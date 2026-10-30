"""Reclaiming the space a deleted HDF5 object leaves behind.

``del group[name]`` unlinks an object; it does not shrink the file.
HDF5 marks the blocks free for *reuse within that file* and the file
keeps every byte it had, so a user who deletes a 2 GB stack and saves
sees a file exactly as large as before. There is no API that shrinks a
file in place -- the only way to get the space back is to write the
surviving objects into a new file, which is what ``h5repack`` does and
what :func:`repack` does here without requiring the HDF5 tools to be
installed.

Repacking is a whole-file rewrite, so it is not free and must not run on
every save. :func:`should_repack` decides, from two numbers:

* the file's size on disk, and
* its *live* bytes -- the sum of ``get_storage_size()`` over every
  dataset, i.e. what the data actually occupies.

The difference is slack: metadata, B-tree nodes, and the holes left by
deletions. Measured on real files, that discriminates cleanly.

    a healthy 307 MB converted scan   0.7 % slack
    the same file, freshly repacked   2.2 % slack
    a file with half its data deleted 50.8 % slack

So a ratio threshold separates "normal overhead" from "something was
deleted" without having to track deletions, which would miss anything
done outside the editor's own undo history. The walk that produces the
live figure costs 0.16 s over the 2518 objects of that 307 MB file,
against a save that already copies the whole file -- cheap enough to run
on every save.

A floor accompanies the ratio because overhead is a large *fraction* of
a small file: a 40 KB NeXus skeleton is mostly metadata and would repack
forever on a threshold expressed only as a percentage, while saving
nothing worth the rewrite.

``h5py``'s ``get_freespace()`` is deliberately not used. Under the
default file-space strategy it reported 0 on a file with half its
datasets deleted, because the freed blocks went to the aggregator rather
than to a tracked free-space manager.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

#: Repack only when slack is at least this fraction of the file. Set
#: well above the ~2 % a freshly repacked file already carries, so a
#: save that changed nothing never triggers a rewrite.
MIN_SLACK_RATIO = 0.10

#: ...and at least this many bytes. Small files are mostly metadata by
#: proportion, so without a floor a 40 KB NeXus skeleton would repack on
#: every save to reclaim a few KB. The floor only has to clear that
#: noise, not to define "a big file": rewriting 20 MB is instant, and
#: setting this high enough to matter would mean a user deleting 5 MB
#: from a 20 MB file gets nothing back, which is the very complaint this
#: module exists to answer.
MIN_SLACK_BYTES = 1024 * 1024


def live_bytes(path: Path | str) -> int:
    """Bytes the datasets actually occupy.

    ``get_storage_size()`` is the on-disk figure, so it counts a
    compressed dataset at its compressed size and a sparse one at what
    was written -- which is what has to be compared against the file
    size. Attributes and group metadata are not counted; they are part
    of the slack this measures, and are small next to array data.
    """
    total = 0

    def visit(_name: str, obj) -> None:
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            total += obj.id.get_storage_size()

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return total


def slack(path: Path | str) -> tuple[int, int]:
    """``(file_size, reclaimable_estimate)`` for ``path``.

    The estimate is an upper bound on what a repack would recover: real
    metadata is in there too, which is why the thresholds sit well above
    what a clean file shows rather than at zero.
    """
    size = os.path.getsize(path)
    return size, max(0, size - live_bytes(path))


def should_repack(path: Path | str) -> bool:
    """Whether repacking ``path`` would recover enough to be worth a
    whole-file rewrite. Never raises: an unreadable or exotic file
    answers False and takes the ordinary copy path."""
    try:
        size, reclaimable = slack(path)
    except (OSError, KeyError, RuntimeError):
        logger.debug("could not measure slack in %s", path, exc_info=True)
        return False
    if size <= 0:
        return False
    return (
        reclaimable >= MIN_SLACK_BYTES
        and reclaimable / size >= MIN_SLACK_RATIO
    )


def repack(src: Path | str, dst: Path | str) -> None:
    """Write every object of ``src`` into a fresh file at ``dst``.

    Uses ``Group.copy``, i.e. ``H5Ocopy``, which carries the dataset
    creation properties across -- chunking, compression and its options,
    fill value -- so a repack is not a silent decompression. Attributes
    ride along with their objects; the root group's own attributes do
    not (it is not copied, only its members are), so they are copied by
    hand.

    Links are copied AS LINKS. ``expand_soft`` / ``expand_external`` /
    ``expand_refs`` all default to False and are passed explicitly
    because the whole point of this file's structure editor is that a
    link stays a link: expanding one would silently turn a reference
    into a duplicate of its target, doubling the data it was written to
    avoid duplicating.

    ``dst`` is overwritten if it exists. Callers write to a sibling temp
    and rename, so a failure never lands on the user's file.
    """
    with h5py.File(src, "r") as source, h5py.File(dst, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for name in source:
            source.copy(
                name, target, name=name,
                expand_soft=False, expand_external=False, expand_refs=False,
            )


def copy_or_repack(src: Path | str, dst: Path | str) -> bool:
    """Put ``src``'s contents at ``dst``, repacking if that would shrink it.

    The unit of "saving a file" in mlgidGUI: a byte copy normally, and a
    repack when the working copy is carrying the holes left by a delete.
    Returns True when it repacked.

    Both paths write to a sibling temp and ``os.replace`` it into place,
    so an interrupted save cannot leave the user with half a file --
    ``os.replace`` is atomic within a filesystem. The temp is removed on
    failure.

    A repack that fails for any reason falls back to the byte copy. The
    user asked to save; losing the save because the space optimisation
    did not work would be the wrong trade.
    """
    src, dst = Path(src), Path(dst)
    wanted = should_repack(src)
    staging = dst.with_name(f".{dst.name}.mlgidgui-save")
    try:
        repacked = False
        if wanted:
            try:
                repack(src, staging)
                repacked = True
            except Exception:
                logger.warning(
                    "could not repack %s; saving it unchanged", src,
                    exc_info=True)
        if not repacked:
            shutil.copy2(src, staging)
        os.replace(staging, dst)
        shutil.copystat(src, dst)
        return repacked
    except Exception:
        try:
            staging.unlink()
        except OSError:
            pass
        raise
