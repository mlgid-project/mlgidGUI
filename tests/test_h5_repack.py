"""Deleting from a NeXus file should give the space back.

``del group[name]`` unlinks an object without shrinking the file: HDF5
marks the blocks reusable *within* that file and its size on disk does
not move. ``NexusSession.save`` was a byte copy of the working copy, so
the holes were reproduced faithfully in the user's file and deleting a
2 GB stack freed nothing.

The only way to reclaim it is to write the survivors into a new file.
That is a whole-file rewrite, so the interesting half of this module is
not "does it shrink" but "does it decline to run when there is nothing
to gain", and "does the rewrite preserve everything".
"""
from __future__ import annotations

import os
import shutil

import h5py
import numpy as np
import pytest

from mlgidgui import h5_repack
from mlgidgui.session import NexusSession


def _frames(path, n=6, side=512, **kw):
    """A file whose datasets dominate its size, like a real scan."""
    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["title"] = "root attribute"
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        for i in range(n):
            d = e.create_dataset(
                f"frames{i}", data=np.zeros((side, side), "f4"), **kw)
            d.attrs["units"] = "counts"
    return path


def _delete(path, *names):
    with h5py.File(path, "r+") as f:
        for name in names:
            del f[name]


# -- the report: deleting then saving frees the space -------------------


def test_saving_after_a_delete_shrinks_the_file(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    before = orig.stat().st_size
    session = NexusSession.open(orig)
    try:
        _delete(session.temp_path, "entry/frames0", "entry/frames1",
                "entry/frames2")
        session.save()
    finally:
        session.close()
    assert orig.stat().st_size < before * 0.7


def test_save_as_shrinks_too(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    before = orig.stat().st_size
    session = NexusSession.open(orig)
    try:
        _delete(session.temp_path, "entry/frames0", "entry/frames1",
                "entry/frames2")
        session.save_as(tmp_path / "out.h5")
    finally:
        session.close()
    assert (tmp_path / "out.h5").stat().st_size < before * 0.7


def test_the_working_copy_is_left_alone(tmp_path):
    """Repacking after every delete would rewrite the whole file each
    time. The working copy is temporary, so it is allowed to carry the
    holes until the save that resolves them."""
    orig = _frames(tmp_path / "scan.h5")
    session = NexusSession.open(orig)
    try:
        size = session.temp_path.stat().st_size
        _delete(session.temp_path, "entry/frames0")
        assert session.temp_path.stat().st_size == size
    finally:
        session.close()


# -- and declines when there is nothing to gain -------------------------


def test_a_clean_save_is_not_repacked(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    session = NexusSession.open(orig)
    try:
        assert h5_repack.should_repack(session.temp_path) is False
        assert h5_repack.copy_or_repack(
            session.temp_path, tmp_path / "out.h5") is False
    finally:
        session.close()


def test_a_clean_save_is_byte_identical(tmp_path):
    """The ordinary path must stay the byte copy it always was -- a
    repack on every save would rewrite gigabytes for nothing."""
    orig = _frames(tmp_path / "scan.h5")
    out = tmp_path / "out.h5"
    h5_repack.copy_or_repack(orig, out)
    assert out.read_bytes() == orig.read_bytes()


def test_a_freshly_repacked_file_does_not_want_repacking_again(tmp_path):
    """Guards the threshold from below: a repacked file still carries a
    couple of percent of real metadata, and if that tripped the check
    every save would rewrite the file forever."""
    orig = _frames(tmp_path / "scan.h5")
    _delete(orig, "entry/frames0", "entry/frames1", "entry/frames2")
    out = tmp_path / "out.h5"
    assert h5_repack.copy_or_repack(orig, out) is True
    assert h5_repack.should_repack(out) is False


def test_a_small_file_is_not_repacked_for_a_few_kb(tmp_path):
    """Overhead is a large *fraction* of a small file, so the ratio
    alone would repack a NeXus skeleton on every save."""
    small = tmp_path / "small.h5"
    with h5py.File(small, "w") as f:
        f.create_group("entry")
    assert small.stat().st_size < h5_repack.MIN_SLACK_BYTES
    assert h5_repack.should_repack(small) is False


def test_both_thresholds_are_required(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    _delete(orig, "entry/frames0", "entry/frames1", "entry/frames2")
    assert h5_repack.should_repack(orig) is True
    # ratio met, floor not
    monkey = h5_repack.MIN_SLACK_BYTES
    try:
        h5_repack.MIN_SLACK_BYTES = 10 ** 12
        assert h5_repack.should_repack(orig) is False
    finally:
        h5_repack.MIN_SLACK_BYTES = monkey
    # floor met, ratio not
    monkey = h5_repack.MIN_SLACK_RATIO
    try:
        h5_repack.MIN_SLACK_RATIO = 0.99
        assert h5_repack.should_repack(orig) is False
    finally:
        h5_repack.MIN_SLACK_RATIO = monkey


# -- the rewrite must not lose anything ---------------------------------


def test_repack_preserves_attributes_at_every_level(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["title"] == "root attribute"
        assert f["entry"].attrs["NX_class"] == "NXentry"
        assert f["entry/frames0"].attrs["units"] == "counts"


def test_repack_preserves_chunking_and_compression(tmp_path):
    """H5Ocopy carries the dataset creation properties, so a repack is
    not a silent decompression of the user's data."""
    orig = _frames(tmp_path / "scan.h5", n=2, chunks=(64, 64),
                   compression="gzip", compression_opts=4)
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        d = f["entry/frames0"]
        assert d.chunks == (64, 64)
        assert d.compression == "gzip"
        assert d.compression_opts == 4


def test_repack_keeps_a_soft_link_a_link(tmp_path):
    """Expanding it would turn a reference into a copy of its target,
    duplicating exactly the data the link exists to avoid duplicating."""
    orig = _frames(tmp_path / "scan.h5", n=2)
    with h5py.File(orig, "r+") as f:
        f["entry/alias"] = h5py.SoftLink("/entry/frames0")
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        link = f["entry"].get("alias", getlink=True)
        assert isinstance(link, h5py.SoftLink)
        assert link.path == "/entry/frames0"


def test_repack_keeps_an_external_link_a_link(tmp_path):
    target = _frames(tmp_path / "target.h5", n=1)
    orig = _frames(tmp_path / "scan.h5", n=1)
    with h5py.File(orig, "r+") as f:
        f["entry/ext"] = h5py.ExternalLink(str(target), "/entry/frames0")
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        assert isinstance(f["entry"].get("ext", getlink=True),
                          h5py.ExternalLink)


def test_repack_preserves_the_data(tmp_path):
    orig = tmp_path / "scan.h5"
    payload = np.arange(4096, dtype="f8").reshape(64, 64)
    with h5py.File(orig, "w") as f:
        f.create_dataset("entry/values", data=payload)
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["entry/values"][...], payload)


def test_survivors_and_only_survivors(tmp_path):
    orig = _frames(tmp_path / "scan.h5", n=4)
    _delete(orig, "entry/frames0", "entry/frames2")
    out = tmp_path / "out.h5"
    h5_repack.repack(orig, out)
    with h5py.File(out, "r") as f:
        assert sorted(f["entry"].keys()) == ["frames1", "frames3"]


# -- a failed save must not cost the user their file --------------------


def test_a_failed_repack_still_saves_the_file(tmp_path, monkeypatch):
    """The user asked to save. Losing the save because the space
    optimisation failed would be the wrong trade."""
    orig = _frames(tmp_path / "scan.h5")
    _delete(orig, "entry/frames0", "entry/frames1", "entry/frames2")
    out = tmp_path / "out.h5"

    def _boom(*_a, **_k):
        raise RuntimeError("repack exploded")
    monkeypatch.setattr(h5_repack, "repack", _boom)

    assert h5_repack.copy_or_repack(orig, out) is False
    with h5py.File(out, "r") as f:
        assert sorted(f["entry"].keys()) == ["frames3", "frames4", "frames5"]


def test_no_staging_file_is_left_behind(tmp_path):
    orig = _frames(tmp_path / "scan.h5", n=2)
    out = tmp_path / "out.h5"
    h5_repack.copy_or_repack(orig, out)
    leftovers = [p.name for p in tmp_path.iterdir()
                 if "mlgidgui-save" in p.name]
    assert leftovers == []


def test_a_failure_removes_the_staging_file(tmp_path, monkeypatch):
    orig = _frames(tmp_path / "scan.h5", n=2)
    out = tmp_path / "out.h5"

    def _boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(h5_repack.os, "replace", _boom)

    with pytest.raises(OSError):
        h5_repack.copy_or_repack(orig, out)
    leftovers = [p.name for p in tmp_path.iterdir()
                 if "mlgidgui-save" in p.name]
    assert leftovers == []


def test_should_repack_never_raises_on_a_bad_file(tmp_path):
    junk = tmp_path / "not-hdf5.h5"
    junk.write_bytes(b"this is not an HDF5 file")
    assert h5_repack.should_repack(junk) is False


def test_a_non_hdf5_file_still_copies(tmp_path):
    """should_repack answering False must route to the byte copy, not
    to an exception -- the save path is shared."""
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not hdf5 at all")
    out = tmp_path / "out.h5"
    assert h5_repack.copy_or_repack(junk, out) is False
    assert out.read_bytes() == b"not hdf5 at all"


# -- the measurement itself ---------------------------------------------


def test_slack_grows_when_data_is_deleted(tmp_path):
    orig = _frames(tmp_path / "scan.h5")
    _size, clean = h5_repack.slack(orig)
    _delete(orig, "entry/frames0", "entry/frames1", "entry/frames2")
    _size, dirty = h5_repack.slack(orig)
    assert dirty > clean


def test_live_bytes_counts_the_data(tmp_path):
    orig = _frames(tmp_path / "scan.h5", n=4, side=256)
    expected = 4 * 256 * 256 * 4
    assert h5_repack.live_bytes(orig) == pytest.approx(expected, rel=0.01)
