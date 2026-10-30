"""Raw files open and browse without GUI-thread metadata walks or
whole-stack reads.

The big-beamtime-file freeze had three layers, each covered here:
``list_raw_entries`` walked the file synchronously on activation (now:
CopyWorker walks once, with progress, and the result rides the session
as ``_raw_entries_cache``); ``load_raw_dataset`` materialized the whole
3D stack to show frame 0 (now: ``LazyRawStack`` reads frames on demand,
warmed off-thread by ``EntryLoadWorker.load_raw``); and the Recent-menu
raw branch bypassed the worker entirely (now: both kinds queue).
Source: file_model.py ``list_raw_entries`` / ``LazyRawStack``,
workers.py ``CopyWorker`` / ``EntryLoadWorker.load_raw``, main_window.py
``_open_recent`` / ``_populate_raw_entries`` / ``_on_raw_entry_loaded``
/ ``_on_open_progress``, session.py ``NexusSession.open(progress=...)``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession


def _linked_master(tmp_path):
    """A Bliss-style master whose scan groups are EXTERNAL links.

    ``visititems`` on the master's root never follows these, so the old
    walker found nothing in beamline masters; the rewritten walker
    resolves each top-level key explicitly.
    """
    scan = tmp_path / "scan0001.h5"
    rng = np.random.default_rng(7)
    with h5py.File(scan, "w") as f:
        f.create_dataset(
            "measurement/eiger",
            data=rng.integers(0, 1000, size=(3, 48, 40), dtype=np.uint16),
        )
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as f:
        f["1.1"] = h5py.ExternalLink(scan.name, "/")
        f["2.1"] = h5py.ExternalLink("gone.h5", "/")  # broken — must be skipped
        f.create_dataset(  # root-level dataset, also a candidate
            "direct", data=np.ones((2, 64, 64), dtype=np.float32)
        )
    return master


# -- list_raw_entries: link following, composed paths, progress --------


def test_list_raw_entries_follows_top_level_external_links(tmp_path):
    entries = file_model.list_raw_entries(_linked_master(tmp_path))
    paths = [e.dataset_path for e in entries]
    # Master-side composed paths (resolvable through the master), in
    # the file's own key order.
    assert paths == ["1.1/measurement/eiger", "direct"]
    assert entries[0].shape == (3, 48, 40)


def test_list_raw_entries_keeps_file_order_not_alphabetical(tmp_path):
    """Entry lists must follow the file's link order (acquisition order
    on track_order beamline masters), not alphabetical sorting — which
    scrambled scan numbering (10.1 before 2.1) in the Selection tree
    and the Display entry combo."""
    path = tmp_path / "ordered.h5"
    with h5py.File(path, "w", track_order=True) as f:
        for name in ("b_scan", "a_scan", "c_scan"):
            f.create_dataset(
                f"{name}/eiger", data=np.ones((2, 32, 32), np.float32)
            )
    entries = file_model.list_raw_entries(path)
    assert [e.dataset_path for e in entries] == [
        "b_scan/eiger", "a_scan/eiger", "c_scan/eiger",
    ]


def test_list_raw_entries_reports_progress_and_survives_broken_link(tmp_path):
    ticks: list[tuple[int, int]] = []
    file_model.list_raw_entries(
        _linked_master(tmp_path),
        progress=lambda done, total: ticks.append((done, total)),
    )
    # One tick per top-level key — including the broken "2.1" link.
    assert ticks == [(1, 3), (2, 3), (3, 3)]


def test_list_raw_entries_unchanged_for_self_contained_file(synthetic_raw):
    entries = file_model.list_raw_entries(synthetic_raw)
    assert [e.dataset_path for e in entries] == ["raw/data0/image"]
    assert entries[0].shape == (4, 64, 64)


# -- LazyRawStack ------------------------------------------------------


def test_lazy_raw_stack_reads_frames_on_demand(synthetic_raw):
    (entry,) = file_model.list_raw_entries(synthetic_raw)
    stack = file_model.LazyRawStack(entry)
    stack.acquire()
    try:
        assert stack.shape == (4, 64, 64) and stack.ndim == 3 and len(stack) == 4
        frame = stack[1]
        assert frame.dtype == np.float32  # uint32 on disk, upcast like load_raw_dataset
        # Values match the eager loader, frame for frame.
        eager = file_model.load_raw_dataset(entry)
        np.testing.assert_array_equal(frame, eager[1])
        # Pixel indexing (cursor readout path).
        assert stack[2, 10, 11] == eager[2, 10, 11]
    finally:
        stack.release()
    assert not stack.is_open
    with pytest.raises(RuntimeError):
        stack[0]


def test_lazy_raw_stack_refuses_materialization(synthetic_raw):
    """np.asarray(stack) would silently read the whole dataset — the
    exact freeze the class prevents — so it must fail loudly."""
    (entry,) = file_model.list_raw_entries(synthetic_raw)
    stack = file_model.LazyRawStack(entry)
    stack.acquire()
    try:
        with pytest.raises(TypeError):
            np.asarray(stack)
    finally:
        stack.release()


def test_lazy_raw_stack_resolves_through_master_link(tmp_path):
    """A RawEntry found through an external link reads via the master."""
    master = _linked_master(tmp_path)
    entry = next(
        e for e in file_model.list_raw_entries(master)
        if e.dataset_path == "1.1/measurement/eiger"
    )
    assert entry.file_path == master
    stack = file_model.LazyRawStack(entry)
    stack.acquire()
    try:
        assert stack[0].shape == (48, 40)
    finally:
        stack.release()


# -- NexusSession.open copy progress ------------------------------------


def test_nexus_open_copy_progress_ticks_and_copies_bytes(synthetic_nexus):
    ticks: list[tuple[int, int]] = []
    session = NexusSession.open(
        synthetic_nexus, progress=lambda d, t: ticks.append((d, t))
    )
    try:
        assert ticks and ticks[-1][0] == ticks[-1][1] == synthetic_nexus.stat().st_size
        assert session.temp_path.read_bytes() == synthetic_nexus.read_bytes()
    finally:
        session.close()
