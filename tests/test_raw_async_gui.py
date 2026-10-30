"""Async raw open + entry display: no raw I/O on the GUI thread.

GUI-side counterpart of ``test_lazy_raw_loading``: CopyWorker carries
the raw-entry walk result to the session, Recent-menu raw opens go
through the worker queue (the old synchronous branch froze the window),
activation consumes the cached walk instead of re-walking, and the
first frame is warmed by ``EntryLoadWorker.load_raw`` off-thread with
stale results dropped. File-browser clicks resolve to detector-dataset
candidates so raw files browse from the tree like NeXus entries.
Source: workers.py ``CopyWorker`` / ``EntryLoadWorker.load_raw``;
main_window.py ``_open_recent`` / ``_on_open_finished`` /
``_finalize_open_batch`` / ``_populate_raw_entries`` /
``_on_raw_entry_loaded`` / ``_on_open_progress`` /
``_activate_raw_entry_for_node``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.session import RawSession
from mlgidgui.workers import CopyWorker, EntryLoadWorker

pytestmark = pytest.mark.gui


class _FakeNode:
    """Stand-in for a silx h5 node: ``_node_h5_path`` reads
    ``local_name`` and ``_node_filename`` reads ``local_filename``."""

    def __init__(self, path: str, filename: str) -> None:
        self.local_name = path
        self.local_filename = filename


def _two_dataset_raw(tmp_path):
    """Raw file with two qualifying detector datasets, distinguishable
    by their constant pixel value (1.0 / 2.0)."""
    path = tmp_path / "two_raw.h5"
    with h5py.File(path, "w", track_order=True) as f:
        f.create_dataset(
            "scan1/eiger", data=np.full((2, 32, 32), 1.0, np.float32)
        )
        f.create_dataset(
            "scan2/eiger", data=np.full((2, 32, 32), 2.0, np.float32)
        )
    return path


def _activate_raw(window, path):
    """Install a RawSession with its walk pre-cached (the CopyWorker
    contract), like ``_finalize_open_batch`` does."""
    session = RawSession.open([path])
    entries = file_model.list_raw_entries(path)
    session._raw_entries_cache = {str(p): entries for p in session.raw_paths}
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_copyworker_caches_raw_entries_and_reports_scan_progress(
    qtbot, synthetic_raw
):
    """Classifying a raw file keeps the walked entry list (the GUI used
    to re-walk it on its own thread) and ticks the scan's progress."""
    worker = CopyWorker(synthetic_raw)
    got: dict = {}
    ticks: list[tuple[int, str]] = []
    worker.finished.connect(got.update)
    worker.progress.connect(lambda pct, label: ticks.append((pct, label)))
    worker.run()

    assert got["kind"] == "raw"
    entries = got["raw_entries"]
    assert [e.dataset_path for e in entries] == ["raw/data0/image"]
    scan_ticks = [t for t in ticks if "Scanning" in t[1]]
    assert scan_ticks, f"no scan progress in {ticks}"
    assert all(0 <= pct <= 100 for pct, _ in ticks)


def test_copyworker_classifies_lima_entry_file_as_raw(qtbot, tmp_path):
    """A detector file with an ``entry_0000``-style root but NO mlgid
    signal (LIMA/Eiger layout: ``entry_0000/measurement/data``) must
    classify as RAW. The shallow name-only classifier called it nexus,
    and the NeXus loader then failed with 'component not found'."""
    path = tmp_path / "eiger_0000.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "entry_0000/measurement/data",
            data=np.ones((3, 64, 64), np.uint32),
        )
    worker = CopyWorker(path)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    assert got["error"] is None
    assert got["kind"] == "raw"
    assert got["session"] is None
    assert [e.dataset_path for e in got["raw_entries"]] == [
        "entry_0000/measurement/data"
    ]


def test_entry_load_worker_load_raw_emits_ready_stack(qtbot, synthetic_raw):
    (entry,) = file_model.list_raw_entries(synthetic_raw)
    worker = EntryLoadWorker()
    got: dict = {}
    worker.raw_loaded.connect(
        lambda rid, label, stack: got.update(rid=rid, label=label, stack=stack)
    )
    worker.load_raw(entry, 9)  # direct call, no thread

    assert got["rid"] == 9 and got["label"] == entry.label
    stack = got["stack"]
    assert stack is not None and stack.is_open
    assert stack[0].shape == (64, 64)
    stack.release()


def test_open_recent_raw_goes_through_worker_queue(
    main_window, qtbot, synthetic_raw, monkeypatch
):
    """End to end from the Recent menu: classify + walk happen exactly
    once (in the worker), the session activates with the cached entries,
    and the first frame renders from the lazy stack."""
    calls: list = []
    orig = file_model.list_raw_entries

    def counting(path, progress=None):
        calls.append(path)
        return orig(path, progress=progress)

    monkeypatch.setattr(file_model, "list_raw_entries", counting)

    main_window._open_recent(str(synthetic_raw), "raw")

    qtbot.waitUntil(
        lambda: main_window.session is not None
        and main_window.session.kind == "raw",
        timeout=30000,
    )
    assert main_window.entry_combo.count() == 1
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=30000
    )
    assert isinstance(main_window.viewer._raw_image_stack, file_model.LazyRawStack)
    assert len(calls) == 1  # the worker's walk; activation reused its result


def test_populate_raw_entries_uses_session_cache_without_walking(
    main_window, qtbot, synthetic_raw, monkeypatch
):
    """Re-activating a raw session must not re-walk the file's metadata
    on the GUI thread — the walk result is cached on the session."""
    session = RawSession.open([synthetic_raw])
    entries = file_model.list_raw_entries(synthetic_raw)
    session._raw_entries_cache = {
        str(p): entries for p in session.raw_paths
    }
    main_window._sessions.append(session)

    def _boom(*a, **k):
        raise AssertionError("no GUI-thread raw walk when the cache is present")

    monkeypatch.setattr(file_model, "list_raw_entries", _boom)
    main_window._set_active_session(session)

    assert main_window.entry_combo.count() == 1
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=30000
    )


def test_on_raw_entry_loaded_drops_stale_stack(main_window, synthetic_raw):
    (entry,) = file_model.list_raw_entries(synthetic_raw)
    stack = file_model.LazyRawStack(entry)
    stack.acquire()
    stale = main_window._entry_req_id - 1

    main_window._on_raw_entry_loaded(stale, entry.label, stack)

    assert not stack.is_open  # released, not installed
    assert main_window.viewer._raw_image_stack is None


def test_open_progress_bar_turns_determinate_on_worker_ticks(main_window):
    """The bar starts as an indeterminate march and flips to a real
    0-100 progress bar on the first CopyWorker percent tick."""
    main_window._show_open_progress("big.h5")
    assert main_window._sb_open_bar.maximum() == 0  # indeterminate

    main_window._on_open_progress(55, "Copying file (700 / 1400 MB)")
    assert main_window._sb_open_bar.maximum() == 100
    assert main_window._sb_open_bar.value() == 55
    assert "Copying" in main_window._sb_open_label.text()

    # The next open starts indeterminate again.
    main_window._show_open_progress("next.h5")
    assert main_window._sb_open_bar.maximum() == 0
    main_window._dismiss_open_progress()


def test_tree_click_on_raw_dataset_switches_entry(main_window, qtbot, tmp_path):
    """Clicking a detector dataset in the file browser selects it in the
    combo and the viewer renders it — same flow as NeXus entry nodes."""
    path = _two_dataset_raw(tmp_path)
    _activate_raw(main_window, path)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=30000
    )
    assert main_window.entry_combo.currentText() == "two_raw.h5::scan1/eiger"

    main_window._activate_entry_for_node(
        _FakeNode("/scan2/eiger", str(path))
    )

    assert main_window.entry_combo.currentText() == "two_raw.h5::scan2/eiger"
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None
        and float(main_window.viewer._raw_image_stack[0].mean()) == 2.0,
        timeout=30000,
    )


def test_tree_click_on_ancestor_group_selects_first_candidate(
    main_window, qtbot, tmp_path
):
    """Clicking a scan GROUP (e.g. a Bliss ``1.1`` node) selects the
    first detector candidate inside it, mirroring how a click anywhere
    inside ``entry_*`` selects that entry."""
    path = _two_dataset_raw(tmp_path)
    _activate_raw(main_window, path)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=30000
    )

    main_window._activate_entry_for_node(_FakeNode("/scan2", str(path)))

    assert main_window.entry_combo.currentText() == "two_raw.h5::scan2/eiger"


def test_tree_click_outside_candidates_is_noop(main_window, qtbot, tmp_path):
    """File root and wrong-file nodes leave the selection alone."""
    path = _two_dataset_raw(tmp_path)
    _activate_raw(main_window, path)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=30000
    )
    before = main_window.entry_combo.currentText()

    main_window._activate_entry_for_node(_FakeNode("/", str(path)))
    main_window._activate_entry_for_node(
        _FakeNode("/scan2/eiger", str(tmp_path / "other.h5"))
    )

    assert main_window.entry_combo.currentText() == before


def test_viewer_clear_releases_lazy_raw_handle(main_window, synthetic_raw):
    (entry,) = file_model.list_raw_entries(synthetic_raw)
    stack = file_model.LazyRawStack(entry)
    stack.acquire()
    stack.get_frame(0)
    main_window.viewer.show_raw_stack(stack)
    assert main_window.viewer.n_frames == 4

    main_window.viewer.clear()

    assert not stack.is_open
    assert main_window.viewer._raw_image_stack is None
