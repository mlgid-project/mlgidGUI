"""Background open: classification + the first-entry read happen in
``CopyWorker`` (off the GUI thread), so the Open click is instant and the
window never freezes. ``CopyWorker.finished`` carries one result dict
``{"path", "kind", "session", "prewarm", "error"}``; ``_on_open_finished``
installs NeXus files from the warm pre-load and collects raw /
unclassifiable files for the end-of-batch finalize.

Entry switching stays synchronous, so the ``_set_active_session`` test
contract is unchanged.
"""

from __future__ import annotations

import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession
from mlgidgui.workers import CopyWorker

pytestmark = pytest.mark.gui


def _nexus(path, kind="nexus", session=None, prewarm=None, entries=None, error=None):
    """Build a CopyWorker-style result dict for ``_on_open_finished``."""
    return {
        "path": path,
        "kind": kind,
        "session": session,
        "prewarm": prewarm,
        "entries": entries,
        "error": error,
    }


def test_copyworker_classifies_and_prewarms_nexus(qtbot, synthetic_nexus):
    """run() returns kind='nexus', the copied session, and a prewarm whose
    FrameSource already has frame 0 readable (the slow read done here)."""
    worker = CopyWorker(synthetic_nexus)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()  # synchronous (no thread) — fine for the test

    assert got["error"] is None
    assert got["kind"] == "nexus"
    assert got["session"] is not None
    entry, source = got["prewarm"]
    assert entry == "entry_0000"
    assert source.is_open
    assert source.get_cartesian(0).ndim == 2
    source.release()
    got["session"].close()


def test_copyworker_passes_entries_and_progress(qtbot, synthetic_nexus):
    """run() returns the q-entry list (so the GUI never re-scans the
    external links on its own thread) and drives the determinate bar to
    100% via the ``progress`` signal."""
    worker = CopyWorker(synthetic_nexus)
    got: dict = {}
    ticks: list[tuple[int, str]] = []
    worker.finished.connect(got.update)
    worker.progress.connect(lambda pct, label: ticks.append((pct, label)))
    worker.run()

    assert got["entries"] == ["entry_0000"]
    # The scan ticks (0..70%) plus the final 100% must have been emitted.
    assert ticks, "no progress emitted"
    assert ticks[-1][0] == 100
    assert all(0 <= pct <= 100 for pct, _ in ticks)
    got["prewarm"][1].release()
    got["session"].close()


def test_copyworker_open_is_shallow_no_external_resolve(qtbot, tmp_path):
    """Classification + entry listing read only the master's link names —
    a master whose external scans are all MISSING still opens as nexus with
    its entry names. This is the freeze fix: the worker never resolves the
    226 external links (each resolve holds the GIL over the network)."""
    import h5py

    master = tmp_path / "master.h5"
    with h5py.File(master, "w", track_order=True) as f:
        f["entry_0000"] = h5py.ExternalLink("missing_a.h5", "/")
        f["entry_0001"] = h5py.ExternalLink("missing_b.h5", "/")

    worker = CopyWorker(master)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    assert got["error"] is None
    assert got["kind"] == "nexus"
    assert got["entries"] == ["entry_0000", "entry_0001"]
    # The (broken) first scan can't be warmed, but the open still succeeds.
    assert got["prewarm"] is None
    got["session"].close()


def test_reopen_same_path_replaces_old_session(main_window, synthetic_nexus):
    """Opening a path that's already open closes the OLD instance (its
    working copy is stale — the conversion appended/replaced the file
    before the auto-open re-opened it). One instance per file."""
    old = NexusSession.open(synthetic_nexus)
    main_window._on_open_finished(_nexus(synthetic_nexus, session=old))
    old_temp = old.temp_path
    assert main_window.session is old

    new = NexusSession.open(synthetic_nexus)
    main_window._on_open_finished(_nexus(synthetic_nexus, session=new))

    assert main_window.session is new
    assert old not in main_window._sessions
    assert not old_temp.exists()  # stale working copy cleaned up
    assert [s for s in main_window._sessions
            if s.display_path == new.display_path] == [new]


def test_reopen_keeps_dirty_old_session_on_cancel(
    main_window, synthetic_nexus, monkeypatch
):
    """A dirty old instance gets the save prompt; cancelling keeps it
    open alongside the new one instead of silently discarding edits."""
    old = NexusSession.open(synthetic_nexus)
    main_window._on_open_finished(_nexus(synthetic_nexus, session=old))
    old.mark_dirty()
    monkeypatch.setattr(
        main_window, "_confirm_discard_changes", lambda s=None: False
    )

    new = NexusSession.open(synthetic_nexus)
    main_window._on_open_finished(_nexus(synthetic_nexus, session=new))

    assert old in main_window._sessions  # kept — user cancelled
    assert new in main_window._sessions
    # Teardown hygiene: a dirty session would block the close prompt
    # (the monkeypatched confirm is gone by then).
    old.dirty = False


def test_copyworker_classifies_raw_without_copying(qtbot, synthetic_raw):
    """A raw file is classified as 'raw' off the GUI thread; the worker does
    not open/copy a session for it (the GUI bundles raw at batch end)."""
    worker = CopyWorker(synthetic_raw)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    assert got["error"] is None
    assert got["kind"] == "raw"
    assert got["session"] is None
    assert got["prewarm"] is None


def test_on_open_finished_uses_prewarm(main_window, synthetic_nexus, monkeypatch):
    """A NeXus result with a prewarm renders the first entry from it;
    ``load_entry`` (the synchronous re-open) is NOT called and the prewarm
    is consumed."""
    session = NexusSession.open(synthetic_nexus)
    source = file_model.FrameSource(file_path=session.temp_path, entry="entry_0000")
    source.acquire()
    source.get_cartesian(0)

    def _boom(*a, **k):
        raise AssertionError("load_entry must not run when a prewarm is present")

    monkeypatch.setattr(file_model, "load_entry", _boom)

    main_window._on_open_finished(
        _nexus(session.temp_path, session=session, prewarm=("entry_0000", source))
    )

    assert main_window.session is session
    assert main_window.viewer.n_frames == 3
    assert getattr(session, "_prewarm") is None


def test_on_open_finished_without_prewarm_loads_sync(main_window, synthetic_nexus):
    """No prewarm → normal synchronous load still works (the path entry
    switches and tests rely on)."""
    session = NexusSession.open(synthetic_nexus)
    main_window._on_open_finished(_nexus(session.temp_path, session=session))
    assert main_window.session is session
    assert main_window.viewer.n_frames == 3


def test_on_open_finished_uses_entry_cache_no_rescan(
    main_window, synthetic_nexus, monkeypatch
):
    """When the worker passes ``entries``, ``_populate_entries`` fills the
    combo from that cache and never re-opens the external links on the GUI
    thread (the residual open freeze). Proven by making ``list_entries``
    raise: if the cache is used, it is never reached."""
    session = NexusSession.open(synthetic_nexus)

    def _boom(*a, **k):
        raise AssertionError("list_entries must not run when entries cached")

    monkeypatch.setattr(file_model, "list_entries", _boom)

    main_window._on_open_finished(
        _nexus(session.temp_path, session=session, entries=["entry_0000"])
    )

    assert main_window.session is session
    assert main_window.entry_combo.count() == 1
    assert main_window.entry_combo.itemText(0) == "entry_0000"
    # Cache is consumed once so later re-populates re-scan a live file.
    assert getattr(session, "_entries_cache", None) is None
