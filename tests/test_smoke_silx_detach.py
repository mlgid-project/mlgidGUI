"""Smoke harness — workstream C: the `_detached_silx_tree` CM.

`test_smoke_session.test_save_writes_back_and_clears_dirty` had to
stamp the *original* file out-of-band because the viewer's FrameSource
holds `temp_path` open read-only and h5py refuses an in-process `r+`
reopen — the exact silx/h5py coupling C's context manager exists to
manage. These are the positive-path proof: inside
`with main_window._detached_silx_tree():` the working copy *is*
writable, and on exit (normal or exception) the FrameSource is
reacquired so the viewer keeps working.
"""

from __future__ import annotations

import h5py
import pytest

from mlgidgui.session import NexusSession


def _open(window, path) -> NexusSession:
    """Open + activate, mirroring `_on_open_finished`: the session is
    appended to `_sessions` *before* activation so `_reattach_silx_tree`
    reinserts its file into the silx tree."""
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_temp_path_writable_inside_detached_scope(
    main_window, synthetic_nexus
):
    """Inside the detached scope the working copy can be opened `r+`
    (no 'already open for read-only'), and the edit round-trips through
    save after the context manager reacquires on exit.

    `_detach_silx_tree` closes the viewer's FrameSource handle
    synchronously and blocks on the background PrefetchWorker's
    ``release`` slot via ``BlockingQueuedConnection``, so by the time
    the CM body runs the worker's h5py handle is provably closed —
    no event-loop pump needed. See
    ``test_prefetch_release_is_synchronous`` for the regression check
    on that guarantee.
    """
    session = _open(main_window, synthetic_nexus)
    assert main_window.viewer.n_frames == 3  # FrameSource acquired

    with main_window._detached_silx_tree():
        with h5py.File(session.temp_path, "r+") as f:
            f["entry_0000/data"].attrs["cm_marker"] = "in_scope"

    # Exit reacquired the FrameSource: the viewer still reads frames.
    main_window.viewer.set_frame(1)
    assert main_window.viewer.current_frame == 1

    # The in-scope edit survives the reattach and a normal save.
    session.mark_dirty()
    assert main_window._save(confirm=False, session=session) is True
    with h5py.File(synthetic_nexus, "r") as f:
        assert f["entry_0000/data"].attrs.get("cm_marker") == "in_scope"


def test_detached_scope_reattaches_on_exception(
    main_window, synthetic_nexus
):
    """An exception raised inside the scope still triggers the
    finally-reattach; the exception propagates and the viewer remains
    usable afterwards."""
    _open(main_window, synthetic_nexus)

    with pytest.raises(RuntimeError, match="boom"):
        with main_window._detached_silx_tree():
            raise RuntimeError("boom")

    main_window.viewer.set_frame(2)
    assert main_window.viewer.current_frame == 2


def test_prefetch_release_is_synchronous(main_window, synthetic_nexus, qtbot):
    """`_detach_silx_tree` must return only after the PrefetchWorker
    has actually closed its h5py handle — not just after a queued
    release signal has been emitted.

    Regression: clearing peaks on a multi-frame entry via Tools →
    Reset used to fail with HDF5 ``Unable to synchronously open
    file`` because the worker's read handle was still open when the
    GUI thread raced ahead to ``h5py.File(..., 'r+')`` in
    ``clear_peaks``. The fix routes the release through
    ``QMetaObject.invokeMethod(..., BlockingQueuedConnection)`` so the
    GUI thread blocks until the worker thread has finished
    ``release()``.

    Inspects the worker's ``_file`` attribute directly — checking the
    h5py open at the next line would silently pass because the test
    conftest sets ``HDF5_USE_FILE_LOCKING=FALSE`` (a second handle
    succeeds even with the race present). The property the fix
    guarantees is the closed handle, so test that.
    """
    _open(main_window, synthetic_nexus)
    # Multi-frame entry — _configure_prefetch_for_active_entry spawned
    # the worker and queued a configure that opens the h5py handle.
    # Pump the event loop so the worker actually opens before we
    # check it; we are asserting the *release* is synchronous, not
    # the configure.
    qtbot.waitUntil(
        lambda: main_window._prefetch_worker is not None
        and main_window._prefetch_worker._file is not None,
        timeout=30000,
    )

    main_window._detach_silx_tree()
    try:
        # No qtbot.wait, no waitUntil — must be closed by the time the
        # call returns, otherwise the bug is back.
        assert main_window._prefetch_worker._file is None
    finally:
        main_window._reattach_silx_tree()
