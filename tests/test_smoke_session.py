"""Smoke harness — workstream A, increment 2: session lifecycle.

Drives the most-used path with a synthetic NeXus file and no heavy
backend: open -> populate entries -> step frames -> save -> close.
Verified against source that none of this touches mlgidbase/pygid
(those are imported lazily only in ``pipeline.py`` when a pipeline
command runs), so the suite stays decoupled from the heavy stack.

The open is driven through ``_set_active_session`` directly rather
than ``_open_paths``: the latter queues a ``CopyWorker`` thread and
spawns a modal ``QProgressDialog``, which would hang an offscreen
run. ``_set_active_session`` is the fully synchronous orchestrator it
ultimately feeds (``_populate_entries`` ->
``_load_entry_into_viewer`` -> ``viewer.show_stack``), so calling it
exercises the same load logic deterministically.
"""

from __future__ import annotations

import os
import subprocess
import sys

from mlgidgui import session as session_mod
from mlgidgui.session import NexusSession


def _open(window, path) -> NexusSession:
    """Open ``path`` synchronously and make it the active session."""
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def test_open_populates_entries_and_viewer(main_window, synthetic_nexus):
    """Opening a valid NeXus file fills the entry combo and loads the
    first entry's stack into the viewer."""
    session = _open(main_window, synthetic_nexus)

    assert main_window.session is session
    assert main_window.entry_combo.count() == 1
    # _load_entry_into_viewer -> viewer.show_stack ran synchronously.
    assert main_window.viewer.n_frames == 3
    assert main_window.viewer.current_frame == 0


def test_frame_navigation(main_window, synthetic_nexus):
    """Direct set_frame and the throttled _step_frame shortcut path
    both move the frame and respect bounds."""
    _open(main_window, synthetic_nexus)

    main_window.viewer.set_frame(2)
    assert main_window.viewer.current_frame == 2

    # _step_frame carries an 80 ms time-throttle (_FRAME_STEP_THROTTLE_S)
    # so OS key-autorepeat can't flood it. The first call always passes
    # (_last_frame_step_t defaults to 0.0). Reset the stamp before the
    # second synthetic step so it isn't dropped as a "too soon" repeat.
    main_window._step_frame(-1)
    assert main_window.viewer.current_frame == 1

    main_window._last_frame_step_t = 0.0
    main_window.viewer.set_frame(2)
    main_window._step_frame(+1)  # at the last frame: bounded no-op
    assert main_window.viewer.current_frame == 2


def test_save_writes_back_and_clears_dirty(main_window, synthetic_nexus):
    """Save copies the temp working file back over the original and
    clears the dirty flag, with no confirm dialog (confirm=False).

    Observability is deliberate:

    * ``NexusSession.save`` uses ``shutil.copy2``, which preserves the
      source mtime, so mtime round-trips unchanged and is *not* a
      valid "save happened" signal.
    * The viewer's FrameSource holds ``temp_path`` open read-only and
      h5py forbids reopening the same file ``r+`` in-process (the very
      silx/h5py coupling the detach dance exists for, workstream C's
      target); a smoke test must not poke that.

    The original file is *not* held open (the viewer only touches the
    per-session ``temp_path`` copy). So: stamp the original
    out-of-band, save, and prove the un-stamped working copy
    overwrote it -- which demonstrates the temp -> original copy ran.
    """
    import h5py

    session = _open(main_window, synthetic_nexus)

    with h5py.File(synthetic_nexus, "r+") as f:
        f["entry_0000/data"].attrs["stale_marker"] = "pre_save"
    session.mark_dirty()
    assert session.dirty is True

    ok = main_window._save(confirm=False, session=session)

    assert ok is True
    assert session.dirty is False
    # Save copied the working copy (which never had the marker) over
    # the original, so the out-of-band marker is gone.
    with h5py.File(synthetic_nexus, "r") as f:
        assert "stale_marker" not in f["entry_0000/data"].attrs


def test_close_event_is_idempotent(main_window, synthetic_nexus):
    """closeEvent must be safe to deliver more than once.

    pytest-qt's teardown calls close() again on every registered
    widget after our own close, and Qt itself can re-emit a close
    event. A second pass used to re-clear already-destroyed pyqtgraph
    widgets, raising "Internal C++ object already deleted" on some
    PySide6 builds during shutdown (seen on CI py3.12 + PySide6
    6.11.1). The second close() is now a no-op.
    """
    _open(main_window, synthetic_nexus)
    main_window.close()
    assert getattr(main_window, "_closed", False) is True
    # Second + third close() must not raise (the fixture teardown will
    # call it once more, also harmlessly).
    main_window.close()
    main_window.close()
    assert main_window._closed is True


def test_pid_alive_probes_on_posix(monkeypatch):
    if sys.platform == "win32":
        return
    assert session_mod._pid_alive(os.getpid()) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert session_mod._pid_alive(child.pid) is False


def test_stale_sweep_leaves_everything_alone_on_windows(tmp_path, monkeypatch):
    """On Windows ``os.kill(pid, 0)`` is Ctrl+C, not a probe. The check
    must not call it and must report "alive", so the sweep removes
    nothing rather than interrupt a live process."""
    monkeypatch.setattr(session_mod.sys, "platform", "win32")

    def _no_kill(pid, sig):
        raise AssertionError("os.kill must not run on Windows")

    monkeypatch.setattr(session_mod.os, "kill", _no_kill)
    monkeypatch.setattr(session_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    leftover = tmp_path / "mlgidgui_999999_abcd"
    leftover.mkdir()
    assert session_mod._pid_alive(999999) is True
    assert session_mod.sweep_stale_temp_dirs() == 0
    assert leftover.is_dir()


def test_stale_sweep_removes_dead_pid_dirs_on_posix(tmp_path, monkeypatch):
    if sys.platform == "win32":
        return
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    monkeypatch.setattr(session_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    dead = tmp_path / f"mlgidgui_{child.pid}_x"
    mine = tmp_path / f"mlgidgui_{os.getpid()}_x"
    dead.mkdir()
    mine.mkdir()
    assert session_mod.sweep_stale_temp_dirs() == 1
    assert not dead.exists()
    assert mine.is_dir()


def test_stale_sweep_also_removes_dirs_the_alphas_leaked(tmp_path, monkeypatch):
    if sys.platform == "win32":
        return
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    monkeypatch.setattr(session_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    old = tmp_path / f"mlgidlab_{child.pid}_x"
    new = tmp_path / f"mlgidgui_{child.pid}_x"
    other = tmp_path / f"unrelated_{child.pid}_x"
    for d in (old, new, other):
        d.mkdir()
    assert session_mod.sweep_stale_temp_dirs() == 2
    assert not old.exists() and not new.exists()
    assert other.is_dir()
