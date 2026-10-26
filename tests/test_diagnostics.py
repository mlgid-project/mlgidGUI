"""Persistent crash / update logs (``mlgidlab.diagnostics``).

The update-install crash reported on Ubuntu 24.04 left nothing behind:
the crash log lived in the temp dir a reboot wipes, and pip's output was
only shown on failure. These pin the per-user log location, the
started / finished blocks around the pip run, and that an unwritable
log location never breaks the worker or startup.
"""
from __future__ import annotations

import faulthandler
import sys
import tempfile
from pathlib import Path

import pytest

from mlgidlab import diagnostics


@pytest.fixture
def no_override(monkeypatch, tmp_path):
    monkeypatch.delenv(diagnostics.LOG_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_linux_uses_xdg_state_home(no_override, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    state = no_override / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    path = diagnostics.log_dir()
    assert path == state / "mlgidlab"
    assert path.is_dir()


def test_linux_falls_back_to_local_state(no_override, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert diagnostics.log_dir() == (
        no_override / "home" / ".local" / "state" / "mlgidlab")
    # A relative XDG_STATE_HOME is invalid per the spec and ignored.
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert diagnostics.log_dir() == (
        no_override / "home" / ".local" / "state" / "mlgidlab")


def test_windows_uses_localappdata(no_override, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(no_override / "appdata"))
    assert diagnostics.log_dir() == (
        no_override / "appdata" / "mlgidlab" / "Logs")


def test_macos_uses_library_logs(no_override, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert diagnostics.log_dir() == (
        no_override / "home" / "Library" / "Logs" / "mlgidlab")


def test_override_wins_and_unwritable_dir_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path / "logs"))
    assert diagnostics.log_dir() == tmp_path / "logs"
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(blocker / "logs"))
    assert diagnostics.log_dir() is None


def test_crash_log_rotates_past_one_mib(monkeypatch, tmp_path):
    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    crash = tmp_path / diagnostics.CRASH_LOG_NAME
    crash.write_bytes(b"x" * (diagnostics.CRASH_LOG_MAX_BYTES + 1))
    stream, path = diagnostics.open_crash_log()
    stream.close()
    assert path == crash
    assert crash.stat().st_size == 0
    assert (tmp_path / "crash.log.1").stat().st_size == (
        diagnostics.CRASH_LOG_MAX_BYTES + 1)


def test_enable_crash_log_writes_to_the_log_dir(monkeypatch, tmp_path):
    import mlgidlab

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    streams: list = []
    # Not the real faulthandler: it would take over pytest's own handler.
    monkeypatch.setattr(
        faulthandler, "enable", lambda file, all_threads: streams.append(file))
    mlgidlab._enable_crash_log()
    assert len(streams) == 1
    streams[0].close()
    text = (tmp_path / diagnostics.CRASH_LOG_NAME).read_text()
    assert f"mlgidLAB {mlgidlab.__version__} started" in text


def _run_worker(command):
    from mlgidlab.update_ui import _UpdateInstallWorker

    results: list = []
    worker = _UpdateInstallWorker(command)
    worker.finished.connect(lambda rc, out: results.append((rc, out)))
    worker.run()
    return results


def test_worker_writes_started_and_finished_blocks(qapp, monkeypatch, tmp_path):
    import mlgidlab

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    command = [sys.executable, "-c", "print('x')"]
    results = _run_worker(command)
    assert results == [(0, "x\n")]
    log = tmp_path / diagnostics.UPDATE_LOG_NAME
    text = log.read_text(encoding="utf-8")
    started = text.index("=== update started")
    finished = text.index("=== update finished")
    assert started < finished
    assert f"mlgidLAB: {mlgidlab.__version__}" in text
    assert f"Python: {sys.version}" in text
    assert "Platform: " in text
    assert f"Command: {' '.join(command)}" in text
    assert "return code 0 ===\nx\n" in text[finished:]
    assert diagnostics.update_log_path() == log


def test_crash_mid_pip_leaves_only_the_started_block(
    qapp, monkeypatch, tmp_path
):
    import subprocess

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    seen: list[str] = []

    def _dies(cmd, **kwargs):
        # The process would be gone here: the block must already be on disk.
        seen.append(
            (tmp_path / diagnostics.UPDATE_LOG_NAME).read_text("utf-8"))
        raise SystemExit("simulated hard crash")

    monkeypatch.setattr(subprocess, "run", _dies)
    with pytest.raises(SystemExit):
        _run_worker(["pip", "install", "x"])
    assert "=== update started" in seen[0]
    assert "=== update finished" not in seen[0]


def test_unwritable_log_location_breaks_neither_worker_nor_startup(
    qapp, monkeypatch, tmp_path
):
    import mlgidlab

    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(blocker / "logs"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(blocker / "tmp"))
    assert diagnostics.append_update_log("x") is None

    results = _run_worker([sys.executable, "-c", "print('x')"])
    assert results == [(0, "x\n")]

    enabled: list = []
    monkeypatch.setattr(
        faulthandler, "enable", lambda **kw: enabled.append(kw))
    mlgidlab._enable_crash_log()  # must not raise
    assert enabled == []


def test_unwritable_log_dir_falls_back_to_temp(qapp, monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(blocker / "logs"))
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
    results = _run_worker([sys.executable, "-c", "print('x')"])
    assert results == [(0, "x\n")]
    fallback = temp / "mlgidlab_update_install.log"
    assert "=== update finished" in fallback.read_text(encoding="utf-8")


def test_failure_dialog_names_the_full_log(main_window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    log = diagnostics.log_update_finished(1, "pip exploded")
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "exec",
        lambda self: shown.append(self.informativeText()))
    main_window._latest_tag = "v99.0.0"
    main_window._on_update_install_finished(1, "pip exploded")
    assert shown and f"Full log: {log}" in shown[0]


def test_restart_prompt_logs_the_request(main_window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    import mlgidlab.main_window_update as mwu

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    # Stop right after the log line: nothing is relaunched or quit here.
    monkeypatch.setattr(
        mwu.QProcess, "startDetached",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(Exception):
        main_window._prompt_restart()
    text = Path(tmp_path / diagnostics.UPDATE_LOG_NAME).read_text("utf-8")
    assert "=== restart requested" in text
