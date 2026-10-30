"""The installer (frozen) build: no in-app pip update, and a start-up self-test.

A PyInstaller build has no pip to run, so an update there is a new installer;
the window must offer the releases page instead of "Update now". The
``--self-test`` switch is what CI runs against the frozen exe to prove its
data files and imports are all there.

Source: update_check.py ``is_frozen`` / ``self_update_supported``;
selftest.py; __init__.py ``main``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from mlgidgui import selftest
from mlgidgui import update_check as uc

pytestmark = pytest.mark.gui


def test_is_frozen_follows_sys_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert uc.is_frozen() is False
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert uc.is_frozen() is True


def test_a_frozen_build_does_not_update_itself(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    supported, reason = uc.self_update_supported()
    assert supported is False
    assert "installer" in reason and "releases page" in reason


def test_update_actions_are_off_and_explain_why_when_frozen(qtbot, monkeypatch):
    from mlgidgui.main_window import MainWindow

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    window = MainWindow()
    qtbot.addWidget(window)
    for action in (window.action_update_now, window.action_auto_update):
        assert not action.isEnabled()
        assert "installer" in action.toolTip()


def test_self_test_passes_in_a_fresh_process(tmp_path):
    """The real entry point, as the frozen exe is started in CI."""
    log = tmp_path / "selftest.log"
    env = {
        "QT_QPA_PLATFORM": "offscreen",
        "MLGIDGUI_SELFTEST_LOG": str(log),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "MLGIDGUI_LOG_DIR": str(tmp_path / "logs"),
    }
    import os

    proc = subprocess.run(
        [sys.executable, "-m", "mlgidgui", "--self-test"],
        env={**os.environ, **env}, cwd=tmp_path,
        capture_output=True, text=True, timeout=180,
    )
    text = log.read_text(encoding="utf-8")
    assert proc.returncode == 0, text + proc.stderr[-1500:]
    assert "self-test passed" in text
    for name in ("window opens a NeXus file", "bundled changelog", "bundled icons"):
        assert f"ok   {name}" in text


def test_self_test_fails_when_a_check_fails(qapp, monkeypatch, tmp_path):
    log = tmp_path / "selftest.log"
    monkeypatch.setenv("MLGIDGUI_SELFTEST_LOG", str(log))

    def broken(path):
        raise OSError("disk full")

    monkeypatch.setattr(selftest, "_write_nexus", broken)
    assert selftest.run(["--self-test"]) == 1
    text = log.read_text(encoding="utf-8")
    assert "FAIL window opens a NeXus file" in text and "disk full" in text
    assert "self-test FAILED" in text
