"""Worker-thread results must be handled on the GUI thread.

The in-app update froze the window after pip had finished: the install
worker's ``finished`` signal reached ``_on_update_install_finished`` on the
worker thread, which closed the progress dialog from there. The handler was
decorated with ``@Slot`` inside ``UpdateMixin``, a plain mixin and not a
QObject, so PySide could not register it on the window and called it
directly on the emitting thread, even through an explicit queued connection.
The earlier tests stubbed the worker and the dialog and could not see it.

Source: main_window_update.py ``_on_update_check_finished`` /
``_on_update_install_finished``; main_window_frames.py ``_on_prefetched``.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QCoreApplication, QObject, QThread

from mlgidgui import diagnostics
from mlgidgui import main_window_update as mwu
from mlgidgui import update_check as uc

pytestmark = pytest.mark.gui


def _on_gui_thread() -> bool:
    return QThread.currentThread() == QCoreApplication.instance().thread()


def test_no_mixin_method_is_a_decorated_slot():
    """A ``@Slot`` on a method of a non-QObject class is called on the
    emitting thread. Every mixin of the window and of the viewer must leave
    its handlers undecorated."""
    from mlgidgui.image_viewer import GIWAXSImageViewer
    from mlgidgui.main_window import MainWindow

    offenders = []
    for root in (MainWindow, GIWAXSImageViewer):
        for cls in root.__mro__:
            if issubclass(cls, QObject):
                continue
            offenders += [
                f"{cls.__name__}.{name}"
                for name, value in vars(cls).items()
                if hasattr(value, "_slots")
            ]
    assert offenders == []


def test_update_install_finish_is_handled_on_the_gui_thread(
    main_window, qtbot, monkeypatch, tmp_path,
):
    """A real install worker on a real QThread, running a real (instant)
    command: the progress dialog is closed and the restart prompt asked
    on the GUI thread, and the thread is released."""
    seen: dict[str, bool] = {}
    asked: list[str] = []

    class Progress:
        def close(self):
            seen["dialog_closed_on_gui_thread"] = _on_gui_thread()

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(mwu, "make_progress_dialog", lambda *a, **k: Progress())
    monkeypatch.setattr(
        uc, "build_upgrade_command", lambda tag: [sys.executable, "-c", "pass"]
    )

    def question(*args, **kwargs):
        seen["prompt_on_gui_thread"] = _on_gui_thread()
        asked.append(args[2])
        return mwu.QMessageBox.StandardButton.No

    monkeypatch.setattr(mwu.QMessageBox, "question", staticmethod(question))
    main_window._latest_tag = "v0.1.0a99"

    main_window._start_update_install("v0.1.0a99")
    qtbot.waitUntil(lambda: main_window._install_thread is None, timeout=30000)

    assert seen == {
        "dialog_closed_on_gui_thread": True,
        "prompt_on_gui_thread": True,
    }
    assert len(asked) == 1


def test_update_check_result_is_handled_on_the_gui_thread(
    main_window, qtbot, monkeypatch,
):
    """The same for the check: its handler shows the banner, the status
    message and (when asked for) a modal confirmation."""
    seen: list[bool] = []

    def is_outdated(current, tag):
        seen.append(_on_gui_thread())
        return False

    monkeypatch.setattr(uc, "latest_release", lambda: ("v0.0.1", "http://x/rel"))
    monkeypatch.setattr(uc, "is_outdated", is_outdated)

    main_window._start_update_check()
    qtbot.waitUntil(lambda: main_window._update_thread is None, timeout=30000)

    assert seen == [True]
