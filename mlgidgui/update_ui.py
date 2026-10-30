"""Update-check/install workers and the in-app update banner.
Moved out of ``main_window`` in the 2026 source split; the underscore
names are kept because tests import them.
"""
from __future__ import annotations

import subprocess
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton
from mlgidgui import diagnostics, update_check
from mlgidgui import icons, theme_tokens

import logging

logger = logging.getLogger(__name__)


class _UpdateCheckWorker(QObject):
    """Runs the blocking GitHub release fetch off the GUI thread.

    Emits ``finished(result)`` where result is ``(tag, url)`` or ``None``
    (offline / no newer release). Lives on a ``QThread``; the host connects
    ``finished`` with a queued connection to marshal back to the GUI thread.
    """

    finished = Signal(object)

    @Slot()
    def run(self) -> None:
        try:
            result = update_check.latest_release()
        except Exception:  # pragma: no cover - defensive; latest_release is safe
            logger.debug("update check worker failed", exc_info=True)
            result = None
        self.finished.emit(result)


class _UpdateInstallWorker(QObject):
    """Runs the blocking ``pip`` command of an update off the GUI thread.

    Emits ``finished(returncode, output)`` where ``output`` is the combined
    stdout+stderr of pip (for the success/failure dialog). Lives on a
    ``QThread`` like ``_UpdateCheckWorker``; the host marshals ``finished``
    back to the GUI thread with a queued connection.
    """

    finished = Signal(int, str)

    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self._command = list(command)

    @Slot()
    def run(self) -> None:
        # On Windows the running gui-scripts launcher (mlgidgui.exe) is
        # locked and pip's uninstall of it fails with WinError 32; rename
        # it aside first and put it back if the install fails. No-op
        # elsewhere.
        renames: list = []
        try:
            renames = update_check.free_locked_launchers()
            # Written before pip starts, so a crash mid-install leaves a
            # started block with no finished one (diagnostics never
            # raises).
            diagnostics.log_update_started(self._command)
            proc = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
                # Without this the pip child pops a transient console
                # window under the console-less pythonw launch.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            diagnostics.log_update_finished(proc.returncode, output)
            if proc.returncode != 0:
                update_check.restore_launchers(renames)
            self.finished.emit(proc.returncode, output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("update install worker failed", exc_info=True)
            update_check.restore_launchers(renames)
            output = f"Failed to launch pip: {exc}"
            diagnostics.log_update_finished(1, output)
            self.finished.emit(1, output)


class _UpdateBanner(QFrame):
    """Dismissible 'a newer version is available' strip above the tabs.

    Carries an "Update now" button that self-installs the release (emitted
    as ``installRequested``); the host shows it only when the install is a
    normal pip-managed one (see ``update_check.self_update_supported``).
    """

    installRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("UpdateBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._url = update_check.RELEASES_PAGE
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 8, 6)
        row.setSpacing(10)
        self._label = QLabel(self)
        # Named so the skin can colour it; the banner sits on its own
        # accent ground, so its text cannot use the panel text token.
        self._label.setObjectName("UpdateBannerText")
        self._label.setWordWrap(True)
        row.addWidget(self._label, 1)
        self._update_btn = QToolButton(self)
        self._update_btn.setText("Update now")
        self._update_btn.setToolTip(
            "Download and install this release into the current environment."
        )
        self._update_btn.clicked.connect(self.installRequested)
        self._update_btn.hide()
        row.addWidget(self._update_btn)
        view_btn = QToolButton(self)
        view_btn.setText("View release")
        view_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self._url))
        )
        row.addWidget(view_btn)
        close_btn = QToolButton(self)
        # The banner sits on its own accent ground, so its glyph
        # takes the banner's foreground rather than the panel text token.
        close_btn.setIcon(icons.icon(
            "close", color=theme_tokens.color("banner_fg")))
        close_btn.setToolTip("Dismiss")
        close_btn.clicked.connect(self.hide)
        row.addWidget(close_btn)

    def show_update(
        self, current: str, latest: str, url: str, *, can_install: bool = False
    ) -> None:
        self._url = url or update_check.RELEASES_PAGE
        self._label.setText(
            f"A newer version of mlgidGUI is available: "
            f"<b>{latest}</b> (you have {current})."
        )
        self._update_btn.setToolTip(
            "Download and install this release into the current "
            "environment."
        )
        self._update_btn.setVisible(can_install)
        self.show()
