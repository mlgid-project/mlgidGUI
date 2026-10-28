"""Help menu, about/diagnostics, update-check and self-update orchestration, post-update changelog dialog.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

from PySide6.QtCore import QProcess, QSettings, QThread, QUrl, Qt
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
)
from mlgidlab import diagnostics, update_check
from mlgidlab.controls_help import ControlsDialog
from mlgidlab.widgets import make_progress_dialog
from mlgidlab.update_ui import (
    _UpdateCheckWorker,
    _UpdateInstallWorker,
)

import logging
import sys

logger = logging.getLogger(__name__)


class UpdateMixin:
    # ------------------------------------------------------------------
    # Help menu
    # ------------------------------------------------------------------

    def _build_help_menu(self) -> None:
        """Build the rightmost top-level Help menu.

        Entries:
        - **Controls & shortcuts…** — modeless, filterable reference of
          every keyboard shortcut and mouse interaction.
        - **About mlgidLAB…** — modal "About" dialog with versions.
        - **mlgidLAB is becoming mlgidGUI…** — the rename notice, which
          also opens once on first launch.
        - **Copy diagnostics** — clipboard-friendly env/session/log
          dump for bug reports.
        """
        help_menu = self.menuBar().addMenu("&Help")
        self.action_controls = QAction("&Controls && shortcuts…", self)
        self.action_controls.setShortcut(QKeySequence("F1"))
        self.action_controls.setToolTip(
            "Reference for every keyboard shortcut, mouse interaction, "
            "and the manual-peak workflow. Type to filter."
        )
        self.action_controls.triggered.connect(self._show_controls)
        help_menu.addAction(self.action_controls)
        self.action_about = QAction("&About mlgidLAB…", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
        # The one-time startup notice, kept reachable so it can be re-read.
        self.action_rename_notice = QAction(
            "mlgidLAB is becoming mlgidGUI…", self
        )
        self.action_rename_notice.triggered.connect(self._show_rename_notice)
        help_menu.addAction(self.action_rename_notice)
        self.action_copy_diagnostics = QAction("&Copy diagnostics", self)
        self.action_copy_diagnostics.setToolTip(
            "Copy environment info + active session details + recent "
            "log lines to the clipboard. Useful for bug reports."
        )
        self.action_copy_diagnostics.triggered.connect(self._copy_diagnostics)
        help_menu.addAction(self.action_copy_diagnostics)
        help_menu.addSeparator()
        self.action_check_updates = QAction("Check for &updates…", self)
        self.action_check_updates.setToolTip(
            "Check GitHub for a newer mlgidLAB release."
        )
        self.action_check_updates.triggered.connect(
            lambda: self._start_update_check(notify_uptodate=True)
        )
        help_menu.addAction(self.action_check_updates)
        supported, reason = update_check.self_update_supported()
        self.action_update_now = QAction("Update &now…", self)
        self.action_update_now.setEnabled(supported)
        self.action_update_now.setToolTip(
            "Check GitHub and, if a newer release exists, install it into "
            "the current environment."
            if supported
            else reason
        )
        # Re-check + install-if-newer in one step (no need to wait for the
        # startup banner). Same off-thread check as "Check for updates…",
        # but on finding a newer release it goes straight to the confirm +
        # install flow instead of showing the banner.
        self.action_update_now.triggered.connect(
            lambda: self._start_update_check(
                notify_uptodate=True, install_when_found=True
            )
        )
        help_menu.addAction(self.action_update_now)
        self.action_auto_update = QAction(
            "Automatically install updates", self
        )
        self.action_auto_update.setCheckable(True)
        self.action_auto_update.setEnabled(supported)
        self.action_auto_update.setChecked(
            supported
            and str(QSettings().value(self._AUTO_UPDATE_KEY, "false")).lower()
            == "true"
        )
        self.action_auto_update.setToolTip(
            "When a newer release is found on launch, install it into the "
            "current environment (after a confirmation)."
            if supported
            else reason
        )
        self.action_auto_update.toggled.connect(self._set_auto_update)
        help_menu.addAction(self.action_auto_update)

    def _set_auto_update(self, enabled: bool) -> None:
        QSettings().setValue(self._AUTO_UPDATE_KEY, bool(enabled))

    def _show_controls(self) -> None:
        """Modeless, filterable reference of every keyboard shortcut and
        mouse interaction (controls_help.ControlsDialog). One cached
        instance per window: Close hides it, so filter text, size and
        position survive re-open; re-invoking (F1) raises + refocuses."""
        dlg = getattr(self, "_controls_dialog", None)
        if dlg is None:
            dlg = ControlsDialog(
                self, theme=getattr(self, "_current_theme", "dark")
            )
            self._controls_dialog = dlg
        if dlg.isMinimized():
            dlg.showNormal()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.focus_filter()

    def _gather_versions(self) -> dict[str, str]:
        """Return a name → version-string map covering the modules
        most likely to matter in a bug report. Each lookup is
        guarded so a missing/older module reports ``(unavailable)``
        instead of breaking the diagnostics dump."""
        import platform
        import sys

        def _v(modname: str, attr: str = "__version__") -> str:
            try:
                mod = __import__(modname)
                # Some packages spell the version attr differently
                # (pyFAI uses both ``version`` and ``__version__``
                # depending on release).
                if attr == "__version__" and not hasattr(mod, "__version__"):
                    if hasattr(mod, "version"):
                        return str(mod.version)
                return str(getattr(mod, attr))
            except Exception:
                logger.debug("suppressed exception in MainWindow._gather_versions._v", exc_info=True)
                return "(unavailable)"

        try:
            from mlgidlab import __version__ as mlgidlab_version
        except Exception:
            logger.debug("suppressed exception in MainWindow._gather_versions", exc_info=True)
            mlgidlab_version = "(unavailable)"

        return {
            "mlgidLAB": mlgidlab_version,
            "Python": sys.version.split()[0],
            "OS": f"{platform.system()} {platform.release()}",
            "PySide6": _v("PySide6"),
            "Qt": _v("PySide6.QtCore", "__version__"),
            "numpy": _v("numpy"),
            "h5py": _v("h5py"),
            "silx": _v("silx"),
            "pyFAI": _v("pyFAI"),
            "pyqtgraph": _v("pyqtgraph"),
            "matplotlib": _v("matplotlib"),
            "mlgidbase": _v("mlgidbase"),
        }

    def _show_about(self) -> None:
        """Modal About dialog. Pure version info; no external links
        embedded yet (the project doesn't have a canonical docs URL
        we'd want to hardcode here)."""
        versions = self._gather_versions()
        rows = "".join(
            f"<tr><td><b>{name}</b></td><td>{ver}</td></tr>"
            for name, ver in versions.items()
        )
        body = (
            f"<h3>mlgidLAB {versions['mlgidLAB']}</h3>"
            "<p>Graphical interface for the mlgidBASE GIWAXS "
            "analysis pipeline.</p>"
            "<p>Use <b>Help → Copy diagnostics</b> to copy this "
            "environment plus the recent log lines for bug "
            "reports.</p>"
            f"<table>{rows}</table>"
        )
        box = QMessageBox(self)
        box.setWindowTitle("About mlgidLAB")
        box.setText(body)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        # The family wordmark instead of Qt's stock information glyph.
        # Two variants ship: the dark one has a transparent ground and
        # lightened arcs, so the mark does not sit in a white box.
        try:
            from importlib import resources

            variant = ("mlgid_logo_mlgidlab_dark.png"
                       if getattr(self, "_current_theme", "dark") == "dark"
                       else "mlgid_logo_mlgidlab.png")
            path = resources.files("mlgidlab").joinpath(f"assets/app/{variant}")
            with resources.as_file(path) as real:
                pixmap = QPixmap(str(real))
            if not pixmap.isNull():
                box.setIconPixmap(pixmap.scaledToWidth(
                    320, Qt.TransformationMode.SmoothTransformation))
        except Exception:
            logger.debug("no wordmark for the About dialog", exc_info=True)
        box.exec()

    def _copy_diagnostics(self) -> None:
        """Build a plain-text diagnostics blob and put it on the
        clipboard. Sections:

        1. Versions — same map as the About dialog.
        2. Active session — file path, mode, entry, frame.
        3. Recent log lines — last 50 lines from the shared Logs
           dock, in chronological order.

        Nothing is uploaded; it's just text the user can paste.
        """
        import datetime

        # Section 1: versions
        versions = self._gather_versions()
        ver_lines = [f"  {k}: {v}" for k, v in versions.items()]

        # Section 2: active session
        session_lines = []
        sess = self._active_session
        if sess is None:
            session_lines.append("  (no file open)")
        else:
            try:
                session_lines.append(f"  kind:         {sess.kind}")
                session_lines.append(f"  display_path: {sess.display_path}")
                if hasattr(sess, "temp_path"):
                    session_lines.append(f"  temp_path:    {sess.temp_path}")
                if hasattr(sess, "raw_paths"):
                    for p in sess.raw_paths:
                        session_lines.append(f"  raw_path:     {p}")
                entry = (
                    self.entry_combo.currentText()
                    if hasattr(self, "entry_combo") else ""
                )
                session_lines.append(f"  entry:        {entry!r}")
                session_lines.append(
                    f"  frame:        {self.viewer.current_frame} / "
                    f"{max(0, self.viewer.n_frames - 1)}"
                )
                session_lines.append(f"  viewer mode:  {self.viewer._mode}")
            except Exception as exc:
                logger.debug("suppressed exception in MainWindow._copy_diagnostics", exc_info=True)
                session_lines.append(f"  (error gathering session info: {exc})")

        # Section 3: recent log lines
        log_lines: list[str] = []
        if hasattr(self, "_log_view"):
            try:
                blob = self._log_view.toPlainText()
                log_lines = blob.splitlines()[-50:]
            except Exception:
                logger.debug("suppressed exception in MainWindow._copy_diagnostics", exc_info=True)
                pass
        if not log_lines:
            log_lines = ["(no log lines)"]

        diagnostics = (
            f"mlgidLAB diagnostics — {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
            "=== Versions ===\n"
            + "\n".join(ver_lines)
            + "\n\n=== Active session ===\n"
            + "\n".join(session_lines)
            + "\n\n=== Recent log lines (last 50) ===\n"
            + "\n".join(log_lines)
            + "\n"
        )

        QApplication.clipboard().setText(diagnostics)
        # Tell the user it landed — status-bar message rather than a
        # modal because copying is a low-friction action.
        self.statusBar().showMessage(
            f"Copied {len(diagnostics)} chars of diagnostics to clipboard",
            5000,
        )

    # -- Update check + post-update changelog --

    def _run_startup_update_checks(self) -> None:
        """Post-update changelog + async online update check.

        Called once from ``main()`` after the window is shown — NOT from
        ``__init__``, so the test fixture (which builds the window directly)
        never triggers a network call or a changelog dialog. Both parts are
        best-effort and silent on failure.
        """
        self._maybe_show_changelog()
        self._maybe_show_rename_notice()
        self._start_update_check(notify_uptodate=False)

    def _start_update_check(
        self,
        *,
        notify_uptodate: bool = False,
        install_when_found: bool = False,
    ) -> None:
        """Fetch the latest GitHub release off-thread; show the banner if a
        newer version exists. ``notify_uptodate`` posts a status-bar
        confirmation when nothing newer is found (Help -> Check for updates).
        ``install_when_found`` skips the banner and goes straight to the
        confirm + install flow when a newer release exists and self-update is
        supported (Help -> Update now…).
        """
        if getattr(self, "_update_thread", None) is not None:
            return  # a check is already in flight
        self._update_notify_uptodate = notify_uptodate
        self._update_install_when_found = install_when_found
        self._update_thread = QThread(self)
        self._update_worker = _UpdateCheckWorker()
        self._update_worker.moveToThread(self._update_thread)
        self._update_thread.started.connect(self._update_worker.run)
        self._update_worker.finished.connect(self._on_update_check_finished)
        self._update_worker.finished.connect(self._update_thread.quit)
        # No worker.deleteLater on thread.finished: the worker is owned by
        # its Python reference, which _clear_update_thread drops on the GUI
        # thread. A deleteLater runs on the finishing worker thread at the
        # same moment, so the two raced into a double delete (segfault).
        self._update_thread.finished.connect(self._clear_update_thread)
        self._update_thread.start()

    def _clear_update_thread(self) -> None:
        thread = getattr(self, "_update_thread", None)
        if thread is not None:
            thread.deleteLater()
        self._update_thread = None
        self._update_worker = None

    # No @Slot here, and on the handlers of the other worker signals: this
    # class is a plain mixin, not a QObject, so PySide cannot register a
    # decorated method on the window and calls it on the thread that emits
    # (even through a queued connection). The handler then ran on the worker
    # thread and closed the progress dialog from there, which froze the
    # window after every in-app update. Undecorated, the connection is
    # queued onto the GUI thread. tests/test_mixin_slots.py guards this.
    def _on_update_check_finished(self, result) -> None:
        from mlgidlab import __version__ as current

        if result is not None:
            tag, url = result
            if update_check.is_outdated(current, tag):
                self._latest_tag = tag
                supported, _ = update_check.self_update_supported()
                # Go straight to the install (via its confirm dialog, never
                # silently) when the user asked for it — Help -> Update now…
                # this run, or the opt-in "Automatically install updates"
                # setting — and self-update is a normal pip install. Otherwise
                # just surface the banner.
                auto = (
                    str(QSettings().value(self._AUTO_UPDATE_KEY, "false")).lower()
                    == "true"
                )
                want_install = (
                    getattr(self, "_update_install_when_found", False) or auto
                )
                if supported and want_install:
                    self._on_install_update_requested()
                else:
                    self._update_banner.show_update(
                        current, tag, url, can_install=supported
                    )
                return
        if getattr(self, "_update_notify_uptodate", False):
            self.statusBar().showMessage(
                f"mlgidLAB is up to date (v{current}).", 5000
            )

    # -- In-app self-update (pip install --upgrade) --

    def _on_install_update_requested(self) -> None:
        """Confirm, then upgrade to the last-found release in place.

        Wired to the banner's "Update now" button. The confirmation names the
        target version and warns that a restart is needed (the running
        process keeps the old code until relaunched).
        """
        tag = getattr(self, "_latest_tag", None)
        if not tag:
            return
        if update_check.is_successor_release(tag):
            # The renamed app: a different distribution, so this is a move
            # rather than an upgrade. Spell that out before anything runs.
            title = "Move to mlgidGUI"
            version = update_check.display_version(tag)
            text = (
                "mlgidLAB is now called mlgidGUI.\n\n"
                f"Install mlgidGUI {version} into the current "
                "environment and remove mlgidLAB?\n\n"
                "mlgidGUI is installed first; mlgidLAB is only removed once "
                "that worked. Your data files are not touched. You can then "
                "restart into mlgidGUI."
            )
        else:
            title = "Update mlgidLAB"
            text = (
                f"Install {tag} into the current environment?\n\n"
                "mlgidLAB will download and pip-install the new version, "
                "then offer to restart."
            )
        reply = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if reply == QMessageBox.StandardButton.Ok:
            self._start_update_install(tag)

    def _start_update_install(self, tag: str) -> None:
        """Run ``pip install --upgrade`` for ``tag`` off the GUI thread.

        Shows a busy progress dialog while pip runs; ``_on_update_install_
        finished`` closes it and reports success (restart prompt) or failure.
        """
        if getattr(self, "_install_thread", None) is not None:
            return  # an install is already in flight
        commands = update_check.build_update_commands(tag)
        if update_check.is_successor_release(tag):
            version = update_check.display_version(tag)
            progress_text = f"Installing mlgidGUI {version}…"
            progress_title = "Moving to mlgidGUI"
        else:
            progress_text = f"Updating mlgidLAB to {tag}…"
            progress_title = "Updating mlgidLAB"
        self._install_progress = make_progress_dialog(
            self, progress_text, title=progress_title,
        )

        self._install_thread = QThread(self)
        self._install_worker = _UpdateInstallWorker(commands)
        self._install_worker.moveToThread(self._install_thread)
        self._install_thread.started.connect(self._install_worker.run)
        self._install_worker.finished.connect(self._on_update_install_finished)
        self._install_worker.finished.connect(self._install_thread.quit)
        # Released by _clear_install_thread, not deleteLater: see
        # _start_update_check.
        self._install_thread.finished.connect(self._clear_install_thread)
        self._install_thread.start()

    def _clear_install_thread(self) -> None:
        thread = getattr(self, "_install_thread", None)
        if thread is not None:
            thread.deleteLater()
        self._install_thread = None
        self._install_worker = None

    def _on_update_install_finished(self, returncode: int, output: str) -> None:
        progress = getattr(self, "_install_progress", None)
        if progress is not None:
            progress.close()
            self._install_progress = None
        if returncode == 0:
            self._update_banner.hide()
            self._prompt_restart()
            return
        # Failure: surface the tail of pip's output plus the manual fallback.
        tail = "\n".join(output.strip().splitlines()[-15:]) or "(no output)"
        tag = getattr(self, "_latest_tag", "") or ""
        # One line per pip call, so the manual fallback is the whole update.
        command = "\n".join(
            " ".join(argv) for argv in update_check.build_update_commands(tag)
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Update failed")
        box.setText(
            "The update could not be installed. You can try again, or "
            "update manually from a terminal:"
        )
        log_path = diagnostics.update_log_path()
        box.setInformativeText(
            f"{command}\n\nFull log: {log_path}" if log_path else command
        )
        box.setDetailedText(tail)
        box.exec()

    def _prompt_restart(self, tag: str | None = None) -> None:
        """Offer to relaunch so the freshly installed version loads.

        The upgrade landed on disk but the running interpreter still holds the
        old modules; only a restart picks up the new code. After the move to
        mlgidGUI the relaunch starts the new app's module, not this one.
        """
        if tag is None:
            tag = getattr(self, "_latest_tag", None)
        module = update_check.restart_module(tag)
        if update_check.is_successor_release(tag):
            version = update_check.display_version(tag)
            text = f"mlgidGUI {version} was installed. Restart now to start it?"
        else:
            text = "mlgidLAB was updated. Restart now to use the new version?"
        reply = QMessageBox.question(
            self,
            "Update installed",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        diagnostics.log_restart_requested(module)
        # Relaunch the module entry point with the same interpreter, then quit
        # this instance. startDetached returns before we exit so the new
        # process survives our shutdown.
        QProcess.startDetached(sys.executable, ["-m", module])
        QApplication.quit()

    def _maybe_show_rename_notice(self) -> None:
        """Once per profile: tell the user mlgidLAB is becoming mlgidGUI.

        Recorded as seen however the dialog is closed, so it never nags.
        The Help menu entry shows it again on request.
        """
        settings = QSettings()
        seen = str(settings.value(self._RENAME_NOTICE_KEY, "false")).lower()
        if seen == "true":
            return
        try:
            self._show_rename_notice()
        finally:
            settings.setValue(self._RENAME_NOTICE_KEY, True)

    def _show_rename_notice(self) -> None:
        """Explain the coming rename (release 1.0.0) and what to do."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("mlgidLAB is becoming mlgidGUI")
        box.setText(
            "mlgidLAB will be renamed mlgidGUI with its next major release, "
            "1.0.0. It is the same application under a new name."
        )
        box.setInformativeText(
            "What you need to do: nothing now. When 1.0.0 is out, the "
            "update banner (or Help, Update now) installs mlgidGUI and "
            "removes mlgidLAB. Your files and settings carry over.\n\n"
            "What changes: the Python package and the command are called "
            "mlgidgui instead of mlgidlab. Scripts that import mlgidlab "
            "have to be updated; there is no compatibility layer."
        )
        more = box.addButton(
            "More information", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is more:
            QDesktopServices.openUrl(QUrl(update_check.RELEASES_PAGE))

    def _maybe_show_changelog(self) -> None:
        """First launch after an update: show what changed since the last
        version this profile ran (bundled CHANGELOG). Records the current
        version so the popup only appears once per update.
        """
        from mlgidlab import __version__ as current

        settings = QSettings()
        raw = settings.value(self._LAST_SEEN_VERSION_KEY, None)
        last_seen = str(raw) if raw else None
        try:
            sections = update_check.whats_new(current, last_seen)
            if sections:
                self._show_changelog_dialog(current, sections)
        finally:
            settings.setValue(self._LAST_SEEN_VERSION_KEY, current)

    def _show_changelog_dialog(
        self, current: str, sections: list[tuple[str, str]]
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"What's new in mlgidLAB {current}")
        dlg.resize(560, 480)
        layout = QVBoxLayout(dlg)
        intro = QLabel("mlgidLAB was updated. Here's what changed:", dlg)
        layout.addWidget(intro)
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(
            "\n\n".join(f"## {head}\n\n{body}" for head, body in sections)
        )
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dlg)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()
