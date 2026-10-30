"""Menu-bar construction and menu-action handlers: File/Edit/Tools/View/Settings menus, clear/reset/export actions, theme and fullscreen, find-peak, recent files.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import json
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QMessageBox,
    QSpinBox,
    QTabBar,
)
from mlgidgui import file_dialogs
from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.main_window_dialogs import (
    _ExportPeaksDialog,
    _SettingsDialog,
)
from mlgidgui.main_window_build import _dirty_dot
from mlgidgui.session import NexusSession
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class MenusMixin:
    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._build_file_menu(file_menu)
        self._build_edit_menu(bar)
        self._build_tools_menu(bar)

    def _build_edit_menu(self, bar) -> None:
        edit_menu = bar.addMenu("&Edit")
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_undo.triggered.connect(self._action_undo)
        edit_menu.addAction(self.action_undo)

        self.action_redo = QAction("&Redo", self)
        # Bind both Ctrl+Y (Win/Linux default) and Ctrl+Shift+Z so muscle
        # memory from either platform works.
        self.action_redo.setShortcuts([
            QKeySequence(QKeySequence.StandardKey.Redo),
            QKeySequence("Ctrl+Shift+Z"),
        ])
        self.action_redo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_redo.triggered.connect(self._action_redo)
        edit_menu.addAction(self.action_redo)

        edit_menu.addSeparator()
        # Ctrl+C / Ctrl+V: copy + paste detected peaks across frames in
        # the same entry. Scoped to detected in this iteration; fitted
        # / matched / manual copy is deferred. Text widgets (QLineEdit,
        # QSpinBox) intercept these for their own clipboard semantics
        # before window-level actions see them, so typing in inputs
        # works as expected.
        self.action_copy_peaks = QAction("&Copy detected peak(s)", self)
        self.action_copy_peaks.setShortcut(QKeySequence.StandardKey.Copy)
        self.action_copy_peaks.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_copy_peaks.setToolTip(
            "Copy the selected detected peak(s) to the in-app clipboard. "
            "Paste on another frame in the same entry."
        )
        self.action_copy_peaks.triggered.connect(self._on_copy_peaks)
        edit_menu.addAction(self.action_copy_peaks)

        self.action_paste_peaks = QAction("&Paste detected peak(s)", self)
        self.action_paste_peaks.setShortcut(QKeySequence.StandardKey.Paste)
        self.action_paste_peaks.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_paste_peaks.setToolTip(
            "Paste copied detected peak(s) onto the current frame. "
            "Same-entry only; cross-entry paste is intentionally blocked."
        )
        self.action_paste_peaks.triggered.connect(self._on_paste_peaks)
        edit_menu.addAction(self.action_paste_peaks)

        # Ctrl+Shift+V: paste the clipboard to every frame in a typed
        # range (e.g. "0-34,37"). One grouped undo reverses the whole
        # range. Out-of-range frames are filtered with a confirmation
        # dialog; duplicates in the input dedup silently.
        self.action_paste_peaks_to_range = QAction(
            "Paste detected peak(s) to &frames…", self,
        )
        self.action_paste_peaks_to_range.setShortcut(
            QKeySequence("Ctrl+Shift+V")
        )
        self.action_paste_peaks_to_range.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_paste_peaks_to_range.setToolTip(
            "Paste copied detected peak(s) to a typed frame range "
            "(e.g. 0-34,37). Same-entry only."
        )
        self.action_paste_peaks_to_range.triggered.connect(
            self._on_paste_peaks_to_range
        )
        edit_menu.addAction(self.action_paste_peaks_to_range)

        edit_menu.addSeparator()
        # Find peak by ID. Opens a modal asking for the kind + numeric
        # ID, then selects that peak in the viewer (jumping to its
        # frame if needed) and switches the Peaks dock to the
        # matching tab via the existing selection-sync wiring.
        self.action_find_peak = QAction("&Find peak by ID…", self)
        self.action_find_peak.setShortcut(QKeySequence("Ctrl+F"))
        self.action_find_peak.setToolTip(
            "Find a Detected or Fitted peak by its numeric ID and "
            "select it in the viewer."
        )
        self.action_find_peak.triggered.connect(self._action_find_peak)
        edit_menu.addAction(self.action_find_peak)

        # T: flip the selection between a detected peak and its fitted
        # peak (same id, needs the peak link). A bare letter, like the
        # J/K frame keys, so text fields consume it before the window.
        self.action_switch_detected_fitted = QAction(
            "&Switch detected / fitted", self,
        )
        self.action_switch_detected_fitted.setShortcut(QKeySequence("T"))
        self.action_switch_detected_fitted.setToolTip(
            "Switch the selection from a detected peak to its fitted "
            "peak, or back. Pairs by id, so it needs the peak link."
        )
        self.action_switch_detected_fitted.triggered.connect(
            self._on_switch_detected_fitted
        )
        edit_menu.addAction(self.action_switch_detected_fitted)

        # One-off repair for older datasets: fitted peaks with no
        # detected peak of the same id. No shortcut, it rewrites many
        # frames at once.
        edit_menu.addSeparator()
        self.action_add_detected_for_fitted = QAction(
            "Add &detected boxes for fitted peaks", self,
        )
        self.action_add_detected_for_fitted.setToolTip(
            "For every fitted peak of the active entry that has no "
            "detected peak with the same id, add a detected box of the "
            "same size, so each fitted peak has a detected partner. "
            "Undoable."
        )
        self.action_add_detected_for_fitted.triggered.connect(
            self._on_add_detected_for_fitted
        )
        edit_menu.addAction(self.action_add_detected_for_fitted)

    def _build_tools_menu(self, bar) -> None:
        """Bulk-edit operations that don't fit the per-peak ROI workflow.

        Currently scoped to "clear all of one kind for the active entry".
        Future additions (export, copy peaks across frames, statistics,
        symmetry ops, etc.) will land here too — see the README for the
        full roadmap.
        """
        tools_menu = bar.addMenu("&Tools")
        # The three per-kind clear-* entries each expand into a scope
        # sub-submenu (Active frame / Active entry / All entries) so the
        # user can wipe exactly the slice they mean. The Reset submenu
        # below still offers a one-click "everything on this scope"
        # for when the user does not care about the kind split.
        clear_menu = tools_menu.addMenu("&Clear peaks")
        clear_menu.setToolTipsVisible(True)

        self._clear_detected_menu = self._build_clear_kind_submenu(
            clear_menu, "Detected", "detected"
        )

        # "Fitted and Matched": clearing fitted necessarily invalidates
        # matched, because matched solutions reference fitted ids and an
        # orphaned matched_* group would render against missing rows.
        # The cascade is one-way (fitted -> matched).
        self._clear_fitted_menu = self._build_clear_kind_submenu(
            clear_menu, "Fitted and Matched", "fitted"
        )

        # "Matched" clears only the matched_* solutions; detected and
        # fitted are left intact (re-match without re-fitting).
        self._clear_matched_menu = self._build_clear_kind_submenu(
            clear_menu, "Matched", "matched"
        )

        # Re-evaluate "Active frame" gates across every per-kind submenu
        # just before the Clear peaks parent is shown. Cheap, and avoids
        # plumbing a frame-count signal into every action.
        clear_menu.aboutToShow.connect(self._refresh_clear_menu_state)

        # Reset submenu — full wipe of det + fit + match (and manual,
        # in-memory) at three scopes. "Active frame" is greyed out
        # when fewer than two frames are loaded since on a single-
        # frame file it would just duplicate "Active entry".
        clear_menu.addSeparator()
        reset_menu = clear_menu.addMenu("&Reset all peaks")
        reset_menu.setToolTipsVisible(True)

        self.action_reset_all = QAction("All entries", self)
        self.action_reset_all.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on every "
            "entry in the active file."
        )
        self.action_reset_all.triggered.connect(
            lambda: self._action_reset_analysis("all")
        )
        reset_menu.addAction(self.action_reset_all)

        self.action_reset_entry = QAction("Active entry (all frames)", self)
        self.action_reset_entry.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on the "
            "currently displayed entry."
        )
        self.action_reset_entry.triggered.connect(
            lambda: self._action_reset_analysis("entry")
        )
        reset_menu.addAction(self.action_reset_entry)

        self.action_reset_frame = QAction("Active frame", self)
        self.action_reset_frame.setToolTip(
            "Clear detected, fitted, and matched peaks on just the "
            "currently displayed frame of the active entry. Manual "
            "peaks are wiped (they live in memory across frames)."
        )
        self.action_reset_frame.triggered.connect(
            lambda: self._action_reset_analysis("frame")
        )
        reset_menu.addAction(self.action_reset_frame)
        # Re-evaluate the per-scope enabled states right before the
        # submenu is shown — n_frames / session state can change between
        # menu opens, and aboutToShow keeps the gate cheap (no signal
        # plumbing on every viewer event).
        reset_menu.aboutToShow.connect(self._refresh_reset_menu_state)

        # Figure export. Replaces the previous pyqtgraph
        # ImageExporter-based PNG capture with a non-modal window
        # built around ``mlgidbase.plot_analysis_results``. Lives at
        # ``mlgidgui.figure_export_window.FigureExportWindow``;
        # imported lazily inside the handler so a missing pipeline
        # dep doesn't break menu construction.
        tools_menu.addSeparator()
        self.action_export_figure = QAction("Export figure…", self)
        self.action_export_figure.triggered.connect(self._action_export_figure)
        tools_menu.addAction(self.action_export_figure)

        # CSV export of detected/fitted/matched peaks. NeXus-only.
        self.action_export_csv = QAction("Export peaks as CSV…", self)
        self.action_export_csv.triggered.connect(self._action_export_csv)
        tools_menu.addAction(self.action_export_csv)

    def _build_clear_kind_submenu(self, parent_menu, label: str, kind: str):
        """Build a per-kind Clear-peaks submenu with three scope choices.

        ``kind`` is one of ``detected``/``fitted``/``matched`` and is
        forwarded to ``_action_clear_file_peaks`` together with the
        chosen scope. Keeps the per-action references on ``self`` so
        ``_refresh_clear_menu_state`` can flip "Active frame" enabled
        when the active entry has a single frame (no per-frame scope
        is meaningful there — it would just duplicate Active entry).
        Returns the QMenu so the caller can stash it for raw-mode
        gating.
        """
        sub = parent_menu.addMenu(label)
        sub.setToolTipsVisible(True)

        act_all = QAction("All entries", self)
        act_all.setToolTip(
            f"Clear every {kind} peak on every entry in the active file."
        )
        act_all.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "all")
        )
        sub.addAction(act_all)

        act_entry = QAction("Active entry (all frames)", self)
        act_entry.setToolTip(
            f"Clear every {kind} peak on the currently displayed entry."
        )
        act_entry.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "entry")
        )
        sub.addAction(act_entry)

        act_frame = QAction("Active frame", self)
        act_frame.setToolTip(
            f"Clear {kind} peaks on just the currently displayed frame "
            "of the active entry."
        )
        act_frame.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "frame")
        )
        sub.addAction(act_frame)

        # Stash per-kind frame action so _refresh_clear_menu_state can
        # gate it on the live frame count.
        setattr(self, f"_clear_{kind}_frame_action", act_frame)
        setattr(self, f"_clear_{kind}_entry_action", act_entry)
        setattr(self, f"_clear_{kind}_all_action", act_all)
        return sub

    def _refresh_clear_menu_state(self) -> None:
        """Gate the per-kind Clear-peaks scope actions.

        Mirrors ``_refresh_reset_menu_state``: "Active frame" is greyed
        out unless there's an open session and the current entry has
        more than one frame. Active-entry / All-entries need only an
        open session. Kept cheap (called on ``aboutToShow``) — no signal
        plumbing on every viewer event.
        """
        has_session = self.session is not None and self._pipe_thread is None
        n_frames = getattr(self.viewer, "n_frames", 0) if has_session else 0
        for kind in ("detected", "fitted", "matched"):
            entry_a = getattr(self, f"_clear_{kind}_entry_action", None)
            all_a = getattr(self, f"_clear_{kind}_all_action", None)
            frame_a = getattr(self, f"_clear_{kind}_frame_action", None)
            if entry_a is not None:
                entry_a.setEnabled(has_session)
            if all_a is not None:
                all_a.setEnabled(has_session)
            if frame_a is not None:
                frame_a.setEnabled(has_session and n_frames > 1)

    def _scoped_peak_targets(self, scope, active_entry, *, error_title):
        """Resolve a peak scope to (targets, scope_label) or None.

        targets is the (entry, frame|None) list the Clear and Reset
        wipe loops iterate; scope_label names it in confirmations.
        Returns None when listing entries failed (the error dialog was
        already shown). Export-CSV keeps its own variant (no label, no
        confirmation).
        """
        if scope == "all":
            try:
                targets = [
                    (e, None)
                    for e in file_model.list_entries(self.session.temp_path)
                ]
            except Exception as exc:
                QMessageBox.critical(
                    self, error_title, f"Could not list entries: {exc}"
                )
                return None
            return targets, f"all {len(targets)} entries"
        if scope == "entry":
            return [(active_entry, None)], f"entry {active_entry}"
        frame_idx = int(self.viewer.current_frame)
        return [(active_entry, frame_idx)], f"frame {frame_idx} of {active_entry}"

    def _action_clear_file_peaks(self, kind: str, scope: str = "entry") -> None:
        """Empty every ``<kind>_peaks`` dataset at the requested scope.

        ``scope`` is one of:
        - ``"entry"`` — active entry, every frame in it.
        - ``"all"``   — every entry in the active file, every frame.
        - ``"frame"`` — active entry, just the active frame.

        Cascade rule (one-way):
        - clearing ``fitted`` also clears ``matched`` (matched rows
          reference fitted ids; orphaned matched_* groups can't render).
        - clearing ``detected`` also clears fitted (and so matched)
          while the fitted/detected link is on — a fit belongs to its
          detection, and clearing every detection on a scope would
          otherwise leave every fit behind with nothing behind it. With
          the link off, detected clears alone, as it always did.
        - clearing ``matched`` clears matched only; detected and fitted
          are left intact (re-match without re-fitting). See the
          Tools-menu wiring above.

        Manual peaks are session-wide and live only in memory — they
        are deliberately *not* touched here; only ``Reset all peaks``
        wipes them.
        """
        if self._is_busy():
            return
        active_entry = self.entry_combo.currentText()
        if scope in ("entry", "frame") and not active_entry:
            return
        if scope == "frame" and getattr(self.viewer, "n_frames", 0) <= 1:
            return

        resolved = self._scoped_peak_targets(
            scope, active_entry, error_title="Clear failed"
        )
        if resolved is None:
            return
        targets, scope_label = resolved

        from mlgidgui.peak_link import link_enabled

        linked = kind == "detected" and link_enabled()
        if not self._confirm_clear(kind, scope_label, linked=linked):
            return

        kinds_to_clear = [kind]
        if kind == "fitted" or linked:
            kinds_to_clear.append("matched")
        if linked:
            kinds_to_clear.insert(1, "fitted")

        with self._detached_silx_tree():
            try:
                removed_total = 0
                for entry, frame in targets:
                    for k in kinds_to_clear:
                        removed_total += file_model.clear_peaks(
                            self.session.temp_path, entry, k, frame=frame
                        )
            except Exception as exc:
                QMessageBox.critical(self, "Clear failed", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        # Bulk wipe invalidates every FileGeomAction and the selection.
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Clearing fitted rows orphans any phase-tracking results
        # (upstream tracks fitted peaks). Detected/matched-only clears
        # leave fitted intact so tracks stay valid.
        if kind == "fitted":
            self._invalidate_scan_tracks()
        if active_entry:
            self._load_entry_into_viewer(active_entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Cleared {' + '.join(kinds_to_clear)} peaks "
            f"({removed_total} rows total) on {scope_label}"
        )

    def _refresh_reset_menu_state(self) -> None:
        """Gate Reset submenu actions on session + frame availability.

        Active-frame is greyed out with a single frame loaded since the
        clear would be identical to Active-entry. Active-entry / All
        entries need only an open session.
        """
        has_session = self.session is not None and self._pipe_thread is None
        n_frames = getattr(self.viewer, "n_frames", 0) if has_session else 0
        self.action_reset_entry.setEnabled(has_session)
        self.action_reset_all.setEnabled(has_session)
        self.action_reset_frame.setEnabled(has_session and n_frames > 1)

    def _action_reset_analysis(self, scope: str) -> None:
        """Wipe det + fit + match (and manual peaks) at the requested scope.

        ``scope`` is one of:
        - ``"entry"`` — active entry, every frame in it.
        - ``"all"``   — every entry in the active file, every frame.
        - ``"frame"`` — active entry, just the active frame.

        Manual peaks are session-wide and live in memory only; every
        scope clears them outright since the user asked for a true reset.
        """
        if self._is_busy():
            return
        active_entry = self.entry_combo.currentText()
        if scope in ("entry", "frame") and not active_entry:
            return
        if scope == "frame" and getattr(self.viewer, "n_frames", 0) <= 1:
            return

        resolved = self._scoped_peak_targets(
            scope, active_entry, error_title="Reset failed"
        )
        if resolved is None:
            return
        targets, scope_label = resolved

        reply = QMessageBox.question(
            self,
            "Reset analysis",
            f"Remove every detected, fitted, matched, and manual peak "
            f"on {scope_label}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Manual peaks are global session state — drop them once,
        # regardless of scope.
        self.viewer.clear_all_manual_peaks()

        with self._detached_silx_tree():
            try:
                removed_total = 0
                for entry, frame in targets:
                    for kind in ("detected", "fitted", "matched"):
                        removed_total += file_model.clear_peaks(
                            self.session.temp_path, entry, kind, frame=frame
                        )
            except Exception as exc:
                QMessageBox.critical(self, "Reset failed", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Reset wipes detected rows — scan-tracking results are orphaned.
        self._invalidate_scan_tracks()
        # Refresh the displayed entry — the cleared one if the user
        # was looking at it, otherwise the currently-active one.
        if active_entry:
            self._load_entry_into_viewer(active_entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Reset analysis: cleared {removed_total} peak rows on {scope_label} "
            f"(plus all manual peaks)"
        )

    def _action_export_figure(self) -> None:
        """Open the non-modal Figure Export window.

        The window is built lazily so cold startup doesn't pay
        matplotlib / mlgidbase import cost. A single instance per
        main window is reused across re-opens so the user's
        settings persist for the GUI session.
        """
        if self.session is None:
            QMessageBox.information(
                self, "No file open",
                "Open a NeXus file before exporting a figure.",
            )
            return
        if not isinstance(self.session, NexusSession):
            QMessageBox.information(
                self, "Figure export needs a NeXus file",
                "The figure exporter renders detected, fitted, and "
                "matched peak overlays from a processed NeXus file. "
                "Run the conversion on your raw data first.",
            )
            return
        if self._figure_export_window is None:
            from mlgidgui.figure_export_window import FigureExportWindow
            self._figure_export_window = FigureExportWindow(self)
        else:
            # Window already exists — refresh its cached mlgidbase
            # handle in case the user swapped files since it was
            # last shown.
            self._figure_export_window.refresh_for_session()
        self._figure_export_window.show()
        self._figure_export_window.raise_()
        self._figure_export_window.activateWindow()

    def _action_export_csv(self) -> None:
        """Pop the kind/scope dialog and write peaks to a CSV.

        NeXus-only — raw sessions don't have peak datasets. The actual
        flatten + write lives in ``file_model.export_peaks_csv`` /
        ``export_matched_csv``; the GUI's job here is dialog wiring,
        scope resolution, and the silx detach/reattach that frees the
        file's HDF5 handle for r-mode reads.
        """
        if self.session is None or self.session.kind != "nexus":
            QMessageBox.information(
                self, "Export peaks",
                "Open a NeXus file first — raw files have no peak datasets.",
            )
            return
        active_entry = self.entry_combo.currentText()
        if not active_entry:
            QMessageBox.information(
                self, "Export peaks", "No active entry to export from."
            )
            return
        n_frames = getattr(self.viewer, "n_frames", 0)
        dlg = _ExportPeaksDialog(self, has_multiple_frames=n_frames > 1)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind = dlg.selected_kind()
        scope = dlg.selected_scope()

        # Suggest a filename rooted at the original-file basename so
        # batched exports from multiple opens don't collide on disk.
        base = self.session.original_path.stem
        suggest = f"{base}_{kind}_{scope}.csv"
        path, _ = file_dialogs.save_file(
            self, "Export peaks as CSV",
            "CSV (*.csv);;All files (*)",
            suggested_name=suggest,
            default_suffix="csv",
        )
        if not path:
            return

        # Resolve the scope into an (entry, frame|None) target list
        # consumed by the file_model exporters.
        if scope == "all":
            try:
                entries = file_model.list_entries(self.session.temp_path)
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Could not list entries: {exc}")
                return
            targets: list[tuple[str, int | None]] = [(e, None) for e in entries]
        elif scope == "entry":
            targets = [(active_entry, None)]
        else:
            targets = [(active_entry, int(self.viewer.current_frame))]

        # silx may hold a read handle on the temp file; detach so h5py
        # can open it without contention.
        with self._detached_silx_tree():
            try:
                if kind == "matched":
                    n = file_model.export_matched_csv(
                        self.session.temp_path, targets, Path(path)
                    )
                else:
                    n = file_model.export_peaks_csv(
                        self.session.temp_path, targets, kind, Path(path)
                    )
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))
                return

        self.statusBar().showMessage(
            f"Wrote {n} {kind} peak rows ({scope}) to {path}", 6000
        )
        self.pipeline_panel.append_log(
            f"Exported {n} {kind} peak rows ({scope}) to {path}"
        )

    def _confirm_clear(
        self, kind: str, scope_label: str = "", linked: bool = False,
    ) -> bool:
        """Confirm a clear, naming exactly what it will remove.

        ``linked`` says the fitted/detected link will widen a detected
        clear to fitted (and so matched) — the dialog has to say so
        before anything goes, since no clear is undoable.
        """
        descriptions = {
            "detected": (
                ("detected + fitted + matched peaks",
                 "every row of detected_peaks AND fitted_peaks AND every "
                 "matched_* solution (each fit belongs to the detected "
                 "peak it came from, and matched references fitted)")
                if linked else
                ("detected peaks", "every row of detected_peaks")
            ),
            "fitted":   ("fitted + matched peaks",
                         "every row of fitted_peaks AND every matched_* "
                         "solution (matched references fitted, so it has "
                         "to go too)"),
            "matched":  ("matched peaks",
                         "every matched_* solution "
                         "(detected and fitted peaks are left intact)"),
        }
        title, body = descriptions.get(kind, (kind, kind))
        scope_suffix = f" on {scope_label}" if scope_label else ""
        reply = QMessageBox.question(
            self,
            f"Clear {title}",
            f"Remove {body}{scope_suffix}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _build_view_menu(self) -> None:
        """Expose dock visibility toggles in a top-level View menu.

        Each dock already has a built-in ``toggleViewAction()`` whose label
        and check state stay in sync with the dock — reusing them keeps the
        menu correct without manual bookkeeping.
        """
        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self._tree_dock,
            self._display_dock,
            self._pipeline_dock,
            self._sim_dock,
            self._conversion_dock,
            self._logs_dock,
            self._peaks_dock,
            self._profile_dock,
            self._scan_tracking_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        # Toggle for the cursor-readout segment of the status bar — some
        # users find the per-pixel readout distracting; on by default.
        self.action_toggle_cursor_readout = QAction(
            "Show cursor readout", self
        )
        self.action_toggle_cursor_readout.setCheckable(True)
        self.action_toggle_cursor_readout.setChecked(True)
        self.action_toggle_cursor_readout.toggled.connect(
            self._set_cursor_readout_visible
        )
        view_menu.addAction(self.action_toggle_cursor_readout)

        # Reset layout — restores the dock arrangement captured at
        # cold startup (see ``_capture_default_layout``). Useful when
        # the user has drag-rearranged things and wants to start
        # over without restarting the app.
        view_menu.addSeparator()
        self.action_reset_layout = QAction("Reset layout", self)
        self.action_reset_layout.setToolTip(
            "Restore the default dock arrangement."
        )
        self.action_reset_layout.triggered.connect(self._reset_layout)
        view_menu.addAction(self.action_reset_layout)

        # F11 fullscreen — hides every dock so the image viewer
        # owns the whole window. Menu bar stays so F11 / View
        # remains reachable. Checkable so the menu reads its state
        # back; toggled() drives the same path as the F11 keypress.
        self.action_fullscreen = QAction("&Fullscreen image viewer", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setToolTip(
            "Maximise the image viewer by hiding every dock. F11 "
            "toggles back."
        )
        self.action_fullscreen.toggled.connect(self._set_fullscreen)
        view_menu.addAction(self.action_fullscreen)

        # Theme submenu — Dark (default) / Light. Both checkable +
        # mutually exclusive via QActionGroup so the menu reads as
        # a radio choice. Selection persists via QSettings; applied
        # at startup in ``__init__``.
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self.action_theme_dark = QAction("&Dark", self)
        self.action_theme_dark.setCheckable(True)
        self.action_theme_light = QAction("&Light", self)
        self.action_theme_light.setCheckable(True)
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        theme_group.addAction(self.action_theme_dark)
        theme_group.addAction(self.action_theme_light)
        theme_menu.addAction(self.action_theme_dark)
        theme_menu.addAction(self.action_theme_light)
        self.action_theme_dark.triggered.connect(lambda: self._set_theme("dark"))
        self.action_theme_light.triggered.connect(lambda: self._set_theme("light"))
        # Sync the menu's check state with whatever's persisted /
        # currently active. ``_apply_persisted_theme`` (called once
        # at startup) writes self._current_theme.
        current = getattr(self, "_current_theme", "dark")
        (self.action_theme_dark if current == "dark"
         else self.action_theme_light).setChecked(True)

    def _set_fullscreen(self, on: bool) -> None:
        """Enter / leave the image-viewer-only fullscreen mode.

        On enter: snapshot every dock's current visibility, then
        hide them. On exit: restore the snapshotted states. The
        menu bar is left alone so the user has a discoverable way
        out beyond the F11 shortcut.
        """
        docks = [
            self._tree_dock,
            self._display_dock,
            self._pipeline_dock,
            self._sim_dock,
            self._conversion_dock,
            self._logs_dock,
            self._peaks_dock,
            self._profile_dock,
        ]
        if on:
            self._dock_visibility_before_fullscreen = {
                id(d): d.isVisible() for d in docks
            }
            for d in docks:
                d.setVisible(False)
        else:
            saved = getattr(self, "_dock_visibility_before_fullscreen", None)
            if saved is None:
                # No snapshot (e.g. user toggled the action via the
                # menu before any fullscreen entry). Fall back to
                # the mode-driven defaults so the layout doesn't end
                # up empty.
                self._apply_session_mode(self._active_session)
            else:
                for d in docks:
                    d.setVisible(bool(saved.get(id(d), True)))
            self._dock_visibility_before_fullscreen = None

    def _set_theme(self, theme: str) -> None:
        """Apply ``"dark"`` or ``"light"`` immediately, then persist
        via QSettings so next launch starts the same way.

        Swaps the full qdarkstyle stylesheet on the live QApplication,
        then **forces a re-polish of every widget** so the change is
        visible right away (Qt does not reliably restyle already-built
        widgets when the app stylesheet is replaced — without this the
        chrome only updates on the next restart). Finally it recolours
        the already-constructed pyqtgraph plots, whose colours are baked
        in at creation from ``pg.setConfigOption`` and so would otherwise
        stay on the old theme until the next render.
        """
        if theme not in ("dark", "light"):
            theme = "dark"
        from mlgidgui.theme import apply_dark_theme, apply_light_theme, pg_colors
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            if theme == "light":
                apply_light_theme(app)
            else:
                apply_dark_theme(app)
            # Force every existing widget to re-evaluate the new stylesheet.
            for w in app.allWidgets():
                try:
                    w.style().unpolish(w)
                    w.style().polish(w)
                    w.update()
                except Exception:
                    logger.debug("suppressed exception repolishing widget", exc_info=True)
        # Repaint the SVG icons. A QIcon is a value, so the widgets
        # holding one keep the old colour until it is re-set; icons.bind
        # recorded them, and this walks that registry rather than every
        # widget in the app.
        try:
            from mlgidgui import icons
            icons.retheme(theme)
            # The wordmark is a coloured asset with a per-theme variant,
            # not a recolourable glyph, so it is swapped rather than
            # retinted.
            view = getattr(self, "welcome_view", None)
            if view is not None:
                view.set_theme(theme)
            # Same for the status bar's unsaved-changes dot: it is a
            # pixmap painted in the accent, so it has to be repainted
            # rather than restyled.
            dot = getattr(self, "_sb_dirty", None)
            if dot is not None:
                dot.setPixmap(_dirty_dot())
            # Dock tab icons are not in that registry: Qt owns those
            # QTabBars and only reads a dock's windowIcon when it builds
            # the tab, so they have to be re-pushed by hand.
            self._apply_dock_tab_icons()
            # Nor are the Structure tree's rows: a QTreeWidgetItem is not
            # a QObject, takes a column with its icon, and is thrown away
            # every time its parent is re-listed. Same fix, same reason.
            panel = getattr(self, "structure_panel", None)
            if panel is not None:
                panel.node_tree.retheme(theme)
        except Exception:
            logger.debug("suppressed exception rethemeing icons", exc_info=True)
        # Recolour the live pyqtgraph plots (config options only affect
        # newly-created items, so existing axes/backgrounds need an
        # explicit push).
        background, foreground = pg_colors(theme)
        for w in (
            getattr(self, "viewer", None),
            getattr(self, "profile_viewer", None),
            getattr(self, "_phase_views_window", None),
            getattr(self, "_controls_dialog", None),
        ):
            fn = getattr(w, "apply_theme_colors", None)
            if fn is not None:
                try:
                    fn(background, foreground)
                except Exception:
                    logger.debug("suppressed exception in apply_theme_colors", exc_info=True)
        self._current_theme = theme
        # Persist.
        try:
            QSettings().setValue(self._THEME_KEY, theme)
        except Exception:
            logger.debug("suppressed exception in MainWindow._set_theme", exc_info=True)
            pass

    def _reset_layout(self) -> None:
        """Restore the dock arrangement captured at cold startup.

        Two steps:

        1. ``restoreState`` with the cached snapshot — pops every
           dock back to its original area, undoes user drags, and
           re-applies original sizes.
        2. Re-run ``_apply_session_mode`` on the active session so
           the mode-specific tabify order (Display | Pipeline |
           Peaks | Logs for NeXus, Display | Conversion | Peaks |
           Logs for raw) is reapplied. The snapshot only captures
           the cold-start layout (no session), so without this
           second step a raw session reset would leave the user
           looking at the Pipeline dock instead of Conversion.
        """
        state = getattr(self, "_default_layout_state", None)
        if state is not None:
            try:
                self.restoreState(state)
            except Exception:
                # restoreState raises on a malformed state blob; we
                # generated this one ourselves so it shouldn't, but
                # don't take down the GUI if it does.
                logger.debug("suppressed exception in MainWindow._reset_layout", exc_info=True)
                pass
        # Reapply the session-mode-specific tab order + show/hide
        # toggles so Conversion/Pipeline visibility lines up with
        # the active session.
        self._apply_session_mode(self._active_session)

    def _build_settings_menu(self) -> None:
        """Build the top-level Settings menu.

        Houses application-wide preferences that don't justify a
        dedicated dock or main-toolbar slot. Currently exposes the
        frame-playback settings; future entries (e.g. default
        colormap, default render quality, log-verbosity toggle) hang
        off the same menu.

        The menu is built after View so it sits at the rightmost
        position, which is where users instinctively reach for
        Settings in cross-platform apps.
        """
        settings_menu = self.menuBar().addMenu("&Settings")
        # Attribute name kept: it is the key this action's icon is
        # registered under in ``_MENU_ICONS``.
        self.action_playback_settings = QAction(
            "&Settings…", self
        )
        self.action_playback_settings.setToolTip(
            "Application settings: how the Display-dock Play button "
            "drives frame advance, and whether each fitted peak is "
            "linked to the detected peak it came from."
        )
        self.action_playback_settings.triggered.connect(
            self._action_playback_settings
        )
        settings_menu.addAction(self.action_playback_settings)

    def _action_playback_settings(self) -> None:
        """Open the application settings dialog.

        On accept, persist the dialog's values via QSettings and, if
        the play timer is currently running, re-apply the new
        interval mid-flight so the change is felt immediately. The
        next press of Play also re-reads via ``_compute_play_schedule``
        so a setting change applied while paused still takes effect.

        The peak-link settings need no re-apply: every consumer reads
        them at the moment it acts (see ``mlgidgui.peak_link``).
        """
        dlg = _SettingsDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dlg.save_to_qsettings()
        # The extra-list registry needs a real re-apply: the viewer's
        # overlay items, the dock's rows and the frame's peak tables all
        # have to move together, and a newly registered list has never
        # been read from the file.
        self._apply_peak_list_specs()
        # The profile-masking option is read when the profiles are
        # drawn, so redraw them now for the change to show.
        self.profile_viewer.refresh()
        # The fit-area outline is read when the overlays are drawn.
        self.viewer.refresh_overlays()
        # If playback is currently running, push the new schedule onto
        # the timer right away. The next tick will use it.
        if self._play_timer.isActive():
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )

    def _action_undo(self) -> None:
        # Two histories, one shortcut. The Structure tab owns Ctrl+Z only
        # while it is the tab in front and has an edit to reverse;
        # everywhere else this is unchanged and reaches the viewer, which
        # covers manual add/remove, manual geom edits, and detected /
        # fitted geom edits. File-level peak deletes stay non-undoable —
        # see the confirmation dialog in _on_delete_peak_requested.
        if self._structure_owns_undo() and self._structure_undo():
            return
        if hasattr(self, "viewer"):
            self.viewer.undo_last_action()

    def _action_redo(self) -> None:
        if self._structure_owns_undo() and self._structure_redo():
            return
        if hasattr(self, "viewer"):
            self.viewer.redo_last_action()

    def _action_find_peak(self) -> None:
        """Modal: pick Kind + ID, select the peak in the viewer.

        Searches the current frame first, then scans every other
        frame in the active entry; on a hit in another frame the
        viewer jumps to that frame before selecting. Matched peaks
        are excluded — matched IDs reference fitted peak ids so the
        Fitted kind covers that case too, and the per-structure
        selection model would need a separate UI.
        """
        if self.session is None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Find peak by ID")
        form = QFormLayout(dlg)
        kind_combo = QComboBox()
        kind_combo.addItems(["Detected", "Fitted"])
        form.addRow("Kind:", kind_combo)
        id_spin = QSpinBox()
        id_spin.setRange(0, 999999)
        # Sensible default: continue from whatever's currently
        # selected so repeated invocations step through IDs.
        cur_sel = self.viewer.selected_peak
        if cur_sel is not None and cur_sel.kind in ("detected", "fitted"):
            kind_combo.setCurrentText(cur_sel.kind.capitalize())
            id_spin.setValue(int(cur_sel.peak_id))
        form.addRow("ID:", id_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind = kind_combo.currentText().lower()
        peak_id = int(id_spin.value())
        self._find_and_select_peak(entry, kind, peak_id)

    def _find_and_select_peak(self, entry: str, kind: str, peak_id: int) -> None:
        """Locate (entry, *, kind, peak_id) across all frames and
        select it. Tries the viewer's in-memory peak tables first
        (cheap) before falling back to per-frame disk reads."""
        current_frame = self.viewer.current_frame
        # In-memory current-frame lookup.
        peaks_now = self.viewer._frame_peaks.get(current_frame, {})
        table = peaks_now.get(kind)
        if table is not None:
            ids = [int(x) for x in table.ids]
            if peak_id in ids:
                self._select_table_row(current_frame, kind, table, ids.index(peak_id))
                return
        # Scan other frames via file_model.
        n_frames = self.viewer.n_frames
        for frame in range(n_frames):
            if frame == current_frame:
                continue
            try:
                peaks = file_model.load_peaks(
                    self.session.temp_path, entry, frame,
                )
            except Exception:
                logger.debug("suppressed exception in MainWindow._find_and_select_peak", exc_info=True)
                continue
            table = peaks.get(kind)
            if table is None or len(table) == 0:
                continue
            ids = [int(x) for x in table.ids]
            if peak_id in ids:
                # Jump to that frame, then select.
                self.viewer.set_frame(frame)
                # _frame_peaks for the new frame is populated lazily
                # by _load_entry_into_viewer at open time; re-read
                # in case the viewer's per-frame cache hasn't been
                # touched yet for this run.
                self._select_table_row(frame, kind, table, ids.index(peak_id))
                return
        QMessageBox.information(
            self,
            "Find peak",
            f"No {kind} peak with id={peak_id} found in entry {entry!r}.",
        )

    def _select_table_row(
        self, frame: int, kind: str, table, idx: int,
    ) -> None:
        """Build a ``SelectedPeak`` from row ``idx`` of ``table`` and
        push it into the viewer. Mirrors the construction inside
        ``GIWAXSImageViewer._on_select_at`` for detected/fitted hits."""
        try:
            score = float(table.score[idx])
        except Exception:
            logger.debug("suppressed exception in MainWindow._select_table_row", exc_info=True)
            score = None
        sel = SelectedPeak(
            kind=kind,
            frame=frame,
            peak_id=int(table.ids[idx]),
            radius=float(table.radius[idx]),
            angle=float(table.angle[idx]),
            radius_width=float(table.radius_width[idx]),
            angle_width=float(table.angle_width[idx]),
            is_ring=bool(table.is_ring[idx]),
            score=score,
        )
        self.viewer._set_selected(sel)

    # Menu action -> shipped glyph. Applied in one pass after the menus
    # are built rather than at each construction site: the actions live
    # in three modules, and a single table is what makes it obvious which
    # entries deliberately have no icon (the clear/reset scope submenus,
    # where an icon per scope would be noise).
    _MENU_ICONS = {
        "action_open": "file-open",
        "action_import_converted": "file-open",
        "action_save": "save",
        "action_save_as": "save-as",
        "action_close_file": "file-close",
        "action_exit": "exit",
        "action_undo": "undo",
        "action_redo": "redo",
        "action_copy_peaks": "export-csv",
        "action_find_peak": "find-peak",
        "action_reset_all": "reset-peaks",
        "action_export_figure": "export-figure",
        "action_export_csv": "export-csv",
        "action_toggle_cursor_readout": "cursor-readout",
        "action_reset_layout": "reset-layout",
        "action_fullscreen": "fullscreen",
        "action_theme_dark": "theme-dark",
        "action_theme_light": "theme-light",
        "action_playback_settings": "playback-settings",
        "action_controls": "help-controls",
        "action_about": "about",
        "action_copy_diagnostics": "copy-diagnostics",
    }

    #: Dock toggle actions carry the dock's own glyph, so the View menu
    #: reads as a list of places rather than a list of checkboxes.
    _DOCK_ICONS = {
        "_tree_dock": "dock-tree",
        "_display_dock": "dock-display",
        "_pipeline_dock": "dock-pipeline",
        "_conversion_dock": "dock-conversion",
        "_logs_dock": "dock-logs",
        "_profile_dock": "dock-profiles",
        "_peaks_dock": "dock-peaks",
        "_sim_dock": "dock-expected",
        "_scan_tracking_dock": "dock-tracking",
    }

    def _apply_menu_icons(self) -> None:
        """Give every mapped menu action its glyph.

        ``icons.bind`` registers each action, so View -> Theme repaints
        them all; an unmapped or missing glyph simply leaves the entry
        text-only.
        """
        from mlgidgui import icons

        for attr, glyph in self._MENU_ICONS.items():
            action = getattr(self, attr, None)
            if action is not None:
                icons.bind(action, glyph)
        recent = getattr(self, "_recent_menu", None)
        if recent is not None:
            icons.bind(recent.menuAction(), "file-recent")
        for attr, glyph in self._DOCK_ICONS.items():
            dock = getattr(self, attr, None)
            if dock is not None:
                icons.bind(dock.toggleViewAction(), glyph)
        self._apply_dock_tab_icons()

    def _apply_dock_tab_icons(self) -> None:
        """Put each dock's glyph on its tab in the tabified rows.

        Both tab rows ("Display | Pipeline | Expected pattern | Logs" and
        "Profiles | Peaks | Scan tracking") are text-only otherwise, which
        makes the docks easy to miss.

        Qt copies a ``QDockWidget``'s ``windowIcon`` onto its tab only
        when that tab is *created*, so a later ``setWindowIcon`` — from a
        theme flip, or from this call, which runs long after the docks
        are built — never reaches an existing tab. The glyph is therefore
        pushed onto the ``QTabBar`` directly, keyed by tab text (which is
        the dock's ``windowTitle``). Widgets whose tab text is not a dock
        title, notably the central Image/Data pair, are left alone.

        Cheap and idempotent, so it is safe to re-run whenever the tab
        set changes: a mode switch re-tabifies, and ``restoreState``
        rebuilds the bars outright.
        """
        from mlgidgui import icons

        wanted = {}
        for attr, glyph in self._DOCK_ICONS.items():
            dock = getattr(self, attr, None)
            if dock is None:
                continue
            glyph_icon = icons.icon(glyph)
            if glyph_icon.isNull():
                continue
            # Also on the dock itself, for when it is floated out.
            dock.setWindowIcon(glyph_icon)
            wanted[dock.windowTitle()] = glyph_icon
        for bar in self.findChildren(QTabBar):
            for index in range(bar.count()):
                glyph_icon = wanted.get(bar.tabText(index))
                if glyph_icon is not None:
                    bar.setTabIcon(index, glyph_icon)

    def _build_file_menu(self, file_menu) -> None:

        # Single Open action — file content is auto-classified as
        # NeXus or raw inside ``_action_open`` so users don't have to
        # pick the right entry point. Raw files are bundled into one
        # ``RawSession`` in the same way the old "Open raw" action did.
        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._action_open)
        file_menu.addAction(self.action_open)

        # For q-space maps produced OUTSIDE mlgidGUI: stack N images as
        # one scan entry in a new .h5 (no conversion runs — the pixels
        # are already reciprocal-space). Opening float-pixel images via
        # the regular Open also offers this flow automatically.
        self.action_import_converted = QAction(
            "&Import images as converted scan…", self
        )
        self.action_import_converted.triggered.connect(
            self._action_import_converted
        )
        file_menu.addAction(self.action_import_converted)

        # Recent-files submenu — populated lazily on aboutToShow so the
        # missing-file filter stays accurate across sessions.
        self._recent_menu = file_menu.addMenu("Open &recent")
        self._recent_menu.setToolTipsVisible(True)
        self._recent_menu.aboutToShow.connect(self._refresh_recent_files_menu)
        # Build once now so the menu shows real entries on first open
        # (aboutToShow only fires when the user actually opens the
        # submenu — but the parent File menu's expansion looks better
        # if the count is right from the start).
        self._refresh_recent_files_menu()

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._action_save)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save &As…", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self._action_save_as)
        file_menu.addAction(self.action_save_as)

        self.action_close_file = QAction("&Close", self)
        self.action_close_file.setShortcut(QKeySequence.StandardKey.Close)
        self.action_close_file.triggered.connect(self._action_close_file)
        file_menu.addAction(self.action_close_file)

        file_menu.addSeparator()

        self.action_exit = action_exit = QAction("E&xit", self)
        action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    # -- Recent files (QSettings-backed) --

    def _load_recent_files(self) -> list[dict]:
        """Return the persisted recent-files list as a list of dicts.

        Each entry is ``{"type": "nexus"|"raw", "path": str}``. The
        list is stored as a JSON string in QSettings to keep the
        serialization explicit and robust across PySide/Qt platforms
        (raw QStringList round-tripping has bitten us before).
        """
        settings = QSettings()
        blob = settings.value(self._RECENT_FILES_KEY, "[]")
        if not isinstance(blob, str):
            return []
        try:
            data = json.loads(blob)
        except Exception:
            logger.debug("suppressed exception in MainWindow._load_recent_files", exc_info=True)
            return []
        if not isinstance(data, list):
            return []
        return [
            d for d in data
            if isinstance(d, dict)
            and d.get("type") in ("nexus", "raw")
            and isinstance(d.get("path"), str)
        ]

    def _save_recent_files(self, items: list[dict]) -> None:
        QSettings().setValue(self._RECENT_FILES_KEY, json.dumps(items))

    def _add_recent_file(self, path: str | Path, kind: str) -> None:
        """Push ``path`` onto the front of the recent list.

        Move-to-front semantics: if ``path`` is already in the list it
        gets bubbled up to the top instead of duplicated. The list is
        capped at ``_MAX_RECENT_FILES``.
        """
        self._add_recent_files([path], kind)

    def _add_recent_files(self, paths: list[str | Path], kind: str) -> None:
        """Push a whole batch onto the recent list in ONE settings write
        and menu rebuild.

        Same net result as calling ``_add_recent_file`` per path (later
        batch items end up higher, duplicates bubble instead of
        repeating) — but a 1000-file batch open must not pay 1000
        QSettings round-trips and menu rebuilds for a list capped at
        ``_MAX_RECENT_FILES``; that loop alone froze the window for ~2 s.
        """
        if kind not in ("nexus", "raw") or not paths:
            return
        batch = [str(p) for p in paths]
        # Last occurrence wins the higher slot, matching sequential
        # move-to-front pushes.
        fresh: list[dict] = []
        seen: set[str] = set()
        for p in reversed(batch):
            if p not in seen:
                seen.add(p)
                fresh.append({"type": kind, "path": p})
        items = self._load_recent_files()
        items = [i for i in items if i.get("path") not in seen]
        items = (fresh + items)[: self._MAX_RECENT_FILES]
        self._save_recent_files(items)
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self) -> None:
        """Rebuild the submenu from the persisted list.

        Existence is intentionally NOT checked here: this runs on the
        menu's ``aboutToShow`` (GUI thread), and ``Path.exists()`` over a
        slow / network share froze the window just opening the submenu. A
        file that has since moved is handled when the user clicks it
        (``_open_recent`` warns + prunes), so stale rows cost nothing
        until then.
        """
        self._recent_menu.clear()
        items = self._load_recent_files()
        if not items:
            empty = QAction("(no recent files)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for entry in items:
            path = entry["path"]
            kind = entry["type"]
            basename = Path(path).name
            # NeXus rows show plain basename; raw rows get a "[raw] "
            # prefix so the two are distinguishable without an icon.
            label = basename if kind == "nexus" else f"[raw]  {basename}"
            action = QAction(label, self)
            # Tooltip shows the full path so the user can disambiguate
            # files with the same basename living in different folders.
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path, k=kind: self._open_recent(p, k)
            )
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear_action = QAction("Clear recent files", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_action)

    def _clear_recent_files(self) -> None:
        self._save_recent_files([])
        self._refresh_recent_files_menu()

    def _open_recent(self, path: str, kind: str) -> None:
        """Open a file from the Recent-files submenu.

        ``kind`` is the recorded classification — kept for the menu
        labels, but the open itself goes through the same worker queue
        for both kinds (the worker re-classifies by content). Drops the
        entry from the list if the file has gone missing since it was
        recorded.
        """
        p = Path(path)
        # Show the indicator BEFORE the existence stat. On an SFTP mount the
        # recent file may be on slow/cold storage, so ``p.exists()`` (a
        # network round-trip) and the load that follows can briefly freeze
        # the window; painting the indicator first means the click visibly
        # registers instead of looking like nothing happened.
        self._show_open_progress(p.name)
        if not p.exists():
            self._dismiss_open_progress()
            QMessageBox.warning(
                self,
                "Recent files",
                f"File no longer exists:\n{path}\n\n"
                "Removing from the recent list.",
            )
            items = [i for i in self._load_recent_files() if i.get("path") != path]
            self._save_recent_files(items)
            self._refresh_recent_files_menu()
            return
        # Both kinds route through the CopyWorker queue. Raw recents used
        # to take a synchronous shortcut here (RawSession.open + insertFile
        # + activate inline) — on a big beamtime file that froze the whole
        # window for the metadata walk and the first full-stack read. The
        # worker re-classifies (cheap relative to the work it moves off
        # this thread) and its raw-entry walk doubles as the combo /
        # Conversion-panel listing, so nothing is scanned twice.
        self._open_queue.append(p)
        self._process_open_queue()
