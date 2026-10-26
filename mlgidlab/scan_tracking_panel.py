"""Scan-tracking dock: mlgidBASE's official cross-frame peak tracking.

Front-end for upstream ``mlgidBASE.track_peaks`` (fitted peaks, IoU
graph clustering — see ``mlgidlab.phase_tracking``). The run goes
through the ordinary pipeline queue as ``PipelineCommand("track_peaks",
...)``; the host pushes the resulting ``TrackingPayload`` back via
``set_payload``. The panel lists one row per surviving track (span,
member count, mean position) and syncs bidirectionally with the image
viewer on FITTED peaks; the richer visualizations (trajectories,
q-map overlay, amplitude evolution, radial waterfall) live in the
separate Phase-tracking views window, opened via "Show views…".

Pure view layer: no file access, no tracking math — signals out,
host-pushed state in, mirroring the ``PeaksTablePanel`` split. Table
sorting reuses the numeric-aware helpers from ``peaks_table_panel``.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QEvent, QModelIndex, Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.peaks_table_panel import (
    _int_item,
    _num_item,
    _PeakProxy,
    _set_header,
    _text_item,
)
from mlgidlab.phase_tracking import TrackingPayload, member_ids  # noqa: F401 (member_ids re-exported for the host)
from mlgidlab.widgets import PRIMARY as _PRIMARY, set_variant as _set_variant
from mlgidlab.widgets import skin_item_view as _skin_item_view
from mlgidlab.widgets import attach_empty_hint as _attach_empty_hint

logger = logging.getLogger(__name__)

# Roles used to round-trip a row back to its track / representative peak.
_ROLE_TRACK_ID = Qt.ItemDataRole.UserRole + 11
_ROLE_FRAME = Qt.ItemDataRole.UserRole + 12
_ROLE_PEAK_ID = Qt.ItemDataRole.UserRole + 13  # fitted id; -1 = unknown


class ScanTrackingPanel(QWidget):
    """Controls + tracks table for mlgidBASE peak tracking."""

    # "Track peaks (mlgidBASE)" clicked: (iou_threshold, min_length).
    trackRequested = Signal(float, int)
    # "Show views…" clicked — host opens/raises the views window.
    showViewsRequested = Signal()
    # Row click: (frame, fitted_peak_id) of a representative member;
    # peak_id is -1 when no id could be reconstructed (jump-only).
    trackRowSelected = Signal(int, int)
    # "Show only tracked peaks" toggled — host pushes/lifts the fitted
    # overlay filter (viewer.set_fitted_visible_only).
    onlyTrackedToggled = Signal(bool)
    # "Interpolate track" clicked — host plans the gap fills and runs
    # the inject + 2D-fit + match pipeline pass (real data, not a
    # display overlay).
    interpolateRequested = Signal()
    # "Load saved tracks" clicked — the file carries a tracking table
    # that was not loaded on its own (it no longer matches the fitted
    # peaks); the host loads it anyway.
    loadSavedRequested = Signal()
    # Delete pressed on a selected table row: (track index). The host
    # removes the track from the tracking RESULTS only — no peaks are
    # deleted from the file.
    trackDeleteRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Controls row ---
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.btn_track = _set_variant(
            QPushButton("Track peaks (mlgidBASE)", self), _PRIMARY)
        self.btn_track.setToolTip(
            "Run mlgidBASE's peak tracking on the active entry: fitted "
            "peaks of every frame are linked into tracks by IoU graph "
            "clustering. Run Fitting with Frames = All frames first."
        )
        self.btn_track.clicked.connect(self._on_track_clicked)
        controls.addWidget(self.btn_track)

        controls.addWidget(QLabel("IoU ≥", self))
        self.spin_threshold = QDoubleSpinBox(self)
        self.spin_threshold.setRange(0.05, 0.95)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setValue(0.50)
        self.spin_threshold.setToolTip(
            "Minimum IoU between two fitted-peak boxes (any two frames) "
            "for them to connect in the tracking graph. Upstream "
            "default 0.50."
        )
        controls.addWidget(self.spin_threshold)

        controls.addWidget(QLabel("Min members >", self))
        self.spin_length = QSpinBox(self)
        self.spin_length.setRange(0, 9999)
        self.spin_length.setValue(10)
        self.spin_length.setToolTip(
            "A track survives only with MORE than this many member "
            "peaks (strictly greater — upstream semantics). Lower it "
            "for short scans."
        )
        controls.addWidget(self.spin_length)

        controls.addStretch(1)

        self.chk_only_tracked = QCheckBox("Show only tracked peaks", self)
        self.chk_only_tracked.setToolTip(
            "Render only the fitted peaks that belong to a surviving "
            "track, so scrubbing frames shows the tracks' consistency "
            "— anything flickering in and out was not tracked. Purely "
            "visual; nothing is changed in the file. Best-effort: a "
            "tracked peak whose fitted id could not be matched back "
            "is hidden too."
        )
        self.chk_only_tracked.setEnabled(False)
        self.chk_only_tracked.toggled.connect(self.onlyTrackedToggled)
        self.chk_only_tracked.toggled.connect(
            lambda _c: self._apply_enabled()
        )
        controls.addWidget(self.chk_only_tracked)

        self.btn_interp_track = QPushButton("Interpolate track", self)
        self.btn_interp_track.setToolTip(
            "Fill each spot track's gap frames with REAL peaks: inject "
            "a detected box at the position interpolated between the "
            "bracketing frames (box size = their mean), run the 2D fit "
            "on it, then re-match the affected frames. The new peaks "
            "are written to the file and join their track, so a peak "
            "tracked from frame x to y shows a fitted box on every "
            "frame in between. Ring tracks are left as-is (the 2D fit "
            "is a spot model). Not undoable per-click — re-run "
            "Detection/Fitting/Matching to rebuild from scratch."
        )
        self.btn_interp_track.setEnabled(False)
        self.btn_interp_track.clicked.connect(self.interpolateRequested)
        controls.addWidget(self.btn_interp_track)

        self.btn_views = QPushButton("Show views…", self)
        self.btn_views.setToolTip(
            "Open the phase-tracking views: trajectories vs frame, "
            "q-map overlay, amplitude evolution, radial waterfall."
        )
        self.btn_views.setEnabled(False)
        self.btn_views.clicked.connect(self.showViewsRequested)
        controls.addWidget(self.btn_views)

        layout.addLayout(controls)

        # Second row: tracking options that would crowd the first (the
        # dock's minimum width, and with it the window's dock layout,
        # follows the widest row).
        options = QHBoxLayout()
        options.setSpacing(6)
        options.addWidget(QLabel("Max frame gap", self))
        self.spin_window = QSpinBox(self)
        self.spin_window.setRange(0, 99999)
        self.spin_window.setValue(0)
        self.spin_window.setSpecialValueText("off")
        self.spin_window.setToolTip(
            "Max frame gap: never link two peaks more than this many "
            "frames apart, "
            "however much their boxes overlap (mlgidBASE window_size). "
            "Off (0) = no limit. Use it when a peak position recurs "
            "later in a long scan and should not be joined to an "
            "earlier track."
        )
        options.addWidget(self.spin_window)
        options.addStretch(1)

        self.btn_load_saved = QPushButton("Load saved tracks", self)
        self.btn_load_saved.setToolTip(
            "The file carries a peak-tracking table that was not loaded "
            "automatically because some of its members no longer match a "
            "fitted peak. Load it anyway (those members are skipped)."
        )
        self.btn_load_saved.clicked.connect(self.loadSavedRequested)
        self.btn_load_saved.hide()
        options.addWidget(self.btn_load_saved)

        layout.addLayout(options)

        # --- Status line ---
        self.lbl_status = QLabel(
            "No tracks yet — run Fitting on all frames, then "
            "\"Track peaks (mlgidBASE)\".",
            self,
        )
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # Where the shown tracks came from / why saved ones were not
        # loaded. Empty (hidden) after a plain "Track peaks" run.
        self.lbl_saved = QLabel("", self)
        self.lbl_saved.setWordWrap(True)
        self.lbl_saved.hide()
        layout.addWidget(self.lbl_saved)

        # --- Tracks table ---
        self._model = QStandardItemModel(self)
        _set_header(
            self._model,
            [("Track",   "Cross-frame track id — the same physical peak "
                         "linked across frames"),
             ("first",   "First frame the track appears in"),
             ("last",    "Last frame the track appears in"),
             ("frames",  "Number of distinct frames the track is "
                         "present in"),
             ("members", "Total member peaks in the track (a frame can "
                         "contribute more than one)"),
             ("|q|",     "Mean q magnitude (Å⁻¹) across the track"),
             ("a",       "Mean polar angle (°) across the track"),
             ("structure", "Matched crystal structures (CIFs) the "
                         "track's peaks belong to, dominant phase "
                         "first — empty until Matching has run")],
        )
        self._proxy = _PeakProxy(self)
        self._proxy.setSourceModel(self._model)
        self._table = QTableView(self)
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QTableView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        _skin_item_view(self._table)
        _attach_empty_hint(
            self._table,
            "No tracks yet.\nRun \"Track peaks\" to follow peaks across "
            "the frames of this scan."
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setMinimumSectionSize(18)
        self._table.selectionModel().currentRowChanged.connect(
            self._on_row_changed
        )
        self._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        # Delete on the table removes the selected track from the
        # RESULTS (host-side; the file is never touched). An event
        # filter rather than a QShortcut: it fires on the key event
        # delivered to the table itself, so Delete elsewhere in the
        # app is unaffected and no focus-dependent shortcut map is
        # involved.
        self._table.installEventFilter(self)
        layout.addWidget(self._table)

        self._payload: TrackingPayload | None = None
        self._busy = False
        self._scan_available = False
        self._repopulating = False
        self._applying_external = False
        self._default_track_tooltip = self.btn_track.toolTip()
        # Reverse lookups for the image -> table sync (fitted peaks):
        # (frame, fitted_id) -> track index, and track index -> table
        # source row. Rebuilt by set_payload.
        self._member_to_track: dict[tuple[int, int], int] = {}
        self._row_by_track: dict[int, int] = {}

    # --- Public API (host-driven) ---

    @property
    def threshold(self) -> float:
        return float(self.spin_threshold.value())

    @property
    def min_length(self) -> int:
        return int(self.spin_length.value())

    def set_saved_note(self, text: str, *, can_load: bool = False) -> None:
        """Say where the shown tracks came from, or why a saved table was
        left alone. ``can_load`` offers the "Load saved tracks" button.
        An empty ``text`` hides both."""
        self.lbl_saved.setText(text)
        self.lbl_saved.setVisible(bool(text))
        self.btn_load_saved.setVisible(bool(text) and can_load)

    def apply_settings(
        self, threshold: "float | None", length: "int | None",
        window_size: "int | None",
    ) -> None:
        """Show the settings a result was made with (loaded from a file),
        so a re-run starts from them. None leaves a control alone."""
        if threshold is not None:
            self.spin_threshold.setValue(float(threshold))
        if length is not None:
            self.spin_length.setValue(int(length))
        self.spin_window.setValue(int(window_size or 0))

    @property
    def window_size(self) -> int:
        """Max frame gap for a link; 0 = no limit."""
        return int(self.spin_window.value())

    @property
    def payload(self) -> TrackingPayload | None:
        return self._payload

    @property
    def only_tracked_checked(self) -> bool:
        return self.chk_only_tracked.isChecked()

    def set_scan_available(self, available: bool, reason: str = "") -> None:
        """Enable / disable the run button (host gates on a multi-frame
        NeXus entry + the pipeline backend being installed)."""
        self._scan_available = bool(available)
        self._apply_enabled()
        if available:
            self.btn_track.setToolTip(self._default_track_tooltip)
        elif reason:
            self.btn_track.setToolTip(reason)

    def set_busy(self, busy: bool) -> None:
        """Pipeline queue running — tracking rides the same queue."""
        self._busy = bool(busy)
        self._apply_enabled()

    def set_payload(
        self,
        payload: TrackingPayload | None,
        ids: list | None = None,
        phases: dict | None = None,
    ) -> None:
        """Repopulate from a completed upstream run.

        ``ids`` is ``phase_tracking.member_ids`` output — the
        best-effort ``(frame, fitted_peak_id)`` per payload member
        (None entries allowed). ``phases`` is
        ``phase_tracking.match_tracks_to_structures`` output
        (track index -> CIF names, dominant first) and fills the
        "structure" column; tracks without a matched member show "—".
        Passing ``payload=None`` clears.
        """
        self._payload = payload
        self._member_to_track = {}
        self._row_by_track = {}
        phases = phases or {}
        self._repopulating = True
        try:
            self._model.removeRows(0, self._model.rowCount())
            if payload is not None:
                ids = ids if ids is not None else [None] * payload.n_members
                for k in range(payload.n_tracks):
                    members = payload.track_members(k)
                    first, last, n_frames, n_members = payload.track_span(k)
                    rep_frame, rep_id = int(first), -1
                    for i in members:
                        tagged = ids[int(i)]
                        if tagged is not None:
                            rep_frame, rep_id = int(tagged[0]), int(tagged[1])
                            break
                    for i in members:
                        tagged = ids[int(i)]
                        if tagged is not None:
                            self._member_to_track[
                                (int(tagged[0]), int(tagged[1]))
                            ] = k
                    id_item = _int_item(k)
                    id_item.setData(int(k), _ROLE_TRACK_ID)
                    id_item.setData(int(rep_frame), _ROLE_FRAME)
                    id_item.setData(int(rep_id), _ROLE_PEAK_ID)
                    self._row_by_track[k] = k
                    struct_names = phases.get(k) or []
                    self._model.appendRow([
                        id_item,
                        _int_item(first),
                        _int_item(last),
                        _int_item(n_frames),
                        _int_item(n_members),
                        _num_item(payload.track_mean("radius", k)),
                        self._angle_item(payload.track_mean("angle", k)),
                        _text_item(
                            ", ".join(str(s) for s in struct_names)
                            if struct_names else "—"
                        ),
                    ])
        finally:
            self._repopulating = False
        self._table.resizeColumnsToContents()
        # Provenance notes belong to one result; the host sets a fresh
        # one after installing a result it loaded from the file.
        self.set_saved_note("")
        # No results -> the filter has nothing to whitelist; untick so
        # the fitted overlay comes back (the toggled emit routes
        # through the host, which lifts the viewer filter).
        if payload is None and self.chk_only_tracked.isChecked():
            self.chk_only_tracked.setChecked(False)
        if payload is None:
            self.lbl_status.setText(
                "No tracks yet — run Fitting on all frames, then "
                "\"Track peaks (mlgidBASE)\"."
            )
        else:
            self.lbl_status.setText(
                f"{payload.n_tracks} tracks from {payload.n_members} "
                f"fitted-peak instances (IoU ≥ {payload.threshold:.2f}, "
                f"members > {payload.length}"
                + (
                    f", gap ≤ {payload.window_size} frames"
                    if payload.window_size else ""
                )
                + f") · entry {payload.entry}."
            )
        self._apply_enabled()

    def clear(self) -> None:
        """Drop all results (entry switch, re-fit: stale references)."""
        self.set_payload(None)

    def track_for_peak(self, frame: int, peak_id: int):
        """Track index containing fitted member ``(frame, peak_id)``,
        or None."""
        return self._member_to_track.get((int(frame), int(peak_id)))

    def set_external_peak(self, frame: int, peak_id: int):
        """Mirror an image-driven FITTED-peak selection onto the table.

        Returns the track index (row highlighted + scrolled) or None
        (selection cleared). Applied under ``_applying_external`` so it
        never bounces back through ``trackRowSelected``.
        """
        track = self.track_for_peak(frame, peak_id)
        self._applying_external = True
        try:
            sel_model = self._table.selectionModel()
            sel_model.clearCurrentIndex()
            sel_model.clearSelection()
            if track is None:
                return None
            src_row = self._row_by_track.get(track)
            if src_row is None:
                return track
            proxy_idx = self._proxy.mapFromSource(
                self._model.index(src_row, 0)
            )
            self._table.selectRow(proxy_idx.row())
            self._table.scrollTo(
                proxy_idx, QTableView.ScrollHint.PositionAtCenter
            )
            return track
        finally:
            self._applying_external = False

    def clear_external_selection(self) -> None:
        """Drop the table's row highlight without emitting."""
        self._applying_external = True
        try:
            sel_model = self._table.selectionModel()
            sel_model.clearCurrentIndex()
            sel_model.clearSelection()
        finally:
            self._applying_external = False

    # --- Internals ---

    @staticmethod
    def _angle_item(value: float):
        return (
            _num_item(value) if np.isfinite(value) else _text_item("—")
        )

    def _apply_enabled(self) -> None:
        self.btn_track.setEnabled(self._scan_available and not self._busy)
        has_tracks = (
            self._payload is not None and self._payload.n_tracks > 0
        )
        self.btn_views.setEnabled(has_tracks)
        self.chk_only_tracked.setEnabled(has_tracks)
        # Gap filling needs tracks to plan from and an idle pipeline
        # (it enqueues an inject+fit pass followed by a re-match).
        self.btn_interp_track.setEnabled(has_tracks and not self._busy)

    def _on_track_clicked(self) -> None:
        self.trackRequested.emit(self.threshold, self.min_length)

    def _on_row_changed(
        self, current: QModelIndex, _previous: QModelIndex
    ) -> None:
        if (
            self._repopulating
            or self._applying_external
            or not current.isValid()
        ):
            return
        id_idx = current.siblingAtColumn(0)
        frame = id_idx.data(_ROLE_FRAME)
        peak_id = id_idx.data(_ROLE_PEAK_ID)
        if frame is None or peak_id is None:
            return
        self.trackRowSelected.emit(int(frame), int(peak_id))

    def eventFilter(self, obj, event) -> bool:
        if (
            obj is self._table
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Delete
        ):
            self._on_delete_pressed()
            return True
        return super().eventFilter(obj, event)

    def _on_delete_pressed(self) -> None:
        """Emit the selected row's track index for host-side removal."""
        current = self._table.selectionModel().currentIndex()
        if not current.isValid():
            return
        track = current.siblingAtColumn(0).data(_ROLE_TRACK_ID)
        if track is None:
            return
        self.trackDeleteRequested.emit(int(track))
