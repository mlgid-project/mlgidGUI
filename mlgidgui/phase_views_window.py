"""Phase-tracking views window (non-modal, lazily built, kept alive).

Renders one ``phase_tracking.TrackingPayload`` — the exact arrays
mlgidBASE's ``track_peaks`` computed — as four interactive pyqtgraph
tabs, following the ideas prototyped in the peak_matching_frames
notebook:

1. **Trajectories** — per-track colored trace of a selectable axis
   (|q| / angle / amplitude / q_xy / q_z) vs frame number (upstream's
   "evolution" figure, interactive). Clicking a point asks the host to
   jump the image viewer to that frame.
2. **q-map overlay** — per-track trajectories and mean-position markers
   in (q_xy, q_z), optionally over the nan-mean Cartesian image of a
   chosen frame window (worker-computed).
3. **Amplitude evolution** — per-track amplitude vs frame,
   median-smoothed and nanmax-normalized like the notebook.
4. **Radial waterfall** — frame × |q| intensity heatmap of the whole
   scan (worker-computed, cached per entry), independent of tracking.

The window never touches the tracking math: the payload is pushed by
the host, and the two image aids are computed by
``workers.ScanProfileWorker`` on QThreads owned here (read-only, own
h5py handles). "Save official mlgidBASE figures…" emits
``saveFiguresRequested`` for the host to run upstream's
``plot_tracked_peaks`` on the GUI thread.
"""
from __future__ import annotations

import csv
import logging
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlgidgui import file_dialogs
from mlgidgui import roi_intensity
from mlgidgui.phase_tracking import (
    AXIS_LABELS,
    AXIS_NAMES,
    TrackingPayload,
    smooth_normalize_amplitude,
)
from mlgidgui.widgets import make_debounced_timer
from mlgidgui.widgets import PRIMARY as _PRIMARY, set_variant as _set_variant
from mlgidgui.workers import RoiTraceWorker, ScanProfileWorker, stop_worker_thread

logger = logging.getLogger(__name__)


# Group key for tracks with no matched phase.
# Re-exports: canonical homes moved in the 2026 source split; kept here
# so tests and older references resolve unchanged.
from mlgidgui.phase_views_dialogs import (
    _UNMATCHED,
    _ExportDialog,
    _TrackSelectDialog,
    _log_display,
    _track_pen,
)


class PhaseViewsWindow(QMainWindow):
    """Four interactive views over one tracking payload."""

    frameJumpRequested = Signal(int)
    # (base_path, axis) — host runs upstream track_peaks with
    # save_fig=True on the GUI thread (silx detached).
    saveFiguresRequested = Signal(str, str)

    def __init__(
        self, parent: QWidget | None = None, busy_probe=None
    ) -> None:
        """``busy_probe``: optional callable returning True while the
        host runs a pipeline command. pygid opens the NeXus file in
        r+ mode even for reads, so a view worker's long-lived read
        handle and a pipeline run must never overlap — the probe lets
        this window refuse to start a worker mid-run (the host guards
        the opposite direction)."""
        super().__init__(parent)
        self._busy_probe = busy_probe
        self.setWindowTitle("Phase tracking views")
        self.resize(900, 640)

        self._payload: TrackingPayload | None = None
        self._file_path = None
        self._entry: str | None = None
        self._n_frames = 0
        # Worker-computed caches, keyed by (file_path, entry).
        self._waterfall: dict | None = None
        self._mean_image: dict | None = None
        self._worker_thread: QThread | None = None
        self._worker: ScanProfileWorker | None = None
        # Phase identity for the q-map + amplitude grouping:
        # {track_index: [cif, ...]} (dominant first) + a stable colour
        # per unique CIF. ``_struct_checks`` maps each structure key
        # (CIF or _UNMATCHED) to its show/hide checkbox; ``_amp_labels``
        # holds the grouped-view TextItems for teardown.
        self._track_phases: dict = {}
        # {member index: [cif, ...]} — which members a matched structure
        # actually claimed ON THEIR FRAME. Non-empty, it switches the
        # member-based plots to per-frame phase attribution (a track
        # matched only from frame z renders its earlier members as
        # unmatched); empty, attribution stays track-level (dominant
        # phase paints the whole track).
        self._member_phases: dict = {}
        self._phase_colors: dict = {}
        # Custom per-CIF colours pushed by the host (picked in the
        # matched-peaks legend); layered over the automatic hue wheel
        # in ``_rebuild_phase_colors`` so both windows agree.
        self._phase_color_overrides: dict = {}
        self._struct_checks: dict = {}
        # {structure key: set of track indices} the user UNTICKED in the
        # Select-tracks dialog — exclusions, so new tracks default to
        # included. Feeds _amp_groups (amplitude tab + its CSV only);
        # reset on every set_payload since track indices shift on
        # deletion and a new run invalidates the choice.
        self._amp_excluded: dict = {}
        self._amp_track_dialog: QDialog | None = None
        # (id(payload), n_tracks) of the last payload push — the reset
        # trigger for the selection (see set_payload).
        self._amp_payload_fp: tuple | None = None
        self._amp_labels: list = []
        # Cached ROI-intensity traces of the "integrated intensity
        # (ROI)" metric: {"frames", "traces", "params", "fp",
        # "target"} — valid only while payload fingerprint, ROI
        # parameters and (file, entry) still match (_roi_cache_valid).
        # ``_roi_pending`` stamps fp/target of an in-flight
        # RoiTraceWorker run.
        self._roi_traces: dict | None = None
        self._roi_pending: dict | None = None
        # Interquartile-band items of the amplitude tab's median mode.
        # FillBetweenItems are NOT pyqtgraph data items, so
        # ``_clear_overlays`` misses them — torn down explicitly.
        self._amp_fills: list = []
        # Track indices that are rings — drawn on the q-map as dashed
        # quarter-circle arcs at their mean |q|, not as point trajectories.
        self._ring_tracks: set = set()
        # Temporary render overrides applied while exporting images
        # (structure subset, line width/style, marker size). Empty in
        # normal operation; the style helpers consult it.
        self._export_overrides: dict = {}
        # Last theme applied by the host — restored after a white-
        # background export.
        self._theme_colors = (
            pg.getConfigOption("background"),
            pg.getConfigOption("foreground"),
        )

        # Central layout: a window-wide structure-selection bar above the
        # tabs (so the per-CIF show/hide toggles govern every phase view —
        # trajectories, q-map, amplitude — from any tab), then the tabs.
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self._struct_toggle_widget = QWidget(central)
        self._struct_toggle_layout = QHBoxLayout(self._struct_toggle_widget)
        self._struct_toggle_layout.setContentsMargins(6, 2, 6, 2)
        self._struct_toggle_layout.addWidget(QLabel("Structures:", central))
        self._struct_toggle_layout.addStretch(1)
        self._struct_toggle_widget.setVisible(False)
        central_layout.addWidget(self._struct_toggle_widget)
        # Window-wide frame interval for the member-based plots
        # (trajectories + amplitude evolution): only members whose frame
        # falls inside [from, to] are drawn. The q-map keeps the full
        # tracks (its mean-image window is a separate control) and the
        # waterfall is frame-complete by nature.
        frame_row_widget = QWidget(central)
        frame_row = QHBoxLayout(frame_row_widget)
        frame_row.setContentsMargins(6, 2, 6, 2)
        frame_row.setSpacing(6)
        frame_row.addWidget(QLabel("Frames:", central))
        self.frame_lo = QSpinBox(frame_row_widget)
        self.frame_hi = QSpinBox(frame_row_widget)
        for spin in (self.frame_lo, self.frame_hi):
            spin.setRange(0, 0)
            spin.setEnabled(False)
            spin.setToolTip(
                "Show only the frame interval [from, to] in the "
                "trajectories and amplitude-evolution plots (the q-map "
                "and waterfall are unaffected). Set from=first and "
                "to=last to see everything again."
            )
            spin.valueChanged.connect(self._on_frame_interval_changed)
        frame_row.addWidget(self.frame_lo)
        frame_row.addWidget(QLabel("to", central))
        frame_row.addWidget(self.frame_hi)
        frame_row.addStretch(1)
        central_layout.addWidget(frame_row_widget)
        self._tabs = QTabWidget(central)
        central_layout.addWidget(self._tabs)
        self.setCentralWidget(central)

        self._build_trajectories_tab()
        self._build_qmap_tab()
        self._build_amplitude_tab()
        self._build_waterfall_tab()

        act_data = QAction("Export plot data (CSV)…", self)
        act_data.setToolTip(
            "Write the numbers behind the selected plots to CSV files "
            "(one per plot) for further analysis or custom plotting."
        )
        act_data.triggered.connect(self._on_export_data)
        act_images = QAction("Export plot images…", self)
        act_images.setToolTip(
            "Render the selected plots to PNG/SVG, with export styling: "
            "structure subset, line width/style, marker size, output "
            "width, white background."
        )
        act_images.triggered.connect(self._on_export_images)
        act_save = QAction("Save official mlgidBASE figures…", self)
        act_save.setToolTip(
            "Rerun mlgidBASE track_peaks and save its own tracking "
            "figure (q_xy vs q_z next to the chosen axis against frame "
            "number) to the file you pick. The amplitude axis is not "
            "available upstream and is drawn as radius."
        )
        act_save.triggered.connect(self._on_save_figures)
        toolbar = self.addToolBar("Export")
        toolbar.setObjectName("PhaseViewsExportBar")
        toolbar.setMovable(False)
        toolbar.addAction(act_data)
        toolbar.addAction(act_images)
        toolbar.addSeparator()
        toolbar.addAction(act_save)

        self.statusBar().showMessage("No tracking results yet.")

    # --- Tab construction ---

    def _controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        return row

    def _make_plot(self, x_label: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(self)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.2)
        return plot

    def _build_trajectories_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Axis:", tab))
        self.axis_combo = QComboBox(tab)
        for name in AXIS_NAMES:
            self.axis_combo.addItem(AXIS_LABELS[name], userData=name)
        self.axis_combo.setCurrentIndex(AXIS_NAMES.index("radius"))
        self.axis_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_trajectories()
        )
        controls.addWidget(self.axis_combo)
        controls.addWidget(QLabel("Colour:", tab))
        self.traj_color = QComboBox(tab)
        self.traj_color.addItem("fitted (per track)", userData="fitted")
        self.traj_color.addItem("matched (per phase)", userData="matched")
        self.traj_color.setToolTip(
            "'fitted' gives each track its own colour; 'matched' colours "
            "tracks by the matched crystal phase (CIF) they belong to, so "
            "tracks of the same phase share a colour (unmatched grey). "
            "Needs Matching to have run."
        )
        self.traj_color.model().item(1).setEnabled(False)  # 'matched' off until phases
        self.traj_color.currentIndexChanged.connect(
            lambda _i: self._refresh_trajectories()
        )
        controls.addWidget(self.traj_color)
        controls.addStretch(1)
        hint = QLabel("Click a point to jump the viewer to that frame.", tab)
        controls.addWidget(hint)
        layout.addLayout(controls)
        self.traj_plot = self._make_plot("Frame #", AXIS_LABELS["radius"])
        layout.addWidget(self.traj_plot)
        self._tabs.addTab(tab, "Trajectories")

    def _build_qmap_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Mean image frames", tab))
        self.qmap_lo = QSpinBox(tab)
        self.qmap_hi = QSpinBox(tab)
        for spin in (self.qmap_lo, self.qmap_hi):
            spin.setRange(0, 0)
            controls.addWidget(spin)
        self.btn_mean_image = _set_variant(
            QPushButton("Compute mean image", tab), _PRIMARY)
        self.btn_mean_image.setToolTip(
            "Average the Cartesian frames in the chosen window "
            "(nan-aware) and draw it under the trajectories."
        )
        self.btn_mean_image.clicked.connect(self._on_compute_mean_image)
        controls.addWidget(self.btn_mean_image)
        self.qmap_log = QCheckBox("Log intensity", tab)
        self.qmap_log.setChecked(True)
        self.qmap_log.toggled.connect(lambda _c: self._refresh_qmap())
        controls.addWidget(self.qmap_log)
        self.qmap_phase = QCheckBox("Color by matched phase", tab)
        self.qmap_phase.setToolTip(
            "Colour each track by the matched crystal phase (CIF) its "
            "tracked peaks belong to, with a legend; unmatched tracks "
            "are grey. Needs Matching to have run."
        )
        self.qmap_phase.setEnabled(False)
        self.qmap_phase.toggled.connect(lambda _c: self._refresh_qmap())
        controls.addWidget(self.qmap_phase)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.qmap_plot = self._make_plot("q_xy (Å⁻¹)", "q_z (Å⁻¹)")
        self.qmap_plot.getPlotItem().getViewBox().setAspectLocked(True)
        self.qmap_image = pg.ImageItem()
        self.qmap_image.setZValue(-10)
        self._apply_image_cmap(self.qmap_image)
        self.qmap_plot.addItem(self.qmap_image)
        self._qmap_legend = self.qmap_plot.getPlotItem().addLegend(
            offset=(-10, 10)
        )
        self._qmap_legend.setVisible(False)
        layout.addWidget(self.qmap_plot)
        self._tabs.addTab(tab, "q-map overlay")

    def _build_amplitude_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Metric:", tab))
        self.amp_metric = QComboBox(tab)
        self.amp_metric.addItem("amplitude", userData="amplitude")
        self.amp_metric.addItem(
            "integrated intensity (ROI)", userData="roi"
        )
        self.amp_metric.setToolTip(
            "What the curves show per tracked peak and frame: the "
            "FITTED peak amplitude (from the per-frame fit results), "
            "or the background-subtracted INTEGRATED intensity of a "
            "q-space ROI box around the tracked position, summed from "
            "the image data itself. The ROI metric has a value on "
            "EVERY frame (a failed fit leaves no gap) and is "
            "proportional to the diffracting volume; it costs one "
            "full read of the scan (computed in the background and "
            "cached until tracking or the ROI settings change)."
        )
        self.amp_metric.currentIndexChanged.connect(
            lambda _i: self._on_metric_changed()
        )
        controls.addWidget(self.amp_metric)
        controls.addWidget(QLabel("Smoothing window", tab))
        self.amp_window = QSpinBox(tab)
        self.amp_window.setRange(1, 99)
        self.amp_window.setSingleStep(2)
        self.amp_window.setValue(7)
        self.amp_window.setToolTip(
            "Median-filter size over each track's frame-ordered "
            "amplitudes (1 = raw)."
        )
        self.amp_window.valueChanged.connect(
            lambda _v: self._refresh_amplitude()
        )
        controls.addWidget(self.amp_window)
        self.amp_normalize = QCheckBox("Normalize per track", tab)
        self.amp_normalize.setChecked(True)
        self.amp_normalize.toggled.connect(
            lambda _c: self._refresh_amplitude()
        )
        controls.addWidget(self.amp_normalize)
        self.amp_zero = QCheckBox("Zeros at start/end", tab)
        self.amp_zero.setToolTip(
            "Median mode: pad each structure's MEDIAN curve with "
            "zero-intensity points from the visible start to its first "
            "observed frame and from its last one to the visible end, "
            "so the evolution rises from and falls back to zero. The "
            "zeros are added to the finished median only — individual "
            "tracks never contribute zeros to the statistics, so an "
            "absent track cannot drag the median down. Gaps inside the "
            "series stay open."
        )
        self.amp_zero.setEnabled(False)
        self.amp_zero.toggled.connect(lambda _c: self._refresh_amplitude())
        controls.addWidget(self.amp_zero)
        self.btn_amp_tracks = QPushButton("Select tracks…", tab)
        self.btn_amp_tracks.setToolTip(
            "Choose per matched structure which tracks feed the "
            "grouped bands and per-structure medians — a sortable "
            "table (including mean amplitude) with a checkbox per "
            "track, applied live. Needs Matching to have run; a new "
            "tracking run resets the selection."
        )
        self.btn_amp_tracks.setEnabled(False)
        self.btn_amp_tracks.clicked.connect(self._on_select_amp_tracks)
        controls.addWidget(self.btn_amp_tracks)
        self.amp_group = QCheckBox("Group by structure", tab)
        self.amp_group.setToolTip(
            "Stack the tracks by matched crystal phase (CIF): each "
            "structure at its own vertical offset (amplitudes are "
            "normalized, so the y-scale is arbitrary), coloured by "
            "phase. Needs Matching to have run."
        )
        self.amp_group.setEnabled(False)
        self.amp_group.toggled.connect(lambda _c: self._refresh_amplitude())
        controls.addWidget(self.amp_group)
        self.amp_median = QCheckBox("Median per structure", tab)
        self.amp_median.setToolTip(
            "Draw ONE curve per matched structure: the per-frame MEDIAN "
            "amplitude across all of the structure's tracks, with a "
            "transparent interquartile band (25th-75th percentile) "
            "around it — instead of every individual track. Respects "
            "the smoothing window and (ungrouped) the normalize toggle; "
            "combines with \"Group by structure\" to stack one median "
            "band per structure. Needs Matching to have run."
        )
        self.amp_median.setEnabled(False)
        self.amp_median.toggled.connect(lambda _c: self._refresh_amplitude())
        controls.addWidget(self.amp_median)
        controls.addStretch(1)
        layout.addLayout(controls)
        # ROI-metric settings on their own row, visible only while the
        # ROI metric is selected. Changes debounce into one recompute.
        self._roi_controls = QWidget(tab)
        roi_row = QHBoxLayout(self._roi_controls)
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.setSpacing(6)
        roi_row.addWidget(QLabel("ROI half-width q_xy", self._roi_controls))
        self.roi_dqxy = QDoubleSpinBox(self._roi_controls)
        self.roi_dqz = QDoubleSpinBox(self._roi_controls)
        for spin, name, default in (
            (self.roi_dqxy, "q_xy", roi_intensity.DEFAULT_HALF_Q_XY),
            (self.roi_dqz, "q_z", roi_intensity.DEFAULT_HALF_Q_Z),
        ):
            spin.setRange(0.001, 1.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.005)
            spin.setValue(default)
            spin.setSuffix(" Å⁻¹")
            spin.setToolTip(
                f"Half-width of the integration box along {name}: the "
                f"box spans the tracked position ± this value."
            )
            spin.valueChanged.connect(
                lambda _v: self._roi_param_timer.start()
            )
        roi_row.addWidget(self.roi_dqxy)
        roi_row.addWidget(QLabel("q_z", self._roi_controls))
        roi_row.addWidget(self.roi_dqz)
        roi_row.addWidget(QLabel("Background gap", self._roi_controls))
        self.roi_gap = QSpinBox(self._roi_controls)
        self.roi_gap.setRange(0, 50)
        self.roi_gap.setValue(roi_intensity.DEFAULT_BG_GAP_PX)
        self.roi_strip = QSpinBox(self._roi_controls)
        self.roi_strip.setRange(1, 50)
        self.roi_strip.setValue(roi_intensity.DEFAULT_BG_STRIP_PX)
        for spin, tip in (
            (self.roi_gap,
             "Pixels between the box edge and each background strip."),
            (self.roi_strip,
             "Width of each background strip in pixels. The local "
             "background is the mean per-pixel intensity of the TWO "
             "strips flanking the box (along q_z for near-axis peaks, "
             "along q_xy otherwise), averaged together so a neighbor "
             "peak under one strip cannot over-subtract the trace."),
        ):
            spin.setSuffix(" px")
            spin.setToolTip(tip)
            spin.valueChanged.connect(
                lambda _v: self._roi_param_timer.start()
            )
        roi_row.addWidget(self.roi_gap)
        roi_row.addWidget(QLabel("strip", self._roi_controls))
        roi_row.addWidget(self.roi_strip)
        roi_row.addStretch(1)
        self._roi_controls.setVisible(False)
        layout.addWidget(self._roi_controls)
        self._roi_param_timer = make_debounced_timer(
            self, 400, self._refresh_amplitude
        )
        # The per-structure show/hide toggles live in a window-wide bar
        # (built in __init__) so they also govern the q-map/trajectories.
        self.amp_plot = self._make_plot(
            "Frame #", "Amplitude (normalized)"
        )
        layout.addWidget(self.amp_plot)
        self._tabs.addTab(tab, "Amplitude evolution")

    def _build_waterfall_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        self.btn_waterfall = _set_variant(
            QPushButton("Compute waterfall", tab), _PRIMARY)
        self.btn_waterfall.setToolTip(
            "Resample every frame to polar and stack the angular-mean "
            "radial profiles into a frame × |q| heatmap. Independent "
            "of tracking; cached per entry."
        )
        self.btn_waterfall.clicked.connect(self._on_compute_waterfall)
        controls.addWidget(self.btn_waterfall)
        self.waterfall_log = QCheckBox("Log intensity", tab)
        self.waterfall_log.setChecked(True)
        self.waterfall_log.toggled.connect(
            lambda _c: self._refresh_waterfall()
        )
        controls.addWidget(self.waterfall_log)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.waterfall_plot = self._make_plot("Frame #", "|q| (Å⁻¹)")
        self.waterfall_image = pg.ImageItem()
        self._apply_image_cmap(self.waterfall_image)
        self.waterfall_plot.addItem(self.waterfall_image)
        layout.addWidget(self.waterfall_plot)
        self._tabs.addTab(tab, "Radial waterfall")

    # --- Host API ---

    def set_context(self, file_path, entry: str, n_frames: int) -> None:
        """Point the window at the active session/entry. Invalidates
        the worker-computed caches when the target changed."""
        changed = (file_path, entry) != (self._file_path, self._entry)
        self._file_path = file_path
        self._entry = entry
        self._n_frames = int(n_frames)
        hi = max(0, self._n_frames - 1)
        for spin in (self.qmap_lo, self.qmap_hi):
            spin.setRange(0, hi)
        if changed:
            self.qmap_lo.setValue(0)
            self.qmap_hi.setValue(hi)
            self._waterfall = None
            self._mean_image = None
            self._refresh_qmap()
            self._refresh_waterfall()

    def set_payload(self, payload: TrackingPayload | None) -> None:
        self._payload = payload
        # Track indices are only meaningful against one payload SHAPE:
        # a new tracking run (new object) or a Delete-key track removal
        # (same object, fewer tracks) shifts them, so the amplitude
        # track selection resets on either — but NOT on a plain re-push
        # of the unchanged payload (re-opening the window), which would
        # needlessly wipe the user's picks.
        fingerprint = (
            (id(payload), payload.n_tracks) if payload is not None else None
        )
        if fingerprint != self._amp_payload_fp:
            self._amp_payload_fp = fingerprint
            self._amp_excluded = {}
            # ROI traces were computed for the old payload's tracks.
            self._roi_traces = None
            if self._amp_track_dialog is not None:
                self._amp_track_dialog.close()
                self._amp_track_dialog = None
            self._update_amp_tracks_button()
        # Re-bound the frame-interval spinboxes and reset them to "show
        # everything" (a NEW tracking run invalidates any previous
        # narrowing). Preferred bounds are the SCAN's frame range (known
        # once the host has pushed set_context) so frame 0 is reachable
        # even when the first track appears late — the zero-baseline
        # padding needs it; payload range is the context-less fallback.
        # Blocked: each setValue would otherwise trigger a refresh of
        # the still-stale plots.
        has_frames = payload is not None and payload.n_members > 0
        if has_frames and self._n_frames > 0:
            lo, hi = 0, self._n_frames - 1
        else:
            lo = int(payload.frame_num.min()) if has_frames else 0
            hi = int(payload.frame_num.max()) if has_frames else 0
        for spin, val in ((self.frame_lo, lo), (self.frame_hi, hi)):
            with QSignalBlocker(spin):
                spin.setRange(lo, hi)
                spin.setValue(val)
            spin.setEnabled(has_frames)
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()
        if payload is None:
            self.statusBar().showMessage("No tracking results yet.")
        else:
            self.statusBar().showMessage(
                f"{payload.n_tracks} tracks · {payload.n_members} "
                f"members · entry {payload.entry}"
            )

    def set_track_phases(self, mapping: dict) -> None:
        """Install ``{track_index: [cif, ...]}`` (dominant phase first)
        for the q-map "Color by matched phase" overlay.

        Assigns a stable colour per unique CIF, enables the toggle only
        when there are phases (untick + disable otherwise), and
        re-renders the q-map.
        """
        self._track_phases = {int(k): list(v) for k, v in mapping.items()}
        self._rebuild_phase_colors()
        has_phases = bool(self._track_phases)
        self._rebuild_structure_toggles()
        # Trajectories 'matched' colour option follows phase availability.
        self.traj_color.model().item(1).setEnabled(has_phases)
        if not has_phases and self.traj_color.currentData() == "matched":
            self.traj_color.setCurrentIndex(0)   # triggers _refresh_trajectories
        else:
            self._refresh_trajectories()
        self.amp_group.setEnabled(has_phases)
        self.amp_median.setEnabled(has_phases)
        self.btn_amp_tracks.setEnabled(has_phases)
        if not has_phases and self.amp_median.isChecked():
            self.amp_median.setChecked(False)  # triggers _refresh_amplitude
        if not has_phases and self.amp_group.isChecked():
            self.amp_group.setChecked(False)   # triggers _refresh_amplitude
        else:
            self._refresh_amplitude()
        self.qmap_phase.setEnabled(has_phases)
        if not has_phases and self.qmap_phase.isChecked():
            self.qmap_phase.setChecked(False)   # triggers _refresh_qmap
        else:
            self._refresh_qmap()

    def set_member_phases(self, mapping: dict) -> None:
        """Install ``{member index: [cif, ...]}`` (from
        ``phase_tracking.member_matched_phases``) and re-render the
        plots that colour by phase. See ``_member_phases`` for the
        semantics; the host pushes this right after
        ``set_track_phases``, so a no-op guard keeps the second
        refresh cascade away when nothing changed."""
        mapping = {int(k): list(v) for k, v in mapping.items()}
        if mapping == self._member_phases:
            return
        self._member_phases = mapping
        self._rebuild_structure_toggles()
        self._refresh_trajectories()
        self._refresh_amplitude()
        self._refresh_qmap()

    def _member_structure_key(self, i: int, k: int) -> str:
        """Structure key of member ``i`` of track ``k``: the phase that
        claimed the member on its frame (preferring the track's dominant
        order so member and track colours agree), ``_UNMATCHED`` for a
        never-claimed member, and the track-level key when no member
        attribution exists at all."""
        if not self._member_phases:
            return self._structure_key_of(k)
        claims = self._member_phases.get(int(i))
        if not claims:
            return _UNMATCHED
        for cif in self._track_phases.get(k, []):
            if cif in claims:
                return cif
        return claims[0]

    def _member_subsets(
        self, k: int, interval: bool = True
    ) -> dict[str, np.ndarray]:
        """Track ``k``'s members split by structure key, insertion
        ordered by first occurrence. One whole-track subset under its
        dominant key when no member attribution exists — the pre-split
        rendering. ``interval=False`` skips the frame-interval filter
        (the q-map stays frame-complete)."""
        members = (
            self._interval_members(k) if interval
            else self._payload.track_members(k)
        )
        out: dict = {}
        for i in members:
            key = self._member_structure_key(int(i), k)
            out.setdefault(key, []).append(int(i))
        return {
            key: np.asarray(v, dtype=int) for key, v in out.items()
        }

    def _any_unmatched_member(self) -> bool:
        """True when the member attribution leaves any tracked member
        unclaimed — the condition for an 'unmatched' toggle/band/legend
        entry beyond the track-level one."""
        if self._payload is None or not self._member_phases:
            return False
        return any(
            not self._member_phases.get(int(i))
            for comp in self._payload.components
            for i in comp
        )

    def _rebuild_phase_colors(self) -> None:
        """Stable colour per unique CIF across the phase map: the
        automatic hue wheel, then the host's custom colours on top."""
        cifs = sorted({c for lst in self._track_phases.values() for c in lst})
        hues = max(9, len(cifs))
        self._phase_colors = {
            cif: pg.intColor(i, hues=hues, maxValue=255)
            for i, cif in enumerate(cifs)
        }
        for cif, hex_color in self._phase_color_overrides.items():
            if cif in self._phase_colors:
                self._phase_colors[cif] = pg.mkColor(hex_color)

    def set_phase_color_overrides(self, mapping: dict) -> None:
        """Install ``{cif: hex}`` custom colours (picked in the host's
        matched-peaks legend) and re-render everything that shows phase
        colours. No-op when nothing changed, so the host can push
        unconditionally."""
        mapping = {str(k): str(v) for k, v in mapping.items()}
        if mapping == self._phase_color_overrides:
            return
        self._phase_color_overrides = mapping
        self._rebuild_phase_colors()
        self._rebuild_structure_toggles()
        self._refresh_trajectories()
        self._refresh_amplitude()
        self._refresh_qmap()

    def _rebuild_structure_toggles(self) -> None:
        """Rebuild the per-structure show/hide checkboxes from the phase
        map (one per CIF present + 'unmatched' when any track has no
        phase). Checkbox text is tinted with the phase colour so it ties
        to the grouped amplitude bands / q-map legend. Hidden when there
        are no phases."""
        for chk in self._struct_checks.values():
            self._struct_toggle_layout.removeWidget(chk)
            chk.setParent(None)
            chk.deleteLater()
        self._struct_checks = {}
        # Toggles are meaningful only once Matching has identified at
        # least one phase; with no matches every track is "unmatched"
        # and the panel is just noise.
        keys = self._structure_keys() if self._track_phases else []
        for key in keys:
            color = self._structure_color(key)
            label = "unmatched" if key == _UNMATCHED else key
            chk = QCheckBox(label, self._struct_toggle_widget)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {pg.mkColor(color).name()};")
            chk.toggled.connect(lambda _c: self._on_structure_toggle())
            # Insert before the trailing stretch.
            self._struct_toggle_layout.insertWidget(
                self._struct_toggle_layout.count() - 1, chk
            )
            self._struct_checks[key] = chk
        self._struct_toggle_widget.setVisible(bool(self._struct_checks))

    def _structure_keys(self) -> list:
        """Ordered structure keys across the tracks: matched CIFs
        (sorted) then 'unmatched' if any track has no phase."""
        if self._payload is None:
            return []
        cifs = sorted({
            v[0] for v in self._track_phases.values() if v
        })
        if self._member_phases:
            # Per-frame attribution can key members to a track's
            # NON-dominant phase; give those their own entry so their
            # band/toggle exists (only keys that actually hold members,
            # so no empty bands appear).
            extra: set = set()
            for k in range(self._payload.n_tracks):
                extra.update(self._member_subsets(k, interval=False))
            extra.discard(_UNMATCHED)
            cifs = sorted(set(cifs) | extra)
        has_unmatched = any(
            not self._track_phases.get(k)
            for k in range(self._payload.n_tracks)
        ) or self._any_unmatched_member()
        return cifs + ([_UNMATCHED] if has_unmatched else [])

    def _structure_color(self, key: str):
        return self._phase_colors.get(
            key, pg.mkColor(*self._UNMATCHED_COLOR)
        )

    def _structure_key_of(self, k: int) -> str:
        cifs = self._track_phases.get(k)
        return cifs[0] if cifs else _UNMATCHED

    def _structure_key_visible(self, key: str) -> bool:
        """Whether a structure key is drawn: the export-time structure
        subset wins over the window's toggle bar while an image export
        is rendering."""
        structs = self._export_overrides.get("structures")
        if structs is not None:
            return key in structs
        chk = self._struct_checks.get(key)
        return chk is None or chk.isChecked()

    def _structure_visible(self, k: int) -> bool:
        return self._structure_key_visible(self._structure_key_of(k))

    # --- Export style overrides (image export) ---

    def _curve_pen(self, color, base_width: float = 1.0,
                   emphasized: bool = False):
        """Pen for a data curve, honouring the export overrides: the
        chosen line width (+1 for emphasized curves like the median)
        and line style. Without overrides, exactly the historical pen."""
        ov = self._export_overrides
        if "line_width" in ov:
            width = float(ov["line_width"]) + (1.0 if emphasized else 0.0)
        else:
            width = float(base_width)
        pen = pg.mkPen(color, width=width)
        style = ov.get("line_style")
        if style is not None:
            pen.setStyle(style)
        return pen

    def _emph_width(self, base: float) -> float:
        """Width for emphasized non-line pens (ring arcs, mean-position
        marker outlines) — these keep their own pen STYLE (a ring's
        dash is semantic) but follow the export line width."""
        ov = self._export_overrides
        if "line_width" in ov:
            return float(ov["line_width"]) + 1.0
        return float(base)

    def _symbol_kwargs(self, base_size: int, color) -> dict:
        """Data-point symbol kwargs for ``plot()``: export marker-size
        override applied; size 0 hides the markers entirely."""
        size = int(self._export_overrides.get("marker_size", base_size))
        if size <= 0:
            return {"symbol": None}
        return {
            "symbol": "o", "symbolSize": size,
            "symbolBrush": color, "symbolPen": None,
        }

    def _on_structure_toggle(self) -> None:
        """A per-structure show/hide checkbox changed: refresh every
        phase view that filters tracks by structure."""
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()

    def _apply_image_cmap(self, image_item) -> None:
        """Apply the GUI's GIWAXS colormap (matches the main image
        viewer's ``DEFAULT_COLORMAP``) to a heatmap ImageItem, so the
        waterfall / mean image read like the rest of the app instead of
        raw greyscale."""
        from mlgidgui.image_viewer import DEFAULT_COLORMAP

        cmap = None
        for source in ("matplotlib", None):
            try:
                cmap = (
                    pg.colormap.get(DEFAULT_COLORMAP, source=source)
                    if source else pg.colormap.get(DEFAULT_COLORMAP)
                )
            except Exception:
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            image_item.setColorMap(cmap)

    def set_ring_tracks(self, ring_tracks) -> None:
        """Mark which track indices are rings (drawn as dashed arcs on
        the q-map). Re-renders the q-map."""
        self._ring_tracks = {int(k) for k in ring_tracks}
        self._refresh_qmap()

    def clear(self) -> None:
        self._ring_tracks = set()
        self.set_track_phases({})
        self.set_member_phases({})
        self.set_payload(None)

    def apply_theme_colors(self, background, foreground) -> None:
        """Runtime theme swap (same contract as the viewer/profile)."""
        self._theme_colors = (background, foreground)
        self._apply_plot_colors(background, foreground)

    def _apply_plot_colors(self, background, foreground) -> None:
        for plot in (self.traj_plot, self.qmap_plot, self.amp_plot,
                     self.waterfall_plot):
            try:
                plot.setBackground(background)
                for axis_name in ("left", "bottom"):
                    axis = plot.getPlotItem().getAxis(axis_name)
                    axis.setPen(foreground)
                    axis.setTextPen(foreground)
            except Exception:
                logger.debug(
                    "suppressed exception in PhaseViewsWindow."
                    "apply_theme_colors", exc_info=True,
                )

    # --- View refreshes ---

    def _clear_overlays(self, plot: pg.PlotWidget, keep=()) -> None:
        for item in list(plot.getPlotItem().listDataItems()):
            plot.removeItem(item)
        # ImageItems are not data items; they persist across refreshes.
        del keep

    def _on_frame_interval_changed(self, _value: int) -> None:
        self._refresh_trajectories()
        self._refresh_amplitude()

    def _frame_interval(self) -> tuple[int, int]:
        """The user's [from, to] frame interval, normalized so a crossed
        pair (from > to) still reads as a valid interval."""
        lo, hi = self.frame_lo.value(), self.frame_hi.value()
        return (lo, hi) if lo <= hi else (hi, lo)

    def _interval_members(self, k: int) -> np.ndarray:
        """Track ``k``'s member indices restricted to the frame
        interval — the member set the trajectories and amplitude plots
        draw (the q-map and waterfall stay frame-complete)."""
        members = self._payload.track_members(k)
        lo, hi = self._frame_interval()
        frames = self._payload.frame_num[members]
        return members[(frames >= lo) & (frames <= hi)]

    def _refresh_trajectories(self) -> None:
        self._clear_overlays(self.traj_plot)
        axis = self.axis_combo.currentData() or "radius"
        self.traj_plot.setLabel("left", AXIS_LABELS[axis])
        payload = self._payload
        if payload is None:
            return
        phase_mode = (
            self.traj_color.currentData() == "matched"
            and bool(self._track_phases)
        )
        values = payload.axis_values(axis)
        for k in range(payload.n_tracks):
            if phase_mode:
                # Per-frame attribution: one curve per structure key the
                # track's members resolve to, so a track matched only
                # from frame z renders z.. in the phase colour and the
                # earlier members as unmatched grey (with member data
                # absent this is one whole-track curve, as before).
                for key, mem in self._member_subsets(k).items():
                    if not self._structure_key_visible(key):
                        continue
                    color = self._structure_color(key)
                    curve = self.traj_plot.plot(
                        payload.frame_num[mem], values[mem],
                        pen=self._curve_pen(color),
                        **self._symbol_kwargs(4, color),
                    )
                    curve.sigPointsClicked.connect(
                        self._on_traj_points_clicked
                    )
                continue
            if not self._structure_visible(k):
                continue
            members = self._interval_members(k)
            if not len(members):
                continue
            frames = payload.frame_num[members]
            color = self._qmap_track_color(k, payload.n_tracks, phase_mode)
            curve = self.traj_plot.plot(
                frames, values[members],
                pen=self._curve_pen(color),
                **self._symbol_kwargs(4, color),
            )
            curve.sigPointsClicked.connect(self._on_traj_points_clicked)

    def _on_traj_points_clicked(self, _item, points) -> None:
        if len(points):
            self.frameJumpRequested.emit(int(round(points[0].pos().x())))

    def _refresh_qmap(self) -> None:
        self._clear_overlays(self.qmap_plot)
        mi = self._mean_image
        if mi is not None:
            data = np.asarray(mi["data"], dtype=np.float32)
            shown = _log_display(data) if self.qmap_log.isChecked() else data
            # ImageItem axis order: axis0 -> x. Data is (n_qz, n_qxy),
            # so transpose puts q_xy on x / q_z on y.
            self.qmap_image.setImage(shown.T, autoLevels=True)
            q_xy, q_z = mi["q_xy"], mi["q_z"]
            x0, x1 = float(np.min(q_xy)), float(np.max(q_xy))
            y0, y1 = float(np.min(q_z)), float(np.max(q_z))
            self.qmap_image.setRect(x0, y0, x1 - x0, y1 - y0)
            self.qmap_image.setVisible(True)
        else:
            self.qmap_image.setVisible(False)
        payload = self._payload
        phase_mode = self.qmap_phase.isChecked() and bool(self._track_phases)
        if payload is None:
            self._qmap_legend.clear()
            self._qmap_legend.setVisible(False)
            return
        for k in range(payload.n_tracks):
            if k in self._ring_tracks:
                if not self._structure_visible(k):
                    continue
                color = self._qmap_track_color(
                    k, payload.n_tracks, phase_mode
                )
                # A ring spans the whole azimuth at a fixed |q|, so it
                # reads as a dashed quarter-circle arc (0deg along q_xy
                # to 90deg along q_z) at its mean radius — not a cluster
                # of points. One arc per track: splitting it by member
                # attribution would just overdraw the same radius.
                r = payload.track_mean("radius", k)
                theta = np.linspace(0.0, np.pi / 2, 100)
                self.qmap_plot.plot(
                    r * np.cos(theta), r * np.sin(theta),
                    pen=pg.mkPen(
                        color, width=self._emph_width(2),
                        style=Qt.PenStyle.DashLine,
                    ),
                )
                continue
            if phase_mode:
                # Per-frame attribution: the track's points split by the
                # phase that claimed each member on its frame (frame-
                # complete — the frame-interval control does not narrow
                # the q-map). The mean marker stays one per track, in
                # the dominant colour.
                for key, mem in self._member_subsets(
                    k, interval=False
                ).items():
                    if not self._structure_key_visible(key):
                        continue
                    self.qmap_plot.plot(
                        payload.q_xy[mem], payload.q_z[mem],
                        pen=self._curve_pen(self._structure_color(key)),
                    )
                if not self._structure_visible(k):
                    continue
                color = self._qmap_track_color(k, payload.n_tracks, True)
            else:
                if not self._structure_visible(k):
                    continue
                color = self._qmap_track_color(
                    k, payload.n_tracks, phase_mode
                )
                members = payload.track_members(k)
                self.qmap_plot.plot(
                    payload.q_xy[members], payload.q_z[members],
                    pen=self._curve_pen(color),
                )
            self.qmap_plot.plot(
                [payload.track_mean("q_xy", k)],
                [payload.track_mean("q_z", k)],
                pen=None, symbol="o", symbolSize=10,
                symbolBrush=None,
                symbolPen=pg.mkPen(color, width=self._emph_width(2)),
            )
        if phase_mode:
            self._rebuild_phase_legend(payload)
        else:
            self._qmap_legend.clear()
            self._qmap_legend.setVisible(False)

    # Neutral grey for tracks with no matched phase.
    _UNMATCHED_COLOR = (150, 150, 150)

    def _qmap_track_color(self, k: int, n_tracks: int, phase_mode: bool):
        """Colour for track ``k`` on the q-map: per-CIF phase colour
        (dominant phase, grey if unmatched) when phase mode is on,
        otherwise the per-track cycling colour."""
        if not phase_mode:
            return _track_pen(k, n_tracks)
        cifs = self._track_phases.get(k)
        if cifs:
            return self._phase_colors[cifs[0]]
        return pg.mkColor(*self._UNMATCHED_COLOR)

    def _rebuild_phase_legend(self, payload) -> None:
        """One legend entry per unique structure key drawn among the
        VISIBLE member subsets (+ 'unmatched' when any visible portion
        is unclaimed) — so hiding a structure drops it from the legend
        too. With no member attribution this reduces to the dominant
        CIF per visible track, the pre-split legend."""
        self._qmap_legend.clear()
        seen: list = []
        has_unmatched = False
        for k in range(payload.n_tracks):
            for key in self._member_subsets(k, interval=False):
                if not self._structure_key_visible(key):
                    continue
                if key == _UNMATCHED:
                    has_unmatched = True
                elif key not in seen:
                    seen.append(key)
        for cif in seen:
            sample = pg.PlotDataItem(
                pen=pg.mkPen(self._structure_color(cif), width=2)
            )
            self._qmap_legend.addItem(sample, cif)
        if has_unmatched:
            sample = pg.PlotDataItem(
                pen=pg.mkPen(self._UNMATCHED_COLOR, width=2)
            )
            self._qmap_legend.addItem(sample, "unmatched")
        self._qmap_legend.setVisible(True)

    def _clear_amp_labels(self) -> None:
        for lbl in self._amp_labels:
            self.amp_plot.removeItem(lbl)
        self._amp_labels = []
        for fill in self._amp_fills:
            self.amp_plot.removeItem(fill)
        self._amp_fills = []

    def _zero_padding_active(self) -> bool:
        """The "Zeros at start/end" toggle is effective only in median
        mode (the checkbox is disabled otherwise but keeps its checked
        state, so both must be tested)."""
        return self.amp_zero.isEnabled() and self.amp_zero.isChecked()

    def _zero_pad(self, frames, *value_arrays):
        """"Zeros at start/end": pad a per-STRUCTURE median series with
        per-frame zero points from the visible start to the first
        observed frame and from the last one to the visible end — the
        structure is genuinely absent before forming / after vanishing,
        so its median evolution rises from and falls back to zero.
        Applied to the RESULT of the per-frame statistics, never to the
        individual tracks feeding them: a track absent on a frame must
        not contribute a zero there, or the median is dragged down
        whenever only some of the structure's tracks are present. Gaps
        inside the series stay open."""
        if not self._zero_padding_active() or frames.size == 0:
            return (frames, *value_arrays)
        lo, hi = self._frame_interval()
        lead = np.arange(lo, int(frames.min()))
        tail = np.arange(int(frames.max()) + 1, hi + 1)
        if not len(lead) and not len(tail):
            return (frames, *value_arrays)
        frames = np.concatenate([lead, frames, tail])
        padded = tuple(
            np.concatenate([np.zeros(len(lead)), arr, np.zeros(len(tail))])
            for arr in value_arrays
        )
        return (frames, *padded)

    def _update_amp_tracks_button(self) -> None:
        n_off = sum(len(v) for v in self._amp_excluded.values())
        self.btn_amp_tracks.setText(
            f"Select tracks… ({n_off} off)" if n_off else "Select tracks…"
        )

    def _on_select_amp_tracks(self) -> None:
        """Open the per-structure track picker — modeless (the plots
        update live next to it), rebuilt fresh on every open so it
        always reflects the current payload/phases."""
        if self._payload is None or not self._track_phases:
            return
        if self._amp_track_dialog is not None:
            self._amp_track_dialog.close()
        dlg = _TrackSelectDialog(self)
        self._amp_track_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _amp_groups(self) -> dict:
        """``{structure key: [member arrays]}`` feeding the amplitude
        displays: the per-frame member subsets minus the tracks the
        user unticked in the Select-tracks dialog (per structure).
        Interval-filtered like the plots themselves."""
        groups: dict = {}
        for k in range(self._payload.n_tracks):
            for key, mem in self._member_subsets(k).items():
                if k in self._amp_excluded.get(key, ()):
                    continue
                groups.setdefault(key, []).append(mem)
        return groups

    # --- Metric abstraction (fitted amplitude vs ROI intensity) ---
    #
    # Every amplitude-tab consumer (curves, medians, CSV export) works
    # on metric-agnostic SERIES — ``(frames, values)`` array pairs —
    # built here. The fitted-amplitude metric reads the payload's
    # member arrays; the ROI metric reads the worker-computed traces,
    # which cover EVERY scan frame (a failed fit leaves no gap).

    def _metric(self) -> str:
        return self.amp_metric.currentData() or "amplitude"

    def _metric_label(self) -> str:
        return (
            "Integrated ROI intensity (arb. u.)"
            if self._metric() == "roi" else AXIS_LABELS["amplitude"]
        )

    def _member_series(self, members) -> tuple:
        """Fitted-amplitude series of a member-index array."""
        payload = self._payload
        return payload.frame_num[members], payload.amplitude[members]

    def _roi_params(self) -> tuple:
        return (
            float(self.roi_dqxy.value()), float(self.roi_dqz.value()),
            int(self.roi_gap.value()), int(self.roi_strip.value()),
        )

    def _roi_cache_valid(self) -> bool:
        cache = self._roi_traces
        return (
            cache is not None
            and cache["params"] == self._roi_params()
            and cache["fp"] == self._amp_payload_fp
            and cache["target"] == (self._file_path, self._entry)
        )

    def _ensure_roi_traces(self) -> bool:
        """True when valid ROI traces are cached. Otherwise kick the
        background computation (when the scan context allows one) and
        return False — the worker's finish re-renders the tab."""
        if self._roi_cache_valid():
            return True
        if (
            self._payload is None or self._file_path is None
            or not self._entry or self._n_frames <= 0
        ):
            self.statusBar().showMessage(
                "The ROI metric needs the scan file — no scan context "
                "is set for this tracking result.", 6000,
            )
            return False
        if self._worker_thread is not None:
            # A view worker is running; its finish handler re-kicks.
            return False
        positions = {
            k: roi_intensity.track_frame_positions(
                self._payload, k, self._n_frames
            )
            for k in range(self._payload.n_tracks)
        }
        params = self._roi_params()
        started = self._start_worker(RoiTraceWorker(
            self._file_path, self._entry, positions, *params
        ))
        if started:
            self._roi_pending = {
                "fp": self._amp_payload_fp,
                "target": (self._file_path, self._entry),
            }
            self.statusBar().showMessage(
                "Computing ROI intensity traces…"
            )
        return False

    def _roi_track_series(self, k: int, frames=None):
        """Track ``k``'s cached ROI trace as a series — restricted to
        the frame interval, or to explicit ``frames``. None while the
        cache is missing or stale."""
        if not self._roi_cache_valid():
            return None
        trace = self._roi_traces["traces"].get(k)
        if trace is None:
            return None
        if frames is None:
            lo, hi = self._frame_interval()
            frames = np.arange(lo, min(hi, trace.size - 1) + 1)
        else:
            frames = np.asarray(frames, dtype=int)
            frames = frames[(frames >= 0) & (frames < trace.size)]
        return frames, trace[frames]

    def _track_series(self, k: int):
        """Metric series of track ``k`` over the frame interval (the
        ungrouped per-track view / export). None with nothing to show."""
        if self._metric() == "roi":
            return self._roi_track_series(k)
        members = self._interval_members(k)
        if not len(members):
            return None
        return self._member_series(members)

    def _roi_key_frames(self, k: int) -> dict:
        """Partition the interval frames of track ``k``'s ROI trace
        among the track's structure keys: each frame follows the key of
        the NEAREST member frame (ties -> earlier) — the ROI companion
        of ``_member_subsets``' per-frame attribution, extended to the
        synthesized gap/lead/tail frames. Single-key tracks (the usual
        case) get the whole trace."""
        lo, hi = self._frame_interval()
        frames = np.arange(lo, hi + 1)
        subsets = self._member_subsets(k, interval=False)
        if len(subsets) <= 1:
            key = next(iter(subsets), self._structure_key_of(k))
            return {key: frames}
        payload = self._payload
        frame_key: dict = {}
        for i in payload.track_members(k):
            f = int(payload.frame_num[i])
            if f not in frame_key:
                frame_key[f] = self._member_structure_key(int(i), k)
        uf = np.array(sorted(frame_key))
        keys = [frame_key[int(f)] for f in uf]
        idx = np.clip(np.searchsorted(uf, frames), 0, uf.size - 1)
        prev = np.clip(idx - 1, 0, uf.size - 1)
        take_prev = np.abs(frames - uf[prev]) <= np.abs(uf[idx] - frames)
        choice = np.where(take_prev, prev, idx)
        out: dict = {}
        for f, c in zip(frames, choice):
            out.setdefault(keys[int(c)], []).append(int(f))
        return {
            key: np.asarray(v, dtype=int) for key, v in out.items()
        }

    def _amp_group_series(self) -> dict:
        """``{structure key: [series]}`` feeding the grouped / median
        displays under the current metric, minus the tracks unticked in
        the Select-tracks dialog."""
        if self._metric() != "roi":
            return {
                key: [self._member_series(mem) for mem in mem_lists]
                for key, mem_lists in self._amp_groups().items()
            }
        groups: dict = {}
        for k in range(self._payload.n_tracks):
            for key, frames in self._roi_key_frames(k).items():
                if k in self._amp_excluded.get(key, ()):
                    continue
                series = self._roi_track_series(k, frames)
                if series is not None and len(series[0]):
                    groups.setdefault(key, []).append(series)
        return groups

    def _on_metric_changed(self) -> None:
        self._roi_controls.setVisible(self._metric() == "roi")
        self._refresh_amplitude()

    def _amp_median_series(self, series_list, window, normalize):
        """Per-frame MEDIAN + interquartile bounds over
        ``series_list`` — one ``(frames, values)`` series per
        contributing track (or track portion, with per-frame phase
        attribution).

        Each series goes through the same smoothing window /
        per-track normalization as the individual curves, then a series
        with several same-frame points contributes their mean once.
        Per frame the statistics run over the series that HAVE a value
        there (a track absent from a frame does not drag the median
        down). Returns ``(frames, median, q25, q75)`` — the 25th/75th
        percentile bounds collapse onto the median where only one track
        contributes — or ``None`` with no data.
        """
        per_frame: dict = {}
        for series in series_list:
            if series is None or not len(series[0]):
                continue
            frames, amps = smooth_normalize_amplitude(
                series[0], series[1],
                window=window, normalize=normalize,
            )
            track_vals: dict = {}
            for f, a in zip(frames, amps):
                track_vals.setdefault(int(f), []).append(float(a))
            for f, vals in track_vals.items():
                per_frame.setdefault(f, []).append(
                    float(np.mean(vals))
                )
        if not per_frame:
            return None
        xs = np.array(sorted(per_frame), dtype=float)
        med = np.array([np.median(per_frame[int(f)]) for f in xs])
        q25 = np.array([np.percentile(per_frame[int(f)], 25) for f in xs])
        q75 = np.array([np.percentile(per_frame[int(f)], 75) for f in xs])
        # Zero baseline on the RESULT only (see _zero_pad): the stats
        # above ran over present tracks, so an absent track never drags
        # the median down; the finished curve then rises from and falls
        # back to zero. The IQR bounds pad to zero too — the band
        # collapses where the structure is absent.
        xs, med, q25, q75 = self._zero_pad(xs, med, q25, q75)
        return xs, med, q25, q75

    def _amp_median_band(
        self, series_list, window, normalize, baseline, color
    ):
        """Draw one structure's median metric curve with a
        transparent interquartile band — 25th to 75th percentile
        (``pg.FillBetweenItem`` between the bound curves, tracked in
        ``_amp_fills`` because fills are not pyqtgraph data items)."""
        series = self._amp_median_series(series_list, window, normalize)
        if series is None:
            return
        xs, med, q25, q75 = series
        qc = pg.mkColor(color)
        band_pen = pg.mkPen(
            (qc.red(), qc.green(), qc.blue(), 0)
        )
        lo = self.amp_plot.plot(xs, q25 + baseline, pen=band_pen)
        hi = self.amp_plot.plot(xs, q75 + baseline, pen=band_pen)
        fill = pg.FillBetweenItem(
            lo, hi,
            brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 60),
        )
        self.amp_plot.addItem(fill)
        self._amp_fills.append(fill)
        self.amp_plot.plot(
            xs, med + baseline,
            pen=self._curve_pen(color, base_width=2, emphasized=True),
            **self._symbol_kwargs(4, color),
        )

    def _amp_extreme_lines(self, series_list, window, baseline, color):
        """Thin horizontal reference lines at a structure's metric
        MAXIMUM and MINIMUM (grouped view): the extremes of its
        per-frame median series (per-track normalized, like the grouped
        curves), spanning the structure's frame range — so every point
        of the band reads off how far it sits from the structure's own
        max/min."""
        series = self._amp_median_series(series_list, window, True)
        if series is None or not len(series[0]):
            return
        xs, med, _q25, _q75 = series
        span = [float(xs.min()), float(xs.max())]
        pen = pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine)
        for y in (float(np.nanmax(med)), float(np.nanmin(med))):
            self.amp_plot.plot(span, [y + baseline] * 2, pen=pen)

    def _amp_curve(self, series, window, normalize, baseline, color):
        if series is None or not len(series[0]):
            return
        frames, amps = smooth_normalize_amplitude(
            series[0], series[1],
            window=window, normalize=normalize,
        )
        self.amp_plot.plot(
            frames, amps + baseline,
            pen=self._curve_pen(color),
            **self._symbol_kwargs(3, color),
        )

    def _fit_amp_xrange(self) -> None:
        """Pin the amplitude plot's x-axis to the tracked frame
        interval. Auto-range used to run away far beyond the data
        (thousands of frames on a 350-frame scan) — the grouped view's
        pixel-sized structure labels fed back into the range
        computation — so the axis is set explicitly to the range the
        curves and the CSV export actually cover."""
        lo, hi = self._frame_interval()
        if hi > lo:
            self.amp_plot.setXRange(float(lo), float(hi), padding=0.02)

    def _refresh_amplitude(self) -> None:
        self._clear_overlays(self.amp_plot)
        self._clear_amp_labels()
        left_axis = self.amp_plot.getPlotItem().getAxis("left")
        payload = self._payload
        grouped = self.amp_group.isChecked() and bool(self._track_phases)
        median = self.amp_median.isChecked() and bool(self._track_phases)
        # Grouped bands are always per-track normalized (the offset needs
        # a fixed 0..1 range), so the manual normalize toggle is moot.
        self.amp_normalize.setEnabled(not grouped)
        # Zero baseline is a median-curve feature (per-structure lead-in
        # and tail); outside median mode the box is inert — shown
        # disabled but keeping its checked state.
        self.amp_zero.setEnabled(median)
        if payload is None:
            left_axis.setStyle(showValues=True)
            self.amp_plot.setLabel("left", "Amplitude")
            return
        if self._metric() == "roi" and not self._ensure_roi_traces():
            # No traces yet: leave the tab empty until the worker's
            # finish handler re-renders (status bar shows the state).
            left_axis.setStyle(showValues=True)
            self.amp_plot.setLabel("left", self._metric_label())
            return
        window = int(self.amp_window.value())
        try:
            self._draw_amplitude(payload, grouped, median, window,
                                 left_axis)
        finally:
            self._fit_amp_xrange()

    def _draw_amplitude(
        self, payload, grouped, median, window, left_axis
    ) -> None:
        if grouped:
            # Stack structures: each CIF group at its own vertical
            # offset, normalized per track (y-scale arbitrary), coloured
            # by phase, with a text label per band. Hidden structures
            # (unticked) drop out and the remaining bands re-pack. In
            # median mode each band is ONE median curve + IQR band
            # instead of the group's individual tracks.
            left_axis.setStyle(showValues=False)
            self.amp_plot.setLabel("left", "Structure (stacked, arb.)")
            # Member-level grouping: a track matched only on part of its
            # span contributes those members to its phase's band and the
            # rest to "unmatched" (one whole-track entry under the
            # dominant key when no member attribution exists). Tracks
            # unticked in the Select-tracks dialog drop out here.
            groups = self._amp_group_series()
            keys = [
                key for key in self._structure_keys()
                if self._structure_key_visible(key)
            ]
            x0 = float(payload.frame_num.min()) if payload.n_members else 0.0
            # With a narrowed frame interval the bands start there, so
            # the labels follow the visible left edge.
            x0 = max(x0, float(self._frame_interval()[0]))
            for g, key in enumerate(keys):
                baseline = g * self._AMP_GROUP_STEP
                color = self._structure_color(key)
                series_list = groups.get(key, [])
                if median:
                    self._amp_median_band(
                        series_list, window, True, baseline, color
                    )
                else:
                    for series in series_list:
                        self._amp_curve(
                            series, window, True, baseline, color
                        )
                # Thin dashed reference lines at this structure's
                # median max / min.
                self._amp_extreme_lines(series_list, window, baseline, color)
                # Bottom-left anchored just above the band's top: the
                # curves and the dashed max reference line stay within
                # baseline..baseline+1.0, so the label sits in the
                # inter-band gap instead of on the curves (it used to
                # be vertically centred IN the band, overlapping them).
                # ignoreBounds: the pixel-sized label must not feed the
                # view's auto-range (see _fit_amp_xrange).
                label = pg.TextItem(
                    text="unmatched" if key == _UNMATCHED else key,
                    color=color, anchor=(0.0, 1.0),
                )
                label.setPos(x0, baseline + 1.02)
                self.amp_plot.addItem(label, ignoreBounds=True)
                self._amp_labels.append(label)
            return

        # Ungrouped: overlaid curves (structure toggles still filter
        # which tracks show). Median mode collapses each structure's
        # tracks into one phase-coloured median curve + IQR band.
        left_axis.setStyle(showValues=True)
        normalize = self.amp_normalize.isChecked()
        if normalize:
            label = (
                "Integrated ROI intensity (normalized)"
                if self._metric() == "roi" else "Amplitude (normalized)"
            )
        else:
            label = self._metric_label()
        self.amp_plot.setLabel("left", label)
        if median:
            groups = self._amp_group_series()
            for key in self._structure_keys():
                if not self._structure_key_visible(key):
                    continue
                self._amp_median_band(
                    groups.get(key, []), window, normalize, 0.0,
                    self._structure_color(key),
                )
            return
        for k in range(payload.n_tracks):
            if not self._structure_visible(k):
                continue
            self._amp_curve(
                self._track_series(k), window, normalize, 0.0,
                _track_pen(k, payload.n_tracks),
            )

    # Vertical spacing between stacked structure bands (each track is
    # normalized to 0..1, so 1.3 leaves a ~0.3 gap).
    _AMP_GROUP_STEP = 1.3

    def _refresh_waterfall(self) -> None:
        wf = self._waterfall
        if wf is None:
            self.waterfall_image.setVisible(False)
            return
        data = np.asarray(wf["data"], dtype=np.float32)
        shown = (
            _log_display(data)
            if self.waterfall_log.isChecked()
            else data
        )
        # data is (n_frames, n_radius): axis0 (frame) -> x, axis1 -> y.
        self.waterfall_image.setImage(shown, autoLevels=True)
        radius = np.asarray(wf["radius"], dtype=float)
        r0, r1 = float(radius.min()), float(radius.max())
        self.waterfall_image.setRect(0.0, r0, data.shape[0], r1 - r0)
        self.waterfall_image.setVisible(True)

    # --- Worker plumbing (waterfall / mean image) ---

    @property
    def worker_active(self) -> bool:
        """True while a waterfall / mean-image worker holds its read
        handle on the file (the host refuses to start pipeline runs
        during that window — see ``busy_probe``)."""
        return self._worker_thread is not None

    def _start_worker(self, worker: ScanProfileWorker) -> bool:
        if self._worker_thread is not None:
            self.statusBar().showMessage(
                "A view computation is already running…", 4000
            )
            return False
        if self._busy_probe is not None and self._busy_probe():
            self.statusBar().showMessage(
                "Wait for the running pipeline command to finish — the "
                "file cannot be read while mlgidbase holds it.", 6000
            )
            return False
        self.btn_waterfall.setEnabled(False)
        self.btn_mean_image.setEnabled(False)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        thread.started.connect(worker.run)
        self._worker_thread, self._worker = thread, worker
        thread.start()
        return True

    def _on_worker_progress(self, done: int, total: int) -> None:
        self.statusBar().showMessage(f"Computing… frame {done}/{total}")

    def _on_worker_finished(self, result: object, error) -> None:
        self._worker_thread, self._worker = stop_worker_thread(
            self._worker_thread, self._worker
        )
        self.btn_waterfall.setEnabled(True)
        self.btn_mean_image.setEnabled(True)
        if error is not None:
            self._roi_pending = None
            self.statusBar().showMessage(f"View computation failed: {error}")
            QMessageBox.warning(self, "Phase tracking views", str(error))
            return
        if not isinstance(result, dict):
            return
        if result.get("kind") == "waterfall":
            self._waterfall = result
            self._refresh_waterfall()
        elif result.get("kind") == "mean_image":
            self._mean_image = result
            self._refresh_qmap()
        elif result.get("kind") == "roi_traces":
            pending = self._roi_pending or {}
            self._roi_pending = None
            self._roi_traces = {
                "frames": result["frames"],
                "traces": result["traces"],
                "params": result["params"],
                "fp": pending.get("fp"),
                "target": pending.get("target"),
            }
        self.statusBar().showMessage("Done.", 3000)
        # The ROI metric may be waiting on a worker slot (only one view
        # worker runs at a time) or on fresher parameters than the run
        # that just landed — the refresh re-kicks in that case, and
        # otherwise renders the new traces.
        if self._metric() == "roi" and (
            result.get("kind") == "roi_traces"
            or not self._roi_cache_valid()
        ):
            self._refresh_amplitude()

    def _on_compute_waterfall(self) -> None:
        if self._file_path is None or not self._entry:
            return
        self._start_worker(ScanProfileWorker(
            self._file_path, self._entry, "waterfall",
        ))

    def _on_compute_mean_image(self) -> None:
        if self._file_path is None or not self._entry:
            return
        lo = int(self.qmap_lo.value())
        hi = int(self.qmap_hi.value())
        if hi < lo:
            lo, hi = hi, lo
        self._start_worker(ScanProfileWorker(
            self._file_path, self._entry, "mean_image",
            frame_lo=lo, frame_hi=hi,
        ))

    # --- Plot data / image export ---

    _VIEW_KEYS = ("trajectories", "qmap", "amplitude", "waterfall")

    def _current_view_key(self) -> str:
        return self._VIEW_KEYS[self._tabs.currentIndex()]

    def _view_availability(self) -> list:
        """``(key, label, enabled, tooltip)`` per view — disabled
        entries carry a hint about what has to run first."""
        payload_ok = self._payload is not None
        need_track = "Run \"Track peaks (mlgidBASE)\" first."
        return [
            ("trajectories", "Trajectories", payload_ok,
             "" if payload_ok else need_track),
            ("qmap", "q-map overlay",
             payload_ok or self._mean_image is not None,
             "" if payload_ok or self._mean_image is not None
             else need_track),
            ("amplitude", "Amplitude evolution", payload_ok,
             "" if payload_ok else need_track),
            ("waterfall", "Radial waterfall", self._waterfall is not None,
             "" if self._waterfall is not None
             else "Compute the waterfall first."),
        ]

    def _structure_dialog_entries(self) -> list:
        """``(key, label, css_color, checked)`` per structure for the
        image-export dialog; initial check state mirrors the window's
        toggle bar."""
        if not self._track_phases:
            return []
        return [
            (
                key,
                "unmatched" if key == _UNMATCHED else key,
                pg.mkColor(self._structure_color(key)).name(),
                self._structure_key_visible(key),
            )
            for key in self._structure_keys()
        ]

    def _track_structure_label(self, k: int) -> str:
        key = self._structure_key_of(k)
        return "unmatched" if key == _UNMATCHED else key

    @staticmethod
    def _write_csv(path, header, rows, comment=None) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if comment:
                fh.write(f"# {comment}\n")
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _write_matrix_csv(
        path, corner, col_axis, row_axis, data, comment=None,
    ) -> None:
        """2D array as CSV: header row = column axis, first column =
        row axis (``corner`` labels both, e.g. ``frame\\|q|``)."""
        data = np.asarray(data, dtype=float)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if comment:
                fh.write(f"# {comment}\n")
            writer = csv.writer(fh)
            writer.writerow(
                [corner] + [f"{float(v):.8g}" for v in col_axis]
            )
            for label, row in zip(row_axis, data):
                writer.writerow([label] + [f"{v:.8g}" for v in row])

    def _member_export_rows(self, extra_columns) -> list:
        """One CSV row per track member (ALL tracks — the structure
        toggles are a viewing aid, the export is complete). ``extra_
        columns(k, m)`` appends the per-view value columns."""
        payload = self._payload
        rows = []
        for k in range(payload.n_tracks):
            label = self._track_structure_label(k)
            all_structs = ";".join(self._track_phases.get(k, []))
            ring = int(k in self._ring_tracks)
            for m in payload.track_members(k):
                rows.append(
                    [k, label, all_structs, ring,
                     int(payload.frame_num[m])] + extra_columns(k, m)
                )
        return rows

    def _trajectories_export(self):
        payload = self._payload
        radius, angle = payload.radius, payload.angle
        header = ["track", "structure", "all_structures", "is_ring",
                  "frame", "radius", "angle", "amplitude", "q_xy", "q_z"]
        rows = self._member_export_rows(lambda k, m: [
            f"{radius[m]:.8g}", f"{angle[m]:.8g}",
            f"{payload.amplitude[m]:.8g}",
            f"{payload.q_xy[m]:.8g}", f"{payload.q_z[m]:.8g}",
        ])
        return header, rows

    def _qmap_export(self):
        payload = self._payload
        header = ["track", "structure", "all_structures", "is_ring",
                  "frame", "q_xy", "q_z", "track_mean_q_xy",
                  "track_mean_q_z", "track_mean_radius"]
        means = {
            k: (payload.track_mean("q_xy", k),
                payload.track_mean("q_z", k),
                payload.track_mean("radius", k))
            for k in range(payload.n_tracks)
        }
        rows = self._member_export_rows(lambda k, m: [
            f"{payload.q_xy[m]:.8g}", f"{payload.q_z[m]:.8g}",
            f"{means[k][0]:.8g}", f"{means[k][1]:.8g}",
            f"{means[k][2]:.8g}",
        ])
        return header, rows

    def _amplitude_export(self):
        """Amplitude tab numbers AS DISPLAYED: honours the metric
        selector, the smoothing window, the normalize toggle (forced on
        when grouped) and the median mode (-> per-structure
        median/quartile columns). The grouped view's vertical offsets
        are display-only and NOT included.

        Every frame of the exported range gets a row (per track /
        structure): frames without a value carry an explicit ``nan``
        instead of being skipped, so naive plots cannot silently
        interpolate across fit gaps. Same-frame duplicate points (a
        track with several members on one frame) average into the
        frame's single row."""
        payload = self._payload
        roi = self._metric() == "roi"
        grouped = self.amp_group.isChecked() and bool(self._track_phases)
        median = self.amp_median.isChecked() and bool(self._track_phases)
        window = int(self.amp_window.value())
        normalize = True if grouped else self.amp_normalize.isChecked()
        lo, hi = self._frame_interval()
        custom = any(self._amp_excluded.values())
        metric_name = "integrated intensity (ROI)" if roi else "amplitude"
        roi_settings = ""
        if roi:
            dqxy, dqz, gap, strip = self._roi_params()
            roi_settings = (
                f", roi=+-{dqxy:g}/+-{dqz:g} 1/A "
                f"bg gap {gap} px strips {strip} px"
            )
        settings = (
            f"metric={metric_name}{roi_settings}, "
            f"window={window}, normalize={normalize}, "
            f"mode={'median per structure' if median else 'per track'}, "
            f"frames={lo}..{hi}, "
            f"zeros={'lead+tail' if self._zero_padding_active() else 'off'}, "
            f"tracks={'custom' if custom else 'all'}; "
            "grouped-display offsets not included; "
            "one row per frame, nan = no value"
        )

        full = np.arange(lo, hi + 1)

        def _frame_complete(frames, values):
            """One value per frame of the exported range: same-frame
            duplicates averaged, missing frames NaN."""
            out = np.full(full.size, np.nan)
            fi = np.asarray(frames, dtype=int) - lo
            ok = (fi >= 0) & (fi < full.size)
            sums = np.zeros(full.size)
            counts = np.zeros(full.size)
            np.add.at(sums, fi[ok], np.asarray(values, dtype=float)[ok])
            np.add.at(counts, fi[ok], 1.0)
            has = counts > 0
            out[has] = sums[has] / counts[has]
            return out

        value_col = "roi_intensity" if roi else "amplitude"
        if median:
            header = ["structure", "frame", "median", "q25", "q75"]
            rows = []
            groups = self._amp_group_series()
            for key in self._structure_keys():
                series = self._amp_median_series(
                    groups.get(key, []), window, normalize,
                )
                if series is None:
                    continue
                label = "unmatched" if key == _UNMATCHED else key
                med = _frame_complete(series[0], series[1])
                q25 = _frame_complete(series[0], series[2])
                q75 = _frame_complete(series[0], series[3])
                for f, m, b_lo, b_hi in zip(full, med, q25, q75):
                    rows.append([
                        label, int(f), f"{m:.8g}",
                        f"{b_lo:.8g}", f"{b_hi:.8g}",
                    ])
            return header, rows, settings
        header = ["track", "structure", "frame", value_col]
        rows = []
        for k in range(payload.n_tracks):
            label = self._track_structure_label(k)
            series = self._track_series(k)
            if series is None or not len(series[0]):
                continue
            frames, amps = smooth_normalize_amplitude(
                series[0], series[1],
                window=window, normalize=normalize,
            )
            for f, a in zip(full, _frame_complete(frames, amps)):
                rows.append([k, label, int(f), f"{a:.8g}"])
        return header, rows, settings

    def _export_data_files(self, keys, base: str) -> list:
        """Write one CSV per selected view (plus the q-map's mean-image
        matrix when computed) as ``<base>_<view>.csv``; returns the
        written paths."""
        entry = self._entry or (
            self._payload.entry if self._payload is not None else ""
        )
        provenance = f"mlgidGUI scan-tracking export, entry={entry}"
        written = []

        def _write(suffix, header, rows, comment):
            path = f"{base}_{suffix}.csv"
            self._write_csv(path, header, rows, comment=comment)
            written.append(path)

        if "trajectories" in keys and self._payload is not None:
            header, rows = self._trajectories_export()
            _write("trajectories", header, rows, provenance)
        if "qmap" in keys:
            if self._payload is not None:
                header, rows = self._qmap_export()
                _write("qmap_tracks", header, rows, provenance)
            if self._mean_image is not None:
                mi = self._mean_image
                path = f"{base}_qmap_mean_image.csv"
                lo, hi = int(self.qmap_lo.value()), int(self.qmap_hi.value())
                self._write_matrix_csv(
                    path, "q_z\\q_xy", mi["q_xy"],
                    [f"{float(v):.8g}" for v in mi["q_z"]], mi["data"],
                    comment=(
                        f"{provenance}; nan-mean Cartesian image of "
                        f"frames {min(lo, hi)}-{max(lo, hi)}, linear "
                        "intensity"
                    ),
                )
                written.append(path)
        if "amplitude" in keys and self._payload is not None:
            header, rows, settings = self._amplitude_export()
            _write("amplitude", header, rows, f"{provenance}; {settings}")
        if "waterfall" in keys and self._waterfall is not None:
            wf = self._waterfall
            data = np.asarray(wf["data"], dtype=float)
            path = f"{base}_waterfall.csv"
            self._write_matrix_csv(
                path, "frame\\|q|", wf["radius"],
                list(range(data.shape[0])), data,
                comment=(
                    f"{provenance}; angular-mean radial profile per "
                    "frame, linear intensity"
                ),
            )
            written.append(path)
        return written

    def _refresh_track_views(self) -> None:
        """Re-render the three payload-driven plots (the waterfall does
        not consume the export overrides)."""
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()

    def _export_image_files(
        self, keys, base: str, ext: str, *, structures=None,
        line_width=None, line_style=None, marker_size=None,
        width_px: int = 1600, white_bg: bool = False,
    ) -> list:
        """Render the selected views to ``<base>_<view><ext>`` with the
        export styling applied, then restore the on-screen rendering.
        Returns the written paths."""
        from pyqtgraph.exporters import ImageExporter, SVGExporter

        overrides: dict = {}
        if structures is not None:
            overrides["structures"] = set(structures)
        if line_width is not None:
            overrides["line_width"] = float(line_width)
        if line_style is not None:
            overrides["line_style"] = line_style
        if marker_size is not None:
            overrides["marker_size"] = int(marker_size)
        plots = {
            "trajectories": self.traj_plot,
            "qmap": self.qmap_plot,
            "amplitude": self.amp_plot,
            "waterfall": self.waterfall_plot,
        }
        written = []
        self._export_overrides = overrides
        if white_bg:
            self._apply_plot_colors("w", "k")
        try:
            if overrides:
                self._refresh_track_views()
            for key in keys:
                path = f"{base}_{key}{ext}"
                item = plots[key].getPlotItem()
                if ext == ".svg":
                    SVGExporter(item).export(path)
                else:
                    exporter = ImageExporter(item)
                    try:
                        exporter.parameters()["width"] = int(width_px)
                    except Exception:
                        logger.debug(
                            "ImageExporter width parameter rejected; "
                            "exporting at native size", exc_info=True,
                        )
                    exporter.export(path)
                written.append(path)
        finally:
            self._export_overrides = {}
            if white_bg:
                self._apply_plot_colors(*self._theme_colors)
            if overrides:
                self._refresh_track_views()
        return written

    def _report_written(self, written) -> None:
        if not written:
            self.statusBar().showMessage("Nothing was exported.", 6000)
            return
        folder = os.path.dirname(written[0])
        names = ", ".join(os.path.basename(p) for p in written)
        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) to {folder}: {names}", 10000
        )

    def _on_export_data(self) -> None:
        views = self._view_availability()
        if not any(enabled for _k, _l, enabled, _t in views):
            QMessageBox.information(
                self, "Export plot data",
                "Nothing to export yet — run \"Track peaks "
                "(mlgidBASE)\" or compute a waterfall first.",
            )
            return
        dlg = _ExportDialog(
            self, "data", views, current_key=self._current_view_key(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keys = dlg.selected_views()
        if not keys:
            return
        path, _filter = file_dialogs.save_file(
            self, "Export plot data (CSV)", "CSV files (*.csv)",
            suggested_name="tracking_data.csv",
        )
        if not path:
            return
        if path.lower().endswith(".csv"):
            path = path[:-4]
        self._report_written(self._export_data_files(keys, path))

    def _on_export_images(self) -> None:
        views = self._view_availability()
        if not any(enabled for _k, _l, enabled, _t in views):
            QMessageBox.information(
                self, "Export plot images",
                "Nothing to export yet — run \"Track peaks "
                "(mlgidBASE)\" or compute a waterfall first.",
            )
            return
        dlg = _ExportDialog(
            self, "image", views,
            structures=self._structure_dialog_entries(),
            current_key=self._current_view_key(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keys = dlg.selected_views()
        if not keys:
            return
        path, chosen_filter = file_dialogs.save_file(
            self, "Export plot images",
            "PNG image (*.png);;SVG image (*.svg)",
            suggested_name="tracking_plots.png",
        )
        if not path:
            return
        ext = (
            ".svg"
            if path.lower().endswith(".svg")
            or "svg" in chosen_filter.lower()
            else ".png"
        )
        for known in (".png", ".svg"):
            if path.lower().endswith(known):
                path = path[: -len(known)]
        style = dlg.style()
        written = self._export_image_files(
            keys, path, ext,
            structures=dlg.selected_structures(),
            line_width=style["line_width"],
            line_style=style["line_style"],
            marker_size=style["marker_size"],
            width_px=style["width_px"],
            white_bg=style["white_bg"],
        )
        self._report_written(written)

    # --- Official figure export ---

    def _on_save_figures(self) -> None:
        if self._payload is None:
            QMessageBox.information(
                self, "Save figures",
                "Run \"Track peaks (mlgidBASE)\" first.",
            )
            return
        path, _filter = file_dialogs.save_file(
            self,
            "Save official mlgidBASE tracking figures",
            "PNG image (*.png);;SVG image (*.svg);;PDF (*.pdf)",
            suggested_name="peak_tracking.png",
        )
        if not path:
            return
        axis = self.axis_combo.currentData() or "radius"
        self.saveFiguresRequested.emit(path, axis)

    # --- Lifecycle ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Let a running view worker finish cleanly; it is read-only
        # and short-lived.
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        super().closeEvent(event)
