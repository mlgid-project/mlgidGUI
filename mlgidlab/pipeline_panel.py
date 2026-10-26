"""Pipeline launcher: collapsible Detection / Fitting / Matching sections.

Each section exposes the full kwarg surface of the underlying ``mlgidBASE``
method (see ``mlgidbase/main.py``: ``run_detection``, ``run_fitting``,
``run_matching``). The panel knows nothing about the active session — the
host wires a ``get_active_entry`` callback so the entry-scope dropdowns can
resolve "Active entry" at click time.

Detection is the exception to "kwarg surface": its parameters do not
arrive as kwargs but as one ``config_detect`` dict, which mlgidBASE
flattens onto an ``mlgiddetect`` ``Config``. The keys, the defaults and
the nesting live in ``mlgidlab.detection_config`` (Qt-free, so they are
testable without a widget); this module only owns the widgets and the
decision of which ones sit inline and which under "Advanced".

Defaults intentionally scope every run to the *active* entry rather than to
all entries (mlgidBASE's own default). The viewer shows one entry at a time
and per-entry runs sidestep failures on incompatible sibling entries — the
user can still pick "All entries" explicitly when they want a sweep.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import file_dialogs
from mlgidlab.detection_config import build_config_detect, flatten_defaults
from mlgidlab.frame_range import parse_frame_range
from mlgidlab.manual_fit import (
    DEFAULT_CLUSTERING_DISTANCE_PEAKS,
    DEFAULT_CLUSTERING_DISTANCE_RINGS,
    DEFAULT_CLUSTERING_EXTEND,
)
from mlgidlab.pipeline import PipelineCommand, is_mlgidbase_available


# Sentinels for the entry-scope and frame-scope dropdowns. The panel resolves
# them to mlgidBASE-shaped kwargs at click time so the host MainWindow stays
# the single source of truth for "what's active right now".
ENTRY_ACTIVE = "Active entry"
ENTRY_ALL = "All entries"

FRAME_ACTIVE = "Active frame"
FRAME_ALL = "All frames"
FRAME_RANGE = "Frame range…"

# Human-readable titles for the mlgidBASE method names emitted via
# ``PipelineWorker.frameProgress`` — used by the progress label and
# the status-bar tail. Falls through to the raw op name when unmapped.
_OP_TITLES: dict[str, str] = {
    "run_detection": "Detection",
    "run_fitting": "Fitting",
    "run_matching": "Matching",
}


# Canonical homes moved to widgets.py in the 2026 source split; the
# underscore aliases keep panel internals and older references working.
from mlgidlab.widgets import CollapsibleSection as _CollapsibleSection
from mlgidlab.widgets import make_form as _make_form
from mlgidlab.widgets import PRIMARY as _PRIMARY, set_variant as _set_variant
from mlgidlab.widgets import skin_progress
from mlgidlab.widgets import spin_double


class PipelinePanel(QWidget):
    """Buttons + parameter controls for the three mlgidbase pipeline stages.

    Emits ``runRequested(PipelineCommand)`` when the user clicks a Run
    button; the main window owns the threading and file-handle juggling.
    """

    runRequested = Signal(PipelineCommand)
    # Emitted when the user clicks "Parse CIFs". The host runs a worker
    # thread to do the actual parsing (slow for raw CIFs) and posts the
    # result back via ``set_cif_pattern``.
    parseCifsRequested = Signal(str)
    # Emitted whenever the CIF cache changes: with the new CifPattern
    # after a successful parse, with None on invalidation (input edit,
    # parse failure, session swap). The Display dock's "Expected
    # pattern" section listens to enable/refresh itself.
    cifCacheChanged = Signal(object)
    # Log routing: panels emit messages and the host forwards them to a
    # shared Logs dock. Keeping ``append_log`` / ``clear_log`` as the
    # public surface so existing call sites don't change.
    logMessage = Signal(str)
    logCleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._available = is_mlgidbase_available()
        # Resolved by the host so panel stays decoupled from MainWindow.
        # If unset (or returns None) "Active entry/frame" silently falls
        # back to mlgidBASE's own None-default (all entries / all frames).
        self._get_active_entry: Callable[[], str | None] | None = None
        self._get_active_frame: Callable[[], int | None] | None = None
        self._get_frame_count: Callable[[], int | None] | None = None
        # CIF cache: ``_cached_cif_input`` is the raw text the user
        # parsed against; ``_cached_cif_obj`` is the resulting CifPattern
        # (or other pre-loaded object). Run Matching forwards the cache
        # when present and the input hasn't changed since the parse.
        self._cached_cif_input: str | None = None
        self._cached_cif_obj: object | None = None
        # Detection parameters: one registry keyed by the flat
        # ``SECTION_KEY`` name mlgidDETECT uses, so building the
        # ``config_detect`` dict and resetting the form are both a walk
        # over the same table rather than two lists of widget names to
        # keep in step. Empty on a backend-less install, where the
        # section is never built.
        self._det_defaults: dict[str, object] = flatten_defaults()
        self._det_param_widgets: dict[str, QWidget] = {}
        # (form, field) pairs for the rows only the dino model reads;
        # see ``_add_dino_row`` and ``_on_det_model_changed``.
        self._det_dino_rows: list[tuple[QFormLayout, QWidget]] = []
        self._build_ui()

    # -- Public API used by MainWindow --

    def set_active_entry_resolver(
        self, fn: Callable[[], str | None]
    ) -> None:
        self._get_active_entry = fn

    def set_active_frame_resolver(
        self, fn: Callable[[], int | None]
    ) -> None:
        self._get_active_frame = fn

    def set_frame_count_resolver(
        self, fn: Callable[[], int | None]
    ) -> None:
        """Resolver for the active entry's frame count — validates the
        "Frame range…" scope at click time."""
        self._get_frame_count = fn

    def append_log(self, msg: str) -> None:
        """Forward ``msg`` to the shared Logs dock via ``logMessage``."""
        self.logMessage.emit(msg)

    def clear_log(self) -> None:
        """Ask the shared Logs dock to wipe its contents."""
        self.logCleared.emit()

    def set_running(self, running: bool) -> None:
        if not self._available:
            return
        self.btn_detect.setEnabled(not running)
        self.btn_fit.setEnabled(not running)
        # Match + run-all both additionally require an active matching
        # source — let _update_match_enabled re-evaluate that gate when
        # a run finishes.
        if running:
            self.btn_match.setEnabled(False)
            self.btn_run_all.setEnabled(False)
        else:
            self._update_match_enabled()
        # Hide both progress rows when a run ends. Either bar will
        # reappear on the next ``on_frame_progress`` /
        # ``on_queue_progress`` call if the upcoming run is
        # multi-frame / multi-entry; single-everything runs leave them
        # hidden. Clear the paint-state memoisation so the next run
        # always repaints (even if its first emit happens to match
        # the last emit of the previous run).
        if not running and hasattr(self, "_progress_bar"):
            self._progress_bar.hide()
            self._progress_label.hide()
            self._frame_progress_last_state = None
            self._frame_progress_last_visible = False
        if not running and hasattr(self, "_entry_progress_bar"):
            self._entry_progress_bar.hide()
            self._entry_progress_label.hide()
            self._entry_progress_last_state = None
            self._entry_progress_last_visible = False

    def on_queue_progress(
        self, current: int, total: int, entry: str, op_name: str
    ) -> None:
        """Update the entry-level progress row.

        Driven by the host whenever a queued "All entries" run starts a
        new entry. ``current`` is 1-indexed (the entry just starting),
        ``total`` is the number of entries in the run. Hidden when
        ``total <= 1`` so single-entry runs do not get a meaningless
        1/1 bar. Label reads e.g. ``Detection · entry 3/8 · entry_DBTTF``.

        Memoises the last paint state and skips redundant widget
        writes so back-to-back identical updates do not trigger paint
        events.
        """
        if not hasattr(self, "_entry_progress_bar"):
            return
        if total <= 1:
            if getattr(self, "_entry_progress_last_visible", True):
                self._entry_progress_bar.hide()
                self._entry_progress_label.hide()
                self._entry_progress_last_visible = False
            return
        new_state = (int(current), int(total), str(entry), str(op_name))
        if getattr(self, "_entry_progress_last_state", None) == new_state:
            return
        self._entry_progress_last_state = new_state
        title = _OP_TITLES.get(op_name, op_name)
        entry_part = f" · {entry}" if entry else ""
        # setRange is cheap when the range is unchanged but still
        # forwards through QProgressBar; only call it when total moves.
        if self._entry_progress_bar.maximum() != total:
            self._entry_progress_bar.setRange(0, total)
        self._entry_progress_bar.setValue(current)
        self._entry_progress_label.setText(
            f"{title} · entry {current}/{total}{entry_part}"
        )
        if not getattr(self, "_entry_progress_last_visible", False):
            self._entry_progress_label.show()
            self._entry_progress_bar.show()
            self._entry_progress_last_visible = True

    def on_frame_progress(
        self, done: int, total: int, op_name: str, entry: str
    ) -> None:
        """Update the multi-frame progress row.

        Hides both widgets when ``total <= 1`` so single-frame runs
        contribute no UI noise. Otherwise paints a determinate bar
        and a one-line label, e.g. ``Detection · entry_4P · 3/12 frames``.
        ``op_name`` is the mlgidBASE method name (``run_detection`` /
        ``run_fitting`` / ``run_matching``) which we map to a human
        title; ``entry`` may be the literal entry name or a sentinel
        like ``"all entries"`` when the run iterates the whole file.

        Memoises the last paint state and skips redundant widget
        writes so a fast pipeline emitting many frame events per
        second does not turn into a paint storm.
        """
        if not hasattr(self, "_progress_bar"):
            return
        if op_name == "track_peaks":
            # track_peaks is one opaque mlgidBASE call dominated by the
            # file open and the all-against-all IoU/graph step, not the
            # (fast) per-frame reads — a determinate "N/M frames" bar
            # just sits at 0 then flickers to full. Show a busy
            # indicator (indeterminate marquee) for the whole run
            # instead; set it up once (later ticks carry no new info).
            if getattr(self, "_frame_progress_last_state", None) != ("track_peaks",):
                self._frame_progress_last_state = ("track_peaks",)
                self._progress_bar.setRange(0, 0)   # 0..0 == busy marquee
                self._progress_label.setText("Tracking peaks…")
                self._progress_label.show()
                self._progress_bar.show()
                self._frame_progress_last_visible = True
            return
        if total <= 1:
            if getattr(self, "_frame_progress_last_visible", True):
                self._progress_bar.hide()
                self._progress_label.hide()
                self._frame_progress_last_visible = False
            return
        new_state = (int(done), int(total), str(op_name), str(entry))
        if getattr(self, "_frame_progress_last_state", None) == new_state:
            return
        self._frame_progress_last_state = new_state
        title = _OP_TITLES.get(op_name, op_name)
        entry_part = f" · {entry}" if entry else ""
        if self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._progress_label.setText(f"{title}{entry_part} · {done}/{total} frames")
        if not getattr(self, "_frame_progress_last_visible", False):
            self._progress_label.show()
            self._progress_bar.show()
            self._frame_progress_last_visible = True

    # -- UI construction --

    def _build_ui(self) -> None:
        # Outer layout hosts only the scroll area — content widget owns
        # all the actual section margins so scrollbars sit flush against
        # the dock edge.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Vertical scroll fires only when content overflows; horizontal
        # is hard-locked off so a narrow dock collapses form rows
        # (labels wrap above fields, see ``_make_form``) instead of
        # introducing an x-axis scrollbar.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)

        # Progress rows — two stacked levels.
        #   * Entry row: only visible during an "All entries" run, shows
        #     "entry K/N · <entry>" and a determinate bar over the queue.
        #   * Frame row: visible during any multi-frame run, shows
        #     "<Op> · <entry> · K/N frames" and a bar over the entry's
        #     frames.
        # Both rows hide on idle and on single-{frame,entry} runs so the
        # progress region collapses to zero pixels in the steady state.
        self._entry_progress_label = QLabel("")
        self._entry_progress_label.setProperty("status", "muted")
        self._entry_progress_label.hide()
        self._entry_progress_bar = skin_progress(QProgressBar())
        self._entry_progress_bar.setRange(0, 1)
        self._entry_progress_bar.setValue(0)
        self._entry_progress_bar.setTextVisible(False)
        self._entry_progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._entry_progress_bar.hide()
        inner.addWidget(self._entry_progress_label)
        inner.addWidget(self._entry_progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setProperty("status", "muted")
        self._progress_label.hide()
        self._progress_bar = skin_progress(QProgressBar())
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._progress_bar.hide()
        inner.addWidget(self._progress_label)
        inner.addWidget(self._progress_bar)

        if not self._available:
            hint = QLabel(
                "<b>mlgidbase</b> is not installed in this environment.<br><br>"
                "Install it to enable detection, fitting, and matching:"
                "<pre>  pip install mlgidbase</pre>"
            )
            hint.setWordWrap(True)
            inner.addWidget(hint)
            inner.addStretch(1)
            scroll.setWidget(content)
            return

        # Run-all button — created up-front because the matching
        # section's initial _update_match_enabled call (during section
        # construction) reaches for self.btn_run_all to set its gated
        # state. The widget is added to the layout further down so it
        # sits pinned to the bottom of the dock.
        self.btn_run_all = _set_variant(QPushButton("Run full pipeline"), _PRIMARY)
        self.btn_run_all.setToolTip(
            "Run Detection, Fitting, and Matching back-to-back using the "
            "current section kwargs.\n\n"
            "Disabled until the active matching source (CIF or pickle) "
            "has a value."
        )
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.clicked.connect(self._on_run_all)

        # Sections are independent now: any combination of them can be
        # open at once. Detection starts open so the user has something
        # actionable on first sight.
        self._sections: list[_CollapsibleSection] = [
            self._build_detection_section(),
            self._build_fitting_section(),
            self._build_matching_section(),
        ]
        for s in self._sections:
            inner.addWidget(s)

        # Logs live in their own dock now (see MainWindow._logs_dock); the
        # trailing stretch keeps the sections at the top of the scroll
        # area when they don't fill the visible height. The run-all
        # button sits below the stretch so it stays pinned to the
        # bottom regardless of how many sections are expanded.
        inner.addStretch(1)
        inner.addWidget(self.btn_run_all)
        scroll.setWidget(content)

    def _build_detection_section(self) -> QWidget:
        section = _CollapsibleSection("Detection", expanded=True)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        # Entry scope — "Active entry" is the default to keep runs aligned
        # with what's on screen and avoid surprises on multi-entry files.
        self.det_entry_scope = QComboBox()
        self.det_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.det_entry_scope)

        self.det_frame_scope = QComboBox()
        self.det_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        self.det_frame_range, det_frames_row = self._frame_scope_widgets(
            self.det_frame_scope
        )
        form.addRow("Frames:", det_frames_row)

        # Model type — empty string means "use mlgidbase default". This
        # one stays a separate ``model_type`` kwarg rather than joining
        # the config dict below: ``load_config`` assigns it *after* the
        # dict, so keeping ``MODEL.TYPE`` out of the dict leaves one
        # source of truth for which weights a run loads (and leaves
        # ``detection_model.resolve_model_key``'s precedence alone).
        self.det_model_type = QComboBox()
        self.det_model_type.addItems(["(default)", "faster_rcnn", "dino"])
        self.det_model_type.setToolTip(
            "Detection model architecture. Leave on (default) unless your "
            "config selects a different backbone.\n\n"
            "(default) runs the dino model. faster_rcnn is the legacy "
            "model: it applies its own fixed post-processing and has no "
            "ensemble, so the score, NMS and ensemble settings are "
            "hidden while it is selected."
        )
        form.addRow("Model:", self.det_model_type)

        # The three parameters that get touched in practice sit inline;
        # everything else mlgidDETECT reads is one click away under
        # "Advanced". Every widget is seeded from ``DETECT_DEFAULTS``,
        # i.e. from mlgidDETECT's own defaults, so an untouched form
        # reproduces exactly what a run did before this section grew
        # fields.
        self.det_score = self._det_spin("POSTPROCESSING_SCORE", 0.0, 1.0, 0.05)
        self.det_score.setToolTip(
            "Confidence a box needs to be kept as a detected peak. "
            "Lower it to pull faint peaks out of a weak pattern, raise "
            "it when the image is full of spurious boxes.\n\n"
            "This decides which boxes are detected and WRITTEN to the "
            "file. The Display dock's minimum-score slider is a view "
            "filter: it only hides boxes that were already written."
        )
        self._add_dino_row(form, "Score threshold:", self.det_score)

        self.det_nms_iou = self._det_spin(
            "POSTPROCESSING_NMSIOU", 0.0, 1.0, 0.05
        )
        self.det_nms_iou.setToolTip(
            "Overlap (intersection over union) above which two boxes of "
            "the same peak are treated as one and the weaker is "
            "dropped. Raise it when a single reflection comes out as a "
            "stack of near-identical boxes; lower it when two genuinely "
            "neighbouring peaks keep being merged into one."
        )
        self._add_dino_row(form, "NMS IoU:", self.det_nms_iou)

        self.det_force_cpu = self._det_check("MODEL_FORCE_CPU")
        self.det_force_cpu.setToolTip(
            "Run inference on the CPU even when a usable GPU is "
            "present. Slower, but it is the documented way out of GPU "
            "trouble (a driver that falls over under load, an "
            "onnxruntime CUDA provider that will not initialize). See "
            "``detection_on_gpu`` in mlgidlab/pipeline.py."
        )
        form.addRow("Force CPU:", self.det_force_cpu)

        section.body_layout.addLayout(form)
        section.body_layout.addWidget(self._build_detection_advanced())
        # Every row exists now, so the model combo can start driving
        # which of them are shown.
        self.det_model_type.currentTextChanged.connect(
            self._on_det_model_changed
        )
        self._on_det_model_changed(self.det_model_type.currentText())
        self.btn_detect = _set_variant(QPushButton("Run detection"), _PRIMARY)
        self.btn_detect.clicked.connect(self._on_run_detection)
        section.body_layout.addWidget(self.btn_detect)
        return section

    def _build_detection_advanced(self) -> QWidget:
        """The rest of mlgidDETECT's configuration, collapsed by default.

        Everything here is real (mlgidDETECT reads every one of these
        attributes), but none of it is part of a normal run, so it would
        cost the inline rows their legibility. Collapsed, it stays
        discoverable without being in the way.
        """
        advanced = _CollapsibleSection("Advanced", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.det_classaware_nms = self._det_check(
            "POSTPROCESSING_CLASSAWARE_NMS"
        )
        self.det_classaware_nms.setToolTip(
            "Run the overlap suppression per class instead of across "
            "all boxes, so a ring and a segment at the same place can "
            "both survive. Applies to the 2-class ring/segment dino "
            "model; leave it off for the legacy single-class model, "
            "where every box is the same class anyway."
        )
        self._add_dino_row(form, "Class-aware NMS:", self.det_classaware_nms)

        self.det_nms_iou_ring = self._det_spin(
            "POSTPROCESSING_NMSIOU_RING", 0.0, 1.0, 0.05
        )
        self.det_nms_iou_ring.setToolTip(
            "Overlap threshold used for ring boxes once class-aware NMS "
            "is on. Rings are long and thin and overlap each other far "
            "more readily than peaks do, which is why their default "
            "(0.10) is stricter than the general one."
        )
        self._add_dino_row(form, "NMS IoU (rings):", self.det_nms_iou_ring)

        self.det_nms_iou_seg = self._det_spin(
            "POSTPROCESSING_NMSIOU_SEG", 0.0, 1.0, 0.05
        )
        self.det_nms_iou_seg.setToolTip(
            "Overlap threshold used for segment (peak) boxes once "
            "class-aware NMS is on."
        )
        self._add_dino_row(form, "NMS IoU (segments):", self.det_nms_iou_seg)

        self.det_ensemble = self._det_check("MODEL_ENSEMBLE_ENABLED")
        self.det_ensemble.setToolTip(
            "Fuse the predictions of two dino models (ONNX base + ONNX "
            "ensemble) instead of running one. Costs a second inference "
            "pass per frame and needs both model files below."
        )
        self._add_dino_row(form, "Ensemble:", self.det_ensemble)

        self.det_onnx_base = self._det_line("MODEL_ONNX_BASE", "base")
        self.det_onnx_base.setToolTip(
            "Path to an .onnx file to run instead of the weights "
            "mlgidDETECT downloads for the chosen Model. Leave empty "
            "for those downloaded weights. A path here also switches "
            "off the GUI's download pre-flight, which has nothing to "
            "fetch or verify for a file you supply yourself."
        )
        form.addRow("ONNX base:", self.det_onnx_base)

        self.det_onnx_ensemble = self._det_line("MODEL_ONNX_ENSEMBLE")
        self.det_onnx_ensemble.setToolTip(
            "Path to the second .onnx file, used only while Ensemble is "
            "ticked. Empty means none."
        )
        self._add_dino_row(form, "ONNX ensemble:", self.det_onnx_ensemble)

        self.det_redownload = self._det_check("MODEL_REDOWNLOAD")
        self.det_redownload.setToolTip(
            "Fetch the model weights again even though a cached copy "
            "exists. The way out of a cached file that loads but "
            "behaves oddly; a cached file that is plainly truncated is "
            "already replaced automatically."
        )
        form.addRow("Re-download weights:", self.det_redownload)

        self.det_log = self._det_check("PREPROCESSING_LOG")
        self.det_log.setToolTip(
            "Take the logarithm of the intensities before the model "
            "sees them. GIWAXS intensities span decades, so this is "
            "how the weights were trained; switching it off is for "
            "experiments, not for runs you want to keep."
        )
        form.addRow("Log scaling:", self.det_log)

        self.det_hist_eq = self._det_check(
            "PREPROCESSING_HISTOGRAMEQUALIZATION"
        )
        self.det_hist_eq.setToolTip(
            "Flatten the intensity histogram so faint and bright "
            "regions carry comparable contrast. Part of the training "
            "preprocessing as well."
        )
        form.addRow("Histogram equalize:", self.det_hist_eq)

        self.det_clipping = self._det_check("PREPROCESSING_PERFORMCLIPPING")
        self.det_clipping.setToolTip(
            "Clip the intensity range to the percentiles below before "
            "scaling, so a few hot pixels or a detector gap cannot set "
            "the contrast for the whole image."
        )
        form.addRow("Clipping:", self.det_clipping)

        self.det_clip_upper = self._det_spin(
            "PREPROCESSING_HIGHERCLIPPINGPERCENTILE", 0.0, 100.0, 0.5
        )
        self.det_clip_upper.setToolTip(
            "Upper clipping percentile: everything above it is pulled "
            "down to it. Lower this when a very bright reflection or a "
            "hot pixel flattens the rest of the pattern."
        )
        form.addRow("Clip upper pct:", self.det_clip_upper)

        self.det_clip_lower = self._det_spin(
            "PREPROCESSING_LOWERCLIPPINGPERCENTILE", 0.0, 100.0, 0.5
        )
        self.det_clip_lower.setToolTip(
            "Lower clipping percentile: everything below it is raised "
            "to it. Raise it to push detector background and masked "
            "zeros out of the contrast range."
        )
        form.addRow("Clip lower pct:", self.det_clip_lower)

        self.det_flip_horizontal = self._det_check(
            "PREPROCESSING_FLIPHORIZONTAL"
        )
        self.det_flip_horizontal.setToolTip(
            "Mirror the image in q_xy before detection. For data whose "
            "missing wedge sits on the other side than the model "
            "expects; converted files already have it on the expected "
            "side, so this stays off."
        )
        form.addRow("Flip horizontal:", self.det_flip_horizontal)

        self.det_debug = self._det_check("GENERAL_DEBUG")
        self.det_debug.setToolTip(
            "Put mlgidDETECT's own logger at DEBUG level for the run. "
            "Verbose, and it reaches the Logs dock."
        )
        form.addRow("Debug:", self.det_debug)

        advanced.body_layout.addLayout(form)
        self.btn_det_reset = QPushButton("Reset to defaults")
        self.btn_det_reset.setToolTip(
            "Put every detection parameter, inline and advanced, back "
            "to mlgidDETECT's own default value."
        )
        self.btn_det_reset.clicked.connect(self._reset_detection_params)
        advanced.body_layout.addWidget(self.btn_det_reset)
        return advanced

    # -- Detection parameter widgets --
    #
    # The three factories register each widget under its flat
    # ``SECTION_KEY`` name and seed it from ``DETECT_DEFAULTS``, so the
    # default, the reset value and the key sent to mlgidDETECT cannot
    # drift apart per widget: there is one table, read three ways.

    def _det_spin(
        self, key: str, lo: float, hi: float, step: float
    ) -> QDoubleSpinBox:
        """A registered detection spin box, two decimals."""
        spin = spin_double(lo, hi, float(self._det_defaults[key]), decimals=2)
        spin.setSingleStep(step)
        self._det_param_widgets[key] = spin
        return spin

    def _det_check(self, key: str) -> QCheckBox:
        """A registered detection check box."""
        box = QCheckBox()
        box.setChecked(bool(self._det_defaults[key]))
        self._det_param_widgets[key] = box
        return box

    def _det_line(self, key: str, placeholder: str = "") -> QLineEdit:
        """A registered detection text field.

        Empty means "mlgidDETECT's default", which is what the
        placeholder shows; nothing is typed in for the user, so the
        pristine state is visibly pristine.
        """
        edit = QLineEdit()
        if placeholder:
            edit.setPlaceholderText(placeholder)
        self._det_param_widgets[key] = edit
        return edit

    def _add_dino_row(
        self, form: QFormLayout, label: str, field: QWidget
    ) -> None:
        """Add a row that only the dino model reads.

        mlgidDETECT's ``faster_rcnn`` branch ignores the score and NMS
        settings (it hardcodes its own post-processing) and never fuses
        an ensemble, so those rows are hidden while it is selected
        rather than left in the form doing nothing.
        """
        form.addRow(label, field)
        self._det_dino_rows.append((form, field))

    def _on_det_model_changed(self, model: str) -> None:
        """Show only the rows the selected model actually reads.

        ``(default)`` resolves to dino, so only an explicit
        ``faster_rcnn`` hides anything. The hidden widgets keep their
        values and still travel in ``config_detect``: the backend never
        reads them under ``faster_rcnn``, and keeping them means an
        edit survives flipping the combo back and forth.
        """
        dino = model != "faster_rcnn"
        for form, field in self._det_dino_rows:
            form.setRowVisible(field, dino)

    def _detection_params(self) -> dict[str, object]:
        """The detection form as a flat ``{"SECTION_KEY": value}`` map."""
        flat: dict[str, object] = {}
        for key, widget in self._det_param_widgets.items():
            if isinstance(widget, QCheckBox):
                flat[key] = bool(widget.isChecked())
            elif isinstance(widget, QDoubleSpinBox):
                flat[key] = float(widget.value())
            else:
                # Text field: an empty one means the default, which for
                # ONNX ensemble is None and gets dropped on nesting.
                flat[key] = widget.text().strip() or self._det_defaults[key]
        return flat

    def _reset_detection_params(self) -> None:
        """Put every detection widget back to mlgidDETECT's default."""
        for key, value in self._det_defaults.items():
            widget = self._det_param_widgets.get(key)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif widget is not None:
                widget.clear()

    def _build_fitting_section(self) -> QWidget:
        section = _CollapsibleSection("Fitting", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.fit_entry_scope = QComboBox()
        self.fit_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.fit_entry_scope)

        self.fit_frame_scope = QComboBox()
        self.fit_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        self.fit_frame_range, fit_frames_row = self._frame_scope_widgets(
            self.fit_frame_scope
        )
        form.addRow("Frames:", fit_frames_row)

        # Match mlgidbase defaults exactly so an unedited form yields the
        # same behaviour as a bare ``analysis.run_fitting()`` call (the
        # clustering distances are 2 since mlgidbase 0.1.11, they were 10;
        # the numbers live in ``manual_fit`` so every fallback agrees).
        self.fit_crit_angle = QDoubleSpinBox()
        self.fit_crit_angle.setDecimals(3)
        self.fit_crit_angle.setRange(0.0, 90.0)
        self.fit_crit_angle.setSingleStep(0.5)
        self.fit_crit_angle.setValue(0.0)
        self.fit_crit_angle.setSuffix(" °")
        self.fit_crit_angle.setToolTip(
            "Critical angle of the sample, in degrees. Together with the "
            "incidence angle and the wavelength it places the sample-horizon "
            "q_z cut, below which the image is masked out before fitting."
        )
        form.addRow("Critical angle:", self.fit_crit_angle)

        self.fit_dist_peaks = QDoubleSpinBox()
        self.fit_dist_peaks.setDecimals(2)
        self.fit_dist_peaks.setRange(0.0, 1000.0)
        self.fit_dist_peaks.setSingleStep(1.0)
        self.fit_dist_peaks.setValue(DEFAULT_CLUSTERING_DISTANCE_PEAKS)
        self.fit_dist_peaks.setToolTip(
            "Distance threshold for peak clustering, in pixels of "
            "pygidfit's polar grid: peaks closer than this are fitted "
            "together. Default 2 (mlgidbase 0.1.11; it was 10 before). "
            "The 2D fit of a single peak and the profile neighbour "
            "mask use the same value."
        )
        form.addRow("Cluster dist (peaks):", self.fit_dist_peaks)

        self.fit_dist_rings = QDoubleSpinBox()
        self.fit_dist_rings.setDecimals(2)
        self.fit_dist_rings.setRange(0.0, 1000.0)
        self.fit_dist_rings.setSingleStep(1.0)
        self.fit_dist_rings.setValue(DEFAULT_CLUSTERING_DISTANCE_RINGS)
        self.fit_dist_rings.setToolTip(
            "Distance threshold for ring clustering, in pixels of "
            "pygidfit's polar grid. Default 2 (mlgidbase 0.1.11; it was "
            "10 before)."
        )
        form.addRow("Cluster dist (rings):", self.fit_dist_rings)

        self.fit_cluster_extend = QSpinBox()
        self.fit_cluster_extend.setRange(0, 100)
        self.fit_cluster_extend.setValue(DEFAULT_CLUSTERING_EXTEND)
        self.fit_cluster_extend.setToolTip(
            "Number of neighboring peaks to include in cluster expansion."
        )
        form.addRow("Cluster extend:", self.fit_cluster_extend)

        self.fit_theta_fixed = QCheckBox()
        self.fit_theta_fixed.setChecked(True)
        self.fit_theta_fixed.setToolTip(
            "Hold theta fixed during clustering. Default in mlgidBASE."
        )
        form.addRow("Theta fixed:", self.fit_theta_fixed)

        self.fit_use_pool = QCheckBox()
        self.fit_use_pool.setChecked(False)
        self.fit_use_pool.setToolTip(
            "Use multiprocessing for fitting (faster on large stacks, "
            "but interleaves logs unpredictably)."
        )
        form.addRow("Use pool:", self.fit_use_pool)

        self.fit_debug = QCheckBox()
        self.fit_debug.setChecked(False)
        form.addRow("Debug:", self.fit_debug)

        section.body_layout.addLayout(form)
        self.btn_fit = _set_variant(QPushButton("Run fitting"), _PRIMARY)
        self.btn_fit.clicked.connect(self._on_run_fitting)
        section.body_layout.addWidget(self.btn_fit)
        return section

    def _build_matching_section(self) -> QWidget:
        section = _CollapsibleSection("Matching", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.match_entry_scope = QComboBox()
        self.match_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.match_entry_scope)

        self.match_frame_scope = QComboBox()
        self.match_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        self.match_frame_range, match_frames_row = self._frame_scope_widgets(
            self.match_frame_scope
        )
        form.addRow("Frames:", match_frames_row)

        # Mutually-exclusive radio buttons in a single QButtonGroup —
        # one sits at the start of each input row, so the user picks
        # the active source by ticking the radio next to the field
        # they want to use. The unticked row's widgets are greyed out
        # to remove any ambiguity about which value Run-Matching
        # consumes.
        self.rb_cif_source = QRadioButton()
        self.rb_pickle_source = QRadioButton()
        self.rb_cif_source.setToolTip(
            "Use the raw CIF input below as the matching source."
        )
        self.rb_pickle_source.setToolTip(
            "Use the pickle file below as the matching source."
        )
        self._source_group = QButtonGroup(self)
        self._source_group.setExclusive(True)
        self._source_group.addButton(self.rb_cif_source)
        self._source_group.addButton(self.rb_pickle_source)
        self.rb_cif_source.setChecked(True)
        # Either toggled signal fires for both buttons in an exclusive
        # group, so we get one update per user click. ``buttonToggled``
        # gives us the QAbstractButton + new state.
        self._source_group.buttonToggled.connect(
            lambda *_: self._on_match_source_changed()
        )

        self.cif_path = QLineEdit()
        self.cif_path.setPlaceholderText("Raw .cif file(s) or folder…")
        self.cif_path.setToolTip(
            "Raw CIF input forwarded as ``cif_prepr`` to "
            "mlgidBASE.run_matching. Accepts:\n"
            "  • one or more raw .cif files (semicolon-separated)\n"
            "  • a folder containing .cif files\n\n"
            "ExpParameters are auto-derived from the active NeXus file's "
            "instrument metadata."
        )
        self._cif_browse_btn = QPushButton("Browse…")
        self._cif_browse_btn.clicked.connect(self._browse_cif)
        self._cif_browse_dir_btn = QPushButton("Folder…")
        self._cif_browse_dir_btn.setToolTip(
            "Pick a directory; every .cif inside is used."
        )
        self._cif_browse_dir_btn.clicked.connect(self._browse_cif_dir)
        cif_row = QWidget()
        cif_h = QHBoxLayout(cif_row)
        cif_h.setContentsMargins(0, 0, 0, 0)
        cif_h.addWidget(self.rb_cif_source)
        cif_h.addWidget(self.cif_path, 1)
        cif_h.addWidget(self._cif_browse_btn)
        cif_h.addWidget(self._cif_browse_dir_btn)
        form.addRow("CIF input:", cif_row)

        # Parse button + cache-state label. CIF preprocessing simulates
        # every input pattern, which is slow — caching the resulting
        # CifPattern across multiple Run-Matching clicks is a huge time
        # saver. The cache is invalidated whenever the input text changes.
        self.btn_parse_cifs = QPushButton("Parse CIFs")
        self.btn_parse_cifs.setToolTip(
            "Pre-load the CIF input into a CifPattern that subsequent "
            "matching runs reuse. Without this, every Run Matching "
            "re-parses the CIFs from scratch.\n\n"
            "Applies only to the CIF input above; pickles are forwarded "
            "directly without preprocessing."
        )
        self.btn_parse_cifs.clicked.connect(self._on_parse_cifs)
        self.btn_parse_cifs.setEnabled(False)
        # Compact busy bar shown while the parse worker runs. CifPattern
        # simulates every CIF in one opaque call (no per-CIF callback to
        # hook), so the bar is indeterminate — but it moves, which the
        # static "Parsing…" text alone does not.
        self.cif_parse_bar = skin_progress(QProgressBar())
        self.cif_parse_bar.setRange(0, 0)
        self.cif_parse_bar.setTextVisible(False)
        self.cif_parse_bar.setFixedWidth(110)
        self.cif_parse_bar.setFixedHeight(14)
        self.cif_parse_bar.hide()
        self.cif_cache_label = QLabel("Not parsed")
        self.cif_cache_label.setProperty("status", "muted")
        cif_parse_row = QWidget()
        cif_parse_h = QHBoxLayout(cif_parse_row)
        cif_parse_h.setContentsMargins(0, 0, 0, 0)
        cif_parse_h.addWidget(self.btn_parse_cifs)
        cif_parse_h.addWidget(self.cif_parse_bar)
        cif_parse_h.addWidget(self.cif_cache_label, 1)
        form.addRow("", cif_parse_row)
        # Any edit invalidates the cache and re-enables the button.
        self.cif_path.textChanged.connect(self._on_cif_input_changed)

        # Separate pickle input — preprocessed CifPattern files skip the
        # raw-CIF preprocessing step entirely; mlgidBASE.run_matching
        # accepts a path-to-pickle string verbatim. Pickle path takes
        # priority over the CIF input above when both are set.
        self.pickle_path = QLineEdit()
        self.pickle_path.setPlaceholderText(
            "Preprocessed CIF pickle (.pickle / .pkl)…"
        )
        self.pickle_path.setToolTip(
            "Path to a preprocessed CifPattern pickle. Forwarded as "
            "``cif_prepr`` to mlgidBASE.run_matching when ``Source`` "
            "above is set to ``Pickle file``."
        )
        self._pickle_browse_btn = QPushButton("Browse…")
        self._pickle_browse_btn.clicked.connect(self._browse_pickle)
        self._pickle_clear_btn = QPushButton("Clear")
        self._pickle_clear_btn.clicked.connect(lambda: self.pickle_path.setText(""))
        pickle_row = QWidget()
        pickle_h = QHBoxLayout(pickle_row)
        pickle_h.setContentsMargins(0, 0, 0, 0)
        pickle_h.addWidget(self.rb_pickle_source)
        pickle_h.addWidget(self.pickle_path, 1)
        pickle_h.addWidget(self._pickle_browse_btn)
        pickle_h.addWidget(self._pickle_clear_btn)
        form.addRow("Pickle input:", pickle_row)

        self.peaks_type = QComboBox()
        self.peaks_type.addItems(["segments", "rings"])
        form.addRow("Peaks type:", self.peaks_type)

        # Probability threshold (mlgidBASE alias: ``threshold``). The two
        # are equivalent on the mlgidBASE side; we send ``threshold`` since
        # that's what the underlying _run_matching prefers.
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        self.threshold.setToolTip(
            "Minimum probability for a CIF candidate to be accepted."
        )
        form.addRow("Probability threshold:", self.threshold)

        self.intensity_threshold = QDoubleSpinBox()
        self.intensity_threshold.setRange(0.0, 1e9)
        self.intensity_threshold.setSingleStep(1.0)
        self.intensity_threshold.setDecimals(3)
        self.intensity_threshold.setValue(0.0)
        self.intensity_threshold.setToolTip(
            "Minimum peak intensity to consider during matching."
        )
        form.addRow("Intensity threshold:", self.intensity_threshold)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("Device:", self.device)

        section.body_layout.addLayout(form)

        self.btn_match = _set_variant(QPushButton("Run matching"), _PRIMARY)
        self.btn_match.setEnabled(False)
        self.btn_match.clicked.connect(self._on_run_matching)
        # Gate run button on the active source having text — matching
        # can't proceed without an input value.
        self.cif_path.textChanged.connect(self._update_match_enabled)
        self.pickle_path.textChanged.connect(self._update_match_enabled)
        section.body_layout.addWidget(self.btn_match)
        # Apply the initial enabled / disabled state now that all
        # widgets exist; running the slot earlier would crash because
        # btn_match isn't built until this line.
        self._on_match_source_changed()
        return section

    def _update_match_enabled(self) -> None:
        # Run-Match and Run-Full-Pipeline are both gated on the *selected*
        # source having a value; text in the inactive row is irrelevant.
        # Run-Full-Pipeline shares the gate because Matching is the last
        # stage of the chain and it can't proceed without a source.
        if self._use_pickle_source():
            has_input = bool(self.pickle_path.text().strip())
        else:
            has_input = bool(self.cif_path.text().strip())
        self.btn_match.setEnabled(has_input)
        self.btn_run_all.setEnabled(has_input)

    def _use_pickle_source(self) -> bool:
        return self.rb_pickle_source.isChecked()

    def _on_match_source_changed(self, *_args) -> None:
        """Enable only the input row matching the active source.

        Greying the unused row out (instead of hiding it) keeps the
        layout stable and lets the user see what's typed in the
        inactive field — useful when alternating between sources.
        Run-Matching gating is recomputed because the enabled-text
        check now points at the other field.
        """
        use_pickle = self._use_pickle_source()
        # CIF inputs + parse machinery are CIF-only.
        for w in (
            self.cif_path,
            self._cif_browse_btn,
            self._cif_browse_dir_btn,
            self.btn_parse_cifs,
            self.cif_cache_label,
        ):
            w.setEnabled(not use_pickle)
        # Re-apply the parse button's text-based gating after the
        # source flip so it doesn't sit enabled on an empty CIF field.
        if not use_pickle:
            self.btn_parse_cifs.setEnabled(bool(self.cif_path.text().strip()))
        # Pickle widgets.
        for w in (
            self.pickle_path,
            self._pickle_browse_btn,
            self._pickle_clear_btn,
        ):
            w.setEnabled(use_pickle)
        self._update_match_enabled()

    # -- Click handlers --

    def _on_run_detection(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.det_entry_scope, kwargs)
        if not self._inject_frame_scope(
            self.det_frame_scope, self.det_frame_range, kwargs
        ):
            return
        # Always sent, never conditional: an untouched form carries
        # mlgidDETECT's own defaults, so there is no state in which
        # omitting the dict would mean something different from sending
        # it. ``pipeline.execute`` builds a fresh mlgidBASE per command,
        # so an edit takes effect on the very next click.
        kwargs["config_detect"] = build_config_detect(
            self._detection_params()
        )
        model = self.det_model_type.currentText()
        if model and not model.startswith("("):
            kwargs["model_type"] = model
        self.runRequested.emit(PipelineCommand("run_detection", kwargs))

    def _on_run_fitting(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.fit_entry_scope, kwargs)
        if not self._inject_frame_scope(
            self.fit_frame_scope, self.fit_frame_range, kwargs
        ):
            return
        kwargs["crit_angle"] = float(self.fit_crit_angle.value())
        kwargs["clustering_distance_peaks"] = float(self.fit_dist_peaks.value())
        kwargs["clustering_distance_rings"] = float(self.fit_dist_rings.value())
        kwargs["clustering_extend"] = int(self.fit_cluster_extend.value())
        kwargs["theta_fixed"] = bool(self.fit_theta_fixed.isChecked())
        kwargs["use_pool"] = bool(self.fit_use_pool.isChecked())
        kwargs["debug"] = bool(self.fit_debug.isChecked())
        self.runRequested.emit(PipelineCommand("run_fitting", kwargs))

    def _on_run_all(self) -> None:
        """Chain Detection → Fitting → Matching as three queued commands.

        Each stage reuses its per-stage handler so kwarg-building stays
        in one place. The host's runRequested dispatcher queues each
        emitted command onto the same worker thread, so the three
        stages run sequentially without overlapping. "All entries"
        scope is still expanded per-entry inside each stage.

        Errors in earlier stages don't abort the chain — the existing
        queue logs and continues, matching the per-entry batch behaviour.
        """
        # Belt-and-braces — the button is gated, but a programmatic
        # caller could still reach this with no source set.
        if self._use_pickle_source():
            if not self.pickle_path.text().strip():
                return
        else:
            if not self.cif_path.text().strip():
                return
        self._on_run_detection()
        self._on_run_fitting()
        self._on_run_matching()

    def matching_kwargs(self) -> dict | None:
        """Base ``run_matching`` kwargs (no entry/frame scope) from the
        panel's current CIF source + parameters, or ``None`` when no
        CIF source is configured.

        The Source selector decides which input is used; the inactive
        row is greyed out and its content ignored. mlgidBASE.run_matching's
        ``load_cif_prepr`` accepts a path-to-pickle string verbatim, so
        the pickle path is forwarded untouched. For raw CIFs we reuse
        the cached CifPattern when the input hasn't changed since the
        last parse, otherwise we send the string and let the worker
        build the pattern. Public so the host can chain a matching pass
        (e.g. after Interpolate-track's gap fills) with the SAME
        settings the panel's own Run matching would use.
        """
        if not self._available:
            # Backend-less install: the matching widgets were never
            # built — report "no CIF source configured".
            return None
        if self._use_pickle_source():
            pkl = self.pickle_path.text().strip()
            if not pkl:
                return None
            cif_value: object = pkl
        else:
            cif = self.cif_path.text().strip()
            if not cif:
                return None
            if self._cached_cif_obj is not None and self._cached_cif_input == cif:
                cif_value = self._cached_cif_obj
            else:
                cif_value = cif
        return {
            "cif_prepr": cif_value,
            "peaks_type": self.peaks_type.currentText(),
            "threshold": float(self.threshold.value()),
            "intensity_threshold": float(self.intensity_threshold.value()),
            "device": self.device.currentText(),
        }

    def cif_source_text(self) -> str:
        """The RAW .cif source text (folder / file / ``;``-list) when
        the CIF input row is active — ``""`` for the pickle source or
        an empty field. Lets the host build CIF-restricted matching
        commands (Interpolate track) even when ``matching_kwargs()``
        returns the cached pre-parsed ``CifPattern`` object, which
        cannot be subset."""
        if not self._available:
            # Backend-less install: the matching widgets were never
            # built (the panel shows the install hint instead).
            return ""
        if self._use_pickle_source():
            return ""
        return self.cif_path.text().strip()

    def _on_run_matching(self) -> None:
        kwargs = self.matching_kwargs()
        if kwargs is None:
            return
        self._inject_entry_scope(self.match_entry_scope, kwargs)
        if not self._inject_frame_scope(
            self.match_frame_scope, self.match_frame_range, kwargs
        ):
            return
        self.runRequested.emit(PipelineCommand("run_matching", kwargs))

    def _on_parse_cifs(self) -> None:
        text = self.cif_path.text().strip()
        if not text:
            return
        # Disable the button + show in-progress text. The host runs the
        # parse on a worker thread and posts the result back via
        # set_cif_pattern.
        self.btn_parse_cifs.setEnabled(False)
        self.btn_parse_cifs.setText("Parsing…")
        self.cif_parse_bar.show()
        self._set_cache_status("Parsing…", "warn")
        self.parseCifsRequested.emit(text)

    def _on_cif_input_changed(self, text: str) -> None:
        """Invalidate the cache + re-enable the parse button on edit."""
        text = text.strip()
        # Only invalidate if the text actually differs from the cache —
        # programmatic Browse → setText that exactly matches the cached
        # input shouldn't blow it away.
        if self._cached_cif_input is not None and text != self._cached_cif_input:
            self._cached_cif_obj = None
            self._cached_cif_input = None
            self._set_cache_status("Input changed; re-parse", "error")
            self.cifCacheChanged.emit(None)
        elif self._cached_cif_input is None:
            self._set_cache_status("Not parsed", "muted")
        self.btn_parse_cifs.setText("Parse CIFs")
        self.btn_parse_cifs.setEnabled(bool(text))

    def clear_cif_cache(self) -> None:
        """Forget any cached CifPattern.

        Used by the host on session swap because ``ExpParameters`` are
        derived from the active NeXus file's instrument metadata — a
        cache built against file A's params can't be safely reused when
        running matching on file B.
        """
        if self._cached_cif_obj is None and self._cached_cif_input is None:
            return
        self._cached_cif_obj = None
        self._cached_cif_input = None
        # Widget updates only when the matching UI was built (backend
        # present) — the cache fields themselves exist either way.
        if self._available:
            self._set_cache_status("Not parsed (active file changed)", "muted")
            self.btn_parse_cifs.setText("Parse CIFs")
            self.btn_parse_cifs.setEnabled(bool(self.cif_path.text().strip()))
        self.cifCacheChanged.emit(None)

    def set_cif_pattern(self, obj: object | None, error: Exception | None) -> None:
        """Host posts the parse result here. None+exception → error state.

        On success, ``obj`` is cached against the current input text and
        every subsequent Run Matching reuses it.

        Widget updates are gated on ``self._available`` (the matching
        widgets are never built backend-less) so tests can seed a fake
        cache through this public API in any environment.
        """
        if self._available:
            self.btn_parse_cifs.setText("Parse CIFs")
            self.btn_parse_cifs.setEnabled(True)
            self.cif_parse_bar.hide()
        if error is not None or obj is None:
            self._cached_cif_obj = None
            self._cached_cif_input = None
            msg = str(error) if error is not None else "(empty result)"
            # Keep the user's input alone so they can retry from the same
            # text; just flag the cache as failed.
            if self._available:
                self._set_cache_status(f"Parse failed — {msg[:60]}", "error")
            self.cifCacheChanged.emit(None)
            return
        self._cached_cif_input = (
            self.cif_path.text().strip() if self._available else ""
        )
        self._cached_cif_obj = obj
        # Surface the CIF count + the kind of input so the user can see
        # what's been cached (CifPattern exposes ``cifs``).
        if self._available:
            n = len(getattr(obj, "cifs", []) or [])
            self._set_cache_status(f"Parsed: {n} CIF(s) cached", "ok")
        self.cifCacheChanged.emit(obj)

    def _set_cache_status(self, text: str, level: str = "muted") -> None:
        """Set the CIF-cache line's text and semantic state.

        The state colours used to be six hardcoded hexes at six call
        sites, all tuned for the dark theme; ``level`` maps to a
        ``QLabel[status=...]`` rule in the skin instead, so the line is
        correct in both themes. Qt only re-evaluates an attribute
        selector when the widget is re-polished, hence the repolish.
        """
        self.cif_cache_label.setText(text)
        self.cif_cache_label.setProperty("status", level)
        style = self.cif_cache_label.style()
        style.unpolish(self.cif_cache_label)
        style.polish(self.cif_cache_label)

    def cached_cif_pattern(self) -> object | None:
        """The parsed CifPattern cache (None when nothing is parsed).

        Read-only accessor for the Display dock's "Expected pattern"
        section; touches no widgets, so it is backend-less-safe.
        """
        return self._cached_cif_obj

    # -- Internals --

    def _inject_entry_scope(self, combo: QComboBox, kwargs: dict) -> None:
        """Translate the entry-scope dropdown into mlgidBASE's ``entry`` kwarg.

        - ``ENTRY_ACTIVE``: insert ``entry=<active>`` (skip if no resolver
          or no active entry — mlgidBASE will then iterate all).
        - ``ENTRY_ALL``: leave ``entry`` out of kwargs so mlgidBASE
          defaults to all entries (the host's queue dispatcher then
          expands this into per-entry runs).
        - any other value: a literal entry name appended after
          ACTIVE/ALL, used when the user explicitly picks one entry.
        """
        choice = combo.currentText()
        if choice == ENTRY_ALL or not choice:
            return
        if choice == ENTRY_ACTIVE:
            if self._get_active_entry is None:
                return
            active = self._get_active_entry()
            if active:
                kwargs["entry"] = active
            return
        # Literal entry name — sent verbatim.
        kwargs["entry"] = choice

    def set_available_entries(self, entries: list[str]) -> None:
        """Refresh the per-entry options in all three entry-scope combos.

        Called by the host whenever the active session changes (or after
        a pipeline op that might have added entries — though no current
        op does). Rebuilds the items as
        ``[ACTIVE, ALL, entry_0000, entry_0001, …]`` while preserving
        the user's prior selection if it's still valid.
        """
        if not self._available:
            return
        for combo in (
            self.det_entry_scope,
            self.fit_entry_scope,
            self.match_entry_scope,
        ):
            previous = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems([ENTRY_ACTIVE, ENTRY_ALL, *entries])
                # Restore the prior selection when still in the new list,
                # otherwise default back to "Active entry".
                idx = combo.findText(previous)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                combo.blockSignals(False)

    def _frame_scope_widgets(
        self, combo: QComboBox
    ) -> tuple[QLineEdit, QWidget]:
        """Extend a Frames scope combo with the "Frame range…" option
        and its expression field (shown only while that scope is
        selected). Returns ``(edit, row_widget)``; the row replaces the
        bare combo in the form so the layout keeps its shape."""
        combo.addItem(FRAME_RANGE)
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 0-34,40")
        edit.setToolTip(
            "Frames to process: comma-separated 0-based indices and "
            "inclusive A-B ranges (like 0-34,40). Checked against the "
            "active entry's frame count when you press Run."
        )
        edit.setVisible(False)
        combo.currentTextChanged.connect(
            lambda text, e=edit: e.setVisible(text == FRAME_RANGE)
        )
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(combo)
        lay.addWidget(edit, 1)
        return edit, row

    def _inject_frame_scope(
        self, combo: QComboBox, edit: QLineEdit, kwargs: dict
    ) -> bool:
        """Resolve the Frames scope into ``kwargs["frame_num"]``.

        Returns False when the run must not start — a malformed
        "Frame range…" expression or one with no frame inside the
        scan (the user is told via a warning box). mlgidBASE accepts
        ``frame_num`` as None / int / list, so the range simply rides
        the existing kwarg."""
        scope = combo.currentText()
        if scope == FRAME_ACTIVE:
            if self._get_active_frame is None:
                return True
            active = self._get_active_frame()
            if active is not None:
                kwargs["frame_num"] = int(active)
            return True
        if scope != FRAME_RANGE:
            return True
        n_frames = (
            self._get_frame_count() if self._get_frame_count else None
        ) or 0
        try:
            valid, dropped = parse_frame_range(
                edit.text(), n_frames=int(n_frames)
            )
        except ValueError as exc:
            QMessageBox.warning(
                self, "Frame range",
                f"Cannot run: invalid frame range — {exc}",
            )
            return False
        if not valid:
            QMessageBox.warning(
                self, "Frame range",
                "Cannot run: no frame in the range lies inside the "
                f"scan (frames 0..{max(int(n_frames) - 1, 0)}).",
            )
            return False
        if dropped:
            self.append_log(
                f"Frame range: ignoring {len(dropped)} frame(s) outside "
                f"0..{int(n_frames) - 1}."
            )
        kwargs["frame_num"] = valid
        return True

    def _browse_cif(self) -> None:
        """Pick one-or-more raw .cif files.

        Multi-select joins paths with ``;``; ``pipeline.execute`` splits
        this back out and wraps the CIFs in a CifPattern at run time.
        Pickle input has its own dedicated picker — see ``_browse_pickle``.
        """
        paths = file_dialogs.open_files(
            self,
            "Select CIF file(s)",
            "CIF files (*.cif);;All files (*)",
        )
        if not paths:
            return
        self.cif_path.setText(";".join(paths))

    def _browse_cif_dir(self) -> None:
        """Pick a directory of .cif files. Forwarded as a single path."""
        directory = file_dialogs.existing_directory(
            self, "Select folder with CIF files"
        )
        if directory:
            self.cif_path.setText(directory)

    def _browse_pickle(self) -> None:
        """Pick a single preprocessed CifPattern pickle file."""
        path = file_dialogs.open_file(
            self,
            "Select preprocessed CIF pickle",
            "Pickle (*.pickle *.pkl);;All files (*)",
        )
        if path:
            self.pickle_path.setText(path)
