from __future__ import annotations

import json
import os

# Pin pyqtgraph to PySide6 before it auto-detects.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSignalBlocker,
    QSize,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QIcon, QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidgui.file_model import (
    EntryStack,
    MatchedStructure,
    PeakTable,
    _LazyImageStack,
    _LazyPolarStack,
)

if TYPE_CHECKING:
    from mlgidgui.file_model import FrameSource
from mlgidgui import peak_lists
from mlgidgui.polar import polar_to_qxyz
from mlgidgui.flow_layout import ToolGroup, wrapping_bar
from mlgidgui.image_viewer_overlays import ViewerOverlaysMixin
from mlgidgui.image_viewer_render import ViewerRenderMixin
from mlgidgui.image_viewer_interact import ViewerInteractMixin
from mlgidgui import icons, simulation_pattern

import logging
logger = logging.getLogger(__name__)

# Re-exports: canonical homes moved in the 2026 source split; kept here
# so the many importers (main_window mixins, panels, tests) resolve
# unchanged.
from mlgidgui.viewer_styles import (
    COLORMAPS,
    DEFAULT_COLORMAP,
    colormap_swatch,
    FITTED_PREVIEW_OPACITY,
    FITTED_PREVIEW_STYLE,
    fit_area_style,
    FIT_AREA_STYLE,
    HOVER_OPACITY,
    hover_style,
    MATCHED_LINE_STYLES,
    MATCHED_LINE_WIDTH,
    MATCHED_MARKER_SIZE,
    MATCHED_PALETTE,
    MATCHED_STYLE,
    MODE_CARTESIAN,
    MODE_POLAR,
    MODE_RAW,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    SELECTION_STYLE,
    selection_style,
    SIM_EXPLAINED_COLOR,
    SIM_MARKER_MAX_PX,
    SIM_MARKER_MIN_PX,
    SIM_MISSED_COLOR,
    SIM_OVERLAY_OPACITY,
    SIM_SELECTED_COLOR,
    UNMATCHED_COLOR,
    UNMATCHED_KEY,
    UNMATCHED_UID,
    _MATCHED_COLORS_KEY,
    _SIM_STATE_COLORS,
    _load_matched_pen_overrides,
    _save_matched_pen_overrides,
    load_overlay_pen_overrides,
    overlay_pen as _overlay_pen,
    save_overlay_pen_overrides,
    _sim_intensity_scale,
    _sim_marker_size,
    matched_pen_for,
)
from mlgidgui.viewer_items import (
    FileGeomAction,
    ManualAddAction,
    ManualGeomAction,
    ManualPeak,
    ManualRemoveAction,
    ManualReplaceAction,
    SelectedPeak,
    _Action,
    _DisplayParams,
    _LabelEventFilter,
    _PeakShapeItem,
    _apply_raw_flips,
    _bin_index,
    _cart_box_contains,
    _cart_table_row_contains,
    _cart_to_polar,
    _clip_angle,
    _disable_viewport_scroll,
    _has_label_modifiers,
    _peaks_from_manual,
    _peaks_subset,
    _polar_box_contains,
    _polar_rect_polygon,
    _polar_table_row_contains,
    _robust_levels,
)


#: Edge of the square colormap chip. The pixmap is rendered at 32 px so
#: it stays crisp if a style asks for more than the 16 px it is shown at.
CMAP_SWATCH = 32
CMAP_ICON_SIZE = 16


class _SwatchCombo(QComboBox):
    """A combo box whose icon size survives a style change.

    Qt clears an explicitly set ``iconSize`` when the widget is polished
    under an application stylesheet — it comes back as 0x0, and a 0x0
    icon is simply not drawn, so the colormap chip disappears from the
    closed box. That happens on the first show and again on every theme
    flip (``_set_theme`` re-polishes every widget), so the size has to be
    re-applied rather than set once at construction.

    The re-apply is deferred by a zero-timer: Qt clears the size *after*
    the style change is delivered, so setting it from inside
    ``changeEvent`` is undone immediately — the chip survived the first
    show and then vanished on the first theme flip. (The QSS
    ``icon-size`` property looks like the tidy answer but does not reach
    this combo; it is nested inside the viewer, not a top-level widget.)
    """

    def __init__(self, icon_size: QSize, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wanted_icon_size = icon_size
        self.setIconSize(icon_size)

    def _restore_icon_size(self) -> None:
        if self.iconSize() != self._wanted_icon_size:
            self.setIconSize(self._wanted_icon_size)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange:
            QTimer.singleShot(0, self._restore_icon_size)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._restore_icon_size()


def _segment_button(text: str, glyph: str, position: str) -> QToolButton:
    """One half of a segmented, mutually exclusive pair.

    ``position`` ("left" / "right") tells the skin which outer corners to
    round and which inner border to drop, so the two halves meet on a
    single shared edge instead of showing a double rule.
    """
    button = QToolButton()
    button.setText(text)
    button.setCheckable(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setProperty("segment", position)
    icons.bind(button, glyph)
    return button


class GIWAXSImageViewer(
    ViewerOverlaysMixin,
    ViewerRenderMixin,
    ViewerInteractMixin,
    QWidget,
):
    """Image viewer with Cartesian ↔ polar mode toggle + peak-box overlays."""

    # QSettings key for the last Custom aspect ratio (the on-screen
    # width:height). The mode itself is not persisted — the viewer always
    # starts in Fit; only the ratio is remembered so re-entering Custom is
    # sensible. Shares the global store with the theme / playback settings.
    _ASPECT_RATIO_KEY = "viewerAspectRatio"

    frameChanged = Signal(int)
    # Cursor readout — emits a dict describing the data point under the
    # cursor (q-mode vs pixel-mode), or None when the pointer leaves
    # the viewport. Consumers (status bar) format the dict for display.
    cursorMoved = Signal(object)
    manualPeakAdded = Signal(int, object)     # frame, ManualPeak
    manualPeakRemoved = Signal(int, object)   # frame, ManualPeak
    selectionChanged = Signal(object)         # SelectedPeak | None
    peakGeometryChanged = Signal(object)      # SelectedPeak whose r/dr/a/da changed
    # Drag-end variant of ``peakGeometryChanged``. Fires once when the
    # user releases the ROI handle (or the user-side undo/redo path
    # settles a geometry mutation). Consumers that want to react only
    # to the final position — not every drag tick — subscribe here.
    # MainWindow's 2D-preview refresh listens to this one because each
    # pygidfit call is ~100-500 ms and per-tick refits would freeze
    # the drag.
    peakGeometryDragFinished = Signal(object)  # SelectedPeak (post-drag)
    # Emitted on drag-end for non-manual peaks: (frame, kind, peak_id,
    # polar_kwargs). MainWindow drives the actual h5py mutation since it
    # owns the silx tree handle that needs releasing first.
    peakRowWriteRequested = Signal(int, str, int, dict)
    # Emitted when the user presses Delete on a single non-manual peak.
    deletePeakRequested = Signal(object)      # SelectedPeak
    # Emitted when Delete is pressed with >= 2 detected peaks selected:
    # carries the full multi-selection so the host can bulk-delete them
    # in one grouped, undoable action.
    deletePeaksRequested = Signal(list)       # list[SelectedPeak]
    # Emitted whenever the *current* frame's matched-structure list might be
    # different from what the UI showed last (frame change, fresh load,
    # re-render after pipeline run). Args: (frame, list[MatchedStructure]).
    matchedStructuresChanged = Signal(int, list)
    # Emitted when the user changes (or resets) the pen of the detected
    # or fitted overlay. Carries the kind; the Display dock's swatch
    # button listens so the legend follows the image.
    overlayPenChanged = Signal(str)
    # Emitted when the set of simulated reflections selected for
    # injection changes (click toggle, Select missed, pattern swap,
    # intensity-cutoff pruning). Carries the sorted reflection indices;
    # the Display dock's count label + Add button listen.
    simulationSelectionChanged = Signal(list)
    # Emitted alongside ``selectionChanged`` with the full multi-selection
    # list (primary + extras). Single-peak consumers stay on the legacy
    # ``selectionChanged(SelectedPeak | None)``; multi-aware consumers
    # (copy handler, batch-fit button enable) subscribe here.
    selectionsChanged = Signal(list)          # list[SelectedPeak]
    # (frame, ManualPeak) — quick select wants this box written to the
    # file before it is dropped. Emitted only while the mode is armed,
    # from every path that would otherwise discard a pending box.
    manualPeakCommitRequested = Signal(int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The control strip wraps. It used to be one QHBoxLayout, whose
        # minimum width is the sum of every control in it — that sum was
        # the floor under how narrow the central column could be dragged.
        # A FlowLayout breaks onto a second row instead, so the floor is
        # the widest single cluster. Controls are grouped into clusters
        # that wrap as a unit; see ``mlgidgui.flow_layout``.
        bar_widget = wrapping_bar(self, margins=(8, 4, 8, 4))
        bar = bar_widget.layout()

        view_group = ToolGroup()
        view_group.add(QLabel("View:"))
        # A segmented pair rather than two radios: the choice is one
        # exclusive view mode, and a segmented control shows the two
        # options and the active one in the space two radios plus their
        # labels took. Still a QButtonGroup, still named ``_radio_*``,
        # so ``set_mode_radios_visible`` and the render path are
        # unchanged.
        self._radio_cart = _segment_button("Cartesian", "view-cartesian", "left")
        self._radio_polar = _segment_button("Polar", "view-polar", "right")
        self._radio_polar.setChecked(True)
        self._radio_group = QButtonGroup(self)
        self._radio_group.setExclusive(True)
        self._radio_group.addButton(self._radio_cart)
        self._radio_group.addButton(self._radio_polar)
        self._radio_cart.toggled.connect(self._on_radio_toggled)
        segmented = QHBoxLayout()
        segmented.setContentsMargins(0, 0, 0, 0)
        segmented.setSpacing(0)          # the two halves share one edge
        segmented.addWidget(self._radio_cart)
        segmented.addWidget(self._radio_polar)
        view_group.add_layout(segmented)
        bar.addWidget(view_group)

        cmap_group = ToolGroup()
        cmap_group.add(QLabel("Colormap:"))
        self._cmap_combo = _SwatchCombo(
            QSize(CMAP_ICON_SIZE, CMAP_ICON_SIZE))
        # With an app-level stylesheet installed, QComboBox takes its
        # width from the CSS box model and does not grow for the icon, so
        # the name ends up clipped. Measure the widest name rather than
        # hardcoding a width, so this survives a font change.
        _metrics = self._cmap_combo.fontMetrics()
        self._cmap_combo.setMinimumWidth(
            max(_metrics.horizontalAdvance(n) for n in COLORMAPS)
            + 56                       # chip, dropdown arrow, padding
        )
        for name in COLORMAPS:
            # The ramp itself, not just its name: a colormap is picked by
            # how it ramps. Falls back to a bare name if the chip cannot
            # be built (a null pixmap makes a text-only row, not a crash).
            swatch = colormap_swatch(name, CMAP_SWATCH)
            if swatch.isNull():
                self._cmap_combo.addItem(name)
            else:
                self._cmap_combo.addItem(QIcon(swatch), name)
        self._cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_group.add(self._cmap_combo)
        bar.addWidget(cmap_group)

        # Log/linear contrast toggle. When checked, the displayed image
        # is log10(clip(data, floor, inf)) and the histogram levels are
        # recomputed on the transformed array so the LUT stays sensible.
        # Coordinates and overlays are unaffected — only the intensity
        # mapping changes.
        # A toggle button rather than a checkbox: it is a mode the image
        # is in, and it reads as on/off at a glance next to the segmented
        # view pair. Same name, same ``toggled`` signal, same persisted
        # setting — ``setChecked`` still drives it.
        self._log_check = QToolButton()
        self._log_check.setText("Log scale")
        self._log_check.setCheckable(True)
        self._log_check.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._log_check.setProperty("variant", "toggle")
        icons.bind(self._log_check, "log-scale")
        self._log_check.setChecked(False)
        self._log_check.setToolTip(
            "Display log10(intensity) instead of linear intensity. "
            "Useful for GIWAXS data with wide dynamic range; coordinate "
            "axes and overlays are unchanged."
        )
        self._log_check.toggled.connect(self._on_log_toggled)
        bar.addWidget(self._log_check)

        # Aspect ratio of the shown image. "Fit" (default) stretches the
        # frame to fill the panel (setAspectLocked(False)); "Custom" locks
        # the y-to-x data-unit scale to the spin value — 1.00 is the
        # undistorted view (equal Å⁻¹ per axis, so q-space rings stay
        # round in Cartesian mode). The lock lives on the PlotItem and so
        # survives mode switches and re-renders. See _on_aspect_changed.
        aspect_group = ToolGroup()
        aspect_group.add(QLabel("Aspect:"))
        self._aspect_combo = QComboBox()
        self._aspect_combo.addItems(["Fit", "Default", "Custom"])
        self._aspect_combo.setToolTip(
            "Fit fills the panel; Default uses a per-mode shape "
            "(Cartesian 1:1, polar 2:1); Custom locks the width:height to "
            "the ratio box. Scrolling over an axis switches to Custom."
        )
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        aspect_group.add(self._aspect_combo)
        self._aspect_spin = QDoubleSpinBox()
        self._aspect_spin.setRange(0.05, 20.0)
        self._aspect_spin.setSingleStep(0.05)
        self._aspect_spin.setDecimals(2)
        self._aspect_spin.setValue(1.0)
        self._aspect_spin.setToolTip(
            "Custom on-screen width:height of the image: >1 wider than "
            "tall, <1 taller than wide."
        )
        self._aspect_spin.valueChanged.connect(self._on_aspect_changed)
        aspect_group.add(self._aspect_spin)
        bar.addWidget(aspect_group)

        # The frame-navigation controls (prev / play / next / slider /
        # label) used to live in the Display dock's Frame row. They
        # now sit here in the toolbar so the user can scrub frames
        # regardless of which right-dock tab is in front. The host
        # injects them via ``insert_frame_controls`` after the
        # Display dock builds them — see MainWindow.
        self._frames_group: ToolGroup | None = None
        # Keep a handle on the layout so the host can splice the
        # frame-navigation widgets in.
        self._toolbar_layout = bar
        # Kept so ``refresh_toolbar_layout`` can nudge the strip when a
        # whole cluster appears or disappears.
        self._toolbar_widget = bar_widget
        self._view_group = view_group
        outer.addWidget(bar_widget)

        self._plot = pg.PlotItem()
        self._view = pg.ImageView(self, view=self._plot)
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        # Hide pyqtgraph's bottom timeline strip — redundant with the
        # Display-dock frame slider. ``setImage`` re-shows it via
        # ``roiClicked`` for any multi-frame stack, so we re-apply our
        # toggle state in ``_apply_params`` after every render.
        self._view.ui.roiPlot.hide()
        # Set the splitter handle width to 0 so even when the strip is
        # hidden there's no grey separator line eating into the image's
        # x-axis label area.
        self._view.ui.splitter.setHandleWidth(0)
        self._view.ui.splitter.setSizes([1, 0])
        # Pyqtgraph's GraphicsView occasionally lets the scene scroll
        # by a few pixels when its sceneRect has drifted from the
        # viewport size — kill scrollbars + frame so the plot is
        # unconditionally pinned inside its tab.
        gv = self._view.ui.graphicsView
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setFrameStyle(QFrame.Shape.NoFrame)
        # Block QAbstractScrollArea-level scrolling without touching
        # ViewBox pan / zoom — see _disable_viewport_scroll docstring.
        _disable_viewport_scroll(gv)
        outer.addWidget(self._view)
        # Remember the user's contrast when they finish dragging the
        # histogram region, so it survives operation re-renders. Connect
        # to the LUT *item*; the widget forwards attribute access but not
        # the Qt signal object itself.
        self._view.getHistogramWidget().item.sigLevelChangeFinished.connect(
            self._on_contrast_changed
        )

        self._plot.invertY(False)
        self._plot.setAspectLocked(False)
        # PyQtGraph occasionally underestimates the bottom axis cell so
        # the axis label ("radius" / "q_xy") gets clipped by the
        # viewport's lower edge — and that clip is what creates the
        # small scrollable region. A small bottom layout margin gives
        # the label guaranteed clearance and keeps the plot fitted
        # inside its tab.
        self._plot.layout.setContentsMargins(0, 0, 0, 12)
        # Restore the persisted aspect choice now that both the toolbar
        # widgets and the plot exist. Starts in Fit, leaving the
        # setAspectLocked(False) baseline above untouched.
        self._restore_aspect_from_settings()
        # Route single-axis (axis-region) wheel events through our handler
        # so scrolling one axis adjusts the Custom ratio. Must come after
        # the viewbox exists.
        self._install_axis_wheel_handler()

        # Hand-set pens for the detected / fitted / manual overlays: one
        # partial pen per kind, layered over ``OVERLAY_STYLE``. Loaded
        # BEFORE the items below, which are built from the effective
        # pens so a customised overlay is right from the first render
        # rather than flicking to it after the first change.
        self._overlay_pen_overrides: dict[str, dict] = (
            load_overlay_pen_overrides()
        )
        self._detected = _PeakShapeItem(**self.overlay_pen("detected"))
        self._fitted = _PeakShapeItem(**self.overlay_pen("fitted"))
        self._manual = _PeakShapeItem(**self.overlay_pen("manual"))
        self._selection = _PeakShapeItem(**SELECTION_STYLE)
        # The area pygidfit really fits around the selected detected
        # peak(s): a dashed red outline, drawn by ``_render_overlays``
        # when the "show fit area" setting is on.
        self._fit_area = _PeakShapeItem(**FIT_AREA_STYLE)
        # Pre-selection preview: outlines the box a bare click would
        # take. Added before the selection item so a box that is both
        # hovered and selected still shows the solid selection pen.
        self._hover = _PeakShapeItem(**hover_style())
        self._hover.setOpacity(HOVER_OPACITY)
        # Explicit z, because the matched overlay items are rebuilt and
        # re-added to the viewbox on *every* render, so they land above
        # anything added once at construction. Without this the white
        # highlight around a selected matched peak was painted over by
        # the structure's own box, leaving only the sliver where the
        # wider pen stuck out — while fitted and detected (added here,
        # before the selection) looked correct.
        self._hover.setZValue(40)
        self._selection.setZValue(45)
        self._fit_area.setZValue(44)
        self._fitted_preview = _PeakShapeItem(**FITTED_PREVIEW_STYLE)
        self._fitted_preview.setOpacity(FITTED_PREVIEW_OPACITY)
        # Above the selection, which is where insertion order used to
        # put it: the preview is a *different* box (where the fit would
        # land), not a highlight, and hiding it under the outline would
        # lose information on the frames where the two nearly coincide.
        self._fitted_preview.setZValue(46)
        vb = self._plot.getViewBox()
        vb.addItem(self._detected, ignoreBounds=True)
        vb.addItem(self._fitted, ignoreBounds=True)
        vb.addItem(self._manual, ignoreBounds=True)
        vb.addItem(self._hover, ignoreBounds=True)
        vb.addItem(self._fit_area, ignoreBounds=True)
        vb.addItem(self._selection, ignoreBounds=True)
        vb.addItem(self._fitted_preview, ignoreBounds=True)

        # One item per user-registered extra peak list, rebuilt by
        # ``set_peak_list_specs`` whenever the registry changes. Kept in
        # a dict rather than as named attributes because the set is
        # decided at runtime by Settings, not at construction. Filled at
        # the END of __init__, once the state it renders against exists.
        self._list_items: dict[str, _PeakShapeItem] = {}
        self._peak_list_specs: list = []

        self._preview_item = pg.QtWidgets.QGraphicsRectItem()
        preview_pen = pg.mkPen(QColor("#ffeb3b"), width=1.0)
        preview_pen.setStyle(Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._preview_item.setPen(preview_pen)
        self._preview_item.setBrush(QColor(255, 235, 59, 40))
        self._preview_item.setZValue(50)
        self._preview_item.setVisible(False)
        vb.addItem(self._preview_item, ignoreBounds=True)

        # Mouse handling lives in a Qt event filter on the graphics-view's
        # viewport — see _LabelEventFilter for why a ViewBox subclass doesn't work.
        self._label_filter = _LabelEventFilter(self._view.ui.graphicsView, vb, self)
        self._label_filter.install()
        self._label_filter.drawStarted.connect(self._on_draw_started)
        self._label_filter.drawUpdated.connect(self._on_draw_updated)
        self._label_filter.drawFinished.connect(self._on_draw_finished)
        self._label_filter.selectAt.connect(self._on_select_at)
        self._label_filter.doubleClicked.connect(self._reset_view_to_default)
        # A double-click delivers a plain click first, which may have
        # stepped the selection through a stack of boxes; take that
        # step back now that the gesture is known.
        self._label_filter.doubleClicked.connect(
            self.revert_cycle_for_double_click)
        self._label_filter.cursorPos.connect(self._on_cursor_pos)
        self._label_filter.cursorLeft.connect(self._on_cursor_left)

        # Apply default colormap immediately.
        self._apply_cmap(DEFAULT_COLORMAP)

        # Need keyboard focus for the Delete shortcut to fire.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._frame_peaks: dict[int, dict[str, PeakTable | None]] = {}
        self._manual_peaks: dict[int, list[ManualPeak]] = {}
        # Per-kind overlay visibility. Registered extra lists are added
        # to this by ``set_peak_list_specs`` and default to visible, so a
        # list the user just registered shows up without a second click.
        self._visibility: dict[str, bool] = {kind: True for kind in OVERLAY_KINDS}

        # Matched-structure overlays. Variable count per frame (one per row in
        # each matched_* dataset), each with its own color and visibility.
        # Items in ``_matched_items`` belong to the *currently rendered* frame
        # only — they're torn down and rebuilt on frame change.
        self._matched_per_frame: dict[int, list[MatchedStructure]] = {}
        # Show/hide state keyed by a structure's CROSS-FRAME identity
        # (``MatchedStructure.color_key`` = CIF + hkl), so unchecking a
        # structure on one frame hides it on every frame it appears.
        # Defaults to True on first sight.
        self._matched_visibility: dict[tuple, bool] = {}
        # Stable colour index per structure identity, assigned first-seen
        # so a given (CIF, hkl) keeps its colour across the whole scan
        # (previously coloured by per-frame position, which shuffled as
        # the structure set changed frame to frame).
        self._matched_color_index: dict[tuple, int] = {}
        # User-set pens per identity, layered over the palette --
        # partial dicts, so overriding only the width leaves the colour
        # cycling. Loaded once here; ``set_matched_pen`` keeps QSettings
        # in sync. Survives ``clear()``: a preference, not file state.
        self._matched_pen_overrides: dict[tuple, dict] = (
            _load_matched_pen_overrides()
        )
        self._matched_master_visible: bool = True
        # How matched structures are drawn: "boxes" (default, per-peak
        # boxes) or "markers" (hollow circles for peaks + dashed arcs for
        # rings, like the q-map). A display preference, not reset on
        # file close.
        self._matched_display_style: str = "boxes"
        # ``unique_id``s currently hidden by the Display-dock search
        # filter. Independent of ``_matched_visibility`` (which
        # tracks checkbox state); the filter set overrides checkbox
        # state but doesn't mutate it, so clearing the filter
        # restores the previous user selection.
        self._matched_filter_hidden: set[str] = set()
        # Minimum detection score to render on the Detected overlay.
        # Driven by the Display dock's Min-score slider. 0.0 (the
        # default) lets every detected peak through; raising it
        # hides weak detections without mutating the file.
        self._detected_score_cutoff: float = 0.0
        # "Expected pattern" simulation overlay state. ``_sim_pattern``
        # is a simulation_pattern.SimulatedPattern (or None); its items
        # are torn down + rebuilt per render exactly like the matched
        # overlays. ``_sim_selected`` holds reflection indices chosen
        # for injection (indices are pattern-scoped, so any pattern
        # swap clears the selection). ``_sim_visible`` and
        # ``_sim_min_intensity`` mirror the Display-dock controls and
        # deliberately survive ``clear()`` (display preferences, not
        # file state).
        self._sim_pattern = None
        self._sim_selected: set[int] = set()
        self._sim_min_intensity: float = 0.01
        self._sim_visible: bool = True
        self._sim_items: list = []
        # When not None, ONLY these fitted rows render, per frame
        # (``{frame: {fitted_id, ...}}``; frames absent from the mapping
        # show no fitted rows at all). Driven by the Scan-tracking
        # dock's "Show only tracked peaks" toggle, so the user can
        # verify track consistency by scrubbing. None = filter off.
        self._fitted_visible_only: dict[int, set[int]] | None = None
        self._matched_items: list[tuple[str, _PeakShapeItem]] = []
        # Identities whose tracked-peak subset is empty on the rendered
        # frame (kept hidden against later visibility refreshes).
        self._matched_empty_uids: set = set()

        self._mode = MODE_POLAR
        self._log_scale: bool = False
        # Raw-preview orientation flips, driven by the Conversion panel's
        # fliplr/flipud/transpose checkboxes. Default False (show the frame
        # exactly as stored); when set, the preview is reoriented to match
        # what pygid's conversion will produce. Only the raw branch of
        # _render_frame reads these; the converted display is unaffected (its
        # orientation is already baked in by pygid). See set_raw_flips and
        # _apply_raw_flips.
        self._raw_flip_lr: bool = False
        self._raw_flip_ud: bool = False
        self._raw_transp: bool = False
        # The contrast (histogram) levels the user has dialed in with the
        # slider, or None to auto-contrast from the data. When set, it is
        # reused across operation re-renders (add-peak, pipeline reattach,
        # frame scrubs) so editing or running the pipeline doesn't snap the
        # contrast back to the robust default. Reset to None whenever the
        # underlying data changes (new stack/entry, log/linear toggle) so
        # those genuinely re-auto-contrast. Captured from the histogram's
        # sigLevelChangeFinished signal; ``setImage(levels=...)`` does not
        # fire that signal, so renders never masquerade as a user change.
        self._user_levels: tuple[float, float] | None = None
        # Guard: True while we push an image via ``setImage``, so the
        # histogram's ``sigLevelChangeFinished`` (which can fire when the
        # data range shifts, e.g. log toggle / new frame) is ignored and
        # only genuine user drags update ``_user_levels``.
        self._suppress_level_capture: bool = False
        self._stack: EntryStack | None = None
        # The FrameSource backing the active EntryStack. Owns the live
        # h5py handle + per-frame LRU; released across the silx detach
        # / reattach dance via release_frame_source / acquire_frame_source.
        self._frame_source: "FrameSource | None" = None  # type: ignore[name-defined]
        # The viewer drives frame indexing itself (the Display-dock slider
        # is the single source of truth). pyqtgraph used to track this via
        # ImageView.currentIndex, but we stopped feeding it 3D stacks once
        # large-file support landed — see the lazy-loading milestone.
        self._frame_index: int = 0
        # Last applied _DisplayParams so per-frame scrubs reuse pos/scale/
        # levels without rebuilding them.
        self._display_params: _DisplayParams | None = None
        # Raw-mode preview state. Held separately from ``_stack`` because raw
        # detector frames have no q-axes — they're rendered in pixel
        # coordinates and carry no overlays. Either an ndarray (small
        # stacks, tests) or a lazily-read file_model.LazyRawStack.
        self._raw_image_stack: "np.ndarray | object | None" = None
        # Polar grid axes derived from the FrameSource. Stored as a
        # (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) get a single-frame index
        # without precomputing the full polar stack.
        self._polar_cache: "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None" = None  # type: ignore[name-defined]
        self._next_manual_id = -1  # negative IDs distinguish manual from detected
        # Quick-select labelling: while True, a pending manual box is
        # committed to the file instead of being dropped. Armed by the
        # host from the Parameter panel's checkbox; the viewer only ever
        # *asks* for the commit (``manualPeakCommitRequested``) — every
        # file write stays on the host side.
        self._quick_select = False
        self._selected: SelectedPeak | None = None
        # Secondary selections for the multi-select (Ctrl+click / Ctrl+A)
        # workflow. Restricted to ``kind == "detected"`` in this iteration
        # (the only kind in scope for copy/paste + batch fit). The primary
        # ``_selected`` can be any kind; extras are only ever appended via
        # ``_toggle_selected`` which gates on detected.
        self._selected_extras: list[SelectedPeak] = []
        self._roi_item: pg.ROI | None = None
        # Stacks of `_Action` objects. Pushing to undo clears redo. ROI drags
        # populate _roi_drag_before on sigRegionChangeStarted and consume it
        # on sigRegionChangeFinished — partial state never lands on the stack.
        self._undo_stack: list[_Action] = []
        self._redo_stack: list[_Action] = []
        self._roi_drag_before: tuple[float, float, float, float] | None = None
        # Set during a pipeline run so we don't allow concurrent ROI edits or
        # Delete keypresses while mlgidbase has the file open for writes.
        self._busy: bool = False

        # Pre-selection preview state. ``_hover_pos`` is the last
        # cursor point in data coordinates (so a re-render can redraw
        # the outline without waiting for the next mouse move) and
        # ``_hover_key`` identifies what is currently outlined, so a
        # move inside the same box costs one hit test and no repaint.
        self._hover_pos: tuple[float, float] | None = None
        self._hover_key: tuple | None = None
        # Click-through stepping: a click never hands back the box that
        # is already selected, it takes the next one under the cursor.
        # ``_cycle_prev`` is the selection to restore if the click that
        # just stepped turns out to have been the first half of a
        # double-click (which resets the zoom), and ``_cycle_time`` is
        # when that step happened.
        self._cycle_prev: SelectedPeak | None = None
        self._cycle_time: float = 0.0

        # Geometry of the fitted-preview box for the current selection
        # (radius_center, fwhm_radial, angle_center, fwhm_angular). Cleared
        # whenever the selection isn't a manual / detected peak with valid
        # 1D fits.
        self._fitted_preview_geom: tuple[float, float, float, float] | None = None
        # When True, the preview is rendered as a ring (full angular sweep
        # at angle = 45°, angle_width = ∞) regardless of the angular fit —
        # mirrors what Add-to-fitted will write when the "Save fitted as
        # ring" toggle is on.
        self._fitted_preview_is_ring: bool = False
        # Cyan dashed previews for the Ctrl+click extras. Same shape
        # as ``_fitted_preview_geom`` but a list — one entry per extra
        # whose pygidfit result the host has computed. Always
        # segments (extras are detected, no ring sentinel).
        self._fitted_preview_extras_geoms: list[
            tuple[float, float, float, float]
        ] = []

        # NB: pyqtgraph's sigTimeChanged is no longer connected — we
        # stopped feeding ImageView 3D stacks once lazy frame loading
        # landed (see the lazy-loading milestone in
        # Documentation/). Frame changes now flow exclusively through
        # MainWindow → viewer.set_frame, and ``frameChanged`` is emitted
        # from there.

        # Add a "Reset zoom" action to the viewbox right-click menu so the
        # user can undo a manual zoom without leaving the keyboard / mouse.
        # pyqtgraph's default "View All" does the same thing under a less
        # discoverable label; this just adds the explicitly-named entry.
        self._install_reset_zoom_action()

        # Last, because it builds overlay items and renders: everything
        # it touches (_mode, _visibility, _frame_peaks) has to exist.
        self.set_peak_list_specs(peak_lists.load_specs())

    # -- Public API --

    def show_stack(self, stack: EntryStack, *, preserve_view: bool = False) -> None:
        """Render ``stack`` in the active mode.

        ``preserve_view`` keeps the current viewbox range and frame
        index across the re-render — used after pipeline ops and
        direct h5py edits where the underlying stack is identical and
        only the peak overlays changed. Default ``False`` (autorange)
        for the entry-switch / file-open paths, where the new stack
        typically has different axes.

        ``stack.image_stack`` must be a ``_LazyImageStack`` (i.e. the
        return value of ``file_model.load_entry``). The viewer extracts
        its backing ``FrameSource`` and stashes it on
        ``self._frame_source`` so frame reads can stream from disk
        without the full stack ever entering RAM.
        """
        # Capture before resetting cached state so the saved range is the
        # one the user is actually looking at right now.
        saved_xrange: tuple[float, float] | None = None
        saved_yrange: tuple[float, float] | None = None
        saved_frame: int | None = None
        if preserve_view and self._stack is not None:
            try:
                xr, yr = self._plot.getViewBox().viewRange()
                saved_xrange = (float(xr[0]), float(xr[1]))
                saved_yrange = (float(yr[0]), float(yr[1]))
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer.show_stack", exc_info=True)
                pass
            saved_frame = self.current_frame

        # Release any prior FrameSource before adopting the new one —
        # an open h5py handle on the previous temp file would block its
        # cleanup during session close on Windows.
        if (
            self._frame_source is not None
            and not isinstance(stack.image_stack, _LazyImageStack)
        ):
            self._frame_source.release()
            self._frame_source = None
        elif (
            self._frame_source is not None
            and isinstance(stack.image_stack, _LazyImageStack)
            and stack.image_stack.source is not self._frame_source
        ):
            self._frame_source.release()
            self._frame_source = None

        self._stack = stack
        if isinstance(stack.image_stack, _LazyImageStack):
            self._frame_source = stack.image_stack.source
        else:
            # Defensive fallback for tests that hand-build an EntryStack
            # with a raw ndarray. The viewer's hot path always goes
            # through _frame_source, so reject this case loudly.
            raise TypeError(
                "show_stack expects an EntryStack whose image_stack is "
                "a _LazyImageStack (i.e. produced by load_entry). Got: "
                f"{type(stack.image_stack).__name__}"
            )
        # ``_raw_image_stack`` belongs to a prior RawSession; clearing it
        # here ensures _render_active_mode never tries to re-render raw
        # pixel data over a NeXus stack.
        self._drop_raw_stack()
        # If the previous session was raw, the mode flag is still
        # ``MODE_RAW`` even though raw rendering ignored the radios.
        # Snap back to whichever Cartesian / Polar radio is checked
        # (Polar is the default at startup) so ``_render_active_mode``
        # takes the converted-data branch.
        if self._mode == MODE_RAW:
            self._mode = (
                MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
            )
        self._polar_cache = None
        self._frame_peaks.clear()
        # A fresh stack (entry change, file open) should re-auto-contrast;
        # a preserve_view re-render (peak edits, pipeline reattach) keeps
        # the user's slider contrast.
        if not preserve_view:
            self._user_levels = None
        # Clamp frame index to the new stack's range. Preserved when
        # the caller asked for it and the prior frame is still valid;
        # otherwise default to 0 so a fresh entry starts at the first
        # frame.
        if preserve_view and saved_frame is not None:
            self._frame_index = max(
                0, min(int(saved_frame), self._stack.n_frames - 1)
            )
        else:
            self._frame_index = 0
        # Skip the setImage autoRange entirely when we're preserving
        # the user's zoom — otherwise the viewbox flashes to the
        # full extent before the post-render setRange snaps it back,
        # which is visually jarring (and breaks if Qt schedules the
        # autoRange via a deferred signal that fires after setRange).
        # The post-render setRange below is the belt to this braces.
        self._render_active_mode(auto_range=not preserve_view)
        if preserve_view and saved_xrange is not None and saved_yrange is not None:
            self._plot.getViewBox().setRange(
                xRange=saved_xrange, yRange=saved_yrange, padding=0
            )

    def set_mode_radios_visible(self, visible: bool) -> None:
        """Show / hide the Cartesian / Polar radios in the top toolbar.

        Used by the host to remove mode controls when a raw session is
        active — raw frames don't carry q-axes, so the toggles would be
        nonsensical. The toolbar's "Colormap" + "Timeline" widgets stay
        visible because they apply equally to raw and converted data.
        """
        # Find the leading "View:" label by walking up from the radio's
        # parent layout — the label was added directly before the radios.
        for w in (self._radio_cart, self._radio_polar):
            w.setVisible(visible)
        # Hide the "View:" prefix label too. It lives in the same toolbar
        # row built by __init__; locate it by text rather than caching a
        # reference at construction time so existing layout code stays put.
        for label in self.findChildren(QLabel):
            if label.text() == "View:":
                label.setVisible(visible)
                break
        self.refresh_toolbar_layout()

    def show_raw_stack(self, arr_3d) -> None:
        """Render a raw detector stack in pixel coordinates.

        Used only for raw-mode (pre-conversion) preview. Wipes any prior
        NeXus-mode state — overlays, peaks, undo history — because none
        of it applies to a raw detector frame. The viewer's frame slider
        and timeline still drive frame navigation across the stack.

        Accepts an ndarray (small stacks, tests) or a
        ``file_model.LazyRawStack`` (the GUI's path — frames are read on
        demand, so a tens-of-GB beamtime stack never gets materialized;
        only ndarrays are copied contiguous here, a lazy stack would be
        materialized by ``ascontiguousarray``).
        """
        if getattr(arr_3d, "ndim", None) != 3:
            raise ValueError(
                f"show_raw_stack expects a 3D (N, H, W) array, "
                f"got shape {getattr(arr_3d, 'shape', None)}"
            )
        # Drop NeXus-mode state (peaks / matched / undo / cached polar).
        # ``clear()`` already covers everything except the raw-stack field
        # itself (and releases any prior lazy raw stack's handle).
        self.clear()
        self._mode = MODE_RAW
        if isinstance(arr_3d, np.ndarray):
            arr_3d = np.ascontiguousarray(arr_3d)
        self._raw_image_stack = arr_3d
        self._render_active_mode()

    def set_raw_flips(
        self, flip_lr: bool, flip_ud: bool, transp: bool = False,
    ) -> None:
        """Set the raw-preview orientation and re-render.

        Driven by the Conversion panel's Orientation row -- fliplr,
        flipud and transpose -- so the preview shows the raw image in the
        orientation the conversion will produce (see
        ``_apply_raw_flips``). The flags persist, so a raw stack loaded
        later is rendered the same way. Re-renders with
        ``auto_range=False`` so the user's zoom/pan survives the toggle.
        """
        flip_lr, flip_ud = bool(flip_lr), bool(flip_ud)
        transp = bool(transp)
        current = (self._raw_flip_lr, self._raw_flip_ud, self._raw_transp)
        if (flip_lr, flip_ud, transp) == current:
            return
        # A transpose swaps the frame's width and height, so the plot
        # extent changes shape and the previous zoom rectangle no longer
        # frames anything sensible. Refit in that case only.
        refit = transp != self._raw_transp
        self._raw_flip_lr, self._raw_flip_ud = flip_lr, flip_ud
        self._raw_transp = transp
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            self._render_active_mode(auto_range=refit)

    def _drop_raw_stack(self) -> None:
        """Forget the raw stack, closing a lazy stack's h5py handle.

        ndarray stacks have nothing to release; ``LazyRawStack`` holds a
        read handle on the raw file that must close before the session
        is dropped (Windows can't delete an open file; on any OS a
        leaked handle pins the LRU frames in memory).
        """
        stack = self._raw_image_stack
        self._raw_image_stack = None
        release = getattr(stack, "release", None)
        if release is not None:
            try:
                release()
            except Exception:
                logger.debug("suppressed release in _drop_raw_stack", exc_info=True)

    def reset_zoom(self) -> None:
        """Auto-fit the viewbox to the current image."""
        try:
            self._plot.getViewBox().autoRange()
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer.reset_zoom", exc_info=True)
            pass

    def apply_theme_colors(self, background, foreground) -> None:
        """Recolour the plot background + axes live for a theme switch.

        pyqtgraph bakes the configured background/foreground into items at
        creation (``pg.setConfigOption``), so an already-built plot keeps
        the old theme's colours until told otherwise — this pushes them
        immediately so a dark→light switch doesn't leave white axis text
        on a white background.
        """
        try:
            self._view.ui.graphicsView.setBackground(background)
        except Exception:
            logger.debug("suppressed exception setting viewer background", exc_info=True)
        pen = pg.mkPen(foreground)
        for name in ("left", "bottom", "right", "top"):
            try:
                ax = self._plot.getAxis(name)
            except Exception:
                ax = None
            if ax is None:
                continue
            try:
                ax.setPen(pen)
                ax.setTextPen(pen)
            except Exception:
                logger.debug("suppressed exception recolouring axis", exc_info=True)
        # The contrast/LUT histogram is its own GraphicsView (the
        # "contrast slider"), so it keeps the construction-time background
        # until told otherwise — recolour it and its level axis too.
        try:
            hist = self._view.getHistogramWidget()
            hist.setBackground(background)
            hist_axis = getattr(hist.item, "axis", None)
            if hist_axis is not None:
                hist_axis.setPen(pen)
                hist_axis.setTextPen(pen)
        except Exception:
            logger.debug("suppressed exception recolouring histogram", exc_info=True)
        # The selection highlight is drawn white on the dark plot ground
        # and near-black on the light one; the overlay item was built at
        # construction time, so re-pen it here. Overlay/matched colours
        # are data, not chrome, and deliberately stay put.
        try:
            self._selection.set_pen_color(selection_style()["color"])
            self._fit_area.set_pen_color(fit_area_style()["color"])
            self._hover.set_pen_color(hover_style()["color"])
        except Exception:
            logger.debug("suppressed exception recolouring selection", exc_info=True)
        # Re-render so the simulation overlay picks up its theme-visible
        # "selected" colour on the next paint.
        try:
            self._render_simulation_overlays(self.current_frame)
        except Exception:
            logger.debug("suppressed exception refreshing sim overlay", exc_info=True)

    # -- Aspect ratio (toolbar "Aspect:" Fit / Default / Custom) --

    _ASPECT_RATIO_MIN = 0.05
    _ASPECT_RATIO_MAX = 20.0
    # Per-mode width:height used by the "Default" preset.
    _ASPECT_DEFAULT_POLAR = 2.0
    _ASPECT_DEFAULT_OTHER = 1.0   # Cartesian + raw

    def aspect(self) -> tuple[str, float]:
        """Return the current aspect choice as ``(mode, ratio)``.

        ``mode`` is ``"fit"``, ``"default"`` or ``"custom"``; ``ratio`` is
        the target on-screen **width:height** of the image (``2`` = twice
        as wide as tall, the natural shape of a polar radius×angle map).
        For Default the ratio reported is the per-mode preset value.
        """
        txt = self._aspect_combo.currentText()
        mode = {"Default": "default", "Custom": "custom"}.get(txt, "fit")
        if mode == "default":
            return mode, self._default_ratio_for_mode()
        return mode, float(self._aspect_spin.value())

    def _default_ratio_for_mode(self) -> float:
        """Width:height the "Default" preset uses for the active mode:
        2:1 for polar (radius×angle), 1:1 for Cartesian / raw. Uses a
        getattr fallback because the aspect control is restored in
        ``__init__`` before ``_mode`` is assigned (the viewer starts
        polar)."""
        if getattr(self, "_mode", MODE_POLAR) == MODE_POLAR:
            return self._ASPECT_DEFAULT_POLAR
        return self._ASPECT_DEFAULT_OTHER

    def set_aspect(self, mode: str, ratio: float | None = None) -> None:
        """Programmatically set the aspect choice (tests / future wiring).
        Seeds the toolbar widgets without retriggering their signals, then
        applies + persists once via the shared path."""
        mode = str(mode).lower()
        label = {"default": "Default", "custom": "Custom"}.get(mode, "Fit")
        with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
            self._aspect_combo.setCurrentText(label)
            if ratio is not None:
                self._aspect_spin.setValue(float(ratio))
        self._on_aspect_changed()

    def _data_extent(self) -> tuple[float, float] | None:
        """Width/height of the current image in *data* units (x, y), from
        the cached ``_DisplayParams``. None when no image is shown yet.

        This is what makes the ratio pixel-based rather than q-based: the
        lock value handed to pyqtgraph is derived from the image's actual
        extent, so a polar map (radius ~0-3 Å⁻¹ vs angle ~0-360°) is not
        collapsed into a sliver — see ``_apply_aspect``.
        """
        p = self._display_params
        if p is None:
            return None
        img = getattr(p, "image_pg", None)
        if img is None or getattr(img, "ndim", 0) < 2:
            return None
        nx, ny = img.shape[0], img.shape[1]   # pyqtgraph col-major: (x, y)
        sx, sy = p.scale
        dx = abs(nx * sx)
        dy = abs(ny * sy)
        if dx <= 0 or dy <= 0:
            return None
        return dx, dy

    def _live_box_ratio(self) -> float | None:
        """The image's *current* on-screen width:height, whatever the
        view state. ViewBox.getAspectRatio() returns
        ``a = (px per x-unit)/(px per y-unit)``; the image box is
        ``(Dx/Dy) * a``. Used to seed Custom when the user starts
        scrolling an axis so the number continues from what is shown."""
        ext = self._data_extent()
        if ext is None:
            return None
        try:
            a = float(self._plot.getViewBox().getAspectRatio())
        except Exception:
            return None
        if a <= 0:
            return None
        return (ext[0] / ext[1]) * a

    def _lock_to_ratio(self, ratio: float) -> None:
        """Lock the plot so the image box width:height == ``ratio``.

        pyqtgraph's ``setAspectLocked(True, a)`` locks ``a = (px per
        x-unit)/(px per y-unit)`` (see ViewBox.updateViewRange: "aspect is
        (widget w/h) / (view range w/h)"); the box is ``(Dx/Dy) * a``, so
        ``a = ratio * Dy / Dx`` from the current extent. Re-derived per
        call because Dx/Dy differ between Cartesian and polar, so the same
        ratio means the same on-screen shape in both. (The reciprocal
        collapsed polar — where Dy >> Dx — to a sliver.)
        """
        ext = self._data_extent()
        if ext is not None and ratio > 0:
            dx, dy = ext
            self._plot.setAspectLocked(True, ratio * dy / dx)
        else:
            # No image yet — defer; re-applied after the first render.
            self._plot.setAspectLocked(False)

    def _apply_aspect(self, refit: bool = False) -> None:
        """Push the current toolbar choice onto the plot. Fit frees the
        lock; Default and Custom lock to a width:height ratio (Default's is
        per-mode, reflected read-only in the spin)."""
        mode, ratio = self.aspect()
        try:
            if mode == "fit":
                self._plot.setAspectLocked(False)
            else:
                if mode == "default":
                    # Mirror the per-mode preset into the (disabled) spin.
                    with QSignalBlocker(self._aspect_spin):
                        self._aspect_spin.setValue(ratio)
                self._lock_to_ratio(ratio)
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._apply_aspect", exc_info=True)
        if refit:
            self.reset_zoom()

    def _on_aspect_changed(self, *_: object) -> None:
        """Toolbar slot: enable the ratio spin only in Custom mode, apply
        the choice (refitting so the whole image is shown at the new
        shape), and persist the Custom ratio."""
        mode, ratio = self.aspect()
        self._aspect_spin.setEnabled(mode == "custom")
        self._apply_aspect(refit=True)
        if mode == "custom":
            QSettings().setValue(self._ASPECT_RATIO_KEY, ratio)

    def _reset_view_to_default(self) -> None:
        """Bare LMB double-click handler: snap the aspect to the per-mode
        **Default** preset. ``set_aspect`` refits, so this also restores the
        zoom to the full extent (replacing the old plain reset-zoom)."""
        self.set_aspect("default")

    def _restore_aspect_from_settings(self) -> None:
        """Seed the toolbar widgets in ``__init__``. The mode starts at
        **Default** (the per-mode preset); only the last Custom ratio is
        remembered so re-entering Custom is sensible."""
        s = QSettings()
        try:
            ratio = float(s.value(self._ASPECT_RATIO_KEY, 1.0))
        except (TypeError, ValueError):
            ratio = 1.0
        if not (self._ASPECT_RATIO_MIN <= ratio <= self._ASPECT_RATIO_MAX):
            ratio = 1.0
        with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
            self._aspect_combo.setCurrentText("Default")
            self._aspect_spin.setValue(ratio)
            self._aspect_spin.setEnabled(False)
        self._apply_aspect(refit=False)

    def _install_axis_wheel_handler(self) -> None:
        """Route single-axis (axis-region) wheels through our handler.

        pyqtgraph forwards an axis-region wheel as
        ``viewbox.wheelEvent(ev, axis=0|1)`` (see ``AxisItem.wheelEvent``);
        a centre-of-image wheel reaches the ViewBox through Qt's C++ vtable
        and is untouched here (it zooms both axes, honouring any lock).
        """
        vb = self._plot.getViewBox()
        vb.wheelEvent = self._axis_wheel_event  # bound method; see docstring

    def _axis_wheel_event(self, ev, axis=None) -> None:
        """Scrolling over a single axis adjusts the aspect ratio and
        switches the control to **Custom**, seeded with the live shown
        ratio so the view continues smoothly. x widens (ratio up), y
        heightens (ratio down); the spin tracks it live."""
        vb = self._plot.getViewBox()
        if axis in (0, 1):
            base = self._live_box_ratio()
            if base is None:
                _, base = self.aspect()
            try:
                s = 1.02 ** (ev.delta() * vb.state["wheelScaleFactor"])
            except Exception:
                s = 1.0
            new_ratio = base / s if axis == 0 else base * s
            new_ratio = min(self._ASPECT_RATIO_MAX, max(self._ASPECT_RATIO_MIN, new_ratio))
            with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
                self._aspect_combo.setCurrentText("Custom")
                self._aspect_spin.setValue(new_ratio)
                self._aspect_spin.setEnabled(True)
            self._apply_aspect(refit=False)   # restretch around current view
            QSettings().setValue(self._ASPECT_RATIO_KEY, new_ratio)
            ev.accept()
            return
        # Centre wheel routed here (axis is None): default both-axis zoom.
        pg.ViewBox.wheelEvent(vb, ev, axis)

    def _install_reset_zoom_action(self) -> None:
        vb = self._plot.getViewBox()
        menu = getattr(vb, "menu", None)
        if menu is None:
            return
        action = QAction("Reset zoom", menu)
        action.triggered.connect(self.reset_zoom)
        # Insert at the top so it lands above pyqtgraph's default entries.
        first = menu.actions()[0] if menu.actions() else None
        if first is not None:
            menu.insertAction(first, action)
            menu.insertSeparator(first)
        else:
            menu.addAction(action)

    def set_peaks(self, frame: int, peaks: dict[str, PeakTable | None]) -> None:
        self._frame_peaks[frame] = peaks
        if frame == self.current_frame:
            self._render_overlays(frame)

    def insert_frame_controls(self, widgets: list[QWidget]) -> None:
        """Add the frame-navigation cluster to the end of the toolbar.

        The host (MainWindow) owns these widgets (previous, play, next,
        slider, index, label) so it can keep their signal wiring intact
        when re-parenting them. They go into one :class:`ToolGroup`, so
        the strip wraps the transport as a block rather than splitting
        it mid-cluster, and so the group vanishes entirely when the host
        hides all six for a single-frame stack.

        The slider carries the stretch inside the group and the group is
        horizontally Expanding, which is what makes it swallow whatever
        width its row has left over.
        """
        if self._frames_group is None:
            group = ToolGroup()
            group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Preferred)
            self._frames_group = group
            self._toolbar_layout.addWidget(group)
        for w in widgets:
            # The slider gets the stretch so it, not the buttons either
            # side of it, absorbs the row's spare width.
            self._frames_group.add(w, 100 if isinstance(w, QSlider) else 0)

    def refresh_overlays(self) -> None:
        """Redraw the peak overlays of the current frame, for a setting
        that is read while they are drawn (the fit-area outline)."""
        if self._stack is not None:
            self._render_overlays(self.current_frame)

    def refresh_toolbar_layout(self) -> None:
        """Re-run the control strip's layout after a cluster's visibility
        changed.

        ``ToolGroup`` reports a zero size hint when all its members are
        hidden, and ``FlowLayout``'s own height depends on which clusters
        are in play — but Qt only re-runs a parent layout when a size
        *hint* changes, and this layout's hint is always one row tall.
        Hiding the six transport widgets for a single-frame stack
        therefore left the strip laid out for the widgets it no longer
        had: the controls were painted in their new places while the
        window still hit-tested the old ones, so clicking Cartesian or
        Polar did nothing. Opening a second file appeared to "fix" it
        only because that triggered a relayout.

        Cheap and idempotent, so every visibility toggle calls it rather
        than reasoning about which ones can get away without it.
        """
        for group in (getattr(self, "_frames_group", None),
                      getattr(self, "_view_group", None)):
            if group is None:
                continue
            # A cluster with nothing left in it should not be on screen
            # at all; leaving it visible is what let an empty one cover
            # its neighbours.
            group.setVisible(group.has_visible_members())
            group.updateGeometry()
        layout = getattr(self, "_toolbar_layout", None)
        if layout is not None:
            layout.invalidate()
            layout.activate()
        holder = getattr(self, "_toolbar_widget", None)
        if holder is not None:
            holder.updateGeometry()

    def set_overlay_visible(self, kind: str, visible: bool) -> None:
        if kind not in OVERLAY_KINDS and kind not in self._list_items:
            return
        self._visibility[kind] = visible
        item = self._overlay_item(kind)
        if item is not None:
            item.setVisible(visible)
        # Hiding the overlay that owns the current selection also clears the
        # selection highlight so it doesn't dangle.
        if (
            not visible
            and self._selected is not None
            and self._selected.kind == kind
        ):
            self.clear_selection()

    # -- File-handle coordination with the silx detach/reattach dance --

    def release_frame_source(self) -> None:
        """Close the FrameSource's h5py handle + clear its LRUs.

        Called by MainWindow._detach_silx_tree before any path that
        opens the same temp file ``r+`` (pipeline runs, ROI commit
        write-throughs, Add-to-fitted, clear-peaks, save-as).
        Idempotent — safe to call when no FrameSource is active.

        Also clears ``self._polar_cache`` so the cursor-readout
        handler's ``self._polar_cache is None`` guard fires while
        the FrameSource is closed. Without this drop, the cached
        ``_LazyPolarStack`` would still satisfy that None-check but
        its ``__getitem__`` would call into a released
        ``FrameSource`` and raise ``RuntimeError("FrameSource not
        acquired")`` every time the user moved the mouse across the
        polar plot during a pipeline run.
        """
        if self._frame_source is not None:
            self._frame_source.release()
        self._polar_cache = None

    def acquire_frame_source(self) -> None:
        """Reopen the FrameSource's h5py handle after a write completes.

        Pairs with ``release_frame_source`` via the silx reattach path.
        The polar cache is invalidated because per-frame polar arrays
        depend on the underlying Cartesian frames — and after a pipeline
        run those may have been overwritten. The polar grid axes
        (radius / angle) stay valid since they're functions of q_xy /
        q_z only, which don't change.

        Re-render runs with ``auto_range=False`` so the user's zoom
        survives the detach/reattach dance. This is the only call
        site for ``acquire_frame_source`` and it's always paired
        with ``release_frame_source`` for an op that the user
        triggered while looking at a specific area (pipeline ops,
        Add-to-fitted, clear-peaks…). Resetting the viewbox there
        is precisely the "zoom jumps to full extent after every
        commit" bug.
        """
        if self._frame_source is not None and not self._frame_source.is_open:
            self._frame_source.acquire()
            # Force a refresh of the polar cache wrapper so consumers
            # holding the previous tuple drop their reference.
            self._polar_cache = None
            # Re-render the current frame so the on-screen image
            # reflects any post-write changes to the underlying data.
            # _render_active_mode rebuilds the display params + LUT
            # but skips setImage's autoRange so the zoom survives.
            if self._stack is not None:
                self._render_active_mode(auto_range=False)

