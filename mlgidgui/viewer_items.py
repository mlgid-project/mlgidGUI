"""Viewer support items: peak dataclasses, undo actions, the label
event filter, the peak shape item and the polar/cartesian hit-test
geometry. Moved out of ``image_viewer`` in the 2026 source split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPolygonF
from PySide6.QtWidgets import QWidget

from mlgidgui import peak_picking
from mlgidgui.file_model import PeakTable
from mlgidgui.polar import polar_to_qxyz
from mlgidgui.viewer_styles import (
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    ANGULAR_SUBDIV_FULL,
    ANGULAR_SUBDIV_MIN,
)


def _disable_viewport_scroll(widget) -> None:
    """Disable QAbstractScrollArea-level scrolling on a pyqtgraph widget.

    Pyqtgraph's GraphicsView / PlotWidget inherit from QAbstractScrollArea,
    so even with the scrollbars hidden the viewport can still slide
    when the scene rect is slightly bigger than the visible area
    (typically by a few pixels of axis-label padding). Overriding
    ``scrollContentsBy`` to a no-op blocks every scroll path —
    scrollbar drag, wheel-on-bar, two-finger gesture, programmatic
    `setValue` — without touching the inner ViewBox's pan / zoom,
    which lives one Qt level deeper as a graphics-item event.

    Implemented by reparenting the instance to a dynamically-created
    subclass; safer than instance-level monkey-patching since Qt
    dispatches virtual methods through the C++ vtable.
    """
    cls = type(widget)
    if cls.__name__.endswith("_NoScroll"):
        return
    new_cls = type(
        cls.__name__ + "_NoScroll",
        (cls,),
        {"scrollContentsBy": lambda self, dx, dy: None},
    )
    widget.__class__ = new_cls


def _bin_index(axis: np.ndarray, value: float) -> int:
    """Floor-based bin index for an evenly-spaced axis.

    The image-display routines call ``setImage(pos=axis[0], scale=step)``,
    so axis[0] is the LOWER edge of pixel 0 and pixel ``i`` covers
    ``[axis[0] + i*step, axis[0] + (i+1)*step)``. Returning the bin
    index by ``floor`` instead of ``argmin(|axis - v|)`` keeps the
    cursor readout constant within a displayed pixel — argmin
    transitions at axis-midpoints, which is half a pixel off from
    where the user sees the boundary.
    """
    n = len(axis)
    if n == 0:
        return 0
    if n == 1:
        return 0
    step = (float(axis[-1]) - float(axis[0])) / (n - 1)
    if step == 0.0:
        return 0
    idx = int(np.floor((float(value) - float(axis[0])) / step))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def _robust_levels(frame: np.ndarray) -> tuple[float, float]:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.5))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _apply_raw_flips(
    frame: np.ndarray,
    flip_lr: bool,
    flip_ud: bool,
    transp: bool = False,
) -> np.ndarray:
    """Orient a file-order (H, W) raw detector frame for preview.

    Mirrors pygid's ``process_image`` **in its order** -- transpose,
    then flipud, then fliplr -- so the preview matches the orientation
    the conversion will produce. The order matters: transposing after a
    flip is a different image. Returns the frame unchanged when nothing
    is set.
    """
    if transp:
        frame = frame.T
    if flip_ud:
        frame = np.flipud(frame)
    if flip_lr:
        frame = np.fliplr(frame)
    return frame


def _raw_pixel_index(
    row: int,
    col: int,
    shape: tuple[int, int],
    flip_lr: bool,
    flip_ud: bool,
    transp: bool = False,
) -> tuple[int, int]:
    """Map a displayed (row, col) back to the file-order pixel.

    ``_apply_raw_flips`` is what the user is looking at, so a cursor
    position is an index into the *oriented* frame; the intensity has to
    be read from the stack, which is in file order. This inverts the
    orientation -- undo fliplr, undo flipud, undo transpose -- against
    the file-order ``shape`` (H, W). Returns ``(-1, -1)`` when the point
    lies outside the oriented frame, which the caller reports as NaN.
    """
    height, width = int(shape[0]), int(shape[1])
    out_h, out_w = (width, height) if transp else (height, width)
    if not (0 <= row < out_h and 0 <= col < out_w):
        return -1, -1
    if flip_lr:
        col = out_w - 1 - col
    if flip_ud:
        row = out_h - 1 - row
    return (col, row) if transp else (row, col)


@dataclass
class _DisplayParams:
    """Per-mode display state. ``image_pg`` is the *single 2D frame*
    for the current frame index — we never feed pyqtgraph a 3D stack
    anymore (see lazy-loading milestone). The viewer keeps the most
    recent ``_DisplayParams`` so per-frame scrubs can re-use ``pos`` /
    ``scale`` / ``levels`` without re-deriving them.
    """
    image_pg: np.ndarray
    pos: tuple[float, float]
    scale: tuple[float, float]
    levels: tuple[float, float]
    x_label: tuple[str, str]
    y_label: tuple[str, str]


@dataclass
class ManualPeak:
    """A user-drawn polar peak box. In-memory only; phase 4c persists these."""

    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    temp_id: int = 0


@dataclass
class SelectedPeak:
    """Snapshot of the currently-selected peak, regardless of source.

    Carries enough geometry for the ROI and parameter panel without forcing
    callers to know which overlay holds the underlying data. ``manual_ref``
    is set only when ``kind == "manual"`` and is the same instance held by
    ``_manual_peaks`` — mutating it propagates to the manual overlay.
    """

    kind: str  # "manual" | "detected" | "fitted" | "matched"
    frame: int
    peak_id: int
    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    structure_uid: str | None = None
    # Human-readable structure label + overlay color, populated only for
    # matched selections so the parameter panel can render the source row
    # without re-deriving these from the viewer's matched bookkeeping.
    structure_label: str | None = None
    structure_color: str | None = None
    manual_ref: ManualPeak | None = None
    # mlgidDETECT confidence score for the underlying row. Populated for
    # detected / fitted / matched selections; left as None for manual
    # peaks (which have no model provenance) so the parameter panel can
    # skip the row instead of showing a misleading zero.
    score: float | None = None
    # Peak amplitude (2D-Gaussian peak height). Populated for detected /
    # fitted / matched selections and shown in the parameter panel.
    # The profile viewer does NOT use it for the curve: a saved peak is
    # drawn from its stored centre and width, with the height read off
    # the box-averaged profile (``fit.gaussian_at_stored_shape``).
    # None for manual peaks (no fit yet).
    amplitude: float | None = None
    # When non-None, the selection represents a *matched structure*
    # rather than a single peak. The list holds every fitted-peak id
    # that belongs to the structure; the overlay highlight is drawn
    # around all of them at once. The single peak_id field still
    # carries a representative id (typically the first one) so the
    # ParameterPanel can render the structure_label + first-peak
    # geometry. Always None for detected / fitted / manual selections.
    multi_peak_ids: list[int] | None = None

    @classmethod
    def from_manual(cls, peak: ManualPeak, frame: int) -> SelectedPeak:
        return cls(
            kind="manual",
            frame=frame,
            peak_id=peak.temp_id,
            radius=peak.radius,
            angle=peak.angle,
            radius_width=peak.radius_width,
            angle_width=peak.angle_width,
            is_ring=peak.is_ring,
            manual_ref=peak,
        )

    def polar_tuple(self) -> tuple[float, float, float, float]:
        return (self.radius, self.angle, self.radius_width, self.angle_width)


# -- Undo/redo actions -----------------------------------------------------
#
# Each action carries the data needed to flip the viewer + (for FileGeom)
# the file. ``undo`` and ``redo`` mirror each other so we can move back and
# forth on the stack without special-casing.


class _Action(Protocol):
    def undo(self, viewer: "GIWAXSImageViewer") -> None: ...
    def redo(self, viewer: "GIWAXSImageViewer") -> None: ...


@dataclass
class ManualAddAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)


@dataclass
class ManualRemoveAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)


@dataclass
class ManualGeomAction:
    frame: int
    peak: ManualPeak
    before: tuple[float, float, float, float]  # (r, a, dr, da)
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.after)


@dataclass
class ManualReplaceAction:
    """Atomic swap of the single manual peak on a frame.

    With the new "at most one manual box per frame" policy, drawing a
    new box replaces any existing one. We model that as a single undo
    entry (instead of separate add + remove entries) so a single
    Ctrl+Z rewinds the whole replace cleanly. ``old_peak`` may be None
    when the user drew the very first manual peak on this frame —
    redo then just adds the new one without removing anything.
    """

    frame: int
    old_peak: ManualPeak | None
    new_peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.new_peak)
        if self.old_peak is not None:
            viewer._undoable_add_manual(self.frame, self.old_peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        if self.old_peak is not None:
            viewer._undoable_remove_manual(self.frame, self.old_peak)
        viewer._undoable_add_manual(self.frame, self.new_peak)


@dataclass
class FileGeomAction:
    frame: int
    kind: str  # "detected" | "fitted" | "list:<dataset>"
    peak_id: int
    before: tuple[float, float, float, float]
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.after)


@dataclass
class FileScoreAction:
    """One file-backed peak's confidence score changing.

    Separate from ``FileGeomAction`` because the two must not be
    conflated: undoing a relabel has to leave the box exactly where it
    is, and undoing a drag has to leave the label alone.
    """

    frame: int
    kind: str
    peak_id: int
    before: float
    after: float

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_score(
            self.frame, self.kind, self.peak_id, self.before
        )

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_score(
            self.frame, self.kind, self.peak_id, self.after
        )


@dataclass
class BatchAction:
    """Several actions the user made in one gesture, undone as one.

    Members undo in REVERSE and redo forwards, the same rule
    ``h5_edit_ops.BatchOp`` follows for the Structure tree: an action
    that depends on an earlier one must be taken back first.
    """

    actions: list

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        for action in reversed(self.actions):
            action.undo(viewer)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        for action in self.actions:
            action.redo(viewer)


def _peaks_subset(table: PeakTable, ids: list[int]) -> PeakTable:
    """Return a row-subset of ``table`` keyed by ``ids``.

    Used to build a multi-peak selection overlay from a matched
    structure's full id list. Rows whose id isn't in ``table.ids``
    are silently skipped (stale matched references — same tolerance
    applied in ``file_model.list_matched_structures``).
    """
    if len(table) == 0 or not ids:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
            score=empty, amplitude=empty,
        )
    want = set(int(x) for x in ids)
    idx = np.array(
        [i for i in range(len(table)) if int(table.ids[i]) in want],
        dtype=int,
    )
    if idx.size == 0:
        return _peaks_subset(table, [])
    return PeakTable(
        q_xy=table.q_xy[idx],
        q_z=table.q_z[idx],
        angle=table.angle[idx],
        radius=table.radius[idx],
        angle_width=table.angle_width[idx],
        radius_width=table.radius_width[idx],
        is_ring=table.is_ring[idx],
        ids=table.ids[idx],
        score=table.score[idx],
        amplitude=table.amplitude[idx],
    )


def _peaks_from_manual(manual: list[ManualPeak]) -> PeakTable:
    """Adapt a list of ManualPeak to the PeakTable shape so the existing
    rendering helpers can draw them without special-casing."""
    if not manual:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
            score=empty,
            amplitude=empty,
        )
    _radius = np.array([m.radius for m in manual], dtype=float)
    _angle = np.array([m.angle for m in manual], dtype=float)
    _qxy, _qz = polar_to_qxyz(_radius, _angle)
    return PeakTable(
        q_xy=_qxy,
        q_z=_qz,
        angle=_angle,
        radius=_radius,
        angle_width=np.array([m.angle_width for m in manual], dtype=float),
        radius_width=np.array([m.radius_width for m in manual], dtype=float),
        is_ring=np.array([m.is_ring for m in manual], dtype=bool),
        ids=np.array([m.temp_id for m in manual], dtype=int),
        score=np.zeros(len(manual), dtype=float),
        amplitude=np.zeros(len(manual), dtype=float),
    )


class _LabelEventFilter(QObject):
    """Qt event filter that emits high-level labelling signals from raw mouse
    events on a graphics-view's viewport.

    Installed instead of subclassing ``pg.ViewBox`` because pyqtgraph's drag
    dispatch only fires ``mouseDragEvent`` when the press was accepted at the
    QGraphicsItem layer — which it isn't for plain LMB on the image area, so a
    ViewBox subclass never sees the drag.
    """

    drawStarted = Signal(QPointF)
    drawUpdated = Signal(QPointF, QPointF)
    drawFinished = Signal(QPointF, QPointF)
    # LMB click (no drag). Carries the data-space click point + the
    # keyboard modifiers that were held at press time. Ctrl+click is
    # multi-select; bare click replaces; Ctrl+Alt+click never gets
    # here because the press branch routes it to ``drawStarted``.
    selectAt = Signal(QPointF, object)  # (QPointF, Qt.KeyboardModifiers)
    # Bare LMB double-click (no modifiers, no drag) — wired to reset zoom.
    doubleClicked = Signal()
    # Hover-aware cursor tracking — fires on every mouse move (with or
    # without a button held). The viewer translates the data-space point
    # into the public ``cursorMoved`` payload consumed by the status bar.
    cursorPos = Signal(QPointF)
    cursorLeft = Signal()

    # Pixel tolerance below which a press+release counts as a click, not a drag.
    CLICK_TOLERANCE_PX = 4

    def __init__(
        self, graphics_view, viewbox: pg.ViewBox, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._gv = graphics_view
        self._vb = viewbox
        self._drawing = False
        self._origin: QPointF | None = None
        self._press_pos: QPoint | None = None
        self._press_mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier

    def install(self) -> None:
        self._gv.viewport().installEventFilter(self)
        # MouseMove only fires with a button held unless tracking is on.
        # The status-bar cursor readout needs hover updates, so force it.
        self._gv.viewport().setMouseTracking(True)

    def eventFilter(self, _obj: QObject, ev: QEvent) -> bool:  # type: ignore[override]
        et = ev.type()
        if (
            et == QEvent.Type.MouseButtonDblClick
            and ev.button() == Qt.MouseButton.LeftButton
            and ev.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            # Bare LMB double-click anywhere on the image resets the zoom.
            # Modifier+double-click and other-button double-click fall
            # through so pyqtgraph's default handlers (e.g. ROI editing)
            # still see the event.
            self.doubleClicked.emit()
            return True
        if et == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            mods = ev.modifiers()
            if _has_label_modifiers(mods):
                pos = self._viewport_to_data(ev.position().toPoint())
                self._origin = pos
                self._drawing = True
                self.drawStarted.emit(pos)
                return True  # consume so pan doesn't engage
            self._press_pos = ev.position().toPoint()
            self._press_mods = mods
            # Ctrl-only LMB press: consume so pyqtgraph's ViewBox
            # doesn't take it as a zoom-rect / pan-modifier drag and
            # swallow the matching release before our click handler
            # sees it. Bare LMB still falls through so pyqtgraph can
            # pan / zoom as before.
            if (
                bool(mods & Qt.KeyboardModifier.ControlModifier)
                and not bool(mods & Qt.KeyboardModifier.AltModifier)
            ):
                return True
            return False
        if et == QEvent.Type.MouseMove:
            # Always emit cursor position for the status-bar readout —
            # independent of whether a draw drag is in progress.
            data_pos = self._viewport_to_data(ev.position().toPoint())
            self.cursorPos.emit(data_pos)
            if self._drawing and self._origin is not None:
                self.drawUpdated.emit(self._origin, data_pos)
                return True
        if et == QEvent.Type.Leave:
            self.cursorLeft.emit()
        if et == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
            if self._drawing and self._origin is not None:
                end = self._viewport_to_data(ev.position().toPoint())
                self.drawFinished.emit(self._origin, end)
                self._drawing = False
                self._origin = None
                return True
            if self._press_pos is not None:
                delta = ev.position().toPoint() - self._press_pos
                is_click = (
                    delta.manhattanLength() <= self.CLICK_TOLERANCE_PX
                )
                press_mods = self._press_mods
                self._press_pos = None
                self._press_mods = Qt.KeyboardModifier.NoModifier
                ctrl_only_press = (
                    bool(press_mods & Qt.KeyboardModifier.ControlModifier)
                    and not bool(press_mods & Qt.KeyboardModifier.AltModifier)
                )
                if is_click:
                    # Emit for any single-modifier combo (Ctrl alone for
                    # multi-select toggle, no modifier for replace). Ctrl+Alt
                    # never reaches here because the press branch routed
                    # to ``drawStarted`` and consumed the event. Other
                    # combinations (e.g. Shift) fall through to the same
                    # slot, which can decide whether to act.
                    pos = self._viewport_to_data(ev.position().toPoint())
                    self.selectAt.emit(pos, press_mods)
                    # Bare clicks don't consume — pyqtgraph still emits
                    # a click for menus, etc. Ctrl+LMB release IS
                    # consumed because we consumed the matching press;
                    # pyqtgraph would otherwise see an orphan release.
                    if ctrl_only_press:
                        return True
        return False

    def _viewport_to_data(self, viewport_pt: QPoint) -> QPointF:
        scene_pt = self._gv.mapToScene(viewport_pt)
        return self._vb.mapSceneToView(scene_pt)


def _has_label_modifiers(mods: Qt.KeyboardModifier) -> bool:
    return bool(
        mods & Qt.KeyboardModifier.ControlModifier
        and mods & Qt.KeyboardModifier.AltModifier
    )


class _PeakShapeItem(pg.GraphicsObject):
    """Draws a collection of peak shapes from a single QPainterPath.

    In polar mode every peak is an axis-aligned rectangle. In Cartesian mode,
    rings become quarter-circle arcs at the central radius and segments become
    polygons formed by tessellating the polar rectangle's angular edges.
    """

    def __init__(self, color: str, style: Qt.PenStyle, width: float) -> None:
        super().__init__()
        pen = pg.mkPen(QColor(color), width=width)
        pen.setStyle(style)
        pen.setCosmetic(True)  # constant pixel width regardless of zoom
        self._pen = pen
        self._path = QPainterPath()
        self._bounding = QRectF()

    def set_pen_color(self, color: str) -> None:
        """Swap the outline colour, keeping style and width.

        Used by the theme switch: the selection highlight is white on the
        dark plot ground and near-black on the light one, and the item is
        built once at construction.
        """
        self.set_pen(color=color)

    def set_pen(
        self,
        color: str | None = None,
        style: Qt.PenStyle | None = None,
        width: float | None = None,
    ) -> None:
        """Replace any part of the outline pen, keeping the rest.

        The item is built once at construction, so every later change --
        a theme switch, a user-picked overlay pen -- comes through here.
        """
        pen = pg.mkPen(
            QColor(color) if color is not None else self._pen.color(),
            width=self._pen.widthF() if width is None else float(width),
        )
        pen.setStyle(self._pen.style() if style is None else style)
        pen.setCosmetic(True)
        self._pen = pen
        self.update()

    def set_polar(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent, clip_finite=False,
                )
                if clip is None:
                    continue
                a_lo, a_hi = clip
                r = float(peaks.radius[i])
                dr = float(peaks.radius_width[i])
                path.addRect(QRectF(r - dr / 2, a_lo, dr, a_hi - a_lo))
        self._update_path(path)

    def set_cartesian(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent, clip_finite=False,
                )
                if clip is None:
                    continue
                a_lo, a_hi = clip
                path.addPath(
                    _polar_rect_polygon(
                        float(peaks.radius[i]),
                        float(peaks.radius_width[i]),
                        a_lo,
                        a_hi,
                    )
                )
        self._update_path(path)

    def clear_path(self) -> None:
        self._update_path(QPainterPath())

    def _update_path(self, path: QPainterPath) -> None:
        self.prepareGeometryChange()
        self._path = path
        self._bounding = path.boundingRect()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding

    def paint(self, painter, *_args) -> None:
        painter.setPen(self._pen)
        painter.drawPath(self._path)


def _box_of(peak: ManualPeak) -> peak_picking.Box:
    return peak_picking.Box(
        radius=float(peak.radius),
        radius_width=float(peak.radius_width),
        angle=float(peak.angle),
        angle_width=float(peak.angle_width),
        is_ring=bool(peak.is_ring),
    )


def _box_of_row(table: PeakTable, i: int) -> peak_picking.Box:
    return peak_picking.Box(
        radius=float(table.radius[i]),
        radius_width=float(table.radius_width[i]),
        angle=float(table.angle[i]),
        angle_width=float(table.angle_width[i]),
        is_ring=bool(table.is_ring[i]),
    )


def _polar_box_contains(
    peak: ManualPeak, x: float, y: float, *, ring_edge_tol: float | None = None,
) -> bool:
    """Hit-test the polar bounding box of a ManualPeak.

    ``ring_edge_tol`` (data units) makes a ring clickable only near its
    inner / outer radius; ``None`` keeps the whole band, which is the
    historical behaviour. See ``mlgidgui.peak_picking`` for why.
    """
    return peak_picking.contains(
        _box_of(peak), x, y, ring_edge_tol=ring_edge_tol)


def _polar_table_row_contains(
    table: PeakTable, i: int, x: float, y: float, *,
    ring_edge_tol: float | None = None,
) -> bool:
    """Hit-test row ``i`` of a PeakTable in polar coordinates."""
    return peak_picking.contains(
        _box_of_row(table, i), x, y, ring_edge_tol=ring_edge_tol)


def _cart_to_polar(q_xy: float, q_z: float) -> tuple[float, float]:
    """Map a Cartesian click ``(q_xy, q_z)`` back to ``(radius, angle_deg)``.

    Matches the convention used everywhere else in the pipeline: the
    Cartesian projections of a polar peak are written as
    ``q_xy = r * cos(a)``, ``q_z = r * sin(a)`` (see
    ``file_model.add_fitted_peak_row``), so inverting with
    ``hypot`` + ``atan2(q_z, q_xy)`` reproduces the original
    ``(radius, angle)``. The polar containment checks downstream
    can then run unchanged.
    """
    r = float(np.hypot(q_xy, q_z))
    a = float(np.degrees(np.arctan2(q_z, q_xy)))
    return r, a


def _cart_box_contains(
    peak: ManualPeak, q_xy: float, q_z: float, *,
    ring_edge_tol: float | None = None,
) -> bool:
    """Cartesian hit-test for a ManualPeak's polar box."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_box_contains(peak, r, a, ring_edge_tol=ring_edge_tol)


def _cart_table_row_contains(
    table: PeakTable, i: int, q_xy: float, q_z: float, *,
    ring_edge_tol: float | None = None,
) -> bool:
    """Cartesian hit-test for row ``i`` of a PeakTable."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_table_row_contains(table, i, r, a, ring_edge_tol=ring_edge_tol)


def _clip_angle(
    a_deg: float,
    da_deg: float,
    extent: tuple[float, float] | None = None,
    clip_finite: bool = True,
) -> tuple[float, float] | None:
    """Clip a polar angular box to the viewer's visible range.

    Treats infinite or non-finite angle_width as 'spans the whole quadrant',
    so rings (whose angle_width is sometimes inf) still draw correctly.
    A finite box is clipped to ``extent`` too unless ``clip_finite`` is
    False, which the peak boxes pass so a box whose angular width
    reaches past the image edge (below 0 degrees, say) is drawn in full
    from its position and width; only a box with no finite width has
    nothing but the extent to span.
    ``extent`` is the actual displayed angular axis of the active polar
    stack — pass it so ring overlays stop at the image edge instead of
    extending to the global ``[-180°, 180°]`` clipping bounds. When
    ``extent`` is None the global bounds are used as a fallback (raw-
    mode renders, unit tests).

    Returns (lo, hi) in degrees, or None if the box is empty/invalid.
    """
    if extent is None:
        ext_lo, ext_hi = ANGLE_MIN_DEG, ANGLE_MAX_DEG
    else:
        ext_lo, ext_hi = float(extent[0]), float(extent[1])
        if ext_hi < ext_lo:
            ext_lo, ext_hi = ext_hi, ext_lo
    if not np.isfinite(a_deg) or not np.isfinite(da_deg):
        a_lo, a_hi = ext_lo, ext_hi
    else:
        a_lo = a_deg - da_deg / 2.0
        a_hi = a_deg + da_deg / 2.0
    if clip_finite:
        a_lo = max(a_lo, ext_lo)
        a_hi = min(a_hi, ext_hi)
    if a_hi <= a_lo:
        return None
    return a_lo, a_hi


def _polar_rect_polygon(
    radius: float, dr: float, a_lo_deg: float, a_hi_deg: float
) -> QPainterPath:
    """Render a polar rectangle (already clipped) as a closed polygon in q-space.

    For full-quadrant rings this becomes a proper quarter-annulus that closes
    along the q_xy and q_z axes. For narrow segments, a thin curved trapezoid.
    """
    a_lo = np.deg2rad(a_lo_deg)
    a_hi = np.deg2rad(a_hi_deg)
    inner = max(radius - dr / 2.0, 0.0)
    outer = radius + dr / 2.0

    span = a_hi_deg - a_lo_deg
    n_sub = max(
        int(np.ceil(span / 90.0 * ANGULAR_SUBDIV_FULL)),
        ANGULAR_SUBDIV_MIN,
    )
    angs = np.linspace(a_lo, a_hi, n_sub)

    path = QPainterPath()
    path.moveTo(QPointF(float(outer * np.cos(angs[0])), float(outer * np.sin(angs[0]))))
    for ang in angs[1:]:
        path.lineTo(QPointF(float(outer * np.cos(ang)), float(outer * np.sin(ang))))
    for ang in angs[::-1]:
        path.lineTo(QPointF(float(inner * np.cos(ang)), float(inner * np.sin(ang))))
    path.closeSubpath()
    return path
