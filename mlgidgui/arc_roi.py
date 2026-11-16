"""Resize / move handles for a peak box in the Cartesian view.

In the polar view a box is an axis-aligned rectangle and a ``pg.ROI``
with four edge handles edits it. In the Cartesian (q_xy, q_z) view the
same box is an arc box (annular sector), which ``pg.ROI`` cannot draw,
so ``ArcBoxROI`` does it: the outline of the box plus four handles --
the outer and inner radius at the middle angle, and the two angular
edges at the middle radius -- and a drag on the body moves the whole
box. Geometry stays polar (radius, radius_width, angle, angle_width),
exactly what the polar ROI edits, so the viewer feeds both into the
same undo / write path.

The three signals carry the same names as ``pg.ROI``'s
(``sigRegionChangeStarted`` / ``sigRegionChanged`` /
``sigRegionChangeFinished``), so the viewer connects and tears both
down the same way.

Rings have radial handles only (their angular extent is the whole
image by definition), and a body drag moves only their radius -- the
same rule as the polar ROI.
"""
from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsItem

from mlgidgui.viewer_items import _cart_to_polar, _polar_rect_polygon

# Handle square, in screen pixels (handles ignore the zoom).
HANDLE_PX = 8.0
# Extra pick margin around the outline, in screen pixels, so the thin
# edge of a narrow box can still be grabbed for a move.
OUTLINE_PICK_PX = 6.0

OUTER, INNER, ANGLE_HI, ANGLE_LO = "outer", "inner", "angle_hi", "angle_lo"


class _ArcHandle(pg.GraphicsObject):
    """One square handle; a drag moves one edge of the parent box."""

    def __init__(self, parent: "ArcBoxROI", edge: str, pen: QPen) -> None:
        super().__init__(parent)
        self._roi = parent
        self.edge = edge
        self._pen = pen
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        h = HANDLE_PX / 2.0 + 2.0
        return QRectF(-h, -h, 2 * h, 2 * h)

    def paint(self, painter, *_args) -> None:
        h = HANDLE_PX / 2.0
        painter.setPen(self._pen)
        painter.setBrush(QBrush(self._pen.color()))
        painter.drawRect(QRectF(-h, -h, 2 * h, 2 * h))

    def hoverEvent(self, ev) -> None:
        if not ev.isExit():
            ev.acceptDrags(Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        pos = self._roi.mapFromScene(ev.scenePos())
        if ev.isStart():
            self._roi.begin_change()
        self._roi.drag_edge(self.edge, pos.x(), pos.y())
        if ev.isFinish():
            self._roi.end_change()


class ArcBoxROI(pg.GraphicsObject):
    """Editable arc box in data coordinates (x = q_xy, y = q_z)."""

    sigRegionChangeStarted = Signal(object)
    sigRegionChanged = Signal(object)
    sigRegionChangeFinished = Signal(object)

    def __init__(
        self,
        radius: float,
        radius_width: float,
        angle: float,
        angle_width: float,
        *,
        is_ring: bool = False,
        angle_extent: tuple[float, float] = (0.0, 90.0),
        pen: QPen | None = None,
    ) -> None:
        super().__init__()
        self._pen = pen if pen is not None else pg.mkPen("w")
        self._pen.setCosmetic(True)
        self.is_ring = bool(is_ring) or not math.isfinite(float(angle_width))
        self._extent = (float(angle_extent[0]), float(angle_extent[1]))
        self._r = float(radius)
        self._dr = abs(float(radius_width))
        self._a = float(angle)
        self._da = abs(float(angle_width)) if not self.is_ring else float(angle_width)
        self._path = QPainterPath()
        self._pick = QPainterPath()
        self._drag_origin: tuple[float, float, float, float, float, float] | None = None
        edges = (OUTER, INNER) if self.is_ring else (OUTER, INNER, ANGLE_HI, ANGLE_LO)
        self.handles = {e: _ArcHandle(self, e, self._pen) for e in edges}
        self.setAcceptHoverEvents(True)
        self._rebuild()

    # -- geometry --------------------------------------------------------

    def box(self) -> tuple[float, float, float, float]:
        """``(radius, radius_width, angle, angle_width)``."""
        return self._r, self._dr, self._a, self._da

    def set_box(
        self, radius: float, radius_width: float,
        angle: float, angle_width: float,
    ) -> None:
        """Set the geometry without emitting (undo, external edits)."""
        self._r = float(radius)
        self._dr = abs(float(radius_width))
        if not self.is_ring:
            self._a = float(angle)
            self._da = abs(float(angle_width))
        self._rebuild()

    def _angle_range(self) -> tuple[float, float]:
        if self.is_ring:
            return self._extent
        return self._a - self._da / 2.0, self._a + self._da / 2.0

    def _rebuild(self) -> None:
        a_lo, a_hi = self._angle_range()
        self.prepareGeometryChange()
        self._path = _polar_rect_polygon(self._r, self._dr, a_lo, a_hi)
        self._pick = QPainterPath()
        self._pick.addPath(self._path)
        a_mid = math.radians((a_lo + a_hi) / 2.0)
        r_lo = max(self._r - self._dr / 2.0, 0.0)
        r_hi = self._r + self._dr / 2.0
        spots = {
            OUTER: (r_hi, a_mid),
            INNER: (r_lo, a_mid),
            ANGLE_HI: (self._r, math.radians(a_hi)),
            ANGLE_LO: (self._r, math.radians(a_lo)),
        }
        for edge, handle in self.handles.items():
            rr, aa = spots[edge]
            handle.setPos(QPointF(rr * math.cos(aa), rr * math.sin(aa)))
        self.update()

    # -- edits -----------------------------------------------------------

    def begin_change(self) -> None:
        self.sigRegionChangeStarted.emit(self)

    def end_change(self) -> None:
        self._drag_origin = None
        self.sigRegionChangeFinished.emit(self)

    def drag_edge(self, edge: str, q_xy: float, q_z: float) -> None:
        """Move one edge to the pointer; the opposite edge stays put.

        Dragging past the opposite edge swaps them (the width stays
        positive), like the polar ROI.
        """
        r_ptr, a_ptr = _cart_to_polar(q_xy, q_z)
        if edge in (OUTER, INNER):
            fixed = (
                self._r - self._dr / 2.0 if edge == OUTER
                else self._r + self._dr / 2.0
            )
            lo, hi = sorted((max(fixed, 0.0), max(r_ptr, 0.0)))
            self._r, self._dr = (lo + hi) / 2.0, hi - lo
        elif not self.is_ring:
            fixed = (
                self._a - self._da / 2.0 if edge == ANGLE_HI
                else self._a + self._da / 2.0
            )
            lo, hi = sorted((fixed, a_ptr))
            self._a, self._da = (lo + hi) / 2.0, hi - lo
        else:
            return
        self._rebuild()
        self.sigRegionChanged.emit(self)

    def move_to(self, start: tuple[float, float], now: tuple[float, float]) -> None:
        """Translate the box (as it was at drag start) by the polar
        distance between two Cartesian pointer positions."""
        if self._drag_origin is None:
            r0, a0 = _cart_to_polar(*start)
            self._drag_origin = (self._r, self._dr, self._a, self._da, r0, a0)
        r_box, _dr, a_box, _da, r0, a0 = self._drag_origin
        r1, a1 = _cart_to_polar(*now)
        self._r = max(r_box + (r1 - r0), self._dr / 2.0)
        if not self.is_ring:
            self._a = a_box + (a1 - a0)
        self._rebuild()
        self.sigRegionChanged.emit(self)

    # -- QGraphicsItem ---------------------------------------------------

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect()

    def shape(self) -> QPainterPath:
        # The filled box plus a band around its outline, so a narrow box
        # can still be grabbed. The band width is in data units derived
        # from the current pixel size.
        stroker = QPainterPathStroker()
        px = self.pixelLength(pg.Point(1, 0)) or 0.0
        stroker.setWidth(max(OUTLINE_PICK_PX * float(px), 1e-9))
        out = QPainterPath(self._pick)
        out.addPath(stroker.createStroke(self._path))
        return out

    def paint(self, painter, *_args) -> None:
        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def hoverEvent(self, ev) -> None:
        if not ev.isExit():
            ev.acceptDrags(Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        start = ev.buttonDownPos()
        now = ev.pos()
        if ev.isStart():
            self._drag_origin = None
            self.begin_change()
        self.move_to((start.x(), start.y()), (now.x(), now.y()))
        if ev.isFinish():
            self.end_change()
