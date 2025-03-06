# -*- coding: utf-8 -*-
from abc import abstractmethod

from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPointF
from pyqtgraph import ViewBox

from ...app.rois.roi import Roi


class MouseEventsController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mouseMoveEvent(self, ev) -> bool:
        return True

    def mouseReleaseEvent(self, ev) -> bool:
        return True

    def mousePressEvent(self, ev) -> bool:
        return True


class ViewBoxMouseEvents(MouseEventsController):
    def __init__(self, view_box: ViewBox = None, parent=None):
        super().__init__(parent)

        self._view_box = None
        self._press_func = None
        self._release_func = None
        self._move_func = None

        if view_box:
            self.attach_to_view_box(view_box)

    def attach_to_view_box(self, view_box: ViewBox):
        self._view_box = view_box
        self._press_func = view_box.mousePressEvent
        self._release_func = view_box.mouseReleaseEvent
        self._move_func = view_box.mouseMoveEvent

        def mouseMoveEvent(ev):
            if self.mouseMoveEvent(ev):
                self._move_func(ev)

        def mouseReleaseEvent(ev):
            if self.mouseReleaseEvent(ev):
                self._release_func(ev)

        def mousePressEvent(ev):
            if self.mousePressEvent(ev):
                self._press_func(ev)

        self._view_box.mousePressEvent = mousePressEvent
        self._view_box.mouseReleaseEvent = mouseReleaseEvent
        self._view_box.mouseMoveEvent = mouseMoveEvent

    def detach_from_view_box(self):
        if self._view_box:
            self._view_box.mousePressEvent = self._press_func
            self._view_box.mouseReleaseEvent = self._release_func
            self._view_box.mouseMoveEvent = self._move_func

        self._view_box = None
        self._press_func = None
        self._release_func = None
        self._move_func = None


class DrawRoiController(ViewBoxMouseEvents):
    sigCreateRoi = pyqtSignal(object)
    sigMoveRoi = pyqtSignal(int, str)

    def __init__(self, view_box=None, parent=None):
        super().__init__(view_box, parent)
        self._init_point: QPointF = None
        self._roi: Roi = None
        self._pressed: bool = False

    def mousePressEvent(self, ev) -> bool:
        if ev.button() == Qt.LeftButton and int(ev.modifiers()) == (Qt.ControlModifier + Qt.AltModifier):
            self._init_point = self._view_box.mapToView(ev.pos())
            self._roi = self._init_roi()
            self._update_roi(self._init_point)
            self.sigCreateRoi.emit(self._roi)
            self._pressed = True
            ev.accept()
            return False
        return True

    def mouseMoveEvent(self, ev) -> bool:
        if not self._pressed:
            return True
        self.update_roi(ev)
        ev.accept()
        return False

    def mouseReleaseEvent(self, ev) -> bool:
        if self._pressed:
            self.update_roi(ev)

            self._init_point: QPointF = None
            self._roi: Roi = None
            self._pressed: bool = False
            ev.accept()
            return False
        return True

    def update_roi(self, ev):
        self._update_roi(self._view_box.mapToView(ev.pos()))
        if self._roi.key is not None:
            self.sigMoveRoi.emit(self._roi.key, '')

    @abstractmethod
    def _update_roi(self, point: QPointF):
        pass

    def _init_roi(self) -> Roi:
        return Roi(0, 0)
