# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from pyqtgraph import LinearRegionItem

from ...app import Roi, App

from .abstract_roi_widget import AbstractRoiWidget


class Roi1D(AbstractRoiWidget, LinearRegionItem):
    OPACITY: bool = 150

    sigRoiMoved = pyqtSignal(int)
    sigSelected = pyqtSignal(int)
    sigShiftSelected = pyqtSignal(int)

    _ACTIVE_Z = 10
    _INACTIVE_Z = 5
    _FIXED_Z = -30

    def __init__(self, roi: Roi, enable_context: bool = True, **kwargs):
        AbstractRoiWidget.__init__(self, roi, enable_context=enable_context, **kwargs)
        LinearRegionItem.__init__(self)
        self.sigRegionChanged.connect(self.roi_is_moving)
        self.update_roi()

        if App().debug_tracker:
            App().debug_tracker.add_object(self, roi.name)

    def roi_is_moving(self):
        if self.moving:
            self.send_move()
        for l in self.lines:
            if l.moving:
                self.send_move()

    def set_color(self, color):
        self.setBrush(color)
        self.update()

    def set_movable(self, movable: bool):
        self.setMovable(movable)

    def send_move(self):
        x1, x2 = self.getRegion()
        r, w = (x1 + x2) / 2, x2 - x1

        self.roi.radius = r
        self.roi.width = w

        self.sigRoiMoved.emit(self.roi.key)

    @pyqtSlot(name='move_roi')
    def move_roi(self):
        r, w = self.roi.radius, abs(self.roi.width)
        x1, x2 = r - w / 2, r + w / 2
        self.setRegion((x1, x2))
        self.update()

    def hoverEvent(self, ev):
        pass

    def mouseDragEvent(self, ev):
        if not ev.modifiers() == Qt.ShiftModifier:
            super().mouseDragEvent(ev)
            self.send_move()
        else:
            self.mouseClickEvent(ev)

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == Qt.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)
            self.send_move()
        elif ev.button() == Qt.RightButton:
            ev.accept()
            self.show_context_menu(ev)
        elif ev.button() == Qt.LeftButton and ev.modifiers() == Qt.ShiftModifier:
            ev.accept()
            self.sigShiftSelected.emit(self.roi.key)

        elif ev.button() == Qt.LeftButton:
            ev.accept()
            self.sigSelected.emit(self.roi.key)
        else:
            ev.ignore()

        self.viewRangeChanged()

    def update_select(self):
        super().update_select()
        self._update_z_value()

    def fix(self):
        super().fix()
        self.setMovable(False)
        self._update_z_value()

    def unfix(self):
        super().unfix()
        self.setMovable(True)
        self._update_z_value()

    def _update_z_value(self):
        if self.roi.movable:
            if self.roi.active:
                self.setZValue(self._ACTIVE_Z)
            else:
                self.setZValue(self._INACTIVE_Z)
        else:
            self.setZValue(self._FIXED_Z)
        self.update()


class Roi1DAngular(Roi1D):

    @pyqtSlot(name='move_roi')
    def move_roi(self):
        a, a_w = self.roi.angle, self.roi.angle_std
        x1, x2 = a - a_w / 2, a + a_w / 2
        self.setRegion((x1, x2))

    def send_move(self):
        x1, x2 = self.getRegion()
        a, a_w = (x1 + x2) / 2, x2 - x1

        self.roi.angle = a
        self.roi.angle_std = a_w

        self.sigRoiMoved.emit(self.roi.key)
