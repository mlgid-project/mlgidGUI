from abc import abstractmethod
from typing import Callable

from PyQt5.QtGui import QColor

from .rois_context_menu import RoiContextMenu
from ...app.rois.roi import Roi
from ..tools import DummySignal


def _color_key(roi: Roi) -> str:
    if roi.active and roi.movable:
        return 'active'
    if not roi.active and roi.movable:
        return 'default'
    if roi.active and not roi.movable:
        return 'fixed_active'
    return 'fixed'


class AbstractRoiWidget(object):
    OPACITY: int = None

    COLOR_DICT = dict(
        default=QColor(0, 0, 255),
        active=QColor(255, 100, 255),
        fixed=QColor(0, 255, 0),
        fixed_active=QColor(255, 0, 255)
    )

    sigRoiMoved = DummySignal()
    sigSelected = DummySignal()
    sigShiftSelected = DummySignal()

    def __init__(self, roi: Roi, enable_context: bool = False, context_menu: Callable[[Roi], None] = RoiContextMenu):
        self._roi = roi
        if enable_context:
            self._context_menu = context_menu
        else:
            self._context_menu = None

    def update_color(self):
        color = self.COLOR_DICT[_color_key(self.roi)]
        if self.OPACITY:
            color.setAlpha(self.OPACITY)
        self.set_color(color)

    def update_roi(self):

        self.move_roi()

        if not self.roi.movable:
            self.fix()
        else:
            self.unfix()
        self.update_color()

    @abstractmethod
    def move_roi(self):
        pass

    @abstractmethod
    def send_move(self):
        pass

    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi: Roi, update: bool = True):
        self._roi = roi
        if update:
            if not self.roi.movable:
                self.fix()
            self.update_color()
            self.move_roi()

    @abstractmethod
    def set_color(self, color):
        pass

    def show_context_menu(self, *args):
        if self._context_menu:
            self._context_menu(self.roi)

    def update_select(self):
        self.update_color()

    def fix(self):
        self.update_color()

    def unfix(self):
        self.update_color()

    def rename(self):
        pass

    def change_cif_file(self):
        pass

    def change_conf_level(self):
        pass

    def change_type(self):
        pass

    # def show_roi(self):
    #     self.show()
    #
    # def hide_roi(self):
    #     self.hide()
