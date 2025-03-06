from abc import abstractmethod
from typing import Dict
import logging

from PyQt5.QtCore import Qt

from .abstract_roi_widget import AbstractRoiWidget
from ...app.app import App
from ...app.rois.roi import Roi


class AbstractRoiHolder(object):
    log = logging.getLogger(__name__)

    def __init__(self, name: str):
        self._name = name
        self._roi_dict = App().roi_dict
        self._roi_widgets: Dict[int, AbstractRoiWidget] = {}
        self._init_connect()
        self._segments_hidden: bool = False
        self._key_patch = _KeyEventsPatch(Qt.Key_H)

    def _init_connect(self):
        self._roi_dict.sig_roi_created.connect(self._create_roi)
        self._roi_dict.sig_roi_deleted.connect(self._delete_roi)
        self._roi_dict.sig_roi_moved.connect(self._move_rois)
        self._roi_dict.sig_selected.connect(self._update_select)
        self._roi_dict.sig_fixed.connect(self._fix_rois)
        self._roi_dict.sig_unfixed.connect(self._unfix_rois)

    def register_key_patch(self, widget=None):
        widget = widget or self
        self._key_patch.attach_to_widget(widget)
        self._key_patch.set_press_func(self.hide_segments)
        self._key_patch.set_release_func(self.show_segments)

    @property
    def segments_hidden(self):
        return self._segments_hidden

    def hide_show_segments(self, hide: bool = None):
        if hide is None:
            hide = not self._segments_hidden
        if hide:
            self.hide_segments()
        else:
            self.show_segments()

    def hide_segments(self, *args):
        for widget in self._roi_widgets.values():
            widget.hide()
        self._segments_hidden = True

    def show_segments(self, *args):
        for widget in self._roi_widgets.values():
            widget.show()
        self._segments_hidden = False

    def _create_roi(self, keys: tuple):
        for key in keys:
            try:
                self._roi_widgets[key] = self._make_roi_widget(self._roi_dict[key])
                self._connect_roi_widget(self._roi_widgets[key])
            except:
                pass

    def _move_rois(self, keys: tuple, name: str):
        if name != self._name:
            for key in keys:
                try:
                    self._roi_widgets[key].move_roi()
                except KeyError as err:
                    self.log.error(f'Key error in {self.__class__.__name__}')
                    self.log.exception(err)

    def _connect_roi_widget(self, roi_widget: AbstractRoiWidget):
        roi_widget.sigRoiMoved.connect(
            lambda k: self._roi_dict.move_roi(k, self._name))
        roi_widget.sigSelected.connect(self._roi_dict.select)
        roi_widget.sigShiftSelected.connect(self._roi_dict.shift_select)

    def _update_select(self, keys: tuple):
        for key in keys:
            try:
                self._roi_widgets[key].update_select()
            except KeyError as err:
                self.log.error(f'Key error in {self.__class__.__name__}')
                self.log.exception(err)

    def _delete_roi(self, keys: tuple):
        for key in keys:
            try:
                self._delete_roi_widget(self._roi_widgets.pop(key))
            except KeyError as err:
                self.log.error(f'Key error in {self.__class__.__name__}')
                self.log.exception(err)

    def _fix_rois(self, keys: tuple):
        for key in keys:
            try:
                self._roi_widgets[key].fix()
            except KeyError as err:
                self.log.error(f'Key error in {self.__class__.__name__}')
                self.log.exception(err)

    def _unfix_rois(self, keys: tuple):
        for key in keys:
            try:
                self._roi_widgets[key].unfix()
            except KeyError as err:
                self.log.error(f'Key error in {self.__class__.__name__}')
                self.log.exception(err)

    @abstractmethod
    def _delete_roi_widget(self, roi_widget) -> None:
        pass

    @abstractmethod
    def _make_roi_widget(self, roi: Roi) -> AbstractRoiWidget:
        pass


class _KeyEventsPatch(object):
    def __init__(self, key: int):
        self._key_listened: int = key

        self._super_key_pressed_func = None
        self._super_key_released_func = None
        self._patched_key_pressed_func = None
        self._patched_key_released_func = None
        self._widget = None

        self._ctrl_pressed: bool = False
        self._key_pressed: bool = False
        self._pressed: bool = False

    def attach_to_widget(self, widget):
        if self._widget:
            self.detach_from_widget()
        self._widget = widget

        self._super_key_pressed_func = widget.keyPressEvent
        self._super_key_released_func = widget.keyReleaseEvent
        widget.keyPressEvent = self._keyPressEvent
        widget.keyReleaseEvent = self._keyReleaseEvent

    def detach_from_widget(self):
        if self._widget:
            self._widget.keyPressEvent = self._super_key_pressed_func
            self._widget.keyReleaseEvent = self._super_key_released_func

            self._widget = None
            self._super_key_pressed_func = None
            self._super_key_released_func = None
            self._ctrl_pressed: bool = False
            self._key_pressed: bool = False
            self._pressed: bool = False

    def set_press_func(self, func):
        self._patched_key_pressed_func = func

    def set_release_func(self, func):
        self._patched_key_released_func = func

    def _keyPressEvent(self, ev):
        if not self._widget:
            return
        if ev.key() == self._key_listened:
            self._key_pressed = True
        if ev.key() == Qt.Key_Control:
            self._ctrl_pressed = True
        if self._patched_key_pressed_func:
            if not self._pressed and (self._key_pressed and self._ctrl_pressed):
                self._patched_key_pressed_func(self._widget, ev)
                self._pressed = True
        self._super_key_pressed_func(ev)

    def _keyReleaseEvent(self, ev):
        if not self._widget:
            return
        if ev.key() == self._key_listened:
            self._key_pressed = False
        if ev.key() == Qt.Key_Control:
            self._ctrl_pressed = False
        if self._patched_key_released_func:
            if self._pressed and not (self._key_pressed and self._ctrl_pressed):
                self._patched_key_released_func(self._widget, ev)
                self._pressed = False
        self._super_key_released_func(ev)
