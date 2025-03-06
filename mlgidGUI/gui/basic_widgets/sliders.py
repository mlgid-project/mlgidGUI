# -*- coding: utf-8 -*-
import logging

from PyQt5.QtWidgets import (QSlider, QLineEdit, QHBoxLayout,
                             QLabel, QWidgetAction, QMenu,
                             QSizePolicy, QWidget)
from PyQt5.QtGui import QPainter, QPainterPath, QColor, QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRectF

import numpy as np

from .buttons import RoundedPushButton

from ..tools import color_animation

logger = logging.getLogger(__name__)


class DoubleSlider(QSlider):
    valueChangedByHand = pyqtSignal(float)
    _MAX_INT = int(1e9)

    def __init__(self, orientation=Qt.Horizontal, parent=None, decimals: int = 5,
                 value: float = 0, bounds: tuple = (0, 1), log_scale: bool = False):
        super().__init__(orientation=orientation, parent=parent)
        self.decimals: int = decimals
        self._pressed: bool = False
        self._log_scale: bool = log_scale
        self.sliderPressed.connect(self._set_pressed)
        self.sliderReleased.connect(self._set_released)
        self.sliderReleased.connect(self.emit_value)
        self.valueChanged.connect(self._check_and_emit)

        super().setMinimum(0)
        super().setMaximum(self._MAX_INT)

        self._min_value = bounds[0]
        self._max_value = bounds[1]
        if self._log_scale and bounds[0] <= 0:
            raise ValueError(f'Wrong minimal bound for log scale: {bounds[0]}')
        self.setValue(value)

    def emit_value(self, *args):
        self.valueChangedByHand.emit(self.value())

    def _set_pressed(self, *args):
        self._pressed = True

    def _set_released(self, *args):
        self._pressed = False

    def _check_and_emit(self, *args):
        if self._pressed:
            self.emit_value()

    def set_decimals(self, decimals):
        self.decimals = decimals

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def _real_to_view(self, value):
        try:
            if self._log_scale:
                return int((np.log10(value) - np.log10(self._min_value)) /
                           (np.log10(self._max_value) - np.log10(self._min_value)) * self._MAX_INT)
            else:
                return int((value - self._min_value) / self._value_range * self._MAX_INT)
        except ZeroDivisionError:
            return 0

    def _view_to_real(self, value):
        if self._log_scale:
            return 10 ** (value / self._MAX_INT * (np.log10(self._max_value) - np.log10(self._min_value)) +
                          np.log10(self._min_value))
        else:
            return value / self._MAX_INT * self._value_range + self._min_value

    def value(self):
        return self._view_to_real(super().value())

    def setValue(self, value):
        super().setValue(self._real_to_view(value))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")
        if self._log_scale and value <= 0:
            raise ValueError(f'Wrong minimal bound for log scale: {value}')
        real_value = self.value()
        self._min_value = value
        self.setValue(real_value)
        if real_value < value:
            self.emit_value()

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        real_value = self.value()
        self._max_value = value
        self.setValue(real_value)
        if real_value > value:
            self.emit_value()

    def setRange(self, p_int, p_int_1):
        real_value = self.value()
        self._min_value = p_int
        self._max_value = p_int_1
        if self._log_scale and p_int <= 0:
            raise ValueError(f'Wrong minimal bound for log scale: {p_int}')
        self.setValue(real_value)

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value


class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(float)

    _HEIGHT = 50
    _WIDTH_MIN = 300
    _WIDTH_MAX = 300

    def __init__(self, name: str, bounds: tuple = (0, 1),
                 value: float = 0,
                 parent=None, orientation=Qt.Horizontal, decimals: int = 3, scientific: bool = False,
                 log_scale: bool = False):
        super().__init__(parent=parent)
        self.name = name
        self._decimals = decimals
        self._scientific: bool = scientific
        self._bounds = bounds
        self.slider = DoubleSlider(orientation, parent, decimals, value, bounds, log_scale=log_scale)
        self.line_edit = QLineEdit(self.get_str_value())
        self.label = QLabel(name)

        self.line_edit.editingFinished.connect(self._set_value_from_text)
        self.line_edit.setStyleSheet('QLineEdit {  border: none; }')

        self.slider.valueChangedByHand.connect(self._set_value_from_slider)

        layout = QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.line_edit)

        self.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Fixed
        )

        self.setFixedHeight(self._HEIGHT)
        self.edit_width = self.fontMetrics().width(
            ' '.join([' '] * (self._decimals + 4))) + int(self._scientific) * 50 + 10
        self.line_edit.setMaximumWidth(self.edit_width)
        self.setMaximumWidth(self._WIDTH_MAX)

    def sizeHint(self) -> QSize:
        return QSize(self._WIDTH_MIN, self._HEIGHT)

    @property
    def value(self) -> float:
        return self.slider.value()

    def get_str_value(self) -> str:
        if self._decimals:
            return f'{self.slider.value():.{self._decimals}{"e" if self._scientific else "f"}}'
        else:
            return str(int(self.slider.value()))

    def _set_value_from_slider(self):
        self.line_edit.setText(self.get_str_value())
        self.valueChanged.emit(self.slider.value())

    def _set_value_from_text(self):
        value = self.line_edit.text()
        try:
            if self._decimals:
                value = float(value)
            else:
                value = int(value)
            self.slider.setValue(value)
            if value < self._bounds[0] or value > self._bounds[1]:
                raise ValueError()
        except (ValueError, OverflowError):
            self.line_edit.setText(self.get_str_value())
            color_animation(self.line_edit)

        self.valueChanged.emit(self.slider.value())

    def set_value(self, value: float, change_bounds: bool = True):
        if change_bounds and value < self.slider.minimum():
            self.slider.setMinimum(value)

        elif change_bounds and value > self.slider.maximum():
            self.slider.setMaximum(value)

        self.slider.setValue(value)
        self.line_edit.setText(self.get_str_value())


class AnimatedSlider(RoundedPushButton):
    valueChanged = pyqtSignal(float)
    statusChanged = pyqtSignal(str)
    _EditMaximumWidth = 90
    _Height = 40
    _padding = 50

    @property
    def value(self):
        return self.slider.value()

    def __init__(self,
                 name: str,
                 bounds: tuple = (0, 1),
                 value: float = 0,
                 parent=None,
                 orientation=Qt.Horizontal,
                 decimals: int = 0,
                 hide: bool = True,
                 context_menu_enabled: bool = True,
                 disable_changing_status: bool = False,
                 min_max_bounds: tuple = (-10000, 10000)):

        super(AnimatedSlider, self).__init__(parent)
        self.context_menu_enabled = context_menu_enabled
        self.name = name
        self.bounds = bounds
        self._decimals = decimals
        self.init_value = value
        self.__init_ui__(orientation)
        self.clicked.connect(self.on_clicked)
        self._status = 'show'
        self._min_max_bounds = min_max_bounds
        self._disable_changing_status = disable_changing_status
        if hide:
            self.hide_and_show('hide')

    def set_min_max_bounds(self, bounds: tuple):
        if bounds[0] < bounds[1]:
            self._min_max_bounds = bounds

    def on_clicked(self):
        self.hide_and_show()
        self.statusChanged.emit(self._status)

    def contextMenuEvent(self, *args, **kwargs):
        if not self.context_menu_enabled:
            return
        event = args[0]
        menu = QMenu()
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        menu.setAttribute(Qt.WA_TranslucentBackground)
        menu.setStyleSheet("QMenu{background:rgba(255, 255, 255, 0%);}")

        decimals_slider = AnimatedSlider(
            'Decimals', (0, 6), self._decimals, hide=False,
            context_menu_enabled=False,
            disable_changing_status=True
        )
        min_slider = AnimatedSlider(
            'Min value', (self._min_max_bounds[0], self.slider.maximum()),
            self.slider.minimum(), hide=False,
            context_menu_enabled=False, disable_changing_status=True)
        max_slider = AnimatedSlider(
            'Max value', (self.slider.minimum(), self._min_max_bounds[1]),
            self.slider.maximum(), hide=False,
            context_menu_enabled=False, disable_changing_status=True)
        decimals_slider.valueChanged.connect(self.set_decimals)
        min_slider.valueChanged.connect(self.slider.setMinimum)
        min_slider.valueChanged.connect(max_slider.slider.setMinimum)
        max_slider.valueChanged.connect(self.slider.setMaximum)
        max_slider.valueChanged.connect(
            lambda x: min_slider.slider.setMaximum(x - 1e-45))

        for w in [decimals_slider, min_slider, max_slider]:
            w_action = QWidgetAction(self)
            w_action.setDefaultWidget(w)
            menu.addAction(w_action)
        menu.exec_(self.mapToGlobal(event.pos()))

    def set_decimals(self, value):
        try:
            value = int(value)
        except ValueError as err:
            logger.exception(err)
            return
        self.slider.set_decimals(value)
        self._decimals = value
        self.editValue.setText(self._get_str_value())
        self._update_label_text()

    def set_bounds(self, bounds: tuple, change_value: bool = False):
        try:
            self.slider.setRange(*bounds)
        except ValueError as err:
            if not change_value:
                logger.exception(err)
            else:
                self.slider.setValue(bounds[0])
                self.slider.setRange(*bounds)

    def hide_and_show(self, new_status: str = None):
        if self._disable_changing_status:
            return
        if new_status in 'show hide'.split():
            self._status = new_status
        else:
            self._status = 'hide' if self._status == 'show' else 'show'

        if self._status == 'hide':
            self.slider.hide()
            self.editValue.hide()
            self._update_label_text()
        else:
            self.slider.show()
            self.editValue.show()
            self.label.setText(self.name)

    def _update_label_text(self, *args):
        if self._status == 'hide':
            new_text = f'{self.name}{" = " if self.name else ""}{self._get_str_value()}'
            self.label.setText(new_text)

    def _get_str_value(self):
        if self._decimals:
            return f'{self.slider.value():.{self._decimals}f}'
        else:
            return str(int(self.slider.value()))

    def set_value(self, value: float, change_bounds: bool = False):
        emit_value = False
        if value < self.slider.minimum():
            if change_bounds:
                self.slider.setMinimum(value)
            else:
                emit_value = True
        if value > self.slider.maximum():
            if change_bounds:
                self.slider.setMaximum(value)
            else:
                emit_value = True
        self.slider.setValue(value)
        self.editValue.setText(self._get_str_value())
        if emit_value:
            self.slider.emit_value()

    def __init_ui__(self, orientation):
        layout = QHBoxLayout(self)

        self.slider = DoubleSlider(orientation, self,
                                   decimals=self._decimals)

        self.slider.setRange(*self.bounds)
        self.slider.setValue(self.init_value)
        self.slider.valueChangedByHand.connect(
            lambda *x: self.valueChanged.emit(self.slider.value()))
        self.editValue = QLineEdit(self._get_str_value())
        self.editValue.setMaximumWidth(self._EditMaximumWidth)
        self.editValue.setStyleSheet('QLineEdit {  border: none; }')
        self.label = QLabel(self.name)
        self.slider.valueChanged.connect(self._update_label_text)

        if not self._decimals:
            self.editValue.setValidator(QIntValidator(*self.bounds))

        self.slider.valueChangedByHand.connect(
            lambda _: self.editValue.setText(self._get_str_value()))
        self.editValue.editingFinished.connect(self._set_slider_value)

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.editValue)

        self.setFixedHeight(self._Height)
        min_width = self.fontMetrics().width(
            ' = '.join([self.name, self._get_str_value()])) + self._padding
        self.setMinimumWidth(min_width)

    def _set_slider_value(self):
        value = self.editValue.text()
        try:
            if self._decimals:
                value = float(value)
            else:
                value = int(value)
            self.slider.setValue(value)
            if value < self.bounds[0] or value > self.bounds[1]:
                raise ValueError()
        except ValueError:
            self.editValue.setText(self._get_str_value())
            color_animation(self.editValue)
        self.valueChanged.emit(self.slider.value())


class ControlSlider(AnimatedSlider):
    def __init__(self, *args, **kwargs):
        AnimatedSlider.__init__(self, *args, **kwargs)
        self._fixed = False

    def set_fixed(self):
        self.hide_and_show('hide')
        self._fixed = True
        self.context_menu_enabled = False
        self.label.setStyleSheet('QLabel {color: green; }')

    def set_unfixed(self):
        self._fixed = False
        self.context_menu_enabled = True
        self.label.setStyleSheet('QLabel {color: black; }')

    def on_clicked(self):
        if not self._fixed:
            AnimatedSlider.on_clicked(self)


class ParametersSlider(QWidget):
    sigLowerValueChanged = pyqtSignal(float)
    sigMiddleValueChanged = pyqtSignal(float)
    sigUpperValueChanged = pyqtSignal(float)

    _rect_width = 10
    _rect_height = 20
    _padding = 30
    _slider_width = 5

    # _path_color = QColor('black')
    _rect_color = QColor(100, 100, 255)

    def __init__(self, x1=0., x2=0.5, x3=0.1,
                 min_value=0, max_value=1, parent=None):
        super().__init__(parent=parent)

        # self.setMouseTracking(True)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum
        )

        self._pressed: int = 0

        self._min_value = min_value
        self._max_value = max_value

        self._x1 = x1
        self._x2 = x2
        self._x3 = x3

    def setValues(self, x1, x2, x3, *, adjust_range: bool = False, new_range: tuple = None):
        if x1 <= x2 <= x3:
            if adjust_range:
                self._min_value = min(self._min_value, x1)
                self._max_value = max(self._max_value, x3)
            elif new_range:
                self._min_value, self._max_value = new_range
                if self._min_value > self._max_value:
                    raise ValueError(f'Wrong range: {self._min_value} > {self._max_value}')

            if x1 < self._min_value or x3 > self._max_value:
                raise ValueError('Out of range.')

            self._x1 = x1
            self._x2 = x2
            self._x3 = x3

            self.update()

        else:
            raise ValueError(f'Wrong order: {x1}, {x2}, {x3}')

    def sizeHint(self):
        return QSize(300, self._rect_height)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            ev.ignore()
        else:
            x = ev.pos().x()
            self._pressed = self._get_idx(x)

            if self._pressed:
                ev.accept()
            else:
                ev.ignore()

    def mouseReleaseEvent(self, ev):
        if self._pressed:
            ev.accept()
            self._pressed = 0
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if not self._pressed:
            # idx = self._get_idx(ev.pos().x())
            # if idx:
            #     print('Hover: ', idx)
            #     ev.accept()
            # else:
            ev.ignore()

        else:
            x = ev.pos().x()
            bounds = self._get_bounds(self._pressed)

            if x < bounds[0]:
                self._set_value(bounds[0])
            elif x > bounds[1]:
                self._set_value(bounds[1])
            else:
                self._set_value(x)

    def _get_idx(self, x):
        w = self._rect_width / 2

        if abs(self._x1_view - x) <= w:
            return 1
        elif abs(self._x2_view - x) <= w:
            return 2
        elif abs(self._x3_view - x) <= w:
            return 3
        else:
            return 0

    def _get_bounds(self, idx: int):
        if not idx:
            return
        w = self._rect_width

        if idx == 1:
            return self._padding, self._x2_view - w
        elif idx == 2:
            return self._x1_view + w, self._x3_view - w
        else:
            return self._x2_view + w, self.width() - self._padding

    def _set_value(self, x):
        idx = self._pressed
        if not idx:
            return

        value = self._scale_from_view(x, idx)

        if idx == 1 and value != self._x1:
            self._x1 = value
            self.sigLowerValueChanged.emit(self._x1)
            self.update()
        elif idx == 2 and value != self._x2:
            self._x2 = value
            self.sigMiddleValueChanged.emit(self._x2)
            self.update()
        elif idx == 3 and value != self._x3:
            self._x3 = value
            self.sigUpperValueChanged.emit(self._x3)
            self.update()

    def paintEvent(self, ev):

        p = QPainter(self)

        self._x1_view = self._scale_to_view(self._x1, 1)
        self._x2_view = self._scale_to_view(self._x2, 2)
        self._x3_view = self._scale_to_view(self._x3, 3)

        p.setRenderHint(QPainter.Antialiasing)

        self._draw_slider(p)
        self._draw_rect(self._x1_view, p)
        self._draw_rect(self._x2_view, p)
        self._draw_rect(self._x3_view, p)

        p.end()

    def _draw_slider(self, p: QPainter):
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(self._padding, self.height() / 2 - self._slider_width / 2,
                   self.width() - self._padding * 2, self._slider_width),
            3, 3)
        # pen = QPen(self._path_color, 2)
        # p.setPen(pen)
        p.fillPath(path, self._rect_color)
        p.drawPath(path)

    def _draw_rect(self, x: float, p: QPainter):
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(x - self._rect_width / 2,
                   self.height() / 2 - self._rect_height / 2,
                   self._rect_width, self._rect_height),
            1, 1)
        # pen = QPen(self._path_color, 2)
        # p.setPen(pen)
        p.fillPath(path, self._rect_color)
        p.drawPath(path)

    @property
    def _length(self):
        return abs(self.width() - 2 * self._padding - 2 * self._rect_width) + 1e-5

    @property
    def _range(self):
        return abs(self._max_value - self._min_value) + 1e-5

    def _scale_to_view(self, x, idx: int):
        return self._padding + self._rect_width * (idx - 1) + \
               (x - self._min_value) / self._range * self._length

    def _scale_from_view(self, x, idx: int):
        return self._min_value + (x - self._padding - self._rect_width * (idx - 1)) * \
               self._range / self._length
