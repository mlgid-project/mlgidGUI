from typing import Dict, Tuple
from itertools import product

from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QColor

from pyqtgraph import ColorButton

from ...app.rois import RoiTypes
from ..basic_widgets import Label



class SelectRoiColorWidget(QWidget):
    sigColorChanged = pyqtSignal(tuple, QColor)
    sigColorChanging = pyqtSignal(tuple, QColor)

    def __init__(self, colors_dict: RoiColorsDict, parent=None, font_size: int = 12):
        super().__init__(parent)
        self.setFixedSize(450, 500)
        self.setWindowTitle('Select roi colors')
        self._color_dict: RoiColorsDict = colors_dict
        self._label_props: dict = {'font_size': font_size}
        self._color_buttons: dict = {}

        self._init_ui()

    def _init_ui(self):
        layout = self._layout = QGridLayout(self)
        layout.addWidget(Label('Type', self, bold=True, **self._label_props), 0, 0)
        layout.addWidget(Label('Selected', self, bold=True, **self._label_props), 0, 1)
        layout.addWidget(Label('Fixed', self, bold=True, **self._label_props), 0, 2)
        layout.addWidget(Label('Color', self, bold=True, **self._label_props), 0, 3)

        for row, (roi_type, selected, fixed) in enumerate(
                product((RoiTypes.ring, RoiTypes.segment, RoiTypes.background),
                        (True, False),
                        (True, False)
                        ), 1
        ):
            self._add_row(row, roi_type, selected, fixed)

    def _add_row(self, row: int, roi_type: RoiTypes, selected: bool, fixed: bool):
        color_button = self._create_color_button((roi_type, selected, fixed))
        layout = self._layout

        layout.addWidget(Label(roi_type.name, self, **self._label_props), row, 0)
        layout.addWidget(Label('Yes' if selected else 'No', self, **self._label_props), row, 1)
        layout.addWidget(Label('Yes' if fixed else 'No', self, **self._label_props), row, 2)
        layout.addWidget(color_button, row, 3)

    def _create_color_button(self, key: tuple) -> ColorButton:
        color: QColor = self._color_dict.get(key, QColor(0, 0, 255))
        self._color_buttons[key] = color_button = ColorButton(self, color)
        color_button.sigColorChanged.connect(lambda btn: self.sigColorChanged.emit(key, btn.color()))
        color_button.sigColorChanging.connect(lambda btn: self.sigColorChanging.emit(key, btn.color()))
        return color_button
