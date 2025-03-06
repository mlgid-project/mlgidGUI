from typing import Dict, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from ..file_manager import FileManager
from .roi import RoiTypes

SELECTED = True
NOT_SELECTED = False
FIXED = True
NOT_FIXED = False

# (type, selected, fixed): (r, g, b)
ROI_COLOR_KEY = Tuple[RoiTypes, bool, bool]
RGB = Tuple[int, int, int]
ROI_COLORS = Dict[ROI_COLOR_KEY, RGB]


_DEFAULT_ROI_COLORS: ROI_COLORS = {
        (RoiTypes.ring, NOT_SELECTED, NOT_FIXED): (0, 0, 255),
        (RoiTypes.ring, SELECTED, NOT_FIXED): (0, 128, 255),
        (RoiTypes.ring, NOT_SELECTED, FIXED): (0, 0, 255),
        (RoiTypes.ring, SELECTED, FIXED): (255, 0, 255),

        (RoiTypes.segment, NOT_SELECTED, NOT_FIXED): (0, 0, 255),
        (RoiTypes.segment, SELECTED, NOT_FIXED): (0, 128, 255),
        (RoiTypes.segment, NOT_SELECTED, FIXED): (0, 0, 255),
        (RoiTypes.segment, SELECTED, FIXED): (255, 0, 255),

        (RoiTypes.background, NOT_SELECTED, NOT_FIXED): (0, 0, 255),
        (RoiTypes.background, SELECTED, NOT_FIXED): (0, 128, 255),
        (RoiTypes.background, NOT_SELECTED, FIXED): (0, 0, 255),
        (RoiTypes.background, SELECTED, FIXED): (255, 0, 255),
    }


class RoiColorsDict(dict):
    def __init__(self, colors: ROI_COLORS = None):
        super().__init__()
        self.update(_DEFAULT_ROI_COLORS)

        if colors:
            self.update(colors)

    def reset(self):
        self.update(_DEFAULT_ROI_COLORS)


class RoiColors(QObject):
    _SAVE_KEY = 'roi_colors'

    sigColorChanged = pyqtSignal(tuple)
    sigColorDictSet = pyqtSignal()

    def __init__(self, fm: FileManager, color_dict: RoiColorsDict = None, parent=None):
        super().__init__(parent)
        self._fm = fm
        self._color_dict: RoiColorsDict = color_dict or RoiColorsDict(fm.config[self._SAVE_KEY])

    def save_colors(self):
        self._fm.config[self._SAVE_KEY] = self._color_dict

    def set_color(self, key: ROI_COLOR_KEY, color: RGB):
        if color != self._color_dict[key]:
            self._color_dict[key] = color
            self.sigColorChanged.emit(key)

    def __getitem__(self, item):
        return self._color_dict[item]

    def __setitem__(self, key: ROI_COLOR_KEY, color: RGB):
        self.set_color(key, color)

    def reset(self):
        self._color_dict.reset()
        self.sigColorDictSet.emit()
