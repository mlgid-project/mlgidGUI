from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QRadioButton,
    QGroupBox,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..tools import Icon
from ...app.rois import Roi


class SetConfidenceLevelWidget(QWidget):
    sigSetConfidenceLevel = pyqtSignal(float)

    def __init__(self, confidence_level_name: str, parent=None):
        flags = Qt.WindowFlags()
        flags |= Qt.FramelessWindowHint

        super().__init__(parent=parent, flags=flags)

        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("mlgidGUI")
        self.setWindowIcon(Icon("window_icon"))

        self._init_ui(confidence_level_name)
        self._init_layout()
        self._init_connect()

    def _init_ui(self, confidence_level_name):
        self.options_list = ConfidenceOptionsList(confidence_level_name, self)
        self.save_btn = QPushButton('Save', self)
        self.cancel_btn = QPushButton('Cancel', self)

    def _init_layout(self):
        layout = QGridLayout(self)
        layout.addWidget(self.options_list, 0, 0, 1, 2)
        layout.addWidget(self.save_btn, 1, 0)
        layout.addWidget(self.cancel_btn, 1, 1)

    def _init_connect(self):
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.cancel_btn.clicked.connect(self.close)

    def _on_save_clicked(self):
        self.sigSetConfidenceLevel.emit(self.options_list.get_level())
        self.close()


class ConfidenceOptionsList(QGroupBox):
    LEVELS = Roi.CONFIDENCE_LEVELS

    sigConfidenceChanged = pyqtSignal(float)

    def __init__(self, confidence_level_name: str, parent=None, horizontal: bool = False):
        super().__init__(parent=parent)
        self._init_ui()
        self._init_layout(horizontal)
        self.set_level(confidence_level_name)
        self._init_connect()

    def _init_connect(self):
        for btn in self.level_bts.values():
            btn.toggled.connect(self._send_value)

    def _send_value(self):
        self.sigConfidenceChanged.emit(self.get_level())

    def set_level(self, level: str or float):
        if not isinstance(level, str):
            level = self.level2name(level)
        current_btn = self.level_bts[self.get_level_name()]

        current_btn.setChecked(False)
        current_btn.setDown(False)

        self.level_bts[level].setChecked(True)
        self.level_bts[level].setDown(True)

    def _init_ui(self):
        self.level_bts = {name: QRadioButton(name, self) for name in self.LEVELS.keys()}

    def _init_layout(self, horizontal: bool = False):
        layout = QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        for name in self.LEVELS.keys():
            layout.addWidget(self.level_bts[name])

    def get_level(self) -> float:
        return self.LEVELS[self.get_level_name()]

    def get_level_name(self) -> str:
        for btn in self.level_bts.values():
            if btn.isChecked():
                return btn.text()
        return Roi.DEFAULT_CONFIDENCE_LEVEL

    def level2name(self, level: float):
        for k, v in self.LEVELS.items():
            if v == level:
                return k
        raise ValueError(f'Unknown confidence level {level}')
