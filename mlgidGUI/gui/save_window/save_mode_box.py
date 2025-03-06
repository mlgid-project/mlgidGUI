from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal

from ...app.data_manager import SaveMode


class SaveModeBox(QComboBox):
    sigSaveModeChanged = pyqtSignal(SaveMode)

    def __init__(self, parent):
        super().__init__(parent)
        self._mode: SaveMode = SaveMode.create
        self.addItems([mode.value for mode in SaveMode])
        self.currentTextChanged.connect(self._mode_changed)
        self.setCurrentText(self._mode.value)

    @property
    def mode(self) -> SaveMode:
        return self._mode

    @pyqtSlot(SaveMode, name='setMode')
    def set_mode(self, mode: SaveMode):
        if self._mode.value != mode.value:
            self._mode = mode
            self.setCurrentText(mode.value)

    @pyqtSlot(name='selectedModeChanged')
    def _mode_changed(self):
        self._mode = SaveMode(self.currentText())
        self.sigSaveModeChanged.emit(self._mode)
