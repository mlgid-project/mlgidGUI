from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal

from ...app.data_manager import SaveFormats, SavingParameters


class FormatBox(QComboBox):
    sigFormatChanged = pyqtSignal(SaveFormats)

    def __init__(self, parent):
        super().__init__(parent)
        self._current_format: SaveFormats = SaveFormats.entire_h5
        self.addItems([save_format.value for save_format in SaveFormats])
        self.currentTextChanged.connect(self._format_changed)
        self.setCurrentText(self._current_format.value)

    @property
    def save_format(self) -> SaveFormats:
        return self._current_format

    @pyqtSlot(SaveFormats, name='setFormat')
    def set_format(self, save_format: SaveFormats):
        if self._current_format.value != save_format.value:
            self._current_format = save_format
            self.setCurrentText(save_format.value)

    def update_params(self, saving_params: SavingParameters):
        saving_params.format = self.save_format

    @pyqtSlot(name='selectedFormatChanged')
    def _format_changed(self):
        self._current_format = SaveFormats(self.currentText())
        self.sigFormatChanged.emit(self._current_format)
