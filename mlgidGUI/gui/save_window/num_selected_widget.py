from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QCheckBox,
    QPushButton,
)

from PyQt5.QtCore import pyqtSignal

from ...app.data_manager import SavingParameters
from ...gui.basic_widgets import Label


class NumSelectedWidget(QWidget):
    sigSkipUnlabelledClicked = pyqtSignal(int)
    sigSelectManuallyClicked = pyqtSignal()

    def __init__(self, saving_params: SavingParameters, parent=None):
        super().__init__(parent)
        self.params = saving_params
        self._init_ui()
        self._init_layout()
        self._init_connect()

    def update_params(self):
        self.num_label.setText(self._get_label())

    def _get_label(self):
        return f'{self.params.num_images} images selected'

    def _init_ui(self):
        self.num_label = Label(self._get_label(), self)
        self.skip_unlabelled_btn = QCheckBox('Skip unlabelled', self)
        self.skip_unlabelled_btn.setChecked(True)
        self.select_manually_btn = QPushButton('Select manually', self)

    def _init_layout(self):
        layout = QGridLayout(self)
        layout.addWidget(self.num_label, 0, 0)
        layout.addWidget(self.skip_unlabelled_btn, 1, 0)
        layout.addWidget(self.select_manually_btn, 2, 0)

    def _init_connect(self):
        self.skip_unlabelled_btn.stateChanged.connect(self.sigSkipUnlabelledClicked)
        self.select_manually_btn.clicked.connect(self.sigSelectManuallyClicked)
