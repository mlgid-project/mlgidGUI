from typing import Dict

from PyQt5.QtWidgets import QWidget, QGridLayout, QCheckBox, QComboBox

from ...app.data_manager import (
    SavingParameters,
    SaveFormats,
    TextFormats,
    MetaTextFormats,
)
from ...gui.basic_widgets import Label


class OptionsWidget(QWidget):
    def __init__(self, saving_parameters: SavingParameters, parent=None):
        super().__init__(parent)
        self.saving_parameters = saving_parameters
        self._current_format = None
        self._init_ui(saving_parameters)
        self.set_format(saving_parameters.format, saving_parameters)


    def _init_ui(self, saving_parameters: SavingParameters):
        self.bool_options = BoolOptionsWidget(saving_parameters, self)
        self.text_options = TextFormatOptions(saving_parameters, self)
        self.text_options.setHidden(True)

    def set_format(self, save_format: SaveFormats, saving_parameters: SavingParameters):
        if self._current_format != save_format:
            self._current_format = save_format
            if save_format.value == SaveFormats.entire_h5.value:
                saving_parameters.set_entire_h5_params()
                self.bool_options.setDisabled(True)
                self.text_options.setHidden(True)
            elif save_format.value == SaveFormats.partial_h5.value:
                saving_parameters.set_partial_h5_params()
                self.bool_options.setDisabled(False)
                self.text_options.setHidden(True)
            elif save_format.value == SaveFormats.text.value:
                self.bool_options.setDisabled(False)
                self.text_options.setHidden(False)
            elif save_format.value in (
                    SaveFormats.object_detection.value,
                    SaveFormats.partial_h5.value,
            ):
                self.bool_options.setDisabled(True)
                self.text_options.setHidden(True)
            else:
                raise ValueError(f'Unknown save format {save_format}.')

    def update_params(self, params: SavingParameters):
        self.bool_options.update_params(params)
        self.text_options.update_params(params)


class BoolOptionsWidget(QWidget):
    def __init__(self, saving_parameters: SavingParameters, parent=None):
        super().__init__(parent)
        self._check_boxes: Dict[SavingParameters, QCheckBox] = {}
        self._init_ui(saving_parameters)

    def _init_ui(self, saving_parameters: SavingParameters):
        layout = QGridLayout(self)

        for i, (attr, name) in enumerate(SavingParameters.BOOL_FLAGS.items()):
            label = Label(name, self)
            check_box = QCheckBox(self)
            value = getattr(saving_parameters, attr, SavingParameters.__dataclass_fields__[attr].default)
            check_box.setChecked(value)
            self._check_boxes[attr] = check_box
            layout.addWidget(label, i, 0)
            layout.addWidget(check_box, i, 1)

    def update_params(self, params: SavingParameters):

        for attr in SavingParameters.BOOL_FLAGS.keys():
            value = getattr(params, attr, SavingParameters.__dataclass_fields__[attr].default)
            self._check_boxes[attr].setChecked(value)


class TextFormatOptions(QWidget):
    def __init__(self, saving_parameters: SavingParameters, parent=None):
        super().__init__(parent)

        self._init_ui(saving_parameters)

    def _init_ui(self, saving_parameters: SavingParameters):
        layout = QGridLayout(self)
        self.text_format = QComboBox(self)
        self.text_format.addItems([value.value for value in TextFormats])

        self.meta_text_format = QComboBox(self)
        self.meta_text_format.addItems([value.value for value in MetaTextFormats])

        self.text_format.setCurrentText(saving_parameters.text_format.value)
        self.meta_text_format.setCurrentText(saving_parameters.meta_text_format.value)

        layout.addWidget(Label('Table format'), 0, 0)
        layout.addWidget(self.text_format, 1, 0)
        layout.addWidget(Label('Configuration format'), 2, 0)
        layout.addWidget(self.meta_text_format, 3, 0)

    def update_params(self, params: SavingParameters):
        params.text_format = TextFormats(self.text_format.currentText())
        params.meta_text_format = MetaTextFormats(self.meta_text_format.currentText())
