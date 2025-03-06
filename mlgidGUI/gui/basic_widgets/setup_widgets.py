# -*- coding: utf-8 -*-
import logging
from copy import deepcopy
from abc import abstractmethod
from typing import NamedTuple

from PyQt5.QtWidgets import (QWidget, QLineEdit, QHBoxLayout,
                             QLabel, QFormLayout,
                             QPushButton, QRadioButton)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal

from ...app.app import App
from .buttons import InfoButton
from ..tools import Icon, center_widget, validate_scientific_value

logger = logging.getLogger(__name__)


class AbstractInputParametersWidget(QWidget):
    class InputParameters(NamedTuple):
        name: str
        label: str
        type: type
        info: str = None
        none: str = False

    @property
    @abstractmethod
    def PARAMETER_TYPES(self):
        pass

    @property
    @abstractmethod
    def NAME(self):
        pass

    @property
    @abstractmethod
    def DEFAULT_DICT(self):
        return {}

    @classmethod
    def get_config(cls):
        d = deepcopy(cls.DEFAULT_DICT)
        config_d = App().fm.config[cls.NAME] or {}
        d.update(config_d)
        return d

    def __init__(self, parent=None, params_dict: dict = None):
        super().__init__(parent)
        self.setWindowIcon(Icon('setup'))
        self.setWindowTitle(self.NAME)
        self.default_dict = params_dict or self.get_config()

    def get_parameters_dict(self):
        parameters_dict = dict()
        for p in self.PARAMETER_TYPES:
            value = getattr(self, p.name)
            if not p.none and value is None:
                return
            parameters_dict[p.name] = value
        return parameters_dict

    def save_to_config(self, parameters_dict: dict):
        App().fm.config[self.NAME] = parameters_dict

    def _get_layout(self,
                    input_parameter: 'InputParameters'):
        current_value = self.default_dict.get(input_parameter.name, '')
        if current_value is None:
            current_value = ''
        label_widget = QLabel(input_parameter.label)
        input_widget = QLineEdit(str(current_value))
        if input_parameter.type is int:
            input_widget.setValidator(QIntValidator())
        layout = QHBoxLayout()
        layout.addWidget(label_widget, Qt.AlignHCenter)
        layout.addWidget(input_widget, Qt.AlignHCenter)
        if input_parameter.info:
            info_button = InfoButton(input_parameter.info)
            layout.addWidget(info_button, Qt.AlignLeft)

        def get_input(s):
            return validate_scientific_value(input_widget, input_parameter.type, input_parameter.none)

        setattr(AbstractInputParametersWidget, input_parameter.name,
                property(get_input))
        return layout


class BasicInputParametersWidget(AbstractInputParametersWidget):
    apply_signal = pyqtSignal(dict)
    close_signal = pyqtSignal()

    def __init__(self, parent=None, params_dict: dict = None):
        super().__init__(parent, params_dict)
        self.form = QFormLayout()
        for p in self.PARAMETER_TYPES:
            self.form.addRow(self._get_layout(p))
        self.save_button = self._init_save_button()
        self.apply_button = self._init_apply_button()
        self.cancel_button = self._init_cancel_button()
        self.form.addWidget(self.save_button)
        self.form.addWidget(self.apply_button)
        self.form.addWidget(self.cancel_button)
        self.setLayout(self.form)
        center_widget(self)

    def _init_apply_button(self):
        apply_button = QPushButton('Apply')

        apply_button.clicked.connect(self.on_apply)
        return apply_button

    def _init_cancel_button(self):
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.close)
        return cancel_button

    def on_apply(self, *args):
        parameters_dict = self.get_parameters_dict()
        if parameters_dict is not None:
            self.apply_signal.emit(parameters_dict)
            logger.debug(parameters_dict)
            if self.save_button.isChecked():
                self.save_to_config(parameters_dict)
            self.close()

    def close(self):
        self.close_signal.emit()
        AbstractInputParametersWidget.close(self)

    @staticmethod
    def _init_save_button():
        save_button = QRadioButton('Save parameters')
        save_button.setChecked(True)
        return save_button
