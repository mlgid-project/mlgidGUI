# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMainWindow, QLabel, QComboBox, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, Qt, QPointF

import numpy as np
from .basic_widgets import (CustomImageViewer, BlackToolBar, BasicInputParametersWidget,
                            InfoButton, AbstractInputParametersWidget,
                            DrawRoiController)
from .roi_widgets.roi_2d_rect_widget import Roi2DRect
from .roi_widgets.abstract_roi_holder import AbstractRoiHolder

from ..app.app import App
from ..app.rois.roi import Roi, RoiTypes
from ..app.polar_image import INTERPOLATION_ALGORITHMS
from .tools import Icon


class _PolarCustomImageViewer(CustomImageViewer):
    def set_default_range(self):
        return self.set_auto_range()


class PolarImageViewer(AbstractRoiHolder, QMainWindow):

    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'PolarImageViewer')
        QMainWindow.__init__(self, parent)
        self.app = parent.app
        self._setup_window = None
        self._interpolation_params_dict: dict = InterpolateSetupWindow.get_config()
        self._image_viewer = _PolarCustomImageViewer(self)

        self.register_key_patch()

        self._draw_roi = PolarDrawRoi(self._image_viewer.view_box, self)
        self._draw_roi.sigCreateRoi.connect(self.app.roi_dict.add_roi)
        self._draw_roi.sigMoveRoi.connect(self.app.roi_dict.move_roi)

        self._image_viewer.image_plot.getAxis('bottom').setLabel(
            text='|Q|', color='white', font_size='large')
        self._image_viewer.image_plot.getAxis('left').setLabel(
            text='&Phi;', color='white', font_size='large')
        self.setCentralWidget(self._image_viewer)
        self.app.image_holder.sigPolarImageChanged.connect(self._update_image)
        self.app.geometry_holder.sigScaleChanged.connect(self._on_scale_changed)
        self.app.image_holder.sigEmptyImage.connect(self._image_viewer.clear_image)
        # self.app.roi_dict.sig_roi_moved.connect(self._image_viewer.set_auto_range)

        self.__init_toolbar()

    def _on_scale_changed(self):
        self.set_axes()

    def _make_roi_widget(self, roi):
        roi_widget = Roi2DRect(roi)
        self._image_viewer.image_plot.addItem(roi_widget)
        return roi_widget

    def _delete_roi_widget(self, roi_widget):
        self._image_viewer.image_plot.removeItem(roi_widget)
        self._image_viewer.set_auto_range()

    def __init_toolbar(self):
        setup_toolbar = BlackToolBar('Setup', self)
        self.addToolBar(setup_toolbar)

        setup_action = setup_toolbar.addAction(Icon('setup'), 'Setup')
        setup_action.triggered.connect(self.open_setup_window)

    @pyqtSlot(name='updateImage')
    def _update_image(self):
        img = self.app.polar_image
        if img is not None:
            self._image_viewer.set_data(img)
            self.set_axes()

    def set_axes(self):
        if self._image_viewer.image_item.image is None:
            return
        p1, p2 = self.app.geometry.phi_range
        r1, r2 = np.nanmin(self.app.geometry.r_axis), self.app.geometry.r_axis.max()
        self._image_viewer.set_x_axis(r1, r2)
        self._image_viewer.set_y_axis(p1 * 180 / np.pi, p2 * 180 / np.pi)
        self._image_viewer.view_box.setAspectLocked(
            True, self.app.geometry.polar_aspect_ratio)
        self._image_viewer.set_auto_range()

    def open_setup_window(self):
        self._setup_window = InterpolateSetupWindow(params_dict=self.app.image_holder.polar_image_params_dict())
        self._setup_window.apply_signal.connect(self.set_fit_parameters)
        self._setup_window.close_signal.connect(self.close_peaks_setup)
        self._setup_window.show()

    def set_fit_parameters(self, params: dict):
        self._interpolation_params_dict = params
        self.app.image_holder.set_polar_image_params(params)

    def close_peaks_setup(self):
        self._setup_window = None


class PolarDrawRoi(DrawRoiController):

    def _update_roi(self, point: QPointF):
        r1, r2 = self._init_point.x(), point.x()
        p1, p2 = self._init_point.y(), point.y()
        self._roi.radius, self._roi.width = (r1 + r2) / 2, abs(r2 - r1)
        self._roi.angle, self._roi.angle_std = (p1 + p2) / 2, abs(p2 - p1)

    def _init_roi(self) -> Roi:
        return Roi(0, 0, 0, 0, type=RoiTypes.segment)


class InterpolateSetupWindow(BasicInputParametersWidget):
    P = BasicInputParametersWidget.InputParameters
    PARAMETER_TYPES = (P('r_size', 'Radius axis size', int),
                       P('phi_size', 'Angle axis size', int),
                       P('mode', 'Interpolation algorithm', str))

    NAME = 'Interpolation parameters'

    DEFAULT_DICT = dict(r_size=1024, phi_size=1024, mode='Bilinear')

    def _get_layout(self,
                    input_parameter: BasicInputParametersWidget.InputParameters):
        if input_parameter.name == 'mode':
            return self._get_mode_layout(input_parameter)
        else:
            return super()._get_layout(input_parameter)

    def _get_mode_layout(self, input_parameter: BasicInputParametersWidget.InputParameters):

        current_value = self.default_dict.get(input_parameter.name, None)
        if current_value is None:
            current_value = next(iter(INTERPOLATION_ALGORITHMS.keys()))
        label_widget = QLabel(input_parameter.label)
        input_widget = QComboBox()
        input_widget.setEditable(False)
        input_widget.addItems(list(INTERPOLATION_ALGORITHMS.keys()))
        input_widget.setCurrentText(current_value)
        layout = QHBoxLayout()
        layout.addWidget(label_widget, Qt.AlignHCenter)
        layout.addWidget(input_widget, Qt.AlignHCenter)
        if input_parameter.info:
            info_button = InfoButton(input_parameter.info)
            layout.addWidget(info_button, Qt.AlignLeft)

        def get_input(*args):
            return input_widget.currentText()

        setattr(AbstractInputParametersWidget, input_parameter.name, property(get_input))
        return layout
