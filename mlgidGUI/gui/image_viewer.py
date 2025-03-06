# -*- coding: utf-8 -*-
import logging
import math
import numpy as np
from pyqtgraph import CircleROI
logger = logging.getLogger(__name__)


from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton, QLabel, QLineEdit
)

from PyQt5.QtGui import QDoubleValidator, QFont

from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
    QPointF
)

from ..app.rois.roi import Roi
from ..app.cif_rois.cif_peaks_calculation import cif_results
from ..app import App
from ..app.transformations import Transformation

from .basic_widgets import (
    CustomImageViewer,
    LabeledSlider,
    BlackToolBar,
    DrawRoiController
)

from .roi_widgets.roi_2d_ring_widget import Roi2DRing, CIF2DRing
from mlgidGUI.gui.roi_widgets.cif_peak_widget import CIFPoint
from .roi_widgets.abstract_roi_holder import AbstractRoiHolder

from .tools import (
    Icon,
    center_widget
)

class ImageViewer(AbstractRoiHolder, CustomImageViewer):
    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'ImageViewer')
        CustomImageViewer.__init__(
            self, parent)
        self.image_plot.getAxis('bottom').setLabel(text='<math>Q<sub>xy</sub>  (A<sup>-1</sup>) </math>', color='white')
        self.image_plot.getAxis('bottom').label.setFont(QFont("", 10))
        self.image_plot.getAxis('left').setLabel(text='<math>Q<sub>z</sub>  (A<sup>-1</sup>) </math>', color='white')
        self.image_plot.getAxis('left').label.setFont(QFont("",10))

    def _make_roi_widget(self, roi: Roi):
        roi_widget = Roi2DRing(roi)
        self.image_plot.addItem(roi_widget)
        return roi_widget

    def _delete_roi_widget(self, roi_widget: Roi2DRing):
        self.image_plot.removeItem(roi_widget)

    def _draw_cif_result(self, int_limit):
        for item in self.image_plot.allChildItems():
            if (type(item) is CIFPoint) or  (type(item) is CIF2DRing):
                self.image_plot.removeItem(item)

        if self.app.fm.current_key and bool(cif_results):
            try:
                q_range = math.sqrt(math.pow(self.app.fm.current_key.get_q_range()[0], 2) + math.pow(self.app.fm.current_key.get_q_range()[1], 2))
                y_max = self.app.geometry.shape[0]
                x_max = self.app.geometry.shape[1]
                y_min = self.app.geometry.beam_center[0]
                x_min = self.app.geometry.beam_center[1]

                for key, cif_result in cif_results.items():
                    if cif_result.intensity_perc > int_limit:
                        if cif_result.is_powder:
                            cif_result.radius = cif_result.q_z *  (x_max-x_min)/self.app.fm.current_key.get_q_range()[0]
                            cif_result.roi_widget = CIF2DRing(cif_result, q_range)
                            self.image_plot.addItem(cif_result.roi_widget)
                        else:
                            rel_qxy = cif_result.q_xy / self.app.fm.current_key.get_q_range()[0]
                            rel_qz = cif_result.q_z / self.app.fm.current_key.get_q_range()[1]
                            cif_result.roi_widget = CIFPoint(self.image_item, self, (rel_qxy, rel_qz, cif_result.intensity), cif_result.cif_file_nr)
                            self.image_plot.addItem(cif_result.roi_widget)
            except:
                pass

class MainImageViewer(ImageViewer):
    class BeamCenterRoi(CircleROI):
        _ROI_SIZE = 1
        def __init__(self, beam_center, parent):
            CircleROI.__init__(self, (beam_center[1], beam_center[0]),
                               self._ROI_SIZE, movable=False, parent=parent)
            self._center = None
            self._scale = 1
            self.set_center(beam_center)

        def set_center(self, value: tuple, y=None, update=True, finish=True, ):
            self._center = value
            radius = self.size().x() / 2
            pos = (value[1] - radius, value[0] - radius)
            super(MainImageViewer.BeamCenterRoi, self).setPos(
                pos, y, update, finish)

        def set_size(self, size: float = None):
            size = size or self._ROI_SIZE
            size *= self._scale
            self.setSize((size, size), update=False, finish=False)
            self.set_center(self._center)

        def set_scale(self, scale: float):
            self._scale = scale
            self.set_size()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.app = App()
        self.register_key_patch()
        self._draw_roi = ImageDrawRoiController(self.view_box, self)
        self._draw_roi.sigCreateRoi.connect(self.app.roi_dict.add_roi)
        self._draw_roi.sigMoveRoi.connect(self.app.roi_dict.move_roi)

        self._segments_hidden = False
        self._geometry_params_widget = None
        self.__init_center_roi()
        self.app.geometry_holder.sigScaleChanged.connect(self._on_scale_changed)
        self.app.geometry_holder.sigBeamCenterChanged.connect(self._on_beam_center_changed)
        self.app.image_holder.sigImageChanged.connect(self._on_image_changed)
        self.app.image_holder.sigEmptyImage.connect(self.clear_image)
        self.app.image_holder.sigCIFPeaksChanged.connect(self._draw_cif_result)

    def _on_scale_changed(self):
        scale = self.app.geometry.scale
        self.set_scale(scale)
        self.center_roi.set_scale(scale)

    def _on_image_changed(self):
        image = self.app.image
        if image is not None:
            self.set_data(image)

    def __init_center_roi(self):
        beam_center = tuple(self.app.geometry.beam_center)
        self.center_roi = self.BeamCenterRoi(beam_center, parent=self.image_item)
        self.center_roi.setZValue(10)
        self.image_plot.addItem(self.center_roi)
        self.center_roi.hide()

    def open_geometry_parameters(self):
        image = self.app.image
        scale = self.app.geometry.scale
        beam_center = tuple(self.app.geometry.beam_center)
        if image is not None and self._geometry_params_widget is None:
            self._geometry_params_widget = GeometryParametersWidget(
                image.shape, beam_center, scale=scale)
            self.center_roi.show()
            self._geometry_params_widget.change_center.connect(lambda x:
                                                               self.app.geometry_holder.set_beam_center(x, False))
            self._geometry_params_widget.scale_changed.connect(
                self.app.geometry_holder.set_scale)
            self._geometry_params_widget.close_event.connect(self._on_closing_geometry_parameters)

    def _on_closing_geometry_parameters(self):
        self.center_roi.set_size()
        self._geometry_params_widget = None
        self.app.geometry_holder.sigGeometryChangeFinished.emit()

    def _on_beam_center_changed(self):
        beam_center = self.app.geometry.beam_center
        self.set_center((beam_center.y, beam_center.z), pixel_units=True)


class ImageDrawRoiController(DrawRoiController):
    def _update_roi(self, point: QPointF):
        r1, r2 = self._init_point.x(), point.x()
        p1, p2 = self._init_point.y(), point.y()

        r1 = np.sqrt(r1 ** 2 + p1 ** 2)
        r2 = np.sqrt(r2 ** 2 + p2 ** 2)

        self._roi.radius, self._roi.width = (r1 + r2) / 2, abs(r2 - r1)


class GeometryParametersWidget(QWidget):
    change_center = pyqtSignal(list)
    change_zero_angle = pyqtSignal(float)
    change_invert_angle = pyqtSignal(bool)
    scale_changed = pyqtSignal(float)

    close_event = pyqtSignal()

    def __init__(self, image_shape: tuple,
                 beam_center: tuple, zero_angle: float = 0,
                 angle_direction: bool = True, scale: float = 1):
        super(GeometryParametersWidget, self).__init__(None, Qt.WindowStaysOnTopHint)
        self.beam_center = list(beam_center)
        self.image_shape = image_shape
        self.zero_angle = zero_angle
        self.scale = scale
        self.angle_direction = angle_direction
        self._init__ui()
        self.setWindowTitle('Set geometry')
        self.setWindowIcon(Icon('setup'))
        center_widget(self)
        self.show()

    def closeEvent(self, a0) -> None:
        self.close_event.emit()
        QWidget.closeEvent(self, a0)

    def _init__ui(self):
        layout = QVBoxLayout(self)

        self.x_slider = LabeledSlider('Y center', (0, self.image_shape[1]),
                                      self.beam_center[1], self, decimals=0)
        self.x_slider.valueChanged.connect(self._connect_func(1))

        self.y_slider = LabeledSlider('Z center', (0, self.image_shape[0]),
                                      self.beam_center[0], self, decimals=0)
        self.y_slider.valueChanged.connect(self._connect_func(0))

        # self.angle_slider = AnimatedSlider('Zero angle', (0, 360),
        #                                    self.zero_angle, self,
        #                                    Qt.Horizontal, disable_changing_status=True)
        # self.angle_slider.valueChanged.connect(self._connect_func(2))
        #
        # self.invert_angle_box = QCheckBox('Invert angle')
        # self.invert_angle_box.toggled.connect(self._connect_func(3))

        self.scale_edit = LabeledSlider('Q to pixel ratio', (1e-10, 10),
                                        self.scale, self,
                                        decimals=4)
        self.scale_edit.valueChanged.connect(self.on_scale_changed)

        layout.addWidget(self.x_slider)
        layout.addWidget(self.y_slider)
        layout.addWidget(self.scale_edit)
        # layout.addWidget(self.angle_slider)
        # layout.addWidget(self.invert_angle_box)

    def on_scale_changed(self, value):
        self.scale = value
        self.scale_changed.emit(value)

    def _connect_func(self, ind: int):
        def beam_center_changed(value):
            self.beam_center[ind] = value
            self.change_center.emit(self.beam_center)

        def angle_zero_changed(value):
            self.zero_angle = value
            self.change_zero_angle.emit(self.zero_angle)

        def angle_direction_changed(value):
            self.angle_direction = value
            self.change_invert_angle.emit(self.angle_direction)

        if ind < 2:
            return beam_center_changed
        elif ind == 2:
            return angle_zero_changed
        else:
            return angle_direction_changed

class Basic2DImageWidget(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.app = parent.app
        self.image_viewer = MainImageViewer(self)
        self.setCentralWidget(self.image_viewer)
        self._init_toolbar()

    def _add_q_range_edits(self, suffix, toolbar):
        label = QLabel('Q<sub>' + suffix +'</sub> (Å<sup>-1</sup>)')
        label.setStyleSheet("font: 10pt;")
        toolbar.addWidget(label)
        qlineedit = QLineEdit('2.7')
        qlineedit.setMaximumWidth(60)
        qlineedit.setMaxLength(4)
        qlineedit.setValidator(QDoubleValidator(0, 100, 2))
        toolbar.addWidget(qlineedit)
        return  qlineedit

    def _init_toolbar(self):
        toolbar = BlackToolBar('Geometry', self)
        self.addToolBar(toolbar)

        rotate_action = toolbar.addAction(Icon('rotate'), 'Rotate')
        rotate_action.triggered.connect(
            lambda: self.app.geometry_holder.add_transform(Transformation.rotate_right))

        flip_h = toolbar.addAction(Icon('flip_horizontal'), 'Horizontal flip')
        flip_h.triggered.connect(
            lambda: self.app.geometry_holder.add_transform(Transformation.horizontal_flip))

        flip_v = toolbar.addAction(Icon('flip_vertical'), 'Vertical flip')
        flip_v.triggered.connect(
            lambda: self.app.geometry_holder.add_transform(Transformation.vertical_flip))

        set_beam_center_action = toolbar.addAction(Icon('center'), 'Beam center')
        set_beam_center_action.triggered.connect(
            self.image_viewer.open_geometry_parameters)

        self._set_default_geometry_button = QPushButton('Save as default geometry')
        self._set_default_geometry_button.clicked.connect(self.app.geometry_holder.save_as_default)
        toolbar.addWidget(self._set_default_geometry_button)

        qz_range_edit = self._add_q_range_edits('z', toolbar)
        qz_range_edit.editingFinished.connect(
            lambda: self.app.fm.current_key.set_qz(float(qz_range_edit.text())))

        qxy_range_edit = self._add_q_range_edits('xy', toolbar)
        qxy_range_edit.editingFinished.connect(
            lambda: self.app.fm.current_key.set_qz(float(qxy_range_edit.text())))

        toolbar.addWidget(QLabel(''))
        self._refresh_sim = QPushButton('Refresh CIF-Simulation')
        self._refresh_sim.clicked.connect(lambda: self.app.fm.sigNewCIFFile.emit())
        toolbar.addWidget(self._refresh_sim)