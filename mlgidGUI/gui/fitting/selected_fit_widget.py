# import logging
# from typing import List, Dict, Iterable, Union
# from enum import Enum, auto
# import gc
#
# import numpy as np
#
# from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton,
#                              QLabel, QFrame, QSplitter, QMenu,
#                              QMessageBox, QComboBox, QScrollArea,
#                              QSizePolicy)
# from PyQt5.QtGui import QColor, QCursor
# from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
#
# from ..roi_widgets.roi_2d_rect_widget import Roi2DRect
# from ..roi_widgets.roi_1d_widget import Roi1D
#
# from ...app.fitting import Fit, FitObject, FittingType, BackgroundType, RangeStrategyType, RangeStrategy
# from ...app.file_manager import ImageKey, FolderKey
# from ...app.rois.roi_data import Roi, RoiTypes
# from ...app import App
# from ..tools import Icon, get_pen, center_widget
# from ..basic_widgets import (Custom1DPlot, CustomImageViewer, PlotBC,
#                              ParametersSlider, LabeledSlider)
#
#
# class SelectedFitMode(Enum):
#     view = auto()
#     fit = auto()
#
#
# class SelectedFitWidget(QWidget):
#     sigFitChanged = pyqtSignal()
#
#     def __init__(self, fit: Fit = None,
#                  mode: SelectedFitMode = SelectedFitMode.view,
#                  parent=None, flags=None):
#         super().__init__(parent, flags=flags)
#         self.__fit: Fit or None = None
#         self.__mode: SelectedFitMode = mode
#
#         if fit:
#             self.set_fit(fit)
#
#
# class SelectedFitLabel(QLabel):
#     def __init__(self, fit: Fit = None, parent=None):
#         super().__init__(parent=parent)
#         self._fit = fit
#         self.update_text()
#
#     @pyqtSlot(name='updateText')
#     def update_text(self):
#         if not self._fit:
#             text = 'Selected Fit'
#         elif self._fit.roi.type == RoiTypes.ring:
#             text = f'Selected Fit: Ring {self._fit.roi.name}'
#         else:
#             text = f'Selected Fit: Segment {self._fit.roi.name} (no baseline correction)'
#         self.setText(text)
#
#     @pyqtSlot(object, name='setFit')
#     def set_fit(self, fit: Fit):
#         self._fit = fit
#         self.update_text()
#
#     @pyqtSlot(name='removeFit')
#     def remove_fit(self):
#         self._fit = None
#         self.update_text()
#
#
# class SlidersWidget(QWidget):
#     sigValueChanged = pyqtSignal()
#
#     DEFAULT_LABEL = 'lower bound; init value; upper bound'
#
#     log = logging.getLogger(__name__)
#
#     def __init__(self, param_names: tuple = (), parent=None):
#         super().__init__(parent=parent)
#         self.fit: Fit = None
#         self.param_names = param_names
#
#         self._sliders: Dict[int, ParametersSlider] = {}
#         self._labels: Dict[int, QLabel] = {}
#         self._param_labels: Dict[int, QLabel] = {}
#
#         self._init_ui()
#
#     @pyqtSlot(object, name='setFit')
#     def set_fit(self, fit: Fit):
#         self.fit = fit
#         self._set_param_names(fit.param_names)
#         self.update_values()
#
#     @pyqtSlot(name='updateFit')
#     def update_fit(self):
#         if self.fit:
#             self._set_param_names(self.fit.param_names)
#             self.update_values()
#
#     @pyqtSlot(name='updateValues')
#     def update_values(self):
#         if not self.fit:
#             return
#         fit = self.fit
#         for idx, (l, i, u) in enumerate(zip(fit.lower_bounds, fit.init_params, fit.upper_bounds)):
#             try:
#                 self._sliders[idx].setValues(l, i, u, new_range=(l, u))
#             except ValueError as err:
#                 self.log.error(f'{self._param_labels[idx].text()} received wrong values.')
#                 self.log.exception(err)
#             self._update_label(idx, l, i, u)
#
#     def remove_fit(self):
#         self.fit = None
#         for l in self._labels.values():
#             l.setText(self.DEFAULT_LABEL)
#
#     def _set_param_names(self, param_names: tuple):
#         if param_names == self.param_names:
#             return
#
#         i = 0
#         for i, name in enumerate(param_names):
#             if i == len(self._param_labels):
#                 self._add_parameter(i, name)
#             else:
#                 self._param_labels[i].setText(name)
#
#         if i < len(self.param_names) - 1:
#             for j in range(i + 1, len(self.param_names)):
#                 self._remove_parameter(j)
#
#         self.param_names = param_names
#
#     def _init_ui(self):
#         self._layout = QGridLayout(self)
#
#         for i, param_name in enumerate(self.param_names):
#             self._add_parameter(i, param_name)
#
#     def _add_parameter(self, i: int, param_name: str):
#         self._param_labels[i] = pl = QLabel(param_name)
#         self._sliders[i] = sl = ParametersSlider()
#         self._labels[i] = ll = QLabel(self.DEFAULT_LABEL)
#         self._layout.addWidget(pl, i, 0, alignment=Qt.AlignCenter)
#         self._layout.addWidget(_SliderWithLabels(sl, ll, self), i, 1)
#         sl.sigLowerValueChanged.connect(lambda x, idx=i: self._send_value(x, 0, idx))
#         sl.sigMiddleValueChanged.connect(lambda x, idx=i: self._send_value(x, 1, idx))
#         sl.sigUpperValueChanged.connect(lambda x, idx=i: self._send_value(x, 2, idx))
#
#         if App().debug_tracker:
#             App().debug_tracker.add_object(pl)
#             App().debug_tracker.add_object(sl)
#             App().debug_tracker.add_object(ll)
#
#     def _remove_parameter(self, i: int):
#         pl = self._param_labels.pop(i)
#         sl = self._sliders.pop(i)
#         ll = self._labels.pop(i)
#         self._layout.removeWidget(pl)
#         self._layout.removeWidget(sl)
#         self._layout.removeWidget(ll)
#         pl.deleteLater()
#         sl.deleteLater()
#         ll.deleteLater()
#
#     def _update_label(self, idx: int, l: float, i: float, u: float):
#         self._labels[idx].setText(f'lower bound = {l:0.3f}; init value = {i:0.3f}; upper bound = {u:0.3f}')
#
#     def _send_value(self, value: float, place: int, idx: int):
#         if not self.fit:
#             return
#         if place == 0:
#             params = self.fit.lower_bounds
#         elif place == 1:
#             params = self.fit.init_params
#         else:
#             params = self.fit.upper_bounds
#         params[idx] = value
#
#         self._update_label(
#             idx, self.fit.lower_bounds[idx], self.fit.init_params[idx], self.fit.upper_bounds[idx])
#
#         self.sigValueChanged.emit()
#
#
# class CurrentFitButtonStatus(Enum):
#     fit = 'Fit roi'
#     unfix = 'Unfix'
#
#
# class FitOptionsWidget(QWidget):
#     def __init__(self, fit: Fit = None, parent=None):
#         super().__init__(parent)
#         self.fit = None
#         self._init_ui()
#         if fit:
#             self.set_fit(fit)
#
#     def _init_ui(self):
#         self.functions_box = QComboBox(self)
#         self.background_box = QComboBox(self)
#         self.range_strategies_box = QComboBox(self)
#
#         self.range_factor_slider = LabeledSlider('Y range factor', (0, 10), parent=self, decimals=2)
#         self.range_factor_slider.valueChanged.connect(self._on_range_slider_moved)
#         self.sigma_slider = LabeledSlider('Sigma', (0, 10), parent=self, decimals=2)
#         self.sigma_slider.valueChanged.connect(self._on_sigma_slider_moved)
#         self.fit_current_button = QPushButton(CurrentFitButtonStatus.fit.value)
#         self.fit_current_button.clicked.connect(self._fit_current_clicked)
#         self.update_data_button = QPushButton('Update fit')
#
#         self.functions_box.addItem('Fitting functions')
#         for t in FittingType:
#             self.functions_box.addItem(t.value)
#         self.background_box.addItem('Backgrounds')
#         for t in BackgroundType:
#             self.background_box.addItem(t.value)
#         self.range_strategies_box.addItem('Range strategy')
#         for t in RangeStrategyType:
#             self.range_strategies_box.addItem(t.value)
#
#     @pyqtSlot(object, name='setFit')
#     def set_fit(self, fit: Fit):
#         pass
#
#
# class FitPlot(Custom1DPlot):
#     log = logging.getLogger(__name__)
#
#     def __init__(self, parent=None):
#         super().__init__(parent=parent)
#         self.fit: Fit or None = None
#         self.fit_plot = self.plot_item.plot()
#         self.background_plot = self.plot_item.plot()
#         self.fit_plot.setPen(get_pen(color='red', style=Qt.DashDotDotLine, width=4))
#         self.background_plot.setPen(get_pen(color='blue', style=Qt.DashDotDotLine, width=4))
#
#     @pyqtSlot(object, name='setFit')
#     def set_fit(self, fit: Fit):
#         self.fit = fit
#         self.update_plot()
#
#     @pyqtSlot(name='updatePlot')
#     def update_plot(self):
#         if not self.fit:
#             return
#         try:
#             self.plot.setData(self.fit.x, self.fit.y)
#             self.fit_plot.setData(self.fit.x, self.fit.init_curve)
#             if self.fit.background_curve is not None:
#                 self.background_plot.setData(self.fit.x, self.fit.background_curve)
#             else:
#                 self.background_plot.clear()
#             self.plot_item.autoRange()
#
#         except Exception as err:
#             self.log.exception(err)
#             self._clear_plot()
#
#     @pyqtSlot(name='removeFit')
#     def remove_fit(self):
#         self.fit = None
#         self._clear_plot()
#
#     def _clear_plot(self):
#         self.plot.clear()
#         self.fit_plot.clear()
#         self.background_plot.clear()
#         self.update()
#
#
# class _SliderWithLabels(QFrame):
#     def __init__(self, slider: ParametersSlider, label: QLabel, parent=None):
#         super().__init__(parent=parent)
#         self.slider = slider
#         self.label = label
#         self._init_ui()
#
#     def _init_ui(self):
#         layout = QGridLayout(self)
#         layout.addWidget(self.label, 0, 0, alignment=Qt.AlignHCenter)
#         layout.addWidget(self.slider, 1, 0)
