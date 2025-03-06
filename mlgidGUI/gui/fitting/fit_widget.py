# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Iterable, Union
from enum import Enum, auto
import gc

import numpy as np

from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton,
                             QLabel, QFrame, QSplitter, QMenu,
                             QMessageBox, QComboBox, QScrollArea,
                             QSizePolicy)
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from ..roi_widgets.roi_2d_rect_widget import Roi2DRect
from ..roi_widgets.roi_1d_widget import Roi1D

from ...app.fitting import Fit, FitObject, FittingType, BackgroundType, RangeStrategyType, RangeStrategy
from ...app.file_manager import ImageKey, FolderKey
from ...app.rois.roi_data import Roi, RoiTypes
from ...app.profiles import BasicProfile, SavedProfile
from ...app import App
from ..tools import Icon, get_pen, center_widget
from ..basic_widgets import (Custom1DPlot, CustomImageViewer, PlotBC,
                             ParametersSlider, LabeledSlider)

class FitImageButtonStatus(Enum):
    fit = 'Fit this image'
    unfix = 'Unfix all ROIs'


class CurrentFitButtonStatus(Enum):
    fit = 'Fit ROI'
    unfix = 'Unfix'


class MoveSource(Enum):
    polar_roi = auto()
    sliders = auto()
    radial_viewer = auto()
    change_fit_type = auto()
    change_roi_type = auto()
    change_range_strategy = auto()
    range_slider = auto()
    sigma_slider = auto()


class DeleteRegime(Enum):
    current = auto()
    following = auto()
    previous = auto()
    all = auto()


class FitWidget(QWidget):
    sigFitApplyActiveImage = pyqtSignal(object)

    log = logging.getLogger(__name__)

    def __init__(self, fit_object: FitObject, parent=None):
        super().__init__(parent=parent)
        self.setGeometry(0, 0, 1500, 700)
        self.setWindowTitle('Fitting parameters')
        self.setWindowIcon(Icon('fit'))
        self.setWindowFlag(Qt.Window, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowModality(Qt.WindowModal)
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow, True)

        self.fit_object: FitObject = fit_object
        self.active_image_key: ImageKey = fit_object.image_key
        self._selected_fit: Fit = None
        self._saved_selected_key: int = None
        self._rect_widgets: dict = {}

        self._init_ui()

        if self.fit_object:
            self._update_data()
            self._update_roi_widgets()
            self._update_fit_button()
            self._update_current_image_label()

            if len(list(self.fit_object.fits.keys())) == 1:
                self._selected_fit = self.fit_object.fits[list(self.fit_object.fits.keys())[0]]
                self._add_selected_fit()

        center_widget(self)
        self.setWindowState(Qt.WindowMaximized)

        if App().debug_tracker:
            App().debug_tracker.add_object(self)

        self.show()

    def _init_ui(self):
        layout = QGridLayout(self)

        self.polar_viewer = CustomImageViewer(parent=self)
        self.radial_viewer = RadialFitWidget(parent=self)
        self.fit_plot = FitPlot(parent=self)
        self.sliders_widget = SlidersWidget(parent=self)
        self.fit_button = QPushButton(FitImageButtonStatus.fit.value)
        self.apply_button = QPushButton('Apply')
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close)
        self.multi_fit_window: MultiFitWindow = MultiFitWindow(self.fit_object, self)

        self.multi_fit_window.sigFitUpdated.connect(self.set_fit)
        self.multi_fit_window.sigClosed.connect(self._close_multi_fit)
        self.fit_button.clicked.connect(self._fit_clicked)
        self.apply_button.clicked.connect(self.apply_results)
        self.radial_viewer.sigUpdateFit.connect(self._radial_roi_moved)
        self.radial_viewer.sigUpdateProfile.connect(self._profile_updated)
        self.sliders_widget.sigValueChanged.connect(self._sliders_changed)

        self.functions_box = QComboBox(self)
        self.background_box = QComboBox(self)
        self.range_strategies_box = QComboBox(self)

        self.set_as_default_button = QPushButton('Set as default options')
        self.set_as_default_button.clicked.connect(self._set_as_default)
        self.range_factor_slider = LabeledSlider('Y range factor', (0, 10), parent=self, decimals=2)
        self.range_factor_slider.valueChanged.connect(self._on_range_slider_moved)
        self.sigma_slider = LabeledSlider('Sigma', (0, 10), parent=self, decimals=2)
        self.sigma_slider.valueChanged.connect(self._on_sigma_slider_moved)
        self.fit_current_button = QPushButton(CurrentFitButtonStatus.fit.value)
        self.fit_current_button.clicked.connect(self._fit_current_clicked)
        self.update_data_button = QPushButton('Update fit')
        self.update_data_button.clicked.connect(self._update_data_button_clicked)

        self.functions_box.currentTextChanged.connect(self._change_function)
        self.background_box.currentTextChanged.connect(self._change_background)
        self.range_strategies_box.currentTextChanged.connect(self._change_range_strategy)

        self.functions_box.addItem('Fitting functions')
        for t in FittingType:
            self.functions_box.addItem(t.value)
        self.background_box.addItem('Backgrounds')
        for t in BackgroundType:
            self.background_box.addItem(t.value)
        self.range_strategies_box.addItem('Range strategy')
        for t in RangeStrategyType:
            self.range_strategies_box.addItem(t.value)

        self.selected_fit_label = QLabel('Selected Fit', self)
        self.current_image_label = QLabel('Current Image', self)

        single_image_widget = QWidget(self)

        q_splitter_h1 = QSplitter(orientation=Qt.Vertical, parent=single_image_widget)
        q_splitter_h1.addWidget(self.polar_viewer)
        q_splitter_h1.addWidget(self.radial_viewer)

        s_layout = QGridLayout(single_image_widget)
        s_layout.addWidget(self.current_image_label, 0, 0, 1, 3)
        s_layout.addWidget(q_splitter_h1, 1, 0, 1, 3)
        s_layout.addWidget(self.fit_button, 2, 0)
        s_layout.addWidget(self.apply_button, 2, 1)
        s_layout.addWidget(self.close_button, 2, 2)

        single_roi_widget = QWidget(self)

        q_scroll_area = QScrollArea(self)
        q_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        q_scroll_area.setWidgetResizable(True)
        q_scroll_area.setGeometry(0, 0, 300, 400)

        options_widget = QWidget(self)
        options_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        scroll_layout = QGridLayout(options_widget)
        scroll_layout.addWidget(self.background_box, 0, 0)
        scroll_layout.addWidget(self.functions_box, 1, 0)
        scroll_layout.addWidget(self.range_strategies_box, 2, 0)
        scroll_layout.addWidget(self.range_factor_slider, 3, 0)

        scroll_layout.addWidget(self.set_as_default_button, 0, 1)
        scroll_layout.addWidget(self.fit_current_button, 1, 1)
        scroll_layout.addWidget(self.update_data_button, 2, 1)
        scroll_layout.addWidget(self.sigma_slider, 3, 1)
        scroll_layout.addWidget(self.sliders_widget, 4, 0, 2, 2)

        q_scroll_area.setWidget(options_widget)

        q_splitter_h2 = QSplitter(orientation=Qt.Vertical, parent=self)
        q_splitter_h2.addWidget(self.fit_plot)
        q_splitter_h2.addWidget(q_scroll_area)
        # q_splitter_h2.addWidget(self.sliders_widget)

        sr_layout = QGridLayout(single_roi_widget)
        sr_layout.addWidget(self.selected_fit_label, 0, 0)
        sr_layout.addWidget(q_splitter_h2, 1, 0)

        q_splitter_v = QSplitter(orientation=Qt.Horizontal, parent=self)
        q_splitter_v.addWidget(self.multi_fit_window)
        q_splitter_v.addWidget(single_image_widget)
        q_splitter_v.addWidget(single_roi_widget)

        q_splitter_v.setSizes((300, 600, self.width() - 900))
        q_splitter_h1.setSizes((400, self.height() - 400))
        q_splitter_h2.setSizes((400, self.height() - 400))
        layout.addWidget(q_splitter_v, 0, 0, 2, 2)

    @pyqtSlot(name='closeMultiFit')
    def _close_multi_fit(self):
        self.multi_fit_window = None

    @property
    def current_image_key(self):
        return self.fit_object.image_key if self.fit_object else None

    @pyqtSlot(object, name='setFit')
    def set_fit(self, fit_object: FitObject):
        self.fit_object = fit_object

        if self._selected_fit:
            key = self.selected_key
        else:
            key = self._saved_selected_key
        selected_fit = self.fit_object.fits.get(key, None)

        self._update_data()
        self._update_roi_widgets()

        if selected_fit:
            self._selected_fit = selected_fit
            self._add_selected_fit()
            self._saved_selected_key = None
        else:
            self._remove_selected_fit()
            self._saved_selected_key = key

        self._update_fit_button()
        self._update_current_image_label()
        gc.collect()

    def _update_data(self):
        self.polar_viewer.set_data(self.fit_object.polar_image)
        self.polar_viewer.set_x_axis(np.nanmin(self.fit_object.r_axis), np.nanmax(self.fit_object.r_axis))
        self.polar_viewer.set_y_axis(np.nanmin(self.fit_object.phi_axis), np.nanmax(self.fit_object.phi_axis))
        self.polar_viewer.view_box.setAspectLocked(True, self.fit_object.aspect_ratio)
        self.polar_viewer.set_auto_range()
        self.radial_viewer.update_data(self.fit_object.r_axis, self.fit_object.r_profile,
                                       self.current_image_key, self.fit_object.saved_profile)

    def _update_roi_widgets(self):
        for fit in self.fit_object.fits.values():
            roi = fit.roi
            roi.active = False
            if roi.key in self._rect_widgets:
                self._rect_widgets[roi.key].set_roi(roi)
            else:
                rect_widget = Roi2DRect(roi, context_menu=self._rect_context)
                rect_widget.sigSelected.connect(self._on_rect_roi_selected)
                rect_widget.sigRoiMoved.connect(self._roi_moved)
                self.polar_viewer.image_plot.addItem(rect_widget)
                self._rect_widgets[roi.key] = rect_widget
            if fit.fitted_params:
                self._fix(fit.roi)
            else:
                self._unfix(fit.roi)

        new_fit_keys = list(self.fit_object.fits.keys())
        for key in list(self._rect_widgets.keys()):
            if key not in new_fit_keys:
                widget = self._rect_widgets.pop(key)
                self.polar_viewer.image_plot.removeItem(widget)

    def _rect_context(self, roi: Roi):
        menu = QMenu()
        delete_menu = menu.addMenu('Delete')
        delete_menu.addAction('Delete on this image',
                              lambda *x, r=roi: self._delete_rois(roi, DeleteRegime.current))
        delete_menu.addAction('Delete on all the following images',
                              lambda *x, r=roi: self._delete_rois(roi, DeleteRegime.following))
        delete_menu.addAction('Delete everywhere',
                              lambda *x, r=roi: self._delete_rois(roi, DeleteRegime.all))
        if roi.movable:
            menu.addAction('Fix', lambda *x, r=roi: self._fix(roi))
            menu.addAction('Fit', lambda *x, r=roi: self._fit(roi))
        else:
            menu.addAction('Unfix', lambda *x, r=roi: self._unfix(roi))
        if not roi.type == RoiTypes.segment:
            menu.addAction(f'Change to segment type',
                           lambda *x, r=roi: self._change_roi_type(roi, RoiTypes.segment))
        if not roi.type == RoiTypes.ring:
            menu.addAction(f'Change to ring type',
                           lambda *x, r=roi: self._change_roi_type(roi, RoiTypes.ring))
        if not roi.type == RoiTypes.background:
            menu.addAction(f'Change to background type',
                           lambda *x, r=roi: self._change_roi_type(roi, RoiTypes.background))
        menu.exec_(QCursor.pos())

    def _delete_rois(self, roi: Roi, delete_regime: DeleteRegime):
        if delete_regime.value == DeleteRegime.current.value:
            image_keys: List[ImageKey] = [self.fit_object.image_key]
        elif delete_regime.value == DeleteRegime.following.value:
            image_keys: List[ImageKey] = [key for key in self.fit_object.image_key.parent.image_children if
                                          key.idx > self.fit_object.image_key.idx]
        elif delete_regime.value == DeleteRegime.all.value:
            image_keys: List[ImageKey] = [key for key in self.fit_object.image_key.parent.image_children]
        else:
            raise ValueError('Unknown delete regime used.')

        self.multi_fit_window.delete_rois(image_keys, roi.key)

        if self.current_image_key in image_keys:
            if roi.key == self.selected_key:
                self._remove_selected_fit()
            del self.fit_object.fits[roi.key]
            self.polar_viewer.image_plot.removeItem(self._rect_widgets.pop(roi.key))

    def _change_roi_type(self, roi: Roi, t: RoiTypes):
        roi.type = t

        if not roi.has_fixed_angles():
            roi.angle, roi.angle_std = self.fit_object.bounds

        if roi.key == self.selected_key:
            self._basic_update(MoveSource.change_roi_type)
        else:
            fit = self.fit_object.fits[roi.key]
            self.fit_object.update_fit_data(fit, update_fit=True)
            self._rect_widgets[roi.key].move_roi()
        self._update_selected_fit_label()

    def _update_current_image_label(self):
        self.current_image_label.setText(f'Current Image: {self.current_image_key.name}'
                                         f' ({self.current_image_key.idx})')

    def _update_selected_fit_label(self):
        if not self._selected_fit:
            text = 'Selected Fit'
        elif self._selected_fit.roi.type == RoiTypes.ring:
            text = f'Selected Fit: Ring {self._selected_fit.roi.name}'
        elif self._selected_fit.roi.type == RoiTypes.segment:
            text = f'Selected Fit: Segment {self._selected_fit.roi.name} (no baseline correction)'
        elif self._selected_fit.roi.type == RoiTypes.background:
            text = f'Selected Fit: Background {self._selected_fit.roi.name}'
        else:
            raise TypeError(f'Unknown roi type {self._selected_fit.roi.type}.')
        self.selected_fit_label.setText(text)

    def _update_combo_boxes(self):
        self.background_box.setCurrentIndex(0)
        self.functions_box.setCurrentIndex(0)
        self.range_strategies_box.setCurrentIndex(0)
        if not self._selected_fit:
            self.background_box.setItemText(0, 'Backgrounds')
            self.functions_box.setItemText(0, 'Fitting functions')
            self.range_strategies_box.setItemText(0, 'Range strategy')
        else:
            self.background_box.setItemText(0, f'Current background: {self._selected_fit.background.TYPE.value}')
            self.functions_box.setItemText(0, f'Current function: {self._selected_fit.fitting_function.TYPE.value}')
            self.range_strategies_box.setItemText(
                0, f'Range strategy: {self._selected_fit.range_strategy.strategy_type.value}')

    @property
    def selected_key(self):
        return self._selected_fit.roi.key if self._selected_fit else None

    @pyqtSlot(name='changeFittingFunction')
    def _change_function(self):
        if not self.functions_box.currentIndex():
            return

        if self._selected_fit:
            if not self._selected_fit.roi.movable:
                self._unfix(self._selected_fit.roi)
            self.fit_object.set_function(FittingType(self.functions_box.currentText()), fit=self._selected_fit)
            self._basic_update(MoveSource.change_fit_type)
            self._update_combo_boxes()

    @pyqtSlot(name='changeRangeStrategy')
    def _change_range_strategy(self):
        if not self.range_strategies_box.currentIndex():
            return
        if self._selected_fit:
            if not self._selected_fit.roi.movable:
                self._unfix(self._selected_fit.roi)
            range_type: RangeStrategyType = RangeStrategyType(self.range_strategies_box.currentText())
            range_factor: float = self.range_factor_slider.value
            strategy = RangeStrategy(range_factor, range_type, False)
            self.fit_object.set_range_strategy(strategy, fit=self._selected_fit)
            self._basic_update(MoveSource.change_range_strategy)
            self._update_combo_boxes()

    @pyqtSlot(name='changeBackground')
    def _change_background(self):
        if not self.background_box.currentIndex():
            return
        if self._selected_fit:
            if not self._selected_fit.roi.movable:
                self._unfix(self._selected_fit.roi)
            self.fit_object.set_background(BackgroundType(self.background_box.currentText()), fit=self._selected_fit)
            self._basic_update(MoveSource.change_fit_type)
            self._update_combo_boxes()

    @pyqtSlot(name='setAsDefault')
    def _set_as_default(self):
        if self._selected_fit:
            self.fit_object.set_function(self._selected_fit.fitting_function.TYPE, update=False)
            self.fit_object.set_background(self._selected_fit.background.TYPE, update=False)
            self.fit_object.set_range_strategy(self._selected_fit.range_strategy, update=True)

    @pyqtSlot(float, name='rangeSliderMoved')
    def _on_range_slider_moved(self, value: float):
        if self._selected_fit:
            if not self._selected_fit.roi.movable:
                self._unfix(self._selected_fit.roi)
            if self._selected_fit.range_strategy.strategy_type.value == RangeStrategyType.fixed.value:
                self._selected_fit.range_strategy.strategy_type = RangeStrategyType.adjust
                self._update_combo_boxes()
            self._selected_fit.range_strategy.range_factor = value
            self.fit_object.update_fit_data(self._selected_fit, update_r_range=True)
            self._basic_update(MoveSource.range_slider)

    @pyqtSlot(float, name='onSigmaSliderMoved')
    def _on_sigma_slider_moved(self, sigma: float):
        if self._selected_fit:
            self.fit_object.set_sigma(sigma, self._selected_fit)
            self._basic_update(MoveSource.sigma_slider)

    def _update_range_slider(self):
        if self._selected_fit:
            self.range_factor_slider.set_value(self._selected_fit.range_strategy.range_factor, True)
        else:
            self.range_factor_slider.set_value(0)

    def _update_sigma_slider(self):
        if not self._selected_fit or self._selected_fit.sigma is None:
            self.sigma_slider.set_value(self.fit_object.default_sigma)
        else:
            self.sigma_slider.set_value(self._selected_fit.sigma)

    @pyqtSlot(name='updateDataButtonClicked')
    def _update_data_button_clicked(self):
        if self._selected_fit:
            if not self._selected_fit.roi.movable:
                self._unfix(self._selected_fit.roi)
            self.fit_object.update_fit_data(self._selected_fit, update_fit=True)
            self._basic_update(MoveSource.sliders)

    @pyqtSlot(name='currentFitButtonClicked')
    def _fit_current_clicked(self):
        if self._selected_fit:
            if self.fit_current_button.text() == CurrentFitButtonStatus.fit.value:
                self._fit(self._selected_fit.roi)
            else:
                self._unfix(self._selected_fit.roi)
            self._update_current_fit_button()

    def _update_current_fit_button(self):
        if self._selected_fit:
            if self._selected_fit.roi.movable:
                self.fit_current_button.setText(CurrentFitButtonStatus.fit.value)
            else:
                self.fit_current_button.setText(CurrentFitButtonStatus.unfix.value)

    @pyqtSlot(object, name='profileUpdated')
    def _profile_updated(self, saved_profile: SavedProfile):
        self.fit_object.set_profile(saved_profile=saved_profile)
        if self._selected_fit:
            self.fit_plot.update_plot()
        self._update_sigma_slider()

    @pyqtSlot(bool, name='updateFit')
    def _radial_roi_moved(self, r_range_changed: bool):
        if r_range_changed:
            self.fit_object.update_fit_data(self._selected_fit, update_r_range=False)
            if self._selected_fit.range_strategy.strategy_type.value == RangeStrategyType.adjust.value:
                self._selected_fit.range_strategy.strategy_type = RangeStrategyType.fixed
                self._update_combo_boxes()
        self._basic_update(MoveSource.radial_viewer)

    def _basic_update(self, source: MoveSource):
        if source == MoveSource.polar_roi or source == MoveSource.change_roi_type:
            self.fit_object.update_fit_data(self._selected_fit)

        if source == MoveSource.change_fit_type:
            self.sliders_widget.set_param_names(self._selected_fit.param_names)
            self.sliders_widget.update_values()

        elif source != MoveSource.sliders:
            self._selected_fit.update_fit(True)
            self.sliders_widget.update_values()
        else:
            self._selected_fit.update_fit(False)

        if source != MoveSource.polar_roi:
            self._rect_widgets[self.selected_key].move_roi()

        if source != MoveSource.radial_viewer:
            self.radial_viewer.update_roi()

        # if source != MoveSource.range_slider:
        #     self._update_range_slider()

        self.fit_plot.update_plot()
        self.multi_fit_window.update_fit(self._selected_fit)

    @pyqtSlot(int, name='roiMoved')
    def _roi_moved(self, key: int):
        if key != self.selected_key:
            self._on_rect_roi_selected(key)
        roi = self._selected_fit.roi
        if roi.should_adjust_angles(*self.fit_object.bounds):
            roi.type = RoiTypes.segment
            self._update_selected_fit_label()

        self._basic_update(MoveSource.polar_roi)

    @pyqtSlot(name='slidersMoved')
    def _sliders_changed(self):
        self._basic_update(MoveSource.sliders)

    def _update_fit_button(self):
        if self.fit_object:
            if self.fit_object.is_fitted:
                self.fit_button.setText(FitImageButtonStatus.unfix.value)
            else:
                self.fit_button.setText(FitImageButtonStatus.fit.value)

    @pyqtSlot(name='fitButtonClicked')
    def _fit_clicked(self):
        if self.fit_button.text() == FitImageButtonStatus.fit.value:
            self._fit()
            self.fit_object.is_fitted = True
            self.fit_button.setText(FitImageButtonStatus.unfix.value)
        else:
            self._unfix()
            self.fit_object.is_fitted = False
            self.fit_button.setText(FitImageButtonStatus.fit.value)

    def _unfix(self, roi: Union[Roi, Iterable[int]] = None):
        if isinstance(roi, Roi):
            widgets = (self._rect_widgets[roi.key],)
        elif roi:
            widgets = (self._rect_widgets[key] for key in roi)
        else:
            widgets = self._rect_widgets.values()

        for widget in widgets:
            roi = widget.roi
            roi.movable = True
            fit = self.fit_object.fits[roi.key]
            fit.fitted_params = []
            widget.unfix()
            if roi.key == self.selected_key:
                self.radial_viewer.roi_widget.unfix()
                self._update_current_fit_button()

        if all(fit.roi.movable for fit in self.fit_object.fits.values()):
            self.fit_object.is_fitted = False
            self._update_fit_button()

    def _fix(self, roi: Union[Roi, Iterable[int]] = None):
        if isinstance(roi, Roi):
            widgets = (self._rect_widgets[roi.key],)
        elif roi:
            widgets = (self._rect_widgets[key] for key in roi)
        else:
            widgets = self._rect_widgets.values()

        for widget in widgets:
            roi = widget.roi
            roi.movable = False
            widget.fix()
            if roi.key == self.selected_key:
                self.radial_viewer.roi_widget.fix()
                self._update_current_fit_button()

        if not any(fit.roi.movable for fit in self.fit_object.fits.values()):
            self.fit_object.is_fitted = True
            self._update_fit_button()

    def _fit(self, roi: Union[Roi, Iterable[int]] = None):
        if isinstance(roi, Roi):
            fits = (self.fit_object.fits[roi.key],)
        elif roi:
            fits = (self.fit_object.fits[key] for key in roi)
        else:
            fits = self.fit_object.fits.values()

        keys_to_fix: List[int] = []

        for fit in fits:
            if not fit.roi.movable:
                continue

            fit.do_fit()
            roi_ = fit.roi
            key = roi_.key
            if fit.fitted_params:
                keys_to_fix.append(key)

            if key == self.selected_key:
                self.fit_plot.update_plot()
                self.radial_viewer.update_roi()
                self.sliders_widget.update_values()

            self.multi_fit_window.update_fit(fit)
            self._rect_widgets[key].move_roi()

        self._fix(keys_to_fix)

    def _on_rect_roi_selected(self, key: int):
        if key == self.selected_key:
            return
        else:
            self._remove_selected_fit()
            self._selected_fit = self.fit_object.fits[key]
            self._add_selected_fit()

    def _add_selected_fit(self):
        roi = self._selected_fit.roi
        roi.active = True
        self._rect_widgets[roi.key].update_color()
        self.radial_viewer.set_fit(self._selected_fit)
        self.fit_plot.set_fit(self._selected_fit)
        self.sliders_widget.set_fit(self._selected_fit)
        self.multi_fit_window.select_fit(self.selected_key)
        self._update_combo_boxes()
        self._update_range_slider()
        self._update_sigma_slider()
        self._update_current_fit_button()
        self._update_selected_fit_label()

    def _remove_selected_fit(self):
        if self._selected_fit:
            roi = self._selected_fit.roi
            roi.active = False

            try:
                self._rect_widgets[roi.key].update_color()
            except KeyError:
                pass

            self.fit_plot.remove_fit()
            self.radial_viewer.remove_fit()
            self.sliders_widget.remove_fit()
            self.multi_fit_window.unselect_fit(self.selected_key)
            self._selected_fit = None
            self._update_combo_boxes()
            self._update_range_slider()
            self._update_sigma_slider()
            self._update_selected_fit_label()

    @pyqtSlot(name='applyResults')
    def apply_results(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Apply fitting')
        msg_box.setWindowIcon(Icon('fit'))
        msg_box.setText("Do you want to apply fits from this image or from all the fitted images?")
        msg_box.setInformativeText("Only fixed fits will be applied.")
        all_btn = msg_box.addButton('All images', QMessageBox.YesRole)
        current_btn = msg_box.addButton('Current image', QMessageBox.YesRole)
        msg_box.addButton(QMessageBox.Cancel)
        msg_box.setDefaultButton(all_btn)
        ret = msg_box.exec()

        if msg_box.clickedButton() == current_btn:
            self._apply_current_fit()
        elif msg_box.clickedButton() == all_btn:
            self._apply_all_fits()

    def _apply_current_fit(self):
        self.log.info('Apply current fit selected')

        if self.fit_object.image_key == self.active_image_key:
            self.sigFitApplyActiveImage.emit(self.fit_object)
            self.multi_fit_window.save_fits([])
        else:
            self.multi_fit_window.save_fits([self.fit_object.image_key])

    def _apply_all_fits(self):
        self.log.info('Apply all fits selected')

        parent_key: FolderKey = self.active_image_key.parent
        image_list = []

        for image_key in parent_key.image_children:
            if image_key == self.active_image_key:
                self.sigFitApplyActiveImage.emit(self.fit_object)
            else:
                image_list.append(image_key)

        self.multi_fit_window.save_fits(image_list)

    def closeEvent(self, a0) -> None:
        msg_box = QMessageBox()
        msg_box.setWindowTitle('Closing the fitting window')
        msg_box.setText("The fitting results will be deleted if you have not applied them yet.")
        msg_box.setInformativeText("Do you want to apply the results and add them to your project? "
                                   "Select Close if you already saved everything.")
        msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Save)
        ret = msg_box.exec()
        if ret == QMessageBox.Cancel:
            a0.ignore()
            return
        if ret == QMessageBox.Save:
            self.apply_results()
            a0.ignore()
            return
        else:
            self.multi_fit_window.close_widget()
            super().closeEvent(a0)


class SliderWithLabels(QFrame):
    def __init__(self, slider: ParametersSlider, label: QLabel, parent=None):
        super().__init__(parent=parent)
        self.slider = slider
        self.label = label
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, alignment=Qt.AlignHCenter)
        layout.addWidget(self.slider, 1, 0)


class SlidersWidget(QWidget):
    sigValueChanged = pyqtSignal()
    DEFAULT_LABEL = 'lower bound; init value; upper bound'

    log = logging.getLogger(__name__)

    def __init__(self, param_names: tuple = (), parent=None):
        super().__init__(parent=parent)
        self.fit: Fit = None
        self.param_names = param_names

        self._sliders: Dict[int, ParametersSlider] = {}
        self._labels: Dict[int, QLabel] = {}
        self._param_labels: Dict[int, QLabel] = {}

        self._init_ui()

    def _init_ui(self):
        self._layout = QGridLayout(self)

        for i, param_name in enumerate(self.param_names):
            self._add_parameter(i, param_name)

    def set_fit(self, fit: Fit):
        self.fit = fit
        self.set_param_names(fit.param_names)
        self.update_values()

    def _add_parameter(self, i: int, param_name: str):
        self._param_labels[i] = pl = QLabel(param_name)
        self._sliders[i] = sl = ParametersSlider()
        self._labels[i] = ll = QLabel(self.DEFAULT_LABEL)
        self._layout.addWidget(pl, i, 0, alignment=Qt.AlignCenter)
        self._layout.addWidget(SliderWithLabels(sl, ll, self), i, 1)
        sl.sigLowerValueChanged.connect(lambda x, idx=i: self._send_value(x, 0, idx))
        sl.sigMiddleValueChanged.connect(lambda x, idx=i: self._send_value(x, 1, idx))
        sl.sigUpperValueChanged.connect(lambda x, idx=i: self._send_value(x, 2, idx))

        if App().debug_tracker:
            App().debug_tracker.add_object(pl)
            App().debug_tracker.add_object(sl)
            App().debug_tracker.add_object(ll)

    def _remove_parameter(self, i: int):
        pl = self._param_labels.pop(i)
        sl = self._sliders.pop(i)
        ll = self._labels.pop(i)
        self._layout.removeWidget(pl)
        self._layout.removeWidget(sl)
        self._layout.removeWidget(ll)
        pl.deleteLater()
        sl.deleteLater()
        ll.deleteLater()

    def set_param_names(self, param_names: tuple):
        if param_names == self.param_names:
            return

        i = 0
        for i, name in enumerate(param_names):
            if i == len(self._param_labels):
                self._add_parameter(i, name)
            else:
                self._param_labels[i].setText(name)

        if i < len(self.param_names) - 1:
            for j in range(i + 1, len(self.param_names)):
                self._remove_parameter(j)

        self.param_names = param_names

    def update_values(self):
        if not self.fit:
            return
        fit = self.fit
        for idx, (l, i, u) in enumerate(zip(fit.lower_bounds, fit.init_params, fit.upper_bounds)):
            try:
                self._sliders[idx].setValues(l, i, u, new_range=(l, u))
            except ValueError as err:
                self.log.error(f'{self._param_labels[idx].text()} received wrong values.')
                self.log.exception(err)
            self._update_label(idx, l, i, u)

    def _update_label(self, idx: int, l: float, i: float, u: float):
        self._labels[idx].setText(f'lower bound = {l:0.3f}; init value = {i:0.3f}; upper bound = {u:0.3f}')

    def _send_value(self, value: float, place: int, idx: int):
        if not self.fit:
            return
        if place == 0:
            params = self.fit.lower_bounds
        elif place == 1:
            params = self.fit.init_params
        else:
            params = self.fit.upper_bounds
        params[idx] = value

        self._update_label(
            idx, self.fit.lower_bounds[idx], self.fit.init_params[idx], self.fit.upper_bounds[idx])

        self.sigValueChanged.emit()

    def remove_fit(self):
        self.fit = None
        for l in self._labels.values():
            l.setText(self.DEFAULT_LABEL)


class FittingProfile(BasicProfile):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_key = None
        self.profile_fm = App().fm.profiles

    def save_state(self):
        if self.current_key and self.raw_y is not None:
            saved_profile = self.to_save()
            self.profile_fm[self.current_key] = saved_profile
            return saved_profile

    def update_data_from_source(self, *args, **kwargs):
        x, y, self.current_key, saved_profile = args
        if not saved_profile:
            saved_profile = self.profile_fm[self.current_key] if self.current_key else None

        if not saved_profile or np.any(saved_profile.x != x):
            self.clear_baseline()
            self.set_data(y, x)

        else:
            self.from_save(saved_profile)


class RadialFitWidget(PlotBC):
    sigUpdateFit = pyqtSignal(bool)
    sigUpdateProfile = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(FittingProfile(), parent=parent)

        self.fit: Fit = None
        self.roi_widget: Roi1D = Roi1D(Roi(0, 0), enable_context=False)
        self.range_roi: Roi = Roi(radius=0, width=1, key=-1)
        self.range_widget: Roi1D = Roi1D(self.range_roi, enable_context=False)
        self.range_widget.set_color(QColor(255, 255, 255, 50))

        self.sigBackgroundChanged.connect(self._on_profile_changed)
        self.sigSigmaChanged.connect(self._on_profile_changed)

        self.range_widget.sigRoiMoved.connect(self._update_fit)
        self.roi_widget.sigRoiMoved.connect(self._update_fit)

        self.image_view.plot_item.addItem(self.range_widget)
        self.image_view.plot_item.addItem(self.roi_widget)

        self.roi_widget.hide()
        self.range_widget.hide()

    @pyqtSlot(name='onBackgroundChanged')
    def _on_profile_changed(self):
        self.sigUpdateProfile.emit(self.profile.save_state())

    @pyqtSlot(object, name='addFit')
    def set_fit(self, fit: Fit):
        self.fit = fit
        self.roi_widget.set_roi(fit.roi)
        r1, r2 = fit.r_range

        self.range_roi.radius = (r1 + r2) / 2
        self.range_roi.width = r2 - r1
        self.range_widget.move_roi()

        self.range_widget.show()
        self.roi_widget.show()

        self.image_view.plot_item.autoRange()

    def update_roi(self):
        if not self.fit:
            return
        self.roi_widget.move_roi()
        r1, r2 = self.fit.r_range
        self.range_roi.radius = (r1 + r2) / 2
        self.range_roi.width = r2 - r1
        self.range_widget.move_roi()

    def _update_fit(self, key: int):
        is_range_widget: bool = (key == -1)
        if is_range_widget:
            r, w = self.range_roi.radius, self.range_roi.width
            self.fit.r_range = r - w / 2, r + w / 2
        self.sigUpdateFit.emit(is_range_widget)

    @pyqtSlot(name='removeFit')
    def remove_fit(self):
        self.roi_widget.hide()
        self.range_widget.hide()
        self.fit = None


class FitPlot(Custom1DPlot):
    log = logging.getLogger(__name__)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.fit: Fit = None
        self.fit_plot = self.plot_item.plot()
        self.background_plot = self.plot_item.plot()
        self.fit_plot.setPen(get_pen(color='red', style=Qt.DashDotDotLine, width=4))
        self.background_plot.setPen(get_pen(color='blue', style=Qt.DashDotDotLine, width=4))

    def set_fit(self, fit: Fit):
        self.fit = fit
        self.update_plot()

    def update_plot(self):
        if not self.fit:
            return
        try:
            self.plot.setData(self.fit.x, self.fit.y)
            self.fit_plot.setData(self.fit.x, self.fit.init_curve)
            if self.fit.background_curve is not None:
                self.background_plot.setData(self.fit.x, self.fit.background_curve)
            else:
                self.background_plot.clear()
            self.plot_item.autoRange()

        except Exception as err:
            self.log.exception(err)
            self.clear_plot()

    def remove_fit(self):
        self.fit = None
        self.clear_plot()

    def clear_plot(self):
        self.plot.clear()
        self.fit_plot.clear()
        self.background_plot.clear()
        self.update()
