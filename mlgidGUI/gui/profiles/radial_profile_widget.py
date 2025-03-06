# -*- coding: utf-8 -*-

import logging

from scipy.signal import find_peaks

from PyQt5.QtGui import QColor
from PyQt5.QtCore import QPointF

from ..basic_widgets import (ConfirmButton, DrawRoiController,
                             RoundedPushButton, PlotBC,
                             BlackToolBar, BasicInputParametersWidget)
from ...app.app import App
from ..tools import Icon
from ..roi_widgets.abstract_roi_holder import AbstractRoiHolder
from ..roi_widgets.roi_1d_widget import Roi1D
from ..tools import show_error

logger = logging.getLogger(__name__)


class RadialProfileWidget(AbstractRoiHolder, PlotBC):

    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'RadialProfile')
        PlotBC.__init__(self, profile=App().radial_profile, parent=parent)
        self.app = App()
        self.app.geometry_holder.sigScaleChanged.connect(self._update_axis)
        self.app.image_holder.sigEmptyImage.connect(self.clear_plot)
        self._init_radial_toolbars()
        self._fit_params_dict: dict = PeaksSetupWindow.get_config()
        self._peaks_setup = None
        self._draw_roi = RadialDrawRoi(self.view_box, self)
        self._draw_roi.sigCreateRoi.connect(self.app.roi_dict.add_roi)
        self._draw_roi.sigMoveRoi.connect(self.app.roi_dict.move_roi)

        self.register_key_patch()

        # self.image_view.plot_item.setTitle('Radial Profile')
        self.image_view.plot_item.getAxis('bottom').setLabel('|Q|', color='white', font_size='large')
        self.image_view.plot_item.getAxis('left').setLabel('Intensity', color='white', font_size='large')

        self.sigBackgroundChanged.connect(self.profile.save_state)

    def _init_radial_toolbars(self):

        fit_toolbar = BlackToolBar('Fitting', self)
        self.addToolBar(fit_toolbar)

        find_peaks_widget = ConfirmButton(Icon('find'), text='Find peaks?')
        find_peaks_widget.label_widget.setStyleSheet(
            'QLabel { color : white ; }')
        find_peaks_widget.clicked.connect(self.find_peaks)
        fit_toolbar.addWidget(find_peaks_widget)
        #
        # fit_peaks_widget = ConfirmButton(Icon('fit'), text='Fit selected peaks?')
        # fit_peaks_widget.label_widget.setStyleSheet(
        #     'QLabel { color : white ; }')
        # fit_peaks_widget.clicked.connect(self.fit_selected)
        # fit_toolbar.addWidget(fit_peaks_widget)
        #
        setup_action = fit_toolbar.addAction(Icon('setup'), 'Fit setup')
        setup_action.triggered.connect(self.open_peaks_setup)

        segments_toolbar = BlackToolBar('Segments', self)
        self.addToolBar(segments_toolbar)

        create_roi_widget = RoundedPushButton(
            icon=Icon('add'), radius=30)
        create_roi_widget.clicked.connect(lambda: self.app.roi_dict.create_roi())
        segments_toolbar.addWidget(create_roi_widget)

        delete_selected_widget = ConfirmButton(
            Icon('delete'), text='Delete selected roi?')
        delete_selected_widget.clicked.connect(self.app.roi_dict.delete_selected_roi)
        delete_selected_widget.label_widget.setStyleSheet(
            'QLabel { color : white ; }')

        segments_toolbar.addWidget(delete_selected_widget)
        fix_all = RoundedPushButton(icon=Icon('fix_all'), radius=120, background_color=QColor(0, 0, 0, 0))
        fix_all.setFixedWidth(60)
        fix_all.setFixedHeight(30)
        fix_all.clicked.connect(self.app.roi_dict.fix_all)
        segments_toolbar.addWidget(fix_all)
        unfix_all = RoundedPushButton(icon=Icon('unfix_all'), radius=120, background_color=QColor(0, 0, 0, 0))
        unfix_all.setFixedWidth(60)
        unfix_all.setFixedHeight(30)
        unfix_all.clicked.connect(self.app.roi_dict.unfix_all)
        segments_toolbar.addWidget(unfix_all)

    # def _on_scale_changed(self):
    #     self.update_x_axis()
    #
    #
    def _update_axis(self):
        self.x = App().geometry.r_axis

    def find_peaks(self):
        if self.y is None or self.x is None:
            return
        if self._fit_params_dict.get('sigma_find', None) is not None:
            self.set_sigma(self._fit_params_dict['sigma_find'])
        peaks = find_peaks(self.y)[0]
        max_num = self._fit_params_dict.get('max_peaks_number', 40)
        if len(peaks) > max_num:
            show_error(
                'Maximum number of peaks exceeded.',
                error_title='Error',
                info_text=f'Number of found peaks ({len(peaks)}) exceeds the maximum number {max_num}. '
                          f'Consider increasing sigma for smoothing the radial profile.')
            return
        w = self._fit_params_dict.get('init_width', 30) * App().geometry.scale
        for r in peaks:
            self._roi_dict.create_roi(radius=self.x[r], width=w)

    def _make_roi_widget(self, roi):
        roi_widget = Roi1D(roi)
        self.image_view.plot_item.addItem(roi_widget)
        return roi_widget

    def _delete_roi_widget(self, roi_widget):
        self.image_view.plot_item.removeItem(roi_widget)

    def open_peaks_setup(self):
        self._peaks_setup = PeaksSetupWindow()
        self._peaks_setup.apply_signal.connect(self.set_fit_parameters)
        self._peaks_setup.close_signal.connect(self.close_peaks_setup)
        self._peaks_setup.show()

    def set_fit_parameters(self, params: dict):
        self._fit_params_dict = params

    def close_peaks_setup(self):
        self._peaks_setup = None


class RadialDrawRoi(DrawRoiController):
    def _update_roi(self, point: QPointF):
        r1, r2 = self._init_point.x(), point.x()
        self._roi.radius, self._roi.width = (r1 + r2) / 2, abs(r2 - r1)


class PeaksSetupWindow(BasicInputParametersWidget):
    P = BasicInputParametersWidget.InputParameters

    PARAMETER_TYPES = (P('max_peaks_number',
                         'Maximum number of peaks', int, 'Do not recommended to put high numbers'),
                       P('init_width', 'Peaks width', float,
                         'Gaussian fitting will start with this number'),
                       P('sigma_find', 'Sigma to find peaks', float,
                         'Default sigma value for gaussian smooth\n'
                         'applied before initial peaks finding to\n'
                         'avoid noise peaks. To use current lambda, \n'
                         'leave empty.', True),
                       P('sigma_fit', 'Sigma to find peaks', float,
                         'Default sigma value for gaussian smooth\n'
                         'applied before gaussian fitting of \n'
                         'found peaks. To use current lambda, \n'
                         'leave empty.', True)
                       )

    NAME = 'Fitting parameters'

    DEFAULT_DICT = dict(max_peaks_number=20, init_width=30, sigma_find=8)
