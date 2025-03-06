# -*- coding: utf-8 -*-

from ..basic_widgets import PlotBC
from ...app.app import App
from ..roi_widgets.abstract_roi_holder import AbstractRoiHolder
from ..roi_widgets.roi_1d_widget import Roi1DAngular


class AngularProfileWidget(AbstractRoiHolder, PlotBC):
    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'AngularProfile')
        PlotBC.__init__(self, App().angular_profile, parent)
        self._selected_key = None

        # self.image_view.plot_item.setTitle('Angular Profile')
        self.image_view.plot_item.getAxis('bottom').setLabel('&Phi;', color='white', font_size='large')
        self.image_view.plot_item.getAxis('left').setLabel('Intensity', color='white', font_size='large')

    def _init_connect(self):
        App().image_holder.sigPolarImageChanged.connect(self._update)
        App().image_holder.sigEmptyImage.connect(self.clear_plot)
        self._roi_dict.sig_roi_created.connect(self._update)
        self._roi_dict.sig_roi_deleted.connect(self._update)
        self._roi_dict.sig_roi_moved.connect(self._update)
        self._roi_dict.sig_roi_moved.connect(self._move_rois)
        self._roi_dict.sig_selected.connect(self._update)
        self._roi_dict.sig_one_selected.connect(self._update)
        self._roi_dict.sig_fixed.connect(self._fix_rois)
        self._roi_dict.sig_unfixed.connect(self._unfix_rois)

    def _update(self):
        selected_rois = self._roi_dict.selected_rois

        if len(selected_rois) == 1:
            selected_roi = selected_rois[0]
            key = selected_roi.key
            if key == self._selected_key:
                self.profile.update()
            else:
                self._clear()
                self._selected_key = key
                self._create_roi((key, ))
                self.profile.update()
        else:
            self._clear()

    def _clear(self):
        if self._selected_key is not None:
            try:
                self._delete_roi_widget(self._roi_widgets.pop(self._selected_key))
            except KeyError:
                pass
            self._selected_key = None
        self.clear_plot()

    def _make_roi_widget(self, roi):
        roi_widget = Roi1DAngular(roi)
        self.image_view.plot_item.addItem(roi_widget)
        return roi_widget

    def _delete_roi_widget(self, roi_widget):
        self.image_view.plot_item.removeItem(roi_widget)
