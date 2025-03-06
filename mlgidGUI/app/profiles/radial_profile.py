# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np

from ..file_manager import FileManager, ImageKey
from .basic_profile import BasicProfile, BaselineParams


@dataclass
class SavedProfile:
    data: np.ndarray
    sigma: float
    baseline_params: BaselineParams
    baseline: np.ndarray = None


class RadialProfile(BasicProfile):
    def __init__(self, image_holder, fm: FileManager, parent=None):
        self.image_holder = image_holder
        self.fm = fm
        self._current_key: ImageKey = None
        super().__init__(parent)

    def save_state(self):
        if self._current_key and self._raw_y is not None:
            self.fm.profiles[self._current_key] = self.to_save()

    def update_data_from_source(self):
        # self.save_state()

        self._current_key = self.image_holder.current_key
        r_axis = self.image_holder.geometry.r_axis

        if r_axis is None:
            return

        self.set_data(self.image_holder.get_radial_profile(), r_axis)
