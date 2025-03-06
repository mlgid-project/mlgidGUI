# -*- coding: utf-8 -*-
import logging
from typing import Tuple
from dataclasses import dataclass, asdict
from abc import abstractmethod

import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject

from ..utils import baseline_correction, smooth_curve

logger = logging.getLogger(__name__)


@dataclass
class BaselineParams:
    smoothness: float = 100
    asymmetry: float = 0.01


@dataclass
class SavedProfile:
    raw_data: np.ndarray
    x: np.ndarray
    x_range: tuple
    sigma: float
    baseline_params: BaselineParams
    baseline: np.ndarray = None


class SmoothedProfile(QObject):
    sigSigmaChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sigma = 0
        self.baseline_params = BaselineParams()
        self.x_range: Tuple[float, float] or None = None
        self._x = self._y = self._raw_y = self._baseline = None

    @property
    def x(self) -> np.ndarray or None:
        return self._x

    @x.setter
    def x(self, value):
        if value is None:
            return
        if self.y is not None and value.size != self.y.size:
            return
        self._x = value

    @property
    def y(self) -> np.ndarray or None:
        return self._y

    @property
    def raw_y(self) -> np.ndarray or None:
        return self._raw_y

    @property
    def baseline(self) -> np.ndarray or None:
        return self._baseline

    @property
    def sigma(self) -> float:
        return self._sigma

    @pyqtSlot(float, name='setSigma')
    def set_sigma(self, sigma: float):
        if sigma <= 0:
            return
        self._sigma = sigma
        self._y = smooth_curve(self._raw_y, self.sigma)

    def to_save(self) -> SavedProfile:
        return SavedProfile(self.raw_y, self.x, self.x_range, self.sigma, self.baseline_params, self.baseline)

    def from_save(self, saved_profile: SavedProfile) -> None:
        self.x_range = saved_profile.x_range
        self.set_data(saved_profile.raw_data, saved_profile.x)
        self._baseline = saved_profile.baseline
        self.baseline_params = saved_profile.baseline_params
        self.set_sigma(saved_profile.sigma)

    def set_parameters(self, smoothness: float, asymmetry: float):
        self.baseline_params.smoothness = smoothness
        self.baseline_params.asymmetry = asymmetry

    def get_parameters(self) -> dict:
        return asdict(self.baseline_params)

    def set_data(self, y: np.ndarray, x: np.ndarray):
        self._raw_y = y
        self._y = smooth_curve(self._raw_y, self.sigma)
        self._x = x

    def update_baseline(self):
        if any(v is None for v in (self.x_range, self.y, self.x)):
            return
        if self.y.size != self.x.size:
            return

        x1, x2 = self._get_coords()

        baseline = baseline_correction(
            self._y[x1:x2], self.baseline_params.smoothness, self.baseline_params.asymmetry)
        self._baseline = np.zeros_like(self._y)
        self._baseline[x1:x2] = baseline

    def clear_baseline(self, clear_range: bool = True):
        self._baseline = None
        if clear_range:
            self.x_range = None

    def _get_coords(self):
        scale_factor = self.x.size / (np.nanmax(self.x) - np.nanmin(self.x))
        x_min = np.nanmin(self.x)
        min_ind, max_ind = 0, self.x.size
        x1 = int((self.x_range[0] - x_min) * scale_factor)
        x2 = int((self.x_range[1] - x_min) * scale_factor)
        x1 = min((max((x1, min_ind)), max_ind))
        x2 = min((max((x2, min_ind)), max_ind))
        xs = (x1, x2)
        return min(xs), max(xs)


class BasicProfile(SmoothedProfile):
    sigDataToBeUpdated = pyqtSignal()
    sigDataUpdated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_shown: bool = True
        self._should_update: bool = False

    @property
    def is_shown(self) -> bool:
        return self._is_shown

    @is_shown.setter
    def is_shown(self, value: bool):
        if value == self.is_shown:
            return
        self._is_shown = value
        if value and self._should_update:
            self.update_data()

    def set_show_status(self, value: bool):
        self.is_shown = value

    def update_data(self, *args, **kwargs):
        self.sigDataToBeUpdated.emit()
        self.clear_baseline()
        self.update_data_from_source(*args, **kwargs)
        self.sigDataUpdated.emit()
        self._should_update = False

    @abstractmethod
    def update_data_from_source(self,  *args, **kwargs):
        pass

    @pyqtSlot(name='update')
    def update(self):
        if self.is_shown:
            self.update_data()
        else:
            self._should_update = True
