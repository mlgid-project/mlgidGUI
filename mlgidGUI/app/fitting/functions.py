from enum import Enum
from abc import abstractmethod
from typing import Tuple, Dict, Callable

import numpy as np

from .utils import Roi, _update_bounds
from .background import Background

__all__ = ['FittingType', 'FittingFunction', 'FITTING_FUNCTIONS', 'Gaussian', 'Lorentzian']


class FittingType(Enum):
    gaussian = 'Gaussian'
    lorentzian = 'Lorentzian'


class FittingFunction(object):
    NAME: str = ''
    PARAM_NAMES: tuple = ()
    NUM: int = 0
    TYPE: FittingType = None
    is_default: bool = True

    def __init__(self, *args, **kwargs):
        pass

    def get_func(self, background: Background) -> Callable:
        def func(x: np.ndarray, *params):
            return self.__call__(x, background, *params)

        return func

    def __call__(self, x: np.ndarray, background: Background, *params) -> np.ndarray:
        return self.func(x, *params[:self.NUM]) + background(x, *params[self.NUM:])

    @staticmethod
    @abstractmethod
    def func(x: np.ndarray, *params):
        pass

    @staticmethod
    @abstractmethod
    def set_roi_from_params(roi: Roi, params: list):
        pass

    @staticmethod
    @abstractmethod
    def set_params_from_roi(roi: Roi, params: list):
        pass

    @staticmethod
    @abstractmethod
    def _bounds(x: np.ndarray, y: np.ndarray, roi: Roi, background: Background):
        pass

    @classmethod
    def bounds(cls, x: np.ndarray, y: np.ndarray, roi: Roi, background: Background, params_from_roi: bool = False):
        if params_from_roi:
            return _update_bounds(roi, list(cls.PARAM_NAMES) + list(background.PARAM_NAMES),
                                  *cls._bounds(x, y, roi, background))
        else:
            return cls._bounds(x, y, roi, background)


class Gaussian(FittingFunction):
    NAME: str = 'Gaussian'
    PARAM_NAMES: tuple = ('peak height', 'radius', 'width')
    NUM = 3
    TYPE = FittingType.gaussian

    @staticmethod
    def func(x: np.ndarray, *params):
        amp, mu, sigma, *_ = params
        return amp * np.exp(- 2 * (x - mu) ** 2 / sigma ** 2)

    @staticmethod
    def set_roi_from_params(roi: Roi, params: list):
        roi.radius = params[1]
        roi.width = params[2]

    @staticmethod
    def _bounds(x: np.ndarray, y: np.ndarray, roi: Roi, background: Background):
        init_b, upper_b, lower_b = background.bounds(x, y, roi)
        amp, amp_max, amp_min = background.amp_bounds(x, y, init_b)

        return ([amp, roi.radius, roi.width] + init_b,
                [amp_max, roi.radius + roi.width / 2, roi.width * 2] + upper_b,
                [amp_min, roi.radius - roi.width / 2, 0] + lower_b)


class Lorentzian(Gaussian):
    NAME: str = 'Lorentzian'
    PARAM_NAMES: tuple = ('peak height', 'radius', 'width')
    TYPE = FittingType.lorentzian

    @staticmethod
    def func(x: np.ndarray, *params):
        amp, mu, sigma, *_ = params
        w = (sigma / 2) ** 2
        return amp * w / (w + (x - mu) ** 2)


FITTING_FUNCTIONS: Dict[FittingType, FittingFunction.__class__] = {
    FittingType.gaussian: Gaussian,
    FittingType.lorentzian: Lorentzian,
}
