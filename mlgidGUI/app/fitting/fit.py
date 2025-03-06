# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple
import logging

import numpy as np
from scipy.optimize import curve_fit

from .functions import FittingFunction
from .background import Background
from .utils import Roi
from .range_strategy import RangeStrategy


@dataclass
class Fit:
    roi: Roi

    r_range: Tuple[float, float]
    x_range: Tuple[int, int]
    x: np.ndarray
    y: np.ndarray

    init_curve: np.ndarray
    init_params: list
    lower_bounds: list
    upper_bounds: list

    fitted_params: list
    fit_errors: list
    fitting_curve: np.ndarray
    background_curve: np.ndarray

    fitting_function: FittingFunction
    background: Background
    range_strategy: RangeStrategy
    sigma: float = None
    x_profile: np.ndarray = None
    y_profile: np.ndarray = None

    @property
    def param_names(self):
        return [*self.fitting_function.PARAM_NAMES, *self.background.PARAM_NAMES]

    @property
    def bounds(self):
        return tuple(self.lower_bounds), tuple(self.upper_bounds)

    def update(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

    def update_fit(self, update_bounds: bool = True):
        if self.roi.movable:
            if update_bounds and self.x.size:
                self.init_params, self.upper_bounds, self.lower_bounds = self.fitting_function.bounds(
                    self.x, self.y, self.roi, self.background)
            else:
                self.fitting_function.set_roi_from_params(self.roi, self.init_params)

        self.background_curve = self.background(self.x, *self.init_params)
        self.init_curve = self.fitting_function(self.x, self.background, *self.init_params)

        if self.fitted_params:
            self.fitting_curve = self.fitting_function(self.x, self.background, *self.fitted_params)

    def set_background(self, background: Background, update: bool = True):
        self.background = background
        if update:
            self.update_fit()

    def set_function(self, fitting_function: FittingFunction, update: bool = True):
        self.fitting_function = fitting_function
        if update:
            self.update_fit()

    def do_fit(self) -> None:
        if not self.y.size or not self.x.size:
            return
        try:
            func = self.fitting_function.get_func(self.background)
            popt, pcov = curve_fit(func, self.x, self.y, self.init_params,
                                   bounds=self.bounds)
            perr = np.sqrt(np.diag(pcov))

            self.fitted_params = popt.tolist()
            self.init_params = self.fitted_params
            self.fit_errors = perr.tolist()
            self.fitting_curve = func(self.x, *popt)
            self.background_curve = self.background(self.x, *popt)
            self.init_curve = self.fitting_curve
            self.update_roi_fit_dict()
            self.fitting_function.set_roi_from_params(self.roi, self.fitted_params)
        except (ValueError, RuntimeError) as err:
            logging.exception(err)
            return

    def set_roi_from_params(self, params=None):
        if params is None:
            if self.fitted_params is None:
                return
            params = self.fitted_params
            if not params:
                return
        self.fitting_function.set_roi_from_params(self.roi, params)

    def set_roi_from_range(self):
        r1, r2 = self.r_range
        roi = self.roi
        roi.radius = (r1 + r2) / 2
        roi.width = (r2 - r1) / 2

    def update_roi_fit_dict(self):
        self.roi.fitted_parameters = {}

        if self.fitted_params is not None:
            self.roi.fitted_parameters.update(dict(zip(self.param_names, self.fitted_params)))
            self.roi.fitted_parameters['fitted_params'] = self.fitted_params
            self.roi.fitted_parameters['fit_errors'] = self.fit_errors

        self.roi.fitted_parameters['fitting_function'] = self.fitting_function.TYPE
        self.roi.fitted_parameters['background'] = self.background.TYPE
        self.roi.fitted_parameters['lower_bounds'] = self.lower_bounds
        self.roi.fitted_parameters['init_params'] = self.init_params
        self.roi.fitted_parameters['upper_bounds'] = self.upper_bounds
        self.roi.fitted_parameters['r_range'] = self.r_range
        self.roi.fitted_parameters['x_range'] = self.x_range
