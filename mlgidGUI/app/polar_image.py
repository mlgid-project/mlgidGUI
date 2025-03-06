from typing import Tuple, NamedTuple

import cv2
import numpy as np

from .geometry import Geometry
from ..app.rois.roi import Roi

INTERPOLATION_ALGORITHMS = {
    'Nearest': cv2.INTER_NEAREST,
    'Bilinear': cv2.INTER_LINEAR,
    'Cubic': cv2.INTER_CUBIC,
    'Lanczos': cv2.INTER_LANCZOS4
}

INTERPOLATION_ALGORITHMS_INVERSED = {v: k for k, v in INTERPOLATION_ALGORITHMS.items()}


class InterpolationParams(NamedTuple):
    shape: Tuple[int, int] = (1024, 1024)
    algorithm: int = cv2.INTER_LINEAR


class PolarImage(object):

    def __init__(self, polar_img: np.ndarray = None,
                 parameters: InterpolationParams = None):
        self._polar_img = polar_img
        self._parameters = parameters or InterpolationParams()

    @property
    def polar_image(self) -> np.ndarray or None:
        return self._polar_img

    @property
    def polar_params(self) -> InterpolationParams:
        return self._parameters

    def set_params(self, shape: Tuple[int, int] = None, algorithm=None):
        shape = shape or self.polar_params.shape
        algorithm = algorithm if algorithm is not None else self.polar_params.algorithm

        self._parameters = InterpolationParams(shape, algorithm)

    def set_algorithm(self, algorithm: int, image: np.ndarray, geometry: Geometry):
        if self.polar_params.algorithm == algorithm:
            return

        self._parameters = InterpolationParams(self.polar_params.shape, algorithm)
        self.update(geometry, image)

    def update(self, geometry: Geometry, image: np.ndarray):
        if image is None:
            self._polar_img = None
            return

        yy, zz = geometry.polar_grids
        self._polar_img = self.calc_polar_image(image, yy, zz, self.polar_params.algorithm)

    def set_polar_image(self, img: np.ndarray):
        self._polar_img = img
        if img.shape != self._parameters.shape:
            self._parameters = InterpolationParams(img.shape, self._parameters.algorithm)

    @staticmethod
    def calc_polar_image(img: np.ndarray, yy: np.ndarray, zz: np.ndarray,
                         algorithm=cv2.INTER_LINEAR) -> np.ndarray or None:
        try:
            return cv2.remap(img.astype(np.float32),
                             yy.astype(np.float32),
                             zz.astype(np.float32),
                             interpolation=algorithm)
        except cv2.error:
            return

    def get_radial_profile(self) -> np.ndarray or None:
        if self.polar_image is None:
            return
        return np.nanmean(self.polar_image,axis=0)

    def get_angular_profile(self, geometry: Geometry, roi: Roi) -> np.ndarray or None:
        if self.polar_image is None:
            return
        r1, r2 = roi.radius - roi.width / 2, roi.radius + roi.width / 2

        r_min, r_max = geometry.r_range
        scale = geometry.scale
        r_size = self.polar_image.shape[1]
        r_ratio = (r_max - r_min) / r_size * scale

        r1, r2 = int((r1 - r_min) / r_ratio), int((r2 - r_min) / r_ratio)
        r1, r2 = max(min((r1, r2)), 0), min(max((r1, r2)), r_size)

        if r1 > r_size or r2 < 0:
            return
        return np.nanmean(self.polar_image[:, r1:r2],axis=1)
