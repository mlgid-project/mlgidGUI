from typing import Tuple, NamedTuple
from copy import deepcopy

import numpy as np

from .transformations import TransformationsHolder


class AxesScale(object):
    def __init__(self, scale: float = 1):
        self._scale = scale
        self._prev_scale = 1.

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self.set_scale(value)

    @property
    def scale_change(self):
        return self.scale / self._prev_scale

    def set_scale(self, scale: float):
        self._scale, self._prev_scale = scale, self._scale


class BeamCenter(NamedTuple):
    z: float = 0
    y: float = 0


class Geometry(object):
    def __init__(self, *, beam_center: tuple = (0, 0),
                 scale: float = 1.,
                 shape: Tuple[int, int] = (10, 10),
                 polar_shape: Tuple[int, int] = (1024, 1024),
                 t_key: str = None, update: bool = True,
                 **kwargs):

        self._beam_center = BeamCenter(*beam_center)
        self._scale = AxesScale(scale)
        self._transforms = TransformationsHolder()
        if t_key:
            self._transforms.update(t_key)
        self._shape = tuple(shape)
        self._polar_shape = tuple(polar_shape)

        self._r_range = (0, 1)
        self._phi_range = (0, 2 * np.pi)
        self._ring_bounds = (0, 2 * np.pi)
        self._r = self._phi = None
        self._y = self._z = None
        self._polar_zz = self._polar_yy = None
        self._polar_aspect_ratio = None

        if update:
            self.update()

    def to_dict(self) -> dict:
        return dict(beam_center=tuple(self.beam_center),
                    shape=self.shape,
                    scale=self.scale,
                    t_key=self.t.key,
                    polar_shape=self.polar_shape)

    @staticmethod
    def keys():
        return 'beam_center', 'shape', 'scale', 't_key', 'polar_shape'

    @classmethod
    def fromdict(cls, d: dict):
        return cls(**d)

    def __eq__(self, other):
        if type(other) != Geometry:
            return False

        return self.to_dict() == other.to_dict()

    @property
    def t(self):
        return self._transforms

    def copy(self) -> 'Geometry':
        return deepcopy(self)

    @property
    def is_available(self) -> bool:
        return self.shape is not None

    @property
    def ring_bounds(self):
        return self._ring_bounds

    @property
    def beam_center(self):
        return self._beam_center

    @property
    def r_range(self) -> Tuple[float, float]:
        return self._r_range

    @property
    def phi_range(self) -> Tuple[float, float]:
        return self._phi_range

    @property
    def polar_grids(self):
        return self._polar_yy, self._polar_zz

    @property
    def r_axis(self):
        return self._r

    @property
    def phi_axis(self):
        return self._phi

    @property
    def polar_aspect_ratio(self):
        return self._polar_aspect_ratio

    @property
    def y_axis(self):
        return self._y

    @property
    def z_axis(self):
        return self._z

    @property
    def shape(self):
        return self._shape

    @property
    def polar_shape(self):
        return self._polar_shape

    @property
    def scale(self):
        return self._scale.scale

    @property
    def scale_change(self):
        return self._scale.scale_change

    def set_scale(self, scale: float, update: bool = True):
        if scale != self.scale:
            self._scale.set_scale(scale)
            if update:
                self._update_axes_on_scale(False)

    def set_beam_center(self, z: float, y: float, update: bool = True):
        if (self._beam_center.z, self._beam_center.y) != (z, y):
            self._beam_center = BeamCenter(z, y)
            if update:
                self.update()
            return True
        return False

    def set_shape(self, shape: Tuple[int, int], update: bool = True):
        if shape != self.shape:
            self._shape = shape
            if update:
                self.update()
            return True
        return False

    def set_polar_shape(self, shape: Tuple[int, int], update: bool = True):
        if shape != self.polar_shape:
            self._polar_shape = shape
            if update:
                self.update_polar()
            return True
        return False

    def update(self):
        if not self.is_available:
            return
        self._update_ranges()
        self._update_polar_grid()
        self._update_axes_on_scale()

    def update_polar(self):
        self._update_polar_grid()
        self._r *= self.scale
        self._polar_aspect_ratio /= self.scale

    def _update_ranges(self):
        self._y = (np.arange(self._shape[1]) - self._beam_center.y)
        self._z = (np.arange(self._shape[0]) - self._beam_center.z)

        yy, zz = np.meshgrid(self._y, self._z)
        rr = np.sqrt(yy ** 2 + zz ** 2)
        phi = np.arctan2(zz, yy)
        self._r_range = (np.nanmin(rr), np.nanmax(rr))
        self._phi_range = p_min, p_max = np.nanmin(phi), np.nanmax(phi)
        angle, angle_std = (p_max + p_min) / 2 * 180 / np.pi, (p_max - p_min) * 180 / np.pi
        self._ring_bounds = (angle, angle_std)

    def _update_polar_grid(self):
        self._phi = np.linspace(*self.phi_range, self.polar_shape[0])
        self._r = np.linspace(*self.r_range, self.polar_shape[1])

        r_matrix = self._r[np.newaxis, :].repeat(self.polar_shape[0], axis=0)
        p_matrix = self._phi[:, np.newaxis].repeat(self.polar_shape[1], axis=1)

        self._polar_yy = r_matrix * np.cos(p_matrix) + self.beam_center.y
        self._polar_zz = r_matrix * np.sin(p_matrix) + self.beam_center.z

        self._phi *= 180 / np.pi

        aspect_ratio = (np.nanmax(self._phi) - np.nanmin(self._phi)) * \
                       self.polar_shape[0] / (self._r_range[1] - self._r_range[0]) / self.polar_shape[1]
        self._polar_aspect_ratio = aspect_ratio

    def _update_axes_on_scale(self, init: bool = True):
        if init:
            scale = self.scale
        else:
            scale = self.scale_change

        self._r = self._r * scale
        self._y = self._y * scale
        self._z = self._z * scale
        self._polar_aspect_ratio = self._polar_aspect_ratio / scale

    def r2p(self, r):
        return (r - self.r_range[0]) / (self.r_range[1] - self.r_range[0]) * self.polar_shape[1]

    def a2p(self, a):
        return (a / 180 * np.pi - self.phi_range[0]) / (self.phi_range[1] - self.phi_range[0]) * self.polar_shape[0]

