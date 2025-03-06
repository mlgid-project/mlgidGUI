from typing import Tuple
import logging

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject

import numpy as np

from .transformations import Transformation
from .file_manager import FileManager, ImageKey
from .geometry import Geometry


class GeometryHolder(QObject):
    sigBeamCenterChanged = pyqtSignal()
    sigGeometryChangeFinished = pyqtSignal()
    sigPolarGeometryChanged = pyqtSignal()
    sigTransformed = pyqtSignal()
    sigScaleChanged = pyqtSignal()
    sigRingBoundsChanged = pyqtSignal(tuple)

    log = logging.getLogger(__name__)

    def __init__(self, fm: FileManager):
        QObject.__init__(self)
        self._fm = fm
        self._default_geometry = Geometry()
        self._current_geometry = None
        self._ring_bounds = (0, np.pi * 2)
        self._current_key: ImageKey or None = None

    def transform_image(self, raw_image: np.ndarray):
        return self.geometry.t(raw_image)

    def add_transform(self, op: Transformation):
        if not self._current_geometry:
            self._current_geometry = self.geometry.copy()
        self.geometry.t.add(op)
        self.sigTransformed.emit()

    def get_geometry(self, image_key: ImageKey):
        return self._fm.geometries[image_key] or self._fm.geometries.default[image_key.parent] or Geometry()

    @property
    def geometry(self) -> Geometry:
        return self._current_geometry or self._default_geometry

    @property
    def is_default(self) -> bool:
        return self._current_geometry is None

    def change_image(self, image_key: ImageKey, image: np.ndarray = None):
        if image_key == self._current_key:
            return

        self.save_state()

        self._current_key = image_key
        if not self._current_key:
            return

        self._default_geometry = self._fm.geometries.default[image_key.parent] or Geometry()
        self._current_geometry = self._fm.geometries[image_key]

        if image is not None:
            image = self.transform_image(image)
            if image.shape != self.geometry.shape:
                self.set_shape(image.shape)
        return image

    def save_as_default(self):
        if self._current_geometry and self._current_key:
            self._default_geometry = self._current_geometry
            self._fm.geometries.default[self._current_key.parent] = self._current_geometry
            del self._fm.geometries[self._current_key]
            self._current_geometry = None
            self.log.info('Geometry saved as default.')

        self.log.info('Geometry not saved (default used)')

    def save_state(self):
        if not (self._current_key and self._current_geometry):
            self.log.info('Geometry not saved (default used)')
            return

        if not self._fm.geometries.default[self._current_key.parent]:
            self.save_as_default()
        elif self._current_geometry != self._default_geometry:
            self._fm.geometries[self._current_key] = self._current_geometry
            self.log.info('Non-default geometry saved')
        else:
            del self._fm.geometries[self._current_key]
            self.log.info('Non-default geometry deleted.')

    @pyqtSlot(tuple, bool, name='changeBeamCenter')
    def set_beam_center(self, beam_center: tuple, finished: bool = True):
        if (beam_center[0] == self.geometry.beam_center.z and
                beam_center[1] == self.geometry.beam_center.y):
            return
        if not self._current_geometry:
            self._current_geometry = self.geometry.copy()
        self._current_geometry.set_beam_center(*beam_center)
        self.sigBeamCenterChanged.emit()
        self.check_ring_bounds()
        if finished:
            self.sigGeometryChangeFinished.emit()

    def set_shape(self, shape: Tuple[int, int]):
        if not self._current_geometry:
            self._current_geometry = self.geometry.copy()
        self._current_geometry.set_shape(shape)

    @pyqtSlot(tuple, name='changePolarShape')
    def set_polar_shape(self, shape: Tuple[int, int]):
        if not self._current_geometry:
            self._current_geometry = self.geometry.copy()
        self._current_geometry.set_polar_shape(shape)
        self.sigGeometryChangeFinished.emit()

    @pyqtSlot(float, name='changeScale')
    def set_scale(self, scale: float):
        if self.geometry.scale == scale:
            return
        if not self._current_geometry:
            self._current_geometry = self.geometry.copy()
        self._current_geometry.set_scale(scale)
        self.sigScaleChanged.emit()

    def check_ring_bounds(self):
        if self.geometry and self.geometry.ring_bounds != self._ring_bounds:
            self._ring_bounds = self.geometry.ring_bounds
            self.sigRingBoundsChanged.emit(self._ring_bounds)
