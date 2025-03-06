import logging
from typing import List
from datetime import datetime as dt

import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject

from .rois.roi_dict import RoiDict, Roi
from .geometry import Geometry
from .geometry_holder import GeometryHolder
from .polar_image import (PolarImage, InterpolationParams,
                          INTERPOLATION_ALGORITHMS, INTERPOLATION_ALGORITHMS_INVERSED)
from .file_manager import FileManager, ImageKey
from .fitting import FitObject


class ImageHolder(QObject):
    sigImageChanged = pyqtSignal()
    sigPolarImageChanged = pyqtSignal()
    sigFitOpen = pyqtSignal(object)
    sigFitSaved = pyqtSignal(tuple)
    sigEmptyImage = pyqtSignal()
    sigCIFPeaksChanged = pyqtSignal(float)

    log = logging.getLogger(__name__)

    def __init__(self, fm: FileManager, g_holder: GeometryHolder, roi_dict: RoiDict):
        QObject.__init__(self)
        self._fm = fm
        self._roi_dict = roi_dict
        self._image = self._raw_image = None
        self._current_key: ImageKey = None
        self._polar_image = PolarImage()
        self._g_holder = g_holder

        self._roi_dict.sigFitRoisOpen.connect(self.open_fit_rois)
        self._g_holder.sigPolarGeometryChanged.connect(self._update_polar_image)
        self._g_holder.sigGeometryChangeFinished.connect(self._update_polar_image)
        self._g_holder.sigTransformed.connect(self._update_image)

    @property
    def current_key(self) -> ImageKey or None:
        return self._current_key

    @property
    def image(self) -> np.ndarray or None:
        return self._image

    @property
    def raw_image(self) -> np.ndarray or None:
        return self._raw_image

    @property
    def polar(self) -> PolarImage:
        return self._polar_image

    @property
    def polar_image(self) -> np.ndarray or None:
        return self._polar_image.polar_image

    @property
    def polar_params(self) -> InterpolationParams:
        return self._polar_image.polar_params

    @property
    def g_holder(self) -> GeometryHolder:
        return self._g_holder

    @property
    def geometry(self) -> Geometry:
        return self.g_holder.geometry

    @pyqtSlot(object, name='changeImage')
    def change_image(self, image_key: ImageKey):
        if self._current_key == image_key:
            return
        self._current_key = image_key

        self._roi_dict.save_and_clear()

        if not image_key:
            self.g_holder.change_image(None)
            self._roi_dict.change_image(None)
            self.sigEmptyImage.emit()
            return

        image = self._fm.images[image_key]

        if image is None:
            self.sigEmptyImage.emit()
            return

        polar_image = self._fm.polar_images[image_key]
        prev_geometry = self.geometry

        self._raw_image = image
        self._image = self.g_holder.change_image(image_key, image)
        self._update_polar_image(polar_image, False)

        if self.geometry.beam_center != prev_geometry.beam_center:
            self.g_holder.sigBeamCenterChanged.emit()
        if self.geometry.scale != prev_geometry.scale:
            self.g_holder.sigScaleChanged.emit()

        self.sigImageChanged.emit()
        self.sigPolarImageChanged.emit()

        # self.g_holder.check_ring_bounds()
        self._roi_dict.change_image(image_key)

    def get_data_by_key(self, image_key: ImageKey, save: bool = False):
        polar_image = self._fm.polar_images[image_key]
        image = self._fm.images[image_key]
        geometry = self.g_holder.get_geometry(image_key)
        if image is None or geometry is None:
            return None, None, None

        image = geometry.t(image)
        if geometry.shape != image.shape:
            geometry.set_shape(image.shape)
        if polar_image is None:
            yy, zz = geometry.polar_grids
            polar_image = self.polar.calc_polar_image(image, yy, zz, self.polar_params.algorithm)
            if save:
                self._fm.polar_images[image_key] = polar_image
        return image, polar_image, geometry

    def _update_polar_image(self, polar_image=None, emit: bool = True):
        if polar_image is not None:
            self._polar_image.set_polar_image(polar_image)
        else:
            self._polar_image.update(self.geometry, self.image)
        if emit:
            self.sigPolarImageChanged.emit()

    def _update_image(self, polar_image: np.ndarray = None, emit: bool = True) -> None:
        if self.raw_image is None:
            return
        old_shape = self.image.shape if self.image is not None else None
        self._image = self.g_holder.transform_image(self._raw_image)
        if old_shape != self.image.shape:
            self.g_holder.set_shape(self.image.shape)
        self._update_polar_image(polar_image, emit=emit)
        if emit:
            self.sigImageChanged.emit()

    def set_polar_image_params(self, params: dict):
        shape = (params.get('phi_size', self.polar_params.shape[0]),
                 params.get('r_size', self.polar_params.shape[1]))
        algorithm = INTERPOLATION_ALGORITHMS.get(params.get('mode', None), self.polar_params.algorithm)
        self.polar.set_params(shape=shape, algorithm=algorithm)
        self.g_holder.set_polar_shape(shape)

    def polar_image_params_dict(self):
        return dict(phi_size=self.geometry.polar_shape[0], r_size=self.geometry.polar_shape[1],
                    algorithm=INTERPOLATION_ALGORITHMS_INVERSED[self.polar_params.algorithm])

    def get_radial_profile(self) -> np.ndarray or None:
        return self.polar.get_radial_profile()

    def get_angular_profile(self, key: int) -> np.ndarray or None:
        if key is None:
            try:
                roi = self._roi_dict.selected_rois[0]
            except IndexError:
                return
        else:
            roi = self._roi_dict[key]

        return self.polar.get_angular_profile(self.geometry, roi)

    @pyqtSlot(list, name='openFitRois')
    def open_fit_rois(self, rois: List[Roi]):
        self.sigFitOpen.emit(self.create_fit_object(rois))

    def create_fit_object(self, rois: List[Roi]) -> FitObject:
        fit_object = FitObject(self._current_key, self.polar_image,
                               self.geometry.r_axis, self.geometry.phi_axis)
        profile = self._fm.profiles[self._current_key]

        if profile:
            fit_object.set_profile(profile)

        for roi in rois:
            fit_object.new_fit(roi)
        return fit_object

    @pyqtSlot(object, name='applyFit')
    def apply_fit(self, fit_object: FitObject):
        name = dt.now().ctime()
        fit_object.name = name
        parent = fit_object.image_key.parent
        fit_object.image_key.remove_parent()
        self._fm.fits[fit_object.image_key, name] = fit_object
        fit_object.image_key.set_parent(parent)

        self._roi_dict.apply_fit([fit.roi for fit in fit_object.fits.values()], fit_object.image_key)

        self.sigFitSaved.emit((fit_object.image_key, name))
