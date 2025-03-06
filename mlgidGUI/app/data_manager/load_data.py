from enum import Flag, auto
from typing import NamedTuple
from numpy import ndarray
from ..geometry import Geometry
from ..image_holder import ImageHolder
from ..file_manager import ImageKey, FolderKey, FileManager
from ..rois import RoiData, RoiMetaData

class ImageData(NamedTuple):
    image_key: ImageKey
    image: ndarray
    polar_image: ndarray
    geometry: Geometry
    default_geometry: bool
    roi_data: RoiData


class ImageDataFlags(Flag):
    IMAGE = auto()
    POLAR_IMAGE = auto()
    GEOMETRY = auto()
    ROI_DATA = auto()
    ALL = IMAGE | POLAR_IMAGE | GEOMETRY | ROI_DATA


class FolderData(NamedTuple):
    folder_key: FolderKey
    roi_metadata: RoiMetaData
    default_geometry: Geometry


class FolderDataFlags(Flag):
    ROI_METADATA = auto()
    GEOMETRY = auto()
    ALL = ROI_METADATA | GEOMETRY


def load_image_data(fm: FileManager, image_holder: ImageHolder, image_key: ImageKey,
                    flags: ImageDataFlags = ImageDataFlags.ALL) -> ImageData:
    image = None
    polar_image = None
    geometry = None
    default_geometry = True
    roi_data = None

    if ImageDataFlags.GEOMETRY in flags or ImageDataFlags.IMAGE in flags:
        geometry = fm.geometries[image_key]
        if geometry:
            default_geometry = False
        else:
            geometry = fm.geometries.default[image_key.parent]

    if ImageDataFlags.POLAR_IMAGE in flags:
        image, polar_image, _ = image_holder.get_data_by_key(image_key)
    else:
        if ImageDataFlags.IMAGE in flags:
            image = image_key.get_image()
            if image is not None:
                if geometry:
                    image = geometry.t(image)
    if ImageDataFlags.ROI_DATA in flags:
        roi_data = fm.rois_data[image_key]

    return ImageData(image_key=image_key, image=image,
                     polar_image=polar_image, geometry=geometry,
                     default_geometry=default_geometry,
                     roi_data=roi_data)

def load_folder_data(self, folder_key: FolderKey, flags: FolderDataFlags = None) -> FolderData:
    roi_metadata = None
    geometry = None

    if FolderDataFlags.ROI_METADATA in flags:
        roi_metadata = self._fm.rois_meta_data[folder_key]
    if FolderDataFlags.GEOMETRY in flags:
        geometry = self._fm.geometries.default[folder_key]
    return FolderData(folder_key=folder_key, roi_metadata=roi_metadata,
                      default_geometry=geometry)
