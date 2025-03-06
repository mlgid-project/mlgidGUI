from typing import Dict, Iterable

from h5py import Group

from .roi_data import RoiData
from .roi import Roi
from ..file_manager.keys import ImageKey, FolderKey


class ImageKeysContainer(object):
    def __init__(self, *image_keys):
        self.__image_dict: Dict[int, ImageKey] = {}
        for key in image_keys:
            self.add(key)

    def add(self, image_key: ImageKey):
        self.__image_dict[hash(image_key)] = image_key

    def remove(self, image_key: ImageKey):
        try:
            del self.__image_dict[hash(image_key)]
        except KeyError:
            raise ValueError(f'Unregistered image key {image_key}')

    def closest(self, image_key: ImageKey) -> ImageKey or None:
        if not self.__image_dict:
            return
        if hash(image_key) in self.__image_dict:
            return image_key
        idx, target_key, min_distance = image_key.idx, None, None

        for i, key in enumerate(self):
            if target_key is None or min_distance >= abs(idx - key.idx):
                target_key = key
                min_distance = abs(idx - target_key.idx)
        return target_key

    def __contains__(self, item):
        return hash(item) in self.__image_dict

    def __iter__(self):
        yield from self.__image_dict.values()


class RoiMetaData(object):
    H5_NAME = 'roi_metadata'

    def __init__(self, folder_key: FolderKey, *,
                 names: Dict[int, str] = None):
        # groups: Dict[int, str] = None):

        self.folder_key = folder_key.clean_copy()
        self.__names: Dict[int, str] = names or {}
        self.__keys_dict: Dict[int, ImageKeysContainer] = {}
        # self.__radii: Dict[int, Dict[ImageKey, float]] = {}
        # self.__widths: Dict[int, Dict[ImageKey, float]] = {}
        # self.__groups: Dict[int, str] = groups or {}

    def roi_keys(self):
        yield from self.__keys_dict.keys()

    def __contains__(self, item: int):
        return item in self.__keys_dict

    def __getitem__(self, item: int):
        try:
            image_set = self.__keys_dict[item]
        except KeyError:
            raise KeyError(f'The roi key {item} is not registered in RoiMetaData.')
        yield from image_set

    def __len__(self):
        return len(self.__keys_dict)

    def add_roi(self, roi: Roi, image_key: ImageKey, *,
                make_copy: bool = True):
        if make_copy:
            image_key = image_key.clean_copy()

        if roi.key is None or roi.key not in self:
            self._create_new_roi(roi, image_key)
        else:
            self.__keys_dict[roi.key].add(image_key)
            roi.name = self.__names[roi.key]

    def add_rois(self, rois: Iterable[Roi], image_key):
        image_key = image_key.clean_copy()
        for roi in rois:
            self.add_roi(roi, image_key, make_copy=False)

    def delete_roi(self, key: int, image_key: ImageKey):
        try:
            self.__keys_dict[key].remove(image_key)
        except KeyError:
            raise ValueError(f'Unknown roi {key}')
        except ValueError:
            raise ValueError(f'Unregistered image key {image_key} for roi {key}')

        if not self.__keys_dict[key]:
            del self.__keys_dict[key]
            del self.__names[key]

    def rename(self, roi: Roi, name: str):
        try:
            self.__names[roi.key] = name
            roi.name = name
        except KeyError:
            raise ValueError(f'Unknown roi key {roi.key}')

    def update_metadata(self, roi_data: RoiData, image_key: ImageKey):
        image_key = image_key.clean_copy()
        for key in roi_data.keys():
            if key not in self:
                self.__keys_dict[key] = ImageKeysContainer(image_key)
            else:
                self.__keys_dict[key].add(image_key)
            roi = roi_data[key]

            try:
                roi.name = self.__names[key]
            except KeyError:
                self.__names[key] = roi.name

    def get_deleted_rois(self, image_key: ImageKey):
        return [(key, self.__names[key]) for key in self.roi_keys() if image_key not in self.__keys_dict[key]]

    def _create_new_roi(self, roi: Roi, image_key: ImageKey):
        if len(self):
            roi.key = key = max(self.roi_keys()) + 1
        else:
            roi.key = key = 0
        if not roi.name:
            roi.name = str(roi.key)
        self.__keys_dict[key] = ImageKeysContainer(image_key)
        self.__names[key] = roi.name

    # methods for storing the object to and retrieving from h5 files (under development)

    def to_h5(self, h5_group: Group):
        pass

    @classmethod
    def from_h5(cls, folder_key: FolderKey, h5_group: Group):
        return cls(folder_key)

    @staticmethod
    def del_h5(h5_group: Group):
        if RoiMetaData.H5_NAME in h5_group.keys():
            del h5_group[RoiMetaData.H5_NAME]
