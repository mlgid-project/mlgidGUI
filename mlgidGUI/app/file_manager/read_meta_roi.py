from h5py import Group
from pathlib import Path
from .object_file_manager import _ObjectFileManager
from .keys import RemoveWeakrefs, FolderKey


class ImportManager(object):
    _class = None

    @classmethod
    def RoiMetaData(cls):
        if not cls._class:
            from ..rois import RoiMetaData
            cls._class = RoiMetaData
        return cls._class

class _ReadMetaData(_ObjectFileManager):
    NAME = 'rois_meta_data'

    def __getitem__(self, item: FolderKey):
        roi_meta_data = super().__getitem__(item)
        if roi_meta_data:
            roi_meta_data.folder_key = item.clean_copy()
        return roi_meta_data

    def __setitem__(self, key: FolderKey, value):
        with RemoveWeakrefs(value.folder_key):
            super().__setitem__(key, value)

    @staticmethod
    def get_h5(h5group: Group, key: FolderKey):
        return ImportManager.RoiMetaData().from_h5(key, h5group)

    @staticmethod
    def set_h5(h5group: Group, key: FolderKey, value: 'RoiMetaData'):
        value.to_h5(h5group)

    @staticmethod
    def del_h5(h5group: Group, key):
        ImportManager.RoiMetaData().del_h5(h5group)
