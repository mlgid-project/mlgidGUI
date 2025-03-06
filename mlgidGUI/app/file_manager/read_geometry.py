from h5py import Group

from .object_file_manager import _ObjectFileManager
from .keys import FolderKey
from ..geometry import Geometry


class _DefaultGeometry(_ObjectFileManager):
    NAME = 'geometries'

    def _get_path(self, key: FolderKey):
        if key is not None:
            if key.parent:
                return self.folder / f'default_geometry_{key.parent.name}'
            else:
                return self.folder / f'default_geometry_{key.name}'
        else:
            return self.folder / f'default_geometry'

    @staticmethod
    def get_h5(h5group: Group, key: FolderKey):
        return Geometry.fromdict(dict(h5group.attrs))

    @staticmethod
    def set_h5(h5group: Group, key: FolderKey, value: Geometry):
        h5group.attrs.update(value.to_dict())

    @staticmethod
    def del_h5(h5group: Group, key: FolderKey):
        for key in Geometry.keys():
            if key in h5group.parent.attrs.keys():
                del h5group.attrs[key]


class _ReadGeometry(_ObjectFileManager):
    NAME = 'geometries'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = _DefaultGeometry(*args, **kwargs)

    @staticmethod
    def get_h5(h5group: Group, key):
        return Geometry.fromdict(dict(h5group.attrs))

    @staticmethod
    def set_h5(h5group: Group, key, value: Geometry):
        h5group.attrs.update(value.to_dict())

    @staticmethod
    def del_h5(h5group: Group, key):
        for key in Geometry.keys():
            if key in h5group.attrs.keys():
                del h5group.attrs[key]
