from h5py import Group
from .npy_file_manager import _ReadNpy
from pathlib import Path

class _ReadImage(_ReadNpy):
    NAME = 'images'

    def init(self):
        path = self.project_structure.path

        if path:
            self.folder: Path = path / self.NAME
    @staticmethod
    def get_h5(h5group: Group, key):
        pass

    @staticmethod
    def set_h5(h5group: Group, key, image):
        _ReadImage.del_h5(h5group, key)
        h5group.create_dataset('image', data=image)

    @staticmethod
    def del_h5(h5group: Group, key):
        if 'image' in h5group.keys():
            del h5group['image']

    def __getitem__(self, key):
        internal_path = self._get_path(key)
        if internal_path.is_file():
            return self._get_pickle(internal_path)
        return key.get_image()
