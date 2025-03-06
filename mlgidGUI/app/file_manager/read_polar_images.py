from h5py import Group
from pathlib import Path
from .npy_file_manager import _ReadNpy


class _ReadPolarImage(_ReadNpy):
    NAME = 'polar_images'

    @staticmethod
    def get_h5(h5group: Group, key):
        if 'polar_image' in h5group.keys():
            return h5group['polar_image'][()]

    @staticmethod
    def set_h5(h5group: Group, key, image):
        _ReadPolarImage.del_h5(h5group, key)
        h5group.create_dataset('polar_image', data=image)

    @staticmethod
    def del_h5(h5group: Group, key):
        if 'polar_image' in h5group.keys():
            del h5group['polar_image']
