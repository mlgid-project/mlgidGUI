import logging

from h5py import Group

from .object_file_manager import _ObjectFileManager
from ..rois.roi_data import RoiData

logger = logging.getLogger(__name__)


class _ReadRoiData(_ObjectFileManager):
    NAME = 'roi_data'

    @staticmethod
    def get_h5(h5group: Group, key):
        if 'roi_data' in h5group.keys():
            roi_dict = {
                k: v[()] for k, v in h5group['roi_data'].items()
            }
            roi_data = RoiData.from_dict(roi_dict)
            return roi_data

    @staticmethod
    def set_h5(h5group: Group, key, value: RoiData):
        if 'roi_data' in h5group.keys():
            del h5group['roi_data']

        rois_dict = value.to_dict()
        roi_group = h5group.create_group('roi_data')
        for key, arr in rois_dict.items():
            try:
                roi_group.create_dataset(key, data=arr)
            except Exception as err:
                logger.exception(err)

    @staticmethod
    def del_h5(h5group: Group, key):
        if 'roi_data' in h5group.keys():
            del h5group['roi_data']
