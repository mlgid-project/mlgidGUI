import os.path
from typing import List
from pathlib import Path
import shutil
from h5py import File, Group
import numpy as np

from ..geometry import Geometry
from .saving_parameters import SavingParameters, SaveMode
from .saving_parameters import h5_saving_dtype
from ..file_manager import (FileManager, FolderKey, ImageKey,
                            IMAGE_PROJECT_KEY, PROJECT_KEY)
from ..image_holder import ImageHolder
from .load_data import ImageData, ImageDataFlags, load_image_data


class SaveH5(object):
    def __init__(self, fm: FileManager, image_holder: ImageHolder):
        self._fm: FileManager = fm
        self._fm = fm
        self._image_holder = image_holder

    def save(self, params: SavingParameters) -> bool:
        filepath = _get_h5_path(params.path)
        path_str: str = str(filepath.resolve())

        save_successfull = _init_h5_project_file(filepath, params)

        for folder_key, image_keys in params.selected_images.items():
            self._save_folder_as_h5(path_str, folder_key, image_keys, params)

        return save_successfull

    def save_for_object_detection(self, params: SavingParameters):
        filepath = _get_h5_path(params.path)
        count: int = 0

        if not filepath.parent.exists():
            raise IOError(f'Parent folder {filepath.parent} does not exist.')

        path_str: str = str(filepath.resolve())

        if filepath.exists():
            with File(path_str, 'r') as f:
                count = len(list(f.keys()))

        with File(path_str, 'a') as f:
            for _, image_keys in params.selected_images.items():
                for image_key in image_keys:
                    if self._save_image_for_object_detection(f, image_key, str(count)):
                        count += 1

    def _save_image_for_object_detection(self, f: File, image_key: ImageKey, name: str) -> bool:
        image_data: ImageData = load_image_data(self._fm, self._image_holder,
            image_key, flags=ImageDataFlags.POLAR_IMAGE | ImageDataFlags.GEOMETRY | ImageDataFlags.ROI_DATA)

        if image_data.polar_image is None or image_data.roi_data is None or image_data.geometry is None:
            return False

        boxes, labels = _get_boxes_n_labels(image_data.roi_data.to_array(), image_data.geometry)

        group = f.create_group(name)
        group.create_dataset('polar_image', data=image_data.polar_image)
        group.create_dataset('boxes', data=boxes)
        group.create_dataset('labels', data=labels)
        group.create_dataset('intensities', data=image_data.roi_data.intensities)
        group.create_dataset('confidence_levels', data=image_data.roi_data.confidence_levels)
        group.create_dataset('cif_files', data=image_data.roi_data.cif_files)
        group.attrs.update(file_key=str(image_key._file_key()))

        try:
            group.attrs.update(path=str(image_key.path))
        except AttributeError:
            pass

        return True

    def _save_folder_as_h5(self, path: str, folder_key: FolderKey,
                           image_keys: List[ImageKey], params: SavingParameters):
        if not image_keys:
            return

        with File(path, 'a') as f:
            for img_idx, image_key in enumerate(image_keys):
                self._save_image_as_h5(f, image_key, img_idx)

    def _save_image_as_h5(self,
                          f,
                          image_key: ImageKey,
                          img_idx: int
                          ):


        h5group = _get_folder_group(f,image_key.parent.path.name)

        h5group.require_group('data').attrs.update({"NX_class": "NXdata", "EX_required": "true", "signal": "img_gid_q"})
        h5group['data'].require_group('analysis').attrs.update({"NX_class": "NXparameters", "EX_required": "true", "signal": "img_gid_q"})
        try:
            h5group['data'].create_dataset('img_gid_q', maxshape=(None, *image_key.get_image().shape), data=image_key.get_image()[np.newaxis, :, :])
        except:
            #append to the existing image array
            h5group['data']['img_gid_q'].resize((h5group['data']['img_gid_q'].shape[0] + image_key.get_image()[np.newaxis, :, :].shape[0]), axis=0)
            h5group['data']['img_gid_q'][-image_key.get_image()[np.newaxis, :, :].shape[0]:] = image_key.get_image()[np.newaxis, :, :]

        try:
            del h5group['data']['analysis']['frame' + str(img_idx).zfill(5)]['detected_peaks']
        except:
            h5group['data']['analysis'].require_group('frame' + str(img_idx).zfill(5))

        diagonal_length = (image_key.get_image().shape[0] ** 2 + image_key.get_image().shape[1] ** 2) ** (1 / 2)
        diagonal_length_q = (image_key.qxy ** 2 + image_key.qz ** 2) ** (1 / 2)
        pixel_per_angstroem = diagonal_length / diagonal_length_q

        if self._fm.rois_data[image_key] is not None:
            rois_data = self._fm.rois_data[image_key].to_dict()
            results_struct = np.zeros(len(rois_data['key']), dtype=h5_saving_dtype)
            results_struct['amplitude'] = [0] * len(rois_data['key'])
            results_struct['angle'] = rois_data['angle']
            results_struct['angle_width'] = rois_data['angle_width']
            results_struct['radius'] = rois_data['radius'] / pixel_per_angstroem
            results_struct['radius_width'] = rois_data['radius_width'] / pixel_per_angstroem
            results_struct['q_xy'] = [0] * len(rois_data['key'])
            results_struct['q_z'] = [0] * len(rois_data['key'])
            results_struct['theta'] = [0] * len(rois_data['key'])
            results_struct['A'] = [0] * len(rois_data['key'])
            results_struct['B'] = [0] * len(rois_data['key'])
            results_struct['C'] = [0] * len(rois_data['key'])
            results_struct['is_cut_qz'] = [0] * len(rois_data['key'])
            results_struct['is_cut_qxy'] = [0] * len(rois_data['key'])
            results_struct['is_ring'] =  np.vectorize({1: True, 2: False,3: False}.get)(rois_data['type'])
            results_struct['visibility'] = np.vectorize({1.0:3, 0.5:2, 0.1:1, -1.0:0}.get)(rois_data['confidence_level'])
            results_struct['score'] = rois_data['score']
            results_struct['id'] = list(range(len(rois_data['key'])))
            h5group['data']['analysis']['frame' + str(img_idx).zfill(5)].create_dataset('detected_peaks', data=results_struct,dtype=h5_saving_dtype)
        try:
            h5group['data']['q_xy'] = np.linspace(0, image_key.qxy, image_key.get_image().shape[1])
            h5group['data']['q_z'] = np.linspace(0, image_key.qz, image_key.get_image().shape[0])
        except OSError:
            pass



def _init_h5_project_file(filepath: Path, params: SavingParameters) -> bool:
    try:
        if not filepath.parent.exists():
            raise IOError(f'Parent folder {filepath.parent} does not exist.')

        path_str: str = str(filepath.resolve())

        if filepath.exists() and params.save_mode.value == SaveMode.create.value:
            raise IOError(f'File {path_str} already exists.')

        if os.path.exists(str(next(iter(params.selected_images)).project_path / 'prototype.h5')):
            shutil.copy(str(next(iter(params.selected_images)).project_path / 'prototype.h5'), path_str)

        if params.save_mode.value == SaveMode.add.value:
            with File(path_str, 'a') as f:
                if PROJECT_KEY not in f.attrs:
                    raise IOError(f'Chosen file {path_str} is not a h5 project file!')

        return True
    except:
        return False

def _get_h5_path(path: Path) -> Path:
    path = path.resolve()
    if path.suffix != '.h5':
        name = '.'.join([path.name.split('.')[0], '.h5'])
        path = path.parent / name
    return path

def _get_folder_group(f: File, name: str) -> Group:
    # TODO carefully handle name collisions by checking Path attribute (or any others)
    return f.create_group(name) if name not in f.keys() else f[name]


def _get_boxes_n_labels(roi_arr: np.ndarray, geometry: Geometry):
    labels, boxes = [], []

    for r, w, a, a_s, key, roi_type in roi_arr:
        x1, x2 = geometry.r2p(r - w / 2), geometry.r2p(r + w / 2)
        y1, y2 = geometry.a2p(a - a_s / 2), geometry.a2p(a + a_s / 2)

        labels.append(roi_type)
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes), np.array(labels)