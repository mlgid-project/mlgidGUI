import os
from h5py import File, Group
from PIL import Image
from pathlib import Path
import numpy as np
from ..geometry import Geometry
from ..rois import RoiData
from ..file_manager import ImagePathKey, FileManager, keys
from itertools import count

class ImportProjectFromH5(object):
    def __init__(self, fm: FileManager):
        self.fm = fm

    def load(self,  h5path: Path, project_path: Path,):
        if os.listdir(project_path):
            raise IOError('New project folder needs to be empty!')

        self.fm.open_project(project_path)

        img_folder = project_path / 'images'
        img_folder.mkdir()

        with File(h5path, 'r') as f:
            for folder_name, group in f.items():
                is_v1 = folder_name + '/' + list(group.keys())[0] + '/' + 'image' in f
                break

        if is_v1:
            with File(h5path, 'r') as f:
                for folder_name, group in f.items():
                    self.import_v1(img_folder, project_path, group, folder_name)
                    continue
        else:
            self.copy_prototype(h5path, project_path)
            with File(h5path, 'r') as f:
                for folder_name, group in f.items():
                    self.import_v2(img_folder, project_path, group, folder_name)

    def import_v1(self, img_folder, project_path, group, folder_name):

        for idx, (img_name, img_data) in enumerate(self.v1_parse_imgs_from_h5_group(group)):
            folder_path = img_folder / keys.validate_filename(folder_name + img_name)
            folder_path.mkdir()
            folder_key = self.fm.add_root_path_to_project(folder_path)
            self.fm.geometries.default[folder_key] = Geometry.fromdict(dict(group.attrs))

            img = img_data['image']
            img_name = keys.validate_filename(img_name)
            img_path = folder_path / str(img_name + '.tiff')
            Image.fromarray(img).save(img_path, 'tiff')
            img_key = ImagePathKey(project_path, folder_key, path=img_path, idx=idx)
            folder_key._image_children.append(img_key)
            self.fm.polar_images[img_key] = img_data.get('polar_image', None)
            self.fm.rois_data[img_key] = img_data.get('roi_data', None)

    def v1_parse_imgs_from_h5_group(self, h5group: Group, skip_empty_imgs: bool = False):
        for img_name, img_group in h5group.items():
            try:
                img_data = {}

                if 'roi_data' in img_group:
                    roi_dict = {
                        k: v[()] for k, v in img_group['roi_data'].items()
                    }
                    roi_dict['angle_width'] = roi_dict['angle_std']
                    del roi_dict['angle_std']
                    roi_dict['radius_width'] = roi_dict['width']
                    del roi_dict['width']

                    img_data['roi_data'] = RoiData.from_dict(roi_dict)

                    if skip_empty_imgs and not len(img_data['roi_data']):
                        continue

                elif skip_empty_imgs:
                    continue

                for key in ('image', 'polar_image'):
                    if key in img_group:
                        img_data[key] = img_group[key][()]

                yield img_name, img_data
            except:
                pass

    def import_v2(self, img_folder, project_path, group, folder_name):
        folder_path = img_folder / folder_name
        if not os.path.isdir(folder_path):
            folder_path.mkdir()
        else:
            for i in count(100):
                if not os.path.isdir(Path(str(folder_path) + str(i))):
                    folder_path = Path(str(folder_path) + str(i))
                    folder_path.mkdir()
                    break
        try:
            image_suffixes = [str(k).zfill(5) for k in list(range(len(group['data']['img_gid_q'])))]

        except:
            return

        img_shape = group['data']['img_gid_q'][0].shape
        diagonal_length_img = (img_shape[0] ** 2 + img_shape[1] ** 2) ** (1 / 2)
        diagonal_length_q = (group['data']['q_xy'][-1] ** 2 + group['data']['q_z'][-1] ** 2) ** (
                    1 / 2)
        pixel_per_angstroem = diagonal_length_img / diagonal_length_q
        for img_nr, image in enumerate(group['data']['img_gid_q']):
            img_name = folder_name + str(image_suffixes[img_nr])
            folder_key = self.fm.add_root_path_to_project(folder_path)
            self.fm.geometries.default[folder_key] = Geometry.fromdict(dict(''))
            img_name = keys.validate_filename(img_name)
            img_path = folder_path / str(img_name + '.tiff')
            Image.fromarray(image).save(img_path, 'tiff')
            img_key = ImagePathKey(project_path, folder_key, path=img_path, idx=img_nr)
            folder_key._image_children.append(img_key)

            attempts = [self.import_peaks_fitted, self.import_peaks_detected, self.deprecated_import_peaks]
            try:
                for func in attempts:
                    try:
                        roi_dict = func(group, folder_name, img_nr, img_key, pixel_per_angstroem)
                        break
                    except:
                        pass

                self.fm.rois_data[img_key] = RoiData.from_dict(roi_dict)
            except:
                pass


    def copy_prototype(self, source, project_path):
        def deep_copy(source, destination):
            with File(destination, 'w') as f_dest:
                with File(source, 'r') as f_src:
                    for folder_name, group in f_src.items():
                        try:
                            f_src.copy(f_src[folder_name], f_dest, folder_name)
                            del f_dest[folder_name]['data']['img_gid_q']
                        except:
                            pass

        deep_copy(source, str(project_path) + '/prototype_intermediate.h5')
        deep_copy(str(project_path) + '/prototype_intermediate.h5' , str(project_path) + '/prototype.h5')
        Path(str(project_path) + '/prototype_intermediate.h5').unlink()

    def import_peaks(self, group, folder_name, img_nr, img_key, q_to_xy_factor, ds_name):
        img_key.set_qxy(group['data']['q_xy'][-1])
        img_key.set_qz(group['data']['q_z'][-1])
        detected_peaks = list(group['data']['analysis'].items())[img_nr][1][ds_name]
        roi_dict = {
            'radius': detected_peaks['radius'] * q_to_xy_factor,
            'radius_width': detected_peaks['radius_width'] * q_to_xy_factor,
            'angle': detected_peaks['angle'],
            'angle_width': detected_peaks['angle_width'],
            'key': detected_peaks['id'],
            'type': np.full((len(detected_peaks['is_ring'])), 2) - detected_peaks['is_ring'],
            'confidence_level': detected_peaks['visibility'] * 0 + .1,
            'score': detected_peaks['score'],
            'cif_file': np.full((len(detected_peaks['id'])), 'not_set')
        }
        return roi_dict

    def import_peaks_detected(self, group, folder_name, img_nr, img_key, q_to_xy_factor):
        return self.import_peaks(group, folder_name, img_nr, img_key, q_to_xy_factor,  'detected_peaks')

    def import_peaks_fitted(self, group, folder_name, img_nr, img_key, q_to_xy_factor):
        return self.import_peaks(group, folder_name, img_nr, img_key, q_to_xy_factor, 'fitted_peaks')

    def deprecated_import_peaks(self, group, folder_name, img_nr, img_key, q_to_xy_factor, ds_name):
        roi_dict = {
            k: v[()] for k, v in
            list(list(group['data']['analysis'].items())[img_nr][1]['fitted_peaks'].items())
        }
        roi_dict['angle_width'] = roi_dict['angle_std']
        del roi_dict['angle_std']
        roi_dict['radius_width'] = roi_dict['width']
        del roi_dict['width']
        return roi_dict