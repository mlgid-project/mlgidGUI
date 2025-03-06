from typing import Dict, List
from pathlib import Path
from os import path
import logging
from shutil import copytree, copyfile
import math

from ..file_manager import FileManager, ImageKey, FolderKey, keys, ImagePathKey
from .load_data import (FolderData, FolderDataFlags,
                        ImageData, ImageDataFlags, load_image_data, load_folder_data)
from ..image_holder import ImageHolder
from .import_from_h5 import ImportProjectFromH5
from .saving_parameters import SavingParameters, SaveFormats
from .save_h5 import SaveH5
from .save_dataset import save_for_object_detection

class DataManager(object):
    def __init__(self, fm: FileManager, image_holder: ImageHolder):
        self.last_save_successfull = False
        self._fm = fm
        self._image_holder = image_holder
        self._log = logging.getLogger(__name__)
        self._save_h5 = SaveH5(fm, image_holder)

    def save(self, params: SavingParameters):
        if params.format.value == SaveFormats.entire_h5.value:
            self.last_save_successfull = self._save_h5.save(params)
        elif params.format.value == SaveFormats.partial_h5.value:
            self.last_save_successfull = self._save_h5.save(params)
        elif params.format.value == SaveFormats.object_detection.value:
            self.last_save_successfull = save_for_object_detection(self._fm, self._image_holder, params)
        else:
            raise ValueError("Save format is not yet supported.")


    def project2h5(self,
                   dest: str or Path,
                   skip_empty_images: bool = True,
                   save_image=True,
                   save_polar_image=True,
                   save_geometries=True,
                   save_roi_keys=True,
                   save_roi_metadata=True,
                   **kwargs):

        dest = Path(dest)

        if not dest.parent.is_dir():
            raise NotADirectoryError(f"Folder {str(dest.parent)} does not exist.")

        if dest.is_file():
            raise FileExistsError(f"File {str(dest)} already exists.")

        params = SavingParameters(
            self.get_paths_dict(self._fm.root, skip_empty_images=skip_empty_images),
            dest,
            save_image=save_image,
            save_polar_image=save_polar_image,
            save_geometries=save_geometries,
            save_roi_keys=save_roi_keys,
            save_roi_metadata=save_roi_metadata,
            **kwargs
        )
        self.save(params)

    def load_project_from_h5(self, h5_path, project_path):
        return ImportProjectFromH5(self._fm).load(project_path, h5_path)

    def import_image(self, source_img_path: Path):
        #self._fm.open_project(self._fm._project_folder)

        # create source images folder if not already created
        dest_img_folder = self._fm._project_folder / 'images'
        dest_img_folder.mkdir(exist_ok=True)

        # save imported image in the source images folder
        img_name = keys.validate_filename(path.basename(source_img_path)) # extract filename from sourcepath and change to allowed filename
        _, extension = path.splitext(source_img_path)
        dest_img_path = dest_img_folder / str(img_name + extension)
        copyfile(source_img_path, dest_img_path)

        img_key = self._fm.add_root_path_to_project(dest_img_path)
        self._fm.change_image(img_key)

    def import_cif(self, source_cif_path: Path):
        # create source folder if not already created
        dest_cif_folder = self._fm._project_folder / 'CIF_files'
        dest_cif_folder.mkdir(exist_ok=True)

        # save imported cif in the source folder
        cif_file_name = keys.validate_filename(path.basename(source_cif_path)) # extract filename from sourcepath and change to allowed filename
        _, extension = path.splitext(source_cif_path)
        dest_cif_path = dest_cif_folder / str(cif_file_name + extension)
        copyfile(source_cif_path, dest_cif_path)

        self._fm.add_root_path_to_project(dest_cif_path)

    def import_folder(self, source_path: Path):
        if not source_path.is_dir():
            self._log.exception('provided folder path is not a directory')
            return None

        self._fm.open_project(self._fm._project_folder)
        folder_name = path.basename(source_path)
        destination_path = self._fm._project_folder / 'images' / folder_name

        try:
            copytree(source_path, destination_path, dirs_exist_ok=True)
        except Exception as e:
            self._log.exception('could not import folder, maybe it already exists')
            return None

        self._fm.add_root_path_to_project(destination_path)

    def estimate_folder_size(self, source_path: Path):
        try:
            size_bytes = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

        except Exception as e:
            self._log.exception('could not estimate folder size')
            return ""


    def load_image_data(self, key: ImageKey,
                        flags: ImageDataFlags = ImageDataFlags.ALL) -> ImageData:
        return load_image_data(self._fm, self._image_holder, key, flags)

    def load_folder_data(self, key: FolderKey,
                         flags: FolderDataFlags = FolderDataFlags.ALL) -> FolderData:
        return load_folder_data(self._fm, self._image_holder, key, flags)

    def get_paths_dict(self,
                       folder_key: FolderKey = None,
                       skip_empty_images: bool = True,
                       ) -> Dict[FolderKey, List[ImageKey]]:
        folder_key = folder_key or self._fm.root
        paths_dict: Dict[FolderKey, List[ImageKey]] = {}
        _fill_paths_dict(paths_dict, folder_key, skip_empty_images, self._fm)
        return paths_dict


def _fill_paths_dict(
        paths_dict: Dict[FolderKey, List[ImageKey]],
        folder_key: FolderKey,
        skip_empty_images: bool,
        fm: FileManager,
):
    if folder_key.images_num:
        img_keys = list(folder_key.image_children)

        if skip_empty_images:
            img_keys = [key for key in img_keys if fm.rois_data[key]]

        paths_dict[folder_key] = img_keys

    for folder_child_key in folder_key.folder_children:
        _fill_paths_dict(paths_dict, folder_child_key, skip_empty_images, fm)
