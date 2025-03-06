import logging
from pathlib import Path
from typing import List

from h5py import Group

from PyQt5.QtCore import pyqtSignal, QObject

from .keys import (FolderKey, FolderH5Key, FolderPathKey, RemoveWeakrefs,
                   ImageKey, ImageH5Key, ImagePathKey, InvalidKey, CIFFileKey,
                   PROJECT_KEY, IMAGE_PROJECT_KEY, GLOB_IMAGE_FORMATS)

from .project_structure import ProjectStructure, ProjectRootKey
from .read_images import _ReadImage, _ReadNpy
from .read_polar_images import _ReadPolarImage
from .read_geometry import _ReadGeometry
from .read_roi_data import _ReadRoiData
from .read_meta_roi import _ReadMetaData
from .read_radial_profile import _ReadRadialProfile
from .config_manager import _GlobalConfigManager
from .read_fits import _ReadFits
from appdirs import user_data_dir
from importlib import metadata


class FileManager(QObject):
    sigActiveImageChanged = pyqtSignal(object)
    sigActiveFolderChanged = pyqtSignal(object)
    sigProjectClosed = pyqtSignal()
    sigProjectIsClosing = pyqtSignal()
    sigProjectOpened = pyqtSignal()
    sigNewFolder = pyqtSignal(object)
    sigNewFile = pyqtSignal(object)
    sigNewCIFFile = pyqtSignal()
    sigRedrawCIFPeaks = pyqtSignal(object)

    log = logging.getLogger(__name__)

    def __init__(self):
        config_path: Path = Path(user_data_dir('mlgidGUI', metadata.metadata('mlgidGUI')['Author']))
        QObject.__init__(self)
        self.config: _GlobalConfigManager = _GlobalConfigManager(config_path)
        self.recent_projects: List[Path] = self.config.get_project_paths()
        self._project_folder: Path or None = None
        self._project_structure: ProjectStructure = ProjectStructure()
        self.project_name: str = None
        self._current_key: ImageKey or None = None
        self.recent_projects = [p for p in self.recent_projects if p.is_dir()]

    @property
    def current_key(self) -> ImageKey or None:
        return self._current_key

    def open_latest_available_project(self):
        self.close_project()
        while self.recent_projects:
            path = self.recent_projects.pop(-1)
            if path.is_dir():
                try:
                    self.open_project(path)
                    return
                except Exception as err:
                    self.log.exception(err)

    @property
    def project_opened(self) -> bool:
        return self._project_structure.project_opened

    @property
    def root(self) -> ProjectRootKey or None:
        return self._project_structure.root

    @property
    def project_path(self) -> Path or None:
        return self._project_folder

    def change_image(self, key: ImageKey) -> None:
        if key != self._current_key:
            if not key or not self._current_key or key.parent != self._current_key.parent:
                new_parent = key.parent if key else None
                self.sigActiveFolderChanged.emit(new_parent)

            self._current_key = key
            self.sigActiveImageChanged.emit(key)

    def open_project(self, path: Path) -> None:
        self.close_project()
        self._init_project(path)
        self.sigProjectOpened.emit()

    def add_root_path_to_project(self, path: Path) -> None:
        if not self.project_opened:
            return
        key, isnew = self._project_structure.root.add_path(path)
        if isnew:
            if isinstance(key, ImageKey):
                self.sigNewFile.emit(key)
            elif isinstance(key, FolderKey):
                self.sigNewFolder.emit(key)
        if isinstance(key, CIFFileKey):
            self.sigNewCIFFile.emit()
        return key

    def remove_key(self, key) -> None:
        self._project_structure.root.remove_key(key)
        if isinstance(key, ImageKey):
            self._delete_image_data(key)
            if key.parent:
                key.parent.remove_image(key)
        else:
            self._delete_folder(key)
        if self._current_key in key:
            self.change_image(None)

    def _delete_image_data(self, key: ImageKey):
        del self.geometries[key]
        del self.rois_data[key]
        del self.images[key]
        del self.polar_images[key]


    def _delete_folder(self, key: FolderKey):
        for folder in key.folder_children:
            self._delete_folder(folder)
        for image in key.image_children:
            self._delete_image_data(image)

    def close_project(self):
        if self.project_opened:
            self.sigProjectIsClosing.emit()
            self._project_structure.save_and_close()
            if self._project_folder not in self.recent_projects:
                self.recent_projects.append(self._project_folder)
            self._project_folder = None
            self.project_name = None
            self.sigProjectClosed.emit()

    def delete_project(self, project_path: Path):
        if self._project_folder == project_path:
            raise ValueError('Cannot delete opened project.')

        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)

        _delete_project(project_path)

    def init_file_viewer(self):

        for key in self._project_structure.root.folder_children:
            self.sigNewFolder.emit(key)
        current_image_key = None
        for key in self._project_structure.root.image_children:
            if not key.path.exists():
                continue
            self.sigNewFile.emit(key)
            current_image_key = key
        if current_image_key is not None:
            self.change_image(current_image_key)
        for key in self._project_structure.root.cif_files:
            if not key.path.exists():
                self._project_structure.root.remove_key(key)
        if self._project_structure.root.cif_files:
            self.sigNewCIFFile.emit()

    def _init_project(self, path: Path):
        self._project_folder = path
        self._project_folder.mkdir(parents=False, exist_ok=True)
        self._project_structure.open_project(path)
        self.images: _ReadImage = _ReadImage(self._project_structure)
        self.geometries: _ReadGeometry = _ReadGeometry(self._project_structure)
        self.polar_images: _ReadPolarImage = _ReadPolarImage(self._project_structure)
        self.rois_data: _ReadRoiData = _ReadRoiData(self._project_structure)
        self.rois_meta_data: _ReadMetaData = _ReadMetaData(self._project_structure)
        self.fits: _ReadFits = _ReadFits(self._project_structure)
        #self.fits = None
        #self.profiles = None
        self.profiles: _ReadRadialProfile = _ReadRadialProfile(self._project_structure)

        self.project_name = self._project_folder.name
        self.init_file_viewer()


    def __len__(self):
        return len(self.paths)

    def close(self):
        self.close_project()

def _give_h5_name(h5group: Group, name):
    if name not in h5group.keys():
        return name
    num = 0
    while True:
        new_name = '_'.join((name, str(num)))
        if new_name not in h5group.keys():
            return new_name
        num += 1


def _delete_project(project_path: Path):
    raise NotImplementedError
