import pickle
from dataclasses import dataclass
from collections import defaultdict
import logging
from pathlib import Path
from os import listdir

from .keys import (AbstractKey, FolderKey, FolderH5Key, FolderPathKey,
                   ImageKey, ImagePathKey, H5_FORMAT, RemoveWeakrefs, CIFFileKey,
                   AVAILABLE_IMAGE_FORMATS, CIF_EXTENSIONS)


class ProjectRootKey(FolderKey):
    def __init__(self, *, project_path: Path = None, h5source_path: Path = None):
        super().__init__(project_path= project_path, path=project_path, parent=None)
        self.path: Path = project_path
        self.h5_path: Path = h5source_path

        self._valid = True

        if self.h5_path:
            self._project_name = self.h5_path.name
        elif self.path:
            self._project_name = self.path.name
        else:
            self._project_name = 'EMPTY PROJECT'
            self._valid = False

        self.update()

    def is_valid(self) -> bool:
        return self._valid

    def __eq__(self, other):
        if isinstance(other, ProjectRootKey):
            return self.h5_path == other.h5_path and self.path == other.path

    def __hash__(self):
        if self.h5_path is not None:
            return hash(self.h5_path)
        if self.path is not None:
            return hash(self.path)

    def _file_key(self) -> str:
        return self.name

    @property
    def name(self):
        return self._project_name

    @property
    def parent(self):
        return

    def add_path(self, path: Path):
        if path.is_dir():
            folder_key = FolderPathKey(project_path = self.path, parent = self, path=path)
            for f in self._folder_children:
                if folder_key.path == f.path:
                    return f, False
            self._folder_children.append(folder_key)
            for image in listdir(folder_key.path):
                if (image.endswith(AVAILABLE_IMAGE_FORMATS)):
                    image_key = ImagePathKey(project_path=self.path, parent=folder_key, path=Path(image))
                    folder_key._image_children.append(image_key)
            return folder_key, True
        elif path.is_file() and path.suffix in AVAILABLE_IMAGE_FORMATS:
            key = ImagePathKey(project_path = self.path, parent = self, path=path)
            for c in self._image_children:
                if key.path == c.path:
                    return c, False
            self._image_children.append(key)
            return key, True
        elif path.is_file() and path.suffix in CIF_EXTENSIONS:
            key = CIFFileKey(project_path = self.path, parent = self, path=path)
            for c in self._cif_files:
                if key.path == c.path:
                    key = CIFFileKey(project_path=self.path, parent=self, path=path, duplicate_num=c.duplicate_num + 1)
            self._cif_files.append(key)
            return key, True
        elif path.is_file() and path.suffix in H5_FORMAT:
            key = FolderH5Key(project_path = self.path, parent =self, h5path=path)
            self._folder_children.append(key)
            return key, True

    def remove_key(self, key: AbstractKey):
        if isinstance(key, FolderKey):
            try:
                self._folder_children.remove(key)
            except ValueError:
                return
        elif isinstance(key, ImageKey):
            try:
                self._image_children.remove(key)
            except ValueError:
                return
        elif isinstance(key, CIFFileKey):
            try:
                self._cif_files.remove(key)
            except ValueError:
                return

    def expand_tree(self, folder_key: FolderKey = None):
        folder_key = folder_key or self

        for child_folder in folder_key.folder_children:
            child_folder.update()
            self.expand_tree(child_folder)


@dataclass
class H5ProjectConfig:
    save_to_h5: bool = False


class ProjectStructure(object):
    log = logging.getLogger(__name__)

    def __init__(self):
        self._root: ProjectRootKey or None = None
        self.config = defaultdict(lambda: False)
        self.path: Path or None = None

    @property
    def project_opened(self):
        return self._root is not None

    @property
    def root(self) -> ProjectRootKey or None:
        return self._root

    def save(self, restore: bool = True):
        if self.path and self.root and self.path.is_dir():
            with RemoveWeakrefs(self.root, restore=restore):
                with open(str(self.path.resolve() / 'project_structure'), 'wb') as f:
                    pickle.dump(self.root, f)

    def open_project(self, path: Path):
        self.close_project()

        self.path = path

        if self.path.is_file():
            raise ValueError(f'Project path is a file: {self.path.resolve()}')

        self.path.mkdir(exist_ok=True, parents=True)

        pickle_file = self.path / 'project_structure'

        if pickle_file.is_file():
            try:
                with open(str(pickle_file.resolve()), 'rb') as f:
                    self._root = pickle.load(f)
                RemoveWeakrefs.restore(self._root)
            except Exception as err:
                pickle_file.unlink()
                self.log.exception(err)

        if not self._root:
            self._root = ProjectRootKey(project_path=self.path)

    def save_and_close(self):
        self.save(False)
        self.close_project()

    def close_project(self):
        self._root = None
        self.path = None
        self.config = defaultdict(lambda: False)
