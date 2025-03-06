import logging
import pickle
from pathlib import Path

from h5py import Group, File

from ..utils import InternalError
from .keys import ImageKey, AbstractKey, ImageH5Key
from .project_structure import ProjectStructure


def _check_empty_project(func):
    def wrapper(self, *args, **kwargs):
        if self.project_structure.project_opened:
            return func(*args, **kwargs)
        else:
            return

    return wrapper


class _ObjectFileManager(object):
    log = logging.getLogger(__name__)

    NAME = ''

    def __init__(self, project_structure: ProjectStructure):
        self.project_structure = project_structure
        self.folder: Path = None
        self.init()

    def init(self):
        path = self.project_structure.path

        if path:
            self.folder: Path = path / self.NAME
            self.folder.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: AbstractKey) -> Path:
        return self.folder / key.file_name()

    @staticmethod
    def _set_pickle(path: Path, value):
        with open(str(path.resolve()), 'wb') as f:
            pickle.dump(value, f)

    @staticmethod
    def _get_pickle(path: Path):
        if path.is_file():
            with open(str(path.resolve()), 'rb') as f:
                return pickle.load(f)

    @staticmethod
    def _del_pickle(path: Path):
        if path.is_file():
            path.unlink()

    @staticmethod
    def get_h5(h5group: Group, key: ImageKey):
        pass

    @staticmethod
    def set_h5(h5group: Group, key: ImageKey, *args, **kwargs):
        pass

    @staticmethod
    def del_h5(h5group: Group, key: ImageKey):
        pass

    def __getitem__(self, key):
        return self._get_pickle(self._get_path(key))

    def __delitem__(self, key):
        return self._del_pickle(self._get_path(key))

    def __setitem__(self, key, value):
        return self._set_pickle(self._get_path(key), value)
