import pytest
from typing import List, Dict, Generator
from pathlib import Path
from dataclasses import dataclass
import shutil
from mlgidGUI.app.file_manager import ImageKey, FolderKey, ImagePathKey, FolderPathKey


__all__ = ['empty_project', 'project_1', 'ProjectInfo']



def lazy_property(func):
    name = f'__lazy_property_{func.__name__}'

    @property
    def wrapper(self):
        if not hasattr(self, name):
            setattr(self, name, func(self))
        return getattr(self, name)

    return wrapper


@dataclass
class ProjectInfo:
    name: str
    root_path: Path
    path_tree: dict

    def root_key(self) -> FolderPathKey:
        root_key = FolderPathKey(None, path=self.root_path, parent=None)
        root_key.update()
        return root_key

    def all_image_keys(self) -> Generator[ImagePathKey, None, None]:
        paths_dict = self.paths_dict()
        for image_key_list in paths_dict.values():
            yield from image_key_list

    def paths_dict(self) -> Dict[FolderKey, List[ImageKey]]:
        root_folder_key = self.root_key()
        paths_dict: Dict[FolderKey, List[ImageKey]] = {}
        self._fill_dict(paths_dict, root_folder_key)
        return paths_dict

    def _fill_dict(self, paths_dict: Dict[FolderKey, List[ImageKey]], folder_key: FolderPathKey):
        child_images = list(self.get_image_keys_by_folder_key(folder_key))
        if child_images:
            paths_dict[folder_key] = child_images

        for p in folder_key.path.iterdir():
            if p.is_dir():
                child_folder_key = FolderPathKey(folder_key, path=p, parent=folder_key)
                self._fill_dict(paths_dict, child_folder_key)

    @staticmethod
    def get_image_keys_by_folder_key(folder_key: FolderPathKey) -> Generator[ImagePathKey, None, None]:
        yield from (ImagePathKey(folder_key.project_path, path=p, parent = folder_key) for p in folder_key.path.iterdir() if p.suffix == '.tiff')


def _get_project_info(project_path, project_name: str):
    root_path = project_path
    path_tree = _get_path_tree(root_path)
    return ProjectInfo(project_name, root_path, path_tree)


def _get_path_tree(root_path: Path, *, _d: dict = None) -> dict:
    if not _d:
        _d = {}

    _d[root_path] = []

    for p in root_path.iterdir():
        if p.is_dir():
            _d[root_path].append(_get_path_tree(p, _d=_d))
    for p in root_path.iterdir():
        if p.is_file():
            _d[root_path].append(p)
    return _d

@pytest.fixture
def empty_project(tmpdir_factory):

    project_path, project_dir = _start_project(tmpdir_factory, 'empty_project_data')
    shutil.copytree((Path(__file__).parents[1] / 'data' / 'empty_project_data'), project_path, dirs_exist_ok=True)
    yield _get_project_info(project_path, 'empty_project_data')


@pytest.fixture
def project_1(tmpdir_factory):
    project_path, project_dir = _start_project(tmpdir_factory, 'project_1_data')
    shutil.copytree((Path(__file__).parents[1] / 'data' / 'project_1_data'), project_path, dirs_exist_ok=True)
    yield _get_project_info(project_path, 'project_1_data')



def _start_project(tmpdir_factory, name: str) -> tuple:

    project_dir = tmpdir_factory.mktemp(name)
    project_path = Path(project_dir.strpath)
    return project_path, project_dir