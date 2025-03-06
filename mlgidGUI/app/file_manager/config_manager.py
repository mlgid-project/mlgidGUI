import pickle
import logging
from typing import List
import shutil
from pathlib import Path


class _GlobalConfigManager(object):
    config_folder: Path
    log = logging.getLogger(__name__)

    def __init__(self, config_path: Path = None):
        if config_path:
            self.config_folder = config_path
        if not self.config_folder.is_dir():
            self.config_folder.mkdir(parents=True)
        self._saved_projects_path = str((Path.home() /'GIWAXSProjects').resolve())
        self._copy_to_new_dir()

    def get_project_paths(self) -> List[Path]:
        project_path = (Path.home() / 'GIWAXSProjects').resolve()
        try:
            return [f for f in project_path.iterdir() if f.is_dir()]
        except:
            return []


    def add_project_path(self, path: Path):
        paths: List[Path] = self.get_project_paths()
        paths.append(path)
        with open(self._saved_projects_path, 'wb') as f:
            pickle.dump(paths, f)

    def __getitem__(self, item):
        try:
            with open(str((self.config_folder / item).resolve()), 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return
        except Exception as err:
            self.log.exception(err)
            return

    def __setitem__(self, key, value):
        with open(str((self.config_folder / key).resolve()), 'wb') as f:
            pickle.dump(value, f)

    def __delitem__(self, key):
        path = self.config_folder / key
        if path.is_file():
            path.unlink()

    def _copy_to_new_dir(self):
        _OLD_CONFIG_PATH: Path = Path(__file__).parents[2] / 'config'
        _OLD_CONFIG_STR: str = str(_OLD_CONFIG_PATH.resolve())
        _OLD_SAVED_PROJECTS: str = str((_OLD_CONFIG_PATH / 'recent_projects').resolve())

        if not _OLD_CONFIG_PATH.is_dir():
            return

        new_config_str_path: str = str(self.config_folder.resolve())

        for p in _OLD_CONFIG_PATH.iterdir():
            if p.name != 'recent_projects':
                shutil.copy(p, new_config_str_path)
                p.unlink()

        try:
            with open(_OLD_SAVED_PROJECTS, 'rb') as f:
                old_project_paths = pickle.load(f)
            self.update_project_paths(old_project_paths + self.get_project_paths())
            Path(_OLD_SAVED_PROJECTS).unlink()
        except FileNotFoundError:
            pass

        _OLD_CONFIG_PATH.rmdir()
