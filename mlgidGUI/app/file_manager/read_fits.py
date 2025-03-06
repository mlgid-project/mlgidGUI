from datetime import datetime as dt
from pathlib import Path

from .object_file_manager import _ObjectFileManager
from .keys import RemoveWeakrefs

class _ReadFits(_ObjectFileManager):
    NAME = 'fits'

    def init(self):
        path = self.project_structure.path

        if path:
            self.folder: Path = path / self.NAME

    def _get_path(self, fit_key: tuple):
        key, name = fit_key
        return self.folder / key.file_name(name)

    def __getitem__(self, item):
        key, name = item
        # if not key.is_project:
        #     return self._get_pickle(self._get_path(key))
        # if self.project_structure.config[key.h5path]:
        #     return self._process_h5_group(key.h5path, key.h5key, self._get_h5, key)
        return self._get_pickle(self._get_path(item))
        # if res is not None:
        #     return res
        # return self._process_h5_group(key.h5path, key.h5key, self._get_h5, key)

    def __delitem__(self, item):
        key, name = item
        # if not key.is_project:
        #     return self._del_pickle(self._get_path(key))
        # if self.project_structure.config[key.h5path]:
        #     return self._process_h5_group(key.h5path, key.h5key, self._del_h5, key)
        # else:
        return self._del_pickle(self._get_path(item))

    def __setitem__(self, item, value):
        # key, name = item
        # if key.is_project and self.project_structure.config[key.h5path]:
        #     return self._process_h5_group(key.h5path, key.h5key, self._set_h5, key, value)
        # else:
        try:
            return self._set_pickle(self._get_path(item), value)
        except Exception as err:
            self.log.exception(err)
            return

    def get_multi_fit(self):
        return MultiFitFileManager(self.project_structure)


class MultiFitFileManager(_ObjectFileManager):
    NAME = 'fits'

    def __init__(self, project_structure):
        super().__init__(project_structure)
        self.folder: Path = self.folder / dt.now().strftime('multi fit %d %m %y - %H %M %S')
        self.folder.mkdir()

    def __getitem__(self, item):
        fit_object = super().__getitem__(item)
        if fit_object:
            fit_object.image_key = item
            return fit_object

    def __setitem__(self, key, value):
        with RemoveWeakrefs(value.image_key):
            super().__setitem__(key, value)

    def delete(self):
        for path in self.folder.iterdir():
            path.unlink()
        self.folder.rmdir()
