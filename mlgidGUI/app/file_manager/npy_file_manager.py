from pathlib import Path

import numpy as np

from .object_file_manager import _ObjectFileManager


class _ReadNpy(_ObjectFileManager):
    @staticmethod
    def _set_pickle(path: Path, value):
        np.save(str(path.resolve()) + '.npy', value)

    @staticmethod
    def _get_pickle(path: Path):
        if path.is_file():
            return np.load(str(path.resolve()) + '.npy', allow_pickle=True)

