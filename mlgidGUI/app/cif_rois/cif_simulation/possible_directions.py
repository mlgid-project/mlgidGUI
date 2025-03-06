from itertools import product
import numpy as np


def get_unique_directions(idx_min: int, idx_max: int,
                          dtype: np.dtype = np.float32
                          ) -> np.ndarray:
    """ Generate possible orientations of crystal growth """
    direct = np.array(_get_all_directions(idx_min, idx_max), dtype=dtype)
    direct_norm = direct / np.linalg.norm(direct, axis=1)[..., np.newaxis]
    dir_unique = np.unique(direct_norm, axis=0)
    return dir_unique


def _get_all_directions(idx_min: int, idx_max: int):
    return list(filter(any, product(list(range(idx_min, idx_max + 1)), repeat=3)))


if __name__ == "__main__":
    directions = get_unique_directions(0, 5)
