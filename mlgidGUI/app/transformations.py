from enum import Enum

import numpy as np

__all__ = ['TransformationsHolder', 'Transformation', 'UnknownTransformation']


class UnknownTransformation(ValueError):
    pass


class Flip(object):
    __slots__ = ('axis',)

    def __init__(self, axis: int):
        self.axis = axis

    def __call__(self, img: np.ndarray):
        return np.flip(img, self.axis)


class Rotate(object):
    __slots__ = ('k',)

    def __init__(self, k: int):
        self.k = k

    def __call__(self, img: np.ndarray):
        return np.rot90(img, self.k)


class Transformation(Enum):
    horizontal_flip = Flip(1)
    vertical_flip = Flip(0)
    rotate_right = Rotate(-1)
    rotate_left = Rotate(1)


_T_DICT = {
    '1234': (),
    '1324': (Flip(1), Rotate(1)),
    '2413': (Rotate(1),),
    '2143': (Flip(1),),
    '3142': (Rotate(-1),),
    '3412': (Flip(0),),
    '4321': (Flip(0), Flip(1)),
    '4231': (Rotate(1), Flip(1))
}


class TransformationsHolder(object):
    def __init__(self, key: str = None):
        self._img = np.array([[1, 2], [3, 4]])
        if key:
            self.update(key)

    def __call__(self, image):
        for op in self._operations:
            image = op(image)
        return image

    def add(self, op: Transformation):
        self._img = op.value(self._img)

    def clear(self):
        self._img = np.array([[1, 2], [3, 4]])

    @property
    def _operations(self) -> tuple:
        return _T_DICT[self.key]

    @property
    def key(self):
        return ''.join(map(str, self._img.ravel()))

    def update(self, key: str):
        self._img = np.array([[1, 2], [3, 4]])
        try:
            for op in _T_DICT[key]:
                self._img = op(self._img)
        except KeyError:
            raise UnknownTransformation(f'Key {key} doesn\'t correspond to any known transformation.')
