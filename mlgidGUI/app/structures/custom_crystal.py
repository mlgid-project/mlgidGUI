# -*- coding: utf-8 -*-

from typing import List
from crystals import Crystal

from .crystal_ring import CrystalRing
from .get_ring_list import get_crystal_rings, get_sorted_rings, RingDict


class CustomCrystal(Crystal):
    def __init__(self, unitcell, lattice_vectors, source=None, **kwargs):
        super().__init__(unitcell, lattice_vectors, source=source, **kwargs)
        self._key = f'{self.chemical_formula}{self.source}'
        self._ring_dict: RingDict = {}
        self._is_updated: bool = False
        self._q_max: float = 5

    @classmethod
    def from_crystal(cls, crystal: Crystal):
        return cls(crystal.unitcell, crystal.lattice_vectors, crystal.source)

    @property
    def is_updated(self) -> bool:
        return self._is_updated

    def update_rings(self, *args, **kwargs) -> None:
        if not self._is_updated:
            self._update_rings(*args, **kwargs)
            self._is_updated = True

    @property
    def rings(self) -> List[CrystalRing]:
        return get_sorted_rings(self._ring_dict, self._q_max)

    @property
    def key(self) -> str:
        return self._key

    def set_q_max(self, q_max) -> None:
        if q_max > self._q_max:
            self._is_updated = False
        self._q_max = q_max

    def _update_rings(self, *args, **kwargs) -> None:
        self._ring_dict = get_crystal_rings(self, self._q_max, ring_dict=self._ring_dict, *args, **kwargs)
