# -*- coding: utf-8 -*-
import logging
from itertools import product
from typing import Dict, Tuple, Union, List, Callable
import cmath
import numpy as np

from crystals import Crystal
from periodictable import cromermann as cr_ff

from .crystal_ring import CrystalRing

MillerIndices = Tuple[int, int, int]
RingDict = Dict[MillerIndices, Union[CrystalRing, None]]
RingCalculationCallback = Callable[[int], bool]

logger = logging.getLogger(__name__)


def get_sorted_rings(ring_dict: RingDict, q_max: float) -> List[CrystalRing]:
    return sorted(list(filter(lambda ring: ring and ring.radius <= q_max, ring_dict.values())),
                  key=lambda ring: ring.radius)


def get_crystal_rings(crystal: Crystal,
                      q_max: float,
                      min_sf: float = 0.01,
                      max_num: int = 10,
                      ring_dict: RingDict = None,
                      process_callback: RingCalculationCallback = None,
                      set_max_callback: RingCalculationCallback = None,
                      ) -> RingDict:
    miller_indices: MillerIndices
    min_num: int
    process_callback = process_callback or _empty_callback()
    set_max_callback = set_max_callback or _empty_callback()

    if ring_dict:
        min_num = max(map(max, ring_dict.keys()))
    else:
        min_num = 0
        ring_dict: RingDict = {}

    for h in range(min_num, max_num):
        miller_indices = (h, 0, 0)
        ring = _get_ring(crystal, miller_indices, min_sf)
        logger.debug(f'Calc miller indices {miller_indices}')
        if ring and ring.radius > q_max:
            max_num = h
            break

        ring_dict[miller_indices] = ring

    set_max_callback(max_num ** 3)

    for i, miller_indices in enumerate(product(list(range(min_num, max_num)), repeat=3)):
        logger.debug(f'Calc miller indices {miller_indices}')
        if miller_indices not in ring_dict:
            ring_dict[miller_indices] = _get_ring(crystal, miller_indices, min_sf)
        process_callback(i)

    return ring_dict


def _empty_callback() -> RingCalculationCallback:
    def func(idx: int):
        return True

    return func


def _get_ring(crystal: Crystal, miller_indices: MillerIndices, min_sf: float) -> Union[CrystalRing, None]:
    scattering_vector = crystal.scattering_vector(miller_indices)
    q = np.linalg.norm(scattering_vector)
    sf = _calc_structure_factor(q, miller_indices, crystal) * _structure_factor_coef(miller_indices)
    if sf >= min_sf:
        return CrystalRing(radius=q, miller_indices=miller_indices, intensity=sf, crystal=crystal)


def _structure_factor_coef(miller_indices: MillerIndices) -> int:
    return 2 ** sum(map(bool, miller_indices))


def _calc_structure_factor(q: float, miller_indices: MillerIndices, crystal: Crystal) -> float:
    sf = 0
    for atom in crystal:
        ff = cr_ff.fxrayatq(atom.element, q, charge=None)
        sf = sf + ff * cmath.exp(2 * np.pi * 1j * np.dot(miller_indices, atom.coords_fractional))
    return abs(sf)
