# -*- coding: utf-8 -*-


from typing import Tuple


class CrystalRing(object):
    def __init__(self,  radius: float, miller_indices: Tuple[int, int, int],
                 intensity: float, crystal: 'CustomCrystal', selected: bool = False):
        self.radius = radius
        self.miller_indices = miller_indices
        self.intensity = intensity
        self.crystal = crystal
        self.key = f'{self.miller_indices}{self.crystal.key}'
        self.selected: bool = selected

    def __repr__(self):
        return f'Ring(radius={self.radius}, miller_indices={self.miller_indices}, intensity={self.intensity},' \
               f'source={self.crystal.source})'
