import numpy as np
from typing import Union

from xrayutilities.materials.cif import CIFFile
from xrayutilities.materials.spacegrouplattice import SGLattice

from .experiment import ExpParameters
from .q_sim import QPos
from .int_sim import Intensity


class Crystal:
    """
    A class to represent the crystal structure
    ...
    Attributes
    ----------
    spgr : int
        space group number (1-230)
    cr_group : int
        crystal group number (1-7)
    lat_par : np.ndarray
        lattice parameters tensor([a,b,c,α,β,γ])
    param_SGL : np.ndarray
        unique lattice parameters that are needed to receive miller indices
    atoms : np.ndarray
        list of elements in the structure
    atoms_positions: np.ndarray
        relative atom coordinates (related with the atoms attribute)
    """

    def __init__(self,
                 spgr: Union[str, int],  # 1-230
                 lat_par: np.ndarray,  # np.array([a,b,c,α,β,γ], dtype=np.float32)
                 atoms: Union[np.ndarray, None] = None,  # list of elements in the structure,
                 atoms_positions: Union[np.ndarray, None] = None  # relative atom coordinates, dtype=np.float32
                 ):
        self.spgr = spgr
        self.cr_group = self._cr_group()
        self.lat_par = lat_par
        self.param_SGL = self._parameters_for_SGLattice()
        self.atoms = atoms
        self.atoms_positions = atoms_positions

    def _cr_group(self) -> Union[str, int]:
        """ Returns the crystal group from the space group """
        cr_list = [(1, 2), (3, 15), (16, 74), (75, 142), (143, 167), (168, 194), (195, 230)]
        spgr_list = str(self.spgr).split(':')
        spgr_num = int(spgr_list[0])
        for idx, (start, end) in enumerate(cr_list):
            if spgr_num <= end:
                if (end == 167) and len(spgr_list) > 1 and spgr_list[1] == 'R':  # trigonal RHOMB
                    return '5:R'
                return idx + 1
        raise AttributeError('space group should be in range 1-231')

    def _parameters_for_SGLattice(self) -> np.ndarray:
        """ Returns the unique lattice parameters that are needed to receive miller indices """

        parameters_for_hkl = {1: [0, 1, 2, 3, 4, 5],  # return [a, b, c, alpha, beta, gamma] - triclinic
                              2: [0, 1, 2, 4],  # return [a, b, c, beta] - monoclinic
                              3: [0, 1, 2],  # return [a, b, c] - orthorhombic
                              4: [0, 2],  # return [a, c] - tetragonal
                              5: [0, 2],  # return [a, с] - trigonal - HEX
                              '5:R': [0, 3],  # return [a, alpha] - trigonal - RHOMB
                              6: [0, 2],  # return [a, c] - hexagonal
                              7: [0]}  # return [a] - cubic

        return self.lat_par[parameters_for_hkl[self.cr_group]]


class GIWAXS:
    """
        A class to calculate the giwaxs image from the crystal structure and experimental parameters
        ...
        Attributes
        ----------
        crystal : Crystal
            crystal structure representation
        exp : ExpParameters
            experiment parameters representation

        Methods
        -------
        giwaxs_sim(rotate=True, orientation='random'):
            calculates the GIWAXS image
        get_mi(q_max):
            returns allowed miller indices
        """

    def __init__(self, crystal: Crystal, exp_par: ExpParameters):
        self.crystal = crystal
        self.exp = exp_par

        self._mi = self.get_mi(self.exp.q_max)
        self._q_sim = QPos(self.crystal.lat_par, self._mi)
        self.q_3d = self._q_sim.q_3d
        self.q_abs = np.linalg.norm(self.q_3d, axis=-1)

    def giwaxs_sim(self,
                   orientation: Union[str, np.ndarray] = np.array([0., 0., 1.]),  # e.g. None, 'random' or np.array([0., 1., 0.]),
                   return_mi = False
                   ):
        """
        Calculates peak positions and their intensities in the GIWAXS image

        Parameters
        ----------
            orientation : Union[str, np.ndarray], optional
                orientation of the crystal growth (e.g. 'random' or np.array([0., 1., 0.]),
                default = np.array([0., 0., 1.])
                if None - no orientation (powder diffraction)
            return_mi : bool, optional
                return miller indices if True, default=False

        Returns
        -------
            q_2d (np.ndarray): peak positions
            int_sum (np.ndarray): peak intensities
        """
        if (self.crystal.atoms is None) or (self.crystal.atoms_positions is None):
            intensity = None
        else:
            intensity = Intensity(self.crystal.atoms,
                                  self.crystal.atoms_positions,
                                  self.q_3d,
                                  self._mi,
                                  self.exp.wavelength,
                                  self.exp.ai,
                                  self.exp.database
                                  ).get_intensities()

        if orientation is None:
            if not return_mi:
                q_1d, int_sum, mi_sum = self.giwaxs_1d(self.q_abs, intensity, None)
                return q_1d, int_sum
            else:
                q_1d, int_sum, mi_sum = self.giwaxs_1d(self.q_abs, intensity, self._mi)
                return q_1d, int_sum, mi_sum
        else:
            q_3d_rot = self._q_sim.rotate_vect(orientation)
            if not return_mi:
                q_2d, int_sum, mi_sum = self.giwaxs_2d(q_3d_rot, intensity, None)
                return q_2d, int_sum
            else:
                q_2d, int_sum, mi_sum = self.giwaxs_2d(q_3d_rot, intensity, self._mi)
                return q_2d, int_sum, mi_sum


    @staticmethod
    def giwaxs_1d(q_1d, intensity, mi):
        x = q_1d[np.newaxis]
        X = np.repeat(x, len(q_1d), axis=0)
        Y = X.T

        closest = np.isclose(X, Y, rtol=1e-04, atol=1e-04).astype(np.int32)
        closest = np.triu(closest)

        indices_sum = np.zeros(len(q_1d), dtype=np.int32)
        all_indices = np.arange(len(q_1d))

        for a, b in zip(*np.where(closest == 1)):
            if a not in all_indices:
                continue  # means that we matched this peak with another one
            indices_sum[b] = np.where(all_indices == a)[0][0]
            if a != b:
                # remove b element from all_indices
                index = np.where(all_indices == b)[0][0]
                all_indices = np.delete(all_indices, index)
        q_1d_fin = q_1d[all_indices]
        if intensity is None:
            int_fin = np.ones_like(q_1d_fin)
        else:
            int_fin = np.bincount(indices_sum, weights=intensity)
            int_fin = int_fin / q_1d_fin**2

        if mi is not None:
            mi_new = []
            for idx, idx_sum in enumerate(indices_sum):
                if len(mi_new) > idx_sum:
                    mi_new[int(idx_sum)] = np.concatenate((mi_new[int(idx_sum)], mi[idx][np.newaxis]), axis=0)
                    continue
                mi_new.append(mi[idx][np.newaxis])
            return q_1d_fin, int_fin, mi_new

        return q_1d_fin, int_fin, mi

    @staticmethod
    def giwaxs_2d(q_3d, intensity, mi):
        """ Convert 3d pattern to 2d space with summation the intensities """
        # q_3d.shape - (points_num, 3)
        q_xy = np.sqrt(q_3d[..., 0] ** 2 + q_3d[..., 1] ** 2) # shape (points_num,)
        q_z = q_3d[..., 2]
        q_2d = np.stack((q_xy, q_z)) # shape (2, points_num)

        x = q_2d[np.newaxis] # (1, 2, points_num)
        X = np.repeat(x, x.shape[2], axis=0) # (points_num, 2, points_num)
        # Y = np.repeat(x.transpose((2, 1, 0)), x.shape[2], axis=2) # (points_num, 2, points_num)
        Y = X.transpose((2, 1, 0))

        # find close points, that could be matched together
        closest = np.isclose(X, Y, rtol=1e-04, atol=1e-04).sum(axis=1)
        closest[closest < 2] = 0
        closest = np.triu(closest)

        indices_sum = np.zeros(len(closest), dtype=np.int32)
        all_indices = np.arange(len(closest))

        for a, b in zip(*np.where(closest == 2)):
            if a not in all_indices:
                continue  # means that we matched this peak with another one
            indices_sum[b] = np.where(all_indices == a)[0][0]
            if a != b:
                # remove b element from all_indices
                index = np.where(all_indices == b)[0][0]
                all_indices = np.delete(all_indices, index)

        q_2d_fin = q_2d[..., all_indices]  # shape (2, points_num)
        if intensity is None:
            int_fin = np.ones_like(q_2d_fin[0])
        else:
            int_fin = np.bincount(indices_sum.astype(np.int32), weights=intensity)

        # take only positive part for z-direction
        idxs_pos = np.where(q_2d_fin[1] >= 0)[0]
        q_2d_pos = q_2d_fin[:, idxs_pos]
        int_pos = int_fin[idxs_pos]

        if mi is not None:
            mi_new = []
            for idx, idx_sum in enumerate(indices_sum):
                if len(mi_new) > idx_sum:
                    mi_new[int(idx_sum)] = np.concatenate((mi_new[int(idx_sum)], mi[idx][np.newaxis]), axis=0)
                    continue
                mi_new.append(mi[idx][np.newaxis])
            mi_pos = [mi_new[i] for i in idxs_pos]
            return q_2d_pos, int_pos, mi_pos

        return q_2d_pos, int_pos, mi

        # # take only positive part for z-direction
        # idxs = np.where(q_2d_fin[1] >= 0)
        # q_2d_pos = q_2d_fin[:, idxs]
        # int_pos = int_fin[idxs]
        # mi_pos = [mi_new[i] for i in idxs[0]]
        #
        # return q_2d_pos, int_pos, mi_pos

    def get_mi(self, q_max):
        """ Returns allowed miller indices"""
        mi = SGLattice(self.crystal.spgr, *self.crystal.param_SGL).get_allowed_hkl(qmax=q_max)
        return np.array(list(mi), dtype=np.float32)


class GIWAXSFromCif:
    """
    A class to calculate the GIWAXS image from the CIF
    ...
    Attributes
    ----------
    crystal : Crystal
        crystal structure representation
    exp : ExpParameters
        experiment parameters representation
    giwaxs : GIWAXS
    """

    def __init__(self, path, exp_par: ExpParameters):
        el = CIFFile(path)
        name = el.default_dataset
        if name is None:
            name = list(el.data.keys())[0]

        lat_par = (np.append(el.data[name].lattice_const, el.data[name].lattice_angles)).astype(np.float32)
        if hasattr(el.data[name], 'sgrp'):
            spgr = el.data[name].sgrp # str type
        else:
            spgr = '1'

        atoms, atoms_positions = zip(*[(atom[0].basename, atom[1]) for atom in el.SGLattice().base()])
        atoms = np.array(atoms)
        atoms_positions = np.array(atoms_positions, dtype=np.float32)

        self.crystal = Crystal(spgr, lat_par, atoms, atoms_positions)
        self.exp = exp_par
        self.giwaxs = GIWAXS(self.crystal, self.exp)
