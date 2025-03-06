import numpy as np
from typing import Union
from .possible_directions import get_unique_directions


class QPos:
    """
    A class to calculate the q positions in 3d reciprocal space
    ...
    Attributes
    ----------
    lat_par : np.ndarray
        lattice parameters
    mi : np.ndarray
        miller indices
    rec : np.ndarray
        reciprocal vectors
    q_3d : np.ndarray
        peak positions in 3d reciprocal space

    Methods
    -------
    calculate_q3d():
        calculates the q vectors in 3d reciprocal space
    rotate_vect(orientation, baz):
        rotate crystal
    """

    def __init__(self,
                 lat_par: np.ndarray,  # np.array([a,b,c,α,β,γ], dtype=np.float32)
                 mi: np.ndarray  # 3d vectors - (E, 3) - [h, k, l]
                 ):
        self.lat_par = lat_par
        self.mi = mi
        self.rec, self.q_3d = self.calculate_q3d()

    def calculate_q3d(self):
        """ Calculate q_3d vectors from lattice parameters and miller indices"""
        try:
            a1, a2, a3 = self._lattice_vectors_from_parameters()
        except TypeError:
            return None

        rec = self._calc_reciprocal_vectors(a1, a2, a3)
        q_vectors = self.mi @ rec
        return rec, q_vectors

    def rotate_vect(self,
                    orientation: Union[None, str, np.ndarray],  # e.g. 'random' or np.array([0., 1., 0.])
                    baz: np.ndarray = np.array([0., 0., 1.], dtype=np.float32),  # basZ
                    dtype=np.float32) -> np.ndarray:
        """
        Rotate crystal

        Parameters
        ----------
            orientation : Union[str, np.ndarray]
                orientation of the crystal growth
            baz : torch.Tensor, optional
                basis vector for the default orientation, default=np.array([0., 0., 1.], dtype=np.float32)

        Returns
        -------
            q_3d (np.ndarray): peak positions in 3d reciprocal space
        """
        if orientation is None:
            return self.q_3d
        elif isinstance(orientation, str):
            directions = get_unique_directions(0, 5)
            orientation = directions[np.random.randint(len(directions))]  # choose one of the possible orientations
        elif isinstance(orientation, np.ndarray) and orientation.shape == (3,):
            orientation = orientation / np.linalg.norm(orientation, axis=0)
        else:
            raise TypeError('orientation is not correct - use np.array with size (3,) or "random"')

        if np.all(baz == orientation):
            return self.q_3d

        baz = baz  #@ self.rec
        orient = orientation @ self.rec

        v1 = orient / np.linalg.norm(orient, axis=0)
        v2 = baz / np.linalg.norm(baz, axis=0)
        _n = np.cross(v1, v2)
        n = _n / np.linalg.norm(_n, axis=0)

        cos_phi = v1 @ v2
        sin_phi = np.sqrt(1 - cos_phi ** 2)

        a_1 = np.stack((n[..., 0] ** 2 * (1 - cos_phi) + cos_phi,
                        n[..., 0] * n[..., 1] * (1 - cos_phi) + n[..., 2] * sin_phi,
                        n[..., 0] * n[..., 2] * (1 - cos_phi) - n[..., 1] * sin_phi))

        a_2 = np.stack((n[..., 0] * n[..., 1] * (1 - cos_phi) - n[..., 2] * sin_phi,
                        n[..., 1] ** 2 * (1 - cos_phi) + cos_phi,
                        n[..., 1] * n[..., 2] * (1 - cos_phi) + n[..., 0] * sin_phi))

        a_3 = np.stack((n[..., 0] * n[..., 2] * (1 - cos_phi) + n[..., 1] * sin_phi,
                        n[..., 1] * n[..., 2] * (1 - cos_phi) - n[..., 0] * sin_phi,
                        n[..., 2] ** 2 * (1 - cos_phi) + cos_phi))

        R = np.stack((a_1, a_2, a_3), dtype=dtype)
        q_rot = self.q_3d @ R

        return q_rot

    def _lattice_vectors_from_parameters(self, vol_min=11, dtype=np.float32):
        """ Returns lattice vectors corresponding to lattice parameters"""
        pi = np.pi
        a, b, c = self.lat_par[:3]
        alpha, beta, gamma = self.lat_par[3:] * pi / 180

        unit_volume = np.sqrt(
            1.0
            + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
        )

        if np.isnan(unit_volume) or (a * b * c * unit_volume < vol_min):
            raise ValueError('Too small unit_volume')

        # reciprocal lattice
        # a_recip = np.sin(alpha) / (a * unit_volume)
        # cos_gamma_recip = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)
        #                    ) / (np.sin(alpha) * np.sin(beta))
        # sin_gamma_recip = np.sqrt(1 - cos_gamma_recip ** 2)

        # a1 = np.array(
        #     [1 / a_recip, -cos_gamma_recip / sin_gamma_recip / a_recip, np.cos(beta) * a], dtype=dtype
        # )
        # a2 = np.array([0, b * np.sin(alpha), b * np.cos(alpha)], dtype=dtype)
        # a3 = np.array([0, 0, c], dtype=dtype)

        # reciprocal lattice
        a1 = np.array([a, 0, 0], dtype=dtype)
        a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0], dtype=dtype)
        a31 = c * np.cos(beta)
        a32 = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        a33 = (c ** 2 - a31 ** 2 - a32 ** 2) ** 0.5
        a3 = np.array([a31, a32, a33], dtype=dtype)

        return a1, a2, a3

    @staticmethod
    def _calc_reciprocal_vectors(a1: np.ndarray,
                                 a2: np.ndarray,
                                 a3: np.ndarray,
                                 dtype=np.float32
                                 ) -> np.ndarray:
        """ Returns vectors in reciprocal space corresponding to lattice vectors """
        pi = np.pi

        unit_volume = np.dot(a1, np.cross(a2, a3, axis=0))
        b1 = np.cross(a2, a3, axis=0)
        b2 = np.cross(a3, a1, axis=0)
        b3 = np.cross(a1, a2, axis=0)
        return np.stack([b1, b2, b3], dtype=dtype) * 2 * pi / unit_volume
