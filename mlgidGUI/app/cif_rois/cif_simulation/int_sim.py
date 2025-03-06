import numpy as np


class Intensity:
    """
    A class to calculate the GIWAXS intensities
    ...
    Attributes
    ----------
    atoms : np.ndarray
        list of elements in the structure
    atoms_positions: np.ndarray
        relative atom coordinates (related with the atoms attribute)
    q_3d : np.ndarray
        peak vectors in 3d reciprocal space
    q_1d : np.ndarray
        absolute peak positions in reciprocal space
    mi : np.ndarray
        miller indices
    wavelength : float
        beam wavelength, Å
    ai : float
        incidence angle, deg
    database: Database
        database with form-factors

    Methods
    -------
    get_intensities():
        returns intensities for each peak position
    """

    def __init__(self,
                 atoms: np.ndarray,  # list of elements in the structure,
                 atoms_positions: np.ndarray,  # relative atom coordinates
                 q_3d: np.ndarray,  # 3d vectors - (E, 3) - [[qx, qy, qz]]
                 mi: np.ndarray,  # 3d vectors - (E, 3) - [[h, k, l]]
                 wavelength: float,  # angstrom
                 ai: float,  # Incidence angle, deg.
                 database
                 ):
        self.atoms = atoms
        self.atoms_positions = atoms_positions
        self.q_3d = q_3d
        self.q_1d = np.linalg.norm(q_3d, axis=-1) # 1d vectors - len(q), shape = N, N-number of peaks in 1D
        self.mi = mi

        self.wavelength = wavelength
        self.ai = ai
        self.database = database

    def get_intensities(self):
        """ Returns intensities for each peak position """
        intensity = self._get_intensities_from_mi()
        #intensity_corrected = intensity * self._lorentz_correction() * self._polarization_correction()
        return intensity

    def _get_intensities_from_mi(self) -> np.ndarray:
        """ Returns intensities for each peak position """

        #full_ff_matrix, atom_list = ff_extract()  # atom_list: list of all possible elements: len = 98
        # full_ff_matrix: shape = (98, 10000)
        # 98 - number of elements, 10000 - calculated ff in range(0,10,0.001)
        sum_fs = self._get_sf(self.database.full_atom_list, self.database.full_ff_matrix)  # structure factor

        intensity = np.abs(sum_fs) ** 2
        return intensity
        # return _normalize_vec(intensity)

    def _get_sf(self,
                atom_list: np.ndarray,  # list of all possible elements: len = 98
                full_ff_matrix: np.ndarray,
                # shape = (98, 10000), 98 - number of elements, 10000 - calculated ff in range(0,10,0.001)
                ) -> np.ndarray:
        """ Returns structure factor as a sum of scattering amplitudes for all atoms """

        ffs = self._get_ff(atom_list, full_ff_matrix)  # form factors: (number of elements, len(mi))

        sf = np.multiply(ffs, np.exp(2 * np.pi * 1j
                                     * np.matmul(self.atoms_positions,
                                                 self.mi.T)))  # scattering factors: (number of elements, len(mi))

        sum_sf = np.sum(sf, axis=0)  # structure factor: (len(mi)) /////// len(mi)==len(q_vectors)

        return sum_sf

    def _get_ff(self,
                atom_list: np.ndarray,  # list of all possible elements: len = 98
                full_ff_matrix: np.ndarray,
                # shape = (98, 10000), 98 - number of elements, 10000 - calculated ff in range(0,10,0.001))
                ) -> np.ndarray:
        """ Return X-ray form factors at specified q """

        # smbl is resolved here
        smbl_num = []
        for smbl in self.atoms:
            if smbl not in atom_list:
                raise KeyError('element {} does not exist'.format(smbl))
            smbl_num.extend(np.where(smbl == atom_list)[0])

        # !!!! WORKS ONLY FOR q_list_default = np.arange(0,10, 0.001)!!!!!
        q_list_num = (self.q_1d * 1000).astype(np.int32)
        ff_matrix = full_ff_matrix[smbl_num][:, q_list_num]

        return ff_matrix  # shape = (len(symbol_list), len(q_vectors))
