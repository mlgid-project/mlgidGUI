import math
import numpy as np
from .data.export_database import calculateFF


class Database:
    """
    A class to represent the form-factors matrix for all possible atoms
    ...
    Attributes
    ----------
    full_atom_list : list
        list of all possible atoms - len = 98
    full_ff_matrix : np.ndarray
        form-factors (ff) matrix for en=18000eV in q_range = (0, 10, 0.001)
        shape = (98, 10000), 98 - number of elements, 10000 - calculated ff in q_range
    """

    def __init__(self, en):
        self.full_ff_matrix, self.full_atom_list = calculateFF(en)


class ExpParameters:
    """
    A class to represent the experiment parameters
    ...
    Attributes
    ----------
    q_xy_max : float
        maximum value for the q in xy direction, Å^{-1}
    q_z_max : float
        maximum value for the q in z direction, Å^{-1}
    ai : float
        incidence angle, deg
    wavelength : float
        beam wavelength, Å
    database: Database
        database with form-factors
    """

    def __init__(self,
                 q_xy_max=2.7,  # Å^{-1}
                 q_z_max=2.7,  # Å^{-1}
                 ai=0.3,  # Incidence angle, deg
                 en=18000,  # Energy, eV
                 ):
        self.q_xy_max = q_xy_max
        self.q_z_max = q_z_max
        self.q_max = math.sqrt(q_xy_max ** 2 + q_z_max ** 2)
        self.ai = ai
        self.wavelength = 12398 / en

        self.database = Database(en=int(en))


if __name__ == "__main__":
    db = ExpParameters().database
