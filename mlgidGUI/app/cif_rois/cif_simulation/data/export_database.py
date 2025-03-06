import numpy as np
import xrayutilities as xu
from datetime import datetime
from .atoms import atoms_list

def calculateFF(en=18000,  # energy in kEv
                q_range=np.arange(0, 10, 0.001)
                ):
    """
    Calculate and save form-factors for each possible atom in atom_list
    """
    # export ff for all possible atoms
    # ff_dict = dict() #
    result_shape = (len(atoms_list), len(q_range))
    full_ff_matrix = np.empty(result_shape, dtype='complex128')

    for idx, atom in enumerate(atoms_list):
        ff = xu.materials.atom.Atom(atom, 1).f(q_range, en=en)  # form factor
        # ff_dict[atom] = ff
        full_ff_matrix[idx] = ff

    return full_ff_matrix, np.array(atoms_list),

if __name__ == "__main__":
    start = datetime.now()
    ff_dict = calculateFF(en=18000,  # energy in kEv
                          q_range=np.arange(0, 10, 0.001))
    print(datetime.now() - start)
