import numpy as np

from .cif_simulation.giwaxs_sim import GIWAXSFromCif
from .cif_simulation.experiment import ExpParameters
from ..rois.roi import CIFPeakRoi

cif_results = {}

def calculate_cif_peaks(project_root_key, image_key):
    cif_results.clear()
    if image_key is not None:
        for file_nr, cif_file_key in enumerate(project_root_key._cif_files):
            if cif_file_key.is_visible():
                try:
                    params = ExpParameters(image_key.get_q_range()[0], image_key.get_q_range()[1], en=cif_file_key.en)
                    el = GIWAXSFromCif(cif_file_key.path, params)
                    orientation = np.array(cif_file_key.orientation)

                    if cif_file_key.powder_diffraction:
                        orientation = None
                    q_2d, intensities, mi_indices = el.giwaxs.giwaxs_sim(orientation, return_mi = True)
                    #normalizes to 1 - 3
                    intensities_perc = np.argsort(np.argsort(intensities))/len(intensities)
                    intensities_raw = intensities
                    intensities = np.clip(np.ceil(((intensities - intensities.min()) / (intensities.max() - intensities.min()))*3), a_min = 1, a_max=3)
                except Exception as e:
                    return cif_file_key

                if cif_file_key.powder_diffraction:
                    q_z = q_2d.squeeze().tolist()
                    q_xy = [0]*len(q_z)
                else:
                    q_z = q_2d.squeeze()[0].tolist()
                    q_xy = q_2d.squeeze()[1].tolist()

                i = 0
                is_powder = [cif_file_key.powder_diffraction]*len(q_z)
                for cif_peak in zip(is_powder, q_xy, q_z, intensities, intensities_raw, intensities_perc, mi_indices):
                    cif_results[cif_file_key.name + 'p' + str(i)] = CIFPeakRoi(
                        radius = 0,
                        width = 0,
                        angle = 0,
                        angle_std = 0,
                        key = cif_file_key.name + 'p' + str(i),
                        name = cif_file_key.name + str(i),
                        movable = False,
                        cif_file_nr = file_nr,
                        cif_file=cif_file_key.name,
                        is_powder = cif_file_key.powder_diffraction,
                        q_xy = cif_peak[1],
                        q_z = cif_peak[2],
                        intensity = cif_peak[3],
                        intensity_raw = cif_peak[4],
                        intensity_perc = cif_peak[5],
                        miller_id = cif_peak[6],
                    )
                    i += 1