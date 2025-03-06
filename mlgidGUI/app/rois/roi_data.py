from typing import List, Tuple, Dict

import numpy as np

from .roi import Roi, RoiTypes


class RoiData(dict):
    def __init__(self, rois: List[Roi] = None):
        super().__init__()
        self._selected_keys = set()
        if rois:
            self.add_rois(rois)

    @property
    def selected_keys(self):
        yield from self._selected_keys

    @property
    def selected_num(self):
        return len(self._selected_keys)

    def select(self, key: int) -> Tuple:
        if len(self._selected_keys) == 1 and key in self._selected_keys:
            return ()
        try:
            for k in self._selected_keys:
                self[k].active = False
            self[key].active = True
            self._selected_keys.add(key)
            change_select = tuple(self._selected_keys)
            self._selected_keys = {key}
            return change_select

        except KeyError:
            raise ValueError(f'The selected roi {key} is missing.')

    def shift_select(self, key: int):
        try:
            self[key].active = not self[key].active
            if self[key].active:
                self._selected_keys.add(key)
            else:
                try:
                    self._selected_keys.remove(key)
                except KeyError:
                    pass
        except KeyError:
            raise ValueError(f'The selected roi {key} is missing.')

    @property
    def selected_rois(self) -> List[Roi]:
        return [self[i] for i in self._selected_keys]

    def select_all(self) -> List[int]:
        to_select = list()
        for roi in self.values():
            if not roi.active:
                roi.active = True
                to_select.append(roi.key)
        self._selected_keys = set(self.keys())
        return to_select

    def unselect_all(self) -> List[int]:
        to_unselect = list()
        for roi in self.values():
            if roi.active:
                roi.active = False
                to_unselect.append(roi.key)
        self._selected_keys = set()
        return to_unselect

    def add_roi(self, roi: Roi) -> None:
        self[roi.key] = roi

        if roi.active:
            self._selected_keys.add(roi.key)

    def add_rois(self, roi_list: List[Roi]) -> List[int]:
        keys = list()
        to_select = list()
        for roi in roi_list:
            self[roi.key] = roi
            if roi.active:
                self._selected_keys.add(roi.key)
                to_select.append(roi.key)
            keys.append(roi.key)
        return to_select

    def delete_roi(self, k: int):
        del self[k]

        if k in self._selected_keys:
            self._selected_keys.remove(k)

    def clear(self):
        super().clear()
        self._selected_keys = set()

    def change_ring_bounds(self, bounds: Tuple[float, float]) -> List[int]:
        angle, angle_std = bounds
        keys = list()
        for k, roi in self.items():
            if roi.should_adjust_angles(angle, angle_std):
                keys.append(k)
                roi.angle, roi.angle_std = bounds
        return keys

    def on_scale_changed(self, scale_change: float):
        for roi in self.values():
            roi.radius *= scale_change
            roi.width *= scale_change

    def apply_fit(self, rois: List[Roi]):
        keys_to_move = []
        rois_to_create = []

        for roi in rois:
            roi.active = True
            roi.movable = False

            if roi.key in self.keys():
                keys_to_move.append(roi.key)
                self[roi.key].update(roi)
                self.select(roi.key)
            else:
                rois_to_create.append(roi)

        self.add_rois(rois_to_create)
        keys_to_create = [roi.key for roi in rois_to_create]

        return tuple(keys_to_create), tuple(keys_to_move)

    def to_array(self) -> np.ndarray:
        return np.array([roi.to_array() for roi in self.values()])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls([Roi.from_array(a) for a in arr])

    def to_dict(self) -> Dict[str, np.ndarray]:
        roi_dicts = [roi.to_dict() for roi in self.values()]
        keys = set().union(*[d.keys() for d in roi_dicts])
        # if not all peaks are fitted, save without fitting parameters
        #if not np.all(np.array(['fitted_params' in d.keys() for d in roi_dicts])):
        keys -= { 'background', 'background (level)', 'background (slope)', 'fit_errors', 'fitted_params', 'fitting_function', 'init_params', 'lower_bounds', 'peak height', 'r_range', 'upper_bounds', 'x_range'}
        return_dict = {}
        for k in keys:
            if type(roi_dicts[0].get(k, np.nan)).__name__ == 'str':
                return_dict[k] = np.array([d.get(k, np.nan) for d in roi_dicts],dtype='S30')
            else:
                return_dict[k] = np.array([d.get(k, np.nan) for d in roi_dicts])
        return return_dict

    @classmethod
    def from_dict(cls, arr_dict: Dict[str, np.ndarray]):
        keys = list(arr_dict.keys())
        if not keys:
            return cls()
        num_rois = len(arr_dict[keys[0]])
        rois = [
            Roi.from_dict({k: arr_dict[k][i] for k in arr_dict.keys()}, name = i)
            for i in range(num_rois)
        ]
        return cls(rois)

    @property
    def intensities(self):
        return np.array([roi.intensity for roi in self.values()], dtype=float)

    @property
    def confidence_levels(self):
        return np.array([roi.confidence_level for roi in self.values()], dtype=float)

    @property
    def cif_files(self):
        return np.array([roi.cif_file for roi in self.values()], dtype=str)
