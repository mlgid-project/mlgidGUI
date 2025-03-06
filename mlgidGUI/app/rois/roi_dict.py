from copy import deepcopy
from typing import List, Tuple, Iterable
import logging

import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from .roi import Roi, RoiTypes
from .roi_data import RoiData
from .roi_meta_data import RoiMetaData
from .roi_colors import RoiColors, RoiColorsDict, ROI_COLOR_KEY
from ..file_manager import FileManager, ImageKey, FolderKey
from ..geometry_holder import GeometryHolder


def _check_non_empty(func):
    def wrapper(self, *args, **kwargs):
        if not self._current_key:
            return
        else:
            return func(self, *args, **kwargs)

    return wrapper


class RoiDict(QObject):
    sig_roi_created = pyqtSignal(tuple)
    sig_roi_deleted = pyqtSignal(tuple)
    sig_roi_moved = pyqtSignal(tuple, str)
    sig_deleted_rois_updated = pyqtSignal(tuple)

    sig_selected = pyqtSignal(tuple)
    sig_one_selected = pyqtSignal(int)
    sig_roi_renamed = pyqtSignal(int)
    sig_cif_renamed = pyqtSignal(int)
    sig_fixed = pyqtSignal(tuple)
    sig_unfixed = pyqtSignal(tuple)

    sig_type_changed = pyqtSignal(int)
    sigFitRoisOpen = pyqtSignal(list)
    sigColorChanged = pyqtSignal(tuple)

    sigConfLevelChanged = pyqtSignal(int)

    EMIT_NAME = 'RoiDict'

    log = logging.getLogger(__name__)

    def __init__(self, file_manager: FileManager, geometry_holder: GeometryHolder):
        super().__init__()
        self._geometry_holder: GeometryHolder = geometry_holder
        self._fm: FileManager = file_manager
        self._roi_data: RoiData = RoiData()
        self._current_key: ImageKey or None = None
        self._meta_data: RoiMetaData or None = None
        self._copied_rois: CopiedRois = CopiedRois()
        self._roi_colors: RoiColors = RoiColors(file_manager, parent=self)
        self._roi_colors.sigColorChanged.connect(self.color_changed)
        self._roi_colors.sigColorDictSet.connect(self.update_colors)

    def __len__(self):
        return len(self._roi_data)

    def __repr__(self):
        return f'<RoiDict({self._roi_data})>'

    @property
    def is_copied(self):
        return len(self._copied_rois) > 0

    @property
    def ring_bounds(self) -> Tuple[float, float]:
        return self._geometry_holder.geometry.ring_bounds

    @_check_non_empty
    def save_state(self):
        if len(self._roi_data):
            self._fm.rois_data[self._current_key] = self._roi_data
            self.log.debug(f'Roi data {self._current_key} saved.')
        else:
            del self._fm.rois_data[self._current_key]
            self.log.debug(f'Empty roi data {self._current_key} deleted.')

    @_check_non_empty
    def save_folder(self):
        if self._meta_data and len(self._meta_data):
            self._fm.rois_meta_data[self._meta_data.folder_key] = self._meta_data
            self.log.debug(f'Roi meta data for {self._meta_data.folder_key} saved.')

    def clear(self):
        if len(self._roi_data):
            self.sig_roi_deleted.emit(tuple(self.keys()))
            self._roi_data.clear()
            self.log.debug(f'Roi dict cleared.')

    def save_and_clear(self):
        self.save_state()
        self.clear()

    @pyqtSlot(object, name='changeFolder')
    def change_folder(self, folder_key: FolderKey):
        self.save_folder()

        if folder_key:
            try:
                self._meta_data = self._fm.rois_meta_data[folder_key]
            except:
                self._meta_data = RoiMetaData(folder_key)
            self.log.debug(f'Loaded roi meta data {folder_key}.')
        else:
            self._meta_data = None
            self.log.debug(f'Empty folder selected')

    def change_image(self, image_key: ImageKey):
        self.clear()
        self._current_key = image_key
        if not self._current_key:
            return
        self._update()
        self.sig_deleted_rois_updated.emit(tuple(self._meta_data.get_deleted_rois(self._current_key)))

    def _update(self):
        self._roi_data = self._fm.rois_data[self._current_key] or RoiData()
        self._roi_data.change_ring_bounds(self._geometry_holder.geometry.ring_bounds)
        self._meta_data = self._fm.rois_meta_data[self._current_key] or RoiMetaData(self._current_key.parent)
        self._meta_data.update_metadata(self._roi_data, self._current_key)
        if len(self._roi_data):
            self.sig_roi_created.emit(tuple(self.keys()))

    @pyqtSlot()
    def select_next(self):
        sorted_keys, current_idx = self._select_by_order()
        if not sorted_keys:
            return
        if current_idx == len(sorted_keys) - 1:
            current_idx = -1

        self.select(sorted_keys[current_idx + 1])

    @pyqtSlot()
    def select_previous(self):
        sorted_keys, current_idx = self._select_by_order()
        if not sorted_keys:
            return
        if current_idx == 0:
            current_idx = len(sorted_keys)
        self.select(sorted_keys[current_idx - 1])

    def _select_by_order(self):
        selected_rois = self.selected_rois
        if len(selected_rois) != 1:
            return None, None
        selected_roi = selected_rois[0]
        sorted_keys = self.get_sorted_keys()
        current_idx = sorted_keys.index(selected_roi.key)
        return sorted_keys, current_idx

    def get_sorted_keys(self):
        keys = list(self.keys())
        indices = np.argsort([self[k].radius for k in keys])
        sorted_keys = [keys[i] for i in indices]
        return sorted_keys

    @property
    def selected_rois(self) -> List[Roi]:
        return self._roi_data.selected_rois

    def keys(self):
        yield from self._roi_data.keys()

    def values(self):
        yield from self._roi_data.values()

    def items(self):
        yield from self._roi_data.items()

    def __getitem__(self, item: int) -> Roi:
        return self._roi_data[item]

    @pyqtSlot(int, name='select')
    def select(self, key: int):
        change_select = self._roi_data.select(key)
        if change_select:
            self.sig_selected.emit(change_select)
            self.log.info(f'Select status changed: {change_select}')
            if self._roi_data.selected_num == 1:
                self.sig_one_selected.emit(next(iter(self._roi_data.selected_keys)))

    @pyqtSlot(int, name='shift_select')
    def shift_select(self, key: int):
        self._roi_data.shift_select(key)
        self.sig_selected.emit((key,))
        self.log.info(f'Select status changed: {key}')
        if self._roi_data.selected_num == 1:
            self.sig_one_selected.emit(next(iter(self._roi_data.selected_keys)))

    @pyqtSlot(name='selectAll')
    @_check_non_empty
    def select_all(self) -> None:
        self._emit_select(self._roi_data.select_all())
        self.log.info(f'Select status changed (select all)')

    @pyqtSlot(name='unselectAll')
    @_check_non_empty
    def unselect_all(self):
        self._emit_select(self._roi_data.unselect_all())

    @pyqtSlot(name='deleteSelected')
    @_check_non_empty
    def delete_selected_roi(self):
        for key in list(self._roi_data.selected_keys):
            self.delete_roi(key)

    @pyqtSlot(int, str, name='changeName')
    def change_name(self, key: int, name: str):
        self._meta_data.rename(self[key], name)
        self.sig_roi_renamed.emit(key)
        self.log.info(f'Roi {key} renamed to {name}')

    @pyqtSlot(int, int, name='changeCIF')
    def change_cif_file(self, key: int, cif_file: str):
        self._roi_data[key].cif_file = cif_file
        self.sig_cif_renamed.emit(key)
        self.log.info(f'Roi {key} cif file changed to {cif_file}')

    @pyqtSlot(int, float, name='changeConfLevel')
    def change_conf_level(self, key: int, level: float):
        self._roi_data[key].confidence_level = level
        self.sigConfLevelChanged.emit(key)
        self.log.info(f'Roi {key} conf level changed to {level}')

    def _emit_select(self, keys: Iterable[int]):
        if keys:
            self.log.debug(f'Emit select {keys}')
            self.sig_selected.emit(tuple(keys))
        if self._roi_data.selected_num == 1:
            self.log.debug(f'Emit select one {keys}')
            self.sig_one_selected.emit(next(iter(self._roi_data.selected_keys)))

    @pyqtSlot(object, name='addRoi')
    @_check_non_empty
    def add_roi(self, roi: Roi) -> None:
        self._meta_data.add_roi(roi, self._current_key)
        self._roi_data.add_roi(roi)
        if not roi.has_fixed_angles():
            roi.angle, roi.angle_std = self.ring_bounds
        self.sig_roi_created.emit((roi.key,))
        if roi.active:
            self._emit_select((roi.key,))

    def add_rois(self, roi_list: List[Roi]):
        self._meta_data.add_rois(roi_list, self._current_key)
        to_select = self._roi_data.add_rois(roi_list)
        self.log.debug(f'Add {len(roi_list)} rois.')
        self.sig_roi_created.emit(tuple(roi.key for roi in roi_list))
        self._emit_select(to_select)

    @_check_non_empty
    def delete_roi(self, key: int):
        self._roi_data.delete_roi(key)
        self._meta_data.delete_roi(key, self._current_key)
        self.sig_roi_deleted.emit((key,))

    @_check_non_empty
    def create_roi(self, radius: float = None, width: float = None, **params) -> Roi:
        d = self._default_params()
        d.update(params)
        if radius and width:
            d.update(radius=radius, width=width)
        roi = Roi(**d)
        self.add_roi(roi)
        return roi

    @pyqtSlot(tuple, name='change_ring_bounds')
    def change_ring_bounds(self, bounds: Tuple[float, float]):
        keys = self._roi_data.change_ring_bounds(bounds)
        if keys:
            self.sig_roi_moved.emit(tuple(keys), self.EMIT_NAME)

    @pyqtSlot(name='on_scale_changed')
    def on_scale_changed(self):
        self._roi_data.on_scale_changed(self._geometry_holder.geometry.scale_change)
        self.sig_roi_moved.emit(tuple(self.keys()), self.EMIT_NAME)

    def _default_params(self) -> dict:
        r0, r1 = self._geometry_holder.geometry.r_range
        r0 *= self._geometry_holder.geometry.scale
        r1 *= self._geometry_holder.geometry.scale
        radius = (r0 + r1) / 2
        width = (r1 - r0) / 10
        angle, angle_std = self.ring_bounds
        return dict(radius=radius, width=width, angle=angle, angle_std=angle_std)

    @pyqtSlot(int, str, name='moveRoi')
    def move_roi(self, key: int, name: str):
        self.select(key)
        self.sig_roi_moved.emit((key,), name)
        roi = self[key]
        if roi.should_adjust_angles(*self.ring_bounds):
            roi.type = RoiTypes.segment
            self.sig_type_changed.emit(key)

    @pyqtSlot(tuple, name='colorChanged')
    def color_changed(self, key: ROI_COLOR_KEY):
        keys = self._get_rois_by_color_key(key)
        if keys:
            self.sigColorChanged.emit(keys)

    @pyqtSlot(name='updateColors')
    def update_colors(self):
        keys = list(self.keys())
        if keys:
            self.sigColorChanged.emit(keys)

    def _get_rois_by_color_key(self, key: ROI_COLOR_KEY) -> List[int]:
        keys = []
        for roi in self.values():
            if (roi.type, roi.active, not roi.movable) == key:
                keys.append(roi.key)
        return keys

    @pyqtSlot(int, name='roiTypeChanged')
    def change_roi_type(self, key: int):
        self.sig_type_changed.emit(key)

    @pyqtSlot(name='fixAll')
    def fix_all(self, only_selected: bool = False):
        keys = []
        for roi in self.values():
            if only_selected and not roi.active:
                continue
            if roi.movable:
                roi.movable = False
                keys.append(roi.key)
        if keys:
            self.sig_fixed.emit(tuple(keys))

    @pyqtSlot(name='fixSelected')
    def fix_selected(self):
        self.fix_all(True)

    @pyqtSlot(name='unfixAll')
    def unfix_all(self, only_selected: bool = False):
        keys = []
        for roi in self.values():
            if only_selected and not roi.active:
                continue
            if not roi.movable:
                roi.movable = True
                keys.append(roi.key)
        if keys:
            self.sig_unfixed.emit(tuple(keys))

    @pyqtSlot(name='unfixSelected')
    def unfix_selected(self):
        self.unfix_all(True)

    @pyqtSlot(int, name='fixRoi')
    def fix_roi(self, key: int):
        try:
            self[key].movable = False
            self.sig_fixed.emit((key,))
        except KeyError:
            return

    @pyqtSlot(int, name='unfixRoi')
    def unfix_roi(self, key: int):
        try:
            self[key].movable = True
            self.sig_unfixed.emit((key,))
        except KeyError:
            return

    def copy_rois(self, key: int or str = 'selected'):
        if key == 'selected':
            rois = self.selected_rois
        elif key == 'all':
            rois = list(self.values())
        else:
            try:
                rois = [self[key]]
            except KeyError:
                return
        if rois:
            self._copied_rois.copy_rois(rois, self._geometry_holder.geometry)

    def paste_rois(self):
        if not self._current_key or not len(self._copied_rois):
            return
        self.add_rois(self._copied_rois.paste(self._geometry_holder.geometry))

    @pyqtSlot(bool, name='openFitRois')
    def open_fit_rois(self, only_selected: bool):
        if only_selected:
            rois = deepcopy(self.selected_rois)
        else:
            rois = deepcopy(list(self.values()))
        if rois:
            self.sigFitRoisOpen.emit(rois)

    def apply_fit(self, rois: List[Roi], image_key: ImageKey):

        if self._current_key == image_key:
            keys_to_create, keys_to_move = self._roi_data.apply_fit(rois)
            self.sig_roi_created.emit(keys_to_create)
            self.sig_roi_moved.emit(keys_to_move, self.EMIT_NAME)
            self.sig_fixed.emit(keys_to_move)
            self.sig_selected.emit(keys_to_move)

        else:
            roi_data = self._fm.rois_data[image_key] or RoiData()
            roi_data.apply_fit(rois)
            self._fm.rois_data[image_key] = roi_data


class CopiedRois(object):
    def __init__(self, rois: List[Roi] = None, geometry=None):
        self._rois = None
        self._scale = None
        self._bounds = None

        if rois and geometry:
            self.copy_rois(rois, geometry)

    def __len__(self):
        return len(self._rois) if self._rois else 0

    def copy_rois(self, rois: List[Roi], geometry):
        self._rois = deepcopy(rois)

        for roi in self._rois:
            roi.active = True

        self._scale = geometry.scale
        self._bounds = geometry.ring_bounds

    def paste(self, geometry, *, clear_keys: bool = True,
              clear: bool = False):
        if not self._rois:
            return

        if clear:
            rois = self._rois
            self.clear()
        else:
            rois = deepcopy(self._rois)

        if self._scale != geometry.scale:
            s = geometry.scale / self._scale
            for roi in rois:
                roi.radius *= s
                roi.width *= s
        if self._bounds != geometry.ring_bounds:
            for roi in rois:
                if not roi.has_fixed_angles():
                    roi.angle, roi.angle_std = geometry.ring_bounds
        if clear_keys:
            for roi in rois:
                roi.key = None

        return rois

    def clear(self):
        self._rois = None
        self._scale = None
        self._bounds = None