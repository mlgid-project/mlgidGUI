from copy import deepcopy
from typing import List, Tuple, Iterable, Dict, Any, Deque
import logging
from collections import deque

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from .roi import Roi, RoiTypes
from .roi_data import RoiData
from .roi_meta_data import RoiMetaData
from .roi_colors import RoiColors, RoiColorsDict
from ..file_manager import FileManager, ImageKey, FolderKey
from ..geometry_holder import GeometryHolder


def _check_non_empty(func):
    def wrapper(self, *args, **kwargs):
        if not self._current_key:
            return
        else:
            return func(self, *args, **kwargs)

    return wrapper


class _RoiDictModel(object):
    log = logging.getLogger(__name__)

    def __init__(self, file_manager: FileManager, geometry_holder: GeometryHolder, history_size: int = 30):
        self.geometry_holder: GeometryHolder = geometry_holder
        self.fm: FileManager = file_manager
        self.roi_data: RoiData = RoiData()
        self.current_key: ImageKey or None = None
        self.meta_data: RoiMetaData or None = None
        self.actions_history: Deque = deque(maxlen=history_size)

    def __len__(self):
        return len(self.roi_data)

    def __repr__(self):
        return f'<_RoiDictModel({self.roi_data})>'

    @property
    def ring_bounds(self) -> Tuple[float, float]:
        return self.geometry_holder.geometry.ring_bounds

    @_check_non_empty
    def save_state(self):
        if len(self.roi_data):
            self.fm.rois_data[self.current_key] = self.roi_data
            self.log.debug(f'Roi data {self.current_key} saved.')
        else:
            del self.fm.rois_data[self.current_key]
            self.log.debug(f'Empty roi data {self.current_key} deleted.')

    @_check_non_empty
    def save_folder(self):
        if self.meta_data and len(self.meta_data):
            self.fm.rois_meta_data[self.meta_data.folder_key] = self.meta_data
            self.log.debug(f'Roi meta data for {self._meta_data.folder_key} saved.')

    def clear(self):
        self.roi_data.clear()
        self.log.debug(f'Roi dict cleared.')

    def update(self, key: ImageKey):
        self.current_key = key
        self.roi_data = self.fm.rois_data[self.current_key] or RoiData()
        self.roi_data.change_ring_bounds(self.geometry_holder.geometry.ring_bounds)
        self.meta_data.update_metadata(self.roi_data, self.current_key)

    def default_params(self) -> dict:
        r0, r1 = self.geometry_holder.geometry.r_range
        r0 *= self.geometry_holder.geometry.scale
        r1 *= self.geometry_holder.geometry.scale
        radius = (r0 + r1) / 2
        width = (r1 - r0) / 10
        angle, angle_std = self.ring_bounds
        return dict(radius=radius, width=width, angle=angle, angle_std=angle_std)

    @property
    def selected_rois(self) -> List[Roi]:
        return self.roi_data.selected_rois

    def keys(self):
        yield from self.roi_data.keys()

    def values(self):
        yield from self.roi_data.values()

    def items(self):
        yield from self.roi_data.items()

    def __getitem__(self, item: int) -> Roi:
        return self.roi_data[item]


class _RoiDictSignals(QObject):
    sigRoiCreated = pyqtSignal(tuple)
    sigRoiDeleted = pyqtSignal(tuple)
    sigRoiMoved = pyqtSignal(tuple, str)
    sigDeletedRoisUpdated = pyqtSignal(tuple)

    sigSelected = pyqtSignal(tuple)
    sigOneSelected = pyqtSignal(int)
    sigRoiRenamed = pyqtSignal(int)
    sigFixed = pyqtSignal(tuple)
    sigUnfixed = pyqtSignal(tuple)

    sigTypeChanged = pyqtSignal(int)
    sigFitRoisOpen = pyqtSignal(list)


class BasicRoiDictAction(object):
    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def undo(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class RoiDictMultiAction(BasicRoiDictAction):
    __slots__ = ('_actions', )

    def __init__(self):
        self._actions = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals,
                *actions: Tuple[BasicRoiDictAction, List[Any]]) -> Any:

        self._actions = [action for action, _ in actions]
        for action, args in actions:
            action.execute(roi_model, roi_signals, *args)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        for action in self._actions:
            action.undo(roi_model, roi_signals)

    def __repr__(self):
        actions_str = ', '.join(map(repr, self._actions))
        return f'{self.__class__.__name__}({actions_str})'


class RoiDict(object):
    def __init__(self, file_manager: FileManager, geometry_holder: GeometryHolder, history_size: int = 30,
                 parent=None):
        self._signals = _RoiDictSignals(parent)
        self._model = _RoiDictModel(file_manager, geometry_holder, history_size)
        self._history = self._model.actions_history

    @property
    def signals(self) -> _RoiDictSignals:
        return self._signals

    @property
    def model(self) -> _RoiDictModel:
        return self._model

    def execute(self, action: BasicRoiDictAction, *args, **kwargs) -> Any:
        res = action.execute(self._model, self._signals, *args, **kwargs)
        self._history.append(action)
        return res

    def undo(self):
        if self._history:
            self._history.pop().undo(self._model, self._signals)


class SelectAction(BasicRoiDictAction):
    __slots__ = ('_selected_previously', '_changed_keys')

    def __init__(self):
        self._selected_previously = None
        self._changed_keys = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, key: int):
        self._selected_previously = [r.key for r in roi_model.selected_rois]
        change_select = self._changed_keys = roi_model.roi_data.select(key)

        if change_select:
            roi_signals.sigSelected.emit(change_select)
            roi_model.log.info(f'Select status changed: {change_select}')
            if roi_model.roi_data.selected_num == 1:
                roi_signals.sigOneSelected.emit(next(iter(roi_model.roi_data.selected_keys)))

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        pass

    def __repr__(self):
        pass


class ShiftSelectAction(BasicRoiDictAction):
    __slots__ = ('_selected_previously', '_changed_keys')

    def __init__(self):
        self._selected_previously = None
        self._changed_keys = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, key: int):
        roi_model.roi_data.shift_select(key)
        roi_signals.sigSelected.emit((key,))
        roi_model.log.info(f'Select status changed: {key}')
        if roi_model.roi_data.selected_num == 1:
            roi_signals.sigOneSelected.emit(next(iter(roi_model.roi_data.selected_keys)))

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        pass

    def __repr__(self):
        pass


class SelectAllAction(BasicRoiDictAction):
    __slots__ = ('_selected_previously', )

    def __init__(self):
        self._selected_previously = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        self._selected_previously = roi_model.roi_data.selected_keys
        to_select = roi_model.roi_data.select_all()
        _emit_select(roi_model, roi_signals, to_select)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        pass

    def __repr__(self):
        pass


class UnselectAllAction(BasicRoiDictAction):
    __slots__ = ('_selected_previously', )

    def __init__(self):
        self._selected_previously = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        self._selected_previously = roi_model.roi_data.selected_keys
        to_unselect = roi_model.roi_data.unselect_all()
        _emit_select(roi_model, roi_signals, to_unselect)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        pass

    def __repr__(self):
        pass


class CreateRoiAction(BasicRoiDictAction):
    __slots__ = ('_add_roi_action', )

    def __init__(self):
        self._add_roi_action = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals,
                radius: float, width: float, params: dict) -> Roi:
        d = roi_model.default_params()
        d.update(params)
        if radius and width:
            d.update(radius=radius, width=width)
        roi = Roi(**d)
        self._add_roi_action = AddRoiAction()
        self._add_roi_action.execute(roi_model, roi_signals, roi)
        return roi

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        self._add_roi_action.undo(roi_model, roi_signals)

    def __repr__(self):
        pass


class AddRoiAction(BasicRoiDictAction):
    __slots__ = ('_added_roi_key', )

    def __init__(self):
        self._added_roi_key = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, roi: Roi):
        roi_model.meta_data.add_roi(roi, roi_model.current_key)
        roi_model.roi_data.add_roi(roi)
        if not roi.has_fixed_angles():
            roi.angle, roi.angle_std = roi_model.ring_bounds

        self._added_roi_key = key = roi.key
        roi_signals.sigRoiCreated.emit((key,))
        if roi.active:
            _emit_select(roi_model, roi_signals, (key,))

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        DeleteRoiAction().execute(roi_model, roi_signals, self._added_roi_key)
        self._added_roi_key = None

    def __repr__(self):
        pass


class AddRoisAction(BasicRoiDictAction):
    __slots__ = ('_added_roi_keys',)

    def __init__(self):
        self._added_roi_keys = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, roi_list: List[Roi]):
        roi_model.meta_data.add_rois(roi_list, roi_model.current_key)
        to_select = roi_model.roi_data.add_rois(roi_list)
        roi_signals.sigRoiCreated.emit(tuple(roi.key for roi in roi_list))
        _emit_select(roi_model, roi_signals, to_select)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        DeleteManyAction().execute(roi_model, roi_signals, self._added_roi_keys)
        self._added_roi_keys = None

    def __repr__(self):
        pass


class DeleteRoiAction(BasicRoiDictAction):
    __slots__ = ('_deleted_roi', )

    def __init__(self):
        self._deleted_roi = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, key: int):
        self._deleted_roi = roi_model.roi_data[key]

        roi_model.roi_data.delete_roi(key)
        roi_model.meta_data.delete_roi(key, roi_model.current_key)
        roi_signals.sigRoiDeleted.emit((key,))

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        AddRoiAction().execute(roi_model, roi_signals, self._deleted_roi)

    def __repr__(self):
        pass


class DeleteManyAction(RoiDictMultiAction):
    __slots__ = ('_deleted_rois', )

    def __init__(self):
        super().__init__()
        self._deleted_rois = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, keys: List[int]):
        super().execute(roi_model, roi_signals, *[(DeleteRoiAction(), [key]) for key in keys])


class DeleteSelectedAction(DeleteManyAction):
    __slots__ = ('_deleted_rois',)

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        super().execute(roi_model, roi_signals, roi_model.roi_data.selected_keys)


class ChangeNameAction(BasicRoiDictAction):
    __slots__ = ('_previous_name', '_key')

    def __init__(self):
        self._previous_name = None
        self._key = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, key: int, name: str):
        roi = roi_model.roi_data[key]
        self._key = key
        self._previous_name = roi.name
        roi_model.meta_data.rename(roi, name)
        roi_signals.sigRoiRenamed.emit(key)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        self.execute(roi_model, roi_signals, self._key, self._previous_name)
        self._previous_name = None
        self._key = None

    def __repr__(self):
        pass


class MoveRoiAction(BasicRoiDictAction):
    __slots__ = ('_previous_position', '_key')

    def __init__(self):
        self._select_action = None
        self._previous_position = None
        self._key = None

    def execute(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, key: int, name: str):
        self._select_action = SelectAction()
        self._select_action.execute(roi_model, roi_signals, key)

        roi_signals.sigRoiMoved.emit((key,), name)
        roi = roi_model.roi_data[key]

        if roi.should_adjust_angles(*roi_model.ring_bounds):
            roi.type = RoiTypes.segment
            roi_signals.sigTypeChanged.emit(key)

    def undo(self, roi_model: _RoiDictModel, roi_signals: _RoiDictSignals):
        pass
        # undo not supported yet.

        # self.execute(roi_model, roi_signals, self._key, self._previous_name)
        # self._previous_name = None
        # self._key = None

    def __repr__(self):
        pass


def _emit_select(roi_model: _RoiDictModel, roi_signals: _RoiDictSignals, keys: Iterable[int]):
    if keys:
        roi_model.log.debug(f'Emit select {keys}')
        roi_signals.sigSelected.emit(tuple(keys))
    if roi_model.roi_data.selected_num == 1:
        roi_model.log.debug(f'Emit select one {keys}')
        roi_signals.sigOneSelected.emit(next(iter(roi_model.roi_data.selected_keys)))


# QObject used to enable pyqtSlot interface
class RoiDictApi(QObject):
    EMIT_NAME = 'RoiDict'

    log = logging.getLogger(__name__)

    def __init__(self, file_manager: FileManager, geometry_holder: GeometryHolder, history_size: int = 30,
                 parent=None):
        super().__init__(parent)
        self._roi_dict = RoiDict(file_manager, geometry_holder, history_size, parent)

    def __len__(self):
        return len(self._roi_dict.model)

    def __repr__(self):
        return f'<RoiDict({self._roi_dict.model.roi_data})>'

    @property
    def signals(self):
        return self._roi_dict.signals

    # save_state does not need to be undone (can be re-saved after reversing all real operations)
    def save_state(self):
        self._roi_dict.model.save_state()

    # save_folder does not need to be undone (can be re-saved after reversing all real operations)
    def save_folder(self):
        self._roi_dict.model.save_folder()

    @pyqtSlot(int, name='select')
    def select(self, key: int):
        self._roi_dict.execute(SelectAction(), key)

    @pyqtSlot(int, name='shiftSelect')
    def shift_select(self, key: int):
        self._roi_dict.execute(ShiftSelectAction(), key)

    @pyqtSlot(name='selectAll')
    def select_all(self) -> None:
        self._roi_dict.execute(SelectAllAction())

    @pyqtSlot(name='unselectAll')
    def unselect_all(self):
        self._roi_dict.execute(UnselectAllAction())

    @pyqtSlot(name='deleteSelected')
    def delete_selected_roi(self):
        self._roi_dict.execute(DeleteSelectedAction())

    @pyqtSlot(int, str, name='changeName')
    def change_name(self, key: int, name: str):
        self._roi_dict.execute(ChangeNameAction(), key, name)
        self.log.info(f'Roi {key} renamed to {name}')

    @pyqtSlot(object, name='addRoi')
    def add_roi(self, roi: Roi) -> None:
        self._roi_dict.execute(AddRoiAction(), roi)

    def add_rois(self, roi_list: List[Roi]):
        self._roi_dict.execute(AddRoisAction(), roi_list)

    @_check_non_empty
    def delete_roi(self, key: int):
        self._roi_dict.execute(DeleteRoiAction(), key)

    @_check_non_empty
    def create_roi(self, radius: float = None, width: float = None, **params) -> Roi:
        return self._roi_dict.execute(CreateRoiAction(), radius, width, params)

    @pyqtSlot(tuple, name='change_ring_bounds')
    def change_ring_bounds(self, bounds: Tuple[float, float]):
        # so far undo not supported. The action is not isolated - it would require to undo another operation
        # at the same time that caused changing the ring bounds (e.g., changing the beam center)

        model = self._roi_dict.model
        keys = model.roi_data.change_ring_bounds(bounds)
        if keys:
            self.signals.sigRoiMoved.emit(tuple(keys), self.EMIT_NAME)

    @pyqtSlot(name='on_scale_changed')
    def on_scale_changed(self):
        # same problem as with change_ring_bounds

        model = self._roi_dict.model
        model.roi_data.on_scale_changed(model.geometry_holder.geometry.scale_change)
        self.signals.sigRoiMoved.emit(tuple(model.roi_data.keys()), self.EMIT_NAME)

    @pyqtSlot(int, str, name='moveRoi')
    def move_roi(self, key: int, name: str):
        self.select(key)
        self.sig_roi_moved.emit((key,), name)
        roi = self[key]
        if roi.should_adjust_angles(*self.ring_bounds):
            roi.type = RoiTypes.segment
            self.sig_type_changed.emit(key)

    @pyqtSlot(int, name='roiTypeChanged')
    def change_roi_type(self, key: int):
        try:
            roi = self[key]
            if roi.should_adjust_angles(*self.ring_bounds):
                roi.angle, roi.angle_std = self.ring_bounds
                self.sig_roi_moved.emit((key,), self.EMIT_NAME)
        except KeyError:
            return
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
