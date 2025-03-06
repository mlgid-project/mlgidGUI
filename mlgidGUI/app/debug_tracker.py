from datetime import datetime as dt
from enum import Enum
import sys
from typing import Dict
import weakref
import gc
import logging

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .utils import SingletonMeta

logger = logging.getLogger(__name__)


class ObjectStatus(Enum):
    EXISTS = 0
    SAFELY_DELETED = 1
    DELETED_BY_PYTHON = 2
    DELETED_BY_C = 3


class TrackSignals(QObject):
    sigObjectAdded = pyqtSignal(str)
    sigStatusUpdated = pyqtSignal(list)


class ObjectTracker(QObject):
    def __init__(self, obj: QObject, name: str):
        super().__init__()
        self.name: str = f'{obj.__class__.__name__}: {name}'
        self.id: str = f'{name}: {id(obj)}, {dt.now().timestamp()}'
        self.obj_ref = weakref.ref(obj)
        self.status: ObjectStatus = ObjectStatus.EXISTS
        self._deleted_by_c: bool = False
        self._deleted_by_python: bool = False
        obj.destroyed.connect(self._on_destroyed)
        self.update_status()

    def update_status(self) -> ObjectStatus:
        try:
            obj = self.obj_ref()
            if obj is None:
                self._deleted_by_python = True
        except LookupError:
            logger.debug('LookupError in tracker')
            self._deleted_by_python = True
        except RuntimeError:
            logger.debug('RuntimeError in tracker')
            self._deleted_by_c = True

        if self._deleted_by_c and self._deleted_by_python:
            self.status = ObjectStatus.SAFELY_DELETED
        elif self._deleted_by_c and not self._deleted_by_python:
            self.status = ObjectStatus.DELETED_BY_C
        elif not self._deleted_by_c and self._deleted_by_python:
            self.status = ObjectStatus.DELETED_BY_PYTHON
        else:
            self.status = ObjectStatus.EXISTS

        return self.status

    def get_referrers(self):
        obj = self.get_obj()
        if obj:
            logger.debug(f'Number of refs = {sys.getrefcount(obj)}')
            return gc.get_referrers([obj])
        else:
            logger.debug('Object is None!')

    def get_obj(self):
        try:
            return self.obj_ref()
        except Exception as err:
            logger.exception(err)

    def is_tracked(self) -> bool:
        obj = self.get_obj()
        if obj:
            return gc.is_tracked(obj)
        else:
            return False

    @pyqtSlot(name='onDestroyed')
    def _on_destroyed(self, *args):
        # logger.debug('destroyed signal in tracker')
        self._deleted_by_c = True


class TrackQObjects(metaclass=SingletonMeta):
    ObjectStatus: ObjectStatus = ObjectStatus

    def __init__(self):
        self.signals: TrackSignals = TrackSignals()
        self.objects: Dict[str, ObjectTracker] = {}

    def add_object(self, obj: QObject, name: str = ''):
        obj_tracker = ObjectTracker(obj, name)
        self.objects[obj_tracker.id] = obj_tracker
        self.signals.sigObjectAdded.emit(obj_tracker.id)

    def update(self, key: str = None, *, emit: bool = True):
        if key:
            self._update(key)
        else:
            updated_keys = []
            for key, obj in self.objects.items():
                if obj.status != obj.update_status():
                    updated_keys.append(key)
            if emit and updated_keys:
                self.signals.sigStatusUpdated.emit(updated_keys)

    def _update(self, key: str):
        try:
            if self.objects[key].status != self.objects[key].update_status():
                self.signals.sigStatusUpdated.emit([key])
        except KeyError:
            pass

    def clear_deleted(self, status: ObjectStatus = ObjectStatus.SAFELY_DELETED):
        to_delete = []
        for key, obj in self.objects.items():
            if obj.status == status:
                to_delete.append(key)

        for key in to_delete:
            obj = self.objects.pop(key)
            obj.deleteLater()
        return to_delete
