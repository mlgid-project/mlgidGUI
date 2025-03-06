# -*- coding: utf-8 -*-
from typing import Hashable, Callable

from PyQt5.QtCore import QThreadPool, QObject, pyqtSlot, pyqtSignal

from ..app.utils import Worker, SingletonMeta


class _BackgroundTasksNotifications(QObject):
    sigAddNotification = pyqtSignal(object, object)
    sigRemoveNotification = pyqtSignal(object)
    sigAddProgressBar = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q_thread_pool = QThreadPool(self)

    @pyqtSlot(str, name='removeNotification')
    def remove_notification(self, name: Hashable):
        self.sigRemoveNotification.emit(name)

    @pyqtSlot(object, object, name='addNotification')
    def add_notification(self, notification, name: Hashable = None):
        self.sigAddNotification.emit(notification, name)

    def add_worker(self, worker: Worker, *, auto_delete: bool = True):
        if auto_delete:
            worker.autoDelete()
        self._q_thread_pool.start(worker)

    def add_progress_bar(self, progress_bar_func: Callable):
        self.sigAddProgressBar.emit(progress_bar_func)


class BackgroundTasks(metaclass=SingletonMeta):
    def __init__(self, parent=None):
        self._tasks = _BackgroundTasksNotifications(parent)

    @property
    def tasks(self) -> _BackgroundTasksNotifications:
        return self._tasks
