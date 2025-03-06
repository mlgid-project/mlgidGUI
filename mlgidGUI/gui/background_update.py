# -*- coding: utf-8 -*-

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..__version import __version__
from ..app.utils import Worker

from ..app import App
from ..app.update import (
    CheckVersionMessage,
    check_outdated,
    update_package
)

from .notifications import (
    NotificationTypes,
    CheckingVersion,
    AskRestart,
    UpdatingProgram,
    UpdateFailed,
    get_check_result_notification
)

from .background_tasks import BackgroundTasks


class BackgroundUpdate(QObject):
    sigRestartAfterUpdate = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks = BackgroundTasks().tasks
        self.app = parent.app
        self._is_updated: bool or None = None

    @property
    def is_updated(self) -> bool:
        if self._is_updated is None:
            self._is_updated = self._check_is_updated()
        return self._is_updated

    def run(self):
        if not self.is_updated:
            self.check_update()
        else:
            self.tasks.add_notification(get_check_result_notification(CheckVersionMessage.latest_version_installed))

    def _check_is_updated(self) -> bool:
        if self.app.fm.config['is_updated']:
            del self.app.fm.config['is_updated']
            return True
        else:
            return False

    def check_update(self):
        self.tasks.add_notification(CheckingVersion(), NotificationTypes.checking_version)
        worker = Worker(check_outdated, version=__version__)
        worker.autoDelete()
        worker.signals.result.connect(self._version_checked)
        self.tasks.add_worker(worker)

    def update_package(self, target_version: str = ''):
        worker = Worker(update_package, version=target_version)
        worker.signals.result.connect(self._package_updated)
        self.tasks.add_notification(UpdatingProgram(target_version), NotificationTypes.updating_program)
        self.tasks.add_worker(worker)

    @pyqtSlot(object, name='versionChecked')
    def _version_checked(self, res: CheckVersionMessage):
        self.tasks.remove_notification(NotificationTypes.checking_version)
        self.tasks.add_notification(get_check_result_notification(res), NotificationTypes.check_result)

        if res.value == CheckVersionMessage.new_version_available.value:
            try:
                self.update_package(res.version)
            except AttributeError:
                return

    @pyqtSlot(object, name='packageUpdated')
    def _package_updated(self, is_updated: bool):
        BackgroundTasks().tasks.remove_notification(NotificationTypes.updating_program)
        if is_updated:
            ask_restart = AskRestart()
            ask_restart.sigRestart.connect(self.sigRestartAfterUpdate)
            self.tasks.add_notification(ask_restart)
        else:
            self.tasks.add_notification(UpdateFailed())
