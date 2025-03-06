from enum import Enum, auto

from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5.QtCore import pyqtSignal

from ..app.update import CheckVersionMessage
from .basic_widgets import PopUpWrapper, TextNotification, Label, HyperlinkLabel


class NotificationTypes(Enum):
    checking_version = auto()
    updating_program = auto()
    check_result = auto()


def get_check_result_notification(res: CheckVersionMessage) -> TextNotification or None:
    if res.value == CheckVersionMessage.new_version_available.value:
        try:
            version = f'{res.version} '
        except AttributeError:
            return
            # version = ''
        return TextNotification('New version available',
                                f'The new version {version}will be installed')
    if res.value == CheckVersionMessage.latest_version_installed.value:
        return TextNotification('Latest version installed',
                                'You use the latest version of the program!')
    if res.value == CheckVersionMessage.error.value or res.value == CheckVersionMessage.no_internet.value:
        return TextNotification('Check version failed',
                                'Please, check your internet connection')
    if res.value == CheckVersionMessage.failed_updating.value:
        return TextNotification('Install failed', 'Please, check your internet connection or update manually')


class CheckingVersion(TextNotification):
    def __init__(self, parent=None):
        super().__init__('Check version', 'The program is looking for updates', spinner=True, parent=parent)


class UpdatingProgram(TextNotification):
    def __init__(self, version: str, parent=None):
        super().__init__('Install', f'Updating the program to version {version}', spinner=True, parent=parent)


class UpdateFailed(TextNotification):
    def __init__(self, parent=None):
        super().__init__('Install failed', 'Please, check your internet connection or use\n'
                                           'pip install --upgrade mlgidGUI',
                         spinner=False, parent=parent)


class AskRestart(QWidget):
    sigRestart = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setSpacing(5)
        title = Label('The program was updated', self, 8.5, True)
        title_font = title.font()
        title_font.setPointSizeF(8.5)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(Label('Please, restart the program to launch the new version.', self, 8.5), 1, 0, 1, 2)
        restart_label = HyperlinkLabel('Restart', self, 8.5)
        layout.addWidget(restart_label, 2, 0)
        restart_label.sigClicked.connect(self.sigRestart)
