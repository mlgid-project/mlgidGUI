from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QFrame, QMessageBox
from PyQt5.QtCore import Qt, QObject, QPoint, pyqtSignal, pyqtSlot
from .spinner import Spinner
from .labels import Label, LabelButton

NOTIFICATION_BACKGROUND_COLOR = 'rgb(80, 80, 80)'


class PopUpWrapper(QObject):
    WIDTH = 400
    HEIGHT = 90
    PADDING = 10

    def __init__(self, widget: QWidget):
        super().__init__(widget)
        self.parent = widget
        self.__popups = []
        self.__popups_dict = {}
        self.__monkeypatch_parent_methods(widget)

    @pyqtSlot(name='movePopup')
    def _move_popups(self):
        win_width, win_height = self.parent.size().width(), self.parent.size().height()
        win_point = self.parent.mapToGlobal(QPoint(win_width, win_height))
        for i, popup in enumerate(self.__popups):
            popup_w, popup_h = self.WIDTH + self.PADDING, (self.HEIGHT + self.PADDING) * (i + 1)
            p = win_point - QPoint(popup_w, popup_h)
            popup.move(p)
            if win_width - popup_w < 0 or win_height - popup_h < 0:
                popup.hide()
            else:
                popup.show()

    def add_notification(self, context: str or QWidget, name=None):
        if not context:
            return
        notification = Notification(self.parent, context)
        notification.sigCloseClicked.connect(self.remove_notification)

        if name is not None:
            self.__popups_dict[name] = notification

        self.__popups.append(notification)
        self._move_popups()

    @pyqtSlot(QWidget, name='removeWidget')
    def remove_notification(self, widget: QWidget):
        try:
            self.__popups.remove(widget)
            widget.deleteLater()
            self._move_popups()
        except ValueError:
            pass
        self.__popups_dict = {k: v for k, v in self.__popups_dict.items() if v is not widget}

    def remove_by_name(self, name):
        if name in self.__popups_dict:
            self.remove_notification(self.__popups_dict.pop(name))

    def clear(self):
        for widget in self.__popups:
            try:
                widget.deleteLater()
            except RuntimeError:
                pass
        self.__popups = []
        self.__popups_dict = {}

    def __monkeypatch_parent_methods(self, parent: QWidget):
        old_resize = parent.resizeEvent
        old_move = parent.moveEvent

        def new_resize(event):
            self._move_popups()
            old_resize(event)

        def new_move(event):
            self._move_popups()
            old_move(event)

        parent.resizeEvent = new_resize
        parent.moveEvent = new_move


class CloseButton(LabelButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setBold(True)
        font.setPointSize(10)
        self.setFont(font)

    def set_visible(self, enable: bool = True):
        if enable:
            self.setStyleSheet('QLabel {color: #808080}')
        else:
            self.setStyleSheet('QLabel {color: %s}' % NOTIFICATION_BACKGROUND_COLOR)


class Notification(QFrame):
    sigCloseClicked = pyqtSignal(QWidget)

    def __init__(self, parent, context: str or QWidget):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self._init_ui(context)
        self.show()

    def enterEvent(self, a0) -> None:
        self.button.set_visible(True)
        super().enterEvent(a0)

    def leaveEvent(self, a0) -> None:
        self.button.set_visible(False)
        super().leaveEvent(a0)

    def _init_ui(self, context: str or QWidget):
        if isinstance(context, str):
            context = QLabel(context, self)
        else:
            context.setParent(self)

        self.setFixedSize(PopUpWrapper.WIDTH, PopUpWrapper.HEIGHT)
        self.setStyleSheet(f"background-color: {NOTIFICATION_BACKGROUND_COLOR}")
        self.setContentsMargins(5, 5, 5, 0)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._get_close_button(), 0, 1, alignment=Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(context, 0, 0, 3, 1)

    def _get_close_button(self):
        close_button = CloseButton('x', self)
        close_button.sigClicked.connect(lambda: self.sigCloseClicked.emit(self))
        close_button.set_visible(False)
        self.button = close_button
        return close_button

def confirmation_box(window_title: str = '', text: str = '', informative_text: str = ','):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(window_title)
    msg_box.setText(text)
    msg_box.setInformativeText(informative_text)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    msg_box.setDefaultButton(QMessageBox.Yes)
    ret = msg_box.exec()
    if ret == QMessageBox.Yes:
        return True
    else:
        return False

class TextNotification(QWidget):
    def __init__(self, title: str, text: str, spinner: bool = False, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setSpacing(5)

        if spinner:
            layout.addWidget(Spinner(self, 20), 0, 0)
            layout.addWidget(Label(title, self, 8.5, True), 0, 1)
            layout.addWidget(Label(text, self, 8.5), 1, 0, 1, 2)
        else:
            layout.addWidget(Label(title, self, 8.5, True), 0, 0)
            layout.addWidget(Label(text, self, 8.5), 1, 0)
