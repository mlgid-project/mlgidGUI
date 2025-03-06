from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, Qt


class Label(QLabel):
    def __init__(self, text: str, parent=None, font_size: float = 8., bold: bool = False):
        super().__init__(text, parent=parent)
        font = self.font()
        font.setPointSizeF(font_size)
        font.setBold(bold)
        self.setFont(font)


class LabelButton(Label):
    sigClicked = pyqtSignal()
    sigPressed = pyqtSignal()

    def mousePressEvent(self, ev) -> None:
        self.sigPressed.emit()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        self.sigClicked.emit()
        ev.accept()

    def enterEvent(self, a0) -> None:
        self.setStyleSheet('QLabel {color : #D3D3D3}')
        super().enterEvent(a0)

    def leaveEvent(self, a0) -> None:
        self.setStyleSheet('QLabel {color : #808080}')
        super().leaveEvent(a0)


class HyperlinkLabel(LabelButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet('QLabel {color : #71bbd4}')
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def enterEvent(self, a0) -> None:
        font = self.font()
        font.setUnderline(True)
        self.setFont(font)
        QLabel.enterEvent(self, a0)

    def leaveEvent(self, a0) -> None:
        font = self.font()
        font.setUnderline(False)
        self.setFont(font)
        QLabel.leaveEvent(self, a0)
