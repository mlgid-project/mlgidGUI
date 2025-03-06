from PyQt5.QtGui import (QPainter, QPixmap, QPen, QBrush,
                         QConicalGradient, QColor)
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import (Qt, QPropertyAnimation, pyqtProperty,
                          pyqtSlot, QRect)


class Spinner(QWidget):
    def __init__(self, parent=None, size: int = 25, padding: int = 2, start: bool = True):
        super().__init__(parent)
        self._size = size
        self._padding = padding

        self.pixmap = QPixmap('wheel.png').scaled(size, size)

        self.setFixedSize(size + padding * 2, size + padding * 2)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self._angle = 0

        self.animation = QPropertyAnimation(self, b"angle", self)
        self.animation.setStartValue(0)
        self.animation.setEndValue(360)
        self.animation.setLoopCount(-1)
        self.animation.setDuration(2000)
        if start:
            self.animation.start()

    @pyqtSlot(name='pauseSpinning')
    def pause(self):
        self.animation.stop()

    @pyqtSlot(name='resumeSpinning')
    def resume(self):
        self.animation.start()

    @pyqtProperty(int)
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self.update()

    def paintEvent(self, ev=None):
        drawing_rect = QRect()
        drawing_rect.setX(self._padding)
        drawing_rect.setY(self._padding)
        drawing_rect.setWidth(self._size)
        drawing_rect.setHeight(self._size)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        gradient = QConicalGradient()
        gradient.setCenter(drawing_rect.center())
        gradient.setAngle(- self._angle - self._size / 10)
        gradient.setColorAt(0, QColor(178, 255, 246))
        gradient.setColorAt(1, QColor(5, 44, 50))

        pen = QPen(QBrush(gradient), self._size // 10)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(drawing_rect, -self._angle * 16, 300 * 16)
