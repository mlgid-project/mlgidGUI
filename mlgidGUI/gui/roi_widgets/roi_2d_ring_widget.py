import numpy as np

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QColor

from pyqtgraph import ROI

from ...app import Roi, App
from .abstract_roi_widget import AbstractRoiWidget


class BasicRoiRing(ROI):
    def __init__(self,
                 radius: float,
                 radius_width: float,
                 angle: float,
                 angle_width: float,
                 center: tuple = (0, 0),
                 movable: bool = False,
                 rotatable: bool = False,
                 resizable: bool = False,
                 parent=None,
                 **kwargs):

        super().__init__(center,
                         (radius, radius),
                         movable=movable,
                         rotatable=rotatable,
                         resizable=resizable,
                         parent=parent,
                         **kwargs)

        self._center = center
        self._radius = radius
        self._radius_width = radius_width
        self._angle = angle
        self._angle_width = angle_width

        self.aspectLocked = True
        self.set_radius(self._radius)

    def set_center(self, center: tuple):
        self._center = center
        d = self._radius + self._radius_width / 2
        pos = (center[1] - d, center[0] - d)
        self.setPos(pos)

    def set_radius(self, radius):
        self._radius = radius if radius > 0 else 0
        s = 2 * radius + self._radius_width
        self.setSize((s, s))
        self.set_center(self._center)

    def set_radius_width(self, radius_width):
        self._radius_width = radius_width
        self.set_radius(self._radius)

    def set_angle(self, angle):
        self._angle = angle
        self.set_center(self._center)

    def set_angle_width(self, angle):
        self._angle_width = angle
        self.set_center(self._center)

    def set_params(self, *,
                   radius: float = None, radius_width: float = None, angle: float = None,
                   angle_width: float = None, center: tuple = None):

        should_set_radius: bool = False

        if radius is not None and radius != self._radius:
            self._radius = radius
            should_set_radius = True
        if radius_width is not None and radius_width != self._radius_width:
            self._radius_width = radius_width
            should_set_radius = True
        if angle is not None and angle != self._angle:
            self._angle = angle
        if angle_width is not None and angle_width != self._angle_width:
            self._angle_width = angle_width
        if center is not None and self._center != center:
            self._center = center
        if should_set_radius:
            self.set_radius(radius)
        else:
            self.set_center(self._center)

    def paint(self, p, opt, widget):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self.currentPen)

        x1, y1 = 0, 0
        x2, y2 = x1 + self._radius_width, y1 + self._radius_width
        x3, y3 = x1 + self._radius_width / 2, y1 + self._radius_width / 2
        d1, d2, d3 = (2 * self._radius + self._radius_width,
                      2 * self._radius - self._radius_width,
                      2 * self._radius)

        # p.scale(self._radius, self._radius)
        r1 = QRectF(x1, y1, d1, d1)
        r2 = QRectF(x2, y2, d2, d2)
        r3 = QRectF(x3, y3, d3, d3)
        angle = - self._angle or 0
        angle_width = self._angle_width or 360
        a1, a2 = int((angle - angle_width / 2) * 16), int(angle_width * 16)
        p.drawArc(r1, a1, a2)
        p.drawArc(r2, a1, a2)
        dash_pen = QPen(self.currentPen)
        dash_pen.setStyle(Qt.DashLine)
        p.setPen(dash_pen)
        p.drawArc(r3, a1, a2)

    def getArrayRegion(self, arr, img=None, axes=(0, 1), **kwds):
        """
        Return the result of ROI.getArrayRegion()
        masked by the elliptical shape
        of the ROI. Regions outside the ellipse are set to 0.
        """
        # Note: we could use the same method as used by PolyLineROI, but this
        # implementation produces a nicer mask.
        arr = ROI.getArrayRegion(self, arr, img, axes, **kwds)
        if arr is None or arr.shape[axes[0]] == 0 or arr.shape[axes[1]] == 0:
            return arr
        w = arr.shape[axes[0]]
        h = arr.shape[axes[1]]
        ## generate an ellipsoidal mask
        mask = np.fromfunction(
            lambda x, y: (((x + 0.5) / (w / 2.) - 1) ** 2 + ((y + 0.5) / (h / 2.) - 1) ** 2) ** 0.5 < 1, (w, h))

        # reshape to match array axes
        if axes[0] > axes[1]:
            mask = mask.T
        shape = [(n if i in axes else 1) for i, n in enumerate(arr.shape)]
        mask = mask.reshape(shape)

        return arr * mask

    def shape(self):
        self.path = QPainterPath()
        self.path.addEllipse(self.boundingRect())
        return self.path


class Roi2DRing(AbstractRoiWidget, BasicRoiRing):
    sigRoiMoved = pyqtSignal(int)
    sigSelected = pyqtSignal(int)
    sigShiftSelected = pyqtSignal(int)

    def __init__(self, roi: Roi, parent=None):
        AbstractRoiWidget.__init__(self, roi)
        BasicRoiRing.__init__(self,
                              radius=roi.radius,
                              radius_width=roi.radius_width,
                              angle=roi.angle,
                              angle_width=roi.angle_width,
                              parent=parent)

        self.update_roi()

        if App().debug_tracker:
            App().debug_tracker.add_object(self, roi.name)

    @pyqtSlot(name='move_roi')
    def move_roi(self):
        self.set_radius(self.roi.radius)
        self.set_radius_width(self.roi.radius_width)
        if self.roi.angle is not None:
            self.set_angle(self.roi.angle)
        if self.roi.angle_width is not None:
            self.set_angle_width(self.roi.angle_width)

    def send_move(self):
        pass

    def set_color(self, color):
        self.setPen(color=color, width=4)

COLOR_DICT = {
    0: QColor(255, 179, 186),
    1: QColor(224, 243, 176),
    2: QColor(191, 213, 232),
    3: QColor(253, 222, 238),
}

class CIF2DRing(Roi2DRing):

    def __init__(self, roi: Roi, file_counter):
        super().__init__(roi)
        roi.roi_widget = self
        self.color = COLOR_DICT[roi.cif_file_nr%4]
        self.color.setAlpha(20)
        self.set_color(self.color)

    def set_selected(self):
        self.setPen(color=QColor(125, 125, 125), width=4)

    def set_unselected(self):
        self.set_color(self.color)

    def reset_color(self):
        self.set_color(COLOR_DICT[self.roi.cif_file_nr % 4])

