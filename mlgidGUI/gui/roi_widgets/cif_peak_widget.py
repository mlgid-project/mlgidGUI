from pyqtgraph import ROI
from PyQt5.QtGui import QPen, QColor

_ROI_SIZE = 5

COLOR_DICT = {
    0: QColor(255, 179, 186),
    1: QColor(224, 243, 176),
    2: QColor(191, 213, 232),
    3: QColor(253, 222, 238),
}

class CIFPoint(ROI):

    def __init__(self, image_item, image_viewer, point: tuple, file_counter):
        self.color_pen = QPen(COLOR_DICT[file_counter%4])
        ROI.__init__(self, (point), 3*point[2], movable=False, parent=image_item, pen=self.color_pen)
        self._center = image_viewer.app.geometry.beam_center
        self._scale = 1
        self.image_viewer = image_viewer
        self.set_pos((point[0],point[1]))

    def set_selected(self):
        self.setPen(color=QColor(125, 125, 125), width=4)

    def set_unselected(self):
        self.setPen(self.color_pen)

    def set_pos(self, q_position):
        y_max = self.image_viewer.app.geometry.shape[0]
        x_max = self.image_viewer.app.geometry.shape[1]
        y_min = self._center[0]
        x_min = self._center[1]
        y = q_position[0] * (y_max - y_min)
        x = q_position[1] * (x_max - x_min)
        self.setPos((x, y))

    def set_scale(self, scale: float):
        self._scale = scale
        self.set_size()

    def zValue(self):
        return 10