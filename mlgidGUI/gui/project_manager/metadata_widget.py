from collections import OrderedDict

from PyQt5.QtWidgets import (QWidget,
                             QGridLayout, QTreeView)
from PyQt5.QtCore import Qt, QObject, QItemSelection
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel

from mlgidGUI.app.rois import Roi
from mlgidGUI.gui.roi_widgets import AbstractRoiHolder, AbstractRoiWidget



class MetaWidget(AbstractRoiHolder, QWidget):
    def __init__(self, parent=None):
        pass