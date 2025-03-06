from collections import OrderedDict

from PyQt5.QtWidgets import (QWidget,
                             QGridLayout, QTreeView, QPushButton)
from PyQt5.QtCore import Qt, QObject, QItemSelection
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel

from mlgidGUI.app.cif_rois.cif_peaks_calculation import cif_results
from mlgidGUI.app.rois import Roi
from mlgidGUI.app.rois.roi import CIFPeakRoi
from mlgidGUI.gui.roi_widgets import AbstractRoiHolder, AbstractRoiWidget


class StandardItem(QStandardItem):
    def __init__(self, text: str, key: int, is_numeric: bool = True):
        super().__init__(text)
        self.key = key
        self.is_numeric = is_numeric

    def __lt__(self, other):
        if self.is_numeric:
            try:
                return float(self.text()) < float(other.text())
            except ValueError:
                pass
        return super().__lt__(other)


class CIFRingWidget(AbstractRoiWidget, QObject):
    COLOR_DICT = dict(
        default=QColor(0, 0, 0, 0),
        active=QColor(0, 128, 255, 100),
        fixed=QColor(0, 255, 0, 100),
        fixed_active=QColor(255, 0, 255, 100)
    )

    PARAM_DICT = OrderedDict([('cif_file', 'CIF File'),
                            ('q_z', 'Qz'),
                            ('q_xy', 'Qxy'),
                            ('miller_id', 'Miller indices'),
                            ('intensity', 'Intensity'),
                            ])

    def __init__(self, roi: Roi, parent):
        AbstractRoiWidget.__init__(self, roi, enable_context=True)
        QObject.__init__(self, parent)

        self.__items = self._init_items()
        self.update_color()
        #self.move_roi()

    @property
    def row(self):
        return self.__items['name'].row()

    def _init_items(self) -> dict:
        roi_key = self.roi.key
        items = dict(name = StandardItem(self.roi.name, roi_key),
                     key = StandardItem(str(self.roi.key), roi_key),
                     type = StandardItem(str(self.roi.type.name), roi_key, False),
                     q_z = StandardItem(f'{self.roi.q_z:.2f}', roi_key, False),
                     q_xy=StandardItem(f'{self.roi.q_xy:.2f}', roi_key, False),
                     confidence_level_name = StandardItem(str(self.roi.confidence_level_name), roi_key),
                     cif_file = StandardItem(str(self.roi.cif_file), roi_key),
                     intensity = StandardItem(f'{self.roi.intensity_raw:.0f}', roi_key),
                     miller_id=StandardItem(str(self.roi.miller_id), roi_key),
                     )
        return items

    def items(self):
        return [self.__items[key] for key in self.PARAM_DICT.keys()]


class CIFRingsWidget(AbstractRoiHolder, QWidget):
    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'RoiMetaWidget')
        QWidget.__init__(self, parent)

        self._model = QStandardItemModel(0, len(CIFRingWidget.PARAM_DICT), self)
        self._model.setHorizontalHeaderLabels(list(CIFRingWidget.PARAM_DICT.values()))
        # self._model.setRowCount(0)

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self._model)

        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._custom_menu)
        self.tree_view.selectionModel().selectionChanged.connect(self._selection_changed)
        self.tree_view.header().setSectionsClickable(True)
        self.tree_view.header().setSortIndicatorShown(True)
        self.tree_view.header().sortIndicatorChanged.connect(self._model.sort)

        self._init_ui()

    def _init_connect(self):
        pass

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.addWidget(self.tree_view)
        layout.addWidget(QPushButton('Add ROI by holding `Ctrl + Alt` and click and hold on the image'))


    def delete_all(self) -> None:
        self._model.removeRows(0, self._model.rowCount())

    def empty_simulation(self) -> None:
        roi_widget = CIFRingWidget(CIFPeakRoi(
            radius = 0,
            width = 0,
            cif_file='Please add a CIF-file to start the simulation.',
        ), self.tree_view)
        self._model.appendRow(roi_widget.items())

    def resizeColumns(self):
        for col in range(self._model.columnCount()):
            self.tree_view.resizeColumnToContents(col)

    def _delete_roi_widget(self, roi_widget) -> None:
        self.tree_view.selectionModel().blockSignals(True)
        self._model.removeRow(roi_widget.row)
        self.tree_view.selectionModel().blockSignals(False)

    def _make_roi_widget(self, roi: Roi) -> AbstractRoiWidget:
        try:
            roi_widget = CIFRingWidget(roi, self.tree_view)
            self._model.appendRow(roi_widget.items())
            return roi_widget
        except:
            return None

    def _custom_menu(self, pos):
        pass

    def _selection_changed(self, item: QItemSelection):
        try:
            cif_results[self.selected_key].roi_widget.set_unselected()
        except:
            pass
        try:
            keys = set(self._model.itemFromIndex(index).key for index in item.indexes())
            if len(keys) == 1:
                cif_results[next(iter(keys))].roi_widget.set_selected()
                self.selected_key = next(iter(keys))
        except:
            pass