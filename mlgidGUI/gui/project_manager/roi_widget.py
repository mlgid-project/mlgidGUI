from collections import OrderedDict

from PyQt5.QtWidgets import (QWidget,
                             QGridLayout, QTreeView, QPushButton)
from PyQt5.QtCore import Qt, QObject, QItemSelection, QItemSelectionModel
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel

from mlgidGUI.app.rois import Roi
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


class RoiWidgetItem(AbstractRoiWidget, QObject):
    COLOR_DICT = dict(
        default=QColor(0, 0, 0, 0),
        active=QColor(0, 128, 255, 100),
        fixed=QColor(0, 255, 0, 100),
        fixed_active=QColor(255, 0, 255, 100)
    )

    PARAM_DICT = OrderedDict([('radius', 'Radius'),
                              ('radius_width', 'Width'),
                              ('angle', 'Angle'),
                              ('angle_width', 'Angle Width'),
                              ('confidence_level_name', 'Confidence level'),
                              ('score', 'Score'),
                              ('key', 'Key'),
                              ('cif_file', 'CIF File'),
                              ('type', 'ROI type')
                              ])

    def __init__(self, roi: Roi, parent):
        AbstractRoiWidget.__init__(self, roi, enable_context=True)
        QObject.__init__(self, parent)

        self.__items = self._init_items()
        self.update_color()
        self.move_roi()

    @property
    def row(self):
        return self.__items['name'].row()

    def _init_items(self) -> dict:
        roi_key = self.roi.key
        items = dict(name=StandardItem(self.roi.name, roi_key),
                     key=StandardItem(str(self.roi.key), roi_key),
                     type=StandardItem(str(self.roi.type.name), roi_key, False),
                     confidence_level_name=StandardItem(str(self.roi.confidence_level_name), roi_key),
                     score=StandardItem(str(self.roi.score), roi_key),
                     cif_file = StandardItem(str(self.roi.cif_file), roi_key)
                     )
        items['score'].setEditable(False)
        for key in 'radius radius_width angle angle_width'.split():
            items[key] = StandardItem('', roi_key)

        return items

    def move_roi(self):
        for key in 'radius radius_width angle angle_width'.split():
            self.__items[key].setText(f'{getattr(self.roi, key):.2f}')

    def items(self):
        return [self.__items[key] for key in self.PARAM_DICT.keys()]

    def send_move(self):
        pass

    def change_conf_level(self):
        self.__items['confidence_level_name'].setText(str(self.roi.confidence_level_name))

    def change_cif_file(self):
        self.__items['cif_file'].setText(str(self.roi.cif_file))

    def rename(self):
        self.__items['name'].setText(self.roi.name)

    def change_type(self):
        self.__items['type'].setText(str(self.roi.type.name))

    def set_color(self, color):
        for item in self.__items.values():
            item.setBackground(color)


class RoiMetaWidget(AbstractRoiHolder, QWidget):
    def __init__(self, parent=None):
        AbstractRoiHolder.__init__(self, 'ROIMetaWidget')
        QWidget.__init__(self, parent)

        self._model = QStandardItemModel(0, len(RoiWidgetItem.PARAM_DICT), self)
        self._model.setHorizontalHeaderLabels(list(RoiWidgetItem.PARAM_DICT.values()))

        self.tree_view = QTreeView(self)
        self.tree_view.setModel(self._model)

        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._custom_menu)
        self.tree_view.selectionModel().selectionChanged.connect(self._selection_changed)
        self.tree_view.header().setSectionsClickable(True)
        self.tree_view.header().setSortIndicatorShown(True)
        self.tree_view.header().sortIndicatorChanged.connect(self._model.sort)
        self._roi_dict.sig_roi_renamed.connect(self._rename)
        self._roi_dict.sig_cif_renamed.connect(self._change_cif_file)
        self._roi_dict.sigConfLevelChanged.connect(self._change_conf_level)
        self._roi_dict.sig_all_rois_deleted.connect(self.clear_all_rois)

        self._init_ui()

    def _init_connect(self):
        super()._init_connect()
        self._roi_dict.sig_type_changed.connect(self._type_changed)

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.addWidget(self.tree_view)
        layout.addWidget(QPushButton('Add ROI by holding `Ctrl + Alt` and click and hold on the image'))

    def _type_changed(self, key: int):
        self._roi_widgets[key].change_type()

    def _delete_roi_widget(self, item) -> None:

        [self._model.item(row, 6).text() for row in range(self._model.rowCount())]
        item_key = item.roi.key
        row_in_model = next((row for row in range(self._model.rowCount()) if self._model.item(row, 6).text() == str(item_key)), None)

        if row_in_model is not None:
            self.tree_view.selectionModel().blockSignals(True)
            self._model.removeRow(row_in_model)
            self.tree_view.selectionModel().blockSignals(False)
            new_index = self._model.index(row_in_model + 1, 0)
            self.tree_view.setCurrentIndex(new_index)
            self.tree_view.viewport().update()

    def _make_roi_widget(self, roi: Roi) -> AbstractRoiWidget:
        roi_widget = RoiWidgetItem(roi, self.tree_view)
        self._model.appendRow(roi_widget.items())
        for col in range(self._model.columnCount()):
            self.tree_view.resizeColumnToContents(col)
        return roi_widget

    def _custom_menu(self, pos):
        item = self._model.itemFromIndex(self.tree_view.indexAt(pos))
        if item:
            self._roi_widgets[item.key].show_context_menu()

    def _selection_changed(self, item: QItemSelection):
        keys = set(self._model.itemFromIndex(index).key for index in item.indexes())
        if len(keys) == 1:
            self._roi_dict.select(next(iter(keys)))

    def _rename(self, key: int):
        try:
            self._roi_widgets[key].rename()
        except KeyError:
            pass

    def _change_conf_level(self, key: int):
        try:
            self._roi_widgets[key].change_conf_level()
        except KeyError:
            pass

    def _change_cif_file(self, key: int):
        try:
            self._roi_widgets[key].change_cif_file()
        except KeyError:
            pass

    def clear_all_rois(self):
        # Block signals to avoid triggering selectionChanged or other slots
        self.tree_view.selectionModel().blockSignals(True)

        # Remove all rows from the model
        self._model.removeRows(0, self._model.rowCount())

        # Clear internal ROI widget dictionary
        self._roi_widgets.clear()

        # Optionally clear selection
        self.tree_view.clearSelection()

        # Unblock signals
        self.tree_view.selectionModel().blockSignals(False)
