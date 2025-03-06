from dataclasses import dataclass
from PyQt5.QtWidgets import (QGridLayout, QTableWidget, QTableWidgetItem,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget, QCheckBox, QMenu, QSlider)
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QCursor
from PyQt5 import QtCore
from mlgidGUI.app.cif_rois.cif_peaks_calculation import calculate_cif_peaks, cif_results
from mlgidGUI.gui.tools import show_info
PARAM_LIST = [' ','Name','Powder\ndiffraction','\nh','contact plane\nk','\nl','energy [eV]','show_only the ..%\nhighest intensities']

@dataclass
class CIFFileRow():
    cif_key: None
    row_num: int
    filename: str
    show_button: QPushButton
    random_button: QCheckBox
    orientation1: QTableWidgetItem
    orientation2: QTableWidgetItem
    orientation3: QTableWidgetItem
    en: QTableWidgetItem

class CIFFileWidget(QWidget):

    def __init__(self, app, parent=None):
        QWidget.__init__(self, parent)
        self._app = app
        self._fm = app.fm
        self.parent = parent
        self._fm.sigNewCIFFile.connect(self.update_view)
        self.input_content_dict = None
        # Create the table widget
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setRowCount(0)  # Set initial row count
        self.tableWidget.setColumnCount(len(PARAM_LIST))  # Set two columns
        self.tableWidget.setHorizontalHeaderLabels(PARAM_LIST)

        # Create the "Add Entry" button
        self.add_button = QPushButton("Add CIF File", self)
        self.add_button.clicked.connect(lambda: self._app.parent.main_window._add_file_dialog(cif=True))

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        layout.addWidget(self.add_button)
        layout.setContentsMargins(0, 0, 0, 0)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        layout = QGridLayout(self)
        layout.addWidget(central_widget)

        self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.mouseMenu)

        self.cif_files = {}

    def update_view(self):
        self.tableWidget.setRowCount(0)
        self.cif_files = {}
        for cif_path_key in self._fm._project_structure.root._cif_files:
            self._add_new_cif_to_view(cif_path_key)
        self._display_cif_peaks()

    def _add_new_cif_to_view(self, cif_key: str):

        show_button = QPushButton("")
        random_button = QPushButton("")
        row_num = self.tableWidget.rowCount()
        int_val = QIntValidator(0,100)
        float_val = QDoubleValidator(0, 90, 4)
        orientation1 = QLineEdit('0')
        orientation1.setValidator(int_val)
        orientation1.setMaxLength(3)
        orientation1.setMaximumWidth(40)
        orientation1.setAlignment(QtCore.Qt.AlignHCenter)
        orientation2 = QLineEdit('0')
        orientation2.setValidator(int_val)
        orientation2.setMaxLength(3)
        orientation2.setMaximumWidth(40)
        orientation2.setAlignment(QtCore.Qt.AlignHCenter)
        orientation3 = QLineEdit('1')
        orientation3.setValidator(int_val)
        orientation3.setMaxLength(3)
        orientation3.setMaximumWidth(40)
        orientation3.setAlignment(QtCore.Qt.AlignHCenter)
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.sliderReleased.connect(lambda: self.slider_released(slider))

        en = QLineEdit('18000')
        en.setValidator(float_val)
        en.setAlignment(QtCore.Qt.AlignHCenter)

        self.cif_files[cif_key.name] = CIFFileRow(
            cif_key = cif_key,
            row_num=row_num,
            filename=cif_key.name,
            show_button = show_button,
            random_button = random_button,
            orientation1 = orientation1,
            orientation2 = orientation2,
            orientation3 = orientation3,
            en = en
            )

        self.tableWidget.insertRow(row_num)
        self._set_show_button_text(show_button, cif_key)
        show_button.clicked.connect(lambda: self._show_button_clicked(show_button, cif_key))

        self.tableWidget.setCellWidget(row_num, 0, show_button)
        self.tableWidget.setItem(row_num, 1, QTableWidgetItem(cif_key.name))

        self.tableWidget.setCellWidget(row_num, 2, random_button)
        self._set_random_button(random_button, cif_key)
        random_button.clicked.connect(lambda: self._random_button_clicked(random_button, cif_key))

        self.tableWidget.setCellWidget(row_num, 7, slider)
        self.tableWidget.setCellWidget(row_num, 3, orientation1)
        self.tableWidget.setCellWidget(row_num, 4, orientation2)
        orientation2.setMaximumWidth(self.tableWidget.columnWidth(4))
        self.tableWidget.setCellWidget(row_num, 5, orientation3)
        self.tableWidget.setCellWidget(row_num, 6, en)
        en.setMaximumWidth(self.tableWidget.columnWidth(6))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

    def slider_released(self, slider):
        self._app.image_holder.sigCIFPeaksChanged.emit(slider.value()/100)

    def _set_show_button_text(self, button, cif_key):
        if cif_key.is_visible():
            button.setText("Hide")
        else:
            button.setText("Show")

    def _set_random_button(self, button, cif_key):
        if cif_key.powder_diffraction:
            button.setText("Deactivate")
        else:
            button.setText("Activate")

    def _random_button_clicked(self, button, cif_key):
        cif_key.change_powder_diffraction()
        self._display_cif_peaks()
        self._set_random_button(button, cif_key)

    def _show_button_clicked(self, button, cif_key):
        orientation = [int(self.cif_files[cif_key.name].orientation1.text()),
                       int(self.cif_files[cif_key.name].orientation2.text()),
                       int(self.cif_files[cif_key.name].orientation3.text())]
        if all(v == 0 for v in orientation):
            raise Exception('all orientations 0 is not allowed')
        cif_key.orientation = orientation
        cif_key.en = float(self.cif_files[cif_key.name].en.text())
        cif_key.change_show_hide()

        self._display_cif_peaks()
        self._set_show_button_text(button, cif_key)

    def _display_cif_peaks(self):
        error_key = calculate_cif_peaks(self._fm._project_structure.root, self._fm.current_key)
        if error_key is not None:
            show_info('Error with CIF File', 'CIF File ' + error_key.name + ' could not be calculated, check content!')
            self._fm._project_structure.root._cif_files.remove(error_key)
            self.update_view()
            return

        self.parent.cif_rois_widget.delete_all()
        if bool(cif_results):
            for key, cif_result in cif_results.items():
                self.parent.cif_rois_widget._make_roi_widget(cif_result)
        else:
            self.parent.cif_rois_widget.empty_simulation()
        self.parent.cif_rois_widget.resizeColumns()
        self._app.image_holder.sigCIFPeaksChanged.emit(0)

    def mouseMenu(self, pos):
        selected = self.tableWidget.selectedIndexes()
        if not selected:
            return
        menu = QMenu()
        deleteAction = menu.addAction('Delete CIF-File')
        deleteAction.triggered.connect(lambda: self.removeRows(selected))
        menu.exec_(QCursor.pos())

    def removeRows(self, indexes):
        rows = set(index.row() for index in indexes)
        for row in sorted(rows, reverse=True):
            self.tableWidget.removeRow(row)
            for cif_file_row in self.cif_files:
                if self.cif_files[cif_file_row].row_num == row:
                    self._fm._project_structure.root._cif_files.remove(self.cif_files[cif_file_row].cif_key)
        self.update_view()