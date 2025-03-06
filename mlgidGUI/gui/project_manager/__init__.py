from PyQt5.QtWidgets import QTabWidget

from .cif_results import CIFRingsWidget
from .project_viewer import FileViewer
from .roi_widget import RoiMetaWidget
from .cif_files_widget import CIFFileWidget


class MainFileWidget(QTabWidget):
    def __init__(self, app , parent=None):
        super().__init__(parent)
        self.cif_rois_widget = CIFRingsWidget(self)
        self.setTabPosition(QTabWidget.West)
        self.addTab(FileViewer(app.fm, self), 'Files')
        self.addTab(RoiMetaWidget(self), 'Labeled ROIs')
        self.addTab(CIFFileWidget(app, self), 'CIF-Files')
        self.addTab(self.cif_rois_widget, 'CIF ROIs')