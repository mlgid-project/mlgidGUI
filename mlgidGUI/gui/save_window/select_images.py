from typing import Iterable, List, Dict, Any

from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QTreeWidgetItemIterator,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from ...app import Roi, App
from ...app.file_manager import ImageKey, FolderKey
from ...app.data_manager import ImageData, ImageDataFlags
from ...gui.image_viewer import ImageViewer


class SelectImagesWindow(QWidget):
    sigApplyClicked = pyqtSignal(dict)

    def __init__(self, path_dict: Dict[FolderKey, List[ImageKey]], parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowModality(Qt.WindowModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.app = App()

        self._init_ui(path_dict)
        self._init_layout()
        self._init_connections()

        if self.app.debug_tracker:
            self.app.debug_tracker.add_object(self, 'SelectImagesWindow')

    def _init_ui(self, path_dict: Dict[FolderKey, List[ImageKey]]):
        self._image_tree_widget = ImageTreeWidget(path_dict, self)
        self._image_viewer = SimpleImageViewer(self)
        self._save_button = QPushButton('Apply')
        self._cancel_button = QPushButton('Cancel')

    def _init_layout(self):
        common_layout = QGridLayout(self)

        splitter = QSplitter(Qt.Horizontal, self)
        common_layout.addWidget(splitter)

        left_widget = QWidget(splitter)
        layout = QGridLayout(left_widget)
        layout.addWidget(self._image_tree_widget, 0, 0, 1, 2)
        layout.addWidget(self._save_button, 1, 0)
        layout.addWidget(self._cancel_button, 1, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(self._image_viewer)
        splitter.setSizes([200, 600])

    def _init_connections(self):
        self._image_tree_widget.sigImageSelected.connect(self._on_image_selected)
        self._save_button.clicked.connect(self._on_save_clicked)
        self._cancel_button.clicked.connect(self.close)

    @pyqtSlot(object, name='imageSelected')
    def _on_image_selected(self, image_key: ImageKey):
        self._image_viewer.clear_rois()

        image_data: ImageData = self.app.data_manager.load_image_data(
            image_key, flags=ImageDataFlags.IMAGE | ImageDataFlags.ROI_DATA)
        if image_data.image is not None:
            self._image_viewer.set_data(image_data.image)
            if image_data.roi_data:
                self._image_viewer.add_rois(image_data.roi_data.values())
        else:
            self._image_viewer.clear_image()

    @pyqtSlot(name='onSavedClicked')
    def _on_save_clicked(self):
        self.sigApplyClicked.emit(self._image_tree_widget.get_path_dict())
        self.close()


class SimpleImageViewer(ImageViewer):
    def _init_connect(self):
        pass

    def add_rois(self, rois: Iterable[Roi]):
        for roi in rois:
            self._roi_widgets[roi.key] = self._make_roi_widget(roi)

    def clear_rois(self):
        for key in list(self._roi_widgets.keys()):
            self._delete_roi_widget(self._roi_widgets.pop(key))


class KeyItem(QTreeWidgetItem):
    def __init__(self, parent, key: ImageKey or FolderKey, is_image: bool = True):
        super().__init__(parent)
        self.key = key
        self.is_image = is_image

    def setData(self, column: int, role: int, value: Any) -> None:
        is_check_change: bool = (column == 0 and
                                 role == Qt.CheckStateRole and
                                 self.data(column, role) is not None and
                                 self.checkState(0) != value)
        super().setData(column, role, value)
        if is_check_change:
            if self.is_image:
                self.treeWidget().sigImageChecked.emit(self.key, self.checkState(0) == Qt.Checked)
            else:
                self.treeWidget().sigFolderChecked.emit(self.key, self.checkState(0) == Qt.Checked)


class MyTree(QTreeWidget):
    sigImageChecked = pyqtSignal(object, bool)
    sigFolderChecked = pyqtSignal(object, bool)

    def __init__(self, parent):
        super().__init__(parent)
        self.setHeaderLabel('Selected images')

    def get_path_dict(self) -> Dict[FolderKey, List[ImageKey]]:
        path_dict: Dict[FolderKey, List[ImageKey]] = {}

        iterator = QTreeWidgetItemIterator(self, QTreeWidgetItemIterator.All)

        while iterator.value():
            item = iterator.value()
            if item.is_image and item.checkState(0) == Qt.Checked:
                folder_key = item.key.parent
                if folder_key not in path_dict:
                    path_dict[folder_key] = []
                path_dict[folder_key].append(item.key)
            iterator += 1

        return path_dict


class ImageTreeWidget(QWidget):
    sigImageSelected = pyqtSignal(object)
    sigPathDictChanged = pyqtSignal(dict)

    def __init__(self, path_dict: Dict[FolderKey, List[ImageKey]], parent):
        super().__init__(parent)
        self.path_dict: Dict[FolderKey, List[ImageKey]] = path_dict
        self._init_ui()

        self.tree.itemSelectionChanged.connect(self._on_item_selected)
        self.tree.sigImageChecked.connect(self._on_image_checked_status_changed)
        self.tree.sigFolderChecked.connect(self._on_folder_checked_status_changed)

    def _init_ui(self):
        layout = QGridLayout(self)
        self.tree = MyTree(self)

        layout.addWidget(self.tree)

        for folder_key, image_keys in self.path_dict.items():
            folder_item = KeyItem(self.tree, folder_key, False)
            folder_item.setText(0, folder_key.name)
            folder_item.setFlags(folder_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            for image_key in image_keys:
                child = KeyItem(folder_item, image_key)
                child.setFlags(child.flags())
                child.setText(0, image_key.name)
                child.setCheckState(0, Qt.Checked)

    @pyqtSlot(object, bool, name='onImageCheckedStatusChanged')
    def _on_image_checked_status_changed(self, image_key: ImageKey, status: bool):
        folder_key = image_key.parent
        if status:
            if folder_key not in self.path_dict:
                self.path_dict[folder_key] = []
            self.path_dict[folder_key].append(image_key)
        else:
            self.path_dict[folder_key].remove(image_key)
        self.sigPathDictChanged.emit(self.path_dict)

    @pyqtSlot(object, bool, name='onFolderCheckedStatusChanged')
    def _on_folder_checked_status_changed(self, folder_key: FolderKey, status: bool):
        self.path_dict = self.tree.get_path_dict()
        self.sigPathDictChanged.emit(self.path_dict)

    def get_path_dict(self) -> Dict[FolderKey, List[ImageKey]]:
        return self.tree.get_path_dict()

    @pyqtSlot(name='onItemClicked')
    def _on_item_selected(self):
        try:
            item = self.tree.selectedItems()[0]
            if item.is_image:
                self.sigImageSelected.emit(item.key)
        except (IndexError, AttributeError):
            return
