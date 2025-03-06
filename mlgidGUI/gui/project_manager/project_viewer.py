from pathlib import Path

from PyQt5.QtWidgets import (QTreeView, QMenu,
                             QWidget, QHBoxLayout, QLabel)
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt

from ..basic_widgets import RoundedPushButton
from ..tools import Icon, get_folder_filepath, get_filepath_dialog
from ...app.file_manager import FileManager, ImageKey, FolderKey


class FileModel(QStandardItemModel):
    def __init__(self):
        super(FileModel, self).__init__()
        self.setHorizontalHeaderLabels([''])
        self.setRowCount(0)

    def new_project(self):
        self.removeRows(1, self.rowCount() - 1)


class ImageItem(QStandardItem):

    def __init__(self, key: ImageKey):
        super().__init__(key.name)
        self.key: ImageKey = key
        self.setIcon(Icon('data'))


class FolderItem(QStandardItem):

    def __init__(self, key: FolderKey):
        super().__init__(key.name)
        self.key = key
        self.setIcon(Icon('folder'))
        self._updated: bool = False

    def on_clicked(self):
        if not self._updated:
            self.update()
        self._updated = True

    def _fill(self):
        for folder in self.key.folder_children:
            self.appendRow(FolderItem(folder))
        for image in self.key.image_children:
            self.appendRow(ImageItem(image))

    def update(self):
        self.clear()
        self.key.update()
        self._fill()

    def clear(self):
        self.removeRows(0, self.rowCount())


class FileViewer(QTreeView):

    def __init__(self, fm: FileManager, parent=None):
        super().__init__(parent)
        self._model = FileModel()
        self.setModel(self._model)
        self.setEditTriggers(QTreeView.NoEditTriggers)
        self._fm = fm
        self._fm.sigNewFile.connect(self._add_new_file)
        self._fm.sigNewFolder.connect(self._add_new_folder)
        self._fm.sigProjectClosed.connect(self._model.new_project)

        self.selectionModel().currentChanged.connect(self._on_clicked)
        self.customContextMenuRequested.connect(self._context_menu)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self._init_ui()
        self.show()

    def _init_ui(self):
        add_file_button = RoundedPushButton(icon=Icon('data'), radius=30,
                                            background_color='transparent')
        add_file_button.clicked.connect(self._open_add_file_menu)
        add_folder_button = RoundedPushButton(icon=Icon('folder'), radius=30,
                                              background_color='transparent')
        add_folder_button.clicked.connect(self._open_add_folder_menu)
        layout = self._get_header_layout(QStandardItem(), 'Files')
        layout.addWidget(add_file_button)
        layout.addWidget(add_folder_button)

    def _get_header_layout(self, item: QStandardItem, label: str):
        header_widget = QWidget(self)
        layout = QHBoxLayout()
        header_widget.setLayout(layout)
        label_widget = QLabel(label)
        layout.addWidget(label_widget, alignment=Qt.AlignLeft)
        layout.addStretch(1)
        self._model.appendRow(item)
        self.setIndexWidget(item.index(), header_widget)
        return layout

    def clear(self):
        self._model.removeRows(0, self._model.rowCount())
        self._init_ui()

    def _add_new_file(self, key: ImageKey):
        self._model.appendRow(ImageItem(key))

    def _add_new_cif_file(self, key):
        pass

    def _add_new_folder(self, key: FolderKey):
        self._model.appendRow(FolderItem(key))

    def _on_clicked(self, index):
        item = self._model.itemFromIndex(index)
        if isinstance(item, ImageItem):
            self._fm.change_image(item.key)
        elif isinstance(item, FolderItem):
            item.on_clicked()
            self.setExpanded(item.index(), True)

    def _open_add_file_menu(self):
        path = get_filepath_dialog(self)
        if path:
            self._fm.add_root_path_to_project(path)

    def _open_add_folder_menu(self):
        path = get_folder_filepath(self, message='Choose directory containing images or h5 files')
        if path:
            self._fm.add_root_path_to_project(path)

    def _context_menu(self, position):
        item = self._model.itemFromIndex(self.indexAt(position))
        menu = QMenu()
        if isinstance(item, FolderItem):
            update_folder = menu.addAction('Update folder')
            update_folder.triggered.connect(item.update)
            close_folder = menu.addAction('Remove from project')
            close_folder.triggered.connect(
                lambda *x, it=item: self._remove_item(it))

        elif isinstance(item, ImageItem):
            close_image = menu.addAction('Remove from project')
            close_image.triggered.connect(
                lambda *x, it=item: self._remove_item(it))
        else:
            return
        menu.exec_(self.viewport().mapToGlobal(position))

    def _remove_item(self, item: FolderItem or ImageItem):
        self._fm.remove_key(item.key)
        self.clear()
        self._fm.init_file_viewer()