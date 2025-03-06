# -*- coding: utf-8 -*-

import logging
from typing import List
from pathlib import Path
from os import path

from PyQt5.QtWidgets import (QWidget, QPushButton, QGridLayout,
                             QVBoxLayout, QFileDialog, QLabel,
                             QLineEdit, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from mlgidGUI.gui.dock_area import AppDockArea
from mlgidGUI.gui.load_window import LoadProjectFolder, LoadFromH5Widget
from mlgidGUI.app import App
from .tools import center_widget, show_error, Icon, color_animation, get_new_project_folder, get_folder_filepath
from .basic_widgets import Label
from ..__version import __version__

class InitWindow(QWidget):
    sigOpenProject = pyqtSignal(object)
    sigExit = pyqtSignal()

    log = logging.getLogger(__name__)

    WIDTH = 600
    HEIGHT = 700

    def __init__(self,
                 recent_projects: List[Path] = None,
                 is_updated: bool = False):
        flags = Qt.WindowFlags()
        flags |= Qt.FramelessWindowHint
        super().__init__(flags=flags)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle('mlgidGUI')
        self.setWindowIcon(Icon('window_icon'))
        self.project_path = get_new_project_folder()
        self.project_name: str = path.basename(self.project_path)
        self.dock_area: AppDockArea = AppDockArea()

        self.__init_ui(recent_projects, is_updated)
        self.setMaximumHeight(self.HEIGHT)
        self.setFixedWidth(self.WIDTH)
        center_widget(self)
        self.show()

    def __init_ui(self, recent_projects: List[Path] or None, is_updated: bool = False):
        layout = QGridLayout(self)
        layout.setContentsMargins(40, 20, 40, 40)
        self.file_line = QLineEdit(str(self.project_path), self)
        self.file_line.textEdited.connect(self._on_text_editing)

        browse_action = self.file_line.addAction(Icon('folder'), QLineEdit.LeadingPosition)

        browse_action.triggered.connect(self._new_project_dialog)
        self.create_button = QPushButton(f'Create project "{self.project_name}"', self)
        self.create_button.clicked.connect(self._create)

        if is_updated:
            title = f'mlgidGUI. Updated to the latest version {__version__}!'
        else:
            title = f'mlgidGUI (version {__version__})'

        layout.addWidget(Label(title, self, 11, True), 0, 0, 1, 3, alignment=Qt.AlignHCenter)
        layout.addWidget(Label('', self, 9.5), 1, 0, 1, 3)
        layout.addWidget(Label('Create new project', self, 9.5), 2, 0, 1, 3)
        layout.addWidget(self.file_line, 3, 0, 1, 2)
        layout.addWidget(self.create_button, 3, 2)
        layout.addWidget(Label('Import from H5', self, 9.5), 5, 0, 1, 3)
        import_h5_button = QPushButton('Import from H5', self)
        import_h5_button.clicked.connect(self.__init__h5_button)
        layout.addWidget(import_h5_button, 6, 0, 1, 3)
        layout.addWidget(Label('Open project folder', self, 9.5), 7, 0, 1, 3)
        open_project_button = QPushButton('Open project folder', self)
        open_project_button.clicked.connect(self.__init__open_project_button)
        layout.addWidget(open_project_button, 8, 0, 1, 3)
        layout.addWidget(Label('Recent projects', self, 9.5), 9, 0, 1, 3)

        if recent_projects:
            layout.addWidget(self._init_recent_projects_area(recent_projects), 10, 0, 1, 3)

        layout.addLayout(self._init_exit_button(), 12 if recent_projects else 11, 0, 1, 3)

    def __init__h5_button(self, ):
        app = App()
        load_window = LoadFromH5Widget(app.data_manager.load_project_from_h5, self)
        if load_window.exec() == 1:
            self.sigOpenProject.emit(load_window.project_path)
        else:
             return None

    def __init__open_project_button(self, ):
        app = App()
        load_window = LoadProjectFolder(app.data_manager.load_project_from_h5, self)
        if load_window.exec() == 1:
            self.sigOpenProject.emit(load_window.project_path)
        else:
             return None

    def _init_exit_button(self):
        e_layout = QVBoxLayout()
        exit_button = QPushButton('Exit')
        exit_button.clicked.connect(self.sigExit)
        e_layout.addWidget(QLabel(''))
        e_layout.addWidget(exit_button)
        return e_layout

    def _init_recent_projects_area(self, recent_projects):
        q_scroll_area = QScrollArea(self)
        q_scroll_area.setWidgetResizable(True)
        q_scroll_area.setGeometry(0, 0, self.WIDTH, self.HEIGHT)

        projects_list_widget = QWidget(self)
        projects_list_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        scroll_layout = QVBoxLayout(projects_list_widget)
        scroll_layout.setContentsMargins(0,0,5,0)
        for path in recent_projects:
            btn = QPushButton(path.name)
            btn.clicked.connect(
                lambda *x, p=path: self.sigOpenProject.emit(p))
            scroll_layout.addWidget(btn)
        scroll_layout.addStretch()
        q_scroll_area.setWidget(projects_list_widget)

        return q_scroll_area

    @pyqtSlot(str, name='onTextEditing')
    def _on_text_editing(self, path: str):
        path = Path(path)
        folder, name = path.parent, path.name
        if not folder.is_dir():
            folder = self.project_path.parent
        self.project_name = name
        self.project_path = folder / name
        self.create_button.setText(f'Create project "{self.project_name}"')

    @pyqtSlot(name='NewProjectDialog')
    def _new_project_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'New project folder', options=
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return
        folder = Path(folder).resolve()
        if folder.is_dir():
            self.project_path = folder / self.project_name
            self.file_line.setText(str(self.project_path))
        else:
            color_animation(self.file_line)
            show_error('Invalid folder. Please, select existing folder',
                       error_title='Wrong path')

    def _create(self):
        try:
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.sigOpenProject.emit(self.project_path.resolve())
        # except FileExistsError:
        #     show_error(f'The folder {str(self.project_path.resolve())} already exists', error_title='Folder exists')
        #     color_animation(self.file_line)
        except PermissionError:
            folder = get_folder_filepath(self, 'Permission denied, please, create a project folder manually',
                                         directory=str(Path('~').expanduser().resolve()))
            if folder and folder.is_dir():
                self.sigOpenProject.emit(folder.resolve())
