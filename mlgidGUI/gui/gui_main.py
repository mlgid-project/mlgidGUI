from pathlib import Path
import logging

from PyQt5.QtWidgets import (QMainWindow, QWidget, QSizePolicy,
                             QApplication, QShortcut, QMessageBox,
                             QFileDialog)
from PyQt5.QtCore import (Qt, pyqtSlot, pyqtSignal,
                          QObject)
from PyQt5.QtGui import QKeySequence

import qdarkstyle
import qdarkgraystyle

from ..app import App

from ..__version import __version__

from .dock_area import AppDockArea
from .basic_widgets import ToolBar, confirmation_box
from .tools import Icon, get_filepath_dialog, get_folder_filepath, save_file_dialog

from .init_window import InitWindow
from .debug_widgets import DebugWindow
from .exception_message import UncaughtHook
from .notifications import PopUpWrapper
from .background_tasks import BackgroundTasks
from .background_update import BackgroundUpdate


class GIWAXSMainController(QObject):
    EXIT_CODE_REBOOT: int = -123456789

    log = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()
        self.app = App(parent=self)
        self.background_tasks = BackgroundTasks(self)
        self.background_update = BackgroundUpdate(self)
        self.background_update.sigRestartAfterUpdate.connect(self.restart_after_update)

        self.init_window = None
        self.main_window = None

        if self.log.level <= logging.DEBUG:
            self.debug_window = DebugWindow()
        self.exception_hook = UncaughtHook()

        self.log.info(f'{"*" * 10}')
        self.log.info(f'Starting mlgidGUI {__version__}!')
        self.log.info(f'{"*" * 10}')
        self.set_style(self.app.fm.config['style'] or 'Gray Dark')
        self.open_init_window()

    def open_main_window(self):
        self.main_window = GIWAXSMainWindow(self.app)

        self.main_window.sigCloseApp.connect(self.close_app)
        self.main_window.sigRestartApp.connect(self.restart_app)
        self.main_window.sigOpenProject.connect(self.open_new_project)
        self.main_window.sigCloseProject.connect(self.close_project)
        self.main_window.sigSetStyle.connect(self.set_style)

        self.background_update.run()

    def open_init_window(self):
        self.init_window = InitWindow(self.app.fm.recent_projects, self.background_update.is_updated)
        self.init_window.sigOpenProject.connect(self.open_new_project)
        self.init_window.sigExit.connect(self.close_app)

    @pyqtSlot(name='closeProject')
    def close_project(self):
        self.app.fm.close_project()
        self.main_window.hide()
        self.open_init_window()

    @pyqtSlot(object, name='NewProject')
    def open_new_project(self, path: Path):
        if self.init_window:
            self.init_window.close()
            self.init_window = None
        if not self.main_window:
            self.open_main_window()
        else:
            self.main_window.show()
        self.app.fm.open_project(path)

    @pyqtSlot(name='restartAppAfterUpdate')
    def restart_after_update(self):
        self.app.fm.config['is_updated'] = True
        self.restart_app()

    @pyqtSlot(name='restartApp')
    def restart_app(self):
        self.app.close()
        q_app = QApplication.instance()
        App._instance = None
        q_app.exit(self.EXIT_CODE_REBOOT)

    @pyqtSlot(name='closeApp')
    def close_app(self):
        self.app.close()
        q_app = QApplication.instance()
        App._instance = None
        q_app.exit(0)

    @pyqtSlot(str, name='setStyle')
    def set_style(self, name: str = 'Gray Dark'):
        q_app = QApplication.instance()
        if not q_app:
            raise RuntimeError('No running application found.')
        if name == 'Dark':
            q_app.setStyleSheet('')
            q_app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        elif name == 'Gray Dark':
            q_app.setStyleSheet('')
            q_app.setStyleSheet(qdarkgraystyle.load_stylesheet())
        else:
            return

        self.app.fm.config['style'] = name


class GIWAXSMainWindow(QMainWindow):
    _MinimumSize = (500, 500)

    sigCloseApp = pyqtSignal()
    sigOpenProject = pyqtSignal(object)
    sigCloseProject = pyqtSignal()
    sigRestartApp = pyqtSignal()
    sigSetStyle = pyqtSignal(str)

    def __init__(self, app: App, parent=None):
        super(GIWAXSMainWindow, self).__init__(parent=parent)
        self.__closing: bool = False
        self.app = app
        self.dock_area = AppDockArea(self)
        self.popups = PopUpWrapper(self)
        self._connect_background_tasks()

        self._init_toolbar()
        self._init_shortcuts()
        self._init_menubar()

        self.app.fm.sigProjectClosed.connect(self.update_window_title)
        self.app.fm.sigProjectOpened.connect(self.update_window_title)

        self.setCentralWidget(self.dock_area)
        self.update_window_title()
        self.setWindowIcon(Icon('window_icon'))
        self.setMinimumSize(*self._MinimumSize)
        self.setWindowState(Qt.WindowMaximized)
        self.show()
        #open the fit view and the angular profile view by default
        self.dock_area.show_hide_docks('fit_view')
        self.dock_area.show_hide_docks('angular_profile')


    def _connect_background_tasks(self):
        background_tasks = BackgroundTasks()
        background_tasks.tasks.sigAddNotification.connect(self.popups.add_notification)
        background_tasks.tasks.sigRemoveNotification.connect(self.popups.remove_by_name)
        background_tasks.tasks.sigAddProgressBar.connect(self._add_progress_bar)

    def _add_progress_bar(self, progress_bar_func):
        return progress_bar_func(self)

    def update_window_title(self):
        if self.app.fm.project_opened:
            self.setWindowTitle(f'{self.app.fm.project_name} - mlgidGUI')
        else:
            self.setWindowTitle('mlgidGUI')

    @pyqtSlot(name='NewProjectDialog')
    def _new_project_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'New project folder')
        if not folder:
            return
        folder = Path(folder)
        if folder.is_dir():
            self.sigOpenProject.emit(folder)

    @pyqtSlot(name='addExSitu')
    def _add_file_dialog(self, cif = False):
        if cif:
            filepath = get_filepath_dialog(self, message='Select CIF file', formats ='Cif file (*.cif)')
            if filepath:
                self.app.data_manager.import_cif(filepath)
        else:
            filepath = get_filepath_dialog(self)
            if filepath:
                self.app.data_manager.import_image(filepath)

    def _save_as_h5(self):
        path = save_file_dialog(self, title='Save project')
        if path:
            self.app.fm.save_as_h5(path)

    def _add_folder_dialog(self):
        filepath = get_folder_filepath(self, message='Choose directory containing images or h5 files')
        if filepath:
            folder_size = self.app.data_manager.estimate_folder_size(filepath)
            confirmation = confirmation_box("Import whole folder", "Do  you whish to import the folder?",
                                           "This would copy " + folder_size + " of data. ")
            if confirmation:
                self.app.data_manager.import_folder(filepath)

    def _init_menubar(self):
        self.menubar = self.menuBar()

        # File menu

        self.file_menu = self.menubar.addMenu('File')
        self.file_menu.addAction('New project', self._new_project_dialog)
        recent_projects_menu = self.file_menu.addMenu('Recent projects')

        for project_path in self.app.fm.recent_projects:
            recent_projects_menu.addAction(
                project_path.name, lambda *x, p=project_path: self.sigOpenProject.emit(p))

        # Save

        self.save_action = self.file_menu.addAction('Save project ...', lambda *x: self.dock_area.open_save_window())

        # Load from h5
        self.save_action = self.file_menu.addAction(
            'Load from h5', lambda *x: self.dock_area.open_load_from_h5_window()
        )

        # Close project

        self.file_menu.addAction('Close project', lambda *x: self.sigCloseProject.emit())

        # Restart

        self.file_menu.addAction('Restart', lambda *x: self.restart())

        # Data menu

        self.data_menu = self.menubar.addMenu('Data')

        self.data_menu.addAction('Add image file', self._add_file_dialog)
        self.data_menu.addAction('Add CIF file', lambda: self._add_file_dialog(cif=True))
        self.data_menu.addAction('Add folder', self._add_folder_dialog)

        # Preferences menu

        self.preferences = self.menubar.addMenu('Preferences')
        self.themes_menu = self.preferences.addMenu('Themes')
        themes = ['Dark', 'Gray Dark']
        # themes = CSS.list_css() + QStyleFactory.keys()
        for theme in themes:
            theme_action = self.themes_menu.addAction(theme)
            theme_action.triggered.connect(lambda *x, t=theme: self.sigSetStyle.emit(t))

    def _init_shortcuts(self):
        self.copy_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.copy_shortcut.activated.connect(lambda *x: self.app.roi_dict.copy_rois('selected'))

        self.paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        self.paste_shortcut.activated.connect(lambda *x: self.app.roi_dict.paste_rois())

        self.select_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.select_shortcut.activated.connect(lambda *x: self.app.roi_dict.select_all())

        self.delete_shortcut = QShortcut(QKeySequence('Del'), self)
        self.delete_shortcut.activated.connect(lambda *x: self.app.roi_dict.delete_selected_roi())

        self.fit_shortcut = QShortcut(QKeySequence('Ctrl+F'), self)
        self.fit_shortcut.activated.connect(lambda *x: self.app.roi_dict.open_fit_rois(True))

        self.save_shortcut = QShortcut(QKeySequence('Ctrl+S'), self)
        self.save_shortcut.activated.connect(self.save_state)

        def raise_err(*_):
            raise ValueError('Congratulations, you found an error combination used for testing! '
                             'Your project will be deleted in 3 seconds...')

        self.raise_shortcut = QShortcut(QKeySequence('Ctrl+R'), self)
        self.raise_shortcut.activated.connect(raise_err)

    def save_state(self, *args):
        self.app.save_state()
        res = QMessageBox.information(
            self, 'Image is saved',
            'The fitted peaks are successfully saved to the project folder.',
             QMessageBox.Ok
        )

    def _init_toolbar(self):

        docks_toolbar = ToolBar('Docks', self)
        docks_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(docks_toolbar)

        control_widget = docks_toolbar.addAction(Icon('folder'), 'Project Manager')
        control_widget.triggered.connect(lambda: self.dock_area.show_hide_docks('file_widget'))

        radial_profile = docks_toolbar.addAction(Icon('radial_profile'), 'Radial Profile')
        radial_profile.triggered.connect(lambda: self.dock_area.show_hide_docks('radial_profile'))

        angular_profile = docks_toolbar.addAction(Icon('angular_profile'), 'Angular Profile')
        angular_profile.triggered.connect(lambda: self.dock_area.show_hide_docks('angular_profile'))

        polar_viewer = docks_toolbar.addAction(Icon('interpolate'), 'Polar Viewer')
        polar_viewer.triggered.connect(lambda: self.dock_area.show_hide_docks('polar'))

        fit_view = docks_toolbar.addAction(Icon('fit'), 'Radial Fit')
        fit_view.triggered.connect(lambda: self.dock_area.show_hide_docks('fit_view'))

        self.gen_toolbar = ToolBar('General')
        self.gen_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(self.gen_toolbar)
        spacer_widget = QWidget()
        spacer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer_widget.setVisible(True)
        self.gen_toolbar.addWidget(spacer_widget)

        self.fullscreen_action = self.gen_toolbar.addAction(Icon('tofullscreen'), 'Full screen')
        self.fullscreen_action.triggered.connect(self._on_fullscreen_changed)

    def _on_fullscreen_changed(self):
        if self.isFullScreen():
            self.setWindowState(Qt.WindowMaximized)
            self.fullscreen_action.setIcon(Icon('tofullscreen'))
        else:
            self.setWindowState(Qt.WindowFullScreen)
            self.fullscreen_action.setIcon(Icon('fromfullscreen'))

    @pyqtSlot(name='restartApp')
    def restart(self):
        self.__closing = True
        self.sigRestartApp.emit()

    def closeEvent(self, a0) -> None:
        if self.__closing:
            super().closeEvent(a0)
        else:
            reply = QMessageBox.question(self, 'Message',
                                         "Are you sure to quit? The project will be saved.",
                                         QMessageBox.Yes |
                                         QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                a0.accept()
                self.__closing = True
                self.sigCloseApp.emit()
            else:
                a0.ignore()
