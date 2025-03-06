import logging
from typing import Dict, Tuple, List, DefaultDict, Set
from collections import defaultdict
import gc

from PyQt5.QtCore import QObject, pyqtSignal, Qt, pyqtSlot

from PyQt5.QtWidgets import (QPlainTextEdit, QTreeWidget,
                             QTreeWidgetItem, QWidget, QPushButton,
                             QVBoxLayout, QApplication, QLabel, QSplitter,
                             QListWidget, QListWidgetItem, QLineEdit, QMenu,
                             QCheckBox, QGridLayout, QComboBox)

from PyQt5.QtGui import QColor, QTextCursor

from ..app import App
from ..app.debug_tracker import ObjectTracker, ObjectStatus


def _set_html_color(message, level):
    return f'<font color="{_COLOR_DICT.get(level, "white")}"> {message}'


_COLOR_DICT = dict(DEBUG='green', INFO='white', WARNING='yellow', ERROR='red')
_LEVEL_DICT = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                   WARNING=logging.WARNING, ERROR=logging.ERROR)


class QTextEditLogger(logging.Handler, QObject):
    sigAppendLog = pyqtSignal(str)
    MAX_LENGTH = 300

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)

        self._init_ui(parent)

        # self.cursor = QTextCursor(self.text_widget.document())
        self.sigAppendLog.connect(self.append_log)
        self.setFormatter(
            logging.Formatter(
                '%(levelname)s %(module)s:%(funcName)s %(message)s'))
        logging.getLogger().addHandler(self)
        self.setLevel(logging.DEBUG)

    def _init_ui(self, parent):
        self.widget = QWidget(parent)
        self.level_box = QComboBox(self.widget)
        self.level_box.addItems('DEBUG INFO WARNING ERROR'.split())
        self.level_box.currentIndexChanged.connect(self.level_changed)
        self.text_widget = QPlainTextEdit(self.widget)
        self.text_widget.setReadOnly(True)
        layout = QGridLayout(self.widget)

        layout.addWidget(self.level_box, 0, 0)
        layout.addWidget(self.text_widget, 1, 0)

    @pyqtSlot(int, name='levelBoxIndexChanged')
    def level_changed(self, idx):
        text = self.level_box.currentText()
        self.setLevel(_LEVEL_DICT[text])

    @pyqtSlot(str, name='appendLog')
    def append_log(self, log: str):
        level, *message = log.split()
        message = _set_html_color(' '.join(message), level)
        if len(message) > self.MAX_LENGTH:
            self.text_widget.appendPlainText('')
        self.text_widget.appendHtml(message)
        if len(message) > self.MAX_LENGTH:
            self.text_widget.appendPlainText('')

    def emit(self, record):
        msg = self.format(record)
        try:
            self.sigAppendLog.emit(msg)
        except RuntimeError:
            pass


class TrackerItem(QListWidgetItem):
    def __init__(self, parent: QListWidget, obj: ObjectTracker):
        super().__init__(obj.name, parent)
        self.id = obj.id


class TrackerWidget(QWidget):
    log = logging.getLogger(__name__)

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.tracker = App().debug_tracker
        self.tracker.signals.sigObjectAdded.connect(self._add_object)
        self.tracker.signals.sigStatusUpdated.connect(self._status_updated)
        self.item_dict: Dict[str, TrackerItem] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)

        self.update_button = QPushButton('Update', self)
        self.clear_button = QPushButton('Clear', self)
        self.collect_checkbox = QCheckBox('GC collect', self)
        self.search_line = QLineEdit(parent=self)
        self.splitter = QSplitter(orientation=Qt.Horizontal, parent=self)

        self.list_exists, w1 = self._get_list('Exist')
        self.list_deleted_by_c, w2 = self._get_list('Deleted by c')
        self.list_deleted_by_python, w3 = self._get_list('Deleted by python')
        self.list_safely_deleted, w4 = self._get_list('Safely deleted')

        self.splitter.addWidget(w1)
        self.splitter.addWidget(w2)
        self.splitter.addWidget(w3)
        self.splitter.addWidget(w4)

        num = 3

        layout.addWidget(self.update_button, 0, 0, 1, num)
        layout.addWidget(self.clear_button, 1, 0, 1, num)
        layout.addWidget(self.collect_checkbox, 2, 0)
        layout.addWidget(QLabel('Search: '), 2, 1)
        layout.addWidget(self.search_line, 2, 2)
        layout.addWidget(self.splitter, 3, 0, 1, num)

        self.update_button.clicked.connect(self._update)
        self.clear_button.clicked.connect(self._clear)
        self.search_line.editingFinished.connect(self._search)

        self.list_exists.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_exists.customContextMenuRequested.connect(self._custom_menu(self.list_exists))

        self.list_deleted_by_c.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_deleted_by_c.customContextMenuRequested.connect(self._custom_menu(self.list_deleted_by_c))

    def _custom_menu(self, widget):
        def menu(pos):
            item = widget.itemAt(pos)
            if not item:
                return

            q_menu = QMenu()
            q_menu.addAction('Referrers', lambda *args, key=item.id: self._log_referrers(key))
            q_menu.addAction('Is tracked', lambda *args, key=item.id: self._is_tracked(key))
            q_menu.addAction('Remove', lambda *args, key=item.id: self._remove(key))
            self.tracker.add_object(q_menu)
            q_menu.exec(widget.mapToGlobal(pos))

        return menu

    @pyqtSlot(name='searchItems')
    def _search(self):
        text = self.search_line.text()
        for item in self.item_dict.values():
            if text in item.text():
                item.setHidden(False)
            else:
                item.setHidden(True)

    @pyqtSlot(name='updateLists')
    def _update(self):
        if self.collect_checkbox.isChecked():
            gc.collect()
        self.tracker.update()

    def _log_referrers(self, key: str):
        try:
            obj = self.tracker.objects[key]
        except KeyError as err:
            self.log.exception(err)
            self._remove(key)
            return

        referrers = obj.get_referrers()
        if not referrers:
            self.log.error('No referrers found!')
        else:
            self.log.debug(f'Referrers to {obj.name}:')
            for r in referrers:
                self.log.debug(f'{r.__class__.__name__}: {r}')
            self.log.debug(f'--------------------')

    def _is_tracked(self, key: str):
        try:
            obj = self.tracker.objects[key]
        except KeyError as err:
            self.log.exception(err)
            return
        if obj.is_tracked():
            self.log.debug(f'{obj.name} is tracked by gc.')
        else:
            self.log.error(f'{obj.name} is NOT tracked by gc!')

    def _remove(self, key: str):
        try:
            self.tracker.objects.pop(key)
        except KeyError:
            self.log.debug(f'{key} not in tracker')
        try:
            item = self.item_dict.pop(key)
            parent = item.listWidget()
            parent.takeItem(parent.row(item))
        except KeyError:
            self.log.debug(f'{key} not in item_dict')
            return

    @pyqtSlot(list, name='statusUpdated')
    def _status_updated(self, key_list: list):
        for key in key_list:
            status = self.tracker.objects[key].status
            item = self.item_dict[key]
            parent = item.listWidget()
            new_parent = self._get_list_by_status(status)
            new_parent.insertItem(new_parent.count(), parent.takeItem(parent.row(item)))

    def _clear(self):
        to_delete = self.tracker.clear_deleted()
        for key in to_delete:
            item = self.item_dict.pop(key)
            item.listWidget().removeItemWidget(item)

        self.list_safely_deleted.clear()

    def _get_list_by_status(self, status: ObjectStatus):
        if status == ObjectStatus.EXISTS:
            return self.list_exists
        if status == ObjectStatus.SAFELY_DELETED:
            return self.list_safely_deleted
        if status == ObjectStatus.DELETED_BY_C:
            return self.list_deleted_by_c
        if status == ObjectStatus.DELETED_BY_PYTHON:
            return self.list_deleted_by_python
        raise ValueError('Unknown status')

    @pyqtSlot(str, name='objectAdded')
    def _add_object(self, key: str):
        self.item_dict[key] = TrackerItem(self.list_exists, self.tracker.objects[key])

    def _get_list(self, name):
        widget = QWidget(self)
        label = QLabel(name, widget)
        list_widget = QListWidget(widget)

        layout = QVBoxLayout(widget)
        layout.addWidget(label)
        layout.addWidget(list_widget)

        return list_widget, widget


class TreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        return int(self.text(2)) < int(other.text(2))


def get_color(fr: float):
    return QColor(0, 255, 0, int(fr * 255))


class WidgetList(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.id_dict: Dict[int, Tuple[List[str], int]] = {}
        self.item_dict: DefaultDict[str, Set[tuple]] = defaultdict(set)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.splitter = QSplitter(orientation=Qt.Vertical, parent=self)

        self.label = QLabel('Number of widgets (not calculated)')
        self.update_button = QPushButton('Update')
        self.widget_list = QTreeWidget(self)
        self.differ_list = QListWidget(self)

        self.search_widget = QWidget(self)
        self.search_line = QLineEdit(self.search_widget)
        self.search_tree = QTreeWidget(self.search_widget)

        self.widget_list.setHeaderLabels(['Class name', 'Object name', 'Number of children', 'id'])
        self.search_tree.setHeaderLabels(['Class name', 'Object name', 'Number of children', 'id'])

        s_layout = QVBoxLayout(self.search_widget)
        s_layout.addWidget(self.search_line)
        s_layout.addWidget(self.search_tree)

        layout.addWidget(self.label)
        layout.addWidget(self.update_button)
        layout.addWidget(self.splitter)

        self.splitter.addWidget(self.widget_list)
        self.splitter.addWidget(self.differ_list)
        self.splitter.addWidget(self.search_widget)

        self.update_button.clicked.connect(self.update_widget_list)
        self.search_line.editingFinished.connect(self._update_search_list)

        self.differ_list.hide()

    @pyqtSlot(name='updateSearchList')
    def _update_search_list(self):
        search: str = self.search_line.text().lower()
        self.search_tree.clear()
        for key, value in self.item_dict.items():
            if key.lower().startswith(search):
                for name_list in value:
                    TreeWidgetItem(self.search_tree, name_list)

    def update_widget_list(self):
        self.widget_list.clear()
        self.differ_list.clear()
        self.item_dict.clear()

        count = 0
        new_id_dict: Dict[int, Tuple[list, int]] = {}

        for obj in QApplication.topLevelWidgets():
            count += self._update_list(obj, self.widget_list, new_id_dict)

        self.widget_list.update()
        self.label.setText(f'Number of widgets = {count}')
        self.widget_list.sortByColumn(2, Qt.DescendingOrder)

        for value in self.id_dict.values():
            QListWidgetItem(' '.join(value[0]), self.differ_list)

        self.id_dict = new_id_dict

    def _update_list(self, parent_obj: QObject,
                     parent_node: QTreeWidget or QTreeWidgetItem, new_id_dict: Dict[int, Tuple[List[str], int]]):
        count = 0

        for obj in parent_obj.children():
            name_list = self._get_name_list(obj)
            item = TreeWidgetItem(parent_node, name_list)

            obj_count = 0

            for o in obj.children():
                obj_count += self._update_list(o, item, new_id_dict)

            item.setText(2, str(obj_count))

            name_list[2] = str(obj_count)
            self.item_dict[str(obj.__class__.__name__)].add(tuple(name_list))

            id_ = id(obj)
            new_id_dict[id_] = [item.text(i) for i in range(4)], obj_count

            _, prev_count = self.id_dict.pop(id_, (None, None))

            if prev_count is None:
                fr = 1
            elif prev_count + obj_count == 0:
                fr = 0
            else:
                fr = abs(prev_count - obj_count) / (prev_count + obj_count)
            item.setBackground(3, get_color(fr))

            count += 1 + obj_count

        return count

    @staticmethod
    def _get_name_list(obj: QObject):
        return [str(obj.__class__.__name__), obj.objectName(), '', str(id(obj))]


class DebugWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.show()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle('Debugging Window')

        self.logging_widget = QTextEditLogger(self)
        # self.widget_list = WidgetList(self)
        self.widget_list = TrackerWidget(self)
        self.splitter = QSplitter(orientation=Qt.Vertical, parent=self)
        self.splitter.addWidget(self.logging_widget.widget)
        self.splitter.addWidget(self.widget_list)

        layout.addWidget(self.splitter)
