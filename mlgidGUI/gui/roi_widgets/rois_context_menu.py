# -*- coding: utf-8 -*-

from PyQt5.QtGui import QCursor, QRegExpValidator
from PyQt5.QtWidgets import QMenu, QWidgetAction, QLineEdit
from PyQt5.QtCore import QRegExp

from ...app.app import App
from ...app.rois.roi import Roi, RoiTypes
from .confidence_level_widget import SetConfidenceLevelWidget


class AbstractRoiContextMenu(QMenu):

    def __init__(self, roi: Roi):
        QMenu.__init__(self)
        self.roi = roi
        self.key = roi.key
        self.roi_dict = App().roi_dict
        self._init_menu()

        if App().debug_tracker:
            App().debug_tracker.add_object(self)

        self.exec_(QCursor.pos())
        # self.exec_(ev.globalPos())


class RoiContextMenu(AbstractRoiContextMenu):
    def _init_menu(self):
        self._init_rename_menu()
        self._init_conf_level_menu()
        self._init_cif_file_menu()
        self.addSeparator()
        self._init_type_menu()
        self.addSeparator()
        self._init_fix_menu()
        self.addSeparator()
        self._init_copy_menu()
        self.addSeparator()
        self._init_select_menu()
        self.addSeparator()
        self._init_fit_menu()
        self.addSeparator()
        self._init_delete_menu()

    def _init_fix_menu(self):
        fix_menu = self.addMenu('Fix/Unfix')
        if self.roi.movable:
            fix_action = fix_menu.addAction('Fix roi')
            fix_action.triggered.connect(lambda: self.roi_dict.fix_roi(self.key))
        else:
            fix_action = fix_menu.addAction('Unfix roi')
            fix_action.triggered.connect(lambda: self.roi_dict.unfix_roi(self.key))

        fix_selected = fix_menu.addAction('Fix selected roi')
        fix_selected.triggered.connect(self.roi_dict.fix_selected)
        unfix_selected = fix_menu.addAction('Unfix selected roi')
        unfix_selected.triggered.connect(self.roi_dict.unfix_selected)

        fix_all = fix_menu.addAction('Fix all roi')
        fix_all.triggered.connect(self.roi_dict.fix_all)

        unfix_all = fix_menu.addAction('Unix all roi')
        unfix_all.triggered.connect(self.roi_dict.unfix_all)

    def _init_delete_menu(self):
        delete_menu = self.addMenu('Delete')
        delete_self = delete_menu.addAction('Delete this roi')
        delete_self.triggered.connect(lambda: self.roi_dict.delete_roi(self.key))
        delete_selected = delete_menu.addAction('Delete selected')
        delete_selected.triggered.connect(self.roi_dict.delete_selected_roi)

    def _init_select_menu(self):
        select_menu = self.addMenu('Select')
        select_all = select_menu.addAction('Select all')
        select_all.triggered.connect(self.roi_dict.select_all)
        unselect_all = select_menu.addAction('Unselect all')
        unselect_all.triggered.connect(self.roi_dict.unselect_all)

    def _init_rename_menu(self):
        rename = self.addMenu('Rename')
        rename_action = QWidgetAction(self)
        line_edit = QLineEdit(self.roi.name)
        line_edit.editingFinished.connect(
            lambda: self.roi_dict.change_name(self.key, line_edit.text())
        )
        rename_action.setDefaultWidget(line_edit)
        rename.addAction(rename_action)

    def _init_conf_level_menu(self):
        conf_level = self.addAction(f'Change confidence ({self.roi.confidence_level_name})')
        conf_level.triggered.connect(self._change_confidence_widget)

    def _init_cif_file_menu(self):
        cif_file = self.addMenu('Change cif_file')
        rename_action = QWidgetAction(self)
        line_edit = QLineEdit(str(self.roi.cif_file))
        rx = QRegExp("^[A-Za-z0-9]*$")
        validator = QRegExpValidator(rx, line_edit)
        line_edit.setValidator(validator)
        line_edit.editingFinished.connect(
            lambda: self.roi_dict.change_cif_file(self.key, line_edit.text())
        )
        rename_action.setDefaultWidget(line_edit)
        cif_file.addAction(rename_action)

    def _change_confidence_widget(self, *args):
        widget = SetConfidenceLevelWidget(self.roi.confidence_level_name)
        widget.sigSetConfidenceLevel.connect(self._conf_level_changed)
        widget.move(QCursor.pos())
        widget.show()
        self._conf_widget = widget

    def _conf_level_changed(self, level: float):
        self.roi_dict.change_conf_level(self.roi.key, level)

    def _init_type_menu(self):
        available_types = []
        roi = self.roi

        if not roi.type == RoiTypes.segment:
            available_types.append(('segment', RoiTypes.segment))
        if not roi.type == RoiTypes.ring:
            available_types.append(('ring', RoiTypes.ring))
        if not roi.type == RoiTypes.background:
            available_types.append(('background', RoiTypes.background))

        for name, new_type in available_types:
            change_type_action = self.addAction(f'Change type to {name}')
            change_type_action.triggered.connect(
                lambda *x, t=new_type: self._change_roi_type(t))

    def _change_roi_type(self, new_type: RoiTypes):
        self.roi.type = new_type
        self.roi_dict.change_roi_type(self.key)

    def _init_fit_menu(self):
        fit_menu = self.addMenu('Fit')
        fit_selected = fit_menu.addAction('Fit selected rois')
        fit_selected.triggered.connect(lambda: self.roi_dict.open_fit_rois(True))
        fit_selected = fit_menu.addAction('Fit all rois')
        fit_selected.triggered.connect(lambda: self.roi_dict.open_fit_rois(False))

    def _init_copy_menu(self):
        copy_menu = self.addMenu('Copy/Paste')
        copy_menu.addAction('Copy roi', lambda: self.roi_dict.copy_rois(self.roi.key))
        copy_menu.addAction('Copy selected rois', lambda: self.roi_dict.copy_rois('selected'))
        copy_menu.addAction('Copy all rois', lambda: self.roi_dict.copy_rois('all'))
        if self.roi_dict.is_copied:
            copy_menu.addAction('Paste rois', lambda: self.roi_dict.paste_rois())
