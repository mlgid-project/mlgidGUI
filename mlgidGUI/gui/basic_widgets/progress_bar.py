# -*- coding: utf-8 -*-
import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from ..tools import center_widget, DummySignal

logger = logging.getLogger(__name__)


class ProgressBar(QWidget):
    sigCancelClicked = pyqtSignal()

    def __init__(self,
                 num: int,
                 progress_label: str,
                 finish_label: str,
                 parent=None,
                 auto_close: bool = False,
                 cancel_btn: bool = False,
                 block_window: bool = True,
                 show: bool = True):
        super().__init__(parent)
        self._progress_label: str = progress_label
        self._finish_label: str = finish_label
        self._auto_close: bool = auto_close
        self._cancel_btn: bool = cancel_btn

        self.setWindowFlag(Qt.Window)
        self.setWindowFlag(Qt.FramelessWindowHint)

        if block_window:
            self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._init_ui(num)
        center_widget(self)
        if show:
            self.show()

    def _init_ui(self, num: int):
        layout = QVBoxLayout(self)
        self.progress = QProgressBar()
        self.progress.setMaximum(num)
        self.label = QLabel(self._progress_label)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        if not self._auto_close:
            self.close_button = self._init_close_button()
            layout.addWidget(self.close_button)

        if self._cancel_btn:
            self.cancel_btn = self._init_cancel_btn()
            layout.addWidget(self.cancel_btn)

        if not num:
            self.progress.setMaximum(1)
            self.progress.setValue(1)
            self.finished()

    def _init_cancel_btn(self):
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.sigCancelClicked.emit)
        cancel_button.hide()
        return cancel_button

    def _init_close_button(self):
        close_button = QPushButton('Close')
        close_button.clicked.connect(self.close)
        close_button.hide()
        return close_button

    @pyqtSlot(int, name='setMax')
    def set_max(self, num: int):
        self.progress.setMaximum(num)
        self.show()
        logger.debug(f'set max called {num}')

    @pyqtSlot(int, name='setProgress')
    def set_progress(self, value: int):
        self.progress.setValue(value)
        logger.debug(f'set_progress {value} ')

    @pyqtSlot(name='finished')
    def finished(self):
        logger.debug('Finished!')
        if self._auto_close:
            self.close()
        else:
            if self._cancel_btn:
                self.cancel_btn.hide()
            self.label.setText(self._finish_label)
            self.progress.setValue(self.progress.maximum())
            self.close_button.show()


def progress_bar_factory(num: int,
                         progress_label: str,
                         finish_label: str,
                         auto_close: bool = False,
                         cancel_btn: bool = False,
                         block_window: bool = True,
                         set_process_sig: DummySignal = None,
                         set_finished_sig: DummySignal = None,
                         set_max_sig: DummySignal = None,
                         cancel_callback=None
                         ):
    def func(parent):
        progress_bar = ProgressBar(parent=parent,
                                   num=num, progress_label=progress_label,
                                   finish_label=finish_label,
                                   auto_close=auto_close,
                                   cancel_btn=cancel_btn,
                                   block_window=block_window)
        if set_process_sig:
            set_process_sig.connect(progress_bar.set_progress)
        if set_finished_sig:
            set_finished_sig.connect(progress_bar.finished)
        if set_max_sig:
            set_max_sig.connect(progress_bar.set_max)
        if cancel_callback:
            progress_bar.sigCancelClicked.connect(cancel_callback)

    return func
