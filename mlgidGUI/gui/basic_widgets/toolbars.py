from PyQt5.QtWidgets import QToolBar
from PyQt5.QtCore import QSize


class ToolBar(QToolBar):
    def __init__(self, name: str, parent=None, color: str = None, *,
                 disable_hide: bool = True, movable: bool = False):
        super().__init__(name, parent)
        if color:
            self.setStyleSheet(f'background-color: {color};')
        if disable_hide:
            self.toggleViewAction().setEnabled(False)
        self.setIconSize(QSize(32,32))
        self.setMovable(movable)


class BlackToolBar(ToolBar):
    def __init__(self, name: str, parent=None, **kwargs):
        super().__init__(name, parent, 'black', **kwargs)