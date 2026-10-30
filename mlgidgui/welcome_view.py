"""The page shown when no file is open.

With nothing loaded the window used to be a void: empty axes, an empty
tree, two empty profile plots. Everything the app can do from a cold
start was already there — Open, image import, a persisted recent-files
list, drag and drop onto the window — but none of it was visible, and
the one thing a new user most needs to know (whether the analysis
backend is even installed) only surfaced when a run failed.

Deliberately not a dashboard: a mark, two actions, the recent list, and
one line of state. Everything here routes to an existing handler on
``MainWindow`` through a signal, so this view holds no file logic.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mlgidgui.widgets import GAP, PAD, PRIMARY, section_label, set_variant

import logging

logger = logging.getLogger(__name__)

#: How many recent files to list. The File menu keeps the full list;
#: this is the "pick up where you left off" shortlist.
RECENT_SHOWN = 6


def _wordmark(theme: str) -> QPixmap:
    """The family wordmark for ``theme`` (null pixmap if unavailable).

    Two variants ship: the dark one has a transparent ground and
    lightened arcs so the mark does not sit in a white box.
    """
    variant = ("mlgid_logo_mlgidgui_dark.png" if theme == "dark"
               else "mlgid_logo_mlgidgui.png")
    try:
        path = resources.files("mlgidgui").joinpath(f"assets/app/{variant}")
        with resources.as_file(path) as real:
            return QPixmap(str(real))
    except Exception:
        logger.debug("no wordmark for the welcome view", exc_info=True)
        return QPixmap()


class WelcomeView(QWidget):
    """The empty-state page for the central area."""

    openRequested = Signal()
    importRequested = Signal()
    #: (path, kind) straight from the persisted recent-files list.
    recentRequested = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(PAD * 4, PAD * 4, PAD * 4, PAD * 4)
        outer.addStretch(1)

        self._mark = QLabel()
        self._mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._mark)
        outer.addSpacing(PAD * 2)

        # No tagline label here: the wordmark asset already carries it.
        outer.addSpacing(PAD * 3)

        actions = QHBoxLayout()
        actions.setSpacing(GAP)
        actions.addStretch(1)
        self.btn_open = set_variant(QPushButton("Open file…"), PRIMARY)
        self.btn_open.clicked.connect(self.openRequested)
        actions.addWidget(self.btn_open)
        self.btn_import = QPushButton("Import images…")
        self.btn_import.setToolTip(
            "Stack a folder of detector images as one scan entry.")
        self.btn_import.clicked.connect(self.importRequested)
        actions.addWidget(self.btn_import)
        actions.addStretch(1)
        outer.addLayout(actions)
        outer.addSpacing(PAD)

        drop_hint = QLabel("or drop a file anywhere on this window")
        drop_hint.setProperty("role", "hint")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(drop_hint)
        outer.addSpacing(PAD * 3)

        # The recent list is held to a readable column width rather than
        # stretched across the window: full-width rows read as banners,
        # not as a list of files.
        recent_column = QWidget()
        recent_column.setMaximumWidth(420)
        self._recent_box = QVBoxLayout(recent_column)
        self._recent_box.setContentsMargins(0, 0, 0, 0)
        self._recent_box.setSpacing(2)
        self._recent_title = section_label("Recent")
        self._recent_box.addWidget(self._recent_title)
        centred = QHBoxLayout()
        centred.addStretch(1)
        centred.addWidget(recent_column)
        centred.addStretch(1)
        outer.addLayout(centred)
        self._recent_buttons: list[QPushButton] = []

        outer.addSpacing(PAD * 2)
        # The one piece of state that decides what this install can do.
        # Without it, Open still works and the viewer still draws — only
        # detection / fitting / matching are gone — so this is a note,
        # not an error.
        self._backend_note = QLabel("")
        self._backend_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._backend_note.setWordWrap(True)
        self._backend_note.hide()
        outer.addWidget(self._backend_note)
        outer.addStretch(2)

    # -- content -------------------------------------------------------

    def set_theme(self, theme: str) -> None:
        """Swap in the wordmark variant for ``theme``."""
        pixmap = _wordmark(theme)
        if pixmap.isNull():
            self._mark.setText("mlgidGUI")
            return
        self._mark.setPixmap(pixmap.scaledToWidth(
            300, Qt.TransformationMode.SmoothTransformation))

    def set_backend_available(self, available: bool) -> None:
        if available:
            self._backend_note.hide()
            return
        self._backend_note.setText(
            "The analysis backend (mlgidbase) is not installed in this "
            "environment, so detection, fitting and matching are "
            "unavailable. Viewing, conversion and manual peaks work as "
            "usual."
        )
        self._backend_note.setProperty("status", "warn")
        self._backend_note.show()

    def set_recent(self, items: list[dict]) -> None:
        """Fill the shortlist from the persisted recent-files entries.

        Existence is deliberately not checked: this runs on the GUI
        thread, and ``Path.exists()`` over a slow or network share froze
        the window in the menu equivalent. A moved file is handled when
        the user clicks it.
        """
        for button in self._recent_buttons:
            self._recent_box.removeWidget(button)
            button.deleteLater()
        self._recent_buttons.clear()

        shown = [i for i in items if i.get("path")][:RECENT_SHOWN]
        self._recent_title.setVisible(bool(shown))
        for item in shown:
            path, kind = item["path"], item.get("type", "nexus")
            name = Path(path).name
            label = name if kind == "nexus" else f"[raw]  {name}"
            button = QPushButton(label)
            button.setFlat(True)
            button.setToolTip(path)
            button.setStyleSheet("text-align: left;")   # a list row, not a bar
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, p=path, k=kind:
                self.recentRequested.emit(p, k)
            )
            self._recent_box.addWidget(button)
            self._recent_buttons.append(button)
