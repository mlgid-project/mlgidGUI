"""Dialog that picks one experimental technique from the PaNET list."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlgidgui import panet
from mlgidgui.widgets import skin_item_view

_NAV_KEYS = (
    Qt.Key.Key_Up,
    Qt.Key.Key_Down,
    Qt.Key.Key_PageUp,
    Qt.Key.Key_PageDown,
)


class TechniquePickerDialog(QDialog):
    """Search the PaNET techniques and select one.

    The first match is selected while typing, so "GIWAXS" + Enter is
    enough; the arrow keys move through the matches without leaving the
    search field. ``selected()`` holds the choice after ``accept``.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        catalogue: panet.Catalogue | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select technique")
        self.resize(760, 520)
        self._catalogue = catalogue or panet.load_catalogue()
        self._shown: list[panet.Technique] = []

        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText(
            "Search by name or abbreviation, e.g. GIWAXS, reflectivity, XRR"
        )
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._refresh)
        self.search_edit.installEventFilter(self)
        layout.addWidget(self.search_edit)

        self.count_label = QLabel(self)
        self.count_label.setProperty("role", "hint")
        layout.addWidget(self.count_label)

        self.tree = skin_item_view(QTreeWidget(self))
        self.tree.setHeaderLabels(["Technique", "Also known as", "ID"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(1, 200)
        header.resizeSection(2, 110)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        self.tree.itemActivated.connect(lambda *_: self.accept())
        layout.addWidget(self.tree, 1)

        self.detail_label = QLabel(self)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.detail_label.setMinimumHeight(
            3 * self.detail_label.fontMetrics().lineSpacing()
        )
        self.detail_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self.detail_label)

        source = QLabel(
            f"PaNET {self._catalogue.version} "
            f"({self._catalogue.released}), "
            f'<a href="{self._catalogue.license}">CC BY 4.0</a>, '
            '<a href="https://pan-ontologies.github.io/PaNET/index-en.html">'
            "full ontology</a>",
            self,
        )
        source.setProperty("role", "hint")
        source.setOpenExternalLinks(True)
        layout.addWidget(source)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("Select")
        ok.setDefault(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._refresh("")
        self.search_edit.setFocus()

    def selected(self) -> panet.Technique | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        return self._shown[self.tree.indexOfTopLevelItem(item)]

    def accept(self) -> None:
        if self.selected() is not None:
            super().accept()

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.search_edit
            and event.type() == QEvent.Type.KeyPress
            and event.key() in _NAV_KEYS
        ):
            QCoreApplication.sendEvent(self.tree, event)
            return True
        return super().eventFilter(watched, event)

    def _refresh(self, query: str) -> None:
        self._shown = panet.search(self._catalogue.techniques, query)
        self.tree.clear()
        items = []
        for technique in self._shown:
            synonyms = ", ".join(technique.synonyms)
            item = QTreeWidgetItem(
                [technique.name, synonyms, technique.short_id]
            )
            item.setToolTip(0, technique.name)
            item.setToolTip(1, synonyms)
            item.setToolTip(2, technique.pid)
            items.append(item)
        self.tree.addTopLevelItems(items)
        total = len(self._catalogue.techniques)
        self.count_label.setText(
            f"{len(self._shown)} of {total} techniques"
            if self._shown else "No technique matches the search."
        )
        # An empty search lists everything; preselecting the first entry
        # there would let a stray Enter add an arbitrary technique.
        if items and query.strip():
            self.tree.setCurrentItem(items[0])
        self._on_current_changed()

    def _on_current_changed(self, *_: object) -> None:
        technique = self.selected()
        self.buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(technique is not None)
        if technique is None:
            self.detail_label.setText("")
            return
        lines = [technique.pid]
        if technique.definition:
            lines.append(technique.definition)
        self.detail_label.setText("\n".join(lines))
