"""The Structure tab: what the file browser's selection is, and editing it.

The third central tab, beside Image and Data. The File-browser dock
stays the one place a file is navigated; this panel answers the
questions the tree cannot show in a row — the full path, the type and
shape, every attribute, what a link points at, whether the pipeline
depends on this node, and what is wrong with the file as a whole — and
is where those things are changed.

The panel performs no I/O. It renders what ``MainWindow``'s
``StructureMixin`` hands it and emits an intent when the user asks for a
change; the mixin owns the file handle, does the write, and re-renders.
The panel therefore holds no authoritative state, and a failed edit
needs no rollback logic here — the re-render reconciles the display with
whatever the file actually says.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings, QSize, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import icons, nexus_schema, skin
from mlgidlab.flow_layout import FlowLayout
from mlgidlab.h5_edit import Issue, NodeInfo, is_inline_editable
from mlgidlab.structure_tree import StructureTree
from mlgidlab.widgets import (
    GAP,
    PAD,
    PRIMARY,
    Card,
    attach_empty_hint,
    boxed,
    make_form,
    set_variant,
    skin_item_view,
)

import logging

logger = logging.getLogger(__name__)

#: Level -> the skin's semantic state name, for the issue rows and the
#: Check card's header status.
_ISSUE_STATUS = {"error": "error", "warning": "warn", "info": "info"}

#: Shown in the header when nothing is selected.
_EMPTY_HINT = (
    "Select a group or a dataset in the File browser to see what it is."
)

#: Types offered when creating an attribute. NeXus metadata is almost
#: entirely text and numbers, and an attribute created with the wrong
#: type is one delete away from being created again — a longer list
#: would cost more in hesitation than it buys.
ATTR_TYPES: tuple[str, ...] = ("str", "float64", "int64", "bool")

#: Attribute names that get a suggestion list rather than a bare field.
_CLASS_ATTR = "NX_class"
_UNITS_ATTR = "units"

#: Stored on each attribute cell so an edit knows which attribute it
#: belongs to even after the name cell itself has been retyped.
_ATTR_NAME_ROLE = Qt.ItemDataRole.UserRole

#: Where each splitter remembers how the user sized it. A workspace the
#: user has arranged must come back arranged, like the workflow rail's
#: fold and the playback settings.
_SPLIT_KEYS = {
    "main": "structure/split_main",
    "top": "structure/split_top",
    "left": "structure/split_left",
    "bottom": "structure/split_bottom",
}

#: How the workspace divides itself the first time, as relative weights.
#: Without these the splitters divide by size hint, which hands Find as
#: much room as the tree — and the tree is the one you work in.
_SPLIT_DEFAULTS = {
    "main": (3, 1),
    "top": (2, 5),
    "left": (1, 3),
    "bottom": (1, 1),
}

#: The action row's buttons, as ``(verb, label, tooltip)``. The verbs
#: are what ``nodeActionRequested`` carries; ``MainWindow`` maps them to
#: the same handlers the context menu calls, so the row and the menu can
#: never offer different things.
NODE_ACTIONS = (
    ("new_group", "Group…", "Add a group inside the selected group."),
    ("new_field", "Field…", "Add a dataset inside the selected group."),
    ("new_link", "Link…", "Add a soft or external link."),
)


def _human_bytes(count: int) -> str:
    """``4600`` -> ``4.6 kB``. Decimal units, matching what h5 tools print."""
    value = float(count)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"  # pragma: no cover - unreachable, kept explicit


def describe_node(info: NodeInfo) -> str:
    """The one-line type summary under the path.

    Everything a person wants before deciding whether to touch a node:
    what it is, what it holds, how big it is, and how it is stored.
    """
    if info.kind == "missing":
        return "not in this file"
    if info.kind == "link":
        return info.link.describe()
    if info.kind == "group":
        children = info.n_children or 0
        parts = [
            info.nx_class or "group",
            f"{children} child" if children == 1 else f"{children} children",
        ]
        if info.link.kind != "hard":
            parts.append(info.link.describe())
        return " · ".join(parts)

    shape = " × ".join(str(n) for n in info.shape) if info.shape else "scalar"
    parts = ["compound" if info.is_compound else info.dtype, shape]
    if info.nbytes:
        parts.append(_human_bytes(info.nbytes))
    if info.chunks:
        parts.append("chunked")
    if info.compression:
        parts.append(str(info.compression))
    if info.maxshape and any(n is None for n in info.maxshape):
        parts.append("resizable")
    if info.link.kind != "hard":
        parts.append(info.link.describe())
    return " · ".join(parts)


def suggestions_for(name: str) -> tuple[tuple[str, ...], str]:
    """``(completions, type)`` worth offering for an attribute called ``name``.

    This is the NeXus-aware half of the form: ``NX_class`` offers the
    base classes instead of a blank box, and anything whose name implies
    a physical quantity offers the units that quantity is written in.
    Both stay editable — the lists are a leg-up, not a constraint.
    """
    if name == _CLASS_ATTR:
        return nexus_schema.NX_CLASSES, "str"
    if name == _UNITS_ATTR:
        return nexus_schema.GENERIC_UNITS, "str"
    return (), ""


class NewAttributeDialog(QDialog):
    """Name, type and value for an attribute being created."""

    def __init__(self, parent: QWidget | None = None, *, node_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("New attribute")
        self._node_name = node_name

        form = make_form(self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("units, NX_class, long_name, …")
        self.name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(ATTR_TYPES)
        form.addRow("Type:", self.type_combo)

        # Editable combo rather than a line edit: for the names NeXus has
        # an opinion about it fills with real values, and for everything
        # else it behaves exactly like a text field.
        self.value_combo = QComboBox(self)
        self.value_combo.setEditable(True)
        form.addRow("Value:", self.value_combo)

        self.hint = QLabel("", self)
        self.hint.setProperty("status", "muted")
        self.hint.setWordWrap(True)
        form.addRow("", self.hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._on_name_changed("")

    def _on_name_changed(self, name: str) -> None:
        completions, forced_type = suggestions_for(name.strip())
        if not completions and name.strip():
            completions = nexus_schema.suggest_units(name.strip())
            # Only offer units when the name actually implies a quantity;
            # the generic fallback would put "counts" under "comment".
            if completions is nexus_schema.GENERIC_UNITS:
                completions = ()
        current = self.value_combo.currentText()
        self.value_combo.clear()
        if completions:
            self.value_combo.addItems([c for c in completions if c])
        self.value_combo.setCurrentText(current)
        if forced_type:
            self.type_combo.setCurrentText(forced_type)
        self.hint.setText(nexus_schema.class_help(self.value_combo.currentText())
                          if name.strip() == _CLASS_ATTR else "")

    def result_values(self) -> tuple[str, str, str]:
        """``(name, type, value_text)`` as typed."""
        return (
            self.name_edit.text().strip(),
            self.type_combo.currentText(),
            self.value_combo.currentText(),
        )


class _NameEdit(QLineEdit):
    """The node's own name, editable where it is printed.

    Renaming already has two routes (the context menu and the action
    row), and both open a dialog. This is the third and the shortest:
    the name is right there at the head of the panel, so typing over it
    is the obvious gesture. It emits, and the window does the work — the
    same commit, the same undo entry, the same protection prompt.

    Read-only it *is* the title: no frame, no ground, the card-title
    weight, so the header reads as one line of "where am I" rather than
    as a form. Editing gives it a real box, because at that point it is
    a field and has to look like one.

    Double-click, not single, exactly as the attribute table decided: a
    single click is how you put the caret somewhere to read a long name,
    and an accidental one-click edit on a file's structure is a bad
    surprise.

    Losing focus reverts. A rename that happened because the user
    clicked somewhere else is a change to their file they did not ask
    for; forgetting to press Enter costs them only the typing.
    """

    #: The user committed a new name for the node on show.
    committed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(skin.NODE_NAME)
        self._name = ""
        self._can_edit = False
        self._settle()
        self.returnPressed.connect(self._commit)

    # -- state -------------------------------------------------------------

    @property
    def editing(self) -> bool:
        return not self.isReadOnly()

    def set_name(self, name: str) -> None:
        """Show ``name``. Any edit in flight belonged to the old node."""
        self._name = name
        self._settle()

    def set_editable(self, can_edit: bool) -> None:
        self._can_edit = bool(can_edit)
        if not can_edit:
            self._settle()
        self.setToolTip(
            "Double-click or F2 to rename" if can_edit else "")

    def begin(self) -> None:
        if not self._can_edit or self.editing:
            return
        self.setReadOnly(False)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self._restyle()
        self.selectAll()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _settle(self) -> None:
        """Back to being a title, showing the name the file has."""
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setText(self._name)
        self.deselect()
        self.home(False)
        self._restyle()
        self.updateGeometry()

    def _restyle(self) -> None:
        # ``:read-only`` is a QSS pseudo-state, and Qt does not re-run
        # the sheet when the flag flips on its own.
        style = self.style()
        style.unpolish(self)
        style.polish(self)

    def _commit(self) -> None:
        if not self.editing:
            return
        typed = self.text().strip()
        old = self._name
        # Settle first, so a rejected or failed rename leaves the header
        # showing what the file still says rather than what was typed.
        self._settle()
        if typed and typed != old:
            self.committed.emit(typed)

    # -- Qt ----------------------------------------------------------------

    def sizeHint(self) -> QSize:
        """As wide as the name, not as wide as a line edit.

        A QLineEdit asks for room for about seventeen characters. Here it
        sits inline after the path, so it has to ask for the width of the
        text it is actually showing.
        """
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance(self.text() or " ")
        return QSize(width + 14, hint.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def mouseDoubleClickEvent(self, event) -> None:
        if not self.editing:
            self.begin()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F2 and not self.editing:
            self.begin()
            return
        if event.key() == Qt.Key.Key_Escape and self.editing:
            self._settle()
            return
        if not self.editing and (
            event.matches(QKeySequence.StandardKey.Undo)
            or event.matches(QKeySequence.StandardKey.Redo)
        ):
            # A QLineEdit has an undo stack of its own, and this one sits
            # in a tab where Ctrl+Z means "undo my last edit to the
            # file". Sitting on the key while showing a title nobody is
            # typing into would be a quiet, infuriating failure, so it is
            # passed up to the window. While the field IS being edited it
            # keeps the key: there the user's own typing is what they
            # mean to take back, and nothing has been written yet.
            event.ignore()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        if self.editing:
            self._settle()
        super().focusOutEvent(event)


class StructurePanel(QWidget):
    """Read-out and editor for one HDF5 node, plus the file's health."""

    #: The user asked for the file to be re-checked.
    recheckRequested = Signal()
    #: ``(attribute_name, typed_text)`` — set this attribute's value.
    attributeEdited = Signal(str, str)
    #: ``(old_name, new_name)``
    attributeRenamed = Signal(str, str)
    #: ``(name, type, typed_text)`` — create this attribute.
    attributeAdded = Signal(str, str, str)
    #: ``(name,)`` — remove this attribute.
    attributeRemoved = Signal(str)
    #: The typed text for a scalar dataset's value.
    scalarValueEdited = Signal(str)
    #: Put the changes list on the clipboard.
    copyChangesRequested = Signal()
    #: Point this link somewhere else.
    retargetLinkRequested = Signal()
    #: Resolve this link and describe what it points at.
    followLinkRequested = Signal()
    #: Open the value grid for this dataset.
    editValuesRequested = Signal()
    #: ``(query, case_sensitive)`` — find nodes matching this.
    searchRequested = Signal(str, bool)
    #: Go to this in-file path.
    searchResultActivated = Signal(str)
    #: One of ``NODE_ACTIONS``' verbs, or copy/cut/paste/rename/delete —
    #: the action row asking for what the context menu also offers.
    nodeActionRequested = Signal(str)
    #: A new name typed over the one in the header.
    renameRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info: NodeInfo | None = None
        self._editable = False
        # Guards the itemChanged signal while the table is being filled,
        # so populating a row never reads as a user edit.
        self._populating = False
        self._splits: dict[str, QSplitter] = {}
        self._splits_applied = False

        # Five regions, no page-level scrolling. The tab used to be one
        # tall scrolling column, which meant that reading the file's
        # health while editing an attribute took a scroll, and that
        # nothing was where it had been left. Splitters instead: each
        # region scrolls inside itself, which is what a table is for,
        # and the divisions are the user's to set and are remembered.
        #
        # Left is where you are (find, tree), right is the node you
        # picked, and the band below is about the file as a whole rather
        # than any one node.
        # Each region goes in a box of its own (``boxed``). Nested boxes
        # would be clutter in a dock, but a pane with no edge leaves its
        # card's title hairline ending in mid-air.
        self._split_left = self._splitter(Qt.Orientation.Vertical, "left")
        self._split_left.addWidget(
            boxed(self._build_search_card(self._split_left)))
        self._split_left.addWidget(
            boxed(self._build_tree_card(self._split_left)))
        self._split_left.setStretchFactor(0, 0)
        self._split_left.setStretchFactor(1, 1)

        # One box for the whole node column: the path, its attributes,
        # its value and its link are four readings of one thing, and
        # four boxes would say they were four things.
        node_column = QWidget(self)
        column = QVBoxLayout(node_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(PAD * 2)
        column.addLayout(self._build_header(node_column))
        # The attributes table is the one thing here that grows: a node
        # has a handful of them or a hundred.
        column.addWidget(self._build_attributes_card(node_column), 1)
        self._value_card = self._build_value_card(node_column)
        column.addWidget(self._value_card)
        self._link_card = self._build_link_card(node_column)
        column.addWidget(self._link_card)

        self._split_top = self._splitter(Qt.Orientation.Horizontal, "top")
        self._split_top.addWidget(self._split_left)
        self._split_top.addWidget(boxed(node_column))
        self._split_top.setStretchFactor(0, 1)
        self._split_top.setStretchFactor(1, 3)

        self._split_bottom = self._splitter(Qt.Orientation.Horizontal, "bottom")
        self._split_bottom.addWidget(
            boxed(self._build_check_card(self._split_bottom)))
        self._split_bottom.addWidget(
            boxed(self._build_changes_card(self._split_bottom)))

        self._split_main = self._splitter(Qt.Orientation.Vertical, "main")
        self._split_main.addWidget(self._split_top)
        self._split_main.addWidget(self._split_bottom)
        self._split_main.setStretchFactor(0, 3)
        self._split_main.setStretchFactor(1, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(PAD * 2, PAD * 2, PAD * 2, PAD * 2)
        outer.addWidget(self._split_main)

        self._restore_splits()
        self.clear()

    # -- the workspace -----------------------------------------------------

    def _splitter(self, orientation, key: str) -> QSplitter:
        """A splitter that remembers where the user put it.

        Children cannot be collapsed to nothing: the whole point of this
        layout is that every region is on screen, and a handle dragged
        one pixel too far must not make a section disappear with no
        obvious way back.
        """
        split = QSplitter(orientation, self)
        split.setObjectName(f"Structure_{key}")
        split.setChildrenCollapsible(False)
        split.splitterMoved.connect(
            lambda *_: self._save_split(key, split))
        self._splits[key] = split
        return split

    def _save_split(self, key: str, split: QSplitter) -> None:
        try:
            QSettings().setValue(
                _SPLIT_KEYS[key], ",".join(str(n) for n in split.sizes()))
        except Exception:  # pragma: no cover - a settings backend failure
            logger.debug("saving the %s splitter failed", key, exc_info=True)

    def _restore_splits(self) -> None:
        """Put the saved divisions back, ignoring anything malformed.

        A stored size list from a different window size is still useful
        — Qt scales it — but a corrupt or stale-arity one is not, and it
        must not stop the tab from building.
        """
        for key, split in self._splits.items():
            sizes = self._stored_sizes(key, split.count())
            if sizes is None:
                sizes = self._default_sizes(key, split)
            split.setSizes(sizes)

    @staticmethod
    def _default_sizes(key: str, split: QSplitter) -> list[int]:
        """The first-run division, as a share of the space there is.

        In real pixels rather than bare weights because a splitter obeys
        its children's size *hints* until it is told otherwise, and the
        Find card's hint is as tall as the tree's — which would hand
        half the left column to a search box nobody has typed in yet.
        """
        weights = _SPLIT_DEFAULTS[key]
        extent = (split.height() if split.orientation() == Qt.Orientation.Vertical
                  else split.width())
        # Before the first show there is no geometry to divide; the
        # ratio is still right, and showEvent re-applies it once there
        # is something real to divide.
        extent = max(int(extent), 400)
        total = sum(weights)
        return [max(1, extent * w // total) for w in weights]

    def showEvent(self, event) -> None:  # type: ignore[override]
        """Size the workspace once, when there is a real window to size it in.

        ``__init__`` runs before the tab has any geometry, so a division
        applied there is a division of nothing.
        """
        super().showEvent(event)
        if not self._splits_applied:
            self._splits_applied = True
            self._restore_splits()

    @staticmethod
    def _stored_sizes(key: str, count: int) -> list[int] | None:
        """The saved division for one splitter, or None if unusable.

        A stored list from a different window size is still worth having
        — Qt scales it — but one with the wrong number of entries came
        from a different layout and would silently misplace a section.
        """
        raw = str(QSettings().value(_SPLIT_KEYS[key], "") or "")
        if not raw:
            return None
        try:
            sizes = [int(n) for n in raw.split(",") if n]
        except ValueError:
            logger.debug("ignoring a malformed %s splitter state", key)
            return None
        if len(sizes) != count or any(n < 0 for n in sizes) or not any(sizes):
            return None
        return sizes

    # -- construction ------------------------------------------------------

    def _build_tree_card(self, parent: QWidget) -> Card:
        """The tab's own tree of every open file.

        It fills itself through a lister that ``MainWindow`` installs,
        so this panel keeps its no-I/O rule: the tree asks, the window
        reads. See ``structure_tree`` for why it is not a silx tree.
        """
        card = Card("Tree", parent=parent)
        self._tree_card = card

        # The same actions the right-click menu carries. Both exist
        # because the menu is faster once you know it is there and the
        # row is the only way to find out that it is.
        #
        # A flow layout, not a QHBoxLayout: a row of five buttons in a
        # box layout sets a hard floor under how narrow the tree column
        # can be dragged, and this column is meant to be narrow. It
        # wraps onto a second line instead — the same reason the image
        # viewer's control strip uses one.
        row = FlowLayout(hspacing=GAP, vspacing=4)
        self._new_btn = QToolButton(card)
        self._new_btn.setText("New")
        self._new_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._new_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self._new_btn.setToolTip(
            "Add a group, a field or a link inside the selected group.")
        new_menu = QMenu(self._new_btn)
        for verb, label, tip in NODE_ACTIONS:
            action = new_menu.addAction(label)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, v=verb: self.nodeActionRequested.emit(v))
        self._new_btn.setMenu(new_menu)
        row.addWidget(self._new_btn)


        self._node_buttons: dict[str, QToolButton] = {}
        for verb, label, tip in (
            ("copy", "Copy", "Copy this node (Ctrl+C). Paste it anywhere, "
                             "including into another open file."),
            ("paste", "Paste", "Paste into the selected group (Ctrl+V)."),
            ("rename", "Rename", "Rename this node."),
            ("delete", "Delete", "Delete this node. Ctrl+Z brings it back."),
        ):
            button = QToolButton(card)
            button.setText(label)
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setAutoRaise(True)
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _checked=False, v=verb: self.nodeActionRequested.emit(v))
            self._node_buttons[verb] = button
            row.addWidget(button)
        card.body_layout.addLayout(row)

        self.node_tree = StructureTree(card)
        self.node_tree.setMinimumWidth(180)
        self.node_tree.setMinimumHeight(80)
        self.node_tree.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        skin_item_view(self.node_tree)
        attach_empty_hint(self.node_tree, "No file open.")
        self.node_tree.setToolTip(
            "Every file you have open. A group can be copied from one and "
            "pasted into another; an external link is only resolved if you "
            "expand it yourself."
        )
        card.body_layout.addWidget(self.node_tree)
        return card

    def _build_search_card(self, parent: QWidget) -> Card:
        """Find a node by name, path, value or attribute.

        Deliberately a jump list rather than a filter on the browser
        tree: filtering silx's model would ask it to resolve rows to
        answer the query, and on a master of external links that is the
        one thing this tab never does.
        """
        card = Card("Find", parent=parent)
        self._search_card = card
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._search_edit = QLineEdit(card)
        self._search_edit.setPlaceholderText(
            "name, value, or a path — e.g. wavelength, sample/material")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.returnPressed.connect(self._on_search)
        row.addWidget(self._search_edit, 1)
        # "Aa" rather than a word: this is the narrowest region of the
        # tab, and the toggle has to leave the field itself usable.
        self._case_btn = QToolButton(card)
        self._case_btn.setText("Aa")
        self._case_btn.setCheckable(True)
        self._case_btn.setProperty("variant", "toggle")
        self._case_btn.setToolTip(
            "Match upper and lower case exactly.\n"
            "Off by default: searching a file someone else wrote means "
            "not knowing how they capitalised things."
        )
        # Flipping it re-runs whatever is in the box, so the toggle
        # answers a question rather than only arming the next search.
        self._case_btn.toggled.connect(lambda _: self._on_search())
        row.addWidget(self._case_btn)
        self._search_btn = QPushButton("Find", card)
        self._search_btn.clicked.connect(self._on_search)
        row.addWidget(self._search_btn)
        card.body_layout.addLayout(row)

        self._search_list = QListWidget(card)
        skin_item_view(self._search_list)
        self._search_list.setMinimumHeight(56)
        self._search_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._search_list.itemActivated.connect(self._on_search_activated)
        self._search_list.itemClicked.connect(self._on_search_activated)
        # Short on purpose: this box is the smallest region of the tab,
        # and a hint that does not fit is worse than no hint. The long
        # version of why links are never followed lives in the tooltip.
        attach_empty_hint(self._search_list, "Nothing searched yet.")
        self._search_list.setToolTip(
            "Searches names, small dataset values, attribute names and "
            "attribute values.\n\n"
            "A query with a slash is a place and then a term: "
            "sample/material finds a match for 'material' under a group "
            "matching 'sample'. The parts do not have to be adjacent.\n\n"
            "Links are never followed, so a master of external scans "
            "answers as fast as any other file."
        )
        card.body_layout.addWidget(self._search_list)
        return card

    def _on_search(self) -> None:
        text = self._search_edit.text().strip()
        if text:
            self.searchRequested.emit(text, self._case_btn.isChecked())

    def case_sensitive(self) -> bool:
        """Whether the Aa toggle is on, for the window and the tests."""
        return self._case_btn.isChecked()

    def _on_search_activated(self, item) -> None:
        path = item.data(_ATTR_NAME_ROLE)
        if path:
            self.searchResultActivated.emit(str(path))

    def set_search_results(self, hits, truncated: bool = False) -> None:
        """Fill the results list. ``hits`` are ``h5_edit.SearchHit``."""
        self._search_list.clear()
        from PySide6.QtWidgets import QListWidgetItem

        for hit in hits:
            label = (hit.path if hit.where == "name"
                     else f"{hit.path}  —  {hit.detail}")
            item = QListWidgetItem(label)
            item.setData(_ATTR_NAME_ROLE, hit.path)
            item.setToolTip(f"{hit.path}\nmatched on {hit.where}")
            self._search_list.addItem(item)
        if truncated:
            self._search_card.set_status(
                f"{len(hits)}, stopped early", "warn")
        else:
            self._search_card.set_status(
                f"{len(hits)}" if hits else "no match",
                "muted" if hits else "warn")

    def search_rows(self) -> list[str]:
        return [
            self._search_list.item(r).text()
            for r in range(self._search_list.count())
        ]

    def _build_header(self, parent: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(0)
        # The header is split where the file is: everything down to the
        # last "/" is where the node lives and is read-only; the last
        # component is the node's own name and can be typed over. The
        # split is what keeps this a rename — a single field holding the
        # whole path would be inviting a move, which is a different
        # operation with a different undo entry.
        self._path_label = QLabel(parent)
        self._path_label.setProperty("role", skin.NODE_PATH)
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        # The path is the one thing worth copying out of this panel, so
        # it stays selectable and wraps rather than eliding into a "…"
        # the user cannot recover.
        self._path_label.setWordWrap(True)
        top.addWidget(self._path_label, 0, Qt.AlignmentFlag.AlignBottom)

        self._name_edit = _NameEdit(parent)
        self._name_edit.committed.connect(self.renameRequested)
        # Bottom-aligned so that when a long path wraps, the name still
        # sits on the last line of it rather than floating beside the
        # middle of the block.
        top.addWidget(self._name_edit, 0, Qt.AlignmentFlag.AlignBottom)
        top.addStretch(1)

        self._badge = QLabel(parent)
        self._badge.setProperty("status", "warn")
        self._badge.hide()
        top.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)

        self._type_label = QLabel(parent)
        self._type_label.setProperty("status", "muted")
        self._type_label.setWordWrap(True)
        box.addWidget(self._type_label)

        self._note_label = QLabel(parent)
        self._note_label.setProperty("status", "muted")
        self._note_label.setWordWrap(True)
        self._note_label.hide()
        box.addWidget(self._note_label)
        return box

    def _build_attributes_card(self, parent: QWidget) -> Card:
        card = Card("Attributes", parent=parent)
        self._attr_card = card
        self._attr_table = QTableWidget(0, 2, card)
        skin_item_view(self._attr_table)
        self._attr_table.setHorizontalHeaderLabels(["Name", "Value"])
        self._attr_table.verticalHeader().setVisible(False)
        self._attr_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        # Double-click to edit, not single: a single click is how the
        # user selects a row to remove, and an accidental one-click edit
        # on a file's metadata is a bad surprise.
        self._attr_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = self._attr_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._attr_table.setMinimumHeight(56)
        self._attr_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._attr_table.itemChanged.connect(self._on_attr_item_changed)
        attach_empty_hint(self._attr_table, "No attributes on this node.")
        card.body_layout.addWidget(self._attr_table)

        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._add_attr_btn = QPushButton("+ Attribute", card)
        self._add_attr_btn.setToolTip(
            "Add an attribute to this node. NX_class and units come with "
            "suggestions."
        )
        self._add_attr_btn.clicked.connect(self._on_add_attribute)
        row.addWidget(self._add_attr_btn)
        self._remove_attr_btn = QPushButton("Remove", card)
        self._remove_attr_btn.setToolTip(
            "Remove the selected attribute (Delete).")
        self._remove_attr_btn.clicked.connect(self._on_remove_attribute)
        row.addWidget(self._remove_attr_btn)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        return card

    def _build_value_card(self, parent: QWidget) -> Card:
        card = Card("Value", parent=parent)
        self._value_label = QLabel(card)
        self._value_label.setWordWrap(True)
        self._value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        card.body_layout.addWidget(self._value_label)

        self._value_row = QWidget(card)
        row = QHBoxLayout(self._value_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(GAP)
        self._value_edit = QLineEdit(self._value_row)
        self._value_edit.setPlaceholderText("value")
        self._value_edit.returnPressed.connect(self._on_set_value)
        row.addWidget(self._value_edit, 1)
        self._value_set_btn = set_variant(
            QPushButton("Set", self._value_row), PRIMARY)
        self._value_set_btn.clicked.connect(self._on_set_value)
        row.addWidget(self._value_set_btn)
        card.body_layout.addWidget(self._value_row)

        grid_row = QHBoxLayout()
        grid_row.setSpacing(GAP)
        self._edit_values_btn = QPushButton("Edit values…", card)
        self._edit_values_btn.setToolTip(
            "Open this dataset in a grid. Compound tables get one column "
            "per field, with row insert and delete."
        )
        self._edit_values_btn.clicked.connect(self.editValuesRequested)
        grid_row.addWidget(self._edit_values_btn)
        grid_row.addStretch(1)
        card.body_layout.addLayout(grid_row)
        # No scrolling view inside, so it takes its natural height and
        # leaves the rest of the column to the attributes table.
        card.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum)
        return card

    def _build_link_card(self, parent: QWidget) -> Card:
        card = Card("Link", parent=parent)
        self._link_label = QLabel(card)
        self._link_label.setWordWrap(True)
        self._link_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        card.body_layout.addWidget(self._link_label)

        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._follow_btn = QPushButton("Follow", card)
        self._follow_btn.setToolTip(
            "Open what this link points at and describe it here. Nothing "
            "is resolved until you ask, which is what keeps a master of "
            "external links quick to browse."
        )
        self._follow_btn.clicked.connect(self.followLinkRequested)
        row.addWidget(self._follow_btn)
        self._retarget_btn = QPushButton("Retarget…", card)
        self._retarget_btn.setToolTip("Point this link at something else.")
        self._retarget_btn.clicked.connect(self.retargetLinkRequested)
        row.addWidget(self._retarget_btn)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        card.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum)
        return card

    def _build_check_card(self, parent: QWidget) -> Card:
        card = Card("Check", parent=parent)
        self._check_card = card
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._recheck_btn = QToolButton(card)
        self._recheck_btn.setText("Re-check")
        self._recheck_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons.bind(self._recheck_btn, "refresh")
        self._recheck_btn.setAutoRaise(True)
        self._recheck_btn.setToolTip(
            "Re-read the file's top-level layout and report anything that "
            "would stop the viewer or the pipeline from opening it."
        )
        self._recheck_btn.clicked.connect(self.recheckRequested)
        row.addWidget(self._recheck_btn)
        row.addStretch(1)
        card.body_layout.addLayout(row)

        self._issue_table = QTableWidget(0, 2, card)
        skin_item_view(self._issue_table)
        self._issue_table.setHorizontalHeaderLabels(["Where", "What"])
        self._issue_table.verticalHeader().setVisible(False)
        self._issue_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._issue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        issue_header = self._issue_table.horizontalHeader()
        issue_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        issue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._issue_table.setMinimumHeight(56)
        self._issue_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        attach_empty_hint(
            self._issue_table,
            "Nothing to report — the viewer and the pipeline can read "
            "this file's layout.",
        )
        card.body_layout.addWidget(self._issue_table)
        return card

    def _build_changes_card(self, parent: QWidget) -> Card:
        card = Card("Changes this session", parent=parent)
        self._changes_card = card
        self._changes_list = QListWidget(card)
        skin_item_view(self._changes_list)
        self._changes_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._changes_list.setMinimumHeight(56)
        self._changes_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        attach_empty_hint(
            self._changes_list,
            "Nothing changed yet. Edits land in the working copy; the "
            "file on disk is untouched until you save.",
        )
        card.body_layout.addWidget(self._changes_list)

        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._copy_changes_btn = QToolButton(card)
        self._copy_changes_btn.setText("Copy as text")
        self._copy_changes_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._copy_changes_btn.setAutoRaise(True)
        self._copy_changes_btn.setToolTip(
            "Put this list on the clipboard — a record of what was done "
            "to the file before saving it."
        )
        self._copy_changes_btn.clicked.connect(self.copyChangesRequested)
        row.addWidget(self._copy_changes_btn)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        return card

    # -- population --------------------------------------------------------

    def clear(self, message: str = "") -> None:
        """Empty every section. ``message`` replaces the default hint."""
        self._info = None
        self._path_label.setText(message or _EMPTY_HINT)
        self._show_name("")
        self._type_label.setText("")
        self._badge.hide()
        self._note_label.hide()
        self._populating = True
        try:
            self._attr_table.setRowCount(0)
        finally:
            self._populating = False
        self._attr_card.set_status("")
        self._value_card.hide()
        self._link_card.hide()
        self._apply_editable()

    def set_editable(self, editable: bool) -> None:
        """Turn every edit control on or off.

        Off for raw sessions, for a file with no session behind it, and
        whenever there is no node selected.
        """
        self._editable = bool(editable)
        self._apply_editable()

    def _show_name(self, name: str) -> None:
        """Put the node's own name in the header, or take it away."""
        self._name_edit.set_name(name)
        self._name_edit.setVisible(bool(name))

    def _apply_editable(self) -> None:
        has_node = self._info is not None
        can_edit = self._editable and has_node and self._info.kind != "missing"
        self._add_attr_btn.setEnabled(can_edit)
        self._remove_attr_btn.setEnabled(can_edit)
        self._attr_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            if can_edit else QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._value_edit.setEnabled(can_edit)
        self._value_set_btn.setEnabled(can_edit)
        # A hard link is the object itself, so there is nothing to point
        # elsewhere; only soft and external links can be retargeted.
        self._retarget_btn.setEnabled(
            can_edit and has_node and self._info.link.kind in ("soft", "external")
        )
        # The action row answers to the same rules the context menu
        # applies: you can only add inside a group, and the file root is
        # not a node you can rename or delete.
        is_group = has_node and self._info.kind == "group"
        is_root = has_node and self._info.path == "/"
        # The header's name follows the Rename button exactly: same
        # operation, same rules, so they can never disagree about
        # whether this node can be renamed.
        self._name_edit.set_editable(can_edit and not is_root)
        self._new_btn.setEnabled(can_edit and is_group)
        for verb, button in self._node_buttons.items():
            if verb in ("rename", "delete"):
                button.setEnabled(can_edit and not is_root)
            elif verb == "paste":
                button.setEnabled(can_edit and is_group)
            else:
                button.setEnabled(can_edit and not is_root)

    def set_note(self, text: str, level: str = "muted") -> None:
        """A line under the type summary — why editing is off, mostly."""
        self._note_label.setText(text)
        self._note_label.setProperty("status", level)
        style = self._note_label.style()
        style.unpolish(self._note_label)
        style.polish(self._note_label)
        self._note_label.setVisible(bool(text))

    def set_node(
        self,
        info: NodeInfo,
        attrs: dict[str, Any],
        preview: str = "",
        *,
        file_label: str = "",
    ) -> None:
        """Show one node. ``file_label`` prefixes the path when set."""
        self._info = info
        path = info.path
        if path == "/":
            # The file root has no name of its own to rename.
            self._path_label.setText(
                f"{file_label}::/" if file_label else "/")
            self._show_name("")
        else:
            parent_path, name = path.rsplit("/", 1)
            head = f"{parent_path}/" if parent_path else "/"
            self._path_label.setText(
                f"{file_label}::{head}" if file_label else head)
            self._show_name(name)
        self._type_label.setText(describe_node(info))

        if info.protection:
            self._badge.setText(f"⛨  {info.protection}")
            self._badge.setToolTip(
                "The viewer or the pipeline reads this node. Editing it is "
                "allowed; anything destructive will ask first."
            )
            self._badge.show()
        else:
            self._badge.hide()

        self._fill_attributes(attrs)

        if info.kind == "dataset":
            self._value_label.setText(preview or "(empty)")
            # One cell is one field to type into. Anything larger is the
            # grid's job (and a 6000 x 2000 image is nobody's job in a
            # single line of text).
            scalar = info.n_cells == 1 and not info.is_compound
            self._value_row.setVisible(scalar)
            if scalar:
                self._value_edit.setText(preview)
            # One cell is a field to type into; anything else is a grid.
            self._edit_values_btn.setVisible(not scalar)
            self._value_card.show()
        else:
            self._value_card.hide()

        if info.link.kind != "hard":
            self._link_label.setText(info.link.describe())
            # A followed link shows its target's type on the line above,
            # so the Follow button has nothing left to do.
            self._follow_btn.setVisible(info.kind == "link")
            self._link_card.show()
        else:
            self._link_card.hide()
        self._apply_editable()

    def _fill_attributes(self, attrs: dict[str, Any]) -> None:
        from mlgidlab.h5_edit import attr_type_label, format_value

        self._populating = True
        try:
            self._attr_table.setRowCount(0)
            for name, value in attrs.items():
                row = self._attr_table.rowCount()
                self._attr_table.insertRow(row)
                name_item = QTableWidgetItem(str(name))
                name_item.setData(_ATTR_NAME_ROLE, str(name))
                name_item.setToolTip(f"type: {attr_type_label(value)}")
                self._attr_table.setItem(row, 0, name_item)

                text = format_value(value)
                value_item = QTableWidgetItem(text)
                value_item.setData(_ATTR_NAME_ROLE, str(name))
                value_item.setToolTip(format_value(value, limit=2000))
                if not is_inline_editable(value):
                    # A 2-D attribute has no one-line spelling. Showing it
                    # read-only is honest; inventing a syntax is not.
                    value_item.setFlags(
                        value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    value_item.setToolTip(
                        "Multi-dimensional attribute — shown read-only.")
                self._attr_table.setItem(row, 1, value_item)
        finally:
            self._populating = False
        count = self._attr_table.rowCount()
        self._attr_card.set_status(f"{count}" if count else "")

    def set_issues(self, issues: list[Issue]) -> None:
        """Fill the Check card. An empty list reads as a clean bill."""
        self._issue_table.setRowCount(0)
        worst = ""
        for issue in issues:
            row = self._issue_table.rowCount()
            self._issue_table.insertRow(row)
            self._issue_table.setItem(row, 0, QTableWidgetItem(issue.path))
            text = issue.message
            if issue.fix:
                text = f"{text}  →  {issue.fix}"
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            self._issue_table.setItem(row, 1, item)
            if issue.level == "error":
                worst = "error"
            elif issue.level == "warning" and worst != "error":
                worst = "warning"
            elif not worst:
                worst = "info"
        if not issues:
            self._check_card.set_status("clean", "ok")
            return
        counts = {
            level: sum(1 for i in issues if i.level == level)
            for level in ("error", "warning", "info")
        }
        summary = ", ".join(
            f"{n} {name}" for name, n in (
                ("error", counts["error"]),
                ("warning", counts["warning"]),
                ("note", counts["info"]),
            ) if n
        )
        self._check_card.set_status(summary, _ISSUE_STATUS.get(worst, "muted"))

    def set_changes(self, entries: list[str]) -> None:
        """Fill the changes list, newest at the bottom."""
        self._changes_list.clear()
        self._changes_list.addItems(entries)
        self._changes_card.set_status(
            f"{len(entries)}" if entries else "", "muted")
        self._copy_changes_btn.setEnabled(bool(entries))
        if entries:
            self._changes_list.scrollToBottom()

    # -- user intent -------------------------------------------------------

    def _on_attr_item_changed(self, item: QTableWidgetItem) -> None:
        """A cell was typed into: rename on the name column, set on the value.

        The attribute's original name travels on the item itself, so a
        rename knows what it is renaming even though the cell now reads
        as the new name.
        """
        if self._populating or not self._editable:
            return
        original = item.data(_ATTR_NAME_ROLE)
        if not original:
            return
        text = item.text().strip()
        if item.column() == 0:
            if text and text != original:
                self.attributeRenamed.emit(str(original), text)
        else:
            self.attributeEdited.emit(str(original), item.text())

    def _on_add_attribute(self) -> None:
        node = self._info.path if self._info is not None else ""
        dialog = NewAttributeDialog(self, node_name=node)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        # Parented for its style, so nothing else frees it.
        dialog.deleteLater()
        if not accepted:
            return
        name, type_name, value = dialog.result_values()
        if name:
            self.attributeAdded.emit(name, type_name, value)

    def _on_remove_attribute(self) -> None:
        name = self.selected_attribute()
        if name:
            self.attributeRemoved.emit(name)

    def _on_set_value(self) -> None:
        if self._editable:
            self.scalarValueEdited.emit(self._value_edit.text())

    def selected_attribute(self) -> str | None:
        """The attribute name of the selected row, if any."""
        items = self._attr_table.selectedItems()
        if not items:
            return None
        value = items[0].data(_ATTR_NAME_ROLE)
        return str(value) if value else None

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Delete removes the selected attribute while the table has focus.

        Scoped to the table's focus so it never collides with the image
        viewer's Delete = "remove the selected peak".
        """
        if (
            event.key() == Qt.Key.Key_Delete
            and self._editable
            and self._attr_table.hasFocus()
            and self.selected_attribute()
        ):
            self._on_remove_attribute()
            event.accept()
            return
        super().keyPressEvent(event)

    # -- read-back, for the tests and for later stages ---------------------

    @property
    def current(self) -> NodeInfo | None:
        return self._info

    @property
    def editable(self) -> bool:
        return self._editable

    def attribute_rows(self) -> list[tuple[str, str]]:
        return [
            (self._attr_table.item(r, 0).text(), self._attr_table.item(r, 1).text())
            for r in range(self._attr_table.rowCount())
        ]

    def issue_rows(self) -> list[tuple[str, str]]:
        return [
            (self._issue_table.item(r, 0).text(),
             self._issue_table.item(r, 1).text())
            for r in range(self._issue_table.rowCount())
        ]

    def change_rows(self) -> list[str]:
        return [
            self._changes_list.item(r).text()
            for r in range(self._changes_list.count())
        ]
