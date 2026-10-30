"""The Structure tab's own file tree.

The tab used to borrow the File-browser dock as its navigator. That put
navigation on one side of the window and editing in the middle, and it
meant the tab could not be used with the dock folded away. This is the
tab's own tree, so the editor is one self-contained surface.

**It holds nothing.** Every row stores two strings — the file it belongs
to and the path inside it — and no h5py object, no group, no file
handle. That is not tidiness, it is the constraint the whole feature
rests on: HDF5 refuses to open a file ``r+`` while any read handle on it
is open, so a tree that kept handles the way silx's does would have to
be torn down and rebuilt every time the editor took its write handle.
This one is never in the way, and it can refresh a single node's
children in place instead of being rebuilt whole.

Reading is therefore somebody else's job. ``set_lister`` installs a
callable — ``MainWindow``'s, which goes through the already-open write
handle when there is one — and the tree calls it when a row is expanded.
A callable rather than a signal because ``select_path`` has to walk
several levels down synchronously to reveal a search hit.

Links are listed, never followed. An external link gets a chevron and is
resolved only if the user expands it themselves, which is the same
bargain the Link card's Follow button makes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from mlgidgui import icons, theme_tokens

import logging

logger = logging.getLogger(__name__)

#: What each row carries. Strings only — see the module docstring.
FILE_ROLE = Qt.ItemDataRole.UserRole
PATH_ROLE = Qt.ItemDataRole.UserRole + 1
KIND_ROLE = Qt.ItemDataRole.UserRole + 2
LOADED_ROLE = Qt.ItemDataRole.UserRole + 3

#: The stand-in child that gives an unopened row its chevron.
_PLACEHOLDER = "placeholder"

#: Glyph per row kind.
_ICONS = {
    "root": "dock-tree",
    "group": "node-group",
    "dataset": "node-dataset",
    "soft": "node-link",
    "external": "node-link",
    "broken": "node-link",
}


@dataclass(frozen=True)
class RootSpec:
    """One open file, as a root row.

    ``label`` is what the row reads (the original filename, not the temp
    working copy's), ``path`` is the file actually on disk that the
    lister will be asked about.
    """

    path: str
    label: str
    editable: bool = True
    detail: str = ""


def _norm(path: str) -> str:
    """``entry/data`` and ``/entry/data/`` both become ``/entry/data``."""
    text = str(path or "").strip()
    if not text.startswith("/"):
        text = "/" + text
    while len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    return text


class StructureTree(QTreeWidget):
    """A lazy, handle-free view of every open file."""

    #: ``(file, h5 path)`` — the user selected a row.
    nodeSelected = Signal(str, str)
    #: ``(file, h5 path, global position)`` — the user asked for a menu.
    contextRequested = Signal(str, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lister: Callable[[str, str], Iterable[Any]] | None = None
        # None means "whatever theme_tokens currently reports"; a theme
        # switch pins it so a row built afterwards matches the rest.
        self._theme: str | None = None
        self._roots: list[RootSpec] = []
        # Set while the tree is changing its own selection, so a restore
        # or a search jump does not read back as a user click.
        self._quiet = False

        self.setHeaderHidden(True)
        self.setColumnCount(1)
        # Uniform rows let Qt skip per-row height queries; the tree may
        # hold a few thousand rows on a master file.
        self.setUniformRowHeights(True)
        # Sorting stays OFF, for the same reason the File browser turns
        # it off: a sort has to compare rows, and comparing rows means
        # asking each one what it is. Native order is acquisition order
        # on the track_order files this targets, which is the useful one.
        self.setSortingEnabled(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setExpandsOnDoubleClick(True)
        # Extended, so a range of siblings can be shift-picked and
        # copied in one go. The panel still follows the CURRENT row —
        # describing a node is a per-node question — while copy, paste
        # and delete act on the whole selection.
        self.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)

        self.itemExpanded.connect(self._on_expanded)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.customContextMenuRequested.connect(self._on_context_menu)

    # -- wiring ------------------------------------------------------------

    def set_lister(self, lister: Callable[[str, str], Iterable[Any]]) -> None:
        """Install what answers "what is inside this group?".

        Called with ``(file, h5 path)`` and expected to return an
        iterable of objects with ``name``, ``kind``, ``expandable`` and
        ``detail`` — ``h5_edit.ChildInfo``. Duck-typed on purpose: this
        module stays free of h5py.
        """
        self._lister = lister

    def _list(self, file_path: str, h5_path: str) -> list[Any]:
        if self._lister is None:
            return []
        try:
            return list(self._lister(file_path, h5_path))
        except Exception:
            logger.debug("listing %s::%s failed", file_path, h5_path,
                         exc_info=True)
            return []

    # -- painting ----------------------------------------------------------

    def _paint_row(self, item: QTreeWidgetItem, kind: str) -> None:
        """Give a row its glyph, and a broken link its colour.

        Not ``icons.bind``: that registry holds widgets weakly and calls
        a one-argument ``setIcon``, and a tree item is neither — it takes
        a column, and it is thrown away and rebuilt every time its parent
        is re-listed. So the rows are repainted by hand from
        ``retheme()``, exactly as the dock tab bars are.
        """
        item.setIcon(0, icons.icon(_ICONS.get(kind, "node-dataset"),
                                   theme=self._theme))
        if kind == "broken":
            # A link whose target is gone is the one row a user has to be
            # able to pick out: it is usually why they opened the editor.
            item.setForeground(
                0, QBrush(QColor(theme_tokens.color("status_error",
                                                    self._theme))))

    def retheme(self, theme: str | None = None) -> None:
        """Repaint every row for a new theme. Called from the theme switch."""
        self._theme = theme

        def walk(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                if self._is_placeholder(child):
                    continue
                self._paint_row(child, str(child.data(0, KIND_ROLE) or ""))
                walk(child)

        for i in range(self.topLevelItemCount()):
            root = self.topLevelItem(i)
            self._paint_row(root, "root")
            walk(root)

    # -- roots -------------------------------------------------------------

    def set_roots(self, roots: list[RootSpec]) -> None:
        """Rebuild the root rows, keeping whatever is still there open.

        Called when a file is opened or closed. Roots come up collapsed
        unless they were open before, so adding a second file never
        costs a walk of the first.

        An unchanged list is a no-op, and that is load-bearing rather
        than an optimisation. This is called from the session-mode
        refresh, which also runs when the *active* session changes — and
        the active session changes when a row in this tree is clicked.
        Rebuilding here would destroy the very item the click is being
        delivered to, mid-signal.
        """
        roots = list(roots)
        if roots == self._roots:
            return
        self._roots = roots
        state = self.capture_state()
        self._quiet = True
        try:
            self.clear()
            for spec in roots:
                item = QTreeWidgetItem(self)
                item.setText(0, spec.label)
                item.setData(0, FILE_ROLE, str(spec.path))
                item.setData(0, PATH_ROLE, "/")
                item.setData(0, KIND_ROLE, "root")
                item.setData(0, LOADED_ROLE, False)
                item.setToolTip(0, spec.detail or str(spec.path))
                self._paint_row(item, "root")
                self._add_placeholder(item)
        finally:
            self._quiet = False
        self.restore_state(state)

    def root_labels(self) -> list[str]:
        """The root rows' text, for the tests."""
        return [
            self.topLevelItem(i).text(0)
            for i in range(self.topLevelItemCount())
        ]

    # -- population --------------------------------------------------------

    @staticmethod
    def _add_placeholder(item: QTreeWidgetItem) -> None:
        child = QTreeWidgetItem(item)
        child.setData(0, KIND_ROLE, _PLACEHOLDER)
        child.setText(0, "…")
        child.setFlags(Qt.ItemFlag.NoItemFlags)

    @staticmethod
    def _is_placeholder(item: QTreeWidgetItem) -> bool:
        return item.data(0, KIND_ROLE) == _PLACEHOLDER

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if not item.data(0, LOADED_ROLE):
            self._populate(item)

    def _populate(self, item: QTreeWidgetItem) -> None:
        """Fill one row's children from the lister. Idempotent."""
        file_path = item.data(0, FILE_ROLE)
        h5_path = item.data(0, PATH_ROLE)
        if not file_path:
            return
        children = self._list(str(file_path), str(h5_path or "/"))
        self._quiet = True
        try:
            item.takeChildren()
            for child in children:
                self._add_child(item, str(file_path), str(h5_path or "/"), child)
        finally:
            self._quiet = False
        item.setData(0, LOADED_ROLE, True)

    def _add_child(
        self, parent: QTreeWidgetItem, file_path: str, parent_path: str,
        info: Any,
    ) -> QTreeWidgetItem:
        name = str(getattr(info, "name", ""))
        kind = str(getattr(info, "kind", "group"))
        detail = str(getattr(info, "detail", ""))
        path = name if parent_path == "/" else f"{parent_path}/{name}"
        path = _norm(path)

        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        item.setData(0, FILE_ROLE, file_path)
        item.setData(0, PATH_ROLE, path)
        item.setData(0, KIND_ROLE, kind)
        item.setData(0, LOADED_ROLE, False)
        item.setToolTip(0, f"{path}\n{detail}" if detail else path)
        self._paint_row(item, kind)
        if getattr(info, "expandable", False):
            self._add_placeholder(item)
        return item

    # -- lookup ------------------------------------------------------------

    def _root_for(self, file_path: str) -> QTreeWidgetItem | None:
        wanted = str(file_path)
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if str(item.data(0, FILE_ROLE)) == wanted:
                return item
        # A file may be named differently on the two sides (a resolved
        # symlink, a relative path) — fall back to comparing what the
        # filesystem says they are.
        try:
            target = Path(wanted).resolve()
        except OSError:
            return None
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            try:
                if Path(str(item.data(0, FILE_ROLE))).resolve() == target:
                    return item
            except OSError:
                continue
        return None

    def find_item(
        self, file_path: str, h5_path: str, *, populate: bool = False
    ) -> QTreeWidgetItem | None:
        """The row for a path, or None.

        With ``populate`` the walk fills each group it passes through, so
        a path nobody has expanded can still be reached — that is how a
        search hit is revealed. Without it, only rows already on screen
        are searched, which is what a refresh wants.
        """
        root = self._root_for(file_path)
        if root is None:
            return None
        current = root
        for part in [p for p in _norm(h5_path).split("/") if p]:
            if populate and not current.data(0, LOADED_ROLE):
                self._populate(current)
            found = None
            for i in range(current.childCount()):
                child = current.child(i)
                if not self._is_placeholder(child) and child.text(0) == part:
                    found = child
                    break
            if found is None:
                return None
            current = found
        return current

    def current_target(self) -> tuple[str, str] | None:
        """``(file, h5 path)`` for the selected row, or None."""
        item = self.currentItem()
        if item is None or self._is_placeholder(item):
            return None
        file_path = item.data(0, FILE_ROLE)
        if not file_path:
            return None
        return (str(file_path), _norm(str(item.data(0, PATH_ROLE) or "/")))

    def selected_targets(self) -> list[tuple[str, str]]:
        """``(file, h5 path)`` for every selected row, in tree order.

        Placeholders and rows with no file behind them are skipped. The
        caller prunes nested picks (``structure_clipboard``); this
        reports what is selected, not what is sensible to act on.
        """
        out: list[tuple[str, str]] = []
        for item in self.selectedItems():
            if self._is_placeholder(item):
                continue
            file_path = item.data(0, FILE_ROLE)
            if not file_path:
                continue
            out.append(
                (str(file_path), _norm(str(item.data(0, PATH_ROLE) or "/"))))
        return out

    # -- refresh and reveal ------------------------------------------------

    def refresh_path(self, file_path: str, h5_path: str) -> None:
        """Re-list one group's children in place.

        What replaces the browser's teardown-and-rebuild after a
        structural edit. Only a row that is already populated is
        touched — one that has never been opened will list correctly
        whenever it is, and re-listing it now would be work for nobody.
        """
        item = self.find_item(file_path, h5_path)
        if item is None or not item.data(0, LOADED_ROLE):
            return
        open_children = {
            item.child(i).text(0)
            for i in range(item.childCount())
            if item.child(i).isExpanded()
        }
        target = self.current_target()
        self._populate(item)
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) in open_children:
                child.setExpanded(True)
        if target is not None:
            self.select_path(target[0], target[1], quiet=True)

    def select_path(
        self, file_path: str, h5_path: str, *, quiet: bool = False
    ) -> bool:
        """Reveal and select a path, expanding only along the way to it.

        ``quiet`` selects without emitting ``nodeSelected`` — used when
        the tree is putting a selection back after a refresh, where the
        panel is already showing that node.

        A row that is ALREADY part of the selection is only made
        current, never re-selected. ``setCurrentItem`` clears everything
        else under ``ExtendedSelection``, and the panel re-syncs to the
        current row on every render — so without this, shift-picking a
        second row would collapse the pick the moment the first row's
        render came back.
        """
        item = self.find_item(file_path, h5_path, populate=True)
        if item is None:
            return False
        previous = self._quiet
        self._quiet = quiet
        try:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            if item.isSelected():
                self.setCurrentItem(
                    item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
            else:
                self.setCurrentItem(item)
            # Selecting emits, and a handler may re-list the branch this
            # row is in — leaving the pointer above dangling. Nothing is
            # lost: whatever rebuilt it put the selection back.
            self.scrollToItem(item)
        except RuntimeError:
            logger.debug("the row went away while it was being selected",
                         exc_info=True)
        finally:
            self._quiet = previous
        return True

    # -- state across a rebuild --------------------------------------------

    def capture_state(self):
        """``(expanded paths, selection)`` — both as plain strings."""
        expanded: list[tuple[str, str]] = []

        def walk(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                if self._is_placeholder(child):
                    continue
                if child.isExpanded():
                    expanded.append((str(child.data(0, FILE_ROLE)),
                                     str(child.data(0, PATH_ROLE))))
                    walk(child)

        for i in range(self.topLevelItemCount()):
            root = self.topLevelItem(i)
            if root.isExpanded():
                expanded.append((str(root.data(0, FILE_ROLE)), "/"))
                walk(root)
        return expanded, self.current_target()

    def restore_state(self, state) -> None:
        """Put expansion and selection back after ``set_roots``."""
        expanded, selected = state
        self._quiet = True
        try:
            # Shallow paths first, so a parent is populated before the
            # child that needs it is looked up.
            for file_path, path in sorted(expanded, key=lambda kv: kv[1].count("/")):
                item = self.find_item(file_path, path, populate=True)
                if item is not None:
                    item.setExpanded(True)
        finally:
            self._quiet = False
        if selected is not None:
            self.select_path(selected[0], selected[1], quiet=True)

    # -- user intent -------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._quiet:
            return
        target = self.current_target()
        if target is not None:
            self.nodeSelected.emit(target[0], target[1])

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is not None and not self._is_placeholder(item):
            # Right-click means "this row", exactly as it does in the
            # File browser — so the row is selected first and the menu is
            # built against it, not against whatever was selected before.
            #
            # Unless it is already part of a multi-row selection: there
            # the click means "these rows", and collapsing the selection
            # would throw away what the user just shift-picked, right as
            # they reach for Copy.
            if item not in self.selectedItems():
                self.setCurrentItem(item)
            else:
                self.setCurrentItem(item, 0,
                                    QItemSelectionModel.SelectionFlag.NoUpdate)
        target = self.current_target()
        if target is None:
            return
        self.contextRequested.emit(
            target[0], target[1], self.viewport().mapToGlobal(pos))
