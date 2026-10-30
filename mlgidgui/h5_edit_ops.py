"""Undoable edits, and the history that holds them.

Every edit the Structure tab performs is also an object that knows how
to reverse itself. The shape mirrors the image viewer's ``_Action``
protocol (``viewer_items.py``): ``undo`` and ``redo`` are exact mirrors,
so the stacks can be walked in both directions without special cases.

Kept separate from the viewer's history on purpose. The two describe
different things — peaks versus file structure — and merging them would
mean a Ctrl+Z on the Image tab could reverse an edit the user made on
the Structure tab three minutes earlier, in a file view they are not
even looking at. ``MainWindow`` routes Ctrl+Z to whichever of the two
the front tab owns.

No Qt: an op takes an open ``h5py.File`` and works on it, which is what
makes the whole history testable without a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import h5py

from mlgidgui import h5_edit
from mlgidgui.h5_edit import MISSING, format_value

import logging

logger = logging.getLogger(__name__)

#: How many edits the history keeps. Old entries fall off the bottom;
#: 200 is far past any single sitting and bounds the memory a run of
#: snapshotted deletes can pin.
HISTORY_LIMIT = 200


class EditOp(Protocol):
    """One reversible edit."""

    def undo(self, f: h5py.File) -> None: ...
    def redo(self, f: h5py.File) -> None: ...
    def describe(self) -> str: ...


def _apply_attr(f: h5py.File, path: str, name: str, value: Any) -> None:
    """Set ``name`` to ``value``, or remove it when ``value`` is MISSING."""
    if value is MISSING:
        try:
            h5_edit.delete_attr(f, path, name)
        except h5_edit.EditError:
            # Already gone. Reaching the intended state is the contract;
            # how it got there is not.
            logger.debug("attribute %s already absent on %s", name, path)
        return
    h5_edit.set_attr(f, path, name, value)


@dataclass
class SetAttrOp:
    """Create, change or delete one attribute.

    One class for all three because they are the same edit with
    ``MISSING`` on one side or the other: creating has no ``before``,
    deleting has no ``after``. That keeps undo symmetric — undoing a
    delete is just applying the ``before`` again.
    """

    path: str
    name: str
    before: Any
    after: Any

    def redo(self, f: h5py.File) -> None:
        _apply_attr(f, self.path, self.name, self.after)

    def undo(self, f: h5py.File) -> None:
        _apply_attr(f, self.path, self.name, self.before)

    def describe(self) -> str:
        where = f"{self.path}@{self.name}"
        if self.before is MISSING:
            return f"+ {where} = {format_value(self.after, limit=40)}"
        if self.after is MISSING:
            return f"- {where}  (was {format_value(self.before, limit=40)})"
        return (
            f"{where}: {format_value(self.before, limit=30)} → "
            f"{format_value(self.after, limit=30)}"
        )


@dataclass
class RenameAttrOp:
    path: str
    before: str
    after: str

    def redo(self, f: h5py.File) -> None:
        h5_edit.rename_attr(f, self.path, self.before, self.after)

    def undo(self, f: h5py.File) -> None:
        h5_edit.rename_attr(f, self.path, self.after, self.before)

    def describe(self) -> str:
        return f"{self.path}@{self.before} renamed to {self.after}"


@dataclass
class WriteCellsOp:
    """A set of cell writes into one dataset, reversed by writing back."""

    path: str
    before: dict[Any, Any]
    after: dict[Any, Any]

    def redo(self, f: h5py.File) -> None:
        h5_edit.write_cells(f, self.path, self.after)

    def undo(self, f: h5py.File) -> None:
        h5_edit.write_cells(f, self.path, self.before)

    def describe(self) -> str:
        if len(self.after) == 1:
            (index, value), = self.after.items()
            old = self.before.get(index)
            where = self.path if index == () else f"{self.path}{_index_text(index)}"
            return (
                f"{where}: {format_value(old, limit=30)} → "
                f"{format_value(value, limit=30)}"
            )
        return f"{self.path}: {len(self.after)} cells changed"


def _index_text(index: Any) -> str:
    if isinstance(index, tuple):
        return "[" + ", ".join(str(part) for part in index) + "]"
    return f"[{index}]"


class EditHistory:
    """The undo/redo stacks, and the session's list of changes.

    The changes list is not a separate log: it *is* the undo stack, read
    oldest-first. Undoing an edit therefore takes its line back off the
    list, which is the only reading of "what I have changed" that stays
    honest.
    """

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        self._limit = limit
        self._undo: list[EditOp] = []
        self._redo: list[EditOp] = []

    def push(self, op: EditOp) -> None:
        """Record an edit that has *already* been applied."""
        self._undo.append(op)
        del self._undo[: max(0, len(self._undo) - self._limit)]
        self._redo.clear()

    def undo(self, f: h5py.File) -> EditOp | None:
        if not self._undo:
            return None
        op = self._undo.pop()
        op.undo(f)
        self._redo.append(op)
        return op

    def redo(self, f: h5py.File) -> EditOp | None:
        if not self._redo:
            return None
        op = self._redo.pop()
        op.redo(f)
        self._undo.append(op)
        return op

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def entries(self) -> list[str]:
        """One line per edit still standing, oldest first."""
        return [op.describe() for op in self._undo]

    def as_text(self) -> str:
        """The changes list as plain text, for the clipboard."""
        return "\n".join(self.entries())

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)


@dataclass
class CreateNodeOp:
    """A group, dataset or link that was created. Undone by deleting it.

    ``recreate`` is a one-argument callable taking the open file, so redo
    replays exactly what the dialog asked for without this class having
    to know which of the three kinds it was.
    """

    path: str
    kind: str
    recreate: Any = None

    def redo(self, f: h5py.File) -> None:
        if self.recreate is not None:
            self.recreate(f)

    def undo(self, f: h5py.File) -> None:
        h5_edit.delete_node(f, self.path)

    def describe(self) -> str:
        return f"+ {self.path}  ({self.kind})"


@dataclass
class BatchOp:
    """Several edits the user made with one gesture.

    Pasting six groups is one action to the person who did it, so it is
    one entry in the changes list and one press of Ctrl+Z. Without this
    a multi-paste would leave six lines and need six undos, which is a
    worse answer to "what did I just do?" than the single line it
    replaces.

    Undo runs the members in REVERSE. It matters: a batch that created
    ``a`` then ``b`` must remove ``b`` first, and a batch that deleted a
    group and then its sibling must put the sibling back before the
    group it was ordered after. Redo replays forwards.

    ``paths`` is what the tree has to re-list afterwards — a batch can
    touch several parents at once, which no single-path op ever does.
    """

    ops: list
    label: str = ""

    @property
    def path(self) -> str:
        """The primary path, for the rules that ask an op where it hit."""
        for op in self.ops:
            candidate = getattr(op, "path", "")
            if candidate:
                return candidate
        return ""

    @property
    def paths(self) -> list[str]:
        """Every path the members touched, including a move's origin."""
        seen: list[str] = []
        for op in self.ops:
            for name in ("path", "before"):
                candidate = getattr(op, name, "")
                if candidate and candidate not in seen:
                    seen.append(candidate)
        return seen

    @property
    def reversible(self) -> bool:
        """False when any member cannot be undone (an oversized delete)."""
        return all(getattr(op, "reversible", True) for op in self.ops)

    def redo(self, f: h5py.File) -> None:
        for op in self.ops:
            op.redo(f)

    def undo(self, f: h5py.File) -> None:
        for op in reversed(self.ops):
            op.undo(f)

    def describe(self) -> str:
        if self.label:
            return self.label
        return f"{len(self.ops)} changes"


@dataclass
class DeleteNodeOp:
    """A node that was deleted, with the bytes needed to bring it back.

    ``snapshot`` is None when the node was too large to hold in memory
    (see ``h5_edit.UNDO_SNAPSHOT_LIMIT``). Such a delete is not
    reversible, and the GUI says so before it happens rather than
    letting the user find out at Ctrl+Z.
    """

    path: str
    kind: str
    snapshot: Any = None

    @property
    def reversible(self) -> bool:
        return self.snapshot is not None

    def redo(self, f: h5py.File) -> None:
        h5_edit.delete_node(f, self.path)

    def undo(self, f: h5py.File) -> None:
        if self.snapshot is None:
            raise h5_edit.EditError(
                f"{self.path} was too large to hold in memory, so this "
                "delete cannot be undone."
            )
        parent, _ = h5_edit.split_path(self.path)
        h5_edit.restore_snapshot(f, parent, self.snapshot)

    def describe(self) -> str:
        tail = "" if self.reversible else "  (not undoable)"
        return f"- {self.path}  ({self.kind}){tail}"


@dataclass
class MoveNodeOp:
    """A rename or a move. Both are one HDF5 operation, so both are one op."""

    before: str
    after: str

    @property
    def path(self) -> str:
        """Where the node is now — what the viewer-refresh rule reads."""
        return self.after

    def redo(self, f: h5py.File) -> None:
        parent, name = h5_edit.split_path(self.after)
        h5_edit.move_node(f, self.before, parent, name)

    def undo(self, f: h5py.File) -> None:
        parent, name = h5_edit.split_path(self.before)
        h5_edit.move_node(f, self.after, parent, name)

    def describe(self) -> str:
        before_parent, _ = h5_edit.split_path(self.before)
        after_parent, after_name = h5_edit.split_path(self.after)
        if before_parent == after_parent:
            return f"{self.before} renamed to {after_name}"
        return f"{self.before} moved to {self.after}"


@dataclass
class DeleteLinkOp:
    """An unlinked soft or external link, remembered as a link.

    Deliberately not a ``DeleteNodeOp``: snapshotting would have to copy
    the link's *target*, which for an external link means opening
    another file — the multi-GB scan the link exists to avoid — and for
    a dangling one is impossible. A link is three strings, so undo just
    writes it again.
    """

    path: str
    kind: str
    target: str
    filename: str = ""

    def redo(self, f: h5py.File) -> None:
        h5_edit.delete_node(f, self.path)

    def undo(self, f: h5py.File) -> None:
        parent, name = h5_edit.split_path(self.path)
        h5_edit.create_link(
            f, parent, name, self.kind, self.target,
            filename=self.filename or None,
        )

    def describe(self) -> str:
        return f"- {self.path}  ({self.kind} link)"


@dataclass
class RetargetLinkOp:
    """A link pointed somewhere else, remembered as both endpoints."""

    path: str
    before_kind: str
    before_target: str
    before_filename: str
    after_kind: str
    after_target: str
    after_filename: str

    def _apply(self, f: h5py.File, kind: str, target: str, filename: str) -> None:
        h5_edit.retarget_link(
            f, self.path, kind, target, filename=filename or None)

    def redo(self, f: h5py.File) -> None:
        self._apply(f, self.after_kind, self.after_target, self.after_filename)

    def undo(self, f: h5py.File) -> None:
        self._apply(f, self.before_kind, self.before_target, self.before_filename)

    def describe(self) -> str:
        def spell(kind: str, target: str, filename: str) -> str:
            return f"{filename}::{target}" if kind == "external" else target
        return (
            f"{self.path}: {spell(self.before_kind, self.before_target, self.before_filename)}"
            f" → {spell(self.after_kind, self.after_target, self.after_filename)}"
        )


@dataclass
class ReplaceDataOp:
    """A dataset's contents replaced wholesale, both sides remembered.

    Used where a sparse cell diff cannot describe the change — a row
    inserted or deleted alters the length, so every index after it moves.
    Both arrays are held in memory, which is affordable because the grid
    only opens on datasets small enough to edit by hand in the first
    place (``h5_edit.VIEW_CELL_LIMIT``).
    """

    path: str
    before: Any
    after: Any

    def redo(self, f: h5py.File) -> None:
        h5_edit.rewrite_dataset(f, self.path, self.after)

    def undo(self, f: h5py.File) -> None:
        h5_edit.rewrite_dataset(f, self.path, self.before)

    def describe(self) -> str:
        rows_before = getattr(self.before, "shape", (0,))[:1]
        rows_after = getattr(self.after, "shape", (0,))[:1]
        if rows_before != rows_after:
            return (
                f"{self.path}: {rows_before[0] if rows_before else 0} → "
                f"{rows_after[0] if rows_after else 0} rows"
            )
        return f"{self.path}: values replaced"
