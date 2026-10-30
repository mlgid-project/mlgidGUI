"""What the Structure tab holds between a Copy and a Paste.

A reference, not a payload: the source file and the path inside it, kept
until the paste actually happens. Copying a 4 GB stack therefore costs
nothing until it is pasted somewhere, and copying an external link keeps
it a link — the same rule the rest of the editor follows.

Modelled on ``peak_clipboard``, which does the same job for detected
peaks. The two are separate on purpose: they hold different things, and
Ctrl+C means "this peak" on the Image tab and "this group" on the
Structure tab.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

ClipKind = Literal["node", "attribute"]
ClipMode = Literal["copy", "cut"]


@dataclass(frozen=True)
class StructureClip:
    """One copied node or attribute, as a reference to where it lives."""

    kind: ClipKind
    file_path: Path
    path: str
    attr: str = ""
    mode: ClipMode = "copy"

    @property
    def name(self) -> str:
        """The name a paste would use by default."""
        if self.kind == "attribute":
            return self.attr
        return self.path.rstrip("/").rpartition("/")[2] or "root"

    def describe(self) -> str:
        """One line for a menu entry or a log message."""
        verb = "Cut" if self.mode == "cut" else "Copied"
        if self.kind == "attribute":
            return f"{verb} attribute {self.attr} from {self.path}"
        return f"{verb} {self.path} from {self.file_path.name}"


def prune_descendants(targets: Iterable[tuple]) -> list[tuple]:
    """Drop any ``(file, path)`` that lives inside another one.

    Shift-selecting a range in a tree picks up whatever rows happen to
    lie between the two clicks, and an expanded group brings its
    children with it. Copying both the group and its child would paste
    the child twice — once inside its parent and once beside it — and
    deleting both would try to delete a node that its parent already
    took with it.

    Order is preserved, so the caller still sees the selection in tree
    order. Comparison is per file: the same path in two files is two
    different nodes.
    """
    kept: list[tuple] = []
    for file_path, path in targets:
        prefix = path.rstrip("/") + "/"
        inside = any(
            str(other_file) == str(file_path)
            and prefix.startswith(other.rstrip("/") + "/")
            and other.rstrip("/") != path.rstrip("/")
            for other_file, other in targets
        )
        if not inside:
            kept.append((file_path, path))
    return kept


def free_name(existing: Iterable[str], wanted: str) -> str:
    """A name like ``wanted`` that is not already in ``existing``.

    ``q_xy`` -> ``q_xy_copy`` -> ``q_xy_copy2`` … Suggested rather than
    imposed: the paste offers it in a dialog, so a copy never silently
    lands under a name the user did not choose.
    """
    taken = set(existing)
    if wanted not in taken:
        return wanted
    stem = f"{wanted}_copy"
    if stem not in taken:
        return stem
    index = 2
    while f"{stem}{index}" in taken:
        index += 1
    return f"{stem}{index}"
