"""In-memory peak clipboard, scoped to the source entry.

Backs the Ctrl+C / Ctrl+V workflow for detected peaks. Same-entry only:
``take_items`` returns ``[]`` when the caller's target entry doesn't
match the entry the items were copied from. This avoids cross-instrument
mistakes (different q-axis ranges, different geometry) without making
the host carry the conditional.

Not a Qt clipboard — paste must be byte-exact polar coords, and we
don't want interop with other apps' text clipboards. Single module-
level snapshot; ``set_items`` overwrites whatever was there.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipboardItem:
    """A frozen snapshot of one detected peak's geometry + source coords.

    ``source_frame`` and ``source_peak_id`` are metadata only — used by
    the host for log lines. The paste target is always the active frame
    and the new peak gets a fresh id from ``add_detected_peak_row``.
    Kind is implicit (always detected in this iteration; expand the
    dataclass if other kinds become copyable).

    ``score`` carries the source peak's mlgidDETECT confidence so a paste
    reproduces it verbatim (copy must not change the score). Defaulted so
    older call sites / tests that omit it still construct.
    """

    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool
    source_frame: int
    source_peak_id: int
    score: float = 0.0


# Module-level state — at most one snapshot at a time.
_items: list[ClipboardItem] = []
_source_entry: str | None = None


def set_items(items: list[ClipboardItem], *, entry: str) -> None:
    """Replace the clipboard contents.

    ``entry`` is the source entry name; ``take_items`` will refuse to
    return the snapshot for any other entry.
    """
    global _items, _source_entry
    _items = list(items)
    _source_entry = entry


def take_items(target_entry: str) -> list[ClipboardItem]:
    """Return the snapshot if it was copied from ``target_entry``, else ``[]``.

    Does not mutate the snapshot — paste-twice on the same frame is
    allowed (the user gets duplicate rows, mirroring how copy-paste
    works in every other GUI). Callers that want one-shot paste should
    call ``clear()`` themselves.
    """
    if _source_entry is None or _source_entry != target_entry:
        return []
    return list(_items)


def clear() -> None:
    """Drop the snapshot. Used by tests for hermetic isolation."""
    global _items, _source_entry
    _items = []
    _source_entry = None


def has_items() -> bool:
    """Whether the clipboard currently holds a snapshot (any entry)."""
    return bool(_items)
