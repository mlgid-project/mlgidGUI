"""The fitted ↔ detected peak link: one fit per detection, same id.

``detected_peaks`` and ``fitted_peaks`` are two independent tables that
each assign their own ``max(id) + 1``, and nothing on disk records which
fit came from which detection. This module holds the two user settings
that change that, plus the one sentence of policy each one carries.

**Link** (``PEAK_LINK_KEY``, default on)
    A fit made from detected peak 42 is written as fitted peak 42.
    Fitting that detection again — with different borders, a different
    mode, whatever — replaces fitted 42 instead of appending a second
    row, so a detection can never accumulate fits. Deleting detected 42
    deletes fitted 42 with it. A pipeline fitting run keeps the
    pairing by itself: pygidfit 0.1.8 writes each fitted row under the
    id of the detected row it came from (older versions numbered fitted
    rows 0..N-1 by position and needed a re-key pass after every run;
    that pass was removed with the 0.1.8 pin).
    A hand-drawn box is written to ``detected_peaks`` first and the fit
    pairs to that new row, so every fitted peak has a detected partner.

**Reverse delete** (``PEAK_LINK_REVERSE_KEY``, default off)
    Deleting a fitted peak also deletes its detected partner. Off by
    default because discarding a fit is the normal way to say "this
    prediction is wrong", which is not the same as saying the detection
    was wrong.

Both are read lazily wherever the behaviour is consumed — the same
shape as ``autoUpdate`` — so a change takes effect on the next action
without any wiring. With the link off every path behaves exactly as it
did before the setting existed; that is the acceptance criterion, and
the tests assert the off case alongside each on case.

Matched peaks are untouched by all of this. Where a linked delete
removes a fitted row, the existing ``prepare_fitted_delete`` /
``remap_matched_peak_lists`` handling runs exactly as it does for a
plain fitted delete — no new matched semantics anywhere.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings

from mlgidgui.main_window_constants import (
    DEFAULT_PEAK_LINK,
    DEFAULT_PEAK_LINK_REVERSE,
    PEAK_LINK_KEY,
    PEAK_LINK_REVERSE_KEY,
)


def read_bool(key: str, default: bool) -> bool:
    """One QSettings boolean, tolerant of the INI backend.

    Values written as Python ``bool`` come back as the strings
    ``"true"`` / ``"false"`` on Linux and as real bools on macOS. The
    repo's older idiom (``str(value(key, "false")).lower() == "true"``)
    only works for a False default; this one is symmetric, so a
    default-on setting stays on until it is explicitly turned off.
    """
    raw = QSettings().value(key, None)
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return bool(default)


def link_enabled() -> bool:
    """One fitted peak per detected peak, paired by id."""
    return read_bool(PEAK_LINK_KEY, DEFAULT_PEAK_LINK)


def reverse_delete_enabled() -> bool:
    """Deleting a fitted peak also deletes its detected partner.

    Only meaningful while :func:`link_enabled` is True — without paired
    ids there is no partner to delete — so callers check both.
    """
    return link_enabled() and read_bool(
        PEAK_LINK_REVERSE_KEY, DEFAULT_PEAK_LINK_REVERSE
    )
