"""Which overlay box did the user click?

Overlay boxes overlap constantly: a fitted box sits on top of the
detected box it refines, a wide detection contains a narrow one, and a
ring contains *every* spot peak at its radius, because mlgidGUI's ring
convention is ``angle = 45, angle_width = inf``. The old rule resolved
those by table order (highest row index wins), which is invisible to
the user: each overlay kind is drawn as a single ``QPainterPath`` with
one pen, so there is no "box on top" on screen to appeal to.

The rules here are the visible ones:

* **Kind first.** manual, then fitted, then detected, then matched, the
  order the app has always used. A refined fit still wins over the
  detection under it.
* **Then the smallest box.** Inside one kind the innermost box wins,
  because it is the one with nowhere else to be clicked; the box around
  it is still reachable everywhere the inner one is not.
* **Rings only on their edges.** A ring's box spans the whole angular
  range, so counting its interior would let it swallow every peak in
  its radius band. It is picked by clicking within a few screen pixels
  of the inner or outer radius instead.

Ties keep the caller's order, so feeding rows in reverse table order
preserves the old "highest row index wins" behaviour for boxes that are
genuinely the same size.

Pure functions over plain floats: no Qt, no PeakTable, so the rules can
be tested without building a viewer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, TypeVar

#: How close to a ring's inner or outer radius a click has to land,
#: in screen pixels. Converted to data units by the caller, which is
#: the only place that knows the current zoom.
RING_EDGE_TOL_PX = 6.0

#: Radial tolerance to fall back on when the view cannot report a pixel
#: size yet (no geometry, so nothing has been rendered to click on).
RING_EDGE_TOL_FALLBACK = 0.02

#: How far outside a box a click still counts as hitting it, in screen
#: pixels. A detected peak is often only a handful of pixels across, so
#: exact containment made a small box inside a larger one effectively
#: unselectable: miss it by three pixels and the click went to the box
#: around it, with no way back — the inner box was never in the stack
#: to step to.
HIT_TOL_PX = 4.0

K = TypeVar("K")


@dataclass(frozen=True)
class Box:
    """One overlay box in polar coordinates: (radius, angle) plus widths."""

    radius: float
    radius_width: float
    angle: float
    angle_width: float
    is_ring: bool = False


def is_ring_box(box: Box) -> bool:
    """A ring by flag or by geometry.

    Rows written by the GUI carry ``is_ring``; rows that arrive from the
    pipeline can instead carry a non-finite ``angle_width``, which means
    the same thing (see ``simulation_pattern`` and ``_clip_angle``).
    """
    return bool(box.is_ring) or not math.isfinite(box.angle_width)


def box_area(box: Box) -> float:
    """Area of the polar box, in Å⁻¹ × degrees.

    Rings return infinity so they sort last among equals: their real
    area is meaningless (the angular extent is unbounded) and they are
    always the outermost thing under the cursor.
    """
    if is_ring_box(box):
        return math.inf
    return abs(box.radius_width) * abs(box.angle_width)


def contains(
    box: Box,
    radius: float,
    angle: float,
    *,
    ring_edge_tol: float | None = None,
    pad_radius: float = 0.0,
    pad_angle: float = 0.0,
) -> bool:
    """Does ``(radius, angle)`` hit ``box``?

    ``ring_edge_tol`` is the radial half-width of a ring's clickable
    edge, in data units. Passing ``None`` keeps the historical
    behaviour where a ring's whole radial band counts, which is what
    non-picking callers (geometry maths, tests of the old helpers)
    still want.

    ``pad_radius`` / ``pad_angle`` grow the box before testing, in data
    units. The picker passes a few screen pixels' worth: peak boxes are
    often only a handful of pixels across, and a box that has to be hit
    exactly is not a box the user can select. Rings are already padded
    by ``ring_edge_tol``, so only their angular bound takes the pad.
    """
    half_r = abs(box.radius_width) / 2.0
    r_lo, r_hi = box.radius - half_r, box.radius + half_r

    if math.isfinite(box.angle_width):
        half_a = abs(box.angle_width) / 2.0 + abs(pad_angle)
        if not (box.angle - half_a <= angle <= box.angle + half_a):
            return False

    if is_ring_box(box) and ring_edge_tol is not None:
        return (
            abs(radius - r_lo) <= ring_edge_tol
            or abs(radius - r_hi) <= ring_edge_tol
        )
    return r_lo - abs(pad_radius) <= radius <= r_hi + abs(pad_radius)


def rank_hits(hits: Sequence[tuple[K, Box]]) -> list[K]:
    """Order candidates of one kind: smallest box first, ties as given."""
    return [key for key, _ in sorted(hits, key=lambda pair: box_area(pair[1]))]
