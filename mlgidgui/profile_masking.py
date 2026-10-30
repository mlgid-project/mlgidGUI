"""Mask the neighbouring peaks out of a selected peak's profiles.

pygidfit fits one cluster at a time and sets every box outside the
cluster to NaN inside the cluster's ROI (``pygidfit.fitting_models.
fit_peak_cluster``). The profile viewer can do the same to the polar
image before it averages, so the radial and angular profiles of peak
``x`` show peak ``x`` and not the peaks next to it. The profile code
already averages with ``nanmean``, so masked pixels simply drop out.

Which rectangle a neighbour masks follows pygidfit's own box
(``pygidfit.box_utils.boxes_preprocessing``, mirrored by
``manual_fit._px_box``): ``radius +- radius_width`` radially (the full
width each side, not half) and ``angle +- angle_width / 2`` angularly.
The ``+-1`` pixel pad pygidfit adds is not reproduced; it is a pixel of
pygidfit's grid, and the viewer's polar grid has a different pitch.

pygidfit does not mask every other box. Peaks whose centres lie within
its clustering distance of the target (the Pipeline panel's "Cluster
dist (peaks)", pixels of its polar grid; 2 by default since mlgidbase
0.1.11) are fitted jointly with it, not masked. The same rule applies here: a
neighbour that close is left in the profile, since it overlaps the
selected peak and cannot be separated from it. Like pygidfit, nothing
else is protected, so a farther neighbour's box masks whatever it
covers, including part of the selected peak's own band.

Rings are never neighbours (``angle_width`` is inf); the caller's box
list is expected to leave them out, and non-finite boxes are skipped
here as well. Chains (A clusters with B, B with the target) are not
followed.

Pure numpy; no Qt, no pygidfit.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from mlgidgui.manual_fit import (
    _PYGIDFIT_POLAR_SHAPE,
    DEFAULT_CLUSTERING_DISTANCE_PEAKS,
    PeakBox,
    _overlaps_drawn,
)


def _span(axis: np.ndarray, lo: float, hi: float) -> slice:
    """Index slice of a monotonic ``axis`` covering ``[lo, hi]``; a
    non-finite bound means the axis end."""
    n = len(axis)
    lo_i = 0 if not math.isfinite(lo) else int(np.searchsorted(axis, lo, side="left"))
    hi_i = n if not math.isfinite(hi) else int(np.searchsorted(axis, hi, side="right"))
    lo_i = max(0, min(lo_i, n))
    hi_i = max(lo_i, min(hi_i, n))
    return slice(lo_i, hi_i)


def _is_twin(target: PeakBox, cand: PeakBox) -> bool:
    """True when ``cand`` is the same physical peak as ``target`` under
    another name (a detected box and the fitted row made from it).

    ``manual_fit._overlaps_drawn`` decides by whether one centre lies in
    the other's drawn box, which is meaningless for a box with no finite
    extent: a ring-like target (``angle_width = inf``) would "contain"
    every spot at its radius and no neighbour would ever be masked. A
    box without a finite extent therefore has no twin here; its own
    detected twin is a ring, which the caller leaves out of the
    neighbours anyway.
    """
    extents = (
        target.radius_width, target.angle_width,
        cand.radius_width, cand.angle_width,
    )
    if not all(math.isfinite(float(v)) for v in extents):
        return False
    return _overlaps_drawn(target, cand)


def _clustered(
    target: PeakBox, cand: PeakBox, q_abs_max: float, distance: float,
) -> bool:
    """True when pygidfit would cluster ``cand`` with ``target``: centres
    within ``distance`` pixels of its polar grid (the ``N - 1`` scale of
    its ``np.linspace(.., N)`` axes)."""
    n_ang, n_rad = _PYGIDFIT_POLAR_SHAPE
    dx = (cand.radius - target.radius) / q_abs_max * (n_rad - 1)
    dy = (cand.angle - target.angle) / 90.0 * (n_ang - 1)
    return math.hypot(dx, dy) <= distance


def neighbour_mask(
    shape: tuple[int, int],
    radius_axis: np.ndarray,
    angle_axis: np.ndarray,
    target: PeakBox,
    neighbours: Iterable[PeakBox],
    cluster_distance: float | None = None,
) -> np.ndarray | None:
    """Boolean mask, ``True`` where a neighbour's box covers the pixel.

    ``cluster_distance`` is pygidfit's clustering distance for peaks, in
    pixels of its polar grid: neighbours closer than that to the target
    are fitted with it, so they stay in the profile. None = the
    mlgidbase default.

    ``shape`` is the polar image's ``(n_radius, n_angle)``; the largest
    radius on ``radius_axis`` stands in for pygidfit's ``q_abs_max``.
    Returns ``None`` when nothing is masked, so callers can skip the
    copy.
    """
    distance = (
        DEFAULT_CLUSTERING_DISTANCE_PEAKS if cluster_distance is None
        else float(cluster_distance)
    )
    q_abs_max = float(radius_axis[-1]) if len(radius_axis) else 0.0
    if not (math.isfinite(q_abs_max) and q_abs_max > 0):
        return None
    mask = np.zeros(shape, dtype=bool)
    for cand in neighbours:
        values = (cand.radius, cand.radius_width, cand.angle, cand.angle_width)
        if not all(math.isfinite(float(v)) for v in values):
            continue
        if cand.radius_width == 0.0 or cand.angle_width == 0.0:
            continue
        if _is_twin(target, cand) or _clustered(
            target, cand, q_abs_max, distance,
        ):
            continue
        dr = abs(float(cand.radius_width))
        da = abs(float(cand.angle_width)) / 2.0
        rs = _span(radius_axis, cand.radius - dr, cand.radius + dr)
        as_ = _span(angle_axis, cand.angle - da, cand.angle + da)
        if rs.stop > rs.start and as_.stop > as_.start:
            mask[rs, as_] = True
    return mask if mask.any() else None
