"""Mask the neighbouring peaks out of a selected peak's profiles.

pygidfit fits one cluster at a time and sets every box outside the
cluster to NaN inside the cluster's ROI (``pygidfit.fitting_models.
fit_peak_cluster``). The profile viewer does the same to the image
pixels before it bins them (``profile_binning``), so the radial and
angular profiles of peak ``x`` show peak ``x`` and not the peaks next to
it. Masked pixels simply leave their bins.

Which rectangle a neighbour masks follows pygidfit's own box
(``pygidfit.box_utils.boxes_preprocessing``, mirrored by
``manual_fit._px_box``): ``radius +- radius_width`` radially (the full
width each side, not half) and ``angle +- angle_width / 2`` angularly.
The ``+-1`` pixel pad pygidfit adds is not reproduced; it is a pixel of
pygidfit's polar grid, which the image pixels do not share.

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
    pixel_radius: np.ndarray,
    pixel_angle: np.ndarray,
    target: PeakBox,
    neighbours: Iterable[PeakBox],
    cluster_distance: float | None = None,
    q_abs_max: float | None = None,
) -> np.ndarray | None:
    """Boolean mask, ``True`` where a neighbour's box covers the pixel.

    ``pixel_radius`` / ``pixel_angle`` hold the |q| (1/A) and angle
    (degrees) of each pixel, in any shape -- the profile viewer passes
    just the pixels of the band it is about to average
    (``profile_binning``), so the cost scales with the band, not the
    image. The mask has the same shape.

    ``cluster_distance`` is pygidfit's clustering distance for peaks, in
    pixels of its polar grid: neighbours closer than that to the target
    are fitted with it, so they stay in the profile. None = the
    mlgidbase default. ``q_abs_max`` stands in for pygidfit's
    ``q_abs_max`` (the image's largest |q|); None = the largest
    ``pixel_radius`` given.

    Returns ``None`` when nothing is masked, so callers can skip the
    copy.
    """
    pixel_radius = np.asarray(pixel_radius)
    pixel_angle = np.asarray(pixel_angle)
    distance = (
        DEFAULT_CLUSTERING_DISTANCE_PEAKS if cluster_distance is None
        else float(cluster_distance)
    )
    if q_abs_max is None:
        q_abs_max = (
            float(np.nanmax(pixel_radius)) if pixel_radius.size else 0.0
        )
    q_abs_max = float(q_abs_max)
    if not (math.isfinite(q_abs_max) and q_abs_max > 0):
        return None
    mask = np.zeros(pixel_radius.shape, dtype=bool)
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
        mask |= (
            (pixel_radius >= cand.radius - dr)
            & (pixel_radius <= cand.radius + dr)
            & (pixel_angle >= cand.angle - da)
            & (pixel_angle <= cand.angle + da)
        )
    return mask if mask.any() else None
