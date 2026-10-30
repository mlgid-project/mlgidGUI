"""View window for zooming the image onto one selected peak.

Pure numpy; no Qt. The caller hands the peak's box and the data limits
of the active display and gets back the ``(x_range, y_range)`` to set on
the viewbox:

* polar display: x = radius (Å⁻¹), y = angle (degrees), so the box is
  the plain rectangle ``radius ± radius_width / 2`` by
  ``angle ± angle_width / 2``;
* Cartesian display: x = q_xy, y = q_z, so the box is an annular sector
  and the window is the bounding box of its arcs.

The window is the box grown by ``PAD`` box-sizes on each side (so the
box fills the middle third), never smaller than ``MIN_FRACTION`` of the
data range per axis (a very narrow box would otherwise zoom to a few
pixels), and shifted, not shrunk, to stay inside the data.
"""
from __future__ import annotations

import math

import numpy as np

PAD = 1.0
MIN_FRACTION = 0.05
_ARC_SAMPLES = 33


def _interval(lo: float, hi: float, data_lo: float, data_hi: float,
              min_extent: float) -> tuple[float, float]:
    centre = 0.5 * (lo + hi)
    width = max((hi - lo) * (1.0 + 2.0 * PAD), min_extent)
    width = min(width, data_hi - data_lo)
    lo, hi = centre - width / 2.0, centre + width / 2.0
    if lo < data_lo:
        lo, hi = data_lo, data_lo + width
    if hi > data_hi:
        lo, hi = data_hi - width, data_hi
    return lo, hi


def peak_view_range(
    mode: str,
    radius: float,
    radius_width: float,
    angle: float,
    angle_width: float,
    *,
    radius_max: float,
    angle_extent: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """``((x0, x1), (y0, y1))`` for the viewbox, or None when the box or
    the data limits are unusable. ``mode`` is ``"polar"`` or
    ``"cartesian"``; anything else returns None."""
    values = (radius, radius_width, radius_max, *angle_extent)
    if not all(math.isfinite(float(v)) for v in values) or radius_max <= 0:
        return None
    a_min, a_max = float(angle_extent[0]), float(angle_extent[1])
    if a_max <= a_min:
        return None
    half_r = abs(float(radius_width)) / 2.0
    if not math.isfinite(float(angle)) or not math.isfinite(float(angle_width)):
        # A ring: no angular extent, the box spans the whole axis.
        a_lo, a_hi = a_min, a_max
    else:
        half_a = abs(float(angle_width)) / 2.0
        a_lo = max(float(angle) - half_a, a_min)
        a_hi = min(float(angle) + half_a, a_max)
    if a_hi <= a_lo:
        return None
    r_lo = max(float(radius) - half_r, 0.0)
    r_hi = float(radius) + half_r

    if mode == "polar":
        return (
            _interval(r_lo, r_hi, 0.0, float(radius_max),
                      MIN_FRACTION * radius_max),
            _interval(a_lo, a_hi, a_min, a_max,
                      MIN_FRACTION * (a_max - a_min)),
        )
    if mode == "cartesian":
        angs = np.deg2rad(np.linspace(a_lo, a_hi, _ARC_SAMPLES))
        xs = np.concatenate([r_lo * np.cos(angs), r_hi * np.cos(angs)])
        ys = np.concatenate([r_lo * np.sin(angs), r_hi * np.sin(angs)])
        lim = float(radius_max)
        return (
            _interval(float(xs.min()), float(xs.max()), -lim, lim,
                      MIN_FRACTION * lim),
            _interval(float(ys.min()), float(ys.max()), -lim, lim,
                      MIN_FRACTION * lim),
        )
    return None
