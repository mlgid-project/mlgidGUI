"""Radial / angular profiles of a q-space image, binned from its pixels.

The Profile dock's curves. A radial profile is intensity against |q|,
taken over the angular extent of a band (the selected box); an angular
profile is intensity against angle, taken over the band's |q| extent.
That is what silx's profile tools do on an image (``silx.gui.plot.tools.
profile.core._alignedFullProfile``: ``nanmean`` -- or ``nansum`` -- of
every pixel row inside the profile width), adapted to a polar box.

Every finite pixel of the converted Cartesian (q_xy, q_z) image inside
the band contributes exactly once, at its own exact |q| and angle, to
the bin it falls in; each bin reports the mean of its pixels (silx's
default ``"mean"`` method; a sum would only rescale each point by its
pixel count). Bins are one image pixel wide: the |q| bin width is the
image's q step, the angle bin width the angle one pixel of arc spans at
the band's radius. Empty bins are NaN, so the curve gaps there.

This replaced averaging a resampled polar grid (``polar.
cartesian_to_polar``, 1024 x 512): its |q| step was ~2.4 image pixels and
its angle step ~3 pixels of arc at 2 1/A, so its bilinear samples
skipped a large share of the pixels in a box and the profiles came out
rough. The polar grid still serves the polar image display and the
cursor readout.

Pure numpy; no Qt, no pygid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Angle bin width limits (degrees). The width is one pixel of arc at the
# band's radius, which grows without bound towards |q| = 0 and shrinks
# to nothing far out; the clamp keeps the bin count sane either way.
MIN_ANGLE_STEP_DEG = 0.01
MAX_ANGLE_STEP_DEG = 2.0


@dataclass
class PixelGeometry:
    """Per-pixel polar coordinates of one entry's Cartesian frames.

    Shared by every frame of the entry (the q axes do not change), so it
    is computed once. Arrays have the frame's shape ``(n_qz, n_qxy)``.
    """

    radius: np.ndarray       # |q| of each pixel centre, 1/A (float32)
    angle: np.ndarray        # atan2(q_z, q_xy) in degrees (float32)
    radial_bin: np.ndarray   # floor(|q| / q_step), int32
    q_step: float            # |q| bin width = the image's q pixel size
    r_max: float             # largest |q| in the image
    angle_min: float
    angle_max: float

    @property
    def n_radial_bins(self) -> int:
        return int(self.radial_bin.max()) + 1 if self.radial_bin.size else 0


def pixel_geometry(q_xy: np.ndarray, q_z: np.ndarray) -> PixelGeometry:
    """Polar coordinates of every pixel of a ``(q_z, q_xy)`` image.

    Same angle convention as ``polar.cartesian_to_polar``: 0 deg along
    +q_xy, 90 deg along +q_z. The q axes may be decreasing or negative.
    """
    q_xy = np.asarray(q_xy, dtype=np.float64)
    q_z = np.asarray(q_z, dtype=np.float64)
    if q_xy.size < 2 or q_z.size < 2:
        raise ValueError("q_xy and q_z need at least 2 samples each")
    qxy, qz = np.meshgrid(q_xy, q_z)
    radius = np.hypot(qxy, qz)
    angle = np.degrees(np.arctan2(qz, qxy))
    step = float(min(abs(q_xy[1] - q_xy[0]), abs(q_z[1] - q_z[0])))
    if not (math.isfinite(step) and step > 0):
        raise ValueError("q axes must have a non-zero, finite step")
    return PixelGeometry(
        radius=radius.astype(np.float32),
        angle=angle.astype(np.float32),
        radial_bin=np.floor(radius / step).astype(np.int32),
        q_step=step,
        r_max=float(radius.max()),
        angle_min=float(angle.min()),
        angle_max=float(angle.max()),
    )


def angle_step_deg(geom: PixelGeometry, at_radius: float) -> float:
    """Angle bin width: one image pixel of arc at ``at_radius``."""
    r = float(at_radius)
    if not (math.isfinite(r) and r > 0):
        r = geom.r_max / 2.0
    step = math.degrees(geom.q_step / max(r, geom.q_step))
    return min(max(step, MIN_ANGLE_STEP_DEG), MAX_ANGLE_STEP_DEG)


def _in_range(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``lo <= values <= hi``; a non-finite bound is that axis end."""
    keep = np.ones(values.shape, dtype=bool)
    if math.isfinite(lo):
        keep &= values >= lo
    if math.isfinite(hi):
        keep &= values <= hi
    return keep


def _bin_mean(
    bins: np.ndarray, weights: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Mean of ``weights`` per bin index; NaN where a bin is empty."""
    sums = np.bincount(bins, weights=weights, minlength=n_bins)[:n_bins]
    counts = np.bincount(bins, minlength=n_bins)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def band_pixels(
    image: np.ndarray, geom: PixelGeometry, *,
    r_lo: float = -math.inf, r_hi: float = math.inf,
    a_lo: float = -math.inf, a_hi: float = math.inf,
) -> np.ndarray:
    """Flat indices of the finite pixels inside a |q| x angle band."""
    keep = np.isfinite(image)
    keep &= _in_range(geom.radius, r_lo, r_hi)
    keep &= _in_range(geom.angle, a_lo, a_hi)
    return np.flatnonzero(keep)


def radial_profile(
    image: np.ndarray, geom: PixelGeometry, pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity per |q| bin over ``pixels`` (flat indices).

    Returns ``(bin centres, means)`` covering 0 .. r_max, so the curve
    spans the whole |q| range whatever the band.
    """
    n = geom.n_radial_bins
    centres = (np.arange(n) + 0.5) * geom.q_step
    flat = image.reshape(-1)
    means = _bin_mean(
        geom.radial_bin.reshape(-1)[pixels],
        flat[pixels].astype(np.float64), n,
    )
    return centres, means


def angular_profile(
    image: np.ndarray, geom: PixelGeometry, pixels: np.ndarray,
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity per angle bin over ``pixels`` (flat indices).

    Bins of ``step_deg`` cover the image's whole angle range.
    """
    lo = geom.angle_min
    n = max(int(math.floor((geom.angle_max - lo) / step_deg)) + 1, 1)
    centres = lo + (np.arange(n) + 0.5) * step_deg
    ang = geom.angle.reshape(-1)[pixels].astype(np.float64)
    bins = np.clip(np.floor((ang - lo) / step_deg).astype(np.int64), 0, n - 1)
    means = _bin_mean(bins, image.reshape(-1)[pixels].astype(np.float64), n)
    return centres, means
