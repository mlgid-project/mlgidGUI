"""Cartesian ↔ polar resampling for q-space GIWAXS images.

Pure numpy/scipy — no Qt, no pygid. Convention: angle in degrees from the q_xy
(sample-plane) axis toward q_z (out-of-plane), so 0° = along q_xy, 90° = along q_z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class PolarImage:
    image: np.ndarray  # shape (n_radius, n_angle)
    radius: np.ndarray  # (n_radius,) Å⁻¹
    angle: np.ndarray   # (n_angle,) degrees


def polar_to_qxyz(radius, angle_deg):
    """Map polar ``(radius, angle_deg)`` to Cartesian ``(q_xy, q_z)``.

    Canonical reference: pygid GIWAXS polar-grid layout at
    ``project_repos/pygid/pygid/coordmaps.py:1005-1036``, which defines

        Q_xy = Q_pol * cos(deg2rad(ang_pol))
        Q_z  = Q_pol * sin(deg2rad(ang_pol))

    with ``Q_pol`` the q magnitude and ``ang_pol`` in degrees,
    measured from +q_xy toward +q_z. Every persisted
    polar↔Cartesian conversion in this codebase MUST go through this
    helper. Do NOT align to ``coordmaps.py:331``
    (``phi = 180 - atan2(...) * 180/pi``); that is the lab-frame
    azimuth, a different quantity in a different frame, and using
    it here would silently invert / shift every polar peak.

    Inputs accept scalars or numpy arrays (broadcasting follows numpy
    rules). Returns ``(q_xy, q_z)`` with the input's broadcast shape.
    Closes physics-audit finding F-01 (single source of truth for
    seven previously-duplicated inline sites) and F-03 (convention
    pinned to upstream by name, not just by prose comment).
    """
    a = np.deg2rad(angle_deg)
    return radius * np.cos(a), radius * np.sin(a)


def _polar_extent(
    q_xy: np.ndarray, q_z: np.ndarray
) -> tuple[float, float, float]:
    """Compute the (radius_max, angle_min_deg, angle_max_deg) covered.

    The polar grid must span the actual data box, not just the
    upper-right quadrant: pygid emits converted images whose
    (q_xy, q_z) box can sit in any quadrant or straddle multiple
    quadrants depending on the ``vert_positive`` / ``hor_positive``
    conversion flags. Sampling outside that box drops data the user
    expects to see (e.g. negative-q_xy peaks vanish from the polar
    view entirely).

    The angular convention matches pygid: ``angle = atan2(q_z, q_xy)``
    in degrees, with 0° along +q_xy and 90° along +q_z. Atan2 returns
    a value in ``(-180, 180]``; for the data ranges produced by
    pygid's ``vert_positive`` / ``hor_positive`` toggles, the four
    corners always lie in a contiguous arc, so simple ``min`` / ``max``
    of corner angles gives the correct grid extent.

    Pygid's own conventions on those toggles
    (``project_repos/pygid/pygid/coordmaps.py:309-315``):

    - ``vert_positive=True``  →  ang in [0°,  180°]
    - ``hor_positive=True``   →  ang in [-90°, 90°]
    - both                    →  ang in [0°,  90°]   (upper-right)
    - neither                 →  full angular range from corners

    The corner-based detection here yields each of those automatically.
    """
    qxy = np.asarray(q_xy, dtype=float)
    qz = np.asarray(q_z, dtype=float)
    qxy_lo, qxy_hi = float(qxy.min()), float(qxy.max())
    qz_lo, qz_hi = float(qz.min()), float(qz.max())
    corners = (
        (qxy_lo, qz_lo),
        (qxy_lo, qz_hi),
        (qxy_hi, qz_lo),
        (qxy_hi, qz_hi),
    )
    radius_max = max(float(np.hypot(xy, z)) for xy, z in corners)
    angles_deg = [
        float(np.rad2deg(np.arctan2(z, xy)))
        for xy, z in corners
        if not (xy == 0.0 and z == 0.0)
    ]
    if not angles_deg:
        # Degenerate: the data box collapses to the origin. Fall
        # back to the upper-right quadrant so callers still get a
        # sensible (empty) polar grid.
        return radius_max, 0.0, 90.0
    return radius_max, min(angles_deg), max(angles_deg)


def cartesian_to_polar(
    image: np.ndarray,
    q_xy: np.ndarray,
    q_z: np.ndarray,
    n_radius: int = 1024,
    n_angle: int = 512,
) -> PolarImage:
    """Resample a Cartesian (q_z, q_xy)-indexed image onto a polar (radius, angle) grid.

    The polar grid spans the actual angular extent of the input data
    box (not just the upper-right quadrant), so converted images that
    sit in any combination of quadrants — including those produced
    by pygid with ``vert_positive`` / ``hor_positive`` left off —
    render in full. The Cartesian-to-polar mapping uses signed
    ``Q_xy`` / ``Q_z``, so the q_xy / q_z arrays may be increasing,
    decreasing, positive, negative, or mixed without any pre-flipping.

    Default resolution matches pygidfit's pipeline polar grid: in
    ``pygidfit.process_scans._get_polar_grid`` the hardcoded
    ``polar_shape=[512, 1024]`` corresponds to ``[n_phi=512,
    n_radius=1024]`` (axis 0 = angle, axis 1 = radius — see
    ``_data2container`` for the scaling). Matching the per-axis
    sample counts here removes the residual centroid drift between
    pygidfit's 2D fit and mlgidGUI's 1D-projected integration that
    we used to see on the 2D live preview. Note that mlgidGUI's
    own image is shape ``(n_radius, n_angle)`` — opposite array
    layout from pygidfit's — but the interpolation density per
    physical axis is now identical.
    """
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    if q_xy.size < 2 or q_z.size < 2:
        raise ValueError(
            f"q_xy and q_z must each have at least 2 samples; got "
            f"q_xy.size={q_xy.size}, q_z.size={q_z.size}"
        )

    radius_max, angle_min, angle_max = _polar_extent(q_xy, q_z)
    radius = np.linspace(0.0, radius_max, n_radius)
    angle = np.linspace(angle_min, angle_max, n_angle)

    R, A = np.meshgrid(radius, np.deg2rad(angle), indexing="ij")
    Q_xy = R * np.cos(A)
    Q_z = R * np.sin(A)

    # ``dx`` may be negative when q_xy is decreasing; the formula
    # ``(target - origin) / step`` recovers the correct fractional
    # column index either way (the negative step cancels).
    dx = float(q_xy[1] - q_xy[0])
    dy = float(q_z[1] - q_z[0])
    col = (Q_xy - float(q_xy[0])) / dx
    row = (Q_z - float(q_z[0])) / dy

    # Out-of-box samples (polar grid points the Cartesian image does not
    # cover) are filled with NaN, not 0. pygid already writes masked /
    # uncovered detector pixels as NaN, so keeping the same sentinel here
    # means "no data" is a single consistent value end to end. The viewer
    # renders NaN as transparent and computes levels / log over finite
    # pixels only; filling with 0 instead would paint these regions as a
    # solid colormap-bottom (black) block, inconsistent with the NaN
    # regions in the same frame. order=1 still smears NaN one pixel along
    # mask/box edges, which is acceptable for display.
    polar = map_coordinates(
        image.astype(np.float32, copy=False),
        [row, col],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return PolarImage(image=polar, radius=radius, angle=angle)
