"""Reference-value test for ``polar.polar_to_qxyz`` (physics-audit F-01 + F-03).

Pins the GIWAXS polar->Cartesian convention against the upstream
reference at ``project_repos/pygid/pygid/coordmaps.py:1005-1036``:

    Q_xy = Q_pol * cos(deg2rad(ang_pol))
    Q_z  = Q_pol * sin(deg2rad(ang_pol))

with angle measured from +q_xy toward +q_z. A future edit that
silently aligned the helper to ``coordmaps.py:331`` (lab-frame
``phi = 180 - atan2(...) * 180/pi``) would flip / shift every polar
peak; the canonical-angle assertions in this file would trip first
and surface the audit reference in the failure output.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.polar import polar_to_qxyz


def test_canonical_angles_scalar():
    """Hand-tabulated (radius, angle) -> (q_xy, q_z) at 0/90/180/270 deg.

    These are the values pygid coordmaps.py:1034-1036 produces for the
    same inputs; they are the convention's load-bearing fixed points.
    """
    r = 1.5
    cases = [
        (0.0,   r,    0.0),    # along +q_xy
        (90.0,  0.0,  r),      # along +q_z
        (180.0, -r,   0.0),    # along -q_xy
        (270.0, 0.0,  -r),     # along -q_z
    ]
    for angle_deg, q_xy_ref, q_z_ref in cases:
        q_xy, q_z = polar_to_qxyz(r, angle_deg)
        assert q_xy == pytest.approx(q_xy_ref, abs=1e-12), (
            f"q_xy at angle={angle_deg} disagrees with pygid "
            f"coordmaps.py:1036 convention"
        )
        assert q_z == pytest.approx(q_z_ref, abs=1e-12), (
            f"q_z at angle={angle_deg} disagrees with pygid "
            f"coordmaps.py:1036 convention"
        )


def test_canonical_45_degrees():
    """A non-trivial angle (45 deg) pins the convention away from the
    quadrant axes — guards against a future edit that uses the wrong
    function on the right corners (cos/sin swap)."""
    r = 2.0
    q_xy, q_z = polar_to_qxyz(r, 45.0)
    # cos(45) = sin(45) = sqrt(2)/2
    expected = r * (2 ** 0.5) / 2
    assert q_xy == pytest.approx(expected, abs=1e-12)
    assert q_z == pytest.approx(expected, abs=1e-12)


def test_array_broadcast():
    """The helper must accept numpy arrays via broadcasting — the
    seven migrated call sites mix scalar and array inputs."""
    radii = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.0, 90.0, 45.0])
    q_xy, q_z = polar_to_qxyz(radii, angles)
    assert q_xy.shape == (3,)
    assert q_z.shape == (3,)
    assert q_xy[0] == pytest.approx(1.0, abs=1e-12)
    assert q_z[0] == pytest.approx(0.0, abs=1e-12)
    assert q_xy[1] == pytest.approx(0.0, abs=1e-12)
    assert q_z[1] == pytest.approx(2.0, abs=1e-12)
    assert q_xy[2] == pytest.approx(3.0 * (2 ** 0.5) / 2, abs=1e-12)
    assert q_z[2] == pytest.approx(3.0 * (2 ** 0.5) / 2, abs=1e-12)


def test_roundtrip_against_reference_formula():
    """Cross-check against the literal formula written in pygid
    coordmaps.py:1034-1036 — a hand-rewritten copy here makes the
    audit reference explicit in the test source."""
    rng = np.random.default_rng(20260526)
    radii = rng.uniform(0.0, 5.0, size=64)
    angles_deg = rng.uniform(-180.0, 180.0, size=64)

    q_xy, q_z = polar_to_qxyz(radii, angles_deg)

    # Verbatim from coordmaps.py:1034-1036 (Q_pol = radius, ang_pol = angle_deg):
    ang_pol_rad = np.deg2rad(angles_deg)
    q_xy_ref = radii * np.cos(ang_pol_rad)
    q_z_ref = radii * np.sin(ang_pol_rad)

    np.testing.assert_allclose(q_xy, q_xy_ref, atol=1e-12)
    np.testing.assert_allclose(q_z, q_z_ref, atol=1e-12)
