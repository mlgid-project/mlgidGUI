"""Profiles binned from the image pixels, the way silx builds them.

Every finite pixel in the band counts exactly once, at its own |q| and
angle, and each bin is the mean of its pixels; bins are one image pixel
wide. The old path averaged a resampled 1024 x 512 polar grid whose
steps were wider than a pixel, so its samples skipped pixels and the
profiles were rough. Source: profile_binning.py; profile_viewer.py
``_recompute_curves``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mlgidgui import profile_binning as pb

Q_XY = np.linspace(0.0, 3.0, 1501)   # 0.002 1/A pixels, like real data
Q_Z = np.linspace(0.0, 3.0, 1501)


@pytest.fixture(scope="module")
def geom():
    return pb.pixel_geometry(Q_XY, Q_Z)


def test_geometry_matches_the_pixel_grid(geom):
    assert geom.q_step == pytest.approx(0.002)
    assert geom.radius.shape == (Q_Z.size, Q_XY.size)
    # Row = q_z, column = q_xy; angle 0 along q_xy, 90 along q_z.
    assert geom.angle[0, -1] == pytest.approx(0.0)
    assert geom.angle[-1, 0] == pytest.approx(90.0)
    assert geom.radius[-1, -1] == pytest.approx(math.hypot(3.0, 3.0), rel=1e-6)
    assert geom.r_max == pytest.approx(math.hypot(3.0, 3.0), rel=1e-6)


def test_every_pixel_in_the_band_counts_once(geom):
    """One bright pixel anywhere in the band shows up with its full
    value: summed back over the bins (mean x count) it is exactly 1.
    The polar grid's bilinear samples missed such pixels."""
    rng = np.random.default_rng(1)
    a_lo, a_hi = 30.0, 36.0
    inside = np.flatnonzero(
        (geom.angle.reshape(-1) >= a_lo) & (geom.angle.reshape(-1) <= a_hi)
        & (geom.radius.reshape(-1) > 0.5)
    )
    for flat in rng.choice(inside, 25, replace=False):
        img = np.zeros(geom.radius.shape, dtype=np.float32)
        img.reshape(-1)[flat] = 1.0
        px = pb.band_pixels(img, geom, a_lo=a_lo, a_hi=a_hi)
        _, means = pb.radial_profile(img, geom, px)
        counts = np.bincount(
            geom.radial_bin.reshape(-1)[px], minlength=means.size,
        )[: means.size]
        assert np.nansum(np.nan_to_num(means) * counts) == pytest.approx(1.0)


def test_bins_are_the_mean_of_their_pixels(geom):
    rng = np.random.default_rng(2)
    img = rng.random(geom.radius.shape).astype(np.float32)
    px = pb.band_pixels(img, geom, r_lo=1.0, r_hi=1.05)
    step = pb.angle_step_deg(geom, 1.025)
    centres, means = pb.angular_profile(img, geom, px, step)
    ang = geom.angle.reshape(-1)[px]
    vals = img.reshape(-1)[px]
    for k in (10, len(centres) // 2, len(centres) - 10):
        in_bin = np.floor((ang - geom.angle_min) / step).astype(int) == k
        assert means[k] == pytest.approx(vals[in_bin].mean(), rel=1e-5)


def test_bin_widths_are_one_pixel(geom):
    centres, _ = pb.radial_profile(
        np.ones(geom.radius.shape, np.float32), geom,
        np.arange(geom.radius.size),
    )
    assert np.diff(centres) == pytest.approx(geom.q_step)
    # One pixel of arc at |q| = 2: 0.002 / 2 rad.
    assert pb.angle_step_deg(geom, 2.0) == pytest.approx(math.degrees(0.001))
    # Clamped near the origin and far out.
    assert pb.angle_step_deg(geom, 1e-6) == pb.MAX_ANGLE_STEP_DEG
    assert pb.angle_step_deg(geom, 1e6) == pb.MIN_ANGLE_STEP_DEG


def test_nan_pixels_drop_out_and_empty_bins_are_nan(geom):
    img = np.full(geom.radius.shape, 7.0, dtype=np.float32)
    img[:, :200] = np.nan                       # a detector gap
    px = pb.band_pixels(img, geom)
    assert np.isfinite(img.reshape(-1)[px]).all()
    centres, means = pb.radial_profile(img, geom, px)
    assert np.nanmax(means) == pytest.approx(7.0)
    # A band with no pixels at small |q| leaves those bins empty.
    px = pb.band_pixels(img, geom, r_lo=1.0, r_hi=2.0)
    _, means = pb.radial_profile(img, geom, px)
    assert np.isnan(means[centres < 0.9]).all()


def test_profile_is_smoother_than_the_old_polar_grid(geom):
    """The point of the change: on a noisy image the binned profile
    uses far more of the data than averaging the 1024 x 512 polar grid
    did. Compared at equal |q| resolution (noise x sqrt(bin width)),
    since a bin half as wide holds half the pixels: the old grid's
    samples skipped pixels, so its noise per unit resolution is
    higher."""
    from mlgidgui.polar import cartesian_to_polar

    rng = np.random.default_rng(3)
    truth = 50.0 + 400.0 * np.exp(-((geom.radius - 2.0) ** 2) / (2 * 0.03**2))
    img = rng.poisson(truth).astype(np.float32)
    a_lo, a_hi = 57.0, 63.0

    px = pb.band_pixels(img, geom, a_lo=a_lo, a_hi=a_hi)
    r_new, y_new = pb.radial_profile(img, geom, px)
    pol = cartesian_to_polar(img, Q_XY, Q_Z)
    cols = (pol.angle >= a_lo) & (pol.angle <= a_hi)
    with warnings.catch_warnings():
        # Polar rows outside the image are all-NaN by design.
        warnings.simplefilter("ignore", RuntimeWarning)
        y_old = np.nanmean(pol.image[:, cols], axis=1)
    r_old = pol.radius

    def rel_error(r, y):
        sel = (r > 1.5) & (r < 2.5) & np.isfinite(y)
        model = 50.0 + 400.0 * np.exp(-((r[sel] - 2.0) ** 2) / (2 * 0.03**2))
        return np.std((y[sel] - model) / np.sqrt(model)) * np.sqrt(r[1] - r[0])

    assert rel_error(r_new, y_new) < 0.75 * rel_error(r_old, y_old)
