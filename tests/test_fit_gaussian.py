"""Coverage of ``fit.fit_gaussian_on_axis`` — pure scipy/numpy, no Qt.

Recovery is asserted on a synthetic Gaussian + linear background with
fixed-seed noise; only the *deterministic* None guards are asserted
(scipy non-convergence is matrix-flaky and never asserted). Source:
``mlgidgui/fit.py`` (``fit_gaussian_on_axis`` and its helpers).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.fit import fit_gaussian_on_axis

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _synthetic(center, sigma, amplitude, slope, intercept, *, n=200, noise=0.0):
    axis = np.linspace(-10.0, 10.0, n)
    clean = (
        amplitude * np.exp(-((axis - center) ** 2) / (2.0 * sigma**2))
        + slope * axis
        + intercept
    )
    if noise:
        clean = clean + np.random.default_rng(42).normal(0.0, noise, size=n)
    return axis, clean


@pytest.mark.parametrize("center", [-3.0, 0.0, 2.5])
def test_recovers_parameters(center):
    sigma, amplitude, slope, intercept = 1.2, 50.0, 0.7, 3.0
    axis, data = _synthetic(
        center, sigma, amplitude, slope, intercept, noise=0.5
    )
    fit = fit_gaussian_on_axis(axis, data, center_init=center, width_init=2.0)
    assert fit is not None
    assert fit.center == pytest.approx(center, abs=0.2)
    assert fit.sigma == pytest.approx(sigma, rel=0.15)
    assert fit.amplitude == pytest.approx(amplitude, rel=0.15)
    assert fit.slope == pytest.approx(slope, abs=0.3)
    assert fit.intercept == pytest.approx(intercept, abs=1.5)


def test_fwhm_property():
    axis, data = _synthetic(0.0, 1.5, 40.0, 0.0, 2.0)
    fit = fit_gaussian_on_axis(axis, data, center_init=0.0, width_init=2.5)
    assert fit is not None
    assert fit.fwhm == pytest.approx(_FWHM_K * fit.sigma)


def test_none_short_axis():
    axis = np.linspace(0.0, 1.0, 5)  # len < 8
    assert fit_gaussian_on_axis(axis, axis, 0.5, 1.0) is None


def test_none_nonpositive_width():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, 0.0, 0.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, -1.0) is None


def test_none_nonfinite_init():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, np.nan, 1.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, np.inf) is None


def test_none_inverted_fit_range():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert (
        fit_gaussian_on_axis(axis, data, 0.0, 1.0, fit_range=(5.0, -5.0))
        is None
    )


def test_none_too_few_samples_in_window():
    # A 10-point axis spanning a huge range with a tiny fit window:
    # fewer than 6 samples fall inside → None.
    axis = np.linspace(0.0, 1000.0, 10)
    data = np.zeros(10)
    assert (
        fit_gaussian_on_axis(
            axis, data, center_init=500.0, width_init=0.01,
            fit_range=(499.9, 500.1),
        )
        is None
    )


# -- Background pinned by the flanks, peak region only bounds the centre --
#
# Both behaviours below were user-reported failures of the old
# five-free-parameter fit inside the box: narrow boxes got no fit at
# all, and the ones that did converge invented a background. The
# synthetic here uses the real radial grid density (1024 samples over
# a 3 x 3 A^-1 q range, so ~0.0041 A^-1 per sample) because the sample
# count inside the box is exactly what used to decide the outcome.

_R_AXIS = np.linspace(0.0, float(np.sqrt(18.0)), 1024)
_R_CENTRE, _R_SIGMA, _R_AMP = 1.5, 0.012, 100.0
_R_SLOPE, _R_INTERCEPT = -5.0, 50.0


def _radial_profile(noise=2.0, extra=None, seed=0):
    """Gaussian on a sloped background, on the real radial grid.
    ``extra`` adds further ``(centre, sigma, amplitude)`` peaks."""
    y = (
        _R_AMP * np.exp(-((_R_AXIS - _R_CENTRE) ** 2) / (2.0 * _R_SIGMA ** 2))
        + _R_SLOPE * _R_AXIS + _R_INTERCEPT
    )
    for centre, sigma, amp in (extra or ()):
        y = y + amp * np.exp(-((_R_AXIS - centre) ** 2) / (2.0 * sigma ** 2))
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.shape)
    return y


def _box(width):
    return (_R_CENTRE - width / 2.0, _R_CENTRE + width / 2.0)


def _true_background_at_centre():
    return _R_SLOPE * _R_CENTRE + _R_INTERCEPT


@pytest.mark.parametrize("width", [0.020, 0.025, 0.030, 0.050, 0.080])
def test_narrow_boxes_still_get_a_fit(width):
    """A 0.020 A^-1 box holds five samples of the radial axis. The old
    fit needed six *inside the box* and returned None; the window is
    now four times the box, so the sample count stops deciding."""
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=_box(width),
    )
    assert fit is not None
    assert fit.amplitude == pytest.approx(_R_AMP, rel=0.1)
    assert fit.sigma == pytest.approx(_R_SIGMA, rel=0.15)


@pytest.mark.parametrize("width", [0.025, 0.050, 0.080])
def test_the_background_lands_on_the_real_baseline(width):
    """The background is estimated from the padding around the box, so
    it has to agree with the data's own baseline under the peak. The
    old fit, with the line free inside a window that was all peak,
    returned slope +393 / intercept -489 here against a true -5 / +50."""
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=_box(width),
    )
    assert fit is not None
    bg_at_centre = fit.slope * _R_CENTRE + fit.intercept
    assert bg_at_centre == pytest.approx(_true_background_at_centre(), abs=8.0)


def test_a_neighbour_in_the_padding_does_not_lift_the_background():
    """The padding can contain another peak. A least-squares line
    through it would ride up on that peak and steal amplitude from the
    target; the lower-envelope baseline drops those points."""
    width = 0.05
    lo, hi = _box(width)
    neighbour_centre = hi + 0.04  # inside the padding (1.5 x box each side)
    data = _radial_profile(extra=[(neighbour_centre, 0.012, 600.0)])
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=(lo, hi),
    )
    assert fit is not None
    bg_at_centre = fit.slope * _R_CENTRE + fit.intercept
    assert bg_at_centre == pytest.approx(_true_background_at_centre(), abs=15.0)
    assert fit.amplitude == pytest.approx(_R_AMP, rel=0.2)
    # The centre is bounded to the box, so the fit cannot walk onto the
    # brighter neighbour however close it sits.
    assert lo <= fit.center <= hi


def test_the_rendered_curve_never_leaves_the_fit_window():
    """A background line extrapolated past the data that constrained it
    is what used to send the overlay off the plot. ``render_range`` is
    clipped to the window the fit actually saw."""
    width = 0.05
    lo, hi = _box(width)
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=(lo, hi),
        render_range=(lo - 3.0 * width, hi + 3.0 * width),
    )
    assert fit is not None
    pad = 1.5 * width  # BG_WINDOW_FACTOR
    assert fit.x[0] >= lo - pad - 1e-9
    assert fit.x[-1] <= hi + pad + 1e-9
    # And the curve stays in the same order of magnitude as the data.
    assert fit.y.min() > data.min() - abs(_R_AMP)
    assert fit.y.max() < data.max() + abs(_R_AMP)


def test_the_baseline_takes_a_real_tilt_but_not_a_contaminated_flank():
    """Unit check on the flank rule. A tilted background raises one
    flank's level while both keep a similar spread, so the tilt is
    used. A peak sitting in one flank raises its level *and* its
    spread, so that flank is not trusted and the background goes flat
    at the lower level."""
    from mlgidgui.fit import _flank_baseline

    x = np.linspace(0.0, 10.0, 120)
    lo, hi = 4.0, 6.0

    tilted = 2.0 * x + 5.0
    slope, intercept = _flank_baseline(x, tilted, lo, hi)
    assert slope == pytest.approx(2.0, rel=0.1)
    assert slope * 5.0 + intercept == pytest.approx(15.0, abs=1.0)

    contaminated = np.full_like(x, 5.0)
    contaminated[x > 8.0] += 400.0
    slope, intercept = _flank_baseline(x, contaminated, lo, hi)
    assert slope == 0.0
    assert intercept == pytest.approx(5.0, abs=0.5)


def test_a_window_with_no_flanks_falls_back_to_a_flat_baseline():
    """When the peak region covers the whole window there is nothing to
    estimate a tilt from, and a tilt nothing constrains is exactly the
    bug this replaced. The baseline goes flat under the data."""
    from mlgidgui.fit import _flank_baseline

    x = np.linspace(0.0, 1.0, 64)
    y = 10.0 * np.exp(-((x - 0.5) ** 2) / (2.0 * 0.05 ** 2)) + 3.0
    slope, intercept = _flank_baseline(x, y, -1.0, 2.0)
    assert slope == 0.0
    assert intercept == pytest.approx(float(y.min()))


# -- gaussian_at_stored_shape: the stored row, drawn as stored ---------


def test_stored_shape_keeps_stored_centre_and_width():
    """The curve is the stored Gaussian whatever the data say: centre
    and sigma come back verbatim (stored width is 2 sigma)."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.2, 50.0, 0.0, 3.0)
    # Stored values deliberately off the real peak (centre 0, 2 sigma 2.4).
    fit = gaussian_at_stored_shape(axis, data, center=2.0, width_2sigma=1.0)
    assert fit is not None
    assert fit.center == 2.0
    assert fit.sigma == pytest.approx(0.5)
    # Off the peak, so it is not dragged onto it as a refit would be.
    assert fit.x[np.argmax(fit.y)] == pytest.approx(2.0, abs=0.1)


def test_stored_shape_maximum_is_the_profile_value_at_the_centre():
    """The curve's maximum sits at the stored centre and equals the
    profile there above the background."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(-1.0, 1.0, 40.0, 0.5, 4.0)
    fit = gaussian_at_stored_shape(axis, data, center=-1.0, width_2sigma=2.0)
    assert fit is not None
    assert fit.amplitude == pytest.approx(40.0, rel=0.05)
    assert fit.slope == pytest.approx(0.5, abs=0.15)
    assert fit.x[np.argmax(fit.y - (fit.slope * fit.x + fit.intercept))] == \
        pytest.approx(-1.0, abs=0.1)


def test_stored_shape_off_the_peak_gets_a_lower_maximum():
    """A stored centre two sigma off the real peak reads the profile
    where it is low, so the curve is visibly lower than the data."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    on = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    off = gaussian_at_stored_shape(axis, data, center=2.0, width_2sigma=2.0)
    assert on is not None and off is not None
    assert off.amplitude < 0.25 * on.amplitude


def test_stored_shape_fwhm_follows_the_stored_width():
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    fit = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    assert fit is not None
    assert fit.fwhm == pytest.approx(_FWHM_K * 1.0)   # sigma = width / 2


def test_stored_shape_amplitude_is_floored_at_zero():
    from mlgidgui.fit import gaussian_at_stored_shape

    axis = np.linspace(-10.0, 10.0, 200)
    data = np.full_like(axis, 5.0)
    data[np.abs(axis) < 1.0] = 0.0          # a dip where the stored peak sits
    fit = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    assert fit is not None
    assert fit.amplitude == 0.0


@pytest.mark.parametrize(
    "center,width", [(np.nan, 1.0), (0.0, np.nan), (0.0, 0.0), (0.0, -1.0),
                     (0.0, np.inf)],
)
def test_stored_shape_rejects_unusable_stored_values(center, width):
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 0.0)
    assert gaussian_at_stored_shape(axis, data, center, width) is None


def test_stored_shape_none_when_peak_is_off_the_axis():
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 0.0)
    assert gaussian_at_stored_shape(axis, data, center=500.0, width_2sigma=1.0) is None


def test_stored_shape_height_is_not_the_single_sample_at_the_centre():
    """One noisy sample exactly at the stored centre must not set the
    curve's height: the window mean damps it."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    spike = data.copy()
    k = int(np.argmin(np.abs(axis)))
    spike[k] += 400.0                       # one hot sample at the centre
    clean = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    hot = gaussian_at_stored_shape(axis, spike, center=0.0, width_2sigma=2.0)
    assert clean is not None and hot is not None
    # A point read would jump by 400; the window (about 10 samples
    # wide here) takes a small fraction of it.
    assert hot.amplitude - clean.amplitude < 0.2 * 400.0


@pytest.mark.parametrize("n", [200, 60])
def test_stored_shape_height_correction_holds_on_coarse_grids(n):
    """The window grows to one sample on a coarse grid; the Gaussian
    correction keeps a matching peak at its true height."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0, n=n)
    fit = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    assert fit is not None
    assert fit.amplitude == pytest.approx(40.0, rel=0.08)


# -- masked (NaN) samples ---------------------------------------------


def _masked(data, axis, lo, hi):
    out = data.copy()
    out[(axis >= lo) & (axis <= hi)] = np.nan
    return out


def test_stored_shape_height_ignores_a_masked_stretch_beside_the_centre():
    """A neighbour masked out of the profile leaves NaN next to the
    peak. The height must come from the finite samples only: the same
    as the unmasked profile, not dragged by a line bridging the gap."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    clean = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    # NaN stretch from the right shoulder outwards (the window is +-0.5).
    gap = _masked(data, axis, 0.2, 4.0)
    masked = gaussian_at_stored_shape(axis, gap, center=0.0, width_2sigma=2.0)
    assert clean is not None and masked is not None
    assert np.isfinite(masked.amplitude)
    assert masked.amplitude == pytest.approx(clean.amplitude, rel=0.08)


def test_stored_shape_is_not_biased_by_a_gap_across_the_centre():
    """With a gap through the centre only the shoulders are finite. The
    Gaussian's mean is taken at those same samples, so the height still
    matches the true peak (an interpolated bridge would read the shoulder
    value, about 0.88 of the peak, as if it were the centre)."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    masked = gaussian_at_stored_shape(
        axis, _masked(data, axis, -0.25, 0.25), center=0.0, width_2sigma=2.0,
    )
    assert masked is not None
    assert masked.amplitude == pytest.approx(40.0, rel=0.05)


def test_stored_shape_height_is_exact_for_a_wide_gap_across_the_centre():
    """The discriminating case: a gap leaving only the outer shoulders of
    the window. Bridging it with a line through the shoulder values read
    37.6 instead of 40 (about 6% low); the finite-sample mean, divided by
    the Gaussian's mean at those same samples, reads 40."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    fit = gaussian_at_stored_shape(
        axis, _masked(data, axis, -0.4, 0.4), center=0.0, width_2sigma=2.0,
    )
    assert fit is not None
    assert fit.amplitude == pytest.approx(40.0, rel=0.015)


def test_stored_shape_has_no_curve_when_the_window_is_fully_masked():
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    assert gaussian_at_stored_shape(
        axis, _masked(data, axis, -0.9, 0.9), center=0.0, width_2sigma=2.0,
    ) is None


def test_stored_shape_spacing_is_read_from_the_axis_not_the_gaps():
    """The window width is at least one sample spacing. Measured after
    dropping NaN samples, a stretch of masked samples would inflate the
    spacing and so the window; here that would swallow the shoulder far
    from the centre and bias the height."""
    from mlgidgui.fit import gaussian_at_stored_shape

    axis, data = _synthetic(0.0, 1.0, 40.0, 0.0, 3.0)
    scattered = data.copy()
    scattered[::2] = np.nan                       # every other sample masked
    fit = gaussian_at_stored_shape(axis, scattered, center=0.0, width_2sigma=2.0)
    clean = gaussian_at_stored_shape(axis, data, center=0.0, width_2sigma=2.0)
    assert fit is not None and clean is not None
    assert fit.amplitude == pytest.approx(clean.amplitude, rel=0.05)
