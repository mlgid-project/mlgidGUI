"""1D Gaussian fitting helpers for the radial / angular profile overlays.

This module's output paints the live preview curves on the profile
viewer for the active manual / detected box. It is not pygidFIT:

  * ``fit.py``        — fast 1D scipy fit, profile overlay for a
                        manual / detected selection (a forecast);
                        ``gaussian_at_stored_shape`` draws a saved
                        fitted / matched row as stored, no fit
  * ``manual_fit.py`` — slow 2D pygidfit fit, the default persisted
                        truth (physics-audit finding F-06)

The default "2D fit (pygidfit)" mode never writes anything from here.
The legacy "1D fit (scipy)" mode of Add-to-fitted does persist these
values (centre + FWHM, with the 2D shape coefficients zero-filled),
which is why the fit below has to be defensible and not merely
plausible-looking on a plot.

Pure scipy/numpy — no Qt. Returns the fit curve sampled on a fine grid plus
the fitted parameters; callers decide where to draw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

import logging
logger = logging.getLogger(__name__)

# Default window for the fit, expressed as a multiplier of the box width on
# each side of the box centre. Wider window gives the fit more context for
# background estimation but costs more CPU.
DEFAULT_FIT_WINDOW_FACTOR = 2.0
# Number of samples on the rendered fit curve. Cheap; smoother is better.
FIT_RENDER_SAMPLES = 250
# How far past the peak region the fit window reaches on each side, as a
# multiple of that region's width. This padding is the *only* data the
# background line is estimated from, so it has to be wide enough to hold
# some actual baseline and narrow enough to stay local.
BG_WINDOW_FACTOR = 1.5
# Fewest samples a fit window may hold. Below this there is nothing to
# fit; note the check is on the padded window, not on the peak region,
# which is what stopped narrow boxes from being refused outright.
MIN_WINDOW_SAMPLES = 6
# Fewest samples the peak region needs to fit three parameters. Below
# this the region grows into the window instead of the fit failing.
MIN_PEAK_SAMPLES = 4
# Fewest samples one flank needs before its background level is used.
MIN_BASELINE_SAMPLES = 4
# Percentile of a flank's samples taken as its background level. Low
# enough that a peak tail reaching into the flank barely moves it.
BG_PERCENTILE = 25.0
# How much more scattered the higher flank may be than the lower one
# before its level is treated as contamination rather than a tilt.
BG_CONTAMINATION_RATIO = 3.0


@dataclass
class GaussianFit:
    x: np.ndarray            # rendered x grid
    y: np.ndarray            # fitted curve at x
    amplitude: float
    center: float
    sigma: float
    slope: float
    intercept: float

    @property
    def fwhm(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(self.sigma))


def gaussian_with_linear_bg(
    x: np.ndarray, amplitude: float, center: float, sigma: float,
    slope: float, intercept: float,
) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) + slope * x + intercept


def gaussian_from_stored_params(
    axis: np.ndarray,
    center: float,
    fwhm: float,
    amplitude: float,
    *,
    data: np.ndarray | None = None,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Render a 1D Gaussian curve **from persisted parameters** — no
    fitting, no data lookup for the Gaussian shape.

    For a peak that already lives in ``fitted_peaks`` or
    ``matched_*``, the profile overlay shows the projection of the
    persisted 2D Gaussian onto the radial / angular axis. The
    projection of an axis-aligned 2D Gaussian along one axis is just
    the 1D Gaussian with the same centre + FWHM in that direction, so
    we sample ``amplitude * exp(-(x-center)^2 / (2*sigma^2))`` over
    ``render_range`` and return it shaped like ``GaussianFit`` so the
    profile viewer can plot it uniformly with the live preview path.

    Baseline. When ``data`` is provided alongside ``axis`` (same
    length), the curve is offset by the local minimum of the data
    inside the rendered window — so the Gaussian sits on top of the
    profile's actual baseline instead of falling to zero in the
    tails. The persisted ``amplitude`` is the 2D fit's peak height
    above local background, so this baseline reconstruction makes
    the overlay visually agree with the data. Without ``data`` we
    fall back to ``intercept=0`` (legacy behaviour).

    Closes the "profile must reflect 2D fit" half of physics-audit
    finding F-06.
    """
    if not (np.isfinite(center) and np.isfinite(fwhm)) or fwhm <= 0:
        return None
    if axis is None or len(axis) == 0:
        return None
    if not np.isfinite(amplitude):
        return None
    sigma = abs(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if render_range is not None:
        rlo = max(float(render_range[0]), float(axis[0]))
        rhi = min(float(render_range[1]), float(axis[-1]))
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi <= rlo:
            rlo, rhi = float(axis[0]), float(axis[-1])
    else:
        rlo, rhi = float(axis[0]), float(axis[-1])
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)

    # Reconstruct the local baseline from the data, when provided.
    # Take the *minimum* of the data inside the render window — the
    # persisted amplitude is peak-above-background, so the baseline
    # is what sits under the peak. Robust to noisy tails because
    # we're aggregating across the full rendered window.
    intercept = 0.0
    if data is not None and len(data) == len(axis):
        axis_arr = np.asarray(axis)
        in_window = (axis_arr >= rlo) & (axis_arr <= rhi)
        if in_window.any():
            window_y = np.asarray(data)[in_window]
            finite = np.isfinite(window_y)
            if finite.any():
                intercept = float(np.nanmin(window_y[finite]))

    y_fine = float(amplitude) * np.exp(
        -((x_fine - float(center)) ** 2) / (2.0 * sigma ** 2)
    ) + intercept
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=float(amplitude), center=float(center), sigma=sigma,
        slope=0.0, intercept=intercept,
    )


def gaussian_at_stored_shape(
    axis: np.ndarray,
    data: np.ndarray,
    center: float,
    width_2sigma: float,
    *,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Render the Gaussian a persisted row describes, over the profile.

    Nothing about the *shape* is fitted: ``center`` and ``width_2sigma``
    are the stored ``radius``/``radius_width`` (or ``angle``/
    ``angle_width``) verbatim, with ``sigma = width_2sigma / 2``
    (the stored width is pygidfit's ``2σ``, for 1D and 2D rows alike;
    the curve's FWHM is therefore ``2 sqrt(2 ln 2) * sigma``). Whatever
    fit made the row, the curve is centred where that fit put it and as
    wide as that fit said.

    Only the *height* comes from the profile, by a sample-aware rule:
    the curve's maximum sits at ``center`` and equals the mean profile
    over ``center ± max(one sample spacing, sigma / 2)`` above the
    background, divided by the Gaussian's own mean over that window.
    A single sample at the centre is noisy; a window narrower than a
    sample would be the same thing. The
    profile is the image averaged over the other axis' band, so the
    stored amplitude (the 2D peak height, or one axis' amplitude for a
    1D row) does not map onto it. The background line is read off the
    flanks exactly as in ``fit_gaussian_on_axis`` (window = peak region
    ``center ± width_2sigma`` padded by ``BG_WINDOW_FACTOR`` of its
    width each side, background from the padding only). The height is
    floored at zero. A stored centre that does not sit on the profile's
    peak therefore shows as a curve whose maximum is lower than the
    profile's own maximum nearby, not as a refit that hides it.

    Non-finite profile samples (masked pixels) are ignored: they are not
    bridged and they do not enter the height. Returns None for
    non-finite or non-positive inputs, when the window holds too few
    samples, or when fewer than two finite samples lie around the
    centre.
    """
    axis = np.asarray(axis, dtype=float)
    data = np.asarray(data, dtype=float)
    if len(axis) < 8 or len(data) != len(axis):
        return None
    if not (np.isfinite(center) and np.isfinite(width_2sigma)) or width_2sigma <= 0:
        return None
    sigma = float(width_2sigma) / 2.0

    peak_lo, peak_hi = center - width_2sigma, center + width_2sigma
    pad = BG_WINDOW_FACTOR * (peak_hi - peak_lo)
    win_lo, win_hi = peak_lo - pad, peak_hi + pad
    lo_idx = max(0, int(np.searchsorted(axis, win_lo, side="left")))
    hi_idx = min(len(axis), int(np.searchsorted(axis, win_hi, side="right")))
    if hi_idx - lo_idx < MIN_WINDOW_SAMPLES:
        return None
    x = axis[lo_idx:hi_idx]
    y = data[lo_idx:hi_idx]
    # The sample spacing is a property of the axis, so it is read before
    # the non-finite samples are dropped: measured after, a masked stretch
    # would look like a very coarse grid.
    step = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    finite = np.isfinite(y) & np.isfinite(x)
    x, y = x[finite], y[finite]
    if len(x) < MIN_WINDOW_SAMPLES:
        return None
    if not (x[0] <= center <= x[-1]):
        return None

    slope, intercept = _flank_baseline(x, y, peak_lo, peak_hi)

    # Height: the mean background-subtracted profile over a window
    # around the stored centre, not the single sample there. The window
    # is at least one sample spacing each side (so it spans real samples
    # even on a coarse grid) and at least half a sigma. Averaging a
    # Gaussian over +-h lowers it, so the mean is divided by the
    # Gaussian's own mean over the same samples: a peak that matches the
    # stored shape reads its true height, not a window-dependent one.
    #
    # Only real, finite samples enter. A masked (NaN) stretch near the
    # centre, e.g. a neighbour masked out of the profile, is skipped, not
    # bridged by a line through the samples either side of it (which would
    # invent values the data never had), and the Gaussian's mean is taken
    # at the samples actually used, so the gap cannot bias the height. A
    # window left with fewer than two finite samples carries no
    # information about this peak, so there is no curve.
    half = max(abs(step), 0.5 * sigma)
    in_window = np.abs(x - center) <= half
    if int(in_window.sum()) < 2:
        return None
    xw = x[in_window]
    residual = y[in_window] - (slope * xw + intercept)
    g = float(np.mean(np.exp(-((xw - center) ** 2) / (2.0 * sigma ** 2))))
    amplitude = max(float(np.mean(residual)) / g, 0.0) if g > 0 else 0.0
    if not np.isfinite(amplitude):
        return None

    rlo, rhi = float(x[0]), float(x[-1])
    if render_range is not None:
        want_lo = max(float(render_range[0]), rlo)
        want_hi = min(float(render_range[1]), rhi)
        if np.isfinite(want_lo) and np.isfinite(want_hi) and want_hi > want_lo:
            rlo, rhi = want_lo, want_hi
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_linear_bg(
        x_fine, amplitude, float(center), sigma, slope, intercept,
    )
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude, center=float(center), sigma=sigma,
        slope=slope, intercept=intercept,
    )


def gaussian_only(
    x: np.ndarray, amplitude: float, center: float, sigma: float,
) -> np.ndarray:
    """The peak term alone, for fits whose background is already fixed."""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def _flank_level(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Robust background level of one flank: ``(level, spread, x_of_level)``.

    The flank's lowest ``BG_PERCENTILE`` of samples are taken as the
    ones least likely to carry peak signal, and the level and position
    are the *median y and median x of that same subset*. Reading both
    off one subset is what keeps the pair consistent: taking the level
    from a percentile but the position from the whole flank tilts the
    line on a sloped background, because the low samples sit at one
    end of the flank, not in its middle.

    ``spread`` is the median absolute deviation about the level over
    the whole flank, used to tell a tilted background (both flanks
    similarly scattered) from a contaminated one (the higher flank far
    more scattered).
    """
    cut = float(np.percentile(y, BG_PERCENTILE))
    low = y <= cut
    if not bool(low.any()):
        low = np.ones(y.size, dtype=bool)
    level = float(np.median(y[low]))
    x_at = float(np.median(x[low]))
    spread = float(np.median(np.abs(y - level)))
    return level, spread, x_at


def _flank_baseline(
    x: np.ndarray, y: np.ndarray, peak_lo: float, peak_hi: float,
) -> tuple[float, float]:
    """Linear background estimated from the two flanks of the window.

    Both flanks present and consistent → the line through their levels.
    "Consistent" means the higher flank is not also the more scattered
    one: a genuinely tilted background raises one flank's *level* while
    both keep a similar spread, whereas a neighbouring peak sitting in
    one flank blows that flank's spread up as well. When the higher
    flank is more than ``BG_CONTAMINATION_RATIO`` times as scattered as
    the lower one it is not trusted for the tilt, and the background
    goes flat at the lower level — an underestimate of a real slope is
    a far smaller error than a background invented from another peak.

    One flank only → flat at its level. Neither → flat at the window
    minimum, which is the last honest statement available.
    """
    left = x < peak_lo
    right = x > peak_hi
    have_left = int(left.sum()) >= MIN_BASELINE_SAMPLES
    have_right = int(right.sum()) >= MIN_BASELINE_SAMPLES

    if have_left and have_right:
        l_level, l_spread, l_x = _flank_level(x[left], y[left])
        r_level, r_spread, r_x = _flank_level(x[right], y[right])
        higher, lower = (
            (r_spread, l_spread) if r_level >= l_level else (l_spread, r_spread)
        )
        trusted = higher <= BG_CONTAMINATION_RATIO * max(lower, 1e-12)
        if trusted and r_x > l_x:
            slope = (r_level - l_level) / (r_x - l_x)
            return float(slope), float(l_level - slope * l_x)
        return 0.0, float(min(l_level, r_level))

    if have_left or have_right:
        side = left if have_left else right
        level, _spread, _x = _flank_level(x[side], y[side])
        return 0.0, float(level)

    return 0.0, float(np.nanmin(y))


def fit_gaussian_on_axis(
    axis: np.ndarray,
    data: np.ndarray,
    center_init: float,
    width_init: float,
    *,
    window_factor: float = DEFAULT_FIT_WINDOW_FACTOR,
    fit_range: tuple[float, float] | None = None,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Fit a 1D Gaussian on a fixed linear background along ``axis``.

    ``fit_range=(lo, hi)`` names the **peak region** — normally the
    user's box. It is not the fit window: the window is that region
    padded by ``BG_WINDOW_FACTOR`` of its width on each side, and the
    background line is estimated from the padding only, where there is
    actually baseline to see. Without ``fit_range`` the window is the
    legacy ``(0.5 + window_factor) × width_init`` half-width around
    ``center_init`` and the peak region is the middle ``width_init``
    of it.

    Two problems this shape exists to avoid, both reported from real
    use:

    * **Fits that refused peaks they should have taken.** Fitting only
      inside the box meant a box narrower than about six samples of the
      profile axis had no fit at all — on the default 1024-sample
      radial grid over a 3 × 3 Å⁻¹ q range that is any box under
      ~0.025 Å⁻¹. The window is four times the box, so the sample count
      is no longer the binding constraint.
    * **Nonsensical backgrounds.** Five free parameters (amplitude,
      centre, sigma, slope, intercept) inside a window that is all
      peak leaves the line unidentifiable, and it was seeded from
      points on the peak's own flanks. Measured on a clean synthetic
      peak, a six-sample box returned slope +393 / intercept −489
      against a true −5 / +50, and lost half the amplitude with it.
      Here the background is pinned by the flanks first and held
      fixed, leaving three free parameters for the peak.

    The peak itself is fitted on the peak region alone, with the
    background held at what the flanks said and the centre bounded to
    that region. Fitting it across the whole window instead would let a
    bright neighbour in the padding inflate the amplitude — measured at
    +82% on a synthetic neighbour four times the target's height. The
    padding informs the background and nothing else.

    A peak region holding fewer than ``MIN_PEAK_SAMPLES`` grows to the
    nearest that many samples of the window rather than returning
    nothing.

    ``render_range=(lo, hi)`` samples the returned curve over a
    different range than the fit used. It is clipped to the fit window:
    the background is a line, and extrapolating it past the data that
    constrained it is how the overlay used to shoot off the plot.

    Returns None if the inputs aren't fittable or scipy can't converge.
    """
    axis = np.asarray(axis, dtype=float)
    data = np.asarray(data, dtype=float)
    if len(axis) < 8 or len(data) != len(axis):
        return None
    if not (np.isfinite(center_init) and np.isfinite(width_init)) or width_init <= 0:
        return None

    if fit_range is not None:
        peak_lo, peak_hi = float(fit_range[0]), float(fit_range[1])
        if not (np.isfinite(peak_lo) and np.isfinite(peak_hi)) or peak_hi <= peak_lo:
            return None
        pad = BG_WINDOW_FACTOR * (peak_hi - peak_lo)
        win_lo, win_hi = peak_lo - pad, peak_hi + pad
    else:
        win_half = max(
            width_init * (0.5 + window_factor), (axis[-1] - axis[0]) * 0.05,
        )
        peak_half = min(width_init / 2.0, win_half)
        peak_lo, peak_hi = center_init - peak_half, center_init + peak_half
        win_lo, win_hi = center_init - win_half, center_init + win_half

    lo_idx = max(0, int(np.searchsorted(axis, win_lo, side="left")))
    hi_idx = min(len(axis), int(np.searchsorted(axis, win_hi, side="right")))
    if hi_idx - lo_idx < MIN_WINDOW_SAMPLES:
        return None
    x = axis[lo_idx:hi_idx]
    y = data[lo_idx:hi_idx]
    finite = np.isfinite(y) & np.isfinite(x)
    x, y = x[finite], y[finite]
    if len(x) < MIN_WINDOW_SAMPLES:
        return None

    # Background from the padding only — the parts of the window the
    # peak region does not claim.
    slope, intercept = _flank_baseline(x, y, peak_lo, peak_hi)

    # The peak itself is fitted on the peak region only. Everything
    # outside it has already done its job (the background), and
    # including it would let a neighbouring peak in the padding pull
    # the amplitude up. When the region holds too few samples to fit
    # three parameters, it grows to the nearest ``MIN_PEAK_SAMPLES``
    # of the window rather than giving up, which is what the old code
    # did to every narrow box.
    peak_mask = (x >= peak_lo) & (x <= peak_hi)
    if int(peak_mask.sum()) < MIN_PEAK_SAMPLES:
        nearest = np.argsort(np.abs(x - center_init))[:MIN_PEAK_SAMPLES]
        if nearest.size < MIN_PEAK_SAMPLES:
            return None
        peak_mask = np.zeros(x.size, dtype=bool)
        peak_mask[nearest] = True
    xp = x[peak_mask]
    residual = y[peak_mask] - (slope * xp + intercept)

    fit_w = float(xp[-1] - xp[0])
    sample_step = float(x[-1] - x[0]) / max(len(x) - 1, 1)
    sigma_lo = max(sample_step * 0.5, 1e-12)
    sigma_hi = max(fit_w, peak_hi - peak_lo, sigma_lo * 2.0)
    sigma_init = float(np.clip(width_init / 2.355, sigma_lo, sigma_hi))
    center_start = float(np.clip(center_init, peak_lo, peak_hi))
    amp_init = float(np.nanmax(residual))
    if not np.isfinite(amp_init) or amp_init <= 0:
        amp_init = float(np.nanmax(y) - np.nanmin(y)) or 1.0

    try:
        popt, _ = curve_fit(
            gaussian_only, xp, residual,
            p0=[amp_init, center_start, sigma_init],
            bounds=(
                [0.0, peak_lo, sigma_lo],
                [np.inf, peak_hi, sigma_hi],
            ),
            maxfev=5000,
        )
        amplitude, center, sigma = (float(v) for v in popt)
    except Exception:
        logger.debug("suppressed exception in fit_gaussian_on_axis", exc_info=True)
        return None

    if not (np.isfinite(amplitude) and np.isfinite(center) and np.isfinite(sigma)):
        return None
    if sigma <= 0:
        return None

    rlo, rhi = float(x[0]), float(x[-1])
    if render_range is not None:
        want_lo = max(float(render_range[0]), rlo)
        want_hi = min(float(render_range[1]), rhi)
        if np.isfinite(want_lo) and np.isfinite(want_hi) and want_hi > want_lo:
            rlo, rhi = want_lo, want_hi
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_linear_bg(
        x_fine, amplitude, center, sigma, slope, intercept,
    )
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude, center=center, sigma=abs(sigma),
        slope=slope, intercept=intercept,
    )
