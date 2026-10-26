"""F-06 closure: ``manual_fit.fit_one_peak`` routes a user-drawn box
through pygidfit's 2D fit so the persisted ``fitted_peaks`` row
carries real A/B/C/theta instead of zero-fill. Plus the
profile-overlay helper that projects persisted 2D params onto the
radial / angular axis without re-fitting the data.

Two pieces tested here:

* ``fit_one_peak`` round-trip on a synthetic 2D Gaussian rendered
  into a polar frame — assert pygidfit recovers something close to
  the seeded centre, widths and amplitude. The exact A/B/C/theta
  values depend on pygidfit's internal coordinate normalisation
  (pixel coords × scale factors), so we don't pin those exactly;
  the assertion is that the fit converged with non-zero shape
  coefficients.
* ``gaussian_from_stored_params`` renders a 1D Gaussian curve with
  the expected centre + FWHM + amplitude on a given axis — no
  data fit, just evaluation.

Module is skipped on CI without pygidfit installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygidfit")  # 2D fit path is private-backend dependent

from mlgidlab.fit import gaussian_from_stored_params  # noqa: E402
from mlgidlab.manual_fit import ManualFitError, fit_one_peak  # noqa: E402


def _synthetic_cartesian_with_gaussian(
    n_qz: int = 256, n_qxy: int = 256,
    q_max: float = 3.0,
    *,
    centre_r: float = 1.5,
    centre_a: float = 45.0,
    sigma_r: float = 0.05,
    sigma_a: float = 4.0,
    amplitude: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic reciprocal-space (img_gid_q) cartesian array
    with a 2D Gaussian peak placed at polar coordinates ``(centre_r,
    centre_a)``. Shape matches what ``FrameSource.get_cartesian``
    returns: ``(n_qz, n_qxy)``. q axes span ``[0, q_max]`` so
    pygidfit's pixel-space polar conversion (which assumes the
    array origin is at the beam centre) round-trips cleanly.
    """
    q_xy = np.linspace(0.0, q_max, n_qxy, dtype=np.float32)
    q_z = np.linspace(0.0, q_max, n_qz, dtype=np.float32)
    QXY, QZ = np.meshgrid(q_xy, q_z)  # (n_qz, n_qxy)
    R = np.sqrt(QXY ** 2 + QZ ** 2)
    A = np.rad2deg(np.arctan2(QZ, QXY))
    img = amplitude * np.exp(
        -0.5 * (((R - centre_r) / sigma_r) ** 2 + ((A - centre_a) / sigma_a) ** 2)
    )
    return img.astype(np.float32), q_xy, q_z


def test_fit_one_peak_recovers_synthetic_gaussian():
    """Render a synthetic 2D Gaussian into a cartesian (img_gid_q)
    array, hand pygidfit a wide box around the true centre, and
    verify it recovers the centre + amplitude approximately. The
    wrapper applies pygidfit's own polar conversion internally so
    the synthetic doesn't have to match mlgidlab's polar resample."""
    centre_r, centre_a = 1.5, 45.0
    sigma_r, sigma_a = 0.04, 3.0
    amplitude = 150.0
    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian(
        centre_r=centre_r, centre_a=centre_a,
        sigma_r=sigma_r, sigma_a=sigma_a, amplitude=amplitude,
    )
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    result = fit_one_peak(
        cart, q_xy, q_z,
        radius=centre_r,
        radius_width=sigma_r * fwhm_factor,
        angle=centre_a,
        angle_width=sigma_a * fwhm_factor,
        wavelength_angstrom=1.0,
        q_xy_max=3.0,
        q_z_max=3.0,
    )
    # Centre recovered to within ~5% of the radial / angular widths.
    assert result.radius == pytest.approx(centre_r, abs=0.05)
    assert result.angle == pytest.approx(centre_a, abs=2.0)
    # Amplitude is finite, positive, non-zero — pygidfit's internal
    # normalisation scales the value, so we don't pin it exactly.
    assert result.amplitude > 0
    assert np.isfinite(result.amplitude)
    # 2D shape coefficients are now real, not zero-fill (this is
    # the F-06 goal). At least one of A/B/C must be non-zero.
    assert (
        abs(result.A) + abs(result.B) + abs(result.C) > 0.0
    ), "A/B/C all zero — the 2D shape was not actually fit"
    # Width conventions match pygidfit's container output verbatim,
    # which is what mlgidbase's pipeline run_fitting path also saves
    # (per ``mlgidbase/pygidfit_functions.py``). The expected stored
    # value is ``2σ`` for both radius_width and angle_width — NOT
    # FWHM and NOT 2×FWHM. Locking this in means manually-added
    # peaks and pipeline-fitted peaks share an identical width
    # convention; an earlier revision tried to convert to FWHM /
    # 2×FWHM here and broke that consistency (the same detected box
    # got a wider box from Add-to-fitted than from run_fitting).
    # The ~3% tolerance soaks up pygidfit's pixel-quantisation on a
    # synthetic peak fit at this grid resolution.
    expected_2sigma_r = 2.0 * sigma_r
    expected_2sigma_a = 2.0 * sigma_a
    assert result.radius_width == pytest.approx(expected_2sigma_r, rel=0.03), (
        f"radius_width {result.radius_width:.4f} not at 2σ "
        f"{expected_2sigma_r:.4f} — pygidfit's container value must "
        f"be stored verbatim so manual peaks match pipeline peaks."
    )
    assert result.angle_width == pytest.approx(expected_2sigma_a, rel=0.03), (
        f"angle_width {result.angle_width:.4f} not at 2σ "
        f"{expected_2sigma_a:.4f} — pygidfit's container value must "
        f"be stored verbatim so manual peaks match pipeline peaks."
    )


def test_fit_one_peak_matches_pipeline_for_same_box():
    """Lock in the byte-identical contract: calling ``fit_one_peak``
    must produce the same fit values pygidfit's
    ``ProcessDataFromFile.process_single_frame`` would for the same
    cartesian frame + same box + same config. The wrapper replicates
    the pipeline's polar conversion chain
    (``img_preprocessing`` + ``_get_polar_grid([512, 1024])`` +
    ``polar_conversion``) so the two paths converge on identical
    pixel-space input.

    Test compares the wrapper output against a direct invocation of
    the same pygidfit primitives — within float-precision tolerance."""
    from pygidfit.process_scans import (
        _get_polar_grid,
        fit_data as _pygidfit_fit_data,
        img_preprocessing,
        polar_conversion,
    )
    import cv2

    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian(
        centre_r=1.5, centre_a=45.0, sigma_r=0.04, sigma_a=3.0,
        amplitude=120.0,
    )

    # Reference: replicate ProcessDataFromFile.process_single_frame
    # inline (same code-shape the pipeline runs).
    wl = 1.0
    ai = 0.0
    crit_angle = 0.0
    img_pre = img_preprocessing(cart, ai, crit_angle, wl, q_z)
    yy, zz, ang_deg_max = _get_polar_grid(img_pre.shape, (512, 1024), [0, 0])
    polar_img = polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)
    q_abs_max = float(
        np.sqrt(np.nanmax(q_z) ** 2 + np.nanmax(q_xy) ** 2)
    )
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    container_ref, _ = _pygidfit_fit_data(
        polar_img,
        np.array([1.5]), np.array([0.04 * fwhm_factor]),
        np.array([45.0]), np.array([3.0 * fwhm_factor]),
        wl, 3.0, 3.0, q_abs_max, ang_deg_max,
        10.0, 10.0, 2, True, False, False, None, ai,
    )

    # Wrapper: same inputs, same config.
    via_wrapper = fit_one_peak(
        cart, q_xy, q_z,
        radius=1.5, radius_width=0.04 * fwhm_factor,
        angle=45.0, angle_width=3.0 * fwhm_factor,
        wavelength_angstrom=wl, q_xy_max=3.0, q_z_max=3.0,
        ai_deg=ai, crit_angle=crit_angle,
        theta_fixed=True,
        clustering_distance_peaks=10.0, clustering_distance_rings=10.0,
        clustering_extend=2,
    )

    # Compare every field within float-precision tolerance.
    assert via_wrapper.radius == pytest.approx(
        float(np.asarray(container_ref.radius).ravel()[0]), abs=1e-8
    )
    assert via_wrapper.radius_width == pytest.approx(
        float(np.asarray(container_ref.radius_width).ravel()[0]), abs=1e-8
    )
    assert via_wrapper.angle == pytest.approx(
        float(np.asarray(container_ref.angle).ravel()[0]), abs=1e-8
    )
    assert via_wrapper.angle_width == pytest.approx(
        float(np.asarray(container_ref.angle_width).ravel()[0]), abs=1e-8
    )
    assert via_wrapper.amplitude == pytest.approx(
        float(np.asarray(container_ref.amplitude).ravel()[0]), abs=1e-8
    )
    assert via_wrapper.A == pytest.approx(
        float(np.asarray(container_ref.A).ravel()[0]), abs=1e-8
    )


def test_fit_one_peak_raises_on_pygidfit_failure(monkeypatch):
    """When pygidfit raises internally, the wrapper must surface a
    ``ManualFitError`` so the caller can fall back to the legacy 1D
    + zero-fill path. The original exception chains through."""
    from mlgidlab import manual_fit
    from pygidfit import process_scans

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic pygidfit failure")

    monkeypatch.setattr(process_scans, "fit_data", _boom)
    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian()
    with pytest.raises(ManualFitError, match="pygidfit.fit_data raised"):
        manual_fit.fit_one_peak(
            cart, q_xy, q_z,
            radius=1.5, radius_width=0.1, angle=45.0, angle_width=10.0,
            wavelength_angstrom=1.0, q_xy_max=3.0, q_z_max=3.0,
        )


def test_fit_one_peak_raises_on_nan_amplitude(monkeypatch):
    """pygidfit may converge with a NaN amplitude (degenerate fit).
    The wrapper treats NaN amplitude as a non-fit and raises so the
    caller falls back rather than persisting a row with NaN values."""
    from mlgidlab import manual_fit
    from pygidfit import process_scans

    class _FakeContainer:
        amplitude = np.array([float("nan")])
        radius = np.array([1.5])
        radius_width = np.array([0.1])
        angle = np.array([45.0])
        angle_width = np.array([10.0])
        A = np.array([0.0])
        B = np.array([0.0])
        C = np.array([0.0])
        theta = np.array([0.0])

    monkeypatch.setattr(
        process_scans, "fit_data",
        lambda *a, **k: (_FakeContainer(), None),
    )
    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian()
    with pytest.raises(ManualFitError, match="NaN amplitude"):
        manual_fit.fit_one_peak(
            cart, q_xy, q_z,
            radius=1.5, radius_width=0.1, angle=45.0, angle_width=10.0,
            wavelength_angstrom=1.0, q_xy_max=3.0, q_z_max=3.0,
        )


# -- gaussian_from_stored_params (no pygidfit dependency) -------------


def test_gaussian_from_stored_params_renders_correct_peak():
    """The renderer must evaluate the Gaussian exactly: peak value at
    ``center`` equals ``amplitude``, FWHM equals the requested value,
    zero baseline."""
    axis = np.linspace(0.0, 4.0, 1000)
    result = gaussian_from_stored_params(
        axis, center=2.0, fwhm=0.4, amplitude=10.0,
        render_range=(0.0, 4.0),
    )
    assert result is not None
    # Peak value at centre. The render uses FIT_RENDER_SAMPLES=250
    # points across the render range, so the nearest sample to the
    # true centre is at most one sample step away — sampling
    # artifact bounds the peak value slightly below ``amplitude``.
    peak_idx = int(np.argmax(result.y))
    assert result.x[peak_idx] == pytest.approx(2.0, abs=0.02)
    assert result.y[peak_idx] == pytest.approx(10.0, rel=2e-3)
    # FWHM check: find the two crossings of half-max. The measurement
    # is quantised by the render's sample step (4 / 249 ≈ 0.016), so
    # the measured FWHM lands within ~2 sample steps of the true 0.4.
    half = 5.0
    above = result.y >= half
    indices = np.where(above)[0]
    fwhm_measured = result.x[indices[-1]] - result.x[indices[0]]
    assert fwhm_measured == pytest.approx(0.4, abs=0.05)
    # Tails approach zero — no baseline term.
    assert result.y[0] == pytest.approx(0.0, abs=1e-3)
    assert result.y[-1] == pytest.approx(0.0, abs=1e-3)


def test_gaussian_from_stored_params_uses_data_baseline_when_provided():
    """When ``data`` is supplied alongside ``axis``, the rendered
    Gaussian must sit on top of the local data baseline (the minimum
    of the data inside the render window). Without this, the
    persisted-peak overlay falls to zero in the tails while the
    profile sits on a non-zero baseline — visually wrong, the
    "background of zero" symptom the user reported."""
    axis = np.linspace(0.0, 4.0, 200)
    # Synthetic profile: a Gaussian peak at x=2 sitting on a
    # baseline of 7.0, with random-ish noise around the baseline
    # in the tails.
    data = 7.0 + 10.0 * np.exp(-((axis - 2.0) ** 2) / (2.0 * 0.17 ** 2))
    result = gaussian_from_stored_params(
        axis, center=2.0, fwhm=0.4, amplitude=10.0,
        data=data,
        render_range=(0.0, 4.0),
    )
    assert result is not None
    # Baseline is the data minimum inside the render window; in this
    # synthetic the tails are flat at 7.0 so intercept must land
    # near 7.0.
    assert result.intercept == pytest.approx(7.0, abs=0.05)
    # The rendered curve in the tails should sit near the baseline,
    # not zero.
    assert result.y[0] == pytest.approx(7.0, abs=0.1)
    assert result.y[-1] == pytest.approx(7.0, abs=0.1)
    # Peak rises ``amplitude`` above the baseline.
    peak_idx = int(np.argmax(result.y))
    assert result.y[peak_idx] == pytest.approx(17.0, abs=0.1)


def test_gaussian_from_stored_params_returns_none_on_bad_inputs():
    """Defensive: invalid centre/fwhm/amplitude → None so the
    profile viewer just clears the curve instead of crashing."""
    axis = np.linspace(0.0, 4.0, 100)
    assert gaussian_from_stored_params(axis, center=float("nan"), fwhm=0.1, amplitude=1.0) is None
    assert gaussian_from_stored_params(axis, center=2.0, fwhm=0.0, amplitude=1.0) is None
    assert gaussian_from_stored_params(axis, center=2.0, fwhm=-0.1, amplitude=1.0) is None
    assert gaussian_from_stored_params(axis, center=2.0, fwhm=0.1, amplitude=float("nan")) is None
    # Empty axis.
    assert gaussian_from_stored_params(np.array([]), center=2.0, fwhm=0.1, amplitude=1.0) is None


def _synthetic_two_peaks(
    n: int = 256, q_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two 2D Gaussians at the same radius, 7° apart: a weak target at
    45° and a neighbour at 52° eight times as bright. Their boxes are
    close enough that the neighbour's intensity falls inside the
    target's fit ROI (pygidfit's ROI is the box padded by
    ``clustering_extend``, and its radial half-extent is
    ``radius_width``, not ``radius_width / 2``).
    """
    q_xy = np.linspace(0.0, q_max, n, dtype=np.float32)
    q_z = np.linspace(0.0, q_max, n, dtype=np.float32)
    QXY, QZ = np.meshgrid(q_xy, q_z)
    R = np.sqrt(QXY ** 2 + QZ ** 2)
    A = np.rad2deg(np.arctan2(QZ, QXY))
    img = np.zeros_like(R)
    for centre_a, amp in ((45.0, 100.0), (52.0, 800.0)):
        img += amp * np.exp(
            -0.5 * (((R - 1.5) / 0.04) ** 2 + ((A - centre_a) / 3.0) ** 2)
        )
    return img.astype(np.float32), q_xy, q_z


def test_a_neighbour_box_keeps_the_fit_inside_the_drawn_box():
    """Regression for the user-reported "fitted peak lies outside the
    marked area".

    pygidfit masks / joint-fits only the boxes handed to one
    ``fit_data`` call, and it bounds a peak centre a quarter of the
    box outside each edge. Passing the target alone therefore let a
    bright neighbour inside the ROI drag the centre out of the drawn
    box and inflate the amplitude. Passing the neighbour switches
    pygidfit's own handling back on.

    Both halves are asserted: the naive single-box call must still
    fail this way (otherwise the test would pass for the wrong
    reason), and the neighbour-aware call must recover the target.
    """
    from mlgidlab.manual_fit import PeakBox

    cart, q_xy, q_z = _synthetic_two_peaks()
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    box = dict(
        radius=1.5, radius_width=0.04 * fwhm_factor,
        angle=45.0, angle_width=3.0 * fwhm_factor,
    )
    geom = dict(wavelength_angstrom=1.0, q_xy_max=3.0, q_z_max=3.0)
    half_angle = 0.5 * box["angle_width"]

    naive = fit_one_peak(cart, q_xy, q_z, **box, **geom)
    assert abs(naive.angle - 45.0) > half_angle, (
        "the unmasked fit no longer drifts out of the drawn box — "
        "this test's premise has changed"
    )

    with_neighbour = fit_one_peak(
        cart, q_xy, q_z, **box, **geom,
        neighbours=[PeakBox(1.5, 0.04 * fwhm_factor, 52.0, 3.0 * fwhm_factor)],
    )
    assert abs(with_neighbour.angle - 45.0) <= half_angle
    assert with_neighbour.angle == pytest.approx(45.0, abs=0.5)
    assert with_neighbour.radius == pytest.approx(1.5, abs=0.01)
    # The naive fit swallowed part of the neighbour's 800-count peak;
    # the masked one recovers the target's own 100.
    assert with_neighbour.amplitude == pytest.approx(100.0, rel=0.15)
    assert naive.amplitude > 2.0 * with_neighbour.amplitude


def test_far_neighbours_do_not_change_the_fit():
    """``select_neighbour_boxes`` drops boxes that cannot reach the
    target's ROI, so handing the wrapper a whole frame's worth of
    distant peaks must produce exactly the single-box result — no
    drift, and no extra cluster fits on every click."""
    from mlgidlab.manual_fit import PeakBox

    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian(
        centre_r=1.5, centre_a=45.0, sigma_r=0.04, sigma_a=3.0,
        amplitude=120.0,
    )
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    box = dict(
        radius=1.5, radius_width=0.04 * fwhm_factor,
        angle=45.0, angle_width=3.0 * fwhm_factor,
    )
    geom = dict(wavelength_angstrom=1.0, q_xy_max=3.0, q_z_max=3.0)
    alone = fit_one_peak(cart, q_xy, q_z, **box, **geom)
    with_far = fit_one_peak(
        cart, q_xy, q_z, **box, **geom,
        neighbours=[
            PeakBox(0.6, 0.05, 20.0, 8.0),
            PeakBox(2.6, 0.05, 70.0, 8.0),
        ],
    )
    assert with_far.radius == pytest.approx(alone.radius, abs=1e-12)
    assert with_far.angle == pytest.approx(alone.angle, abs=1e-12)
    assert with_far.amplitude == pytest.approx(alone.amplitude, abs=1e-12)


# -- the 0.1.8 stack: defaults, hot-pixel parity, N-1 box scale -------------


def test_the_shared_fit_defaults_follow_mlgidbase():
    """The Pipeline panel and every fallback take the clustering numbers
    from manual_fit; they must be what a bare run_fitting() uses."""
    import inspect

    mlgidbase = pytest.importorskip("mlgidbase")
    from mlgidlab import manual_fit

    sig = inspect.signature(mlgidbase.mlgidBASE.run_fitting)
    assert sig.parameters["clustering_distance_peaks"].default == (
        manual_fit.DEFAULT_CLUSTERING_DISTANCE_PEAKS
    )
    assert sig.parameters["clustering_distance_rings"].default == (
        manual_fit.DEFAULT_CLUSTERING_DISTANCE_RINGS
    )
    assert sig.parameters["clustering_extend"].default == (
        manual_fit.DEFAULT_CLUSTERING_EXTEND
    )


def test_the_hot_pixel_percentile_is_the_pipelines():
    """mlgidbase never passes hot_pixel_percentile, so the pipeline runs
    with fit_data's default; the manual fit passes the same value
    explicitly. If pygidfit changes its default this fails and the
    parity contract has to be re-decided."""
    import inspect

    from pygidfit.process_scans import fit_data

    from mlgidlab import manual_fit

    default = inspect.signature(fit_data).parameters[
        "hot_pixel_percentile"
    ].default
    assert manual_fit.HOT_PIXEL_PERCENTILE == default
    assert inspect.signature(fit_one_peak).parameters[
        "hot_pixel_percentile"
    ].default == default


def test_fit_one_peak_hands_the_hot_pixel_percentile_to_pygidfit(monkeypatch):
    import mlgidlab.manual_fit as mf

    seen: dict = {}
    import pygidfit.process_scans as ps

    def _spy(*a, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(ps, "fit_data", _spy)
    cart, q_xy, q_z = _synthetic_cartesian_with_gaussian(
        centre_r=1.5, centre_a=45.0, sigma_r=0.04, sigma_a=3.0,
        amplitude=120.0,
    )
    for pct in (mf.HOT_PIXEL_PERCENTILE, None):
        with pytest.raises(ManualFitError):
            fit_one_peak(
                cart, q_xy, q_z, radius=1.5, radius_width=0.1,
                angle=45.0, angle_width=7.0, wavelength_angstrom=1.0,
                q_xy_max=3.0, q_z_max=3.0, hot_pixel_percentile=pct,
            )
        assert seen["hot_pixel_percentile"] == pct


def test_px_box_matches_pygidfits_boxes_preprocessing():
    """pygidfit >= 0.1.8 scales boxes by N - 1; the GUI's pixel box
    must agree with it to the pixel."""
    from pygidfit.box_utils import boxes_preprocessing
    from pygidfit.process_scans import DetectedPeaks

    from mlgidlab.manual_fit import PeakBox, _px_box

    shape = (512, 1024)
    q_abs_max = 4.2
    for box in (
        PeakBox(1.5, 0.06, 45.0, 7.0),
        PeakBox(0.4, 0.02, 3.0, 4.0),
        PeakBox(3.9, 0.1, 87.0, 9.0),
        PeakBox(2.0, 0.05, 30.0, 12.5),
    ):
        det = DetectedPeaks(
            radius=np.array([box.radius]),
            radius_width=np.array([box.radius_width]),
            angle=np.array([box.angle]),
            angle_width=np.array([box.angle_width]),
        )
        ref = boxes_preprocessing(det, shape, 1.0, q_abs_max, 3.0, 3.0, 0.0)[0]
        xmin, ymin, xmax, ymax = (float(v) for v in ref.limits)
        assert _px_box(box, q_abs_max, shape) == (xmin, xmax, int(ymin), int(ymax))
