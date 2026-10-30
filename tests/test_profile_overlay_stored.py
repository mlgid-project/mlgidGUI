"""A fitted / matched selection shows its stored row in both profiles:
a Gaussian at the stored centre and width (height read off the profile
at that centre) plus marks at the centre and the two box edges. A
manual / detected selection still gets the live scipy fit."""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.profile_viewer import ProfileViewer
from mlgidgui.viewer_items import SelectedPeak

pytestmark = pytest.mark.gui


def _viewer(qtbot):
    radius = np.linspace(0.0, 4.0, 400)
    angle = np.linspace(0.0, 90.0, 180)
    R, A = np.meshgrid(radius, angle, indexing="ij")
    # True peak at r=2.0 (sigma 0.1), a=40 (sigma 3).
    img = 100.0 * np.exp(-((R - 2.0) ** 2) / (2 * 0.1**2)
                         - ((A - 40.0) ** 2) / (2 * 3.0**2)) + 5.0
    viewer = ProfileViewer()
    qtbot.addWidget(viewer)
    viewer.set_polar_stack(img[None], radius, angle)
    return viewer


def _peak(kind, radius, angle, rw, aw):
    return SelectedPeak(
        kind=kind, frame=0, peak_id=0, radius=radius, angle=angle,
        radius_width=rw, angle_width=aw,
    )


def _marks(marks):
    return [m.value() for m in marks], [m.isVisible() for m in marks]


def test_fitted_selection_marks_stored_centre_and_box_edges(qtbot):
    viewer = _viewer(qtbot)
    # Stored values deliberately off the data's real peak (2.0 / 40).
    viewer.set_selected_peak(_peak("fitted", 2.3, 47.0, 0.3, 8.0))
    (lo, mid, hi), vis = _marks(viewer._radial_marks)
    assert (lo, mid, hi) == pytest.approx((2.15, 2.3, 2.45))
    assert all(vis)
    (lo, mid, hi), vis = _marks(viewer._angular_marks)
    assert (lo, mid, hi) == pytest.approx((43.0, 47.0, 51.0))
    assert all(vis)


def test_saved_peak_curve_is_centred_on_the_stored_centre(qtbot):
    viewer = _viewer(qtbot)
    # Stored values deliberately off the data's real peak (2.0 / 40).
    viewer.set_selected_peak(_peak("matched", 2.3, 47.0, 0.3, 8.0))
    fits = viewer.last_fit_params()
    assert fits["radial"].center == 2.3
    assert fits["radial"].sigma == pytest.approx(0.15)
    assert fits["angular"].center == 47.0
    assert fits["angular"].sigma == pytest.approx(4.0)
    assert len(viewer._radial_fit_curve.getData()[0]) > 0


def test_saved_peak_on_the_data_peak_reaches_the_profile_height(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_selected_peak(_peak("fitted", 2.0, 40.0, 0.2, 6.0))
    prof = np.asarray(viewer._radial_curve.getData()[1])
    fit_y = np.asarray(viewer._radial_fit_curve.getData()[1])
    assert np.nanmax(fit_y) == pytest.approx(np.nanmax(prof), rel=0.08)


def test_marks_hide_for_a_live_selection_and_on_clear(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_selected_peak(_peak("fitted", 2.3, 47.0, 0.3, 8.0))
    viewer.set_selected_peak(_peak("detected", 2.05, 41.0, 0.6, 16.0))
    assert not any(m.isVisible() for m in viewer._radial_marks)
    assert not any(m.isVisible() for m in viewer._angular_marks)
    viewer.set_selected_peak(_peak("fitted", 2.3, 47.0, 0.3, 8.0))
    viewer.set_selected_peak(None)
    assert not any(m.isVisible() for m in viewer._radial_marks)


def test_detected_selection_still_gets_the_live_fit(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_selected_peak(_peak("detected", 2.05, 41.0, 0.6, 16.0))
    fit = viewer.last_fit_params()["radial"]
    assert fit is not None
    # Refit moves onto the data's peak; the stored-shape path would not.
    assert fit.center == pytest.approx(2.0, abs=0.03)


def test_ring_fitted_selection_marks_radial_only(qtbot):
    viewer = _viewer(qtbot)
    ring = _peak("fitted", 2.0, 45.0, 0.2, float("inf"))
    ring.is_ring = True
    viewer.set_selected_peak(ring)
    assert all(m.isVisible() for m in viewer._radial_marks)
    assert not any(m.isVisible() for m in viewer._angular_marks)


def test_saved_peak_profile_averages_over_two_sigma_not_one(qtbot):
    """A saved peak's other-axis band is centre +- the full stored
    width (+-2 sigma), so its profile height is close to the true peak
    height; a detected box keeps width / 2 each side."""
    viewer = _viewer(qtbot)
    # True peak: r=2.0 (sigma 0.1), a=40 (sigma 3). Stored angular
    # width 2 sigma = 6 deg.
    viewer.set_selected_peak(_peak("fitted", 2.0, 40.0, 0.2, 6.0))
    y_fitted = np.asarray(viewer._radial_curve.getData()[1])
    peak_fitted = np.nanmax(y_fitted)
    # Mean of exp(-x^2/18) over +-6 deg is erf(sqrt(2)) * 3*sqrt(2*pi)
    # / 12 = 0.598, so 100 * 0.598 + background 5 = ~64.8.
    assert peak_fitted == pytest.approx(5.0 + 100.0 * 0.598, abs=3.0)

    # The same numbers as a detected box average over only +-3 deg.
    viewer.set_selected_peak(_peak("detected", 2.0, 40.0, 0.2, 6.0))
    peak_detected = np.nanmax(np.asarray(viewer._radial_curve.getData()[1]))
    # +-3 deg is +-1 sigma: 100 * 0.856 + 5 = ~90.6.
    assert peak_detected == pytest.approx(5.0 + 100.0 * 0.856, abs=3.0)


# -- neighbour masking in the profiles ---------------------------------


def _two_peak_viewer(qtbot):
    radius = np.linspace(0.0, 4.0, 400)
    angle = np.linspace(0.0, 90.0, 180)
    R, A = np.meshgrid(radius, angle, indexing="ij")
    g = lambda r0, a0: 100.0 * np.exp(
        -((R - r0) ** 2) / (2 * 0.1**2) - ((A - a0) ** 2) / (2 * 3.0**2)
    )
    img = g(2.0, 40.0) + g(2.0, 60.0) + 5.0   # same radius, 20 deg apart
    viewer = ProfileViewer()
    qtbot.addWidget(viewer)
    viewer.set_polar_stack(img[None], radius, angle)
    return viewer


def test_masking_off_by_default_leaves_the_profile_alone(qtbot):
    viewer = _two_peak_viewer(qtbot)
    viewer.set_selected_peak(_peak("detected", 2.0, 40.0, 0.2, 40.0))
    plain = np.asarray(viewer._radial_curve.getData()[1]).copy()
    viewer.set_neighbour_provider(lambda frame, sel: [])
    viewer.refresh()
    assert np.allclose(np.nan_to_num(plain),
                       np.nan_to_num(np.asarray(viewer._radial_curve.getData()[1])))
    assert "masked" not in viewer._radial_plot.getPlotItem().titleLabel.text


def test_masking_blanks_the_neighbour_in_the_angular_profile(qtbot):
    from mlgidgui.manual_fit import PeakBox

    viewer = _two_peak_viewer(qtbot)
    viewer.set_selected_peak(_peak("detected", 2.0, 40.0, 0.2, 10.0))
    angle = viewer._angle
    around_60 = (angle > 55.0) & (angle < 65.0)
    ang = np.asarray(viewer._angular_curve.getData()[1])
    assert np.nanmax(ang[around_60]) > 20.0       # neighbour visible
    viewer.set_neighbour_provider(
        lambda frame, sel: [PeakBox(2.0, 0.1, 60.0, 20.0)]
    )
    viewer.refresh()
    ang = np.asarray(viewer._angular_curve.getData()[1])
    assert np.isnan(ang[around_60]).all()          # masked out
    assert np.nanmax(ang[(angle > 35.0) & (angle < 45.0)]) > 20.0   # target kept
    assert "neighbours masked" in viewer._angular_plot.getPlotItem().titleLabel.text
    # The cached frame itself was not edited.
    assert np.isfinite(viewer._polar_stack[0]).all()


def test_neighbour_mask_leaves_clustered_neighbours_in():
    """Centres within pygidfit's clustering distance (pixels of its polar
    grid) cluster with the target (joint fit), so they are not masked.
    The distance is the Pipeline setting, 2 px by default since mlgidbase
    0.1.11 (it was 10)."""
    from mlgidgui.manual_fit import PeakBox
    from mlgidgui.profile_masking import neighbour_mask

    radius = np.linspace(0.0, 4.0, 400)
    angle = np.linspace(0.0, 90.0, 180)
    target = PeakBox(2.0, 0.1, 40.0, 0.4)       # drawn extent +-0.2 deg
    near = PeakBox(2.0, 0.1, 41.0, 1.2)         # 1 deg = ~5.7 px away
    far = PeakBox(2.0, 0.1, 50.0, 4.0)          # 10 deg = ~57 px: masked
    # With the old distance of 10 px `near` clusters and stays in...
    assert neighbour_mask(
        (400, 180), radius, angle, target, [near], cluster_distance=10.0,
    ) is None
    mask = neighbour_mask(
        (400, 180), radius, angle, target, [near, far], cluster_distance=10.0,
    )
    assert mask is not None
    ir = int(np.searchsorted(radius, 2.0))
    assert mask[ir, int(np.searchsorted(angle, 50.0))]
    assert not mask[ir, int(np.searchsorted(angle, 41.0))]
    # ...with the default 2 px it is a separate cluster, so it is masked.
    default = neighbour_mask((400, 180), radius, angle, target, [near])
    assert default is not None
    assert default[ir, int(np.searchsorted(angle, 41.0))]


def test_the_profile_viewer_asks_for_the_panels_cluster_distance(qtbot):
    """A neighbour 12 polar pixels away (~2.3 deg) is masked at the
    default 2 px, and left in the profile at 10 px? No: 12 > 10, so both
    mask it; at 20 px it clusters with the target and stays."""
    from mlgidgui.manual_fit import PeakBox

    viewer = _two_peak_viewer(qtbot)
    viewer.set_selected_peak(_peak("detected", 2.0, 40.0, 0.2, 2.0))
    # Not a twin (centre outside the target's drawn +-1 deg), 2.3 deg
    # = ~13 px of pygidfit's 512-row angle grid.
    neighbour = PeakBox(2.0, 0.1, 42.3, 0.4)
    viewer.set_neighbour_provider(lambda frame, sel: [neighbour])
    title = lambda: viewer._angular_plot.getPlotItem().titleLabel.text

    viewer.set_cluster_distance_provider(lambda: 2.0)
    viewer.refresh()
    assert "neighbours masked" in title()

    viewer.set_cluster_distance_provider(lambda: 20.0)
    viewer.refresh()
    assert "neighbours masked" not in title()

    # A provider that raises means "use the default", not a broken plot.
    viewer.set_cluster_distance_provider(lambda: 1 / 0)
    viewer.refresh()
    assert "neighbours masked" in title()


def test_neighbour_mask_skips_duplicates_of_the_target_and_bad_boxes():
    from mlgidgui.manual_fit import PeakBox
    from mlgidgui.profile_masking import neighbour_mask

    radius = np.linspace(0.0, 4.0, 400)
    angle = np.linspace(0.0, 90.0, 180)
    target = PeakBox(2.0, 0.2, 40.0, 10.0)
    same_peak = PeakBox(2.01, 0.3, 40.5, 14.0)    # the detected twin
    ring = PeakBox(1.0, 0.1, 45.0, float("inf"))
    zero = PeakBox(3.0, 0.0, 20.0, 5.0)
    assert neighbour_mask((400, 180), radius, angle, target,
                          [same_peak, ring, zero]) is None


def test_neighbour_mask_masks_spots_for_a_target_without_angular_extent():
    """A saved ring-like row has angle_width = inf. It must not count
    as the 'same peak' as every spot at its radius (peak 197 / 198 of
    the perovskite validation set): the spot is masked."""
    from mlgidgui.manual_fit import PeakBox
    from mlgidgui.profile_masking import neighbour_mask

    radius = np.linspace(0.0, 4.0, 400)
    angle = np.linspace(0.0, 90.0, 180)
    target = PeakBox(0.45, 0.06, 45.0, float("inf"))
    spot = PeakBox(0.447, 0.02, 88.0, 4.0)
    mask = neighbour_mask((400, 180), radius, angle, target, [spot])
    assert mask is not None
    assert mask[int(np.searchsorted(radius, 0.447)),
                int(np.searchsorted(angle, 88.0))]
    # The ring's own duplicate (infinite box) is never a neighbour.
    twin = PeakBox(0.45, 0.06, 45.0, float("inf"))
    assert neighbour_mask((400, 180), radius, angle, target, [twin]) is None


def test_saved_peak_shows_its_linear_background(qtbot):
    """The background line under the curve is drawn on its own: a
    straight line (constant slope) over the curve's span, below the
    Gaussian's maximum, and gone for a live (detected) fit."""
    viewer = _viewer(qtbot)
    viewer.set_selected_peak(_peak("fitted", 2.0, 40.0, 0.2, 6.0))
    for bg, fit in ((viewer._radial_bg_curve, viewer._radial_fit_curve),
                    (viewer._angular_bg_curve, viewer._angular_fit_curve)):
        x, y = (np.asarray(v) for v in bg.getData())
        fx, fy = (np.asarray(v) for v in fit.getData())
        assert len(x) > 0
        assert np.allclose(x, fx)
        slopes = np.diff(y) / np.diff(x)
        assert np.allclose(slopes, slopes[0], rtol=1e-6, atol=1e-9)   # a line
        assert np.all(y <= fy + 1e-9)                 # the peak sits on top

    viewer.set_selected_peak(_peak("detected", 2.05, 41.0, 0.6, 16.0))
    assert len(viewer._radial_bg_curve.getData()[0] or []) == 0
    viewer.set_selected_peak(None)
    assert len(viewer._angular_bg_curve.getData()[0] or []) == 0
