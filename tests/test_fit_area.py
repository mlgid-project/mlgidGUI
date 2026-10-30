"""The dashed red outline of the area pygidfit really fits.

``manual_fit.fit_area`` turns a detected box into the rectangle pygidfit's
``boxes_preprocessing`` fits (radially twice the drawn width, a pixel more
on every side). The viewer draws it around a selected detected peak when the
"show fit area" setting is on.

Source: manual_fit.py ``fit_area``; image_viewer_render.py
``_fit_area_table``; main_window_dialogs.py ``_SettingsDialog``.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mlgidgui.manual_fit import PeakBox, fit_area

SHAPE = (512, 1024)
Q_ABS_MAX = 5.0
PX_R = Q_ABS_MAX / (SHAPE[1] - 1)       # one radial pixel, in 1/Angstrom
PX_A = 90.0 / (SHAPE[0] - 1)            # one angular pixel, in degrees


def _limits(area):
    r, wr, a, wa = area
    return r - wr / 2, r + wr / 2, a - wa / 2, a + wa / 2


# -- the geometry -----------------------------------------------------------


def test_the_area_is_twice_as_wide_radially_plus_a_pixel_each_side():
    r_lo, r_hi, a_lo, a_hi = _limits(
        fit_area(PeakBox(2.0, 0.2, 45.0, 10.0), Q_ABS_MAX))
    # Drawn: 1.9 .. 2.1. Fitted: radius +- radius_width, one pixel further.
    assert r_lo == pytest.approx(1.8 - PX_R, abs=PX_R)
    assert r_hi == pytest.approx(2.2 + PX_R, abs=PX_R)
    # Angularly the drawn extent, 40 .. 50 degrees, plus a pixel each side.
    assert a_lo == pytest.approx(40.0 - PX_A, abs=PX_A)
    assert a_hi == pytest.approx(50.0 + PX_A, abs=PX_A)


def test_the_area_is_larger_than_the_drawn_box_in_both_directions():
    box = PeakBox(2.0, 0.2, 45.0, 10.0)
    _, wr, _, wa = fit_area(box, Q_ABS_MAX)
    assert wr > 1.9 * box.radius_width     # twice, minus nothing: pad adds
    assert wa > box.angle_width


def test_a_box_at_the_edge_is_clipped_into_the_image():
    r_lo, _, a_lo, _ = _limits(
        fit_area(PeakBox(0.05, 0.2, 1.0, 4.0), Q_ABS_MAX))
    assert r_lo == pytest.approx(0.0, abs=1e-9)
    assert a_lo == pytest.approx(0.0, abs=1e-9)


def test_a_ring_is_fitted_across_the_whole_angular_range():
    r, wr, a, wa = fit_area(
        PeakBox(2.0, 0.2, 45.0, float("inf")), Q_ABS_MAX)
    assert (a, wa) == (45.0, float("inf"))
    assert r == pytest.approx(2.0, abs=PX_R)
    assert wr > 0.39


@pytest.mark.parametrize("box, q_abs_max", [
    (PeakBox(2.0, 0.2, 45.0, 10.0), 0.0),
    (PeakBox(2.0, 0.2, 45.0, 10.0), float("nan")),
    (PeakBox(float("nan"), 0.2, 45.0, 10.0), Q_ABS_MAX),
    (PeakBox(2.0, float("inf"), 45.0, 10.0), Q_ABS_MAX),
])
def test_an_unusable_box_or_scale_has_no_area(box, q_abs_max):
    assert fit_area(box, q_abs_max) is None


def test_the_area_agrees_with_pygidfits_own_boxes():
    """The strongest check: pygidfit's ``boxes_preprocessing`` on the same
    detected box, converted back to physical units, is the drawn area."""
    pytest.importorskip("pygidfit")
    from pygidfit.box_utils import boxes_preprocessing
    from pygidfit.process_scans import DetectedPeaks

    for box in (
        PeakBox(1.5, 0.06, 45.0, 7.0),
        PeakBox(3.9, 0.1, 87.0, 9.0),
        PeakBox(2.0, 0.05, 30.0, 12.5),
    ):
        det = DetectedPeaks(
            radius=np.array([box.radius]),
            radius_width=np.array([box.radius_width]),
            angle=np.array([box.angle]),
            angle_width=np.array([box.angle_width]),
        )
        ref = boxes_preprocessing(
            det, SHAPE, 1.0, Q_ABS_MAX, 3.0, 3.0, 0.0)[0]
        x0, y0, x1, y1 = (float(v) for v in ref.limits)
        expected = (
            x0 / (SHAPE[1] - 1) * Q_ABS_MAX, x1 / (SHAPE[1] - 1) * Q_ABS_MAX,
            y0 / (SHAPE[0] - 1) * 90.0, y1 / (SHAPE[0] - 1) * 90.0,
        )
        assert _limits(fit_area(box, Q_ABS_MAX)) == pytest.approx(
            expected, abs=1e-9)


# -- the viewer -------------------------------------------------------------


@pytest.fixture
def viewer(qtbot):
    from mlgidgui.image_viewer import GIWAXSImageViewer
    from mlgidgui.viewer_styles import MODE_POLAR

    v = GIWAXSImageViewer()
    qtbot.addWidget(v)
    v._mode = MODE_POLAR
    v._polar_cache = (None, np.linspace(0, 5, 400), np.linspace(0, 90, 180))
    # q_abs_max = hypot(4, 3) = 5, as pygidfit computes it.
    v._stack = SimpleNamespace(
        q_xy=np.linspace(-1.0, 3.0, 24), q_z=np.linspace(0.0, 4.0, 16))
    return v


def _sel(kind, peak_id=0, r=2.0, a=45.0, wa=10.0):
    from mlgidgui.image_viewer import SelectedPeak
    return SelectedPeak(
        kind=kind, frame=0, peak_id=peak_id, radius=r, angle=a,
        radius_width=0.2, angle_width=wa,
    )


def _rects(item):
    """The rectangles of a polar-mode outline: ``(x0, y0, x1, y1)`` each."""
    out = []
    for poly in item._path.toSubpathPolygons():
        xs = [p.x() for p in poly]
        ys = [p.y() for p in poly]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


@pytest.fixture(autouse=True)
def _setting_on_by_default():
    from PySide6.QtCore import QSettings

    from mlgidgui.main_window_constants import SHOW_FIT_AREA_KEY

    QSettings().remove(SHOW_FIT_AREA_KEY)
    yield
    QSettings().remove(SHOW_FIT_AREA_KEY)


@pytest.mark.gui
def test_a_selected_detected_peak_gets_the_outline(viewer):
    viewer._set_selected(_sel("detected"))
    (x0, y0, x1, y1), = _rects(viewer._fit_area)
    assert x1 - x0 == pytest.approx(0.4, abs=0.02)      # twice the 0.2 drawn
    assert (x0 + x1) / 2 == pytest.approx(2.0, abs=0.01)
    assert y1 - y0 == pytest.approx(10.0, abs=0.5)
    assert (y0 + y1) / 2 == pytest.approx(45.0, abs=0.2)


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["fitted", "matched", "manual"])
def test_only_detected_peaks_have_the_outline(viewer, kind):
    viewer._set_selected(_sel(kind))
    assert _rects(viewer._fit_area) == []


@pytest.mark.gui
def test_the_outline_goes_when_the_selection_does(viewer):
    viewer._set_selected(_sel("detected"))
    assert _rects(viewer._fit_area)
    viewer._set_selected(None)
    assert _rects(viewer._fit_area) == []


@pytest.mark.gui
def test_the_setting_turns_the_outline_off_and_on(viewer):
    from PySide6.QtCore import QSettings

    from mlgidgui.main_window_constants import SHOW_FIT_AREA_KEY

    viewer._set_selected(_sel("detected"))
    assert _rects(viewer._fit_area)
    QSettings().setValue(SHOW_FIT_AREA_KEY, False)
    viewer.refresh_overlays()
    assert _rects(viewer._fit_area) == []
    QSettings().setValue(SHOW_FIT_AREA_KEY, True)
    viewer.refresh_overlays()
    assert _rects(viewer._fit_area)


@pytest.mark.gui
def test_a_ring_outline_spans_the_displayed_angles(viewer):
    viewer._set_selected(_sel("detected", wa=float("inf")))
    (_, y0, _, y1), = _rects(viewer._fit_area)
    lo, hi = viewer.angular_extent()
    assert (y0, y1) == pytest.approx((lo, hi), abs=0.01)


@pytest.mark.gui
def test_in_the_cartesian_view_the_outline_is_an_annular_sector(viewer):
    from mlgidgui.viewer_styles import MODE_CARTESIAN

    viewer._mode = MODE_CARTESIAN
    viewer._set_selected(_sel("detected"))
    points = [p for poly in viewer._fit_area._path.toSubpathPolygons()
              for p in poly]
    assert points
    radii = [math.hypot(p.x(), p.y()) for p in points]
    assert min(radii) == pytest.approx(1.8, abs=0.02)
    assert max(radii) == pytest.approx(2.2, abs=0.02)


@pytest.mark.gui
def test_no_stack_no_outline(viewer):
    viewer._stack = None
    viewer._set_selected(_sel("detected"))
    assert _rects(viewer._fit_area) == []


@pytest.mark.gui
def test_the_outline_is_dashed_and_red(viewer):
    from PySide6.QtCore import Qt

    pen = viewer._fit_area._pen
    assert pen.style() == Qt.PenStyle.DashLine
    colour = pen.color()
    assert colour.red() > 180 and colour.green() < 120 and colour.blue() < 120


# -- the setting -------------------------------------------------------------


@pytest.mark.gui
def test_the_settings_dialog_has_the_switch(main_window):
    from PySide6.QtCore import QSettings

    from mlgidgui.main_window_constants import SHOW_FIT_AREA_KEY
    from mlgidgui.main_window_dialogs import _SettingsDialog

    dlg = _SettingsDialog(main_window)
    assert dlg._chk_fit_area.isChecked()            # on by default
    dlg._chk_fit_area.setChecked(False)
    dlg.save_to_qsettings()
    assert str(QSettings().value(SHOW_FIT_AREA_KEY)).lower() == "false"
    assert not _SettingsDialog(main_window)._chk_fit_area.isChecked()


@pytest.mark.gui
def test_in_a_real_window_a_detected_peak_gets_the_outline(
    main_window, synthetic_nexus_with_peaks, qtbot
):
    from mlgidgui.image_viewer import SelectedPeak
    from mlgidgui.session import NexusSession

    window = main_window
    window._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    qtbot.wait(50)
    det = window.viewer._frame_peaks[0]["detected"]
    i = 1                                          # radius 2.0, angle 45
    window.viewer._set_selected(SelectedPeak(
        kind="detected", frame=0, peak_id=int(det.ids[i]),
        radius=float(det.radius[i]), angle=float(det.angle[i]),
        radius_width=float(det.radius_width[i]),
        angle_width=float(det.angle_width[i]),
    ))
    (x0, y0, x1, y1), = _rects(window.viewer._fit_area)
    assert x1 - x0 == pytest.approx(2 * float(det.radius_width[i]), abs=0.05)
    assert y1 - y0 > float(det.angle_width[i])
    window.viewer._set_selected(None)
    assert _rects(window.viewer._fit_area) == []
