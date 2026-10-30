"""Selecting a detected / fitted peak zooms the image onto its box; a ring zooms back out.

``peak_view_range`` is pure geometry (no Qt); the viewer tests drive
``_set_selected`` on a bare ``GIWAXSImageViewer`` with the polar axes
installed directly, so the zoom is checked without loading a file.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.peak_zoom import MIN_FRACTION, PAD, peak_view_range

_KW = dict(radius_max=4.0, angle_extent=(0.0, 90.0))


def test_polar_window_is_the_box_plus_padding_centred_on_it():
    (x0, x1), (y0, y1) = peak_view_range("polar", 2.0, 0.2, 40.0, 10.0, **_KW)
    assert (x0 + x1) / 2 == pytest.approx(2.0)
    assert (y0 + y1) / 2 == pytest.approx(40.0)
    assert x1 - x0 == pytest.approx(0.2 * (1 + 2 * PAD))
    assert y1 - y0 == pytest.approx(10.0 * (1 + 2 * PAD))


def test_a_very_narrow_box_does_not_zoom_to_nothing():
    (x0, x1), (y0, y1) = peak_view_range("polar", 2.0, 0.001, 40.0, 0.05, **_KW)
    assert x1 - x0 == pytest.approx(MIN_FRACTION * 4.0)
    assert y1 - y0 == pytest.approx(MIN_FRACTION * 90.0)


def test_the_window_is_shifted_not_shrunk_at_the_data_edge():
    (x0, x1), (y0, y1) = peak_view_range("polar", 0.05, 0.1, 2.0, 4.0, **_KW)
    assert x0 == 0.0 and y0 == 0.0
    assert x1 - x0 == pytest.approx(0.1 * (1 + 2 * PAD))
    assert y1 - y0 == pytest.approx(4.0 * (1 + 2 * PAD))


def test_cartesian_window_bounds_the_annular_sector():
    (x0, x1), (y0, y1) = peak_view_range("cartesian", 2.0, 0.2, 45.0, 10.0, **_KW)
    # The sector sits near (1.41, 1.41); the window must contain it.
    c = 2.0 * np.cos(np.deg2rad(45.0))
    assert x0 < c < x1 and y0 < c < y1
    assert x1 - x0 < 1.5 and y1 - y0 < 1.5            # zoomed, not the whole image


def test_a_ring_row_spans_the_whole_angular_axis():
    _, (y0, y1) = peak_view_range(
        "polar", 2.0, 0.1, 45.0, float("inf"), **_KW,
    )
    assert (y0, y1) == (0.0, 90.0)


@pytest.mark.parametrize("bad", [
    dict(mode="raw"),
    dict(radius=float("nan")),
    dict(radius_width=float("inf")),
])
def test_unusable_input_returns_none(bad):
    args = dict(mode="polar", radius=2.0, radius_width=0.2, angle=40.0,
                angle_width=10.0)
    args.update(bad)
    mode = args.pop("mode")
    assert peak_view_range(
        mode, args["radius"], args["radius_width"], args["angle"],
        args["angle_width"], **_KW,
    ) is None


# -- the viewer ---------------------------------------------------------

pytestmark_gui = pytest.mark.gui


@pytest.fixture
def viewer(qtbot):
    from mlgidgui.image_viewer import GIWAXSImageViewer
    from mlgidgui.viewer_styles import MODE_POLAR

    v = GIWAXSImageViewer()
    qtbot.addWidget(v)
    v._mode = MODE_POLAR
    v._polar_cache = (None, np.linspace(0, 4, 400), np.linspace(0, 90, 180))
    return v


def _sel(kind, peak_id=0, r=2.0, a=40.0):
    from mlgidgui.image_viewer import SelectedPeak
    return SelectedPeak(
        kind=kind, frame=0, peak_id=peak_id, radius=r, angle=a,
        radius_width=0.2, angle_width=10.0,
    )


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["detected", "fitted"])
def test_selecting_a_detected_or_fitted_peak_zooms_to_it(viewer, kind):
    viewer._set_selected(_sel(kind))
    xr, yr = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(0.2 * (1 + 2 * PAD), rel=0.02)
    assert yr[1] - yr[0] == pytest.approx(10.0 * (1 + 2 * PAD), rel=0.02)
    assert (xr[0] + xr[1]) / 2 == pytest.approx(2.0, abs=0.02)


@pytest.mark.gui
def test_a_manual_selection_does_not_move_the_view(viewer):
    from mlgidgui.image_viewer import SelectedPeak

    before = viewer._plot.getViewBox().viewRange()
    sel = SelectedPeak(
        kind="matched", frame=0, peak_id=0, radius=2.0, angle=40.0,
        radius_width=0.2, angle_width=10.0,
    )
    viewer._set_selected(sel)
    assert viewer._plot.getViewBox().viewRange() == before


@pytest.mark.gui
def test_reselecting_the_same_peak_does_not_re_zoom(viewer):
    viewer._set_selected(_sel("fitted"))
    viewer._plot.getViewBox().setRange(xRange=(0, 4), yRange=(0, 90), padding=0)
    viewer._set_selected(_sel("fitted"))          # same identity: no-op
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(4.0, rel=0.02)
    viewer._set_selected(_sel("fitted", peak_id=1, r=3.0))   # new peak: zooms
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] < 1.0


# -- rings and the Settings switch ----------------------------------------


def _with_image(viewer):
    """Give the viewbox something to auto-range to, as a loaded frame does."""
    import pyqtgraph as pg

    viewer._plot.getViewBox().addItem(pg.ImageItem(np.zeros((400, 180))))
    viewer.reset_zoom()
    xr, _ = viewer._plot.getViewBox().viewRange()
    return xr[1] - xr[0]


def _ring(kind="fitted", peak_id=0, angle_width=float("inf")):
    from mlgidgui.image_viewer import SelectedPeak

    return SelectedPeak(
        kind=kind, frame=0, peak_id=peak_id, radius=2.0, angle=45.0,
        radius_width=0.2, angle_width=angle_width, is_ring=True,
    )


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["detected", "fitted"])
def test_selecting_a_ring_zooms_back_out_after_a_box(viewer, kind):
    full = _with_image(viewer)
    viewer._set_selected(_sel(kind))
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] < 0.1 * full                  # zoomed in on the box

    viewer._set_selected(_ring(kind, peak_id=1))
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(full, rel=0.02)


@pytest.mark.gui
def test_a_ring_flag_with_a_finite_width_also_zooms_out(viewer):
    full = _with_image(viewer)
    viewer._set_selected(_sel("fitted"))
    viewer._set_selected(_ring("detected", peak_id=1, angle_width=80.0))
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(full, rel=0.02)


@pytest.mark.gui
def test_a_ring_selected_from_the_table_zooms_out(viewer):
    """The Peaks table hands rows to ``_set_selected`` too."""
    full = _with_image(viewer)
    viewer._set_selected(_sel("detected"))
    viewer._set_selected(_ring("fitted", peak_id=2))
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(full, rel=0.02)


@pytest.mark.gui
def test_reselecting_the_same_ring_does_not_reset_the_view_again(viewer):
    full = _with_image(viewer)
    viewer._set_selected(_ring())
    viewer._plot.getViewBox().setRange(xRange=(0, 40), yRange=(0, 20), padding=0)
    viewer._set_selected(_ring())                      # same identity: no-op
    xr, _ = viewer._plot.getViewBox().viewRange()
    assert xr[1] - xr[0] == pytest.approx(40.0, rel=0.02)
    assert full > 100


@pytest.mark.gui
def test_the_setting_turns_the_ring_zoom_out_off_too(viewer):
    from PySide6.QtCore import QSettings
    from mlgidgui.main_window_constants import ZOOM_TO_SELECTED_PEAK_KEY

    full = _with_image(viewer)
    settings = QSettings()
    old = settings.value(ZOOM_TO_SELECTED_PEAK_KEY, None)
    viewer._set_selected(_sel("fitted"))               # setting still on
    settings.setValue(ZOOM_TO_SELECTED_PEAK_KEY, False)
    try:
        before = viewer._plot.getViewBox().viewRange()
        viewer._set_selected(_ring(peak_id=1))
        assert viewer._plot.getViewBox().viewRange() == before
        assert before[0][1] - before[0][0] < 0.1 * full
    finally:
        if old is None:
            settings.remove(ZOOM_TO_SELECTED_PEAK_KEY)
        else:
            settings.setValue(ZOOM_TO_SELECTED_PEAK_KEY, old)


@pytest.mark.gui
def test_the_setting_turns_the_zoom_off(viewer):
    from PySide6.QtCore import QSettings
    from mlgidgui.main_window_constants import ZOOM_TO_SELECTED_PEAK_KEY

    settings = QSettings()
    old = settings.value(ZOOM_TO_SELECTED_PEAK_KEY, None)
    settings.setValue(ZOOM_TO_SELECTED_PEAK_KEY, False)
    try:
        before = viewer._plot.getViewBox().viewRange()
        viewer._set_selected(_sel("fitted"))
        assert viewer._plot.getViewBox().viewRange() == before
        settings.setValue(ZOOM_TO_SELECTED_PEAK_KEY, True)
        viewer._set_selected(_sel("fitted", peak_id=1, r=3.0))
        xr, _ = viewer._plot.getViewBox().viewRange()
        assert xr[1] - xr[0] < 1.0
    finally:
        if old is None:
            settings.remove(ZOOM_TO_SELECTED_PEAK_KEY)
        else:
            settings.setValue(ZOOM_TO_SELECTED_PEAK_KEY, old)


@pytest.mark.gui
def test_the_settings_dialog_has_the_zoom_option_on_the_display_tab(qtbot):
    from PySide6.QtWidgets import QWidget
    from mlgidgui.main_window_dialogs import _SettingsDialog

    parent = QWidget()
    qtbot.addWidget(parent)
    dlg = _SettingsDialog(parent)
    qtbot.addWidget(dlg)
    titles = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
    assert titles == ["Display", "Peaks", "Interface"]
    display = dlg._tabs.widget(0)
    assert display.isAncestorOf(dlg._chk_zoom_peak)
    assert display.isAncestorOf(dlg._chk_profile_mask)
    assert dlg._tabs.widget(1).isAncestorOf(dlg._chk_peak_link)
    assert dlg._tabs.widget(2).isAncestorOf(dlg._chk_rail_run_opens_dock)
