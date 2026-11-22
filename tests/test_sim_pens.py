"""The Expected-pattern boxes take a user pen, like detected / fitted.

Each reflection state ("missed", "explained", "selected") has a
built-in look (state colour, dashed, 1.4 wide, 2.2 selected) and an
optional partial user pen on top: colour, line style and width, set
from the swatches in the Expected-pattern dock (the same PenPopup the
Detected / Fitted swatches use), persisted app-wide.
Source: viewer_styles.py ``sim_pen`` / ``load_sim_pen_overrides``;
image_viewer_overlays.py ``set_sim_pen`` and the three simulation
renderers; main_window_build.py ``_open_sim_pen_popup``.
"""

from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QSettings, Qt

from mlgidgui.image_viewer import _PeakShapeItem
from mlgidgui.pen_picker import PenPopup
from mlgidgui.viewer_styles import (
    SIM_STATES,
    load_sim_pen_overrides,
    save_sim_pen_overrides,
    sim_pen,
    sim_state_colors,
)

from test_sim_overlay_gui import (
    _fake_cifpattern,
    _markers_style,
    _open,
    _structure,
    _type_user_hkl,
)

pytestmark = pytest.mark.gui


# --- the pure pen -----------------------------------------------------


def test_the_built_in_look_is_what_the_overlay_always_had():
    colors = sim_state_colors()
    for state, width in (("missed", 1.4), ("explained", 1.4),
                         ("selected", 2.2)):
        pen = sim_pen(state)
        assert pen["color"] == colors[state]
        assert pen["style"] == Qt.PenStyle.DashLine
        assert pen["width"] == pytest.approx(width)


def test_a_user_pen_is_partial_and_layered_on_top():
    over = {"missed": {"width": 4.0}}
    pen = sim_pen("missed", over)
    assert pen["width"] == 4.0
    assert pen["color"] == sim_state_colors()["missed"]      # untouched
    assert sim_pen("explained", over) == sim_pen("explained")


def test_overrides_round_trip_through_settings(clean_matched_colors):
    assert load_sim_pen_overrides() == {}
    save_sim_pen_overrides({
        "missed": {"color": "#112233", "style": Qt.PenStyle.DotLine,
                   "width": 3.0},
        "explained": {},
    })
    assert load_sim_pen_overrides() == {
        "missed": {"color": "#112233", "style": Qt.PenStyle.DotLine,
                   "width": 3.0},
    }
    QSettings().setValue("simPens", "not json")
    assert load_sim_pen_overrides() == {}


# --- the viewer -------------------------------------------------------


def _overlay_on(mw, synthetic_nexus, hkl="1 1 1"):
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    _type_user_hkl(mw, hkl)


def test_a_pen_restyles_the_boxes_at_once(
    main_window, synthetic_nexus, clean_matched_colors,
):
    mw = main_window
    _overlay_on(mw, synthetic_nexus)
    box = mw.viewer._sim_items[0]
    assert isinstance(box, _PeakShapeItem)
    assert box._pen.widthF() == pytest.approx(1.4)
    seen: list[str] = []
    mw.viewer.simPenChanged.connect(seen.append)

    mw.viewer.set_sim_pen("missed", {
        "color": "#ff00ff", "width": 4.0, "style": Qt.PenStyle.SolidLine,
    })
    assert seen == ["missed"]
    box = mw.viewer._sim_items[0]          # rebuilt by the render
    assert box._pen.color().name() == "#ff00ff"
    assert box._pen.widthF() == pytest.approx(4.0)
    assert box._pen.style() == Qt.PenStyle.SolidLine
    assert box._pen.isCosmetic()

    # Back to the built-in look.
    mw.viewer.set_sim_pen("missed", None)
    box = mw.viewer._sim_items[0]
    assert box._pen.color().name() == sim_state_colors()["missed"]
    assert box._pen.widthF() == pytest.approx(1.4)
    assert box._pen.style() == Qt.PenStyle.DashLine
    assert mw.viewer.sim_pen_override("missed") == {}


def test_each_state_has_its_own_pen(
    main_window, synthetic_nexus, clean_matched_colors,
):
    """Missed and explained reflections sit in separate items, so a pen
    for one leaves the other alone."""
    mw = main_window
    _open(mw, synthetic_nexus)
    # One matched box on the (1,1,1) reflection of In2O3 -> "explained".
    mw.viewer.set_matched_structures(0, [
        _structure("In2O3", (1, 1, 1), boxes=[(0.7071, 45.0, 0.2, 10.0, False)]),
    ])
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    _type_user_hkl(mw, "1 1 1")
    mw.viewer.set_sim_pen("explained", {"color": "#0000ff", "width": 5.0})
    colours = sorted(
        it._pen.color().name() for it in mw.viewer._sim_items
        if isinstance(it, _PeakShapeItem)
    )
    assert "#0000ff" in colours
    for it in mw.viewer._sim_items:
        if it._pen.color().name() != "#0000ff":
            assert it._pen.widthF() == pytest.approx(1.4)   # untouched


def test_markers_style_takes_colour_width_and_style(
    main_window, synthetic_nexus, clean_matched_colors,
):
    mw = main_window
    _overlay_on(mw, synthetic_nexus, hkl="0 0 2")
    _markers_style(mw)
    scatter = [i for i in mw.viewer._sim_items
               if isinstance(i, pg.ScatterPlotItem)]
    assert scatter
    assert scatter[0].opts["pen"].widthF() == pytest.approx(1.5)

    mw.viewer.set_sim_pen("missed", {
        "color": "#00ff00", "width": 3.0, "style": Qt.PenStyle.DotLine,
    })
    scatter = [i for i in mw.viewer._sim_items
               if isinstance(i, pg.ScatterPlotItem)][0]
    pen = scatter.opts["pen"]
    assert pen.color().name() == "#00ff00"
    assert pen.widthF() == pytest.approx(3.0)
    assert pen.style() == Qt.PenStyle.DotLine
    # The diamond SIZE is still the intensity coding.
    assert max(scatter.data["size"]) > 5.0


def test_ring_curves_take_the_pen_and_drop_the_intensity_width(
    main_window, synthetic_nexus, clean_matched_colors,
):
    mw = main_window
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)            # powder rings
    _markers_style(mw)
    curves = [i for i in mw.viewer._sim_items
              if isinstance(i, pg.PlotCurveItem)]
    assert len({round(c.opts["pen"].widthF(), 3) for c in curves}) > 1  # coded

    mw.viewer.set_sim_pen("missed", {"width": 2.0, "color": "#123456"})
    curves = [i for i in mw.viewer._sim_items
              if isinstance(i, pg.PlotCurveItem)]
    assert {round(c.opts["pen"].widthF(), 3) for c in curves} == {2.0}
    assert {c.opts["pen"].color().name() for c in curves} == {"#123456"}
    assert all(c.opts["pen"].style() == Qt.PenStyle.DashLine for c in curves)


def test_an_unknown_state_is_refused(main_window, clean_matched_colors):
    with pytest.raises(ValueError):
        main_window.viewer.set_sim_pen("nope", {"width": 2.0})


def test_a_new_viewer_starts_with_the_saved_pens(
    main_window, clean_matched_colors,
):
    from mlgidgui.image_viewer import GIWAXSImageViewer

    main_window.viewer.set_sim_pen("explained", {"color": "#abcdef"})
    fresh = GIWAXSImageViewer()
    try:
        assert fresh.sim_pen("explained")["color"] == "#abcdef"
    finally:
        fresh.deleteLater()


# --- the dock ---------------------------------------------------------


def test_the_dock_has_a_swatch_per_state(main_window, clean_matched_colors):
    mw = main_window
    assert list(mw._sim_pen_swatches) == list(SIM_STATES)
    assert [b.text() for b in mw._sim_pen_swatches.values()] == [
        "missed", "explained", "selected",
    ]


def test_a_swatch_opens_the_editor_and_follows_the_image(
    main_window, clean_matched_colors,
):
    mw = main_window
    before = mw._sim_pen_swatches["explained"].icon().pixmap(40, 20).toImage()

    mw._open_sim_pen_popup("explained")
    popup = mw.findChild(PenPopup)
    assert popup is not None
    # Seeded with the pen in effect.
    assert popup._width_spin.value() == pytest.approx(1.4)

    popup._width_spin.setValue(4.5)
    assert mw.viewer.sim_pen("explained")["width"] == pytest.approx(4.5)
    after = mw._sim_pen_swatches["explained"].icon().pixmap(40, 20).toImage()
    assert after != before                              # swatch redrawn

    popup._auto_button.click()
    assert mw.viewer.sim_pen("explained") == sim_pen("explained")
    again = mw._sim_pen_swatches["explained"].icon().pixmap(40, 20).toImage()
    assert again == before                              # back to preset
