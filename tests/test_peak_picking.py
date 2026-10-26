"""Which box a click takes when several are stacked under the cursor.

Overlay boxes overlap constantly: a fitted box sits on the detection it
refines, a wide detection contains a narrow one, and a ring's box spans
every angle at its radius, so it contains every spot peak in that band.
The old rule resolved that by table order, which is invisible on screen
because each overlay kind is drawn as one path with one pen. These tests
pin the rules that replaced it: kind priority first, smallest box inside
a kind, rings only on their radial edges, and a repeat click walking
down the stack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication

from mlgidlab import peak_picking, theme_tokens
from mlgidlab.image_viewer import GIWAXSImageViewer
from mlgidlab.peak_picking import Box, box_area, contains, rank_hits
from mlgidlab.viewer_styles import OVERLAY_STYLE, hover_style, selection_style
from mlgidlab.theme import apply_dark_theme
from mlgidlab.file_model import MatchedStructure
from mlgidlab.viewer_items import ManualPeak, _peaks_from_manual

pytestmark = pytest.mark.gui

RING = Box(radius=2.0, radius_width=0.2, angle=45.0,
           angle_width=math.inf, is_ring=True)
SPOT = Box(radius=2.0, radius_width=0.05, angle=30.0, angle_width=4.0)
WIDE = Box(radius=2.0, radius_width=0.50, angle=30.0, angle_width=20.0)


# -- the rules, without Qt ------------------------------------------------

def test_a_ring_is_taken_only_near_its_radial_edges():
    """The whole point: a ring's interior must not swallow the peaks
    sitting inside its band."""
    assert not contains(RING, 2.0, 30.0, ring_edge_tol=0.03)
    assert contains(RING, 1.9, 30.0, ring_edge_tol=0.03)   # inner edge
    assert contains(RING, 2.1, 30.0, ring_edge_tol=0.03)   # outer edge
    assert not contains(RING, 1.5, 30.0, ring_edge_tol=0.03)


def test_without_a_tolerance_a_ring_keeps_its_whole_band():
    """Non-picking callers (geometry maths, the old helpers) still want
    plain containment, so the edge rule is opt-in."""
    assert contains(RING, 2.0, 30.0)


def test_a_non_finite_angular_width_counts_as_a_ring():
    """Rows from the pipeline carry ``angle_width = inf`` instead of the
    GUI's ``is_ring`` flag; both mean the same shape."""
    implied = Box(radius=2.0, radius_width=0.2, angle=45.0,
                  angle_width=math.inf, is_ring=False)
    assert peak_picking.is_ring_box(implied)
    assert not contains(implied, 2.0, 10.0, ring_edge_tol=0.03)


def test_a_ring_with_a_clamped_angular_width_still_checks_the_angle():
    """mlgidbase 0.1.5 clamps a ring's infinite width to a finite one,
    and a finite width is a real angular bound."""
    clamped = Box(radius=2.0, radius_width=0.2, angle=45.0,
                  angle_width=45.0, is_ring=True)
    assert contains(clamped, 1.9, 45.0, ring_edge_tol=0.03)
    assert not contains(clamped, 1.9, 170.0, ring_edge_tol=0.03)


def test_a_pad_grows_the_box_before_testing():
    """Peak boxes are often a handful of pixels across, and a box that
    has to be hit exactly is not a box the user can select."""
    assert not contains(SPOT, 2.04, 30.0)
    assert contains(SPOT, 2.04, 30.0, pad_radius=0.02)
    assert not contains(SPOT, 2.0, 34.0)
    assert contains(SPOT, 2.0, 34.0, pad_angle=2.0)


def test_the_pad_leaves_a_rings_edge_rule_alone():
    """A ring is already padded by its edge tolerance; adding the hit
    pad on top would widen the band twice."""
    assert not contains(RING, 2.0, 30.0, ring_edge_tol=0.03, pad_radius=0.5)


def test_the_smallest_box_ranks_first_and_rings_rank_last():
    assert box_area(SPOT) < box_area(WIDE) < box_area(RING)
    assert rank_hits([("wide", WIDE), ("ring", RING), ("spot", SPOT)]) == [
        "spot", "wide", "ring"]


def test_equal_boxes_keep_the_order_they_were_given():
    """Ties fall back to the caller's order, and the caller feeds rows in
    reverse table order — so same-sized boxes resolve exactly as they did
    before ranking existed."""
    twin = Box(radius=SPOT.radius, radius_width=SPOT.radius_width,
               angle=SPOT.angle, angle_width=SPOT.angle_width)
    assert rank_hits([("second", twin), ("first", SPOT)]) == ["second", "first"]


# -- the viewer ------------------------------------------------------------

@pytest.fixture
def picker(qtbot):
    """A viewer with a known zoom, so pixel-sized tolerances are known."""
    apply_dark_theme(QApplication.instance())
    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(900, 600)
    viewer.show()
    qtbot.waitExposed(viewer)
    viewer._plot.getViewBox().setRange(
        xRange=(0.0, 5.0), yRange=(-90.0, 90.0), padding=0)
    return viewer


def _peak(radius, angle, dr, da, temp_id, is_ring=False):
    return ManualPeak(radius=radius, angle=angle, radius_width=dr,
                      angle_width=da, is_ring=is_ring, temp_id=temp_id)


def _install(viewer, detected=(), fitted=()):
    viewer.set_peaks(0, {
        "detected": _peaks_from_manual(list(detected)) if detected else None,
        "fitted": _peaks_from_manual(list(fitted)) if fitted else None,
        "manual": None,
    })


def _click(viewer, r, a, mods=Qt.KeyboardModifier.NoModifier):
    viewer._on_select_at(QPointF(r, a), mods)


def test_the_inner_box_wins_where_two_are_nested(picker):
    """A box fully inside another used to be unreachable when it came
    first in the table; now it is the one you get."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 2

    # ... and the outer box is still reachable everywhere the inner is not
    _click(picker, 2.2, 30.0)
    assert picker.selected_peak.peak_id == 1


def test_kind_priority_still_outranks_size(picker):
    """A fitted box wins over a detected one even when it is the larger
    of the two: the refined result is what the user works with."""
    _install(picker,
             detected=[_peak(2.0, 30.0, 0.05, 4.0, 7)],
             fitted=[_peak(2.0, 30.0, 0.5, 20.0, 9)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.kind == "fitted"
    assert picker.selected_peak.peak_id == 9


def test_a_ring_no_longer_swallows_the_peaks_in_its_band(picker):
    _install(picker, detected=[_peak(2.0, 45.0, 0.2, math.inf, 1, is_ring=True),
                               _peak(2.0, 30.0, 0.04, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 2, "the spot, not the ring"

    # Empty part of the ring's band: nothing at all, rather than the ring
    _click(picker, 2.0, 80.0)
    assert picker.selected_peak is None


def test_a_ring_is_selected_by_clicking_its_edge(picker):
    _install(picker, detected=[_peak(2.0, 45.0, 0.2, math.inf, 1, is_ring=True)])
    tol = picker._ring_edge_tol()
    assert tol > 0
    _click(picker, 2.0 - 0.2 / 2 + tol / 2, 80.0)
    assert picker.selected_peak is not None
    assert picker.selected_peak.peak_id == 1


def test_clicking_again_walks_down_the_stack(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    seen = []
    for _ in range(3):
        _click(picker, 2.0, 30.0)
        seen.append(picker.selected_peak.peak_id)
    assert seen == [2, 1, 2], "smallest first, then out, then round again"


def test_a_click_never_hands_back_the_selected_box(picker):
    """The rule in one line: if the box you clicked is the one already
    selected, the click takes the next box under the cursor."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 1
    # Elsewhere inside the outer box, still over both: the outer one is
    # selected, so this hands back the inner one.
    _click(picker, 2.01, 31.0)
    assert picker.selected_peak.peak_id == 2


def test_a_double_click_takes_its_cycle_step_back(picker):
    """A bare double-click resets the zoom, and Qt delivers a plain
    click first — which cycles. Refusing to cycle on fast clicks was
    worse (it swallowed deliberate ones), so the step is taken back once
    the gesture turns out to be a double-click."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 1, "the click stepped down"
    picker.revert_cycle_for_double_click()
    assert picker.selected_peak.peak_id == 2, "and the double-click undid it"


def test_only_a_real_cycle_step_is_reverted(picker):
    """A first selection is not a step, so a double-click on a fresh
    box must not clear it."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    picker.revert_cycle_for_double_click()
    assert picker.selected_peak.peak_id == 2


def test_ctrl_click_keeps_its_kind_and_takes_the_smaller_box(picker):
    """Ctrl+click still extends one kind (so a detected multi-select is
    not broken by the fitted box drawn over it), and inside that kind it
    follows the same smallest-box rule."""
    _install(picker,
             detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                       _peak(2.0, 30.0, 0.05, 4.0, 2)],
             fitted=[_peak(2.0, 30.0, 0.06, 5.0, 5)])
    _click(picker, 2.0, 30.0, Qt.KeyboardModifier.ControlModifier)
    assert picker.selected_peak.kind == "fitted"

    picker.clear_selection()
    _click(picker, 2.2, 30.0)                       # a detected primary
    assert picker.selected_peak.kind == "detected"
    _click(picker, 2.0, 30.0, Qt.KeyboardModifier.ControlModifier)
    ids = [s.peak_id for s in picker.selected_peaks()]
    assert 2 in ids, "extended detected, and took the inner detected box"


# -- the nested detected / fitted case the user hit ------------------------

def test_a_detected_box_inside_a_fitted_one_is_reachable(picker):
    """Kind priority hands the first click to the fitted box, so the
    detected box under it is only reachable by stepping down."""
    _install(picker,
             fitted=[_peak(2.0, 35.0, 0.12, 10.0, 5)],
             detected=[_peak(2.0, 35.0, 0.03, 3.0, 7)])
    _click(picker, 2.0, 35.0)
    assert picker.selected_peak.kind == "fitted"
    _click(picker, 2.0, 35.0)
    assert picker.selected_peak.kind == "detected"
    assert picker.selected_peak.peak_id == 7


def test_a_hand_that_drifts_a_few_pixels_still_steps_down(picker):
    """Stepping asks nothing of the hand: any click that still covers
    the selected box moves on to the next one."""
    _install(picker,
             fitted=[_peak(2.0, 35.0, 0.12, 10.0, 5)],
             detected=[_peak(2.0, 35.0, 0.03, 3.0, 7)])
    x_px, y_px = picker._plot.getViewBox().viewPixelSize()
    _click(picker, 2.0, 35.0)
    _click(picker, 2.0 + 6 * x_px, 35.0 + 4 * y_px)
    assert picker.selected_peak.kind == "detected"


def test_a_near_miss_still_hits_a_small_box(picker):
    """The case that stayed broken: an 8 px box inside a big one. Exact
    containment sent a click three pixels off to the box around it —
    and since the stack then held only that box, there was nothing to
    step to and the inner box could not be selected at all."""
    x_px, y_px = picker._plot.getViewBox().viewPixelSize()
    _install(picker, detected=[_peak(2.0, 35.0, 60 * x_px, 60 * y_px, 1),
                               _peak(2.0, 35.0, 8 * x_px, 8 * y_px, 2)])
    _click(picker, 2.0 + 7 * x_px, 35.0)          # 3 px outside the small box
    assert picker.selected_peak.peak_id == 2


def test_the_tolerance_is_only_a_few_pixels(picker):
    """It must not turn every click into a selection."""
    x_px, y_px = picker._plot.getViewBox().viewPixelSize()
    _install(picker, detected=[_peak(2.0, 35.0, 8 * x_px, 8 * y_px, 2)])
    _click(picker, 2.0 + 14 * x_px, 35.0)         # 10 px outside
    assert picker.selected_peak is None


def test_a_click_on_a_box_that_is_not_selected_takes_that_box(picker):
    """Stepping applies only to the box you already have: a click that
    has moved onto a different one selects it outright, innermost
    first."""
    x_px, _ = picker._plot.getViewBox().viewPixelSize()
    # The third box sits 8 px away, outside the hit tolerance of the
    # first click.
    elsewhere = 2.0 + 8 * x_px
    _install(picker, detected=[_peak(2.0, 35.0, 0.12, 10.0, 5),
                               _peak(2.0, 35.0, 0.03, 3.0, 7),
                               _peak(elsewhere, 35.0, 2 * x_px, 0.6, 9)])
    _click(picker, 2.0, 35.0)
    assert picker.selected_peak.peak_id == 7, "the inner box"
    # Now over a box that is not the selection: taken outright.
    _click(picker, elsewhere, 35.0)
    assert picker.selected_peak.peak_id == 9


# -- matched structures ----------------------------------------------------

def _match(picker, peaks, ids):
    """Install ``peaks`` as fitted rows and claim them as one structure."""
    table = _peaks_from_manual(list(peaks))
    picker.set_peaks(0, {"detected": None, "fitted": table, "manual": None})
    picker.set_matched_structures(0, [MatchedStructure(
        solution_field="matched_segments_0000", local_idx=0, cif="PbI2",
        h=1, k=0, l=0, probability=0.9, peaks=table,
        peak_list=np.asarray(ids, dtype=int))])
    picker.set_overlay_visible("fitted", False)   # so matched is what a click takes
    return table


def test_the_selection_outline_sits_above_the_matched_boxes(picker):
    """Regression: matched overlay items are rebuilt and re-added to the
    viewbox on *every* render, so they land above anything added once at
    construction. The white highlight around a selected matched peak was
    painted over by the structure's own box, leaving only the sliver
    where the wider pen stuck out — while fitted and detected, added
    before the selection item, looked right."""
    _match(picker, [_peak(1.95, 32.0, 0.06, 5.0, 10),
                    _peak(2.05, 38.0, 0.06, 5.0, 11)], [0, 1])
    _click(picker, 1.95, 32.0)
    assert picker.selected_peak.kind == "matched"
    matched_z = [item.zValue() for _uid, item in picker._matched_items]
    assert matched_z, "the structure really is drawn"
    painted = _scan(picker, 1.95, 32.0)
    assert selection_style()["color"] in painted
    structure_colour = picker._pen_for_key(("PbI2", 1, 0, 0))["color"]
    assert structure_colour not in painted, "covered, not painted over"
    assert picker._selection.zValue() > max(matched_z)
    assert picker._hover.zValue() > max(matched_z)
    assert picker._selection.zValue() > picker._hover.zValue(), (
        "a selected box keeps its solid highlight over the preview")


def test_the_hover_outlines_the_whole_matched_structure(picker):
    """Clicking a matched peak selects the structure, so previewing one
    box of it would promise the wrong thing."""
    _match(picker, [_peak(1.95, 32.0, 0.06, 5.0, 10),
                    _peak(2.05, 38.0, 0.06, 5.0, 11)], [0, 1])
    depth = picker._update_hover(1.95, 32.0)
    assert depth == 1
    assert picker._hover_key[0] == "matched"
    outlined = picker._hover_boxes(picker._hit_candidates(1.95, 32.0)[0])
    assert sorted(b.temp_id for b in outlined) == [10, 11]


def test_a_single_peak_hover_outlines_only_itself(picker):
    """The structure expansion must not leak into ordinary candidates."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    outlined = picker._hover_boxes(picker._hit_candidates(2.0, 30.0)[0])
    assert [b.temp_id for b in outlined] == [2]


# -- how a highlight looks -------------------------------------------------

def _scan(picker, x, y, span=40):
    """Colours along a horizontal line through the data point ``(x, y)``.

    Reads the rendered widget rather than the pen settings, because the
    complaint was about what ends up on screen: a translucent preview
    blended with the box under it instead of covering it.
    """
    QApplication.processEvents()
    image = picker.grab().toImage()
    view = picker._view.ui.graphicsView
    scene = picker._plot.getViewBox().mapViewToScene(QPointF(x, y))
    centre = view.mapTo(picker, view.mapFromScene(scene))
    seen = set()
    for dx in range(-span, span + 1):
        colour = image.pixelColor(centre.x() + dx, centre.y())
        if colour.red() + colour.green() + colour.blue() > 160:
            seen.add(colour.name())
    return seen


@pytest.mark.parametrize("kind", ["detected", "fitted", "manual"])
def test_every_kind_gets_the_same_highlight_over_its_own_box(picker, kind):
    """One look for every overlay kind, and it covers the box rather
    than blending with it: the translucent preview over the dashed red
    detection came out pink, and every kind read differently."""
    box = _peak(2.0, 35.0, 0.3, 20.0, 3)
    if kind == "manual":
        picker._manual_peaks[0] = [box]
        picker.set_peaks(0, {"detected": None, "fitted": None, "manual": None})
    else:
        _install(picker, **{kind: [box]})
    own = OVERLAY_STYLE[kind]["color"]
    if kind != "manual":
        assert own in _scan(picker, 2.0, 35.0), "the box paints its own colour"
    # A manual box is a scratch label: deliberately invisible until it
    # is the selection (see ``_render_overlays``), so there is nothing
    # to cover yet — the check that matters for it is after the click.

    picker._update_hover(2.0, 35.0)
    hovered = _scan(picker, 2.0, 35.0)
    assert hover_style()["color"] in hovered
    assert own not in hovered, "the source colour is covered, not blended"

    _click(picker, 2.0, 35.0)
    selected = _scan(picker, 2.0, 35.0)
    assert selection_style()["color"] in selected
    assert own not in selected


def test_the_highlight_pens_outrank_every_overlay_pen():
    widest = max(style["width"] for style in OVERLAY_STYLE.values())
    assert selection_style()["width"] > widest
    assert hover_style()["width"] > widest


def test_the_hover_colour_is_the_accent_and_follows_the_theme(qtbot,
                                                              main_window):
    """White is the selection's colour; the preview takes the accent so
    "under the cursor" and "selected" are not the same signal. The item
    is built once, so the pen has to be swapped on a theme change."""
    assert hover_style("dark")["color"] == theme_tokens.color("accent", "dark")
    assert hover_style("light")["color"] != hover_style("dark")["color"]

    viewer = main_window.viewer
    for theme in ("light", "dark"):
        main_window._set_theme(theme)
        assert viewer._hover._pen.color().name() == hover_style(theme)["color"]


# -- hover preview ---------------------------------------------------------

def test_the_hover_outline_marks_what_a_click_would_take(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    depth = picker._update_hover(2.0, 30.0)
    assert depth == 2, "two boxes stacked here"
    assert picker._hover_key == ("detected", 2, None)
    assert not picker._hover.boundingRect().isNull()


def test_the_hover_outline_clears_on_empty_space(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    picker._update_hover(2.0, 30.0)
    assert picker._update_hover(4.5, -80.0) == 0
    assert picker._hover_key is None
    assert picker._hover.boundingRect().isNull()


def test_the_hover_outline_steps_aside_for_the_selection(picker):
    """The selected box already carries the solid highlight; drawing the
    preview on top of it would just thicken the line."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker._update_hover(2.0, 30.0) == 1
    assert picker._hover_key is None


def test_leaving_the_image_drops_the_outline(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    picker._update_hover(2.0, 30.0)
    assert picker._hover_key is not None
    picker._on_cursor_left()
    assert picker._hover_key is None
    assert picker._hover_pos is None


def _outlined(picker) -> bool:
    return not picker._hover.boundingRect().isNull()


def _pointer_over(picker, over: bool) -> None:
    """Fake the pointer being on (or off) the plot.

    ``underMouse`` reads ``WA_UnderMouse``, which Qt sets from real
    enter/leave dispatch — offscreen there is no pointer to dispatch,
    so the tests set the attribute the same way Qt would.
    """
    picker._view.ui.graphicsView.setAttribute(
        Qt.WidgetAttribute.WA_UnderMouse, over)


def test_a_re_render_while_busy_takes_the_outline_with_it(picker):
    """Regression: ``_refresh_hover`` zeroes the hover key before
    re-running, and ``_clear_hover`` used to skip the repaint when the
    key was already None — so any early return in ``_update_hover``
    (busy, dragging, mid draw-drag) left the outline painted on."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    assert _outlined(picker)

    picker._busy = True
    picker._render_overlays(0)
    assert not _outlined(picker)
    picker._busy = False


def test_starting_a_run_drops_the_outline(picker):
    """Hit-testing is off while the pipeline runs, so an outline left up
    would promise a click that does nothing — and the user watching a
    run is not moving the mouse to trigger the next update."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    picker.set_busy(True)
    assert not _outlined(picker)
    picker.set_busy(False)


def test_a_re_render_does_not_resurrect_an_outline_for_a_cursor_that_left(picker):
    """A Leave is not guaranteed — the pointer can exit fast, or the
    window can lose it to a popup or a keyboard switch. Without the
    ``underMouse`` check, the next frame change faithfully redrew the
    outline for a cursor that had gone."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    assert _outlined(picker)

    _pointer_over(picker, False)          # gone, no Leave delivered
    picker._render_overlays(0)
    assert not _outlined(picker)


def test_the_outline_survives_a_re_render_under_the_cursor(picker):
    """The other half: overlays are rebuilt on frame changes and
    pipeline results, and a cursor sitting on a box should keep its
    preview."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    picker._render_overlays(0)
    assert _outlined(picker)


def test_leaving_the_viewer_drops_the_outline(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    picker.leaveEvent(QEvent(QEvent.Type.Leave))
    assert not _outlined(picker)
    assert picker._hover_pos is None


def test_switching_view_mode_drops_the_outline(picker):
    """The remembered point is in the space just left — polar is
    (r, angle), Cartesian is (q_xy, q_z) — so re-running the hover on
    it would outline nonsense."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    picker.set_mode("cartesian")
    assert not _outlined(picker)
    assert picker._hover_pos is None


def test_closing_the_file_drops_the_outline(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.3, 20.0, 1)])
    _pointer_over(picker, True)
    picker._update_hover(2.0, 30.0)
    picker.clear()
    assert not _outlined(picker)


def test_the_readout_reports_a_stacked_cursor(main_window):
    """The count is what makes the repeat click discoverable."""
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 2.0, "theta": 30.0, "intensity": 5.0,
         "overlapping": 3})
    assert "3 boxes here" in main_window._sb_cursor.text()
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 2.0, "theta": 30.0, "intensity": 5.0})
    assert "boxes here" not in main_window._sb_cursor.text()


# -- Boxes are drawn in full, past the image edge --------------------------

def test_a_finite_box_past_the_image_edge_is_drawn_in_full():
    from mlgidlab.viewer_items import _PeakShapeItem

    class _T:                                   # one row: angle 3, width 10
        angle, angle_width = [3.0], [10.0]
        radius, radius_width = [1.0], [0.2]

        def __len__(self):
            return 1

    item = _PeakShapeItem("#fff", Qt.PenStyle.SolidLine, 1)
    item.set_polar(_T(), extent=(0.0, 90.0))
    rect = item.boundingRect()
    assert rect.top() == pytest.approx(-2.0)    # not clipped at 0
    assert rect.bottom() == pytest.approx(8.0)
    item.set_cartesian(_T(), extent=(0.0, 90.0))
    assert item.boundingRect().top() < 0.0      # below the q_xy axis (q_z < 0)


def test_a_ring_without_a_finite_width_still_stops_at_the_image_edge():
    from mlgidlab.viewer_items import _clip_angle

    assert _clip_angle(45.0, float("inf"), extent=(0.0, 90.0), clip_finite=False) == (0.0, 90.0)
    assert _clip_angle(float("nan"), float("inf"), extent=(0.0, 90.0)) == (0.0, 90.0)


def test_clip_angle_still_clips_a_finite_box_by_default():
    from mlgidlab.viewer_items import _clip_angle

    assert _clip_angle(3.0, 10.0, extent=(0.0, 90.0)) == (0.0, 8.0)
    assert _clip_angle(3.0, 10.0, extent=(0.0, 90.0), clip_finite=False) == (-2.0, 8.0)
