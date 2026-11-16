"""Drawing and editing peak boxes in the Cartesian view.

Ctrl+Alt-drag draws a box in both views: a rectangle in polar, an arc
box between the two drag corners in Cartesian. A selected manual /
detected box gets handles in both: a rectangle ROI in polar, an
``ArcBoxROI`` (inner / outer radius, two angular edges, body move) in
Cartesian. The box is the same polar ManualPeak / table row either way,
so undo, the file write and the profiles are shared.
Source: viewer_items.py ``_arc_box_from_corners``; arc_roi.py;
image_viewer_interact.py ``_on_draw_*`` / ``_sync_roi`` /
``_apply_roi_box``.
"""
from __future__ import annotations

import math

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.arc_roi import ANGLE_HI, ANGLE_LO, INNER, OUTER, ArcBoxROI
from mlgidgui.session import NexusSession
from mlgidgui.viewer_items import ManualPeak, SelectedPeak, _arc_box_from_corners
from mlgidgui.viewer_styles import MODE_CARTESIAN, MODE_POLAR

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def q(r: float, a_deg: float) -> QPointF:
    a = math.radians(a_deg)
    return QPointF(r * math.cos(a), r * math.sin(a))


def _cart_window(window, path):
    window._set_active_session(NexusSession.open(path))
    window.viewer.set_mode(MODE_CARTESIAN)
    return window.viewer


def _box(peak):
    return (peak.radius, peak.radius_width, peak.angle, peak.angle_width)


def test_arc_box_from_corners_is_order_free():
    p0, p1 = q(1.0, 20.0), q(1.5, 40.0)
    for a, b in ((p0, p1), (p1, p0)):
        r, dr, ang, da = _arc_box_from_corners(a.x(), a.y(), b.x(), b.y())
        assert (r, dr, ang, da) == pytest.approx((1.25, 0.5, 30.0, 20.0))


def test_ctrl_alt_drag_draws_an_arc_box(main_window, synthetic_nexus_with_peaks):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    p0, p1 = q(1.0, 20.0), q(1.5, 40.0)
    v._on_draw_started(p0)
    v._on_draw_updated(p0, p1)
    # The arc preview shows, not the polar rectangle.
    assert v._arc_preview_item.isVisible()
    assert not v._preview_item.isVisible()
    assert not v._arc_preview_item._path.isEmpty()
    v._on_draw_finished(p0, p1)
    assert not v._arc_preview_item.isVisible()

    boxes = v.manual_peaks(0)
    assert len(boxes) == 1
    assert _box(boxes[0]) == pytest.approx((1.25, 0.5, 30.0, 20.0))
    assert not boxes[0].is_ring
    sel = v.selected_peak
    assert sel.kind == "manual" and sel.manual_ref is boxes[0]
    assert isinstance(v._roi_item, ArcBoxROI)

    # A second box replaces the first (single-box policy), one undo each.
    v._on_draw_finished(q(2.0, 50.0), q(2.4, 60.0))
    assert [_box(b) for b in v.manual_peaks(0)] == [
        pytest.approx((2.2, 0.4, 55.0, 10.0))
    ]
    v.undo_last_action()
    assert [_box(b) for b in v.manual_peaks(0)] == [
        pytest.approx((1.25, 0.5, 30.0, 20.0))
    ]
    v.undo_last_action()
    assert v.manual_peaks(0) == []


def test_zero_size_drag_draws_nothing(main_window, synthetic_nexus_with_peaks):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    p = q(1.0, 20.0)
    v._on_draw_finished(p, p)
    v._on_draw_finished(q(1.0, 20.0), q(1.5, 20.0))   # no angular extent
    assert v.manual_peaks(0) == []


def _select_manual(v, peak):
    v.add_manual_peak(0, peak)
    v._set_selected(SelectedPeak.from_manual(peak, 0))
    return v._roi_item


def test_arc_handles_move_one_edge_and_undo(
    main_window, synthetic_nexus_with_peaks,
):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    box = ManualPeak(radius=1.25, angle=30.0, radius_width=0.5,
                     angle_width=20.0, temp_id=-1)
    roi = _select_manual(v, box)
    assert isinstance(roi, ArcBoxROI)
    assert set(roi.handles) == {OUTER, INNER, ANGLE_HI, ANGLE_LO}
    seen = []
    v.peakGeometryChanged.connect(lambda s: seen.append(s))

    # Outer edge out to |q| = 2.0; the inner edge (1.0) stays.
    roi.begin_change()
    roi.drag_edge(OUTER, q(2.0, 33.0).x(), q(2.0, 33.0).y())
    roi.end_change()
    assert _box(box) == pytest.approx((1.5, 1.0, 30.0, 20.0))
    assert seen
    # Upper angle edge to 50 deg; the lower (20) stays.
    roi.begin_change()
    roi.drag_edge(ANGLE_HI, q(1.5, 50.0).x(), q(1.5, 50.0).y())
    roi.end_change()
    assert _box(box) == pytest.approx((1.5, 1.0, 35.0, 30.0))
    # Dragging the inner edge past the outer one swaps them.
    roi.begin_change()
    roi.drag_edge(INNER, q(2.5, 35.0).x(), q(2.5, 35.0).y())
    roi.end_change()
    assert _box(box) == pytest.approx((2.25, 0.5, 35.0, 30.0))
    # Body move: +0.1 in |q|, +5 deg.
    roi.begin_change()
    start, now = q(2.25, 35.0), q(2.35, 40.0)
    roi.move_to((start.x(), start.y()), (now.x(), now.y()))
    roi.end_change()
    assert _box(box) == pytest.approx((2.35, 0.5, 40.0, 30.0))

    # One undo per gesture, back to the start.
    for _ in range(4):
        v.undo_last_action()
    assert _box(box) == pytest.approx((1.25, 0.5, 30.0, 20.0))
    # The handles followed the undo.
    assert roi.box() == pytest.approx((1.25, 0.5, 30.0, 20.0))


def test_detected_box_edit_reaches_the_file(
    main_window, synthetic_nexus_with_peaks,
):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    v._on_select_at(q(2.0, 45.0))          # detected peak r=2.0, a=45
    sel = v.selected_peak
    assert sel.kind == "detected"
    roi = v._roi_item
    assert isinstance(roi, ArcBoxROI)
    writes = []
    v.peakRowWriteRequested.connect(lambda *a: writes.append(a))
    roi.begin_change()
    roi.drag_edge(OUTER, q(2.3, 45.0).x(), q(2.3, 45.0).y())
    roi.end_change()
    assert writes and writes[-1][1] == "detected"
    t = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    i = list(t.ids).index(int(sel.peak_id))
    # Inner edge 1.9 kept, outer now 2.3.
    assert float(t.radius[i]) == pytest.approx(2.1, abs=1e-5)
    assert float(t.radius_width[i]) == pytest.approx(0.4, abs=1e-5)
    assert float(t.angle[i]) == pytest.approx(45.0, abs=1e-5)


def test_ring_gets_radial_handles_only(main_window, synthetic_nexus_with_peaks):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    ring = ManualPeak(radius=1.5, angle=45.0, radius_width=0.2,
                      angle_width=float("inf"), is_ring=True, temp_id=-1)
    roi = _select_manual(v, ring)
    assert set(roi.handles) == {OUTER, INNER}
    roi.begin_change()
    roi.drag_edge(OUTER, q(1.8, 10.0).x(), q(1.8, 10.0).y())
    roi.end_change()
    assert ring.radius == pytest.approx(1.6)
    assert math.isinf(ring.angle_width)   # never turned into a segment


def test_mode_switch_swaps_the_handles(main_window, synthetic_nexus_with_peaks):
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    box = ManualPeak(radius=1.25, angle=30.0, radius_width=0.5,
                     angle_width=20.0, temp_id=-1)
    _select_manual(v, box)
    assert isinstance(v._roi_item, ArcBoxROI)
    v.set_mode(MODE_POLAR)
    assert isinstance(v._roi_item, pg.ROI)
    v.set_mode(MODE_CARTESIAN)
    assert isinstance(v._roi_item, ArcBoxROI)
    assert v._roi_item.box() == pytest.approx((1.25, 0.5, 30.0, 20.0))


def test_external_edit_moves_the_arc_handles(
    main_window, synthetic_nexus_with_peaks,
):
    """A profile-region drag (or any external edit) re-syncs the arc."""
    v = _cart_window(main_window, synthetic_nexus_with_peaks)
    box = ManualPeak(radius=1.25, angle=30.0, radius_width=0.5,
                     angle_width=20.0, temp_id=-1)
    roi = _select_manual(v, box)
    box.radius, box.radius_width = 1.4, 0.3
    v.update_peak_geometry_external(box)
    assert roi.box() == pytest.approx((1.4, 0.3, 30.0, 20.0))


def test_polar_axes_known_when_opened_in_cartesian(
    main_window, synthetic_nexus_with_peaks,
):
    """Regression: with the Cartesian view chosen before a file opens,
    the polar axes (ring outlines, zoom-to-peak, handles) must still be
    known; they used to be filled only by rendering the polar view."""
    main_window.viewer.set_mode(MODE_CARTESIAN)
    main_window._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    v = main_window.viewer
    assert v._mode == MODE_CARTESIAN
    assert v.angular_extent() is not None
    v._on_select_at(q(2.0, 45.0))
    assert isinstance(v._roi_item, ArcBoxROI)


def _mouse_drag(qtbot, viewport, p0, delta, steps=10):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, p0)
    for k in range(1, steps + 1):
        QTest.mouseMove(viewport, p0 + QPoint(delta.x() * k // steps,
                                              delta.y() * k // steps))
        qtbot.wait(5)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, p0 + delta)
    qtbot.wait(30)


def test_real_mouse_drags_reach_handles_and_body(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    """Plain LMB passes the viewer's label filter to the arc item, as it
    does for the polar ROI: an edge handle drag changes only the radial
    range, a drag inside the box moves it, each one undo step."""
    from PySide6.QtCore import QPoint

    w = main_window
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    v = _cart_window(w, synthetic_nexus_with_peaks)
    qtbot.wait(50)
    box = ManualPeak(radius=1.25, angle=30.0, radius_width=0.5,
                     angle_width=20.0, temp_id=-1)
    roi = _select_manual(v, box)
    gv = v._view.ui.graphicsView

    p_outer = gv.mapFromScene(roi.handles[OUTER].scenePos())
    _mouse_drag(qtbot, gv.viewport(), p_outer, QPoint(40, -20))
    assert box.radius_width > 0.6                       # outer edge moved out
    assert box.radius - box.radius_width / 2 == pytest.approx(1.0)  # inner kept
    assert (box.angle, box.angle_width) == pytest.approx((30.0, 20.0))

    before = _box(box)
    centre = q(box.radius, box.angle)
    p_body = gv.mapFromScene(v._plot.getViewBox().mapViewToScene(centre))
    _mouse_drag(qtbot, gv.viewport(), p_body, QPoint(-30, -30))
    after = _box(box)
    assert after[1] == pytest.approx(before[1])         # widths unchanged
    assert after[3] == pytest.approx(before[3])
    assert after[0] != pytest.approx(before[0]) or after[2] != pytest.approx(before[2])

    v.undo_last_action()
    assert _box(box) == pytest.approx(before)
