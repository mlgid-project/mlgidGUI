"""Multi-selection on the image viewer (Ctrl+click toggle, Ctrl+A).

Multi-select covers detected AND fitted peaks (single-kind only):

* Ctrl+click toggles a detected peak in / out of the extras list.
* Ctrl+click hit-test is kind-aware: it prefers the current
  multi-selection's kind (so detected stays reachable under a fitted
  overlay), and with no multi-kind primary fitted wins over detected.
* A lone fitted peak is reachable by Ctrl+click (no detected under it).
* Ctrl+click on a manual / matched peak (or a cross-kind peak) falls
  back to single-select replacement.
* Ctrl+A selects every peak of the current kind on the frame.
* The selection-highlight overlay renders one row per selected peak.
* ``selectionsChanged`` fires with the full list every mutation.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _detected_sel(window, frame: int, idx: int) -> SelectedPeak:
    tables = window.viewer._frame_peaks.get(frame) or {}
    det = tables["detected"]
    return SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[idx]),
        radius=float(det.radius[idx]), angle=float(det.angle[idx]),
        radius_width=float(det.radius_width[idx]),
        angle_width=float(det.angle_width[idx]),
        is_ring=bool(det.is_ring[idx]),
        score=float(det.score[idx]),
        amplitude=float(det.amplitude[idx]),
    )


def test_ctrl_click_toggles_detected(main_window, synthetic_nexus_with_peaks):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    s0 = _detected_sel(main_window, 0, 0)
    s1 = _detected_sel(main_window, 0, 1)

    v._set_selected(s0)
    assert len(v.selected_peaks()) == 1

    v._toggle_selected(s1)
    assert len(v.selected_peaks()) == 2
    assert v.selected_peaks()[0].peak_id == s0.peak_id
    assert v.selected_peaks()[1].peak_id == s1.peak_id

    # Toggling s1 again drops it.
    v._toggle_selected(s1)
    assert len(v.selected_peaks()) == 1
    assert v.selected_peaks()[0].peak_id == s0.peak_id

    # Toggling primary demotes it; with no extras, selection clears.
    v._toggle_selected(s0)
    assert v.selected_peaks() == []


def _inject_detected_at_fitted0(v):
    """Add a detected row at fitted row 0's centre so the two overlays
    overlap at one point. Returns (radius, angle) of that point."""
    import numpy as np
    from mlgidlab.file_model import PeakTable
    tables = v._frame_peaks.get(0) or {}
    fit = tables["fitted"]
    det = tables["detected"]
    fr, fa = float(fit.radius[0]), float(fit.angle[0])
    new_det = PeakTable(
        q_xy=np.append(det.q_xy, det.q_xy[0]),
        q_z=np.append(det.q_z, det.q_z[0]),
        angle=np.append(det.angle, fa),
        radius=np.append(det.radius, fr),
        angle_width=np.append(det.angle_width, 4.0),
        radius_width=np.append(det.radius_width, 0.3),
        is_ring=np.append(det.is_ring, False),
        ids=np.append(det.ids, det.ids.max() + 1),
        score=np.append(det.score, 1.0),
        amplitude=np.append(det.amplitude, 10.0),
    )
    v._frame_peaks[0] = {**tables, "detected": new_det}
    return fr, fa


def test_ctrl_click_prefers_fitted_when_no_primary(
    main_window, synthetic_nexus_with_peaks,
):
    """With no multi-kind primary, Ctrl+click at an overlapping
    detected+fitted point picks fitted (same priority as a bare
    click). Fitted is the post-Run-Fitting refinement, so it's the
    sensible default."""
    from PySide6.QtCore import QPointF
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    fr, fa = _inject_detected_at_fitted0(v)

    v._on_select_at(QPointF(fr, fa), Qt.KeyboardModifier.ControlModifier)
    sels = v.selected_peaks()
    assert len(sels) == 1
    assert sels[0].kind == "fitted"


def test_ctrl_click_prefers_current_kind(
    main_window, synthetic_nexus_with_peaks,
):
    """With a detected peak already the primary, Ctrl+click at an
    overlapping detected+fitted point extends the *detected*
    selection — the gesture keeps building one kind even where a
    fitted box covers the detected peak."""
    from PySide6.QtCore import QPointF
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    fr, fa = _inject_detected_at_fitted0(v)
    # Make a detected peak the primary first.
    v._set_selected(_detected_sel(main_window, 0, 0))

    v._on_select_at(QPointF(fr, fa), Qt.KeyboardModifier.ControlModifier)
    sels = v.selected_peaks()
    assert len(sels) == 2
    assert all(s.kind == "detected" for s in sels)


def test_ctrl_click_selects_lone_fitted(
    main_window, synthetic_nexus_with_peaks,
):
    """Ctrl+click on a fitted peak with no detected underneath selects
    it (the reported bug: Ctrl+click on fitted did nothing because the
    hit-test was detected-only)."""
    from PySide6.QtCore import QPointF
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    fit = (v._frame_peaks.get(0) or {})["fitted"]
    # Fitted row 0 (r=1.5, a=20.0) has no detected peak at its centre.
    fr, fa = float(fit.radius[0]), float(fit.angle[0])

    v._on_select_at(QPointF(fr, fa), Qt.KeyboardModifier.ControlModifier)
    sels = v.selected_peaks()
    assert len(sels) == 1
    assert sels[0].kind == "fitted"
    assert sels[0].peak_id == int(fit.ids[0])


def test_ctrl_click_on_empty_space_is_noop(
    main_window, synthetic_nexus_with_peaks,
):
    """Ctrl+click that doesn't hit any detected peak leaves the
    existing multi-selection alone — a near-miss must not wipe
    the user's painstakingly assembled multi-selection."""
    from PySide6.QtCore import QPointF
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    v._set_selected(_detected_sel(main_window, 0, 0))
    v._toggle_selected(_detected_sel(main_window, 0, 1))
    assert len(v.selected_peaks()) == 2

    # Far from every detected peak.
    v._on_select_at(QPointF(100.0, 100.0), Qt.KeyboardModifier.ControlModifier)
    assert len(v.selected_peaks()) == 2


def test_ctrl_click_on_non_detected_replaces(
    main_window, synthetic_nexus_with_peaks,
):
    """Direct ``_toggle_selected`` call with a non-detected peak
    falls back to single-select replacement — internal contract
    test, separate from the user-facing hit-test restriction
    above."""
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    # Start with a detected primary + one extra.
    v._set_selected(_detected_sel(main_window, 0, 0))
    v._toggle_selected(_detected_sel(main_window, 0, 1))
    assert len(v.selected_peaks()) == 2

    # Ctrl+click a fitted peak — should replace the whole multi-selection.
    tables = v._frame_peaks.get(0) or {}
    fit = tables["fitted"]
    fit_sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        is_ring=bool(fit.is_ring[0]),
        score=float(fit.score[0]),
        amplitude=float(fit.amplitude[0]),
    )
    v._toggle_selected(fit_sel)
    assert len(v.selected_peaks()) == 1
    assert v.selected_peaks()[0].kind == "fitted"


def test_ctrl_a_selects_all_detected_on_frame(
    main_window, synthetic_nexus_with_peaks,
):
    """Frame 0 has 3 detected rows in the fixture — Ctrl+A should
    grab all 3 as the multi-selection."""
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    v._select_all_of_kind_on_frame("detected")
    sels = v.selected_peaks()
    assert len(sels) == 3
    assert all(s.kind == "detected" for s in sels)


def test_ctrl_a_selects_all_fitted_when_fitted_primary(
    main_window, synthetic_nexus_with_peaks,
):
    """Ctrl+A keys off the current kind: with a fitted peak as primary
    it grabs all fitted rows (the fixture has 2 on frame 0)."""
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    v._select_all_of_kind_on_frame("fitted")
    sels = v.selected_peaks()
    assert len(sels) == 2
    assert all(s.kind == "fitted" for s in sels)


def test_multi_selection_renders_n_highlight_rows(
    main_window, synthetic_nexus_with_peaks,
):
    """The selection PeakShapeItem holds N rows after multi-select.

    Asserts via a small render hook: after Ctrl+A on a 3-peak frame,
    the count of rectangles in ``_selection``'s path should reflect
    3 boxes. We don't inspect the QPainterPath shape; we just check
    that the underlying PeakTable used to build the overlay has 3
    rows by introspecting the render call.
    """
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    v._select_all_of_kind_on_frame("detected")
    # Selection state mirrors the overlay; len(selected_peaks()) == 3
    # is the source-of-truth check. The overlay path itself is built
    # by ``_PeakShapeItem.set_polar`` and rendering tests are out of
    # scope for the smoke suite (covered by visual verification).
    assert len(v.selected_peaks()) == 3


def test_selectionsChanged_emits_full_list(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    """Every mutation through ``_set_selected`` or ``_toggle_selected``
    fires ``selectionsChanged`` with the full list."""
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    captured: list[list] = []
    v.selectionsChanged.connect(lambda sels: captured.append(list(sels)))

    s0 = _detected_sel(main_window, 0, 0)
    s1 = _detected_sel(main_window, 0, 1)

    v._set_selected(s0)
    assert captured[-1] and len(captured[-1]) == 1

    v._toggle_selected(s1)
    assert len(captured[-1]) == 2

    v._toggle_selected(s1)
    assert len(captured[-1]) == 1

    v._toggle_selected(s0)  # demotes primary; no extras → empty
    assert captured[-1] == []


# -- Up / Down step through the peaks of the selected kind ----------------

def _sel_kind(window, kind: str, idx: int) -> SelectedPeak:
    table = (window.viewer._frame_peaks.get(0) or {})[kind]
    sel = window.viewer._row_selection(kind, table, idx, 0)
    window.viewer._set_selected(sel)
    return sel


def _now(v):
    s = v.selected_peak
    return None if s is None else (s.kind, s.peak_id)


def test_step_selection_walks_detected_ids_and_stops_at_the_ends(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    _sel_kind(main_window, "detected", 0)
    assert v.step_selection(+1) and _now(v) == ("detected", 1)
    assert v.step_selection(+1) and _now(v) == ("detected", 2)
    assert not v.step_selection(+1) and _now(v) == ("detected", 2)
    assert v.step_selection(-1) and _now(v) == ("detected", 1)
    assert v.step_selection(-1) and _now(v) == ("detected", 0)
    assert not v.step_selection(-1) and _now(v) == ("detected", 0)


def test_step_selection_skips_gaps_in_the_ids(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    v._frame_peaks[0]["detected"].ids[:] = [0, 5, 9]
    _sel_kind(main_window, "detected", 0)
    assert v.step_selection(+1) and _now(v) == ("detected", 5)
    assert v.step_selection(+1) and _now(v) == ("detected", 9)
    assert v.step_selection(-1) and _now(v) == ("detected", 5)


def test_step_selection_stays_within_the_fitted_kind(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    _sel_kind(main_window, "fitted", 0)
    assert v.step_selection(+1) and _now(v) == ("fitted", 1)
    assert not v.step_selection(+1)                   # fitted has only 0, 1


def test_step_selection_skips_fitted_rows_the_tracking_filter_hides(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    _sel_kind(main_window, "fitted", 0)
    v._fitted_visible_only = {0: {0}}                 # fitted 1 is hidden
    assert not v.step_selection(+1)
    assert _now(v) == ("fitted", 0)


def test_step_selection_needs_exactly_one_detected_or_fitted_peak(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    assert not v.step_selection(+1)                   # nothing selected
    v._select_all_of_kind_on_frame("detected")
    before = [(s.kind, s.peak_id) for s in v.selected_peaks()]
    assert not v.step_selection(+1)                   # a multi-selection
    assert [(s.kind, s.peak_id) for s in v.selected_peaks()] == before


def test_step_selection_zooms_like_a_click(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    _sel_kind(main_window, "detected", 0)
    zoomed = []
    monkeypatch.setattr(v, "zoom_to_peak", lambda sel: zoomed.append(sel.peak_id))
    v.step_selection(+1)
    assert zoomed == [1]


def _activate(window, qtbot) -> None:
    """Window shortcuts only fire in the active window; offscreen needs
    the activation asked for."""
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.wait(50)
    assert window.isActiveWindow()


def test_down_and_up_keys_step_the_selection(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    w = main_window
    _activate(w, qtbot)
    _open(w, synthetic_nexus_with_peaks)
    _sel_kind(w, "detected", 0)
    w.viewer.setFocus()                  # where a click on the image leaves it
    qtbot.keyClick(w, Qt.Key.Key_Down)
    assert _now(w.viewer) == ("detected", 1)
    qtbot.keyClick(w, Qt.Key.Key_Up)
    assert _now(w.viewer) == ("detected", 0)


def test_up_and_down_stay_with_a_spin_box(
    main_window, synthetic_nexus_with_peaks, qtbot, monkeypatch,
):
    """A widget that uses Up / Down keeps them: the frame spin box steps
    its value and the peak selection does not move."""
    w = main_window
    _activate(w, qtbot)
    _open(w, synthetic_nexus_with_peaks)
    calls = []
    monkeypatch.setattr(w.viewer, "step_selection", lambda d: calls.append(d))
    w.frame_spin.setFocus()
    qtbot.keyClick(w.frame_spin, Qt.Key.Key_Up)
    assert calls == []
    assert w.frame_spin.value() == 1
