"""Correcting a detected peak's confidence score.

The Score row used to be a read-only readout for an existing peak, so a
model's call could not be corrected without re-labelling from scratch.
It is now an editor (High / Medium / Low, or a number), it applies to
the whole selection, and it lands as ONE undo entry -- which is what
makes labelling a validation set workable.

The invariant threaded through most of these: a score edit rewrites the
score column and NOTHING else. Moving a peak because its label changed
would be silent data loss.
"""
from __future__ import annotations

import pytest

from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _detected(window, frame: int = 0):
    return (window.viewer._frame_peaks.get(frame) or {})["detected"]


def _select_detected(window, frame: int, n: int) -> list[SelectedPeak]:
    det = _detected(window, frame)
    sels = [
        SelectedPeak(
            kind="detected", frame=frame, peak_id=int(det.ids[i]),
            radius=float(det.radius[i]), angle=float(det.angle[i]),
            radius_width=float(det.radius_width[i]),
            angle_width=float(det.angle_width[i]),
            is_ring=bool(det.is_ring[i]),
            score=float(det.score[i]),
            amplitude=float(det.amplitude[i]),
        )
        for i in range(n)
    ]
    window.viewer._set_selected(sels[0])
    window.viewer._selected_extras = list(sels[1:])
    return sels


def _scores_on_disk(session, entry="entry_0000", frame=0):
    table = file_model.load_peaks(session.temp_path, entry, frame)["detected"]
    return [float(s) for s in table.score]


# --- the write itself -------------------------------------------------


def test_a_score_write_leaves_the_geometry_alone(synthetic_nexus_with_peaks):
    """The whole point of the optional-field signature.

    ``update_peak_row`` used to demand all four polar fields, so a score
    edit could only be expressed as "rewrite everything", which risks
    writing back a rounded copy of the geometry.
    """
    path = synthetic_nexus_with_peaks
    before = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    peak_id = int(before.ids[1])
    kept = (
        float(before.radius[1]), float(before.angle[1]),
        float(before.q_xy[1]), float(before.q_z[1]),
    )

    file_model.update_peak_row(
        path, "entry_0000", 0, "detected", peak_id, score=0.25,
    )

    after = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    assert float(after.score[1]) == pytest.approx(0.25)
    assert (
        float(after.radius[1]), float(after.angle[1]),
        float(after.q_xy[1]), float(after.q_z[1]),
    ) == pytest.approx(kept)


def test_a_geometry_write_still_leaves_the_score_alone(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    before = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    peak_id = int(before.ids[0])
    score = float(before.score[0])

    file_model.update_peak_row(
        path, "entry_0000", 0, "detected", peak_id,
        radius=1.5, angle=30.0, radius_width=0.1, angle_width=5.0,
    )

    after = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    assert float(after.score[0]) == pytest.approx(score)
    assert float(after.radius[0]) == pytest.approx(1.5)
    # q_xy/q_z ARE recomputed here, because the polar pair was given.
    assert float(after.q_xy[0]) != pytest.approx(float(before.q_xy[0]))


# --- through the GUI --------------------------------------------------


def test_a_preset_relabels_the_whole_selection_in_one_undo(
    main_window, synthetic_nexus_with_peaks,
):
    """Ctrl-select a handful, press Low, change your mind, one Ctrl+Z."""
    window = main_window
    session = _open(window, synthetic_nexus_with_peaks)
    before = _scores_on_disk(session)
    assert len(before) >= 3

    _select_detected(window, 0, 3)
    window.parameter_panel._score_preset_buttons[2].click()  # "Low" = 0.1

    assert _scores_on_disk(session)[:3] == pytest.approx([0.1, 0.1, 0.1])
    # The cached table follows too -- the min-score filter and the Peaks
    # table read it, so without this the peak keeps its old score on
    # screen until the frame is reloaded.
    assert [float(s) for s in _detected(window).score][:3] == pytest.approx(
        [0.1, 0.1, 0.1]
    )

    window.viewer.undo_last_action()
    assert _scores_on_disk(session)[:3] == pytest.approx(before[:3])
    window.viewer.redo_last_action()
    assert _scores_on_disk(session)[:3] == pytest.approx([0.1, 0.1, 0.1])


def test_the_presets_write_the_numbers_the_app_already_meant_by_them(
    main_window, synthetic_nexus_with_peaks,
):
    """High / Medium / Low were already 1.0 / 0.5 / 0.1 for a manual
    commit; the editor must not invent a second meaning."""
    from mlgidgui.parameter_panel import ParameterPanel

    window = main_window
    session = _open(window, synthetic_nexus_with_peaks)
    for i, (_label, value) in enumerate(ParameterPanel.CONFIDENCE_LEVELS):
        _select_detected(window, 0, 1)
        window.parameter_panel._score_preset_buttons[i].click()
        assert _scores_on_disk(session)[0] == pytest.approx(value)


def test_the_spin_box_writes_any_value_in_range(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    session = _open(window, synthetic_nexus_with_peaks)
    _select_detected(window, 0, 1)
    window.parameter_panel._score_spin.setValue(0.375)
    window.parameter_panel._score_spin.editingFinished.emit()
    assert _scores_on_disk(session)[0] == pytest.approx(0.375)


def test_a_mixed_selection_relabels_only_its_detected_members(
    main_window, synthetic_nexus_with_peaks,
):
    """A mixed selection is normal; refusing the whole write would be
    useless, so non-detected members are skipped."""
    window = main_window
    session = _open(window, synthetic_nexus_with_peaks)
    fitted_before = [
        float(s)
        for s in file_model.load_peaks(
            session.temp_path, "entry_0000", 0
        )["fitted"].score
    ]

    sels = _select_detected(window, 0, 2)
    fit = (window.viewer._frame_peaks.get(0) or {})["fitted"]
    window.viewer._selected_extras = list(sels[1:]) + [
        SelectedPeak(
            kind="fitted", frame=0, peak_id=int(fit.ids[0]),
            radius=float(fit.radius[0]), angle=float(fit.angle[0]),
            radius_width=float(fit.radius_width[0]),
            angle_width=float(fit.angle_width[0]),
            score=float(fit.score[0]),
        )
    ]
    window.parameter_panel._score_preset_buttons[0].click()  # High = 1.0

    assert _scores_on_disk(session)[:2] == pytest.approx([1.0, 1.0])
    after = file_model.load_peaks(session.temp_path, "entry_0000", 0)["fitted"]
    assert [float(s) for s in after.score] == pytest.approx(fitted_before)


# --- when the editor is and is not offered ----------------------------


def test_the_editor_shows_for_a_detected_peak_and_not_for_a_fitted_one(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _open(window, synthetic_nexus_with_peaks)
    panel = window.parameter_panel

    _select_detected(window, 0, 1)
    assert panel._score_stack.currentWidget() is panel._score_editor

    fit = (window.viewer._frame_peaks.get(0) or {})["fitted"]
    window.viewer._set_selected(SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        score=float(fit.score[0]),
    ))
    assert panel._score_stack.currentWidget() is panel._score_label


def test_the_spin_box_cannot_write_to_whatever_was_selected_next(
    main_window, synthetic_nexus_with_peaks,
):
    """``editingFinished`` also fires when focus leaves on a selection
    change, so the emit is gated on the editor still being the page."""
    window = main_window
    session = _open(window, synthetic_nexus_with_peaks)
    panel = window.parameter_panel

    _select_detected(window, 0, 1)
    panel._score_spin.setValue(0.42)
    before = _scores_on_disk(session)

    window.viewer.clear_selection()
    panel._score_spin.editingFinished.emit()

    assert _scores_on_disk(session) == pytest.approx(before)


def test_the_count_line_appears_only_for_a_real_batch(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _open(window, synthetic_nexus_with_peaks)
    panel = window.parameter_panel

    _select_detected(window, 0, 1)
    panel.set_selection_count(1)
    assert panel._score_count_label.text() == ""

    _select_detected(window, 0, 3)
    panel.set_selection_count(3)
    assert "3 peaks selected" in panel._score_count_label.text()
