"""Batch 2D fit of the multi-selected detected peaks.

Five scenarios (all stub out ``_run_pygidfit_for_selection`` so the
real pygidfit isn't invoked). The row-count ones take
``peak_link_off``: they count appended rows, which is the unlinked
contract. With the fitted/detected link on (the shipped default) a
batch fit writes each fit under its detected peak's id and replaces
that peak's previous fit, so the count is the number of *detected*
peaks — covered in ``test_peak_linking.py``.

* One fitted row appended per selected detected peak on success.
* Button is disabled when 'Save fitted as ring' is on (ring forces 1D).
* Cancel mid-batch keeps already-written rows; the rest are skipped.
* Single Ctrl+Z undo removes every appended row.
* Partial failure — pygidfit returns ``(None, err)`` on some peaks,
  succeeds on others — continues through the batch and reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

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


@dataclass(frozen=True)
class _StubFitResult:
    radius: float = 1.5
    radius_width: float = 0.22
    angle: float = 45.0
    angle_width: float = 6.0
    amplitude: float = 100.0
    A: float = 1.0
    B: float = 0.0
    C: float = 1.0
    theta: float = 0.0


def _select_all_three_detected(window):
    window.viewer._select_all_of_kind_on_frame("detected")
    assert len(window.viewer.selected_peaks()) == 3


def test_batch_fit_appends_one_fitted_row_per_selected(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """3 detected selected, pygidfit stubbed → 3 fitted rows added."""
    from mlgidgui import file_model
    _open(main_window, synthetic_nexus_with_peaks)
    _select_all_three_detected(main_window)

    n_fit_before = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_StubFitResult(
            radius=float(sel.radius), angle=float(sel.angle),
        ), None),
    )
    main_window._on_batch_fit_2d()
    n_fit_after = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    assert n_fit_after == n_fit_before + 3


def test_batch_fit_button_disabled_when_ring_on(
    main_window, synthetic_nexus_with_peaks,
):
    """Ring storage forces 1D, which doesn't batch — button hides
    (and disables; both happen together). ``main_window.show()`` so
    ``isVisible()`` doesn't return False just because the parent
    isn't on screen yet."""
    _open(main_window, synthetic_nexus_with_peaks)
    main_window.show()
    _select_all_three_detected(main_window)
    # Selection alone shows + enables the button.
    main_window._refresh_fit_buttons()
    assert main_window.parameter_panel.btn_fit_selected_2d.isVisible()
    assert main_window.parameter_panel.btn_fit_selected_2d.isEnabled()
    # Tick 'Save fitted as ring' — button hides + disables.
    main_window.parameter_panel.chk_save_as_ring.setChecked(True)
    main_window._refresh_fit_buttons()
    assert not main_window.parameter_panel.btn_fit_selected_2d.isVisible()
    assert not main_window.parameter_panel.btn_fit_selected_2d.isEnabled()


def test_fit_buttons_visibility_swap(
    main_window, synthetic_nexus_with_peaks,
):
    """Single fittable selection shows Add to fitted; ≥2 detected
    selections swap to Fit selected (2D)."""
    _open(main_window, synthetic_nexus_with_peaks)
    main_window.show()  # buttons need a visible parent for isVisible()
    v = main_window.viewer
    # Single detected → Add to fitted visible, batch hidden
    v._set_selected(_detected_sel(main_window, 0, 0))
    main_window._refresh_fit_buttons()
    assert main_window.parameter_panel.btn_add_fitted.isVisible()
    assert not main_window.parameter_panel.btn_fit_selected_2d.isVisible()
    # Add a second via Ctrl+click toggle → swap
    v._toggle_selected(_detected_sel(main_window, 0, 1))
    main_window._refresh_fit_buttons()
    assert not main_window.parameter_panel.btn_add_fitted.isVisible()
    assert main_window.parameter_panel.btn_fit_selected_2d.isVisible()
    # Clear selection → both hidden
    v.clear_selection()
    main_window._refresh_fit_buttons()
    assert not main_window.parameter_panel.btn_add_fitted.isVisible()
    assert not main_window.parameter_panel.btn_fit_selected_2d.isVisible()


def test_batch_fit_cancel_mid_loop_keeps_partial(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """Cancel after the first write keeps row 1 but not 2/3."""
    from mlgidgui import file_model
    _open(main_window, synthetic_nexus_with_peaks)
    _select_all_three_detected(main_window)

    call_count = {"n": 0}

    def _fit(sel, entry, frame):
        call_count["n"] += 1
        return (_StubFitResult(
            radius=float(sel.radius), angle=float(sel.angle),
        ), None)

    monkeypatch.setattr(main_window, "_run_pygidfit_for_selection", _fit)

    n_fit_before = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])

    # Cancel the dialog after the first iteration completes by
    # monkeypatching QProgressDialog.wasCanceled. Pattern: the loop
    # consults wasCanceled() at the top of each iteration; return True
    # after one successful write.
    from PySide6.QtWidgets import QProgressDialog
    state = {"steps_done": 0}
    original_setValue = QProgressDialog.setValue

    def _setValue(self, v):
        state["steps_done"] = v
        original_setValue(self, v)

    def _wasCanceled(self):
        # Cancel as soon as one row has been written.
        return state["steps_done"] >= 1

    monkeypatch.setattr(QProgressDialog, "setValue", _setValue)
    monkeypatch.setattr(QProgressDialog, "wasCanceled", _wasCanceled)

    main_window._on_batch_fit_2d()

    n_fit_after = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    # Exactly one row written before the cancel kicked in.
    assert n_fit_after == n_fit_before + 1
    assert call_count["n"] == 1


def test_batch_fit_undo_removes_all_appended(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """One Ctrl+Z reverses the whole batch — all N appended rows go."""
    from mlgidgui import file_model
    _open(main_window, synthetic_nexus_with_peaks)
    _select_all_three_detected(main_window)
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_StubFitResult(
            radius=float(sel.radius), angle=float(sel.angle),
        ), None),
    )
    n_before = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    main_window._on_batch_fit_2d()
    n_after_fit = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    assert n_after_fit == n_before + 3

    main_window.viewer.undo_last_action()
    n_after_undo = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    assert n_after_undo == n_before


def test_batch_fit_partial_failure_continues(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """pygidfit fails on the 2nd peak; the 1st and 3rd still write."""
    from mlgidgui import file_model
    _open(main_window, synthetic_nexus_with_peaks)
    _select_all_three_detected(main_window)

    calls = {"i": 0}

    def _fit(sel, entry, frame):
        i = calls["i"]
        calls["i"] += 1
        if i == 1:
            return (None, "no convergence")
        return (_StubFitResult(
            radius=float(sel.radius), angle=float(sel.angle),
        ), None)

    monkeypatch.setattr(main_window, "_run_pygidfit_for_selection", _fit)
    n_before = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    main_window._on_batch_fit_2d()
    n_after = len(file_model.load_peaks(
        main_window.session.temp_path, "entry_0000", 0,
    )["fitted"])
    # 2 successes (peaks 1 and 3) + 1 failure (peak 2) → 2 new rows.
    assert n_after == n_before + 2


def test_batch_fit_leaves_no_progress_dialog_behind(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """The progress dialog is parented to the window; once the batch is
    done it has to go, or every batch leaves one behind."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QProgressDialog

    _open(main_window, synthetic_nexus_with_peaks)
    _select_all_three_detected(main_window)
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_StubFitResult(
            radius=float(sel.radius), angle=float(sel.angle),
        ), None),
    )
    before = len(main_window.findChildren(QProgressDialog))
    main_window._on_batch_fit_2d()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(main_window.findChildren(QProgressDialog)) == before
