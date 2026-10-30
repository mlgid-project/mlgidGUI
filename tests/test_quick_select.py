"""Quick-select labelling: drawing the next box commits the previous one.

Labelling a frame used to cost four gestures per peak — draw, cross to
the Display dock, press Add to detected, come back. With quick select on
it is draw, draw, draw: each box is written to the file the moment the
user starts the next one.

The mechanism is the single-manual-box policy that was already there
(``_on_draw_finished`` replaces any existing box on the frame). Quick
select changes what happens to the box being replaced — committed rather
than discarded — and every other path that would silently drop a pending
box now commits it too.

Everything here is asserted twice: with the mode on, and with it off to
prove the mode is genuinely inert. Tests drive the viewer's real draw
handler rather than poking ``_manual_peaks``, because *when* the commit
fires is the whole feature.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.image_viewer import ManualPeak, SelectedPeak
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui

ENTRY = "entry_0000"


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    """A labelling run must never be interrupted by a dialog — including
    the failure paths, which is why this is autouse here."""
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


@pytest.fixture(autouse=True)
def _clean_quick_target():
    """The target persists in QSettings; keep one test's pick out of the
    next one's default."""
    from PySide6.QtCore import QSettings
    from mlgidgui.main_window_constants import QUICK_TARGET_KEY

    QSettings().remove(QUICK_TARGET_KEY)
    yield
    QSettings().remove(QUICK_TARGET_KEY)


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window.session.temp_path


def _ids(path, kind, frame=0) -> list[int]:
    t = file_model.load_peaks(path, ENTRY, frame)[kind]
    return [] if t is None else [int(v) for v in t.ids]


def _rows(path, kind, frame=0):
    return file_model.load_peaks(path, ENTRY, frame)[kind]


def _quick(window, on: bool, target: str | None = None):
    if target is not None:
        combo = window.parameter_panel.combo_quick_target
        combo.setCurrentIndex(combo.findData(target))
    window.parameter_panel.chk_quick_select.setChecked(on)


def _draw(window, radius: float, angle: float, dr: float = 0.2, da: float = 6.0):
    """Draw a box through the viewer's real draw handler.

    Polar mode: x = radius, y = angle — the same convention
    ``_on_draw_finished`` reads off the drag.
    """
    window.viewer._on_draw_finished(
        QPointF(radius - dr / 2, angle - da / 2),
        QPointF(radius + dr / 2, angle + da / 2),
    )
    return window.viewer._manual_peaks[window.viewer.current_frame][0]


def _manual_count(window, frame=0) -> int:
    return len(window.viewer._manual_peaks.get(frame, []))


# -- The panel controls -------------------------------------------------

def test_the_mode_is_off_and_the_target_is_detected_by_default(main_window):
    panel = main_window.parameter_panel
    assert panel.quick_select_enabled() is False
    assert panel.quick_select_target() == ParameterPanel.QUICK_TARGET_DETECTED
    # The target only means something while the mode is on.
    assert not panel.combo_quick_target.isEnabled()
    panel.chk_quick_select.setChecked(True)
    assert panel.combo_quick_target.isEnabled()


def test_the_target_is_remembered_but_the_mode_is_not(main_window, qtbot):
    """A labelling run outlives one session, so re-picking the target
    every launch is friction. Coming back *in* the mode would be a
    surprise, since it changes what a drag writes."""
    panel = main_window.parameter_panel
    _quick(main_window, True, ParameterPanel.QUICK_TARGET_FITTED)

    fresh = ParameterPanel()
    qtbot.addWidget(fresh)
    assert fresh.quick_select_target() == ParameterPanel.QUICK_TARGET_FITTED
    assert fresh.quick_select_enabled() is False


def test_turning_the_mode_on_arms_the_viewer(main_window):
    assert main_window.viewer.quick_select is False
    _quick(main_window, True)
    assert main_window.viewer.quick_select is True
    _quick(main_window, False)
    assert main_window.viewer.quick_select is False


# -- The trigger: drawing the next box ----------------------------------

def test_drawing_the_next_box_commits_the_previous_one(
    main_window, synthetic_nexus_with_peaks,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    before = _ids(path, "detected")

    first = _draw(main_window, 1.4, 30.0)
    assert _ids(path, "detected") == before  # nothing yet
    _draw(main_window, 2.6, 70.0)

    ids = _ids(path, "detected")
    assert len(ids) == len(before) + 1
    det = _rows(path, "detected")
    i = len(det) - 1
    assert float(det.radius[i]) == pytest.approx(first.radius)
    assert float(det.angle[i]) == pytest.approx(first.angle)
    assert float(det.score[i]) == pytest.approx(
        main_window.parameter_panel.confidence_score()
    )
    # Only the new box is still a manual candidate.
    assert _manual_count(main_window) == 1


def test_with_the_mode_off_the_box_is_replaced_as_before(
    main_window, synthetic_nexus_with_peaks,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    before = _ids(path, "detected")
    _draw(main_window, 1.4, 30.0)
    _draw(main_window, 2.6, 70.0)
    assert _ids(path, "detected") == before
    assert _manual_count(main_window) == 1


def test_a_box_drawn_over_the_pending_one_is_a_correction(
    main_window, synthetic_nexus_with_peaks,
):
    """'The next box in a different location' is what commits. A box on
    top of the pending one is the user fixing an attempt they were not
    happy with, and must cost nothing."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    before = _ids(path, "detected")

    _draw(main_window, 1.4, 30.0)
    corrected = _draw(main_window, 1.42, 31.0)  # centre inside the first box

    assert _ids(path, "detected") == before
    assert _manual_count(main_window) == 1
    assert main_window.viewer._manual_peaks[0][0] is corrected


# -- The other triggers -------------------------------------------------

def test_clicking_away_commits_the_pending_box(
    main_window, synthetic_nexus_with_peaks,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    n_before = len(_ids(path, "detected"))
    _draw(main_window, 1.4, 30.0)

    # Selecting anything else is the "done with that box" gesture.
    main_window.viewer._set_selected(None)
    assert len(_ids(path, "detected")) == n_before + 1
    assert _manual_count(main_window) == 0


def test_escape_still_discards_the_box(
    main_window, synthetic_nexus_with_peaks,
):
    """Esc is how you throw away a bad label, and quick select must not
    turn it into a commit."""
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    before = _ids(path, "detected")
    _draw(main_window, 1.4, 30.0)

    main_window.viewer.keyPressEvent(QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert _ids(path, "detected") == before
    assert _manual_count(main_window) == 0


def test_enter_commits_the_pending_box(
    main_window, synthetic_nexus_with_peaks,
):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    n_before = len(_ids(path, "detected"))
    _draw(main_window, 1.4, 30.0)

    main_window.viewer.keyPressEvent(QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return,
        Qt.KeyboardModifier.NoModifier,
    ))
    assert len(_ids(path, "detected")) == n_before + 1
    assert _manual_count(main_window) == 0


def test_changing_frame_commits_against_the_frame_it_was_drawn_on(
    main_window, synthetic_nexus_with_peaks,
):
    """Manual boxes survive a frame switch, so without this the box
    would be committed later against whatever frame the user ends on."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    n_before = len(_ids(path, "detected", 0))
    _draw(main_window, 1.4, 30.0)

    main_window.viewer.set_frame(1)
    assert len(_ids(path, "detected", 0)) == n_before + 1
    assert _manual_count(main_window, 0) == 0


def test_turning_the_mode_off_commits_what_is_pending(
    main_window, synthetic_nexus_with_peaks,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    n_before = len(_ids(path, "detected"))
    _draw(main_window, 1.4, 30.0)

    _quick(main_window, False)
    assert len(_ids(path, "detected")) == n_before + 1


# -- Undo ---------------------------------------------------------------

def test_undo_turns_the_committed_peak_back_into_a_box(
    main_window, synthetic_nexus_with_peaks,
):
    """One entry covers the write and the box's retirement, so a single
    Ctrl+Z gives the user back exactly what they drew."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    before = _ids(path, "detected")
    first = _draw(main_window, 1.4, 30.0)
    main_window.viewer._set_selected(None)     # commits it
    assert len(_ids(path, "detected")) == len(before) + 1

    main_window.viewer.undo_last_action()
    assert _ids(path, "detected") == before
    assert _manual_count(main_window) == 1
    assert main_window.viewer._manual_peaks[0][0] is first

    main_window.viewer.redo_last_action()
    assert len(_ids(path, "detected")) == len(before) + 1
    assert _manual_count(main_window) == 0


def test_the_undo_ladder_after_commit_and_draw(
    main_window, synthetic_nexus_with_peaks,
):
    """Draw B (which commits A): the first Ctrl+Z removes B, the second
    turns A back into a box. Each press undoes one visible thing."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _quick(main_window, True)
    before = _ids(path, "detected")
    first = _draw(main_window, 1.4, 30.0)
    second = _draw(main_window, 2.6, 70.0)
    assert len(_ids(path, "detected")) == len(before) + 1

    main_window.viewer.undo_last_action()
    assert second not in main_window.viewer._manual_peaks.get(0, [])
    assert len(_ids(path, "detected")) == len(before) + 1

    main_window.viewer.undo_last_action()
    assert _ids(path, "detected") == before
    assert main_window.viewer._manual_peaks[0][0] is first


# -- Targets ------------------------------------------------------------

def _stub_fit(window, monkeypatch, row=None):
    monkeypatch.setattr(
        window, "_build_fitted_row_2d",
        lambda sel, entry, frame, **_kw: dict(row or {
            "radius": 1.4, "radius_width": 0.2, "angle": 30.0,
            "angle_width": 6.0, "amplitude": 12.0,
            "theta": 0.0, "A": 1.0, "B": 0.0, "C": 1.0,
        }),
    )


def test_the_fitted_target_writes_a_linked_pair(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """With the peak link on (the default), the fitted path writes the
    detected partner itself — so one box becomes one pair sharing an id,
    and the two features compose."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _quick(main_window, True, ParameterPanel.QUICK_TARGET_FITTED)
    det_before = set(_ids(path, "detected"))
    fit_before = set(_ids(path, "fitted"))

    _draw(main_window, 1.4, 30.0)
    main_window.viewer._set_selected(None)

    new_det = set(_ids(path, "detected")) - det_before
    new_fit = set(_ids(path, "fitted")) - fit_before
    assert len(new_det) == 1
    assert new_det == new_fit


def test_a_failed_fit_keeps_the_box_as_detected(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """The Add-to-fitted button is strict on purpose (F-06): it refuses
    and says so. A quick commit inverts that trade — the user has moved
    on, so the box is kept as detected rather than lost, and no modal
    interrupts the run (the autouse fixture fails the test if one
    appears)."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    # Fail where the real failure happens, so the no-modal promise is
    # asserted against the code that would raise the dialog rather than
    # against a stub that skips it.
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (None, "did not converge"),
    )
    _quick(main_window, True, ParameterPanel.QUICK_TARGET_FITTED)
    det_before = len(_ids(path, "detected"))
    fit_before = len(_ids(path, "fitted"))

    _draw(main_window, 1.4, 30.0)
    main_window.viewer._set_selected(None)

    assert len(_ids(path, "detected")) == det_before + 1
    assert len(_ids(path, "fitted")) == fit_before
    assert _manual_count(main_window) == 0


def test_the_status_cell_shows_the_mode_and_a_pending_box(
    main_window, synthetic_nexus_with_peaks,
):
    """The Display dock is tabbed and scrollable, so the mode needs
    somewhere it cannot hide."""
    _open(main_window, synthetic_nexus_with_peaks)
    assert not main_window._sb_quick.isVisible() or main_window._sb_quick.text() == ""

    main_window.show()
    _quick(main_window, True)
    assert "detected" in main_window._sb_quick.text()
    assert "pending" not in main_window._sb_quick.text()

    _draw(main_window, 1.4, 30.0)
    assert "pending" in main_window._sb_quick.text()

    _quick(main_window, False)
    assert not main_window._sb_quick.isVisible()
