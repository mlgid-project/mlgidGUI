"""Add-to-detected writes a fresh detected_peaks row at the confidence
score chosen in the Parameter-panel Score-row dropdown, and is undoable.

The Score row shows a read-only model score for an existing detected /
fitted / matched peak, but becomes a High / Medium / Low dropdown while a
manual box is selected. "Add to detected" then writes that box as a
detected peak carrying the chosen score (High = 1.0 / Medium = 0.5 /
Low = 0.1). Copy/paste is unaffected (it preserves the source score).
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.image_viewer import ManualPeak, SelectedPeak
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _open(window, path):
    window._set_active_session(NexusSession.open(path))


def _n_detected(path, entry, frame):
    t = file_model.load_peaks(path, entry, frame)["detected"]
    return 0 if t is None else len(t)


def _new_row_score(path, entry, frame):
    """Score of the last (highest-id) detected row on the frame."""
    t = file_model.load_peaks(path, entry, frame)["detected"]
    return float(t.score[len(t) - 1])


def _select_manual_box(window, frame=0):
    p = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    window.viewer.add_manual_peak(frame, p)
    window.viewer._set_selected(SelectedPeak.from_manual(p, frame))
    return p


def _set_confidence(window, label):
    combo = window.parameter_panel._confidence_combo
    idx = next(i for i in range(combo.count()) if combo.itemText(i) == label)
    combo.setCurrentIndex(idx)


@pytest.mark.parametrize("label,expected", [("High", 1.0), ("Medium", 0.5), ("Low", 0.1)])
def test_add_to_detected_writes_chosen_confidence(
    main_window, synthetic_nexus_with_peaks, label, expected,
):
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    _select_manual_box(main_window, 0)
    _set_confidence(main_window, label)
    assert _n_detected(path, "entry_0000", 0) == 3

    main_window._on_add_to_detected()
    assert _n_detected(path, "entry_0000", 0) == 4
    assert _new_row_score(path, "entry_0000", 0) == pytest.approx(expected)


def test_add_to_detected_is_undoable(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    v = main_window.viewer
    _select_manual_box(main_window, 0)
    _set_confidence(main_window, "Medium")

    main_window._on_add_to_detected()
    assert _n_detected(path, "entry_0000", 0) == 4

    v.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 3
    v.redo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 4
    # Redo re-adds with the same score.
    assert _new_row_score(path, "entry_0000", 0) == pytest.approx(0.5)


def test_add_to_detected_leaves_manual_box(
    main_window, synthetic_nexus_with_peaks,
):
    """The source manual box stays so it can also go to fitted / be tweaked."""
    _open(main_window, synthetic_nexus_with_peaks)
    _select_manual_box(main_window, 0)
    _set_confidence(main_window, "High")
    main_window._on_add_to_detected()
    assert len(main_window.viewer.manual_peaks(0)) == 1


def test_confidence_score_getter():
    panel = ParameterPanel()
    combo = panel._confidence_combo
    for label, expected in (("High", 1.0), ("Medium", 0.5), ("Low", 0.1)):
        idx = next(i for i in range(combo.count()) if combo.itemText(i) == label)
        combo.setCurrentIndex(idx)
        assert panel.confidence_score() == pytest.approx(expected)
