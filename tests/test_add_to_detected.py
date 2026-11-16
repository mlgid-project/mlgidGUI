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


def test_score_of_committed_box_editable_after_the_fact(
    main_window, synthetic_nexus_with_peaks, tmp_path,
):
    """A detected peak added from a manual box keeps an editable score.

    The manual box stays on screen exactly over the new row, and manual
    boxes come first in the click order, so a later click used to select
    the box -- whose dropdown only sets the score of a FUTURE commit --
    and the stored score could not be reached. The committed box now
    ranks below detected: the click selects the stored peak, its Score
    row is the editor, and the edit reaches the file and survives a save.
    A second click on the same spot still cycles to the manual box.
    """
    from PySide6.QtCore import QPointF

    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    v = main_window.viewer
    # Clear of the fixture's own boxes, so only this box and its
    # committed row sit under the click.
    box = ManualPeak(
        radius=3.6, angle=30.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    v.add_manual_peak(0, box)
    v._set_selected(SelectedPeak.from_manual(box, 0))
    _set_confidence(main_window, "High")
    main_window._on_add_to_detected()
    new_id = int(file_model.load_peaks(path, "entry_0000", 0)["detected"].ids[-1])
    assert box.committed_detected_id == new_id

    # Later: select something else, then click the box's spot again.
    v.clear_selection()
    v._mode = "polar"
    v._on_select_at(QPointF(box.radius, box.angle))
    sel = v.selected_peak
    assert (sel.kind, int(sel.peak_id)) == ("detected", new_id)
    panel = main_window.parameter_panel
    assert panel._score_stack.currentWidget() is panel._score_editor

    main_window._on_score_edit_requested(0.3)
    assert _new_row_score(path, "entry_0000", 0) == pytest.approx(0.3, abs=1e-6)
    out = tmp_path / "saved.h5"
    main_window.session.save_as(out)
    t = file_model.load_peaks(out, "entry_0000", 0)["detected"]
    assert float(t.score[list(t.ids).index(new_id)]) == pytest.approx(0.3, abs=1e-6)

    # Same spot again: the manual box is still reachable.
    v._on_select_at(QPointF(box.radius, box.angle))
    assert v.selected_peak.kind == "manual"


def test_uncommitted_manual_box_still_wins_the_click(
    main_window, synthetic_nexus_with_peaks,
):
    """Only committed boxes are demoted; a fresh one keeps priority."""
    from PySide6.QtCore import QPointF

    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    t = file_model.load_peaks(main_window.session.temp_path, "entry_0000", 0)["detected"]
    r, a = float(t.radius[0]), float(t.angle[0])
    p = ManualPeak(radius=r, angle=a, radius_width=0.3, angle_width=8.0, temp_id=7)
    v.add_manual_peak(0, p)
    v.clear_selection()
    v._mode = "polar"
    v._on_select_at(QPointF(r, a))
    assert v.selected_peak.kind == "manual"


def _dock_cell(window, kind, peak_id, col):
    from mlgidgui.peaks_table_panel import _find_row_for_id_in_model

    panel = window.peaks_table_panel
    model = panel._detected_model if kind == "detected" else panel._fitted_model
    row = _find_row_for_id_in_model(model, peak_id)
    return model.item(row, col).text()


def test_score_edit_updates_peaks_dock_in_place(
    main_window, synthetic_nexus_with_peaks,
):
    """Editing the score of the selected detected peak (Display dock)
    shows the new value in the Peaks dock at once, keeps the dock's
    row selection, and is what lands in the file."""
    _open(main_window, synthetic_nexus_with_peaks)
    w = main_window
    path = w.session.temp_path
    t = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    pid = int(t.ids[1])
    w.peaks_table_panel._detected_table.selectRow(1)  # dock-driven select
    sel = w.viewer.selected_peak
    assert (sel.kind, int(sel.peak_id)) == ("detected", pid)
    assert _dock_cell(w, "detected", pid, 5) == f"{float(t.score[1]):.3f}"

    w._on_score_edit_requested(0.123)
    assert _dock_cell(w, "detected", pid, 5) == "0.123"
    t2 = file_model.load_peaks(path, "entry_0000", 0)["detected"]
    assert float(t2.score[1]) == pytest.approx(0.123, abs=1e-6)
    rows = w.peaks_table_panel._detected_table.selectionModel().selectedRows()
    assert len(rows) == 1

    # Undo restores the dock as well as the file.
    w.viewer.undo_last_action()
    assert _dock_cell(w, "detected", pid, 5) == f"{float(t.score[1]):.3f}"


def test_drag_write_updates_peaks_dock(main_window, synthetic_nexus_with_peaks):
    """A geometry write (what a drag commits) patches the row too."""
    _open(main_window, synthetic_nexus_with_peaks)
    w = main_window
    pid = int(file_model.load_peaks(
        w.session.temp_path, "entry_0000", 0)["detected"].ids[0])
    w._on_peak_row_write_requested(
        0, "detected", pid, {"radius": 1.234, "angle": 12.5},
    )
    assert _dock_cell(w, "detected", pid, 1) == "1.234"
    assert _dock_cell(w, "detected", pid, 3) == "12.50"
