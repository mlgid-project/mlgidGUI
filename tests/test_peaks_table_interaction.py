"""Selecting and deleting peaks from the Peaks dock, not just the image.

Two additions, both asked for after the tables proved more precise than
clicking small boxes on the image:

* **Ctrl / Shift multi-select** on the Detected and Fitted tabs, driving
  the same multi-selection Ctrl+click on the image drives. The Matched
  tab stays single-select — a row there is a whole structure, so
  "several rows" and "delete" would both mean something else.
* **Delete** with rows selected in the table, routed into the same host
  handlers the image-side Delete uses, so both routes share one
  confirmation, one write path and one undo entry.

``QTableView`` swallows unhandled key presses instead of letting them
bubble, so the Delete tests press the real key on the real widget
rather than calling the handler: that swallowing is exactly what the
``_PeakTableView`` subclass exists to undo.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QTableView

from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _confirm(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window.session.temp_path


def _ids(path, kind, frame=0) -> list[int]:
    t = file_model.load_peaks(path, "entry_0000", frame)[kind]
    return [] if t is None else [int(v) for v in t.ids]


def _table(window, kind):
    return {
        "detected": window.peaks_table_panel._detected_table,
        "fitted": window.peaks_table_panel._fitted_table,
        "matched": window.peaks_table_panel._matched_table,
    }[kind]


def _select_rows(window, kind: str, rows: list[int]):
    """Select ``rows`` (view order) the way Ctrl+click would, with the
    last one current — that is the row the panel promotes to primary."""
    table = _table(window, kind)
    model = table.selectionModel()
    model.clearSelection()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    for r in rows:
        model.select(table.model().index(r, 0), flags)
    model.setCurrentIndex(
        table.model().index(rows[-1], 0),
        QItemSelectionModel.SelectionFlag.NoUpdate,
    )
    return table


def test_the_detected_and_fitted_tabs_take_more_than_one_row(main_window):
    panel = main_window.peaks_table_panel
    for kind in ("detected", "fitted"):
        assert (
            _table(main_window, kind).selectionMode()
            == QTableView.SelectionMode.ExtendedSelection
        ), kind
    # A matched row is a structure, not a peak.
    assert (
        panel._matched_table.selectionMode()
        == QTableView.SelectionMode.SingleSelection
    )


def test_selecting_two_rows_multi_selects_them_on_the_image(
    main_window, synthetic_nexus_with_peaks,
):
    _open(main_window, synthetic_nexus_with_peaks)
    _select_rows(main_window, "detected", [0, 1])
    sels = main_window.viewer.selected_peaks()
    assert len(sels) == 2
    assert {s.kind for s in sels} == {"detected"}
    # The current row is the primary, so the Parameter panel and the
    # resize ROI follow the peak the user touched last.
    assert main_window.viewer.selected_peak.peak_id == 1
    # And the frame is stamped: the panel builds every row with frame=0
    # because it has no viewer context.
    assert all(s.frame == main_window.viewer.current_frame for s in sels)


def test_a_single_row_still_goes_through_the_old_path(
    main_window, synthetic_nexus_with_peaks,
):
    """One row must not emit the multi signal as well, or the viewer
    would do the same work twice for every ordinary click."""
    _open(main_window, synthetic_nexus_with_peaks)
    seen: list = []
    main_window.peaks_table_panel.peaksSelectedFromTable.connect(seen.append)
    _select_rows(main_window, "detected", [0])
    assert seen == []
    assert len(main_window.viewer.selected_peaks()) == 1


def test_an_image_multi_selection_shows_up_in_the_table(
    main_window, synthetic_nexus_with_peaks,
):
    """Without this the table would highlight one row while the image
    showed three, and a Delete in the table would act on a set the
    user cannot see."""
    _open(main_window, synthetic_nexus_with_peaks)
    main_window.viewer._select_all_of_kind_on_frame("detected")
    table = _table(main_window, "detected")
    assert len(table.selectionModel().selectedRows()) == 3
    assert main_window.peaks_table_panel._tabs.currentIndex() == 0


def test_delete_in_the_table_removes_every_selected_peak(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    table = _select_rows(main_window, "detected", [0, 1])
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _ids(path, "detected") == [2]


def test_delete_in_the_table_works_on_one_row_too(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    table = _select_rows(main_window, "fitted", [0])
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _ids(path, "fitted") == [1]


def test_delete_in_the_table_is_undoable_like_the_image_one(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Both routes end in the same handler, so the undo entry is the
    same one — this asserts the routing, not a second implementation."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    table = _select_rows(main_window, "detected", [0, 1])
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _ids(path, "detected") == [2]
    main_window.viewer.undo_last_action()
    assert sorted(_ids(path, "detected")) == [0, 1, 2]


def test_delete_asks_before_removing_anything(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    table = _select_rows(main_window, "detected", [0, 1])
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _ids(path, "detected") == [0, 1, 2]


def test_delete_on_the_matched_tab_does_nothing(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """A matched row is a structure. Deleting one is a different
    operation with different semantics, so the key is not wired there
    and the tab is left as it was."""
    _open(main_window, synthetic_nexus_with_peaks)
    seen: list = []
    main_window.peaks_table_panel.deletePeaksRequested.connect(seen.append)
    table = _table(main_window, "matched")
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert seen == []


def test_delete_is_ignored_while_a_pipeline_run_is_in_flight(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    table = _select_rows(main_window, "detected", [0, 1])
    monkeypatch.setattr(main_window, "_is_busy", lambda: True)
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _ids(path, "detected") == [0, 1, 2]


def test_a_modifier_delete_is_left_to_the_view(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Only a bare Delete deletes; Ctrl+Delete and friends stay
    available to the view (and to any future binding)."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    table = _select_rows(main_window, "detected", [0, 1])
    QTest.keyClick(table, Qt.Key.Key_Delete, Qt.KeyboardModifier.ControlModifier)
    assert _ids(path, "detected") == [0, 1, 2]


def test_a_real_ctrl_click_adds_the_second_row(
    main_window, synthetic_nexus_with_peaks,
):
    """The programmatic tests above set the selection model directly.
    This one goes through the mouse, because the two signals a
    Ctrl+click emits do not arrive in a fixed order and the panel has
    to end up on the multi-selection either way."""
    _open(main_window, synthetic_nexus_with_peaks)
    # The dock is tabbed with the Profiles dock; a click only lands on
    # the table when its tab is the visible one.
    main_window.show()
    main_window._peaks_dock.show()
    main_window._peaks_dock.raise_()
    QApplication.processEvents()
    table = _table(main_window, "detected")
    for row, modifier in (
        (0, Qt.KeyboardModifier.NoModifier),
        (1, Qt.KeyboardModifier.ControlModifier),
    ):
        idx = table.model().index(row, 0)
        QTest.mouseClick(
            table.viewport(), Qt.MouseButton.LeftButton, modifier,
            table.visualRect(idx).center(),
        )
    assert len(table.selectionModel().selectedRows()) == 2
    sels = main_window.viewer.selected_peaks()
    assert [s.peak_id for s in sels] == [1, 0]
