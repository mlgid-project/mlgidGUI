"""After a peak is deleted, the peak with the next id is selected.

Deleting used to leave nothing selected, so working through a frame
meant clicking the next box after every delete. Now the selection moves
to the peak of the same kind whose id is the next one after the deleted
one (gaps left by earlier deletes are skipped, like the Down / Up
stepping); when nothing comes after, the last remaining peak is
selected, and with none left the selection stays empty. A bulk delete
goes by the highest deleted id. Undo selects the restored peak(s) only,
redo goes back to the neighbour. Source: image_viewer_interact.py
``select_neighbour_of_deleted``; main_window_peaks.py
``_delete_peaks_scoped`` / ``_push_delete_undo``.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _confirm_delete(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window


def _row_sel(window, kind: str, peak_id: int, frame: int = 0) -> SelectedPeak:
    table = window.viewer._frame_peaks[frame][kind]
    i = [int(x) for x in table.ids].index(peak_id)
    return window.viewer._row_selection(kind, table, i, frame)


def _select(window, kind: str, *peak_ids: int) -> list[SelectedPeak]:
    sels = [_row_sel(window, kind, p) for p in peak_ids]
    window.viewer._set_selected(sels[0])
    window.viewer._selected_extras = list(sels[1:])
    return sels


def _selected(window) -> list[tuple[str, int]]:
    return [(s.kind, int(s.peak_id)) for s in window.viewer.selected_peaks()]


def _delete(window, sels):
    if len(sels) == 1:
        window._on_delete_peak_requested(sels[0])
    else:
        window._on_delete_peaks_requested(sels)


# -- which peak is picked ----------------------------------------------------


@pytest.mark.parametrize("deleted, expected", [
    (0, 1),       # the next id after it
    (1, 2),       # from the middle
    (2, 1),       # nothing after the last one: the last remaining
])
def test_the_next_id_is_selected(
    main_window, synthetic_nexus_with_peaks, peak_link_off, deleted, expected,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", deleted))
    assert _selected(w) == [("detected", expected)]


def test_gaps_left_by_earlier_deletes_are_skipped(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 1))           # ids 0, 2 remain
    assert _selected(w) == [("detected", 2)]
    _delete(w, _select(w, "detected", 0))           # next after 0 is 2, not 1
    assert _selected(w) == [("detected", 2)]


def test_nothing_is_selected_when_nothing_is_left(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "fitted", 0))
    assert _selected(w) == [("fitted", 1)]
    _delete(w, _select(w, "fitted", 1))
    assert _selected(w) == []


def test_the_kind_stays_the_kind_that_was_deleted(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "fitted", 0))
    assert [k for k, _ in _selected(w)] == ["fitted"]


def test_a_linked_delete_selects_in_the_deleted_kind(
    main_window, synthetic_nexus_with_peaks,
):
    """With the fitted/detected link on (the default) deleting a
    detection takes its fit too; the selection is the next DETECTED."""
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 0))
    assert _selected(w) == [("detected", 1)]


def test_a_bulk_delete_goes_by_the_highest_deleted_id(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 0, 1))
    assert _selected(w) == [("detected", 2)]

    w2 = main_window
    path = w2.session.temp_path
    assert len(file_model.load_peaks(path, "entry_0000", 0)["detected"]) == 1


def test_a_bulk_delete_of_the_top_ids_falls_back_to_the_last_left(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 1, 2))
    assert _selected(w) == [("detected", 0)]


def test_fitted_rows_hidden_by_the_tracked_filter_are_skipped(
    main_window, synthetic_nexus_with_peaks, peak_link_off, monkeypatch,
):
    """A row hidden by 'show only tracked peaks' cannot be selected on
    screen, so it is never picked."""
    w = _open(main_window, synthetic_nexus_with_peaks)
    monkeypatch.setattr(
        w.viewer, "_fitted_row_hidden", lambda frame, pid: int(pid) == 1,
    )
    _delete(w, _select(w, "fitted", 0))
    assert _selected(w) == []


# -- what is shown ------------------------------------------------------------


def test_the_parameter_panel_follows_the_new_selection(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 0))
    sel = w.viewer.selected_peak
    assert sel is not None and int(sel.peak_id) == 1
    assert w.viewer._roi_item is not None          # editable: handles shown


# -- undo and redo ---------------------------------------------------------


def test_undo_selects_the_restored_peak_not_the_neighbour_too(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 1))
    assert _selected(w) == [("detected", 2)]

    w.viewer.undo_last_action()
    restored = sorted(
        int(i) for i in w.viewer._frame_peaks[0]["detected"].ids
    )
    assert len(restored) == 3
    assert len(_selected(w)) == 1                   # the restored peak only
    assert _selected(w)[0][0] == "detected"
    assert _selected(w)[0][1] not in (0, 2)         # neither of the others


def test_redo_selects_the_neighbour_again(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 1))
    w.viewer.undo_last_action()
    w.viewer.redo_last_action()
    assert _selected(w) == [("detected", 2)]


def test_undo_keeps_a_selection_the_user_made_after_the_delete(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    """The restored peak joins what the user picked since, as it always
    did; only the automatic neighbour is replaced."""
    w = _open(main_window, synthetic_nexus_with_peaks)
    _delete(w, _select(w, "detected", 1))           # auto-selects id 2
    w.viewer._set_selected(_row_sel(w, "detected", 0))   # user picks id 0
    w.viewer.undo_last_action()
    ids = sorted(i for _, i in _selected(w))
    assert 0 in ids and len(ids) == 2


# -- the other delete routes --------------------------------------------------


def test_the_peaks_table_delete_selects_the_next_too(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    w = _open(main_window, synthetic_nexus_with_peaks)
    sels = [_row_sel(w, "detected", 0)]
    w._on_delete_peaks_from_table(sels)
    assert _selected(w) == [("detected", 1)]



# -- the Peaks table keeps up (a real mouse click and the real Delete key) ----


def _table_ids(table) -> list[int]:
    from PySide6.QtCore import Qt
    return [
        int(ix.sibling(ix.row(), 0).data(Qt.ItemDataRole.DisplayRole))
        for ix in table.selectionModel().selectedRows()
    ]


def _click_row(qtbot, window, peak_id: int):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    panel = window.peaks_table_panel
    panel._tabs.setCurrentIndex(0)
    table = panel._detected_table
    model = table.model()
    row = next(
        r for r in range(model.rowCount())
        if int(model.index(r, 0).data(Qt.ItemDataRole.DisplayRole)) == peak_id
    )
    QTest.mouseClick(
        table.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        table.visualRect(model.index(row, 1)).center(),
    )
    qtbot.wait(50)
    return table


def test_peaks_table_delete_leaves_the_table_on_the_next_peak(
    main_window, synthetic_nexus_with_peaks, peak_link_off, qtbot,
):
    """Reported: deleting peak 5 from the Peaks table selected the FIRST
    peak. The table ignored the selection the delete made (it treated it
    as its own echo), kept no current row, and the next key press or
    focus change started again from its first row."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = main_window
    w.resize(1300, 800)
    w.show()
    qtbot.waitExposed(w)
    _open(w, synthetic_nexus_with_peaks)
    table = _click_row(qtbot, w, 0)
    assert _selected(w) == [("detected", 0)]

    table.setFocus()
    QTest.keyClick(table, Qt.Key.Key_Delete)
    qtbot.wait(100)

    assert _selected(w) == [("detected", 1)]
    assert _table_ids(table) == [1]                 # the row is highlighted
    assert table.currentIndex().isValid()

    # Keeps working from there: Down goes on to the next row (id 2), not
    # back to the top, and a second Delete removes the highlighted peak.
    QTest.keyClick(table, Qt.Key.Key_Down)
    qtbot.wait(50)
    assert _selected(w) == [("detected", 2)]
    QTest.keyClick(table, Qt.Key.Key_Delete)
    qtbot.wait(100)
    ids = sorted(int(i) for i in w.viewer._frame_peaks[0]["detected"].ids)
    assert ids == [1]                               # 0 and 2 are gone
    assert _selected(w) == [("detected", 1)]
    assert _table_ids(table) == [1]


def test_deleting_twice_in_a_row_walks_through_the_ids(
    main_window, synthetic_nexus_with_peaks, peak_link_off, qtbot,
):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = main_window
    w.resize(1300, 800)
    w.show()
    qtbot.waitExposed(w)
    _open(w, synthetic_nexus_with_peaks)
    table = _click_row(qtbot, w, 0)
    table.setFocus()
    seen = []
    for _ in range(2):
        QTest.keyClick(table, Qt.Key.Key_Delete)
        qtbot.wait(100)
        seen.append(_selected(w))
    assert seen == [[("detected", 1)], [("detected", 2)]]
    assert _table_ids(table) == [2]
