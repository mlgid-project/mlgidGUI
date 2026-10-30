"""The value grid: editing what a dataset holds.

Stage seven of the editor. The grid itself is tested directly, then
driven through ``MainWindow`` to pin the part that matters most — that
an editing session in the grid becomes ONE undoable edit, sparse when
the shape is unchanged and a whole-array replace when rows moved.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt
from PySide6.QtWidgets import QDialog, QMessageBox

from mlgidgui import h5_edit
from mlgidgui.array_edit_dialog import ArrayEditDialog, CompoundTableModel
from mlgidgui.h5_edit_ops import ReplaceDataOp
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui

PEAK_DTYPE = np.dtype([("amplitude", "f4"), ("id", "i4"), ("is_ring", "bool")])


def _peak_rows(n=3):
    rows = np.zeros(n, dtype=PEAK_DTYPE)
    rows["amplitude"] = np.arange(n, dtype="f4") + 1.0
    rows["id"] = np.arange(n, dtype="i4")
    return rows


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


# -- the compound model ----------------------------------------------------


@pytest.fixture
def model(qtbot) -> CompoundTableModel:
    return CompoundTableModel(_peak_rows())


def test_one_column_per_field(model):
    assert model.columnCount() == 3
    assert model.rowCount() == 3
    assert [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(3)] == ["amplitude", "id", "is_ring"]


def test_a_cell_edit_keeps_the_fields_type(model):
    index = model.index(0, 0)
    assert model.setData(index, "2.5")
    assert model.array()["amplitude"][0] == pytest.approx(2.5)
    assert model.array().dtype == PEAK_DTYPE


def test_a_boolean_field_takes_words(model):
    assert model.setData(model.index(1, 2), "true")
    assert bool(model.array()["is_ring"][1])
    assert model.setData(model.index(1, 2), "no")
    assert not bool(model.array()["is_ring"][1])


def test_nonsense_is_refused_rather_than_stored_as_zero(model):
    before = float(model.array()["amplitude"][0])
    assert not model.setData(model.index(0, 0), "abc")
    assert float(model.array()["amplitude"][0]) == before


def test_a_read_only_model_has_no_editable_flag():
    model = CompoundTableModel(_peak_rows(), editable=False)
    assert not (model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable)


def test_insert_copies_the_row_above(model):
    model.insert_rows(2)
    assert model.rowCount() == 4
    assert int(model.array()["id"][2]) == 1, "the row above is duplicated"


def test_insert_at_the_top_makes_a_blank_row(model):
    model.insert_rows(0)
    assert int(model.array()["id"][0]) == 0
    assert float(model.array()["amplitude"][0]) == 0.0


def test_delete_removes_the_named_rows(model):
    model.delete_rows([0, 2])
    assert model.rowCount() == 1
    assert int(model.array()["id"][0]) == 1


def test_delete_ignores_rows_that_are_not_there(model):
    model.delete_rows([99])
    assert model.rowCount() == 3


# -- the dialog ------------------------------------------------------------


def test_a_compound_dataset_gets_the_field_table(qtbot):
    dialog = ArrayEditDialog(path="/t", array=_peak_rows())
    qtbot.addWidget(dialog)
    assert dialog.is_compound
    assert dialog._grid is None and dialog._model is not None
    assert "amplitude" in dialog._header.text()


def test_a_numeric_dataset_gets_the_silx_grid(qtbot):
    dialog = ArrayEditDialog(
        path="/t", array=np.arange(24, dtype="f4").reshape(2, 3, 4))
    qtbot.addWidget(dialog)
    assert not dialog.is_compound
    assert dialog._grid is not None
    assert "2 × 3 × 4" in dialog._header.text()


def test_the_numeric_grid_hands_the_array_back(qtbot):
    array = np.arange(6, dtype="f4").reshape(2, 3)
    dialog = ArrayEditDialog(path="/t", array=array)
    qtbot.addWidget(dialog)
    assert np.array_equal(dialog.result_array(), array)


def test_a_read_only_dialog_says_why_and_can_be_unlocked(qtbot):
    dialog = ArrayEditDialog(
        path="/t", array=_peak_rows(), editable=False,
        read_only_reason="1,000,000 values",
    )
    qtbot.addWidget(dialog)
    assert dialog._reason.isVisibleTo(dialog)
    assert not dialog._insert_btn.isEnabled()

    dialog._unlock()
    assert dialog._insert_btn.isEnabled()
    assert not dialog._reason.isVisibleTo(dialog)
    assert dialog._model.flags(dialog._model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable


def test_row_buttons_act_on_the_selection(qtbot):
    dialog = ArrayEditDialog(path="/t", array=_peak_rows())
    qtbot.addWidget(dialog)
    dialog._table.selectionModel().select(
        dialog._model.index(1, 0),
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    dialog._on_insert()
    assert dialog._model.rowCount() == 4
    dialog._table.selectionModel().clearSelection()
    dialog._table.selectionModel().select(
        dialog._model.index(0, 0),
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    dialog._on_delete()
    assert dialog._model.rowCount() == 3


# -- through the window ----------------------------------------------------


class _FakeNode:
    def __init__(self, filename, local_name: str) -> None:
        self.local_filename = str(filename)
        self.local_name = local_name


@pytest.fixture
def editing(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    return main_window


def _show(window, h5_path, monkeypatch):
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


def _read(window):
    handle = window._h5_edit_handle
    if handle is not None and handle.is_open:
        return handle.file
    return h5py.File(window.session.temp_path, "r")


def _stub_grid(monkeypatch, transform, *, accept=True):
    """Replace the dialog with one that applies ``transform`` to the array."""
    from mlgidgui import main_window_structure as mws

    class _Stub:
        def __init__(self, parent=None, *, path="", array=None,
                     editable=True, read_only_reason=""):
            self.array = np.asarray(array)
            self.editable = editable
            self.reason = read_only_reason

        def deleteLater(self):
            pass

        def exec(self):
            return (QDialog.DialogCode.Accepted if accept
                    else QDialog.DialogCode.Rejected)

        def result_array(self):
            return transform(np.array(self.array, copy=True))

    monkeypatch.setattr(mws, "ArrayEditDialog", _Stub)
    return _Stub


def _add_array(window, path="/entry_0000/values"):
    """An ordinary, unprotected 1-D dataset to edit.

    The q axes are protected, so editing one asks first — which is a
    different behaviour, tested on its own below.
    """
    with window._detached_silx_tree():
        with h5py.File(window.session.temp_path, "r+") as f:
            parent, name = path.rsplit("/", 1)
            f[parent].create_dataset(
                name, data=np.arange(5, dtype="f4"))


def _add_table(window, path="/entry_0000/peaks", rows=None):
    with window._detached_silx_tree():
        with h5py.File(window.session.temp_path, "r+") as f:
            parent, name = path.rsplit("/", 1)
            f[parent].create_dataset(
                name, data=_peak_rows() if rows is None else rows)


def test_editing_a_cell_writes_it(editing, monkeypatch):
    def bump(array):
        array[0] = 9.0
        return array

    _add_array(editing)
    _stub_grid(monkeypatch, bump)
    _show(editing, "/entry_0000/values", monkeypatch)
    editing._on_structure_edit_values()
    assert float(_read(editing)["entry_0000/values"][0]) == pytest.approx(9.0)


def test_an_unchanged_grid_records_nothing(editing, monkeypatch):
    """OK on a grid nobody typed into asks nothing and records nothing,
    even on a protected dataset."""
    _stub_grid(monkeypatch, lambda array: array)
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_edit_values()
    assert editing.structure_panel.change_rows() == []


def test_a_cancelled_grid_changes_nothing(editing, monkeypatch):
    def wreck(array):  # pragma: no cover - never applied
        array[:] = 0
        return array

    _stub_grid(monkeypatch, wreck, accept=False)
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    before = np.array(_read(editing)["entry_0000/data/q_xy"][()])
    editing._on_structure_edit_values()
    assert np.array_equal(_read(editing)["entry_0000/data/q_xy"][()], before)


def test_a_whole_session_of_cell_edits_is_one_undo(editing, monkeypatch):
    def edit(array):
        array[0] = 9.0
        array[1] = 8.0
        return array

    _add_array(editing)
    _stub_grid(monkeypatch, edit)
    _show(editing, "/entry_0000/values", monkeypatch)
    before = np.array(_read(editing)["entry_0000/values"][()])
    editing._on_structure_edit_values()
    assert editing.structure_panel.change_rows() == [
        "/entry_0000/values: 2 cells changed"]
    editing._structure_undo()
    assert np.array_equal(_read(editing)["entry_0000/values"][()], before)


def test_a_row_insert_becomes_a_whole_array_replace(editing, monkeypatch):
    _add_table(editing)

    def grow(array):
        return np.concatenate([array, array[-1:]])

    _stub_grid(monkeypatch, grow)
    _show(editing, "/entry_0000/peaks", monkeypatch)
    editing._on_structure_edit_values()
    assert _read(editing)["entry_0000/peaks"].shape == (4,)
    assert editing.structure_panel.change_rows() == [
        "/entry_0000/peaks: 3 → 4 rows"]


def test_a_row_change_is_undoable(editing, monkeypatch):
    _add_table(editing)
    _stub_grid(monkeypatch, lambda array: array[:1])
    _show(editing, "/entry_0000/peaks", monkeypatch)
    editing._on_structure_edit_values()
    assert _read(editing)["entry_0000/peaks"].shape == (1,)
    editing._structure_undo()
    assert _read(editing)["entry_0000/peaks"].shape == (3,)
    assert list(_read(editing)["entry_0000/peaks"]["id"]) == [0, 1, 2]


def test_a_compound_cell_edit_stays_sparse(editing, monkeypatch):
    _add_table(editing)

    def edit(array):
        array["amplitude"][1] = 42.0
        return array

    _stub_grid(monkeypatch, edit)
    _show(editing, "/entry_0000/peaks", monkeypatch)
    editing._on_structure_edit_values()
    assert editing.structure_panel.change_rows() == [
        "/entry_0000/peaks[1, amplitude]: 2.0 → 42.0"]
    editing._structure_undo()
    assert float(_read(editing)["entry_0000/peaks"]["amplitude"][1]) == 2.0


def test_a_protected_dataset_asks_before_the_grid_is_applied(
    editing, monkeypatch
):
    answers = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: answers.append(a[1:3])
                     or QMessageBox.StandardButton.Cancel),
    )
    _stub_grid(monkeypatch, lambda array: array * 2)
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    before = np.array(_read(editing)["entry_0000/data/q_xy"][()])
    editing._on_structure_edit_values()
    assert answers, "q_xy is protected and must be confirmed"
    assert np.array_equal(_read(editing)["entry_0000/data/q_xy"][()], before)


def test_a_dataset_too_big_for_the_grid_is_refused(editing, monkeypatch):
    monkeypatch.setattr(h5_edit, "VIEW_CELL_LIMIT", 4)
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_edit_values()
    assert seen and "Data tab" in str(seen[0])


def test_a_large_dataset_opens_read_only(editing, monkeypatch):
    """Between the two limits: loadable, but not something to type into."""
    monkeypatch.setattr(h5_edit, "EDITABLE_CELL_LIMIT", 4)
    captured = {}

    from mlgidgui import main_window_structure as mws

    class _Stub:
        def __init__(self, parent=None, *, path="", array=None,
                     editable=True, read_only_reason=""):
            captured["editable"] = editable
            captured["reason"] = read_only_reason

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def result_array(self):  # pragma: no cover - never reached
            return None

    monkeypatch.setattr(mws, "ArrayEditDialog", _Stub)
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_edit_values()
    assert captured["editable"] is False
    assert "read-only" in captured["reason"]


def test_the_grid_button_shows_only_for_multi_cell_datasets(editing, monkeypatch):
    panel = editing.structure_panel
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    assert panel._edit_values_btn.isVisibleTo(panel._value_card)

    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            f["entry_0000"].create_dataset("exposure", data=np.float32(0.1))
    _show(editing, "/entry_0000/exposure", monkeypatch)
    assert not panel._edit_values_btn.isVisibleTo(panel._value_card)


# -- the op, on its own ----------------------------------------------------


def test_replace_data_op_round_trips(tmp_path):
    path = tmp_path / "ops.h5"
    before = _peak_rows()
    with h5py.File(path, "w") as f:
        f.create_dataset("peaks", data=before)
    after = np.concatenate([before, before[-1:]])
    op = ReplaceDataOp("/peaks", before, after)
    with h5py.File(path, "r+") as f:
        op.redo(f)
        assert f["peaks"].shape == (4,)
        op.undo(f)
        assert f["peaks"].shape == (3,)


def test_replace_data_op_describes_a_row_change():
    before, after = _peak_rows(3), _peak_rows(5)
    assert ReplaceDataOp("/p", before, after).describe() == "/p: 3 → 5 rows"


def test_replace_data_op_describes_a_same_shape_replace():
    before = _peak_rows(3)
    assert ReplaceDataOp("/p", before, before).describe() == (
        "/p: values replaced")


# -- the diff --------------------------------------------------------------


def test_diff_finds_changed_cells_in_an_nd_array(main_window):
    before = np.zeros((2, 3), dtype="f4")
    after = before.copy()
    after[1, 2] = 5.0
    changed = main_window._diff_cells(before, after)
    assert changed == {(1, 2): np.float32(5.0)}


def test_diff_is_empty_for_an_untouched_array(main_window):
    array = np.arange(6, dtype="f4").reshape(2, 3)
    assert main_window._diff_cells(array, array.copy()) == {}


def test_diff_of_a_compound_array_is_keyed_by_row_and_field(main_window):
    before = _peak_rows()
    after = before.copy()
    after["id"][2] = 99
    changed = main_window._diff_cells(before, after)
    assert changed == {(2, "id"): np.int32(99)}


def test_diff_of_a_scalar(main_window):
    before = np.array(1.0, dtype="f4")
    after = np.array(2.0, dtype="f4")
    assert main_window._diff_cells(before, after) == {(): np.float32(2.0)}
    assert main_window._diff_cells(before, before.copy()) == {}
