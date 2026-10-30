"""Reversible edits and the history that holds them.

No Qt: an op takes an open ``h5py.File``, so the whole undo model is
testable without a window. The invariant every test here is really
checking is that ``undo`` and ``redo`` are exact mirrors — walk the
stack in either direction, any number of times, and the file matches.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import h5_edit
from mlgidgui.h5_edit import MISSING
from mlgidgui.h5_edit_ops import (
    EditHistory,
    RenameAttrOp,
    SetAttrOp,
    WriteCellsOp,
)


@pytest.fixture
def file_handle(tmp_path):
    path = tmp_path / "ops.h5"
    with h5py.File(path, "w", track_order=True) as f:
        group = f.create_group("meta", track_order=True)
        group.attrs["units"] = "counts"
        group.attrs["scale"] = np.float64(2.5)
        group.create_dataset("counts", data=np.arange(5, dtype="i4"))
        group.create_dataset("temperature", data=np.float32(297.5))
    with h5py.File(path, "r+") as f:
        yield f


# -- SetAttrOp covers create, change and delete ---------------------------


def test_change_an_attribute_and_reverse_it(file_handle):
    op = SetAttrOp("/meta", "units", "counts", "photons")
    op.redo(file_handle)
    assert file_handle["meta"].attrs["units"] == "photons"
    op.undo(file_handle)
    assert file_handle["meta"].attrs["units"] == "counts"


def test_creating_an_attribute_is_a_set_with_no_before(file_handle):
    op = SetAttrOp("/meta", "comment", MISSING, "hello")
    op.redo(file_handle)
    assert file_handle["meta"].attrs["comment"] == "hello"
    op.undo(file_handle)
    assert "comment" not in file_handle["meta"].attrs


def test_deleting_an_attribute_is_a_set_with_no_after(file_handle):
    op = SetAttrOp("/meta", "units", "counts", MISSING)
    op.redo(file_handle)
    assert "units" not in file_handle["meta"].attrs
    op.undo(file_handle)
    assert file_handle["meta"].attrs["units"] == "counts"


def test_undoing_a_delete_that_already_happened_is_not_an_error(file_handle):
    """Reaching the intended state is the contract, not how it got there."""
    del file_handle["meta"].attrs["units"]
    SetAttrOp("/meta", "units", "counts", MISSING).redo(file_handle)
    assert "units" not in file_handle["meta"].attrs


@pytest.mark.parametrize(
    "op,expected",
    [
        (SetAttrOp("/meta", "units", "counts", "photons"),
         "/meta@units: counts → photons"),
        (SetAttrOp("/meta", "note", MISSING, "hi"), "+ /meta@note = hi"),
        (SetAttrOp("/meta", "note", "hi", MISSING), "- /meta@note  (was hi)"),
        (RenameAttrOp("/meta", "a", "b"), "/meta@a renamed to b"),
    ],
)
def test_descriptions_read_as_a_changelog(op, expected):
    assert op.describe() == expected


# -- rename ----------------------------------------------------------------


def test_rename_round_trips(file_handle):
    op = RenameAttrOp("/meta", "units", "unit")
    op.redo(file_handle)
    assert "unit" in file_handle["meta"].attrs
    assert "units" not in file_handle["meta"].attrs
    op.undo(file_handle)
    assert file_handle["meta"].attrs["units"] == "counts"


# -- cell writes -----------------------------------------------------------


def test_write_cells_round_trips(file_handle):
    op = WriteCellsOp("/meta/counts", {0: 0, 3: 3}, {0: 90, 3: 93})
    op.redo(file_handle)
    assert list(file_handle["meta/counts"][()]) == [90, 1, 2, 93, 4]
    op.undo(file_handle)
    assert list(file_handle["meta/counts"][()]) == [0, 1, 2, 3, 4]


def test_a_single_cell_write_describes_both_sides():
    op = WriteCellsOp("/meta/counts", {2: 7}, {2: 9})
    assert op.describe() == "/meta/counts[2]: 7 → 9"


def test_a_multi_cell_write_describes_the_count():
    op = WriteCellsOp("/x", {0: 1, 1: 2}, {0: 3, 1: 4})
    assert op.describe() == "/x: 2 cells changed"


def test_a_scalar_write_names_the_dataset_without_an_index():
    op = WriteCellsOp("/meta/temperature", {(): 297.5}, {(): 300.0})
    assert op.describe().startswith("/meta/temperature: 297.5 → 300.0")


# -- the history -----------------------------------------------------------


def test_undo_and_redo_walk_the_stack(file_handle):
    history = EditHistory()
    first = SetAttrOp("/meta", "units", "counts", "photons")
    second = SetAttrOp("/meta", "scale", np.float64(2.5), np.float64(4.0))
    for op in (first, second):
        op.redo(file_handle)
        history.push(op)

    assert history.undo(file_handle) is second
    assert file_handle["meta"].attrs["scale"] == 2.5
    assert history.undo(file_handle) is first
    assert file_handle["meta"].attrs["units"] == "counts"
    assert history.undo(file_handle) is None

    assert history.redo(file_handle) is first
    assert file_handle["meta"].attrs["units"] == "photons"
    assert history.redo(file_handle) is second
    assert history.redo(file_handle) is None


def test_a_new_edit_drops_the_redo_branch(file_handle):
    history = EditHistory()
    history.push(SetAttrOp("/meta", "units", "counts", "photons"))
    history.undo(file_handle)
    assert history.can_redo
    history.push(SetAttrOp("/meta", "note", MISSING, "x"))
    assert not history.can_redo


def test_the_changes_list_is_the_undo_stack(file_handle):
    """Undoing takes the line back off the list — the only reading of
    'what I have changed' that stays honest."""
    history = EditHistory()
    history.push(SetAttrOp("/meta", "units", "counts", "photons"))
    history.push(SetAttrOp("/meta", "note", MISSING, "x"))
    assert history.entries() == [
        "/meta@units: counts → photons", "+ /meta@note = x"
    ]
    history.undo(file_handle)
    assert history.entries() == ["/meta@units: counts → photons"]


def test_as_text_joins_the_entries():
    history = EditHistory()
    history.push(SetAttrOp("/meta", "a", MISSING, "1"))
    history.push(SetAttrOp("/meta", "b", MISSING, "2"))
    assert history.as_text() == "+ /meta@a = 1\n+ /meta@b = 2"


def test_the_history_is_bounded(file_handle):
    history = EditHistory(limit=3)
    for i in range(6):
        history.push(SetAttrOp("/meta", f"a{i}", MISSING, str(i)))
    assert len(history) == 3
    assert history.entries()[0] == "+ /meta@a3 = 3"


def test_clear_empties_both_stacks(file_handle):
    history = EditHistory()
    history.push(SetAttrOp("/meta", "units", "counts", "photons"))
    history.undo(file_handle)
    history.clear()
    assert not history.can_undo and not history.can_redo


# -- attribute typing ------------------------------------------------------


def test_an_edited_attribute_keeps_its_type(file_handle):
    """Typing 4 into a float64 attribute must not turn it into a string."""
    template = file_handle["meta"].attrs["scale"]
    value = h5_edit.parse_attr_value("4", template)
    assert isinstance(value, np.floating)
    h5_edit.set_attr(file_handle, "/meta", "scale", value)
    assert file_handle["meta"].attrs["scale"].dtype.kind == "f"


def test_a_string_attribute_takes_the_text_as_typed(file_handle):
    assert h5_edit.parse_attr_value(" 2.5 ", "counts") == " 2.5 "


def test_a_flat_array_attribute_is_comma_separated(file_handle):
    file_handle["meta"].attrs["window"] = np.array([1, 2, 3], dtype="i4")
    template = file_handle["meta"].attrs["window"]
    value = h5_edit.parse_attr_value("4, 5", template)
    assert list(value) == [4, 5]
    assert value.dtype == np.dtype("i4")


def test_a_new_attribute_with_no_template_is_text():
    assert h5_edit.parse_attr_value("hello", MISSING) == "hello"


def test_a_two_dimensional_attribute_is_not_inline_editable():
    assert not h5_edit.is_inline_editable(np.zeros((2, 2)))
    assert h5_edit.is_inline_editable(np.zeros(3))
    assert h5_edit.is_inline_editable("text")


@pytest.mark.parametrize(
    "value,label",
    [("x", "str"), (np.float32(1), "float32"), (True, "bool"),
     (np.array([1, 2], dtype="i4"), "int32[2]"), (MISSING, "str")],
)
def test_attr_type_labels(value, label):
    assert h5_edit.attr_type_label(value) == label
