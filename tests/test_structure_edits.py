"""Editing a file from the Structure tab, end to end through MainWindow.

Stage three of the editor: attributes and single-cell values. Everything
here goes through the real signal handlers, so what is pinned is the
whole chain — panel intent, protected-path guard, write through the
long-lived handle, flush, dirty mark, undo history, changes list.

The last section is the one that matters most for the rest of the app:
holding an ``r+`` handle open while the user edits must leave peak
writes, saves and session teardown behaving exactly as they did before.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model, h5_edit
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    """Any unexpected modal is a failure, not a hang."""
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


@pytest.fixture
def confirm_yes(monkeypatch):
    """Answer every protected-path confirmation with Yes."""
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


@pytest.fixture
def confirm_no(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )


class _FakeNode:
    def __init__(self, filename, local_name: str) -> None:
        self.local_filename = str(filename)
        self.local_name = local_name


@pytest.fixture
def editing(main_window, synthetic_nexus):
    """A window with a registered NeXus session, showing the Structure tab.

    The session goes into ``_sessions`` — unlike the older browser tests,
    which only set it active — because the editor asks "which session owns
    this file?" before it will write anything.

    That registration is also why the save prompt is stubbed out on the
    instance. An editor test leaves unsaved changes behind by definition,
    and ``closeEvent`` prompts once per dirty session — a hang, not a
    dialog, in a headless run. It has to be an instance attribute rather
    than a monkeypatch: pytest-qt closes registered widgets from its own
    ``pytest_runtest_teardown`` hook, which runs *before* fixture
    finalizers, so anything monkeypatch would undo is already undone by
    the time the window closes. Shadowing the bound method on this one
    window is ordering-independent and cannot leak into another test.
    """
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    return main_window


def _show(window, h5_path: str, monkeypatch) -> None:
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


def _read(window, h5_path: str, attr: str | None = None):
    """Read straight from the working copy, through the open handle when
    there is one — a second ``r`` open behind the editor's ``r+`` is fine,
    but reusing the handle is what the app itself does."""
    handle = window._h5_edit_handle
    if handle is not None and handle.is_open:
        obj = handle.file[h5_path]
        return obj.attrs[attr] if attr else obj[()]
    with h5py.File(window.session.temp_path, "r") as f:
        return f[h5_path].attrs[attr] if attr else f[h5_path][()]


# -- attributes ------------------------------------------------------------


def test_editing_an_attribute_writes_it(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeEdited.emit("NX_class", "NXentry")
    assert _read(editing, "/entry_0000", "NX_class") == "NXentry"


def test_an_edit_marks_the_session_dirty(editing, monkeypatch):
    assert not editing.session.dirty
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "PM6:Y6")
    assert editing.session.dirty
    assert "•" in editing.windowTitle() or editing.session.dirty


def test_adding_an_attribute_shows_up_in_the_panel(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "PM6:Y6 as-cast")
    assert ("title", "PM6:Y6 as-cast") in editing.structure_panel.attribute_rows()


def test_adding_a_typed_attribute_stores_that_type(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("exposure", "float64", "0.5")
    value = _read(editing, "/entry_0000", "exposure")
    assert float(value) == pytest.approx(0.5)
    assert np.asarray(value).dtype.kind == "f"


def test_editing_keeps_an_attributes_type(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("exposure", "float64", "0.5")
    editing.structure_panel.attributeEdited.emit("exposure", "2")
    value = _read(editing, "/entry_0000", "exposure")
    assert np.asarray(value).dtype.kind == "f"
    assert float(value) == pytest.approx(2.0)


def test_renaming_an_attribute(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("titel", "str", "typo")
    editing.structure_panel.attributeRenamed.emit("titel", "title")
    names = [name for name, _ in editing.structure_panel.attribute_rows()]
    assert "title" in names and "titel" not in names


def test_removing_an_attribute(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("scratch", "str", "x")
    editing.structure_panel.attributeRemoved.emit("scratch")
    names = [name for name, _ in editing.structure_panel.attribute_rows()]
    assert "scratch" not in names


def test_a_bad_value_is_reported_and_changes_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("exposure", "float64", "0.5")
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: seen.append(a[1:3]) or QMessageBox.StandardButton.Ok),
    )
    editing.structure_panel.attributeEdited.emit("exposure", "not a number")
    assert seen, "the user must be told why the edit did not take"
    assert float(_read(editing, "/entry_0000", "exposure")) == pytest.approx(0.5)


# -- values ----------------------------------------------------------------


def test_setting_a_scalar_dataset_value(editing, monkeypatch):
    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            f["entry_0000"].create_dataset("exposure", data=np.float32(0.1))
    _show(editing, "/entry_0000/exposure", monkeypatch)
    editing.structure_panel.scalarValueEdited.emit("0.75")
    assert float(_read(editing, "/entry_0000/exposure")) == pytest.approx(0.75)


def test_the_value_field_only_appears_for_a_single_cell(editing, monkeypatch):
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    panel = editing.structure_panel
    assert not panel._value_row.isVisibleTo(panel._value_card)
    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            f["entry_0000"].create_dataset("exposure", data=np.float32(0.1))
    _show(editing, "/entry_0000/exposure", monkeypatch)
    assert panel._value_row.isVisibleTo(panel._value_card)


# -- the protected-path guard ---------------------------------------------


def test_changing_the_signal_attribute_asks_first(editing, monkeypatch, confirm_no):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel.attributeEdited.emit("signal", "something_else")
    assert _read(editing, "/entry_0000/data", "signal") == "img_gid_q"


def test_confirming_lets_the_change_through(editing, monkeypatch, confirm_yes):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel.attributeEdited.emit("signal", "img_gid_q_2")
    assert _read(editing, "/entry_0000/data", "signal") == "img_gid_q_2"


def test_removing_a_protected_attribute_asks_first(editing, monkeypatch, confirm_no):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel.attributeRemoved.emit("signal")
    assert "signal" in [n for n, _ in editing.structure_panel.attribute_rows()]


def test_an_ordinary_attribute_is_not_guarded(editing, monkeypatch):
    """The autouse fixture fails the test if any modal appears."""
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel.attributeAdded.emit("long_name", "str", "detector")
    assert _read(editing, "/entry_0000/data", "long_name") == "detector"


# -- undo ------------------------------------------------------------------


def test_undo_reverses_an_attribute_edit(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing.structure_panel.attributeEdited.emit("title", "second")
    assert _read(editing, "/entry_0000", "title") == "second"
    editing._action_undo()
    assert _read(editing, "/entry_0000", "title") == "first"
    editing._action_undo()
    assert "title" not in dict(_attrs(editing, "/entry_0000"))


def _attrs(window, path):
    handle = window._h5_edit_handle
    return dict(handle.file[path].attrs)


def test_redo_replays_it(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing._action_undo()
    editing._action_redo()
    assert _read(editing, "/entry_0000", "title") == "first"


def test_undo_belongs_to_the_viewer_on_the_image_tab(editing, monkeypatch):
    """The Structure tab owns Ctrl+Z only while it is in front."""
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing.tabs.setCurrentWidget(editing.viewer)
    calls = []
    monkeypatch.setattr(
        editing.viewer, "undo_last_action", lambda: calls.append(1))
    editing._action_undo()
    assert calls == [1]
    assert _read(editing, "/entry_0000", "title") == "first"


def test_undo_falls_through_when_the_editor_has_no_history(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    calls = []
    monkeypatch.setattr(
        editing.viewer, "undo_last_action", lambda: calls.append(1))
    editing._action_undo()
    assert calls == [1]


# -- the changes list ------------------------------------------------------


def test_every_edit_lands_in_the_changes_list(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing.structure_panel.attributeEdited.emit("title", "second")
    rows = editing.structure_panel.change_rows()
    assert rows == ["+ /entry_0000@title = first",
                    "/entry_0000@title: first → second"]


def test_undo_takes_the_line_back_off(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing.structure_panel.attributeEdited.emit("title", "second")
    editing._action_undo()
    assert editing.structure_panel.change_rows() == [
        "+ /entry_0000@title = first"]


def test_copy_as_text_reaches_the_clipboard(editing, monkeypatch):
    from PySide6.QtWidgets import QApplication

    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first")
    editing.structure_panel.copyChangesRequested.emit()
    assert "title" in QApplication.clipboard().text()


# -- the handle, and everything that must not change ----------------------


def test_the_handle_is_taken_once_and_reused(editing, monkeypatch):
    """Opening the tab takes it; editing never pays for it again.

    Acquiring detaches and rebuilds the file browser, so it is done on
    the tab switch — where a moment's work is expected — rather than in
    the middle of the user's first edit.
    """
    handle = editing._h5_edit_handle
    assert handle is not None and handle.is_open, (
        "the fixture switches to the Structure tab, which takes the handle")
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "x")
    assert editing._h5_edit_handle is handle, "no re-acquire on the first edit"
    editing.structure_panel.attributeEdited.emit("title", "y")
    assert editing._h5_edit_handle is handle, "no re-acquire between edits"


def test_detaching_the_tree_releases_the_handle(editing, monkeypatch):
    """The one hook that keeps every other write path working."""
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "x")
    assert editing._h5_edit_handle.is_open
    editing._detach_silx_tree()
    assert not editing._h5_edit_handle.is_open
    editing._reattach_silx_tree()
    # ...and the next edit simply takes it again.
    editing.structure_panel.attributeEdited.emit("title", "y")
    assert editing._h5_edit_handle.is_open
    assert _read(editing, "/entry_0000", "title") == "y"


def test_a_peak_add_still_works_with_the_editor_handle_open(editing, monkeypatch):
    """The coexistence guarantee: peak writes open r+ of their own."""
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "x")
    assert editing._h5_edit_handle.is_open

    new_id = file_model.add_detected_peak_row(
        editing.session.temp_path, "entry_0000", 0,
        score=0.9, radius=1.0, angle=45.0,
        radius_width=0.2, angle_width=10.0, is_ring=False,
    )
    assert new_id == 0
    table = file_model.load_peaks(
        editing.session.temp_path, "entry_0000", 0)["detected"]
    assert table is not None and len(table) == 1


def test_saving_with_an_open_handle_writes_a_consistent_file(
    editing, monkeypatch, synthetic_nexus
):
    """Save is a plain copy2 and never detaches, so every edit flushes."""
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "kept")
    editing._save(confirm=False)
    with h5py.File(synthetic_nexus, "r") as f:
        assert f["entry_0000"].attrs["title"] == "kept"
    assert not editing.session.dirty


def test_closing_the_session_releases_the_handle_and_the_history(
    editing, monkeypatch
):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "x")
    temp = str(editing._h5_edit_handle.path)
    editing._action_close_file()
    assert editing._h5_edit_handle is None
    assert temp not in editing._structure_histories


def test_a_raw_session_cannot_be_edited(main_window, synthetic_raw, monkeypatch):
    from mlgidgui.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    node = _FakeNode(synthetic_raw, "/")
    monkeypatch.setattr(main_window, "_safe_selected_h5_nodes", lambda: [node])
    main_window._on_tree_selection_changed()

    assert not main_window.structure_panel.editable
    assert main_window._structure_target is None
    # Even if an intent arrives anyway, nothing is written and no handle
    # is opened against the user's original detector file.
    main_window.structure_panel.attributeAdded.emit("title", "str", "nope")
    assert main_window._h5_edit_handle is None
    with h5py.File(synthetic_raw, "r") as f:
        assert "title" not in f.attrs


def test_editing_a_q_axis_reloads_the_viewer(editing, monkeypatch, confirm_yes):
    """A protected edit in the current entry must reach the image."""
    reloads = []
    monkeypatch.setattr(
        editing, "_load_entry_into_viewer",
        lambda entry, **kw: reloads.append(entry),
    )
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing.structure_panel.attributeAdded.emit("units", "str", "1/Angstrom")
    assert reloads == ["entry_0000"]


def test_an_unrelated_edit_leaves_the_viewer_alone(editing, monkeypatch):
    reloads = []
    monkeypatch.setattr(
        editing, "_load_entry_into_viewer",
        lambda entry, **kw: reloads.append(entry),
    )
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "x")
    assert reloads == []


def test_the_check_card_reruns_after_an_edit(editing, monkeypatch, confirm_yes):
    _show(editing, "/entry_0000/data", monkeypatch)
    assert not any("signal" in what
                   for _, what in editing.structure_panel.issue_rows())
    editing.structure_panel.attributeEdited.emit("signal", "not_a_dataset")
    assert any("signal" in what
               for _, what in editing.structure_panel.issue_rows())


def test_two_files_keep_separate_histories(editing, monkeypatch, tmp_path):
    """Switching files must not wipe either list."""
    second = tmp_path / "second.h5"
    with h5py.File(second, "w") as f:
        data = f.create_group("entry_0000/data")
        data.attrs["signal"] = "img_gid_q"
        # Non-zero data: the viewer's log-scale range goes to nan on an
        # all-zero frame and pyqtgraph raises out of setRange.
        data.create_dataset(
            "img_gid_q",
            data=np.linspace(0.1, 1.0, 16, dtype="f4").reshape(1, 4, 4),
        )
        data.create_dataset("q_xy", data=np.linspace(-1, 1, 4, dtype="f4"))
        data.create_dataset("q_z", data=np.linspace(0, 2, 4, dtype="f4"))
    other = NexusSession.open(second)
    editing._sessions.append(other)
    editing._confirm_discard_changes = lambda session=None: True

    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel.attributeAdded.emit("title", "str", "first file")
    first_temp = str(editing.session.temp_path)

    editing._set_active_session(other)
    node = _FakeNode(other.temp_path, "/entry_0000")
    monkeypatch.setattr(editing, "_safe_selected_h5_nodes", lambda: [node])
    editing._on_tree_selection_changed()
    editing.structure_panel.attributeAdded.emit("title", "str", "second file")

    assert editing._structure_histories[first_temp].entries() == [
        "+ /entry_0000@title = first file"]
    assert editing.structure_panel.change_rows() == [
        "+ /entry_0000@title = second file"]


# -- the panel follows the tab's tree, not the folded browser --------------


def test_a_re_render_follows_the_tab_not_the_browser(editing, monkeypatch):
    """The browser is folded away here and may point somewhere else.

    Before the tab had its own tree, a re-render after an edit fell back
    to the File browser's selection. With the browser hidden that is a
    stale answer, and the panel would jump to a node the user was not
    editing the moment they changed an attribute.
    """
    stale = _FakeNode(editing.session.temp_path, "/entry_0000")
    monkeypatch.setattr(editing, "_safe_selected_h5_nodes", lambda: [stale])

    editing._on_structure_tree_selected(
        str(editing.session.temp_path), "/entry_0000/data/q_xy")
    assert editing.structure_panel.current.path == "/entry_0000/data/q_xy"

    editing._rerender_structure()

    assert editing.structure_panel.current.path == "/entry_0000/data/q_xy"


def test_an_edit_from_the_tab_writes_to_the_node_on_show(editing, monkeypatch):
    stale = _FakeNode(editing.session.temp_path, "/entry_0000")
    monkeypatch.setattr(editing, "_safe_selected_h5_nodes", lambda: [stale])

    editing._on_structure_tree_selected(
        str(editing.session.temp_path), "/entry_0000/data/q_xy")
    editing.structure_panel.attributeAdded.emit("units", "str", "1/A")

    assert _read(editing, "/entry_0000/data/q_xy", "units") == "1/A"
    assert "units" not in dict(
        _read_attrs(editing, "/entry_0000"))


def _read_attrs(window, h5_path):
    handle = window._h5_edit_handle
    if handle is not None and handle.is_open:
        return dict(handle.file[h5_path].attrs)
    with h5py.File(window.session.temp_path, "r") as f:
        return dict(f[h5_path].attrs)
