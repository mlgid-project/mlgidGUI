"""Structure edits: creating, renaming, moving and deleting nodes.

Stage four of the editor. The dialogs are driven by stubbing ``exec``
and the result getters, so what is tested is the handler chain and not
Qt's own form widgets — the same way the rest of this suite drives the
app's modal flows.

The tree-state section is the one worth reading: a structural edit
rebuilds the File browser, and the user is working *in* that tree, so
losing their place mid-edit would be the feature's worst behaviour.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from mlgidlab import h5_edit
from mlgidlab.h5_edit_ops import CreateNodeOp, DeleteNodeOp, MoveNodeOp
from mlgidlab.session import NexusSession
from mlgidlab.structure_dialogs import parse_shape

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


@pytest.fixture
def confirm_yes(monkeypatch):
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
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    # See test_structure_edits: pytest-qt closes registered widgets from
    # its own teardown hook, before fixture finalizers, so the unsaved
    # changes prompt has to be stubbed on the instance.
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    # Populate the browser the way the real open path does. Tests that
    # only drive handlers do not need it, but the tree-state ones read
    # actual rows.
    main_window._reattach_silx_tree()
    return main_window


def _target(window, h5_path: str):
    from pathlib import Path
    return (Path(window.session.temp_path), h5_path)


def _show(window, h5_path: str, monkeypatch) -> None:
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


def _stub_group_dialog(monkeypatch, name, nx_class="", template=None):
    from mlgidlab import main_window_structure as mws

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_values(self):
            return (name, nx_class, template)

    monkeypatch.setattr(mws, "NewGroupDialog", _Stub)


def _stub_dataset_dialog(
    monkeypatch, name, dtype="float64", shape="", value="0",
    units="", resizable=False,
):
    from mlgidlab import main_window_structure as mws

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_values(self):
            return (name, dtype, shape, value, units, resizable)

    monkeypatch.setattr(mws, "NewDatasetDialog", _Stub)


def _read(window):
    return window._h5_edit_handle.file


# -- shape parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("", ()), ("3", (3,)), ("3, 4", (3, 4)), (" (2,5) ", (2, 5)), ("[7]", (7,))],
)
def test_parse_shape(text, expected):
    assert parse_shape(text) == expected


def test_parse_shape_explains_itself():
    with pytest.raises(ValueError, match="comma-separated"):
        parse_shape("3 by 4")
    with pytest.raises(ValueError, match="negative"):
        parse_shape("-1")


# -- creating --------------------------------------------------------------


def test_new_group_is_created_with_its_class(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    f = _read(editing)
    assert "notes" in f["entry_0000"]
    assert f["entry_0000/notes"].attrs["NX_class"] == "NXnote"


def test_a_template_creates_its_fields(editing, monkeypatch):
    """Picking Sample (NXsample) is meant to leave you editing values."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "sample2", "NXsample", "NXsample")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    group = _read(editing)["entry_0000/sample2"]
    assert group.attrs["NX_class"] == "NXsample"
    assert set(group.keys()) == {"name", "chemical_formula", "temperature"}
    assert group["temperature"].attrs["units"] == "K"


def test_new_field_with_a_shape_and_units(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_dataset_dialog(
        monkeypatch, "window", dtype="float32", shape="2, 3",
        value="1.5", units="1/Angstrom",
    )
    editing._on_structure_new_dataset(_target(editing, "/entry_0000"))
    dataset = _read(editing)["entry_0000/window"]
    assert dataset.shape == (2, 3)
    assert dataset[0, 0] == pytest.approx(1.5)
    assert dataset.attrs["units"] == "1/Angstrom"


def test_new_text_field(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_dataset_dialog(monkeypatch, "operator", dtype="str", value="Nico")
    editing._on_structure_new_dataset(_target(editing, "/entry_0000"))
    value = _read(editing)["entry_0000/operator"][()]
    assert h5_edit._as_text(value) == "Nico"


def test_a_growable_field_is_resizable(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_dataset_dialog(
        monkeypatch, "log", dtype="int32", shape="4", resizable=True)
    editing._on_structure_new_dataset(_target(editing, "/entry_0000"))
    assert _read(editing)["entry_0000/log"].maxshape == (None,)


def test_a_bad_shape_is_reported_and_creates_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_dataset_dialog(monkeypatch, "bad", shape="3 by 4")
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: seen.append(a[1:3])),
    )
    editing._on_structure_new_dataset(_target(editing, "/entry_0000"))
    assert seen, "the user must be told why nothing was created"
    assert "bad" not in _read(editing)["entry_0000"]


def test_a_duplicate_name_is_refused(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "data")
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: seen.append(a[1:3])),
    )
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    assert seen and "already exists" in str(seen[0])


def test_creating_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    assert "notes" in _read(editing)["entry_0000"]
    editing._action_undo()
    assert "notes" not in _read(editing)["entry_0000"]
    editing._action_redo()
    assert "notes" in _read(editing)["entry_0000"]


# -- renaming and moving ---------------------------------------------------


def test_rename_a_group(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))
    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))
    f = _read(editing)
    assert "remarks" in f["entry_0000"] and "notes" not in f["entry_0000"]


def test_a_cancelled_rename_changes_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))
    assert "notes" in _read(editing)["entry_0000"]


def test_rename_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))
    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))
    editing._action_undo()
    assert "notes" in _read(editing)["entry_0000"]


def test_renaming_a_protected_node_asks_first(editing, monkeypatch, confirm_no):
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("qxy", True)))
    editing._on_structure_rename(_target(editing, "/entry_0000/data/q_xy"))
    assert "q_xy" in _read(editing)["entry_0000/data"]


# The header's name is the third route to a rename and the only one
# without a dialog. It has to be the SAME rename: same write, same undo
# entry, same question about a protected node. What is pinned here is
# that it goes through the shared commit rather than growing a second
# one that could drift.


def test_the_header_name_renames_the_node(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)

    editing.structure_panel.renameRequested.emit("remarks")

    f = _read(editing)
    assert "remarks" in f["entry_0000"] and "notes" not in f["entry_0000"]


def test_a_header_rename_is_undoable_like_any_other(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)

    editing.structure_panel.renameRequested.emit("remarks")
    editing._action_undo()

    assert "notes" in _read(editing)["entry_0000"]


def test_a_header_rename_of_a_protected_node_asks_first(
    editing, monkeypatch, confirm_no
):
    """No dialog to type in is not a reason to skip the warning."""
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)

    editing.structure_panel.renameRequested.emit("qxy")

    assert "q_xy" in _read(editing)["entry_0000/data"]


def test_the_dialog_rename_updates_the_header_too(editing, monkeypatch):
    """The panel follows the node whichever route moved it.

    It did not before: every commit ends by re-reading the node the
    panel is showing, and a rename empties that path, so the header went
    on naming a node that was no longer there.
    """
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))

    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))

    assert editing.structure_panel._name_edit.text() == "remarks"
    assert editing.structure_panel.current.kind != "missing"


def test_undoing_a_rename_puts_the_header_back(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)
    editing.structure_panel.renameRequested.emit("remarks")

    editing._action_undo()

    assert editing.structure_panel._name_edit.text() == "notes"


def test_redoing_a_rename_takes_the_header_forward_again(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)
    editing.structure_panel.renameRequested.emit("remarks")
    editing._action_undo()

    editing._action_redo()

    assert editing.structure_panel._name_edit.text() == "remarks"


def test_renaming_something_else_leaves_the_panel_where_it_was(
    editing, monkeypatch
):
    """A context-menu rename can target a row the panel is not showing."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000", monkeypatch)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))

    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))

    assert editing.structure_panel._name_edit.text() == "entry_0000"


def test_the_header_shows_the_new_name_after_the_rename(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)

    editing.structure_panel.renameRequested.emit("remarks")

    assert editing.structure_panel._name_edit.text() == "remarks"


# Deleting a shift-picked selection. A multi-row selection that quietly
# deleted one row would be the worst kind of surprise in an editor, so
# Delete takes the whole thing — as one confirm and one Ctrl+Z.


def _pick(window, paths, monkeypatch):
    window.tabs.setCurrentWidget(window.structure_panel)
    picked = [(str(window.session.temp_path), path) for path in paths]
    monkeypatch.setattr(
        window.structure_panel.node_tree, "selected_targets",
        lambda: list(picked))


def test_deleting_a_selection_takes_all_of_it(editing, monkeypatch,
                                              confirm_yes):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "one")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _stub_group_dialog(monkeypatch, "two")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _pick(editing, ["/entry_0000/one", "/entry_0000/two"], monkeypatch)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))

    editing._on_structure_delete(None)

    entry = _read(editing)["entry_0000"]
    assert "one" not in entry and "two" not in entry


def test_a_multi_delete_is_one_undo(editing, monkeypatch, confirm_yes):
    _show(editing, "/entry_0000", monkeypatch)
    for name in ("one", "two"):
        _stub_group_dialog(monkeypatch, name)
        editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _pick(editing, ["/entry_0000/one", "/entry_0000/two"], monkeypatch)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    editing._on_structure_delete(None)

    editing._action_undo()

    entry = _read(editing)["entry_0000"]
    assert "one" in entry and "two" in entry


def test_declining_the_confirm_deletes_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    for name in ("one", "two"):
        _stub_group_dialog(monkeypatch, name)
        editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _pick(editing, ["/entry_0000/one", "/entry_0000/two"], monkeypatch)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))

    editing._on_structure_delete(None)

    entry = _read(editing)["entry_0000"]
    assert "one" in entry and "two" in entry


def test_deleting_one_row_still_asks_nothing_extra(editing, monkeypatch):
    """A single pick keeps the old path: no batch, no count dialog."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "one")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _pick(editing, ["/entry_0000/one"], monkeypatch)
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: asked.append(1) or
                     QMessageBox.StandardButton.Yes))

    editing._on_structure_delete(None)

    assert asked == []
    assert "one" not in _read(editing)["entry_0000"]


# -- deleting --------------------------------------------------------------


def test_delete_a_group_and_undo_it(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "sample2", "NXsample", "NXsample")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))

    editing._on_structure_delete(_target(editing, "/entry_0000/sample2"))
    assert "sample2" not in _read(editing)["entry_0000"]

    editing._action_undo()
    group = _read(editing)["entry_0000/sample2"]
    assert group.attrs["NX_class"] == "NXsample"
    assert "temperature" in group, "the whole subtree must come back"


def test_deleting_a_protected_node_asks_first(editing, monkeypatch, confirm_no):
    _show(editing, "/entry_0000/data/q_z", monkeypatch)
    editing._on_structure_delete(_target(editing, "/entry_0000/data/q_z"))
    assert "q_z" in _read(editing)["entry_0000/data"]


def test_a_confirmed_protected_delete_goes_through(
    editing, monkeypatch, confirm_yes
):
    _show(editing, "/entry_0000/data/q_z", monkeypatch)
    editing._on_structure_delete(_target(editing, "/entry_0000/data/q_z"))
    assert "q_z" not in _read(editing)["entry_0000/data"]


def test_a_delete_too_big_to_snapshot_warns_that_it_is_final(
    editing, monkeypatch
):
    """The cap is what makes 'undoable' an honest promise."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_dataset_dialog(monkeypatch, "bulk", dtype="float64", shape="16")
    editing._on_structure_new_dataset(_target(editing, "/entry_0000"))
    monkeypatch.setattr(h5_edit, "UNDO_SNAPSHOT_LIMIT", 8)

    asked = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(
            lambda *a, **k: asked.append(a[1:3]) or QMessageBox.StandardButton.Yes),
    )
    editing._on_structure_delete(_target(editing, "/entry_0000/bulk"))
    assert asked and "cannot be reversed" in str(asked[0])
    assert "bulk" not in _read(editing)["entry_0000"]
    assert editing.structure_panel.change_rows()[-1].endswith("(not undoable)")


def test_undoing_an_unsnapshotted_delete_says_why_it_cannot(editing):
    op = DeleteNodeOp("/entry_0000/bulk", "dataset", snapshot=None)
    assert not op.reversible
    with pytest.raises(h5_edit.EditError, match="too large"):
        op.undo(None)


# -- what a structural change makes stale ---------------------------------


def test_a_new_entry_reaches_the_entry_dropdown(editing, monkeypatch):
    """A top-level group that looks like an entry must become selectable."""
    before = editing.entry_combo.count()
    _show(editing, "/", monkeypatch)
    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            data = f.create_group("entry_0001/data")
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q",
                data=np.linspace(0.1, 1, 16, dtype="f4").reshape(1, 4, 4))
            data.create_dataset("q_xy", data=np.linspace(-1, 1, 4, dtype="f4"))
            data.create_dataset("q_z", data=np.linspace(0, 2, 4, dtype="f4"))
    _stub_group_dialog(monkeypatch, "scratch")
    editing._on_structure_new_group(_target(editing, "/"))
    assert editing.entry_combo.count() == before + 1
    assert editing.entry_combo.currentText() == "entry_0000", (
        "the entry the user was on must survive the repopulate")


def test_the_changes_list_records_structural_edits(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    assert editing.structure_panel.change_rows() == [
        "+ /entry_0000/notes  (NXnote)"]


# -- the browser tree ------------------------------------------------------


def test_the_tree_keeps_its_open_rows_across_an_edit(editing, monkeypatch):
    """The user is working in this tree; an edit must not collapse it."""
    view = editing.tree
    model = view.model()
    root = model.index(0, 0, view.rootIndex())
    view.expand(root)
    entry = model.index(0, 0, root)
    view.expand(entry)
    assert view.isExpanded(root) and view.isExpanded(entry)

    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))

    root = model.index(0, 0, view.rootIndex())
    assert view.isExpanded(root)
    entry = model.index(0, 0, root)
    assert view.isExpanded(entry)


def test_the_new_node_is_findable_in_the_rebuilt_tree(editing, monkeypatch):
    view = editing.tree
    model = view.model()
    root = model.index(0, 0, view.rootIndex())
    view.expand(root)

    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))

    # A row is keyed by its file plus the chain of display names, which
    # is what survives the model being torn down and rebuilt.
    key = (str(editing.session.temp_path), ("entry_0000", "notes"))
    index = editing._find_tree_index(key)
    assert index is not None and index.isValid()
    assert model.data(index) == "notes"


def test_finding_a_path_that_no_longer_exists_returns_none(editing):
    key = (str(editing.session.temp_path), ("entry_0000", "gone"))
    assert editing._find_tree_index(key) is None


def test_capturing_state_of_a_collapsed_tree_is_cheap_and_empty(editing):
    expanded, _ = editing._capture_tree_state()
    assert expanded == []


# -- the context menu ------------------------------------------------------


class _MenuEvent:
    def __init__(self, node, menu):
        self._node, self._menu = node, menu

    def hoveredObject(self):
        return self._node

    def menu(self):
        return self._menu


def _menu_labels(window, h5_path):
    from PySide6.QtWidgets import QMenu

    menu = QMenu()
    node = _FakeNode(window.session.temp_path, h5_path)
    window._structure_context_actions(_MenuEvent(node, menu))
    return [a.text() for a in menu.actions() if a.text()]


def test_the_structure_tab_menu_is_not_kept_after_it_closes(
    editing, monkeypatch,
):
    """Each right-click builds a fresh menu parented to the window; it
    has to go once it closes, or every click leaves one behind."""
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint
    from PySide6.QtWidgets import QMenu

    import mlgidlab.main_window_structure as mws

    shown = []

    class _ShownMenu(QMenu):
        def exec(self, *args):
            shown.append(self)

    monkeypatch.setattr(mws, "QMenu", _ShownMenu)
    before = len(editing.findChildren(QMenu))
    for _ in range(3):
        editing._on_structure_tree_context(
            str(editing.session.temp_path), "/entry_0000", QPoint(0, 0))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(shown) == 3
    assert len(editing.findChildren(QMenu)) == before


def test_a_closed_structure_dialog_is_not_kept(editing, monkeypatch):
    """The New group / field / link dialogs are parented to the window
    and used once; each use has to leave nothing behind."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QDialog

    import mlgidlab.main_window_structure as mws

    class _Cancelled(mws.NewGroupDialog):
        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(mws, "NewGroupDialog", _Cancelled)
    before = len(editing.findChildren(mws.NewGroupDialog))
    for _ in range(3):
        editing._on_structure_new_group(_target(editing, "/entry_0000"))
    assert len(editing.findChildren(mws.NewGroupDialog)) == before + 3
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(editing.findChildren(mws.NewGroupDialog)) == before


#: What only a group can offer, since only a group holds children.
CREATE_ACTIONS = {"New group…", "New field…", "New link…"}
#: What any node offers.
NODE_ACTIONS = {"Rename…", "Delete", "Copy", "Cut", "Paste"}


def test_a_group_offers_the_create_actions(editing):
    labels = set(_menu_labels(editing, "/entry_0000"))
    assert CREATE_ACTIONS <= labels
    assert NODE_ACTIONS <= labels


def test_a_dataset_cannot_hold_children(editing):
    labels = set(_menu_labels(editing, "/entry_0000/data/q_xy"))
    assert not (CREATE_ACTIONS & labels)
    assert NODE_ACTIONS <= labels


def test_the_file_root_cannot_be_renamed_or_deleted(editing):
    from PySide6.QtWidgets import QMenu

    menu = QMenu()
    node = _FakeNode(editing.session.temp_path, "/")
    editing._structure_context_actions(_MenuEvent(node, menu))
    states = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
    assert states["New group…"] and not states["Rename…"] and not states["Delete"]


def test_a_raw_file_gets_no_edit_menu(main_window, synthetic_raw, monkeypatch):
    from PySide6.QtWidgets import QMenu
    from mlgidlab.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    menu = QMenu()
    node = _FakeNode(synthetic_raw, "/")
    main_window._structure_context_actions(_MenuEvent(node, menu))
    assert [a.text() for a in menu.actions() if a.text()] == []


# -- undo after a structural edit ------------------------------------------
#
# Reported from a manual pass: Ctrl+Z did nothing after a rename or a
# delete, while it worked after a create. Both cases leave the panel
# showing a path that no longer exists, and the undo route used to hang
# off "what is on show" rather than "which file is being edited".


def test_undo_works_after_a_rename(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    # Select the group being renamed, which is what a user does before
    # right-clicking it. The panel then shows a path that stops existing
    # the moment the rename lands.
    _show(editing, "/entry_0000/notes", monkeypatch)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))
    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))

    assert editing._structure_owns_undo(), "the editor must still own Ctrl+Z"
    assert editing._structure_undo() is True
    f = _read(editing)
    assert "notes" in f["entry_0000"] and "remarks" not in f["entry_0000"]


def test_undo_works_after_a_delete(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/notes", monkeypatch)
    editing._on_structure_delete(_target(editing, "/entry_0000/notes"))
    assert "notes" not in _read(editing)["entry_0000"]

    assert editing._structure_owns_undo()
    assert editing._structure_undo() is True
    assert "notes" in _read(editing)["entry_0000"]


def test_undoing_a_rename_puts_the_old_name_back_in_the_tree(editing, monkeypatch):
    """The browser is the surface being edited; undo has to reach it."""
    view = editing.tree
    view.expand(view.model().index(0, 0, view.rootIndex()))
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("remarks", True)))
    editing._on_structure_rename(_target(editing, "/entry_0000/notes"))
    editing._structure_undo()

    key = (str(editing.session.temp_path), ("entry_0000", "notes"))
    index = editing._find_tree_index(key)
    assert index is not None and index.isValid()
    gone = (str(editing.session.temp_path), ("entry_0000", "remarks"))
    assert editing._find_tree_index(gone) is None


def test_undoing_a_create_takes_the_row_out_of_the_tree(editing, monkeypatch):
    view = editing.tree
    view.expand(view.model().index(0, 0, view.rootIndex()))
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    key = (str(editing.session.temp_path), ("entry_0000", "notes"))
    assert editing._find_tree_index(key) is not None

    editing._structure_undo()
    assert editing._find_tree_index(key) is None


def test_undoing_a_delete_puts_the_row_back_in_the_tree(editing, monkeypatch):
    view = editing.tree
    view.expand(view.model().index(0, 0, view.rootIndex()))
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))
    editing._on_structure_delete(_target(editing, "/entry_0000/notes"))
    editing._structure_undo()

    key = (str(editing.session.temp_path), ("entry_0000", "notes"))
    assert editing._find_tree_index(key) is not None


def test_the_editor_owns_undo_while_the_browser_has_focus(editing, monkeypatch):
    """Edits start from the dock's context menu, so the dock has to count.

    Otherwise creating a group from the browser and pressing Ctrl+Z hands
    the key to the image viewer, which has nothing to do with it.
    """
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))

    editing.tabs.setCurrentWidget(editing.viewer)
    monkeypatch.setattr(type(editing.tree), "hasFocus", lambda self: True)
    assert editing._structure_owns_undo()
    editing._action_undo()
    assert "notes" not in _read(editing)["entry_0000"]


def test_the_viewer_still_owns_undo_elsewhere(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_group_dialog(monkeypatch, "notes", "NXnote")
    editing._on_structure_new_group(_target(editing, "/entry_0000"))

    editing.tabs.setCurrentWidget(editing.viewer)
    monkeypatch.setattr(type(editing.tree), "hasFocus", lambda self: False)
    calls = []
    monkeypatch.setattr(editing.viewer, "undo_last_action", lambda: calls.append(1))
    editing._action_undo()
    assert calls == [1]
    assert "notes" in _read(editing)["entry_0000"]
