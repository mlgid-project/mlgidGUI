"""Copy and paste in the Structure tab, within a file and between files.

Stage six of the editor. The clipboard holds a reference, not a payload,
so copying a large stack costs nothing until it is pasted and copying an
external link keeps it a link.

The sharing of Ctrl+C / Ctrl+V with the peak clipboard is the part that
most needs pinning: those keys already mean "this detected peak" on the
Image tab, and that must not change.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QInputDialog, QMenu, QMessageBox

from mlgidlab import h5_edit
from mlgidlab.session import NexusSession
from mlgidlab.structure_clipboard import StructureClip, free_name

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


class _FakeNode:
    def __init__(self, filename, local_name: str) -> None:
        self.local_filename = str(filename)
        self.local_name = local_name


def _second_nexus(tmp_path, name="second.h5"):
    path = tmp_path / name
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=np.linspace(0.1, 1.0, 16, dtype="f4").reshape(1, 4, 4))
        data.create_dataset("q_xy", data=np.linspace(-1, 1, 4, dtype="f4"))
        data.create_dataset("q_z", data=np.linspace(0, 2, 4, dtype="f4"))
        meta = f.create_group("beamline", track_order=True)
        meta.attrs["NX_class"] = "NXinstrument"
        meta.attrs["operator"] = "Nico"
        meta.create_dataset("wavelength", data=np.float64(1.54))
    return path


@pytest.fixture
def editing(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window._reattach_silx_tree()
    return main_window


def _target(window, h5_path: str):
    from pathlib import Path
    return (Path(window.session.temp_path), h5_path)


def _show(window, h5_path: str, monkeypatch) -> None:
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


def _read(window):
    handle = window._h5_edit_handle
    if handle is not None and handle.is_open:
        return handle.file
    return h5py.File(window.session.temp_path, "r")


# -- free_name -------------------------------------------------------------


def test_free_name_leaves_an_unused_name_alone():
    assert free_name(["a", "b"], "c") == "c"


def test_free_name_walks_up_the_copies():
    assert free_name(["q_xy"], "q_xy") == "q_xy_copy"
    assert free_name(["q_xy", "q_xy_copy"], "q_xy") == "q_xy_copy2"
    assert free_name(
        ["q_xy", "q_xy_copy", "q_xy_copy2"], "q_xy") == "q_xy_copy3"


def test_a_clip_describes_itself(tmp_path):
    clip = StructureClip("node", tmp_path / "a.h5", "/entry_0000/data")
    assert clip.name == "data"
    assert "Copied /entry_0000/data from a.h5" == clip.describe()
    cut = StructureClip("node", tmp_path / "a.h5", "/x", mode="cut")
    assert cut.describe().startswith("Cut ")
    attr = StructureClip("attribute", tmp_path / "a.h5", "/x", attr="units")
    assert attr.name == "units"
    assert "attribute units" in attr.describe()


# -- copy and paste within one file ---------------------------------------


def test_copy_a_group_and_paste_it_elsewhere(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data"))
    editing._on_structure_paste(_target(editing, "/"))
    f = _read(editing)
    assert "data" in f
    assert np.array_equal(f["data/q_xy"][()], f["entry_0000/data/q_xy"][()])


def test_pasting_onto_a_dataset_lands_beside_it(editing, monkeypatch):
    """Dropping onto a leaf means 'here', as in any file manager."""
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data/q_xy"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("q_xy_copy", True)))
    editing._on_structure_paste(_target(editing, "/entry_0000/data/q_z"))
    assert "q_xy_copy" in _read(editing)["entry_0000/data"]


def test_a_name_collision_asks_with_a_suggestion(editing, monkeypatch):
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data/q_xy"))
    asked = []

    def _ask(parent, title, label, **kwargs):
        asked.append((label, kwargs.get("text")))
        return (kwargs.get("text", ""), True)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(_ask))
    editing._on_structure_paste(_target(editing, "/entry_0000/data"))
    assert asked and asked[0][1] == "q_xy_copy"
    assert "q_xy_copy" in _read(editing)["entry_0000/data"]


def test_backing_out_of_the_name_prompt_pastes_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data/q_xy"))
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    before = set(_read(editing)["entry_0000/data"].keys())
    editing._on_structure_paste(_target(editing, "/entry_0000/data"))
    assert set(_read(editing)["entry_0000/data"].keys()) == before


def test_a_paste_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data"))
    editing._on_structure_paste(_target(editing, "/"))
    assert "data" in _read(editing)
    editing._structure_undo()
    assert "data" not in _read(editing)
    editing._structure_redo()
    assert "data" in _read(editing)


def test_pasting_with_an_empty_clipboard_does_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    before = set(_read(editing).keys())
    editing._on_structure_paste(_target(editing, "/"))
    assert set(_read(editing).keys()) == before


# -- cut, which is a move --------------------------------------------------


def test_cut_and_paste_moves_within_the_file(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            f["entry_0000"].create_group("strays").create_dataset(
                "note", data=np.int32(1))

    editing._on_structure_copy(
        _target(editing, "/entry_0000/strays"), cut=True)
    editing._on_structure_paste(_target(editing, "/"))
    f = _read(editing)
    assert "strays" in f and "note" in f["strays"]
    assert "strays" not in f["entry_0000"]


def test_a_move_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    with editing._detached_silx_tree():
        with h5py.File(editing.session.temp_path, "r+") as f:
            f["entry_0000"].create_group("strays")
    editing._on_structure_copy(
        _target(editing, "/entry_0000/strays"), cut=True)
    editing._on_structure_paste(_target(editing, "/"))
    editing._structure_undo()
    f = _read(editing)
    assert "strays" in f["entry_0000"] and "strays" not in f


def test_a_group_cannot_be_cut_into_itself(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    editing._on_structure_copy(_target(editing, "/entry_0000"), cut=True)
    editing._on_structure_paste(_target(editing, "/entry_0000/data"))
    assert seen and "inside itself" in str(seen[0])


def test_cut_across_files_is_refused_with_a_way_forward(
    editing, monkeypatch, tmp_path
):
    """Undoing a cross-file move would need both files writable at once."""
    other = _second_nexus(tmp_path)
    editing._structure_clip = [StructureClip(
        "node", other, "/beamline", mode="cut")]
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste(_target(editing, "/"))
    assert seen and "copy it and delete the original" in str(seen[0])


# -- across files ----------------------------------------------------------


def test_paste_a_group_from_another_file(editing, monkeypatch, tmp_path):
    other = _second_nexus(tmp_path)
    editing._structure_clip = [StructureClip("node", other, "/beamline")]
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste(_target(editing, "/"))
    f = _read(editing)
    assert f["beamline"].attrs["NX_class"] == "NXinstrument"
    assert f["beamline"].attrs["operator"] == "Nico"
    assert float(f["beamline/wavelength"][()]) == pytest.approx(1.54)


def test_a_cross_file_paste_is_undoable_and_redoable(
    editing, monkeypatch, tmp_path
):
    """Redo has to reopen the source file, which is still on disk."""
    other = _second_nexus(tmp_path)
    editing._structure_clip = [StructureClip("node", other, "/beamline")]
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste(_target(editing, "/"))
    editing._structure_undo()
    assert "beamline" not in _read(editing)
    editing._structure_redo()
    assert "beamline" in _read(editing)


def test_a_paste_from_a_file_that_vanished_is_reported(
    editing, monkeypatch, tmp_path
):
    other = _second_nexus(tmp_path)
    editing._structure_clip = [StructureClip("node", other, "/beamline")]
    other.unlink()
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste(_target(editing, "/"))
    assert seen, "a missing source must be reported, not swallowed"
    assert "beamline" not in _read(editing)


def test_copying_an_external_link_keeps_it_a_link(editing, monkeypatch, tmp_path):
    """The rule that stops a paste dragging a multi-GB scan along."""
    source = tmp_path / "master.h5"
    with h5py.File(source, "w") as f:
        f["scan"] = h5py.ExternalLink("/nowhere/absent.h5", "/entry_0000")
    editing._structure_clip = [StructureClip("node", source, "/scan")]
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste(_target(editing, "/"))
    link = _read(editing).get("scan", getlink=True)
    assert isinstance(link, h5py.ExternalLink)


# -- attributes ------------------------------------------------------------


def test_copy_and_paste_an_attribute(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel._attr_table.selectRow(0)
    editing._on_structure_copy()
    clip = editing._structure_clip[0]
    assert clip.kind == "attribute"

    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_paste(_target(editing, "/entry_0000"))
    f = _read(editing)
    assert f["entry_0000"].attrs[clip.attr] == f["entry_0000/data"].attrs[clip.attr]


def test_an_attribute_paste_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel._attr_table.selectRow(0)
    editing._on_structure_copy()
    name = editing._structure_clip[0].attr

    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_paste(_target(editing, "/entry_0000"))
    assert name in _read(editing)["entry_0000"].attrs
    editing._structure_undo()
    assert name not in _read(editing)["entry_0000"].attrs


def test_an_attribute_pastes_onto_a_dataset_not_its_parent(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel._attr_table.selectRow(0)
    editing._on_structure_copy()
    name = editing._structure_clip[0].attr

    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    editing._on_structure_paste(_target(editing, "/entry_0000/data/q_xy"))
    assert name in _read(editing)["entry_0000/data/q_xy"].attrs


# -- the shared shortcuts --------------------------------------------------


def test_the_editor_takes_ctrl_c_on_its_own_tab(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel._attr_table.clearSelection()
    editing._on_copy_peaks()
    assert editing._structure_clip
    assert editing._structure_clip[0].path == "/entry_0000/data"


def test_the_editor_takes_ctrl_c_while_the_browser_has_focus(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing.structure_panel._attr_table.clearSelection()
    editing.tabs.setCurrentWidget(editing.viewer)
    monkeypatch.setattr(type(editing.tree), "hasFocus", lambda self: True)
    editing._on_copy_peaks()
    assert editing._structure_clip is not None


def test_the_peak_clipboard_keeps_ctrl_c_elsewhere(editing, monkeypatch):
    """The Image tab's Ctrl+C must go on meaning 'this detected peak'."""
    editing.tabs.setCurrentWidget(editing.viewer)
    monkeypatch.setattr(type(editing.tree), "hasFocus", lambda self: False)
    calls = []
    monkeypatch.setattr(
        editing.viewer, "selected_peaks", lambda: calls.append(1) or [])
    editing._on_copy_peaks()
    assert calls == [1], "the peak path must still run"
    assert not editing._structure_clip


def test_the_peak_clipboard_keeps_ctrl_v_elsewhere(editing, monkeypatch):
    editing.tabs.setCurrentWidget(editing.viewer)
    monkeypatch.setattr(type(editing.tree), "hasFocus", lambda self: False)
    calls = []
    monkeypatch.setattr(editing, "_is_busy", lambda: calls.append(1) or True)
    editing._on_paste_peaks()
    assert calls == [1]


# -- the context menu ------------------------------------------------------


class _MenuEvent:
    def __init__(self, node, menu):
        self._node, self._menu = node, menu

    def hoveredObject(self):
        return self._node

    def menu(self):
        return self._menu


def _menu(window, h5_path) -> QMenu:
    """The built menu itself.

    Returned rather than its action map: the actions are children of the
    menu, so letting it fall out of scope destroys them and reading one
    afterwards raises "Internal C++ object already deleted". Callers keep
    the menu alive for as long as they inspect it.
    """
    menu = QMenu()
    node = _FakeNode(window.session.temp_path, h5_path)
    window._structure_context_actions(_MenuEvent(node, menu))
    return menu


def _actions(menu: QMenu) -> dict:
    return {a.text(): a for a in menu.actions() if a.text()}


def test_the_menu_offers_the_clipboard_actions(editing):
    menu = _menu(editing, "/entry_0000")
    assert {"Copy", "Cut", "Paste", "Paste from file…"} <= set(_actions(menu))


def test_paste_is_disabled_with_an_empty_clipboard(editing):
    menu = _menu(editing, "/entry_0000")
    assert not _actions(menu)["Paste"].isEnabled()


def test_paste_is_enabled_once_something_is_copied(editing, monkeypatch):
    _show(editing, "/entry_0000/data", monkeypatch)
    editing._on_structure_copy(_target(editing, "/entry_0000/data"))
    menu = _menu(editing, "/entry_0000")
    assert _actions(menu)["Paste"].isEnabled()


def test_the_file_root_cannot_be_copied(editing):
    menu = _menu(editing, "/")
    actions = _actions(menu)
    assert not actions["Copy"].isEnabled() and not actions["Cut"].isEnabled()
    # ...but it is a perfectly good place to paste into.
    assert actions["Paste from file…"].isEnabled()


# -- paste from file -------------------------------------------------------


def test_paste_from_file_picks_a_node_and_copies_it(
    editing, monkeypatch, tmp_path
):
    other = _second_nexus(tmp_path)
    from mlgidlab import main_window_structure as mws

    monkeypatch.setattr(
        mws.file_dialogs, "open_file", lambda *a, **k: str(other),
    )

    class _Picker:
        def __init__(self, *a, **k):
            pass

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_path(self):
            return "/beamline"

    monkeypatch.setattr(mws, "PickNodeDialog", _Picker)
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste_from_file(_target(editing, "/"))
    assert _read(editing)["beamline"].attrs["operator"] == "Nico"


def test_paste_from_file_cancelled_at_the_picker_does_nothing(
    editing, monkeypatch, tmp_path
):
    other = _second_nexus(tmp_path)
    from mlgidlab import main_window_structure as mws

    monkeypatch.setattr(
        mws.file_dialogs, "open_file", lambda *a, **k: str(other),
    )

    class _Picker:
        def __init__(self, *a, **k):
            pass

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def selected_path(self):  # pragma: no cover - never reached
            return "/beamline"

    monkeypatch.setattr(mws, "PickNodeDialog", _Picker)
    _show(editing, "/", monkeypatch)
    editing._on_structure_paste_from_file(_target(editing, "/"))
    assert "beamline" not in _read(editing)


# -- across files, entirely from the tab -----------------------------------
#
# The workflow the tab's own tree exists for: the File browser is folded
# away while the Structure tab is up, so the second file has to be
# reachable, selectable and copyable from inside the tab itself.


def test_a_second_open_file_is_a_second_root(editing, tmp_path):
    from mlgidlab.session import NexusSession

    other = NexusSession.open(_second_nexus(tmp_path))
    editing._sessions.append(other)
    editing._refresh_structure_tree_roots()

    assert editing.structure_panel.node_tree.root_labels() == [
        "synthetic.h5", "second.h5"]


def test_copy_from_one_root_and_paste_into_another(editing, tmp_path,
                                                    monkeypatch):
    from mlgidlab.session import NexusSession

    first = editing.session
    other = NexusSession.open(_second_nexus(tmp_path))
    editing._sessions.append(other)
    editing._refresh_structure_tree_roots()
    tree = editing.structure_panel.node_tree

    # Click into the second file and copy a group from it.
    assert tree.select_path(str(other.temp_path), "/beamline")
    editing._on_structure_action("copy")
    assert editing._structure_clip is not None

    # Back into the first — held from before the click, because clicking
    # into the second file made *it* the active session, which is the
    # point of that promotion — and paste at its root.
    assert tree.select_path(str(first.temp_path), "/")
    assert editing.session is first
    editing._on_structure_action("paste")

    with h5py.File(first.temp_path, "r") as f:
        assert f["beamline"].attrs["operator"] == "Nico"
        assert float(f["beamline/wavelength"][()]) == pytest.approx(1.54)


def test_selecting_the_second_file_promotes_its_session(editing, tmp_path):
    """An edit marks the *active* session dirty, so it has to follow."""
    from mlgidlab.session import NexusSession

    other = NexusSession.open(_second_nexus(tmp_path))
    editing._sessions.append(other)
    editing._refresh_structure_tree_roots()

    editing.structure_panel.node_tree.select_path(
        str(other.temp_path), "/beamline")

    assert editing.session is other


# -- copying several nodes at once -----------------------------------------
#
# The tab's tree is multi-select, so one gesture can put a whole range on
# the clipboard. What is pinned here is that the batch lands as ONE
# change: six pastes needing six undos would be a worse account of one
# gesture than the single line it replaces.


def _select_many(window, paths, monkeypatch):
    """Make the tab's tree report ``paths`` as its selection.

    Stubbed rather than clicked: building a real multi-row selection
    means populating branches, and populating a branch re-lists it and
    discards the very rows just picked. That is scaffolding trouble, not
    product behaviour — what a real click does is pinned by
    ``test_adding_a_row_to_the_selection_survives_the_render`` — and
    these tests are about the clipboard, not about Qt's item lifetimes.
    """
    window.tabs.setCurrentWidget(window.structure_panel)
    picked = [(str(window.session.temp_path), path) for path in paths]
    monkeypatch.setattr(
        window.structure_panel.node_tree, "selected_targets",
        lambda: list(picked))
    return picked


def test_adding_a_row_to_the_selection_survives_the_render(editing,
                                                           monkeypatch):
    """Shift-picking a second row must not collapse the first.

    Selecting emits, the panel renders the new row, and the render
    re-syncs the tree to what it is showing. That sync used to call
    ``setCurrentItem``, which under ExtendedSelection clears everything
    else — so the pick came apart as fast as it was made.
    """
    editing.tabs.setCurrentWidget(editing.structure_panel)
    editing._refresh_structure_tree_roots()
    tree = editing.structure_panel.node_tree
    tree.clearSelection()
    first = tree.find_item(str(editing.session.temp_path),
                           "/entry_0000/data/q_xy", populate=True)
    second = tree.find_item(str(editing.session.temp_path),
                            "/entry_0000/data/q_z", populate=True)

    first.setSelected(True)
    second.setSelected(True)

    assert {t[1] for t in tree.selected_targets()} == {
        "/entry_0000/data/q_xy", "/entry_0000/data/q_z"}


def test_copying_a_selection_puts_every_node_on_the_clipboard(editing,
                                                              monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data/q_xy", "/entry_0000/data/q_z"], monkeypatch)

    editing._on_structure_copy()

    assert [c.path for c in editing._structure_clip] == [
        "/entry_0000/data/q_xy", "/entry_0000/data/q_z"]


def test_pasting_a_selection_lands_every_node(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data/q_xy", "/entry_0000/data/q_z"], monkeypatch)
    editing._on_structure_copy()

    editing._on_structure_paste(_target(editing, "/entry_0000"))

    entry = _read(editing)["entry_0000"]
    assert "q_xy" in entry and "q_z" in entry


def test_a_multi_paste_is_one_undo(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data/q_xy", "/entry_0000/data/q_z"], monkeypatch)
    editing._on_structure_copy()
    editing._on_structure_paste(_target(editing, "/entry_0000"))
    assert "q_xy" in _read(editing)["entry_0000"]

    editing._structure_undo()

    entry = _read(editing)["entry_0000"]
    assert "q_xy" not in entry and "q_z" not in entry


def test_a_multi_paste_is_one_line_in_the_changes_list(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data/q_xy", "/entry_0000/data/q_z"], monkeypatch)
    editing._on_structure_copy()
    before = len(editing.structure_panel.change_rows())

    editing._on_structure_paste(_target(editing, "/entry_0000"))

    rows = editing.structure_panel.change_rows()
    assert len(rows) == before + 1
    assert "2 nodes pasted" in rows[-1]


def test_redo_puts_a_whole_batch_back(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data/q_xy", "/entry_0000/data/q_z"], monkeypatch)
    editing._on_structure_copy()
    editing._on_structure_paste(_target(editing, "/entry_0000"))
    editing._structure_undo()

    editing._structure_redo()

    entry = _read(editing)["entry_0000"]
    assert "q_xy" in entry and "q_z" in entry


def test_a_group_and_its_child_are_not_copied_twice(editing, monkeypatch):
    """A shift-picked range drags in whatever lies between the clicks."""
    _show(editing, "/entry_0000", monkeypatch)
    _select_many(editing, ["/entry_0000/data", "/entry_0000/data/q_xy"], monkeypatch)

    editing._on_structure_copy()

    assert [c.path for c in editing._structure_clip] == ["/entry_0000/data"]


# -- pruning and batching, as plain functions ------------------------------


def test_prune_drops_a_child_of_a_selected_group():
    from mlgidlab.structure_clipboard import prune_descendants
    picked = [("a.h5", "/entry"), ("a.h5", "/entry/data"),
              ("a.h5", "/other")]
    assert prune_descendants(picked) == [("a.h5", "/entry"), ("a.h5", "/other")]


def test_prune_keeps_the_same_path_in_another_file():
    from mlgidlab.structure_clipboard import prune_descendants
    picked = [("a.h5", "/entry"), ("b.h5", "/entry/data")]
    assert prune_descendants(picked) == picked


def test_prune_keeps_siblings_and_order():
    from mlgidlab.structure_clipboard import prune_descendants
    picked = [("a.h5", "/e/z"), ("a.h5", "/e/a"), ("a.h5", "/e/m")]
    assert prune_descendants(picked) == picked


def test_a_batch_undoes_in_reverse():
    """A batch that made a then b must remove b first."""
    from mlgidlab.h5_edit_ops import BatchOp

    order = []

    class _Op:
        def __init__(self, tag): self.tag = tag; self.path = f"/{tag}"
        def redo(self, f): order.append(("redo", self.tag))
        def undo(self, f): order.append(("undo", self.tag))
        def describe(self): return self.tag

    batch = BatchOp([_Op("a"), _Op("b")], label="two things")
    batch.undo(None)
    batch.redo(None)

    assert order == [("undo", "b"), ("undo", "a"),
                     ("redo", "a"), ("redo", "b")]
    assert batch.describe() == "two things"
    assert batch.paths == ["/a", "/b"]

