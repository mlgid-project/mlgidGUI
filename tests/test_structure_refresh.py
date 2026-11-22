"""The Structure tab's Refresh button.

A detection, fitting or matching run, or a conversion, writes into the
file without telling the Structure tab, so a group it created (the
``mlgidDETECT`` metadata, say) was missing from the tree until the file
was reopened. ``StructureTree.refresh_all`` re-lists every open group,
keeping what is open and selected, and the window re-reads the node on
show. Source: structure_tree.py ``refresh_all``; structure_panel.py
``_refresh_btn`` / ``refreshRequested``; main_window_structure.py
``_on_structure_refresh``.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import h5_edit
from mlgidgui.session import NexusSession
from mlgidgui.structure_tree import RootSpec, StructureTree

pytestmark = pytest.mark.gui


@pytest.fixture
def tree(qtbot):
    widget = StructureTree()
    qtbot.addWidget(widget)
    calls: list[tuple[str, str]] = []

    def lister(file_path, h5_path):
        calls.append((str(file_path), str(h5_path)))
        with h5py.File(file_path, "r") as f:
            return h5_edit.list_children(f, h5_path)

    widget.set_lister(lister)
    widget.listed = calls
    return widget


def _children(item) -> list[str]:
    return [item.child(i).text(0) for i in range(item.childCount())]


# -- the tree --------------------------------------------------------------


def test_refresh_all_finds_groups_added_behind_its_back(tree, synthetic_nexus):
    path = str(synthetic_nexus)
    tree.set_roots([RootSpec(path, "synthetic.h5")])
    tree.select_path(path, "/entry_0000/data/q_xy", quiet=True)
    with h5py.File(path, "r+") as f:
        f["/entry_0000"].create_group("mlgidDETECT")
        f["/entry_0000/data"].create_dataset("extra", data=1.0)
        f.create_group("late_entry")

    tree.refresh_all()

    assert "late_entry" in _children(tree.find_item(path, "/"))
    assert "mlgidDETECT" in _children(tree.find_item(path, "/entry_0000"))
    assert "extra" in _children(tree.find_item(path, "/entry_0000/data"))
    # What was open and selected is still open and selected.
    assert tree.find_item(path, "/entry_0000/data").isExpanded()
    assert tree.current_target()[1] == "/entry_0000/data/q_xy"


def test_a_root_opened_and_nothing_deeper_is_re_listed(tree, synthetic_nexus):
    """An already-open root keeps its open state through the reset, and
    opening it again emits nothing: it must be collapsed first or it
    would stay on its placeholder."""
    path = str(synthetic_nexus)
    tree.set_roots([RootSpec(path, "synthetic.h5")])
    tree.find_item(path, "/").setExpanded(True)
    assert "late_entry" not in _children(tree.find_item(path, "/"))
    with h5py.File(path, "r+") as f:
        f.create_group("late_entry")

    tree.refresh_all()

    root = tree.find_item(path, "/")
    assert root.isExpanded()
    assert "late_entry" in _children(root)


def test_an_unopened_group_stays_unread(tree, synthetic_nexus):
    path = str(synthetic_nexus)
    tree.set_roots([RootSpec(path, "synthetic.h5")])
    tree.find_item(path, "/").setExpanded(True)
    tree.listed.clear()

    tree.refresh_all()

    # Only the open root was listed again; nothing below it.
    assert tree.listed == [(path, "/")]


def test_refresh_all_is_silent(tree, synthetic_nexus, qtbot):
    """Putting the selection back must not read as a user click."""
    path = str(synthetic_nexus)
    tree.set_roots([RootSpec(path, "synthetic.h5")])
    tree.select_path(path, "/entry_0000/data/q_z", quiet=True)
    seen: list = []
    tree.nodeSelected.connect(lambda f, p: seen.append((f, p)))

    tree.refresh_all()

    assert seen == []


def test_a_selected_node_that_was_deleted_does_not_break_it(
    tree, synthetic_nexus,
):
    path = str(synthetic_nexus)
    tree.set_roots([RootSpec(path, "synthetic.h5")])
    tree.select_path(path, "/entry_0000/data/q_xy", quiet=True)
    with h5py.File(path, "r+") as f:
        del f["/entry_0000/data/q_xy"]

    tree.refresh_all()

    assert "q_xy" not in _children(tree.find_item(path, "/entry_0000/data"))
    assert tree.find_item(path, "/entry_0000/data").isExpanded()


def test_refresh_all_with_no_files_is_a_no_op(tree):
    tree.refresh_all()
    assert tree.topLevelItemCount() == 0


# -- the window ------------------------------------------------------------


@pytest.fixture
def opened(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    yield main_window
    if main_window._h5_edit_handle is not None:
        main_window._h5_edit_handle.release()


def _attr_names(window) -> list[str]:
    table = window.structure_panel._attr_table
    return [table.item(r, 0).text() for r in range(table.rowCount())]


def test_the_button_exists_and_emits(opened, qtbot):
    panel = opened.structure_panel
    assert panel._refresh_btn.text() == "Refresh"
    with qtbot.waitSignal(panel.refreshRequested, timeout=1000):
        panel._refresh_btn.click()


def test_refresh_shows_what_a_run_wrote_while_the_tab_was_open(opened):
    """The reported case: a group appears in the file after the tree was
    listed. Written through the edit handle, which is how the file is
    held while the tab is up."""
    window = opened
    panel = window.structure_panel
    path = str(window.session.temp_path)
    window._on_structure_tree_selected(path, "/entry_0000")
    tree = panel.node_tree
    # The user has the entry open in the tree, as in the report.
    tree.find_item(path, "/entry_0000", populate=True).setExpanded(True)
    entry_rows = _children(tree.find_item(path, "/entry_0000"))
    assert "data" in entry_rows                      # really listed
    assert "mlgidDETECT" not in entry_rows

    handle = window._h5_edit_handle
    assert handle is not None and handle.is_open
    handle.file["/entry_0000"].create_group("mlgidDETECT")
    handle.file["/entry_0000"].attrs["written_by_run"] = "yes"
    handle.file.flush()

    panel._refresh_btn.click()

    assert "mlgidDETECT" in _children(tree.find_item(path, "/entry_0000"))
    # The node on show was re-read too: its attributes are current.
    assert "written_by_run" in _attr_names(window)
    assert tree.current_target() == (path, "/entry_0000")


def test_refresh_rechecks_the_layout(opened):
    window = opened
    path = str(window.session.temp_path)
    window._on_structure_tree_selected(path, "/entry_0000")
    window._structure_checked_file = path        # "already checked"
    window.structure_panel._refresh_btn.click()
    # The cache was dropped and the render re-ran the check.
    assert window._structure_checked_file == path


def test_refresh_is_refused_while_a_run_owns_the_file(opened):
    """A read during a run fails, and a tree re-listed from a failed
    read would come up empty."""
    window = opened
    panel = window.structure_panel
    path = str(window.session.temp_path)
    window._on_structure_tree_selected(path, "/entry_0000")
    before = _children(panel.node_tree.find_item(path, "/entry_0000"))
    assert before

    window._pipe_thread = object()
    try:
        panel._refresh_btn.click()
    finally:
        window._pipe_thread = None

    assert _children(panel.node_tree.find_item(path, "/entry_0000")) == before
    assert "run is writing" in window.statusBar().currentMessage()


def test_refresh_follows_a_file_opened_elsewhere(opened, tmp_path):
    """The roots are refreshed too, so a file the tab has not listed yet
    shows up."""
    window = opened
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("entry_0000").attrs["NX_class"] = "NXentry"
    other_session = NexusSession.open(other)
    window._sessions.append(other_session)
    window.structure_panel._refresh_btn.click()
    labels = window.structure_panel.node_tree.root_labels()
    assert len(labels) == 2
