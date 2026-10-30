"""The Structure tab's own tree: lazy, link-blind, and holding nothing.

Three properties are pinned here, in order of how much they would cost
to lose:

1. **The tree opens no file handle.** HDF5 refuses ``r+`` while any read
   handle is open, so a tree that kept one would break editing outright —
   silently, and only on the second edit. The guard is a fully expanded
   tree followed by an acquire that must still succeed.
2. **A link is listed, not followed.** A master whose entries are
   external links to multi-GB scans must list as fast as any other file.
   Asserted against links whose target does not exist at all: resolving
   one would raise, so a clean listing is proof nothing was opened.
3. **A group is re-listed in place.** That is what replaces the File
   browser's teardown-and-rebuild after a structural edit, and it has to
   keep the rows around it open.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import h5_edit
from mlgidgui.session import NexusSession
from mlgidgui.structure_tree import KIND_ROLE, PATH_ROLE, RootSpec, StructureTree

pytestmark = pytest.mark.gui


@pytest.fixture
def linked_master(tmp_path):
    """A file whose children are links, one of them into a missing file.

    The missing target is the point: any code that resolved it would
    raise, so a listing that survives is proof the link was only read.
    """
    path = tmp_path / "master.h5"
    with h5py.File(path, "w", track_order=True) as f:
        real = f.create_group("real")
        real.attrs["NX_class"] = "NXentry"
        real.create_dataset("value", data=np.arange(4.0))
        f["soft"] = h5py.SoftLink("/real")
        f["soft_broken"] = h5py.SoftLink("/gone")
        f["ext"] = h5py.ExternalLink(str(tmp_path / "absent.h5"), "/entry")
    return path


@pytest.fixture
def tree(qtbot, tmp_path):
    """A standalone tree with a lister that opens ``r`` per call."""
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


def _kind(tree, file_path, h5_path) -> str:
    item = tree.find_item(str(file_path), h5_path)
    return str(item.data(0, KIND_ROLE))


# -- listing ---------------------------------------------------------------


def test_roots_start_collapsed_and_unread(tree, linked_master):
    """Adding a file must not walk it — that is the whole cost model."""
    tree.set_roots([RootSpec(str(linked_master), "master.h5")])
    assert tree.root_labels() == ["master.h5"]
    assert tree.listed == []


def test_expanding_a_root_lists_it_once(tree, linked_master):
    tree.set_roots([RootSpec(str(linked_master), "master.h5")])
    root = tree.topLevelItem(0)
    root.setExpanded(True)
    # Creation order, not alphabetical: the fixture writes the file with
    # ``track_order=True`` and sorting is off, so what the tree shows is
    # what the file says — acquisition order on a real scan.
    assert _children(root) == ["real", "soft", "soft_broken", "ext"]
    assert tree.listed == [(str(linked_master), "/")]


def test_links_are_classified_without_being_followed(tree, linked_master):
    """The external target does not exist; listing it must still work."""
    tree.set_roots([RootSpec(str(linked_master), "master.h5")])
    tree.topLevelItem(0).setExpanded(True)
    assert _kind(tree, linked_master, "/ext") == "external"
    assert _kind(tree, linked_master, "/soft") == "soft"
    assert _kind(tree, linked_master, "/soft_broken") == "broken"
    assert _kind(tree, linked_master, "/real") == "group"
    # Only the root was ever listed: no link was descended into.
    assert tree.listed == [(str(linked_master), "/")]


def test_an_external_link_is_expandable_but_unread(tree, linked_master):
    """It gets a chevron; opening that file stays the user's decision."""
    tree.set_roots([RootSpec(str(linked_master), "master.h5")])
    tree.topLevelItem(0).setExpanded(True)
    ext = tree.find_item(str(linked_master), "/ext")
    assert ext.childCount() == 1  # the placeholder
    assert tree.listed == [(str(linked_master), "/")]


def test_two_files_show_as_two_roots(tree, linked_master, synthetic_nexus):
    tree.set_roots([
        RootSpec(str(linked_master), "master.h5"),
        RootSpec(str(synthetic_nexus), "synthetic.h5"),
    ])
    assert tree.root_labels() == ["master.h5", "synthetic.h5"]


# -- reveal and refresh ----------------------------------------------------


def test_select_path_expands_only_its_own_branch(tree, synthetic_nexus):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    assert tree.select_path(str(synthetic_nexus), "/entry_0000/data/q_xy")
    assert tree.current_target()[1] == "/entry_0000/data/q_xy"
    # Root, entry, data — and nothing else.
    assert tree.listed == [
        (str(synthetic_nexus), "/"),
        (str(synthetic_nexus), "/entry_0000"),
        (str(synthetic_nexus), "/entry_0000/data"),
    ]


def test_select_path_returns_false_for_a_path_that_is_not_there(
    tree, synthetic_nexus
):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    assert not tree.select_path(str(synthetic_nexus), "/entry_0000/nope")


def test_selecting_a_row_emits_the_target(tree, synthetic_nexus, qtbot):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    seen = []
    tree.nodeSelected.connect(lambda f, p: seen.append((f, p)))
    tree.select_path(str(synthetic_nexus), "/entry_0000")
    assert seen == [(str(synthetic_nexus), "/entry_0000")]


def test_a_quiet_select_does_not_emit(tree, synthetic_nexus):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    seen = []
    tree.nodeSelected.connect(lambda f, p: seen.append((f, p)))
    tree.select_path(str(synthetic_nexus), "/entry_0000", quiet=True)
    assert seen == []


def test_refresh_picks_up_a_new_child_and_keeps_the_selection(
    tree, synthetic_nexus
):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    tree.select_path(str(synthetic_nexus), "/entry_0000/data/q_xy")
    with h5py.File(synthetic_nexus, "r+") as f:
        f["/entry_0000/data"].create_dataset("extra", data=1.0)

    tree.refresh_path(str(synthetic_nexus), "/entry_0000/data")

    data = tree.find_item(str(synthetic_nexus), "/entry_0000/data")
    assert "extra" in _children(data)
    assert tree.current_target()[1] == "/entry_0000/data/q_xy"


def test_refresh_drops_a_deleted_child(tree, synthetic_nexus):
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    tree.select_path(str(synthetic_nexus), "/entry_0000/data/q_xy")
    with h5py.File(synthetic_nexus, "r+") as f:
        del f["/entry_0000/data/q_xy"]

    tree.refresh_path(str(synthetic_nexus), "/entry_0000/data")

    data = tree.find_item(str(synthetic_nexus), "/entry_0000/data")
    assert "q_xy" not in _children(data)


def test_refresh_leaves_an_unopened_group_alone(tree, synthetic_nexus):
    """Re-listing what nobody has expanded would be work for nobody."""
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    tree.refresh_path(str(synthetic_nexus), "/entry_0000/data")
    assert tree.listed == []


def test_state_survives_a_root_rebuild(tree, synthetic_nexus, linked_master):
    """Opening a second file must not collapse the first."""
    tree.set_roots([RootSpec(str(synthetic_nexus), "synthetic.h5")])
    tree.select_path(str(synthetic_nexus), "/entry_0000/data/q_z")

    tree.set_roots([
        RootSpec(str(synthetic_nexus), "synthetic.h5"),
        RootSpec(str(linked_master), "master.h5"),
    ])

    assert tree.current_target()[1] == "/entry_0000/data/q_z"
    assert tree.find_item(str(synthetic_nexus), "/entry_0000/data").isExpanded()


# -- the handle guard ------------------------------------------------------


def test_a_fully_expanded_tree_still_lets_the_editor_take_r_plus(
    main_window, synthetic_nexus, qtbot
):
    """The property the whole design rests on.

    If any row held a read handle, this acquire would fail with
    "file is already open for read-only" — the exact failure a second
    silx tree would have introduced.
    """
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True

    tree = main_window.structure_panel.node_tree
    main_window._refresh_structure_tree_roots()
    tree.expandAll()
    assert tree.select_path(str(session.temp_path), "/entry_0000/data/q_xy")

    handle = main_window._acquire_edit_handle(session.temp_path)
    assert handle.is_open
    assert handle.file.mode == "r+"


def test_the_lister_reuses_the_open_write_handle(
    main_window, synthetic_nexus
):
    """With ``r+`` held, a listing must go through it, not open a second."""
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True

    handle = main_window._acquire_edit_handle(session.temp_path)
    names = [
        child.name
        for child in main_window._structure_list_children(
            str(session.temp_path), "/entry_0000")
    ]
    assert names == ["data"]
    # Still the same handle, still writable.
    assert main_window._h5_edit_handle is handle
    assert handle.is_open


def test_listing_an_unreadable_file_is_empty_not_an_error(main_window, tmp_path):
    missing = tmp_path / "gone.h5"
    assert main_window._structure_list_children(str(missing), "/") == []


# -- roots from the open sessions ------------------------------------------


def test_a_raw_batch_of_images_contributes_no_roots(main_window, synthetic_raw):
    """An image file has no structure, and 6000 of them are not a tree.

    Also the reason the extension test runs before ``h5py.is_hdf5``:
    that call opens the file, and a batch of thousands would mean
    thousands of opens every time this list is rebuilt.
    """
    from mlgidgui.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    specs = main_window._structure_root_specs()
    assert all(spec.editable is False for spec in specs)


def test_a_nexus_session_shows_under_its_original_name(main_window,
                                                       synthetic_nexus):
    """The working copy is what is edited; the original is what is known."""
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)

    specs = main_window._structure_root_specs()

    assert [s.label for s in specs] == ["synthetic.h5"]
    assert specs[0].path == str(session.temp_path)
    assert specs[0].path != str(synthetic_nexus)


def test_the_raw_root_list_is_capped(main_window, tmp_path, monkeypatch):
    """A batch can hold thousands of files; a tree of them is not a tree."""
    from mlgidgui.session import RawSession

    paths = []
    for i in range(6):
        path = tmp_path / f"raw_{i}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.arange(4.0))
        paths.append(path)

    main_window._sessions.append(RawSession.open(paths))
    monkeypatch.setattr(type(main_window), "_MAX_RAW_ROOTS", 3)

    specs = main_window._structure_root_specs()
    assert len(specs) == 3
    assert all(not spec.editable for spec in specs)
