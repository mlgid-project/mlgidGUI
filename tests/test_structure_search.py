"""Find: locating a node by name, attribute name or attribute value.

Stage eight of the editor. Search is a jump list, not a filter on the
browser tree — filtering silx's model would ask it to resolve rows to
answer the query, which is the one thing this tab never does. The tests
that matter here are the ones proving a link is never followed and that
a partial answer says so.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import h5_edit
from mlgidgui.session import NexusSession

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


@pytest.fixture
def editing(main_window, synthetic_nexus):
    with h5py.File(synthetic_nexus, "r+") as f:
        sample = f["entry_0000"].create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("chemical_formula", data="C62H68N2O2S5")
        f["entry_0000/data/q_xy"].attrs["units"] = "1/Angstrom"
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window._reattach_silx_tree()
    return main_window


def _show(window, h5_path, monkeypatch):
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


# -- what it finds ---------------------------------------------------------


def test_a_name_match(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("q_xy")
    rows = editing.structure_panel.search_rows()
    assert any("/entry_0000/data/q_xy" in row for row in rows)


def test_an_attribute_name_match(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("NX_class")
    rows = editing.structure_panel.search_rows()
    assert any("sample" in row and "NX_class" in row for row in rows)


def test_an_attribute_value_match(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("1/Angstrom")
    rows = editing.structure_panel.search_rows()
    assert any("q_xy" in row and "1/Angstrom" in row for row in rows)


def test_the_search_is_case_insensitive_by_default(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("SAMPLE")
    assert editing.structure_panel.search_rows()
    assert not editing.structure_panel.case_sensitive()


# The two things that make a common word usable in a real file. Searching
# "Si" in a silicon dataset also matches every "signal" and every
# "silicon" in the file, and no amount of reading the list fixes that.


def test_a_path_query_narrows_a_common_word(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)

    editing._on_structure_search("formula")
    broad = editing.structure_panel.search_rows()
    editing._on_structure_search("sample/formula")
    narrow = editing.structure_panel.search_rows()

    assert any("chemical_formula" in row for row in narrow)
    assert len(narrow) <= len(broad)
    # The lead has to be somewhere on the path.
    editing._on_structure_search("detector/formula")
    assert editing.structure_panel.search_rows() == []


def test_a_dataset_value_is_searchable(editing, monkeypatch):
    """``chemical_formula`` holds its value in the dataset, not an
    attribute — which is where NeXus keeps most of its metadata."""
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("C62H68N2O2S5")
    rows = editing.structure_panel.search_rows()
    assert any("chemical_formula" in row for row in rows)


def test_the_case_toggle_changes_the_answer(editing, monkeypatch):
    panel = editing.structure_panel
    _show(editing, "/entry_0000", monkeypatch)
    panel._search_edit.setText("SAMPLE")

    panel._on_search()
    assert panel.search_rows()

    panel._case_btn.setChecked(True)
    assert panel.case_sensitive()
    assert panel.search_rows() == []


def test_flipping_the_toggle_re_runs_the_search(editing, monkeypatch):
    """It answers a question rather than only arming the next search."""
    panel = editing.structure_panel
    _show(editing, "/entry_0000", monkeypatch)
    asked = []
    panel.searchRequested.connect(lambda text, cs: asked.append((text, cs)))
    panel._search_edit.setText("sample")

    panel._case_btn.setChecked(True)

    assert asked == [("sample", True)]


def test_an_empty_box_is_not_re_run_by_the_toggle(editing, monkeypatch):
    panel = editing.structure_panel
    _show(editing, "/entry_0000", monkeypatch)
    asked = []
    panel.searchRequested.connect(lambda text, cs: asked.append((text, cs)))

    panel._case_btn.setChecked(True)

    assert asked == []


def test_no_match_says_so(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("zzzz_nothing")
    assert editing.structure_panel.search_rows() == []
    assert editing.structure_panel._search_card._status.text() == "no match"


def test_an_empty_term_searches_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing.structure_panel._search_edit.setText("   ")
    editing.structure_panel._on_search()
    assert editing.structure_panel.search_rows() == []


# -- what it must not do ---------------------------------------------------


def test_an_external_link_is_never_followed(main_window, tmp_path, monkeypatch):
    """The rule that keeps a master of linked scans searchable."""
    linked = tmp_path / "scan.h5"
    with h5py.File(linked, "w") as f:
        f.create_dataset("wavelength", data=np.float64(1.54))
    master = tmp_path / "master.h5"
    with h5py.File(master, "w", track_order=True) as f:
        f.create_dataset("wavelength", data=np.float64(1.0))
        f["scan"] = h5py.ExternalLink(str(linked), "/")
        # A real entry too, so opening it is an ordinary session rather
        # than one the app warns about for having nothing to display.
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=np.linspace(0.1, 1.0, 16, dtype="f4").reshape(1, 4, 4))
        data.create_dataset("q_xy", data=np.linspace(-1, 1, 4, dtype="f4"))
        data.create_dataset("q_z", data=np.linspace(0, 2, 4, dtype="f4"))
    linked.unlink()  # following the link now would raise

    session = NexusSession.open(master)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    node = _FakeNode(session.temp_path, "/")
    monkeypatch.setattr(main_window, "_safe_selected_h5_nodes", lambda: [node])
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window._on_tree_selection_changed()

    main_window._on_structure_search("wavelength")
    rows = main_window.structure_panel.search_rows()
    assert any(row.startswith("/wavelength") for row in rows)
    assert not any("/scan/" in row for row in rows), (
        "the linked file must not be walked")


def test_a_truncated_search_says_it_is_partial(editing, monkeypatch):
    monkeypatch.setattr(
        h5_edit, "walk_search",
        lambda f, q, **k: ([h5_edit.SearchHit("/x", "name", "x")], True),
    )
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("anything")
    assert "stopped early" in editing.structure_panel._search_card._status.text()


def test_a_failed_search_leaves_the_panel_usable(editing, monkeypatch):
    def _boom(*a, **k):
        raise OSError("gone")

    monkeypatch.setattr(h5_edit, "walk_search", _boom)
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search("q_xy")
    assert editing.structure_panel.search_rows() == []


# -- jumping to a hit ------------------------------------------------------


def test_activating_a_hit_selects_it_in_the_browser(editing, monkeypatch):
    view = editing.tree
    view.expand(view.model().index(0, 0, view.rootIndex()))
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search_result("/entry_0000/data")
    key = (str(editing.session.temp_path), ("entry_0000", "data"))
    assert view.currentIndex() == editing._find_tree_index(key)


def test_a_hit_inside_a_collapsed_subtree_is_still_reached(editing, monkeypatch):
    """No row is visible for it, but the walk populates as it descends.

    The tree model fills a group's children the first time anything asks
    for them, so a hit three levels down resolves without the user
    having expanded anything.
    """
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search_result("/entry_0000/sample/chemical_formula")
    key = (str(editing.session.temp_path),
           ("entry_0000", "sample", "chemical_formula"))
    index = editing._find_tree_index(key)
    assert index is not None and index.isValid()
    assert editing.tree.currentIndex() == index


def test_a_hit_that_no_longer_exists_is_reported_not_raised(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    editing._on_structure_search_result("/entry_0000/gone")
    assert editing.structure_panel.current.kind == "missing"
