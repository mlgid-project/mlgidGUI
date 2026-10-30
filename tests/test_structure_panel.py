"""The Structure tab, read-only: what it shows and what it refuses to do.

Two halves. The first drives ``StructurePanel`` directly, since it is a
pure display widget. The second drives the real ``MainWindow`` to pin
the wiring — a file-browser click reaches the panel, the work is
deferred while the tab is hidden, and nothing about the Image or Data
tabs changed.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent

from mlgidgui import h5_edit
from mlgidgui.h5_edit import Issue, LinkInfo, NodeInfo
from mlgidgui.structure_panel import StructurePanel, describe_node

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qtbot) -> StructurePanel:
    widget = StructurePanel()
    qtbot.addWidget(widget)
    return widget


def _info(**kwargs) -> NodeInfo:
    base = dict(path="/meta", kind="group", link=LinkInfo("hard"))
    base.update(kwargs)
    return NodeInfo(**base)


# -- describe_node ---------------------------------------------------------


def test_describe_a_dataset_reads_as_one_line():
    text = describe_node(_info(
        path="/entry_0000/data/img_gid_q", kind="dataset", dtype="float32",
        shape=(3, 16, 24), nbytes=4608, chunks=(1, 16, 24), compression="gzip",
    ))
    assert "float32" in text
    assert "3 × 16 × 24" in text
    assert "4.6 kB" in text
    assert "chunked" in text and "gzip" in text


def test_describe_a_scalar_and_a_group():
    assert "scalar" in describe_node(_info(kind="dataset", dtype="float64"))
    assert describe_node(
        _info(nx_class="NXsample", n_children=1)
    ) == "NXsample · 1 child"


def test_describe_a_link_never_pretends_to_know_its_contents():
    text = describe_node(_info(
        kind="link", link=LinkInfo("external", target="/entry", filename="scan.h5")
    ))
    assert text == "external link -> scan.h5::/entry"


# -- the panel -------------------------------------------------------------


def test_empty_panel_says_what_to_do(panel):
    assert "Select a group or a dataset" in panel._path_label.text()
    assert panel.current is None


def test_set_node_fills_path_type_and_attributes(panel):
    info = _info(path="/meta", kind="group", nx_class="NXcollection", n_children=2)
    panel.set_node(info, {"NX_class": "NXcollection", "comment": "hi"},
                   file_label="scan.h5")
    # The header is split at the last "/": where the node lives, then
    # its own name, which is the part that can be typed over.
    assert panel._path_label.text() == "scan.h5::/"
    assert panel._name_edit.text() == "meta"
    assert "NXcollection" in panel._type_label.text()
    assert panel.attribute_rows() == [
        ("NX_class", "NXcollection"), ("comment", "hi")
    ]
    assert panel.current is info


def test_the_value_card_shows_only_for_datasets(panel):
    panel.set_node(_info(kind="group"), {})
    assert not panel._value_card.isVisibleTo(panel)
    panel.set_node(_info(kind="dataset", dtype="int32", shape=(3,)), {},
                   preview="0, 1, 2")
    assert panel._value_card.isVisibleTo(panel)
    assert panel._value_label.text() == "0, 1, 2"


def test_the_link_card_shows_only_for_links(panel):
    panel.set_node(_info(kind="dataset"), {})
    assert not panel._link_card.isVisibleTo(panel)
    panel.set_node(
        _info(kind="link", link=LinkInfo("soft", target="/meta/counts")), {}
    )
    assert panel._link_card.isVisibleTo(panel)
    assert "/meta/counts" in panel._link_label.text()


def test_a_protected_node_carries_the_badge(panel):
    panel.set_node(_info(path="/entry_0000/data/q_xy", kind="dataset",
                         protection="the q_xy axis every fit reads"), {})
    assert panel._badge.isVisibleTo(panel)
    assert "q_xy axis" in panel._badge.text()
    panel.set_node(_info(path="/meta"), {})
    assert not panel._badge.isVisibleTo(panel)


def test_clear_empties_every_section(panel):
    panel.set_node(_info(kind="dataset", shape=(2,)), {"units": "m"},
                   preview="1, 2")
    panel.clear()
    assert panel.current is None
    assert panel.attribute_rows() == []
    assert not panel._value_card.isVisibleTo(panel)


def test_issues_are_listed_with_their_fix(panel):
    panel.set_issues([
        Issue("error", "/bad", "no 'data' group", fix="Add a 'data' group."),
        Issue("info", "/other", "no NX_class attribute"),
    ])
    rows = panel.issue_rows()
    assert rows[0][0] == "/bad"
    assert "no 'data' group" in rows[0][1] and "Add a 'data' group." in rows[0][1]
    assert len(rows) == 2


def test_a_clean_file_says_so(panel):
    panel.set_issues([])
    assert panel.issue_rows() == []
    assert panel._check_card._status.text() == "clean"


def test_recheck_button_emits(panel, qtbot):
    with qtbot.waitSignal(panel.recheckRequested, timeout=500):
        panel._recheck_btn.click()


# -- the wiring ------------------------------------------------------------
#
# Driven with a stand-in node and a patched ``_safe_selected_h5_nodes``,
# the idiom the existing browser tests use (``test_browser_entry_select``):
# the real ``_on_tree_selection_changed`` runs, so the deferral and the
# panel wiring are genuinely exercised, without needing a populated silx
# model whose rows resolve h5py objects.


class _FakeNode:
    """What the mixin reads off a silx node: the file and the in-file path."""

    def __init__(self, filename, local_name: str) -> None:
        self.local_filename = str(filename)
        self.local_name = local_name


@pytest.fixture
def opened(main_window, synthetic_nexus):
    from mlgidgui.session import NexusSession

    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    return main_window


def _click(window, h5_path: str, monkeypatch) -> _FakeNode:
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()
    return node


def test_the_window_has_the_three_tabs_in_order(main_window):
    labels = [main_window.tabs.tabText(i) for i in range(main_window.tabs.count())]
    assert labels == ["Image", "Data", "Structure"]


def test_a_click_is_deferred_while_the_structure_tab_is_hidden(opened, monkeypatch):
    """Browsing on the Image tab must not pay for a panel nobody sees."""
    opened.tabs.setCurrentWidget(opened.viewer)
    _click(opened, "/entry_0000/data", monkeypatch)
    assert opened._pending_structure_node is not None
    assert opened.structure_panel.current is None


def test_switching_to_the_tab_renders_the_deferred_node(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.viewer)
    _click(opened, "/entry_0000/data", monkeypatch)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    info = opened.structure_panel.current
    assert info is not None and info.path == "/entry_0000/data"
    assert opened._pending_structure_node is None


def test_a_click_while_the_tab_is_up_renders_immediately(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data/q_xy", monkeypatch)
    info = opened.structure_panel.current
    assert info is not None and info.path == "/entry_0000/data/q_xy"
    assert info.kind == "dataset" and info.shape == (24,)
    assert opened.structure_panel._value_label.text().startswith("-1.0")


def test_the_signal_attribute_is_shown(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data", monkeypatch)
    assert ("signal", "img_gid_q") in opened.structure_panel.attribute_rows()


def test_a_pipeline_path_is_badged(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data/img_gid_q", monkeypatch)
    panel = opened.structure_panel
    assert panel._badge.isVisibleTo(panel)
    assert panel.current.protection


def test_a_clean_synthetic_file_reports_nothing_serious(opened, monkeypatch):
    """The fixture entry has no NX_class, which is an information note,
    not something that stops the viewer or the pipeline."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data", monkeypatch)
    rows = opened.structure_panel.issue_rows()
    assert [w for _, w in rows] == ["no NX_class attribute  →  Set NX_class to NXentry."]


def test_closing_the_session_empties_the_panel(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data", monkeypatch)
    assert opened.structure_panel.current is not None
    opened._action_close_file()
    assert opened.structure_panel.current is None
    assert opened._structure_checked_file is None


def test_a_missing_node_is_reported_not_raised(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/nowhere", monkeypatch)
    assert opened.structure_panel.current.kind == "missing"


def test_structure_read_falls_back_to_a_plain_open(opened):
    """With no edit handle held, reads still work through a short 'r' open."""
    assert getattr(opened, "_h5_edit_handle", None) is None
    with opened._structure_read(opened.session.temp_path) as f:
        assert f is not None
        assert "entry_0000" in f


def test_structure_read_yields_none_for_an_unreadable_file(opened, tmp_path):
    with opened._structure_read(tmp_path / "gone.h5") as f:
        assert f is None


def test_an_image_row_says_there_is_no_structure(main_window, tmp_path):
    from PySide6.QtGui import QIcon

    from mlgidgui.browser_widgets import _ImageFileNode

    node = _ImageFileNode(str(tmp_path / "frame_0001.tif"), QIcon())
    main_window._render_structure_node(node)
    assert "not an HDF5 file" in main_window.structure_panel._path_label.text()
    assert main_window.structure_panel.current is None


def test_the_data_tab_still_gets_its_own_deferred_node(opened, monkeypatch):
    """The Data tab's existing behaviour is untouched by the new wiring."""
    opened.tabs.setCurrentWidget(opened.viewer)
    _click(opened, "/entry_0000/data/q_z", monkeypatch)
    assert opened._pending_data_node is not None
    opened.tabs.setCurrentWidget(opened.data_viewer)
    assert opened._pending_data_node is None


def test_recheck_re_runs_the_layout_check(opened, monkeypatch):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data", monkeypatch)
    assert opened._structure_checked_file is not None
    assert not any("signal" in w for _, w in opened.structure_panel.issue_rows())
    # Break the file behind the GUI's back, then ask for a re-check.
    # The detached scope is how every write in this app gets the file
    # free of read handles — the viewer's FrameSource holds one, and r+
    # behind an open r is exactly the case that fails.
    with opened._detached_silx_tree():
        with h5py.File(opened.session.temp_path, "r+") as f:
            del f["entry_0000/data"].attrs["signal"]
    opened._on_structure_recheck()
    assert any("signal" in what for _, what in opened.structure_panel.issue_rows())


def test_the_check_is_not_repeated_on_every_click(opened, monkeypatch):
    """One file, many clicks: the layout check runs once."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    _click(opened, "/entry_0000/data", monkeypatch)
    calls = []
    monkeypatch.setattr(
        h5_edit, "validate", lambda f: calls.append(1) or [])
    _click(opened, "/entry_0000/data/q_xy", monkeypatch)
    _click(opened, "/entry_0000/data/q_z", monkeypatch)
    assert calls == []


def test_a_raw_session_is_marked_read_only(main_window, synthetic_raw, monkeypatch):
    from mlgidgui.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    node = _FakeNode(synthetic_raw, "/")
    monkeypatch.setattr(main_window, "_safe_selected_h5_nodes", lambda: [node])
    main_window._on_tree_selection_changed()
    assert "read-only" in main_window.structure_panel._note_label.text()


def test_an_external_link_is_described_without_being_opened(main_window, tmp_path):
    """The rule that keeps a 123 GB master usable, end to end."""
    linked = tmp_path / "scan.h5"
    with h5py.File(linked, "w") as f:
        f.create_dataset("payload", data=np.zeros(4))
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as f:
        f["scan"] = h5py.ExternalLink(str(linked), "/payload")
    linked.unlink()  # opening it now would raise

    main_window._render_structure_node(_FakeNode(master, "/scan"))
    panel = main_window.structure_panel
    assert panel.current.kind == "link"
    assert "external link" in panel._type_label.text()


# -- the name in the header ------------------------------------------------
#
# Renaming already had two routes, both through a dialog. This is the
# third and the shortest: the name is printed at the head of the panel,
# so typing over it is the obvious gesture. What is pinned here is that
# it is hard to do BY ACCIDENT — a rename writes to the user's file.


def _named(panel, path="/entry_0000/data/q_xy"):
    panel.set_node(_info(path=path, kind="dataset", dtype="f4", shape=(4,)), {})
    panel.set_editable(True)
    return panel._name_edit


def test_the_root_has_no_name_to_change(panel):
    panel.set_node(_info(path="/", kind="group"), {}, file_label="scan.h5")
    assert panel._path_label.text() == "scan.h5::/"
    assert not panel._name_edit.isVisibleTo(panel)


def test_the_name_starts_read_only(panel):
    name = _named(panel)
    assert name.text() == "q_xy"
    assert not name.editing


def test_a_committed_name_is_emitted_once(panel, qtbot):
    name = _named(panel)
    seen = []
    panel.renameRequested.connect(seen.append)

    name.begin()
    name.setText("q_par")
    name.returnPressed.emit()

    assert seen == ["q_par"]
    # And the field is a title again, showing what the file still says
    # until the window comes back with the rename applied.
    assert not name.editing
    assert name.text() == "q_xy"


def test_escape_puts_the_old_name_back(panel, qtbot):
    name = _named(panel)
    seen = []
    panel.renameRequested.connect(seen.append)

    name.begin()
    name.setText("nonsense")
    qtbot.keyClick(name, Qt.Key.Key_Escape)

    assert seen == []
    assert name.text() == "q_xy"
    assert not name.editing


def test_clicking_away_reverts_rather_than_renames(panel):
    """Losing focus must not write to the file.

    A rename that happened because the user clicked somewhere else is a
    change they did not ask for. Forgetting Enter costs them the typing.
    """
    name = _named(panel)
    seen = []
    panel.renameRequested.connect(seen.append)

    name.begin()
    name.setText("half-typed")
    name.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))

    assert seen == []
    assert name.text() == "q_xy"


def test_the_same_name_is_not_a_rename(panel):
    name = _named(panel)
    seen = []
    panel.renameRequested.connect(seen.append)

    name.begin()
    name.returnPressed.emit()
    name.begin()
    name.setText("   ")
    name.returnPressed.emit()

    assert seen == []


def test_a_read_only_node_cannot_be_typed_over(panel):
    """Raw inputs and files with no session behind them."""
    panel.set_node(_info(path="/entry_0000", kind="group"), {})
    panel.set_editable(False)

    panel._name_edit.begin()

    assert not panel._name_edit.editing


def test_a_single_click_does_not_start_an_edit(panel, qtbot):
    """Same rule the attribute table already applies."""
    name = _named(panel)
    qtbot.mouseClick(name, Qt.MouseButton.LeftButton)
    assert not name.editing


def test_the_name_field_does_not_sit_on_ctrl_z(panel):
    """A QLineEdit has an undo stack of its own.

    In this tab Ctrl+Z means "undo my last edit to the file", so a
    read-only title that swallowed the key while it happened to hold
    focus would be a quiet, infuriating failure. It passes the key up
    instead. While the field IS being edited it keeps it: there the
    user's own typing is what they mean to take back, and nothing has
    been written yet.
    """
    name = _named(panel)
    undo = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z,
                     Qt.KeyboardModifier.ControlModifier)

    name.keyPressEvent(undo)
    assert not undo.isAccepted()

    name.begin()
    kept = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z,
                     Qt.KeyboardModifier.ControlModifier)
    name.keyPressEvent(kept)
    assert kept.isAccepted()

