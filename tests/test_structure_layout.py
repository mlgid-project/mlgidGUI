"""What the Structure tab does to the window around it.

Two behaviours asked for after the first manual pass: the side and
bottom docks fold away while the tab is up, because nothing in them
applies to editing a file's structure; and the workflow strip above the
image can be folded away for good, because not everyone works from it.

The restoring half is what these tests really guard. "Hide them" is
easy; "put back exactly what was open, and nothing else" is where this
kind of feature usually goes wrong.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from mlgidgui.main_window_build import RAIL_EXPANDED_KEY
from mlgidgui.session import NexusSession
from mlgidgui.workflow_rail import WorkflowRail

pytestmark = pytest.mark.gui

FOLDED = (
    "_tree_dock", "_display_dock", "_pipeline_dock", "_sim_dock",
    "_conversion_dock", "_logs_dock", "_profile_dock", "_peaks_dock",
    "_scan_tracking_dock",
)


@pytest.fixture
def opened(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    yield main_window
    # Entering the Structure tab now takes the write handle, and the
    # temp dir is removed by the session-cleanup fixture.
    if main_window._h5_edit_handle is not None:
        main_window._h5_edit_handle.release()


def _visible(window):
    return {
        name for name in FOLDED
        if getattr(window, name, None) is not None
        and getattr(window, name).isVisible()
    }


# -- the docks -------------------------------------------------------------


def test_the_structure_tab_folds_the_side_and_bottom_docks(opened):
    before = _visible(opened)
    assert before, "the fixture must start with docks open"
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()


def test_the_file_browser_folds_away_too(opened):
    """It used to stay open as this tab's navigator. The tab has its own
    tree now, so leaving the browser up would show one file twice."""
    assert opened._tree_dock.isVisible()
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert not opened._tree_dock.isVisible()
    opened.tabs.setCurrentWidget(opened.viewer)
    assert opened._tree_dock.isVisible()


def test_leaving_the_tab_puts_them_back(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_a_dock_the_user_had_closed_stays_closed(opened):
    """Restoring must not open something the user deliberately shut."""
    opened._logs_dock.hide()
    before = _visible(opened)
    assert "_logs_dock" not in before
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.data_viewer)
    assert _visible(opened) == before
    assert not opened._logs_dock.isVisible()


def test_switching_between_image_and_data_changes_nothing(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.data_viewer)
    assert _visible(opened) == before
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_a_mode_change_cannot_pop_the_docks_back_up(opened):
    """_apply_session_mode decides visibility from scratch."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()
    opened._apply_session_mode(opened.session)
    assert _visible(opened) == set()


def test_the_docks_survive_a_round_trip_through_a_mode_change(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened._apply_session_mode(opened.session)
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_folding_is_a_no_op_with_no_session(main_window):
    """The welcome page is up and the docks are already hidden."""
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window.tabs.setCurrentWidget(main_window.viewer)


# -- the workflow strip ----------------------------------------------------


@pytest.fixture
def rail(qtbot) -> WorkflowRail:
    widget = WorkflowRail()
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_the_strip_starts_open(rail):
    assert rail.is_expanded()
    assert all(block.isVisible() for block in rail._blocks)
    assert not rail._folded_label.isVisible()


def test_folding_hides_the_chips_and_names_itself(rail):
    rail.set_expanded(False)
    assert not any(block.isVisible() for block in rail._blocks)
    assert rail._folded_label.isVisible()
    assert rail._folded_label.text() == "Workflow"


def test_folding_reclaims_the_height(rail):
    open_height = rail.sizeHint().height()
    rail.set_expanded(False)
    assert rail.sizeHint().height() < open_height


def test_the_toggle_emits_once_per_change(rail, qtbot):
    with qtbot.waitSignal(rail.expandedChanged, timeout=500) as caught:
        rail.set_expanded(False)
    assert caught.args == [False]
    # Setting the same value again is not a change and must not re-emit.
    seen = []
    rail.expandedChanged.connect(seen.append)
    rail.set_expanded(False)
    assert seen == []


def test_the_chips_still_answer_while_folded(rail):
    """Folding is presentation only — the state underneath keeps updating."""
    rail.set_expanded(False)
    rail.set_state("detect", "12 this frame", "ok")
    rail.set_expanded(True)
    assert rail.chips["detect"].state.text() == "12 this frame"


def test_the_choice_is_remembered(main_window):
    settings = QSettings()
    previous = settings.value(RAIL_EXPANDED_KEY, None)
    try:
        main_window.workflow_rail.set_expanded(False)
        assert str(settings.value(RAIL_EXPANDED_KEY)).lower() in ("false", "0")
        main_window.workflow_rail.set_expanded(True)
        assert str(settings.value(RAIL_EXPANDED_KEY)).lower() in ("true", "1")
    finally:
        if previous is None:
            settings.remove(RAIL_EXPANDED_KEY)
        else:
            settings.setValue(RAIL_EXPANDED_KEY, previous)


# -- what folding the docks must not leave behind -------------------------
#
# Reported from a manual pass: controls above the image were frequently
# unclickable. Hiding and re-showing docks that share a tab group is a
# re-tabify as far as Qt is concerned, and this window carries two fixes
# for what that leaves behind — a stale QTabBar painted into a corner,
# and fresh tabs with no glyph. The fold has to run both.


def test_folding_runs_the_dock_chrome_cleanup(opened, monkeypatch):
    calls = []
    monkeypatch.setattr(
        opened, "_hide_stale_dock_tab_bars", lambda: calls.append("bars"))
    monkeypatch.setattr(
        opened, "_apply_dock_tab_icons", lambda: calls.append("icons"))
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert calls == ["bars", "icons"]
    calls.clear()
    opened.tabs.setCurrentWidget(opened.viewer)
    assert calls == ["bars", "icons"]


def test_the_cleanup_is_skipped_when_nothing_was_folded(opened, monkeypatch):
    """No dock changed, so there is nothing for Qt to have mangled."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    calls = []
    monkeypatch.setattr(
        opened, "_hide_stale_dock_tab_bars", lambda: calls.append("bars"))
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert calls == []


def test_a_failing_cleanup_does_not_break_the_switch(opened, monkeypatch):
    def _boom():
        raise RuntimeError("Qt said no")

    monkeypatch.setattr(opened, "_hide_stale_dock_tab_bars", _boom)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()


def test_no_dock_tab_bar_is_left_showing_an_empty_tab_set(opened):
    """A visible bar with no tabs is a ghost with nothing to click."""
    from PySide6.QtWidgets import QTabBar

    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.viewer)
    ghosts = [
        tb for tb in opened.findChildren(QTabBar)
        if tb.parent() is opened and tb.isVisible() and tb.count() == 0
    ]
    assert ghosts == []


def test_the_rail_has_no_widget_outside_its_layout(qtbot):
    """An unmanaged child sits at (0, 0) on top of whatever is there."""
    from PySide6.QtWidgets import QWidget

    rail = WorkflowRail()
    qtbot.addWidget(rail)
    rail.show()
    managed = set()
    layout = rail.layout()
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            managed.add(id(widget))
    stray = [c for c in rail.children()
             if isinstance(c, QWidget) and id(c) not in managed]
    assert stray == []


# -- taking the write handle up front -------------------------------------


def test_opening_the_tab_takes_the_write_handle(opened):
    """So the first edit is as immediate as every edit after it."""
    assert opened._h5_edit_handle is None
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert opened._h5_edit_handle is not None
    assert opened._h5_edit_handle.is_open


def test_a_raw_session_is_not_opened_for_writing(main_window, synthetic_raw):
    from mlgidgui.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    assert main_window._h5_edit_handle is None


def test_a_failure_to_take_the_handle_is_silent(opened, monkeypatch):
    """The user has asked for nothing yet; the next edit reports it."""
    from mlgidgui.h5_edit import EditError

    def _boom(_path):
        raise EditError("nope")

    monkeypatch.setattr(opened, "_acquire_edit_handle", _boom)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert opened.structure_panel is opened.tabs.currentWidget()


# -- the control strip above the image ------------------------------------
#
# Reported twice: the view buttons were dead on the first file loaded and
# came alive after loading a second one. On a single-frame stack every
# transport widget is hidden, so their cluster reports a zero size hint
# and the flow layout SKIPS it — and a skipped widget keeps its default
# 640x480 geometry at the parent's origin, floating over the strip and
# eating the clicks. A second, multi-frame file gave the cluster real
# content, which is why it looked like a fix.


def _single_frame_nexus(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "one_frame.h5"
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=np.linspace(0.1, 1.0, 64, dtype="f4").reshape(1, 8, 8))
        data.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype="f4"))
        data.create_dataset("q_z", data=np.linspace(0, 2, 8, dtype="f4"))
    return path


@pytest.fixture
def single_frame(main_window, tmp_path):
    session = NexusSession.open(_single_frame_nexus(tmp_path))
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    return main_window


def test_a_single_frame_file_hides_the_transport(single_frame):
    assert single_frame.viewer.n_frames == 1
    assert not single_frame.frame_slider.isVisible()


def test_the_view_buttons_are_hit_testable_on_the_first_file(single_frame):
    """The reported bug, as a hit test: the click must reach the button."""
    button = single_frame.viewer._radio_cart
    point = button.mapTo(single_frame, button.rect().center())
    assert single_frame.childAt(point) is button


def test_an_emptied_cluster_covers_nothing(single_frame):
    group = single_frame.viewer._frames_group
    assert not group.has_visible_members()
    assert group.isHidden() or group.size().isEmpty(), (
        "an empty cluster must not keep a default 640x480 rectangle")


def test_the_cluster_comes_back_when_the_transport_does(single_frame):
    """Collapsing is geometry only, so a refilled cluster lays out again."""
    viewer = single_frame.viewer
    single_frame._set_frame_slider_visible(True)
    assert viewer._frames_group.has_visible_members()
    assert not viewer._frames_group.isHidden()
    button = viewer._radio_cart
    point = button.mapTo(single_frame, button.rect().center())
    assert single_frame.childAt(point) is button


def test_hiding_the_view_cluster_also_settles_the_strip(single_frame):
    """The same path a raw session takes."""
    viewer = single_frame.viewer
    viewer.set_mode_radios_visible(False)
    assert viewer._view_group.isHidden() or viewer._view_group.size().isEmpty()
    viewer.set_mode_radios_visible(True)
    button = viewer._radio_cart
    point = button.mapTo(single_frame, button.rect().center())
    assert single_frame.childAt(point) is button


# -- the workspace ---------------------------------------------------------
#
# The tab was one tall scrolling column. It is now five resizable
# regions, and the promise is that all five are on screen at once: only
# the tables and lists inside them scroll. These pin that promise, since
# it is the kind that erodes one added widget at a time.


@pytest.fixture
def workspace(opened):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.resize(1100, 760)
    opened.show()
    return opened.structure_panel


def _regions(panel):
    return {
        "find": panel._search_card,
        "tree": panel._tree_card,
        "attributes": panel._attr_card,
        "check": panel._check_card,
        "changes": panel._changes_card,
    }


def test_the_page_itself_does_not_scroll(workspace):
    """No QScrollArea anywhere in the tab — that is the whole request."""
    from PySide6.QtWidgets import QScrollArea

    assert workspace.findChildren(QScrollArea) == []


def test_every_region_is_on_screen_at_once(workspace):
    for name, card in _regions(workspace).items():
        assert card.isVisible(), name
        assert card.height() > 0 and card.width() > 0, name


def test_the_regions_still_fit_a_small_window(opened):
    """A laptop-sized window must not push a section off the tab."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.resize(900, 620)
    opened.show()
    panel = opened.structure_panel
    for name, card in _regions(panel).items():
        assert card.isVisible(), name
    assert panel.minimumSizeHint().height() <= 620


def test_the_tables_keep_their_own_scrollbars(workspace):
    """Scrolling moved inside the sections; it did not go away."""
    from PySide6.QtCore import Qt

    for view in (workspace._attr_table, workspace._issue_table,
                 workspace._changes_list, workspace._search_list,
                 workspace.node_tree):
        assert view.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_a_section_cannot_be_collapsed_out_of_existence(workspace):
    for split in workspace._splits.values():
        assert not split.childrenCollapsible()


def test_the_division_is_remembered(workspace):
    from mlgidgui.structure_panel import _SPLIT_KEYS, StructurePanel

    split = workspace._splits["top"]
    split.setSizes([300, 700])
    workspace._save_split("top", split)
    stored = StructurePanel._stored_sizes("top", 2)
    assert stored == split.sizes()

    QSettings().remove(_SPLIT_KEYS["top"])


def test_a_stale_division_is_ignored_rather_than_applied(workspace):
    """A stored list from another layout would misplace a section."""
    from mlgidgui.structure_panel import _SPLIT_KEYS, StructurePanel

    QSettings().setValue(_SPLIT_KEYS["top"], "1,2,3")
    assert StructurePanel._stored_sizes("top", 2) is None
    QSettings().setValue(_SPLIT_KEYS["top"], "not,sizes")
    assert StructurePanel._stored_sizes("top", 2) is None
    QSettings().remove(_SPLIT_KEYS["top"])


# -- the action row --------------------------------------------------------


def test_the_action_row_routes_to_the_same_handlers(workspace, opened,
                                                    monkeypatch):
    """The row and the context menu must not be two sets of rules.

    Per-node verbs are handed the selected target. Copy, cut and delete
    are handed nothing, on purpose: they act on the whole tree
    selection, so each reads it for itself rather than being pinned to
    whichever row a shift-pick left current.
    """
    called = []
    monkeypatch.setattr(
        opened, "_structure_selected_target", lambda: ("f.h5", "/entry"))
    monkeypatch.setattr(
        opened, "_structure_selected_targets", lambda: [("f.h5", "/entry")])
    for verb, handler in opened._STRUCTURE_ACTIONS.items():
        monkeypatch.setattr(
            opened, handler,
            lambda target, v=verb: called.append((v, target)))
        workspace.nodeActionRequested.emit(verb)
    assert [v for v, _ in called] == list(opened._STRUCTURE_ACTIONS)
    for verb, target in called:
        if verb in opened._STRUCTURE_MULTI_VERBS:
            assert target is None, verb
        else:
            assert target == ("f.h5", "/entry"), verb


def test_the_action_row_no_ops_with_nothing_selected(workspace, opened,
                                                     monkeypatch):
    monkeypatch.setattr(opened, "_structure_selected_target", lambda: None)
    monkeypatch.setattr(
        opened, "_on_structure_delete",
        lambda target: pytest.fail("deleted with nothing selected"))
    workspace.nodeActionRequested.emit("delete")


def test_add_actions_are_off_unless_a_group_is_selected(workspace, opened,
                                                        monkeypatch):
    from mlgidgui import h5_edit

    class _Node:
        local_filename = ""
        local_name = ""

    node = _Node()
    node.local_filename = str(opened.session.temp_path)
    node.local_name = "/entry_0000/data/q_xy"
    opened._render_structure_node(node)
    assert not workspace._new_btn.isEnabled()

    node.local_name = "/entry_0000/data"
    opened._render_structure_node(node)
    assert workspace._new_btn.isEnabled()
    assert h5_edit.normalize_path(node.local_name) == "/entry_0000/data"


# -- the two deferrals -----------------------------------------------------
#
# Both exist because the Structure tab hides what they update. Neither
# may be skipped: a browser holding deleted nodes is a crash waiting for
# a click, and an entry never applied means the Image tab shows a
# different scan than the one just edited.


def _select(window, h5_path):
    window.structure_panel.node_tree.select_path(
        str(window.session.temp_path), h5_path)


@pytest.fixture
def rename_to(monkeypatch):
    """Answer the rename dialog, and the protected-path confirm behind it.

    ``q_xy`` is a node the viewer reads, so renaming it asks first —
    which is the behaviour under test everywhere else, and a hang here.
    """
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **k: ("renamed", True)))
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))


@pytest.fixture
def editing_tab(opened, rename_to):
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened._refresh_structure_tree_roots()
    return opened


def test_an_edit_behind_the_folded_browser_defers_its_rebuild(editing_tab,
                                                              monkeypatch):
    rebuilt = []
    monkeypatch.setattr(
        editing_tab, "_rebuild_tree_preserving",
        lambda *a, **k: rebuilt.append(True))

    _select(editing_tab, "/entry_0000/data/q_xy")
    editing_tab._on_structure_rename(
        (editing_tab.session.temp_path, "/entry_0000/data/q_xy"))

    assert rebuilt == []
    assert editing_tab._browser_needs_rebuild


def test_leaving_the_tab_pays_the_deferred_rebuild(editing_tab):
    _select(editing_tab, "/entry_0000/data/q_xy")
    editing_tab._on_structure_rename(
        (editing_tab.session.temp_path, "/entry_0000/data/q_xy"))
    assert editing_tab._browser_needs_rebuild

    editing_tab.tabs.setCurrentWidget(editing_tab.viewer)

    assert not editing_tab._browser_needs_rebuild
    # And the browser really does show the new name.
    key = (str(editing_tab.session.temp_path), ("entry_0000", "data", "renamed"))
    assert editing_tab._find_tree_index(key) is not None


def test_reopening_the_browser_by_hand_pays_it_too(editing_tab):
    _select(editing_tab, "/entry_0000/data/q_xy")
    editing_tab._on_structure_rename(
        (editing_tab.session.temp_path, "/entry_0000/data/q_xy"))
    assert editing_tab._browser_needs_rebuild

    editing_tab._tree_dock.show()

    assert not editing_tab._browser_needs_rebuild


def test_an_edit_with_the_browser_open_rebuilds_at_once(opened, rename_to,
                                                        monkeypatch):
    """Nothing is deferred when the browser is on screen — the old path."""
    rebuilt = []
    monkeypatch.setattr(
        opened, "_rebuild_tree_preserving",
        lambda *a, **k: rebuilt.append(True))
    assert opened._tree_dock.isVisible()

    opened._on_structure_rename(
        (opened.session.temp_path, "/entry_0000/data/q_xy"))

    assert rebuilt == [True]
    assert not opened._browser_needs_rebuild


def test_clicking_the_tree_does_not_reload_the_viewer(editing_tab, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        editing_tab, "_activate_entry_for_node",
        lambda node: loaded.append(node))

    _select(editing_tab, "/entry_0000/data/q_xy")

    assert loaded == []
    assert editing_tab._structure_pending_activation is not None


def test_leaving_the_tab_applies_the_entry_once(editing_tab, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        editing_tab, "_activate_entry_for_node",
        lambda node: loaded.append(node))

    _select(editing_tab, "/entry_0000/data/q_xy")
    _select(editing_tab, "/entry_0000/data/q_z")
    editing_tab.tabs.setCurrentWidget(editing_tab.viewer)

    assert len(loaded) == 1
    assert loaded[0].local_name == "/entry_0000/data/q_z"
    assert editing_tab._structure_pending_activation is None


def test_a_click_from_another_tab_still_switches_at_once(opened, monkeypatch):
    """Only the Structure tab defers; the browser's own clicks do not."""
    loaded = []
    monkeypatch.setattr(
        opened, "_activate_entry_for_node", lambda node: loaded.append(node))
    opened.tabs.setCurrentWidget(opened.viewer)
    opened._refresh_structure_tree_roots()

    opened._on_structure_tree_selected(
        str(opened.session.temp_path), "/entry_0000/data/q_xy")

    assert len(loaded) == 1


# The entry combo stays on the toolbar while every dock folds away, so
# it is the one control that can still move the viewer from inside the
# tab. The read behind it is already off the GUI thread; the install is
# not, and that is what the user feels.


def _entry(window):
    return window.entry_combo.currentText()


def test_switching_entry_in_the_tab_does_not_load_the_frame(editing_tab,
                                                            monkeypatch):
    loaded = []
    monkeypatch.setattr(
        editing_tab, "_load_entry_async", lambda entry: loaded.append(entry))

    editing_tab._on_entry_changed(_entry(editing_tab))

    assert loaded == []
    assert editing_tab._structure_pending_entry == _entry(editing_tab)


def test_leaving_the_tab_loads_the_entry_once(editing_tab, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        editing_tab, "_load_entry_async", lambda entry: loaded.append(entry))
    entry = _entry(editing_tab)
    editing_tab._on_entry_changed(entry)
    editing_tab._on_entry_changed(entry)

    editing_tab.tabs.setCurrentWidget(editing_tab.viewer)

    assert loaded == [entry]
    assert editing_tab._structure_pending_entry is None


def test_an_entry_change_outside_the_tab_loads_at_once(opened, monkeypatch):
    """Only this tab defers. Everywhere else the viewer is on screen."""
    loaded = []
    monkeypatch.setattr(
        opened, "_load_entry_async", lambda entry: loaded.append(entry))
    opened.tabs.setCurrentWidget(opened.viewer)

    opened._on_entry_changed(_entry(opened))

    assert loaded == [_entry(opened)]
    assert opened._structure_pending_entry is None


def test_a_file_closed_behind_the_tab_leaves_nothing_to_catch_up_on(
    editing_tab, monkeypatch
):
    """The combo is the source of truth, not the remembered name.

    A name remembered from a session that has since been closed would
    be loaded out of a file that is no longer open.
    """
    loaded = []
    monkeypatch.setattr(
        editing_tab, "_load_entry_async", lambda entry: loaded.append(entry))
    editing_tab._on_entry_changed(_entry(editing_tab))
    editing_tab.entry_combo.blockSignals(True)
    editing_tab.entry_combo.clear()
    editing_tab.entry_combo.blockSignals(False)

    editing_tab.tabs.setCurrentWidget(editing_tab.viewer)

    assert loaded == []
    assert editing_tab._structure_pending_entry is None


# Putting the browser's selection back after a rebuild used to re-enter
# the rebuild itself and then use an index the reset had freed. It
# crashed the process rather than raising: 2026-08-26 10:13:26,
# ``segfault at 20 ... in libQt6Core.so.6``, which resolves to
# ``QSortFilterProxyModel::parent`` — the proxy walking a dead internal
# pointer. A test cannot observe a segfault, so it pins the cause.


def test_putting_the_browser_selection_back_is_not_a_click(opened, monkeypatch):
    """The restore must not run the selection handler.

    The handler promotes the selected node's file to the active
    session; with the Structure tab in front that reaches
    ``_detach_silx_tree`` and resets the model mid-restore, leaving the
    line after it holding a freed index.
    """
    from PySide6.QtCore import QItemSelectionModel, QModelIndex

    promoted = []
    monkeypatch.setattr(
        opened, "_activate_session_for_node", lambda node: promoted.append(node))
    # The fixture opens a session but not the browser row for it; the
    # real open path does this in ``_on_open_finished``.
    opened.tree.findHdf5TreeModel().insertFile(str(opened.session.temp_path))
    root = opened.tree.model().index(0, 0, QModelIndex())
    assert root.isValid()
    opened.tree.selectionModel().select(
        root,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    state = opened._capture_tree_state()
    assert state[1] is not None, "nothing selected — the test proves nothing"
    promoted.clear()

    opened._rebuild_tree_preserving(state)

    assert promoted == []
    assert not opened._restoring_tree_selection


def test_the_flag_is_lowered_even_when_the_restore_blows_up(opened,
                                                            monkeypatch):
    """A stuck flag would silently deafen every later click."""
    monkeypatch.setattr(
        opened, "_find_tree_index",
        lambda key: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        opened._restore_tree_state(([("f", ())], None))

    assert not opened._restoring_tree_selection


# -- the region boxes ------------------------------------------------------


def test_every_region_sits_in_a_box(workspace):
    """A card's only chrome is a hairline under its title.

    Stacked in a dock that is right. Given a pane of its own it is not:
    the hairline runs out into open space and nothing says where the
    section ends, which is exactly how the first cut of this layout
    looked.
    """
    from PySide6.QtWidgets import QFrame

    from mlgidgui.skin import REGION_NAME

    boxes = [
        f for f in workspace.findChildren(QFrame)
        if f.objectName() == REGION_NAME
    ]
    assert len(boxes) == 5
    for card in _regions(workspace).values():
        assert any(box.isAncestorOf(card) for box in boxes), card.title()
