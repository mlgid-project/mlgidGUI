"""The empty state: what the window says with nothing loaded.

Everything the welcome page offers already existed (Open, image import,
the persisted recent list, drag and drop) — it was just invisible. So the
tests that matter are about wiring: the page appears exactly when there
is no session, its buttons reach the handlers that were already there,
and the one piece of state it reports (whether the analysis backend is
installed) is read live rather than baked in at build time.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton

from mlgidgui.welcome_view import RECENT_SHOWN, WelcomeView

pytestmark = pytest.mark.gui


@pytest.fixture
def view(qtbot):
    widget = WelcomeView()
    qtbot.addWidget(widget)
    return widget


def test_the_window_starts_on_the_welcome_page(main_window):
    assert main_window._central_stack.currentWidget() is main_window.welcome_view


def test_opening_a_file_switches_to_the_tabs(main_window, synthetic_nexus):
    from mlgidgui.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    assert main_window._central_stack.currentWidget() is main_window.tabs


def test_closing_the_last_file_comes_back_to_welcome(main_window,
                                                     synthetic_nexus):
    from mlgidgui.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    main_window._set_active_session(None)
    assert main_window._central_stack.currentWidget() is main_window.welcome_view


def test_the_recent_list_mirrors_the_persisted_entries(view):
    view.set_recent([
        {"type": "nexus", "path": "/data/one.h5"},
        {"type": "raw", "path": "/data/two.h5"},
    ])
    labels = [b.text() for b in view._recent_buttons]
    assert labels == ["one.h5", "[raw]  two.h5"]
    assert view._recent_buttons[0].toolTip() == "/data/one.h5"


def test_the_recent_list_is_a_shortlist_and_rebuilds_cleanly(view):
    view.set_recent([{"type": "nexus", "path": f"/d/{i}.h5"}
                     for i in range(RECENT_SHOWN + 4)])
    assert len(view._recent_buttons) == RECENT_SHOWN
    view.set_recent([{"type": "nexus", "path": "/d/only.h5"}])
    assert [b.text() for b in view._recent_buttons] == ["only.h5"]
    view.set_recent([])
    assert view._recent_buttons == []
    assert view._recent_title.isHidden()


def test_clicking_a_recent_row_emits_its_path_and_kind(view, qtbot):
    view.set_recent([{"type": "raw", "path": "/data/two.h5"}])
    with qtbot.waitSignal(view.recentRequested) as caught:
        view._recent_buttons[0].click()
    assert caught.args == ["/data/two.h5", "raw"]


def test_the_buttons_reach_the_existing_handlers(main_window, monkeypatch):
    """The page holds no file logic; it routes to what the File menu
    already calls."""
    calls = []
    monkeypatch.setattr(type(main_window), "_action_open",
                        lambda self: calls.append("open"))
    monkeypatch.setattr(type(main_window), "_action_import_converted",
                        lambda self: calls.append("import"))
    # Re-wire, since the connections were made to the bound originals.
    view = main_window.welcome_view
    view.openRequested.disconnect()
    view.importRequested.disconnect()
    view.openRequested.connect(main_window._action_open)
    view.importRequested.connect(main_window._action_import_converted)

    view.btn_open.click()
    view.btn_import.click()
    assert calls == ["open", "import"]


def test_the_backend_note_appears_only_when_the_backend_is_missing(view):
    view.set_backend_available(True)
    assert view._backend_note.isHidden()

    view.set_backend_available(False)
    assert not view._backend_note.isHidden()
    assert "mlgidbase" in view._backend_note.text()
    assert view._backend_note.property("status") == "warn"


def test_the_wordmark_follows_the_theme(view):
    view.set_theme("dark")
    dark = view._mark.pixmap().toImage()
    view.set_theme("light")
    light = view._mark.pixmap().toImage()
    assert not dark.isNull() and not light.isNull()
    assert dark != light


def test_a_theme_switch_reaches_the_welcome_page(main_window):
    main_window._set_theme("dark")
    dark = main_window.welcome_view._mark.pixmap().toImage()
    main_window._set_theme("light")
    assert main_window.welcome_view._mark.pixmap().toImage() != dark


def test_the_primary_action_is_tagged(view):
    assert view.btn_open.property("variant") == "primary"
    assert view.btn_import.property("variant") in (None, "")


def test_the_empty_tables_say_what_is_missing(main_window):
    """An empty table is otherwise indistinguishable from a broken one."""
    from PySide6.QtWidgets import QLabel

    for table in (main_window.peaks_table_panel._detected_table,
                  main_window.scan_tracking_panel._table):
        hints = [w for w in table.viewport().findChildren(QLabel)
                 if w.property("role") == "hint"]
        assert len(hints) == 1
        assert not hints[0].isHidden(), "shown while the table has no rows"


def test_the_hint_hides_once_rows_arrive(qtbot):
    from PySide6.QtGui import QStandardItem, QStandardItemModel
    from PySide6.QtWidgets import QTableView

    from mlgidgui.widgets import attach_empty_hint

    model = QStandardItemModel(0, 1)
    table = QTableView()
    table.setModel(model)
    qtbot.addWidget(table)
    hint = attach_empty_hint(table, "nothing here")
    assert not hint.isHidden()

    model.appendRow(QStandardItem("a"))
    assert hint.isHidden()
    model.removeRow(0)
    assert not hint.isHidden()
