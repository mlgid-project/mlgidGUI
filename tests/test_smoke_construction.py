"""Smoke harness — workstream A, increment 1: construction + teardown.

The riskiest path in mlgidGUI is the ``MainWindow`` constructor itself:
it wires the silx tree, the image viewer, eight docks and six menus,
and that coupling is exactly what every feature has had to tiptoe
around. Zero fixture data is needed to exercise it, so it is the
highest-value first net: "does the app start, and does it shut down
cleanly without a session loaded".

Later A increments add synthetic-file open/frame/save tests once the
FrameSource HDF5 schema is mapped.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDockWidget

EXPECTED_MENUS = {"File", "Edit", "Tools", "View", "Settings", "Help"}

EXPECTED_DOCKS = {
    "_tree_dock",
    "_display_dock",
    "_pipeline_dock",
    "_conversion_dock",
    "_logs_dock",
    "_profile_dock",
    "_peaks_dock",
    "_scan_tracking_dock",
}


def test_main_window_constructs_clean(main_window):
    """Constructor builds the full shell with no session loaded."""
    # No data loaded yet: the session must be absent, not half-built.
    # ``session`` is the public property over ``_active_session``.
    assert main_window._active_session is None
    assert main_window.session is None

    # Core child widgets exist and were assigned (not left as None by a
    # swallowed exception in __init__).
    assert main_window.viewer is not None
    assert main_window.profile_viewer is not None
    assert main_window.peaks_table_panel is not None
    assert main_window.entry_combo is not None


def test_menu_bar_has_all_top_level_menus(main_window):
    """All six top-level menus are present (ampersands stripped)."""
    titles = {
        a.text().replace("&", "")
        for a in main_window.menuBar().actions()
        if a.text()
    }
    assert EXPECTED_MENUS.issubset(titles), (
        f"missing menus: {EXPECTED_MENUS - titles}"
    )


def test_all_docks_created_and_registered(main_window):
    """Each named dock attribute exists and is a real QDockWidget that
    the window actually parents."""
    registered = set(main_window.findChildren(QDockWidget))
    for attr in EXPECTED_DOCKS:
        dock = getattr(main_window, attr, None)
        assert isinstance(dock, QDockWidget), f"{attr} is not a QDockWidget"
        assert dock in registered, f"{attr} not registered on the window"


def test_no_ghost_dock_tab_bars(main_window):
    """Qt can leave the Profiles/Peaks group's OLD two-tab bar behind
    when the Scan-tracking dock is tabified into it — a stale QTabBar
    painted into the file-browser corner. After show, no visible
    MainWindow-child dock tab bar may have a tab set that is a strict
    subset of another's (``_hide_stale_dock_tab_bars`` sweeps them on
    show and on every dock re-tabify)."""
    from PySide6.QtWidgets import QApplication, QTabBar

    main_window.show()
    for _ in range(20):
        QApplication.processEvents()
    visible = [
        {tb.tabText(i) for i in range(tb.count())}
        for tb in main_window.findChildren(QTabBar)
        if tb.parent() is main_window and tb.isVisible()
    ]
    for labs in visible:
        assert not any(
            labs < other for other in visible if other is not labs
        ), f"ghost dock tab bar still visible: {sorted(labs)}"
    # The real bottom group (with Scan tracking) survived the sweep.
    assert {"Profiles", "Peaks", "Scan tracking"} in visible


def test_clean_close_does_not_raise(qtbot):
    """The closeEvent shutdown path (silx detach, worker quit) must run
    without raising or prompting when no session is loaded.

    Constructed here directly rather than via the ``main_window``
    fixture so the close is the assertion under test, not teardown.
    """
    from mlgidgui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.close() is True
