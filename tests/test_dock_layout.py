"""How wide the side columns open, and why it is measured.

The right-hand docks share one column, and the panels in them sit in
resizable scroll areas — so a column that is too narrow does not scroll,
it compresses and elides. The pinned 350 px squeezed the Pipeline form
(the "Config (yaml)" field it carried at the time came out as a stub),
which is what "most of the content is obstructed" looked like.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from mlgidlab.main_window import MainWindow
from mlgidlab.widgets import Card

pytestmark = pytest.mark.gui


def _cap(window) -> int:
    """The widest the column is allowed to open for this window.

    Mirrors the clamps in ``_preferred_right_dock_width``: a share of
    the window, an absolute cap, and a floor.
    """
    return min(
        MainWindow._RIGHT_DOCK_MAX_PX,
        max(window.width() // MainWindow._RIGHT_DOCK_WINDOW_SHARE,
            MainWindow._RIGHT_DOCK_MIN_PX),
    )


def _panel_content(dock):
    inner = dock.widget()
    scroll = inner.findChild(QScrollArea)
    if scroll is not None and scroll.widget() is not None:
        return scroll.widget()
    return inner


def test_the_right_column_is_measured_from_its_panels(main_window):
    wanted = main_window._preferred_right_dock_width()
    for dock in (main_window._display_dock, main_window._pipeline_dock,
                 main_window._sim_dock, main_window._logs_dock):
        assert wanted >= min(_panel_content(dock).sizeHint().width(),
                             _cap(main_window))


def test_it_counts_the_sections_that_start_closed(main_window):
    """Pipeline's Fitting section wants ~465 px and starts collapsed, so
    measuring only what is open reports far too little — and the column
    would be too narrow the moment the user opens one.

    Written against whichever right-hand panel has a collapsed section:
    without the analysis backend the Pipeline panel builds a stub with
    no sections at all, and the Conversion dock carries the point just
    as well."""
    widest_closed = 0
    for dock in (main_window._pipeline_dock, main_window._conversion_dock,
                 main_window._display_dock, main_window._sim_dock):
        content = _panel_content(dock)
        for card in content.findChildren(Card):
            if not card.is_expanded():
                widest_closed = max(widest_closed, card.open_width_hint())
    if widest_closed <= 0:
        pytest.skip("no collapsed sections in this build")
    assert main_window._preferred_right_dock_width() >= min(
        widest_closed, _cap(main_window))


def test_the_column_stays_inside_its_bounds(main_window):
    wanted = main_window._preferred_right_dock_width()
    assert MainWindow._RIGHT_DOCK_MIN_PX <= wanted <= MainWindow._RIGHT_DOCK_MAX_PX
    main_window.resize(900, 800)
    assert main_window._preferred_right_dock_width() >= MainWindow._RIGHT_DOCK_MIN_PX


def test_the_request_lands_once_the_window_has_room(qtbot, main_window):
    """A resizeDocks call made during construction is advisory — the
    window has no geometry yet and QMainWindow scales it down. It is
    re-applied on the first show, and this is the check that the number
    survives.

    Asserted against the room the docks actually got, not against the
    raw request. ``resizeDocks`` cannot hand out more than is left
    beside the centre widget's minimum, and when the total does not fit
    it scales **both** columns by one factor. The headroom here is
    large (1600 px window, ~1142 px left for 760 px of request) but it
    is not guaranteed: the offscreen platform's screen is fixed at
    800x800, the centre widget's minimum moves with font metrics, and
    which tests built windows earlier in the same process changes both.
    A CI runner once gave the two columns 614 px, where the unscaled
    260 came out as 210 and this test failed for a layout that was in
    fact correct.

    So the invariant is the *split*: each column gets its share of
    whatever room exists, in the proportion the request asked for. With
    full headroom ``scale`` is 1.0 and these are the exact original
    assertions; with less, they still catch a request made in the wrong
    proportions.

    The absolute net is the ``> 350`` floor at the end, and it is worth
    knowing which assertion does what. Verified by sabotage: with the
    body of ``_apply_default_dock_widths`` stubbed out, so the docks
    keep the construction-time widths (197 + 403, scaled down before
    the window had geometry), the two proportional checks still pass -
    197 of a 600 px allotment is about the share 260 of 760 asks for -
    and the floor is what fails, on ``205 > 350``. The proportions
    describe the shape; the floor is what says the re-apply happened.
    """
    main_window.resize(1600, 950)
    main_window.show()
    qtbot.waitExposed(main_window)
    qtbot.wait(100)
    main_window._apply_default_dock_widths()
    qtbot.wait(150)

    tree = main_window._tree_dock.width()
    # Measured on the front tab: one behind it keeps a stale width.
    right = main_window._pipeline_dock.width()
    wanted_right = main_window._preferred_right_dock_width()
    # What the two columns were actually allotted, versus what was
    # asked for. Qt never hands out more than the request, so the
    # ratio is <= 1 and 1.0 means "there was room".
    asked = MainWindow._TREE_DOCK_PX + wanted_right
    allotted = tree + right
    scale = min(1.0, allotted / asked)

    # Within a few pixels rather than to the pixel: QMainWindow trims
    # for separators.
    assert abs(tree - MainWindow._TREE_DOCK_PX * scale) <= 20
    assert abs(right - wanted_right * scale) <= 20
    assert right > 350, "wider than the pinned width it replaced"


def test_the_default_window_grows_with_the_screen():
    width, height = MainWindow._default_window_size()
    assert 1400 <= width <= 1600
    assert 900 <= height <= 950


# -- which right-hand tab is in front -------------------------------------


def _front_right_tab(window) -> str:
    """The current tab of the right-hand dock group, by its title."""
    from PySide6.QtWidgets import QApplication, QTabBar

    for _ in range(5):
        QApplication.processEvents()
    for bar in window.findChildren(QTabBar):
        if bar.parent() is not window or not bar.isVisible():
            continue
        titles = {bar.tabText(i) for i in range(bar.count())}
        if "Logs" in titles:
            return bar.tabText(bar.currentIndex())
    raise AssertionError("no visible right-hand dock tab bar")


def _open_nexus(window, path):
    from mlgidlab.session import NexusSession

    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_pipeline_is_in_front_at_startup(main_window):
    main_window.show()
    assert _front_right_tab(main_window) == "Pipeline"


def test_opening_a_file_keeps_pipeline_in_front(main_window, synthetic_nexus):
    main_window.show()
    _open_nexus(main_window, synthetic_nexus)
    assert _front_right_tab(main_window) == "Pipeline"


def test_switching_sessions_keeps_the_tab_the_user_chose(
    main_window, synthetic_nexus, synthetic_nexus_with_peaks
):
    main_window.show()
    first = _open_nexus(main_window, synthetic_nexus)
    main_window._display_dock.raise_()
    assert _front_right_tab(main_window) == "Display"
    assert main_window._right_front_dock is main_window._display_dock
    _open_nexus(main_window, synthetic_nexus_with_peaks)
    assert _front_right_tab(main_window) == "Display"
    main_window._logs_dock.raise_()
    main_window._set_active_session(first)
    assert _front_right_tab(main_window) == "Logs"


def test_switching_entries_keeps_the_tab_the_user_chose(
    main_window, synthetic_nexus
):
    import h5py

    with h5py.File(synthetic_nexus, "a") as f:
        f.copy("entry_0000", "entry_0001")
    main_window.show()
    _open_nexus(main_window, synthetic_nexus)
    combo = main_window.entry_combo
    assert combo.count() == 2
    main_window._sim_dock.raise_()
    combo.setCurrentIndex(1 - combo.currentIndex())
    assert _front_right_tab(main_window) == "Expected pattern"


def test_raw_shows_conversion_and_nexus_falls_back_to_pipeline(
    main_window, synthetic_raw, synthetic_nexus
):
    from mlgidlab.session import RawSession

    main_window.show()
    raw = RawSession.open([synthetic_raw])
    main_window._sessions.append(raw)
    main_window._set_active_session(raw)
    assert _front_right_tab(main_window) == "Conversion"
    _open_nexus(main_window, synthetic_nexus)
    assert _front_right_tab(main_window) == "Pipeline"


def test_a_structure_tab_round_trip_keeps_the_tab_the_user_chose(
    main_window, synthetic_nexus
):
    """Unfolding shows the docks one by one, each coming forward in turn."""
    main_window.show()
    main_window._confirm_discard_changes = lambda session=None: True
    _open_nexus(main_window, synthetic_nexus)
    main_window._logs_dock.raise_()
    try:
        main_window.tabs.setCurrentWidget(main_window.structure_panel)
        main_window.tabs.setCurrentWidget(main_window.viewer)
        assert _front_right_tab(main_window) == "Logs"
        assert main_window._right_front_dock is main_window._logs_dock
    finally:
        if main_window._h5_edit_handle is not None:
            main_window._h5_edit_handle.release()
