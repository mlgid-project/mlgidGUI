"""The SVG icon loader: packaging, recolouring, and theme refresh.

Three failure modes are worth pinning. The assets can go missing from a
wheel (``package-data`` is easy to forget, and the app then shows blank
buttons with no error). ``currentColor`` can fail to resolve, which looks
fine in dark mode and invisible in light. And a ``QIcon`` already handed
to a widget does not follow a theme flip on its own, because it is a
value type — that is what ``bind``/``retheme`` exist for.
"""

from __future__ import annotations

from importlib import resources

import pytest
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolButton

from mlgidgui import icons, theme_tokens

pytestmark = pytest.mark.gui

# Glyphs the application actually asks for by name. If one is renamed or
# dropped from the package, these buttons silently lose their icon.
IN_USE = ["play", "pause", "prev", "next", "refresh", "close",
          "chevron-down", "chevron-right"]


@pytest.fixture(autouse=True)
def _isolate_icon_state():
    icons.clear_cache()
    yield
    icons.clear_cache()


def test_the_assets_are_reachable_as_package_data():
    """Catches a wheel built without the package-data stanza."""
    directory = resources.files("mlgidgui").joinpath("assets/icons")
    assert directory.is_dir()
    shipped = icons.available()
    assert len(shipped) >= 40
    for name in IN_USE:
        assert name in shipped, f"{name}.svg is not shipped"


@pytest.mark.parametrize("name", IN_USE)
def test_every_glyph_in_use_renders(name, qapp):
    ic = icons.icon(name)
    assert not ic.isNull()
    pixmap = ic.pixmap(24, 24)
    assert not pixmap.isNull()
    assert pixmap.toImage().constBits() is not None


def test_currentColor_is_actually_substituted(qapp):
    """The same glyph must differ between themes; if the substitution
    failed, Qt would render both identically (and invisibly on light)."""
    dark = icons.icon("play", theme="dark").pixmap(32, 32).toImage()
    light = icons.icon("play", theme="light").pixmap(32, 32).toImage()
    assert dark != light

    # And the ink is the theme's text colour, sampled from the most
    # opaque pixel of the glyph.
    def ink(image):
        best, colour = 0, None
        for y in range(image.height()):
            for x in range(image.width()):
                px = image.pixelColor(x, y)
                if px.alpha() > best:
                    best, colour = px.alpha(), px
        return colour.name().lower()

    assert ink(dark) == theme_tokens.color("text", "dark")
    assert ink(light) == theme_tokens.color("text", "light")


def test_an_unknown_glyph_is_a_null_icon_not_a_crash(qapp):
    """A missing asset costs an icon, never a traceback: a QToolButton
    with a null icon renders text-only, which is what these buttons did
    before they had icons."""
    assert icons.icon("no-such-glyph").isNull()


def test_icons_are_cached_per_colour(qapp):
    first = icons.icon("play")
    assert icons.icon("play") is first
    assert icons.icon("play", theme="light") is not first


def test_bind_then_retheme_repaints_the_target(qapp):
    button = QToolButton()
    action = QAction("Next")
    icons.bind(button, "next")
    icons.bind(action, "next")
    assert not button.icon().isNull()

    before = button.icon().pixmap(24, 24).toImage()
    assert icons.retheme("light") >= 2
    after = button.icon().pixmap(24, 24).toImage()
    assert before != after
    assert not action.icon().isNull()


def test_unbind_stops_a_widget_being_repainted(qapp):
    """After unbind, a theme switch must leave that widget alone.

    Asserted per widget rather than by the registry's size: the registry
    is global, so any other live window contributes to the count and the
    test would only pass in isolation.
    """
    button = QToolButton()
    icons.bind(button, "play")
    icons.retheme("dark")
    before = button.icon().pixmap(24, 24).toImage()

    icons.unbind(button)
    icons.retheme("light")
    after = button.icon().pixmap(24, 24).toImage()
    assert before == after


def test_the_registry_holds_widgets_weakly(qapp):
    """A closed window must not be pinned in memory by its icons."""
    import weakref

    button = QToolButton()
    icons.bind(button, "play")
    ref = weakref.ref(button)
    del button
    assert ref() is None


def test_a_theme_switch_repaints_the_transport_icons(main_window):
    """End-to-end: the buttons built in main_window_build follow
    View -> Theme."""
    button = main_window.next_frame_button
    main_window._set_theme("dark")
    dark = button.icon().pixmap(24, 24).toImage()
    main_window._set_theme("light")
    light = button.icon().pixmap(24, 24).toImage()
    assert dark != light


def test_the_play_button_glyph_follows_playback_state(main_window):
    """The transport button swaps between two glyphs, so it asks for the
    icon by name at swap time rather than caching one at build time."""
    play = icons.icon("play").pixmap(24, 24).toImage()
    pause = icons.icon("pause").pixmap(24, 24).toImage()
    assert play != pause
    assert isinstance(main_window.play_button.icon(), QIcon)
    assert not main_window.play_button.icon().isNull()


def test_menu_actions_carry_their_glyphs(main_window):
    """The View menu should read as a list of places, and File/Edit as a
    list of operations — which only works if the mapping is applied."""
    for attr in ("action_open", "action_save", "action_undo",
                 "action_find_peak", "action_export_figure",
                 "action_fullscreen", "action_about"):
        action = getattr(main_window, attr, None)
        assert action is not None, attr
        assert not action.icon().isNull(), f"{attr} has no icon"

    assert not main_window._tree_dock.toggleViewAction().icon().isNull()


def test_every_mapped_glyph_actually_ships(main_window):
    """A typo in the mapping would silently leave an entry text-only."""
    shipped = set(icons.available())
    mapped = set(main_window._MENU_ICONS.values()) | set(
        main_window._DOCK_ICONS.values())
    assert mapped <= shipped, f"missing glyphs: {mapped - shipped}"


def _dock_tab_bars(window):
    """Every QTabBar carrying a dock title, keyed by tab text.

    Qt owns these bars (they are built by QMainWindow when docks are
    tabified), so there is no handle to ask for directly.
    """
    from PySide6.QtWidgets import QTabBar

    titles = {getattr(window, attr).windowTitle(): attr
              for attr in window._DOCK_ICONS
              if getattr(window, attr, None) is not None}
    found = {}
    for bar in window.findChildren(QTabBar):
        for index in range(bar.count()):
            if bar.tabText(index) in titles:
                found[bar.tabText(index)] = (bar, index)
    return found


def test_the_tabified_docks_carry_their_glyph(main_window):
    """Both tab rows are text-only without this, which is what makes the
    docks easy to miss in the first place."""
    tabs = _dock_tab_bars(main_window)
    assert len(tabs) >= 6, f"expected both tab rows, found {sorted(tabs)}"
    for title, (bar, index) in tabs.items():
        assert not bar.tabIcon(index).isNull(), f"{title} tab has no icon"


def test_dock_tab_icons_follow_a_theme_switch(main_window):
    """Qt copies the dock's windowIcon onto its tab only when the tab is
    created, so a flip has to re-push them explicitly."""
    bar, index = _dock_tab_bars(main_window)["Display"]
    main_window._set_theme("dark")
    dark = bar.tabIcon(index).pixmap(16, 16).toImage()
    main_window._set_theme("light")
    light = bar.tabIcon(index).pixmap(16, 16).toImage()
    assert dark != light


def test_a_mode_switch_keeps_the_tab_glyphs(main_window):
    """``_apply_session_mode`` re-tabifies the right-hand chain, and a
    rebuilt tab comes up bare."""
    main_window._apply_session_mode(None)
    for title, (bar, index) in _dock_tab_bars(main_window).items():
        assert not bar.tabIcon(index).isNull(), f"{title} lost its icon"


def test_menu_icons_follow_a_theme_switch(main_window):
    main_window._set_theme("dark")
    dark = main_window.action_open.icon().pixmap(20, 20).toImage()
    main_window._set_theme("light")
    light = main_window.action_open.icon().pixmap(20, 20).toImage()
    assert dark != light
