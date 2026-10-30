"""Theme application — light is genuinely light, and a runtime switch
takes effect live (not only after restart).

Regression for two bugs: (1) the old light theme set an *empty*
stylesheet and fell back to the OS palette, so on a dark desktop "light
mode" stayed dark — both themes now apply a full qdarkstyle palette
(Light / Dark), independent of the host; (2) `_set_theme` now forces a
re-polish and recolours the already-built pyqtgraph plots, so the chrome
and plot backgrounds change immediately.

Mirrors `test_theme_persistence.py` (the `main_window` fixture provides a
QApplication; conftest redirects XDG_CONFIG_HOME).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from mlgidgui.theme import apply_dark_theme, apply_light_theme

pytestmark = pytest.mark.gui


def test_light_and_dark_are_distinct_real_stylesheets(main_window):
    """Both themes are full stylesheets (not the strip-to-OS empty
    fallback), and they differ: light carries the light background,
    dark the dark one."""
    app = QApplication.instance()

    apply_light_theme(app)
    light = app.styleSheet().lower()
    apply_dark_theme(app)
    dark = app.styleSheet().lower()

    assert light != dark
    assert len(light) > 1000 and len(dark) > 1000   # not an empty fallback
    assert "fafafa" in light                        # genuine light palette
    assert "19232d" in dark                         # dark palette


def test_set_theme_switches_chrome_and_plots_live(main_window):
    """A runtime `_set_theme` updates the app stylesheet *and* recolours
    the live viewer / profile plot backgrounds without a restart."""
    app = QApplication.instance()
    app.setOrganizationName("mlgidGUI")
    app.setApplicationName("mlgidGUI")
    v = main_window.viewer
    pv = main_window.profile_viewer

    def gv_bg():
        return v._view.ui.graphicsView.backgroundBrush().color().name().lower()

    def radial_bg():
        return pv._radial_plot.backgroundBrush().color().name().lower()

    def hist_bg():
        # The contrast/LUT histogram is its own GraphicsView.
        return v._view.getHistogramWidget().backgroundBrush().color().name().lower()

    main_window._set_theme("light")
    assert "fafafa" in app.styleSheet().lower()
    assert gv_bg() == "#fafafa"
    assert radial_bg() == "#fafafa"
    assert hist_bg() == "#fafafa"

    main_window._set_theme("dark")
    assert "19232d" in app.styleSheet().lower()
    assert gv_bg() == "#19232d"
    assert radial_bg() == "#19232d"
    assert hist_bg() == "#19232d"
