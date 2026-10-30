"""The viewer's control strip: segmented view pair, log toggle, swatches.

These controls changed shape (radios -> a segmented pair, a checkbox ->
a toggle button) while keeping their names, signals and persisted
settings. That is exactly the change a type-blind refactor breaks
quietly, so the tests here drive them the way the rest of the app does:
``setChecked`` on ``_log_check``, ``isChecked`` on ``_radio_cart``, and
the mode round-trip through ``set_mode_radios_visible``.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QToolButton

from mlgidgui import skin
from mlgidgui.image_viewer import GIWAXSImageViewer
from mlgidgui.theme import apply_dark_theme
from mlgidgui.viewer_styles import (
    COLORMAPS,
    DEFAULT_COLORMAP,
    MODE_CARTESIAN,
    MODE_POLAR,
    colormap_swatch,
    resolve_colormap,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def viewer(qtbot):
    apply_dark_theme(QApplication.instance())
    view = GIWAXSImageViewer()
    qtbot.addWidget(view)
    return view


def test_the_view_pair_is_segmented_and_exclusive(viewer):
    assert isinstance(viewer._radio_cart, QToolButton)
    assert viewer._radio_cart.property("segment") == "left"
    assert viewer._radio_polar.property("segment") == "right"
    assert viewer._radio_group.exclusive()

    assert viewer._radio_polar.isChecked()
    viewer._radio_cart.setChecked(True)
    assert not viewer._radio_polar.isChecked(), "one mode at a time"
    viewer._radio_polar.setChecked(True)
    assert not viewer._radio_cart.isChecked()


def test_the_pair_still_drives_the_render_mode(viewer):
    viewer._radio_cart.setChecked(True)
    assert viewer._mode == MODE_CARTESIAN
    viewer._radio_polar.setChecked(True)
    assert viewer._mode == MODE_POLAR


def test_hiding_the_mode_controls_still_works(viewer):
    """Raw sessions have no q-axes, so the host hides the pair. It walks
    the widgets by name, which the reshape had to preserve."""
    viewer.show()
    viewer.set_mode_radios_visible(False)
    assert not viewer._radio_cart.isVisible()
    assert not viewer._radio_polar.isVisible()
    viewer.set_mode_radios_visible(True)
    assert viewer._radio_cart.isVisible()


def test_the_log_control_is_a_toggle_but_keeps_its_api(viewer):
    """``setChecked`` is how the rest of the app and the persistence test
    drive it; a QToolButton keeps that and the ``toggled`` signal."""
    assert isinstance(viewer._log_check, QToolButton)
    assert viewer._log_check.isCheckable()
    assert viewer._log_check.property("variant") == "toggle"

    seen = []
    viewer._log_check.toggled.connect(seen.append)
    viewer._log_check.setChecked(True)
    assert seen == [True]
    assert viewer._log_check.isChecked()


@pytest.mark.parametrize("name", COLORMAPS)
def test_every_offered_colormap_resolves_and_previews(name):
    """A name the resolver cannot find would show as an empty row in the
    dropdown and silently keep the previous ramp on the image."""
    assert resolve_colormap(name) is not None
    swatch = colormap_swatch(name)
    assert not swatch.isNull()
    # Square, because Qt sizes item icons from a single length: a wide
    # strip is either squeezed into the style's 16 px box or forces the
    # popup rows to 58 px.
    assert swatch.width() == swatch.height()

    image = swatch.toImage()
    seen = {image.pixelColor(x, image.height() // 2).name()
            for x in range(image.width())}
    assert len(seen) > 4, "a gradient, not a flat block"


def test_an_unknown_colormap_gives_a_null_swatch_not_a_crash():
    assert colormap_swatch("no-such-map").isNull()


def test_the_dropdown_carries_the_ramps(viewer):
    combo = viewer._cmap_combo
    assert combo.count() == len(COLORMAPS)
    assert combo.currentText() == DEFAULT_COLORMAP
    for index in range(combo.count()):
        assert not combo.itemIcon(index).isNull(), combo.itemText(index)


def test_the_chip_survives_a_theme_flip(qtbot, main_window):
    """Regression: Qt clears an explicitly set ``iconSize`` when a widget
    is polished under an application stylesheet, and it clears it *after*
    the style change is delivered — so the chip showed on first launch
    and then disappeared from the closed box on the first View -> Theme.
    A 0x0 icon is not drawn at all."""
    combo = main_window.viewer._cmap_combo
    main_window.show()
    qtbot.waitUntil(lambda: not combo.iconSize().isEmpty(), timeout=2000)

    for theme in ("light", "dark", "light"):
        main_window._set_theme(theme)
        qtbot.waitUntil(lambda: not combo.iconSize().isEmpty(), timeout=2000)
        assert combo.iconSize().width() > 0, f"chip lost on {theme}"
    assert not combo.itemIcon(combo.currentIndex()).isNull()


def test_the_skin_defines_the_new_control_rules():
    for theme in ("dark", "light"):
        qss = skin.build_qss(theme)
        assert 'QToolButton[segment="left"]' in qss
        assert 'QToolButton[segment="right"]' in qss
        assert 'QToolButton[variant="toggle"]' in qss
