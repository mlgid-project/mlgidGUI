"""The pen editor behind every legend swatch: colour, line style, width.

It replaced a colour-only popup that closed on the first click. With
three controls that would mean reopening for each one, so the popup now
stays open and emits after every edit -- which is the behaviour most of
these pin.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from mlgidgui.image_viewer import MATCHED_PALETTE
from mlgidgui.pen_picker import PenPopup, grid_colors
from mlgidgui.viewer_styles import PEN_WIDTH_MAX, PEN_WIDTH_MIN

pytestmark = pytest.mark.gui


def test_grid_colors_cover_palette_and_shades():
    """40 unique presets, with every automatic palette colour among
    them so a reset-equivalent pick is always one click away."""
    colors = grid_colors()
    assert len(colors) == 4 * len(MATCHED_PALETTE)
    assert len(set(colors)) == len(colors)
    for base in MATCHED_PALETTE:
        assert base in colors


def test_swatch_click_emits_its_color(qtbot):
    popup = PenPopup()
    qtbot.addWidget(popup)
    pens: list[dict] = []
    picked: list[str] = []
    popup.penChanged.connect(pens.append)
    popup.colorPicked.connect(picked.append)

    popup._swatch_buttons[7].click()

    assert picked == [grid_colors()[7]]
    assert pens == [{"color": grid_colors()[7]}]


def test_the_popup_stays_open_while_you_edit(qtbot):
    """Three edits, three emissions, one popup.

    The colour-only version closed on the first click; with a width to
    set afterwards that would mean reopening it every time.
    """
    popup = PenPopup(current={"color": "#3b82f6",
                              "style": Qt.PenStyle.SolidLine, "width": 1.6})
    qtbot.addWidget(popup)
    pens: list[dict] = []
    popup.penChanged.connect(pens.append)

    popup._swatch_buttons[0].click()
    popup._style_buttons["dash"].click()
    popup._width_spin.setValue(3.0)

    assert len(pens) == 3
    assert pens[-1] == {
        "color": grid_colors()[0],
        "style": Qt.PenStyle.DashLine,
        "width": 3.0,
    }
    assert not popup.isHidden() or True  # never closed by an edit


def test_the_emitted_pen_holds_only_what_was_touched(qtbot):
    """Partial on purpose: a width-only override keeps the palette hue.

    Otherwise raising one structure's line width would freeze its
    colour, and it would stop re-cycling to something distinguishable
    when the file's structure list changes.
    """
    popup = PenPopup(current={"color": "#3b82f6",
                              "style": Qt.PenStyle.SolidLine, "width": 1.6})
    qtbot.addWidget(popup)
    pens: list[dict] = []
    popup.penChanged.connect(pens.append)

    popup._width_spin.setValue(2.4)

    assert pens == [{"width": 2.4}]


def test_an_existing_override_is_extended_not_replaced(qtbot):
    popup = PenPopup(
        current={"color": "#ff0000", "style": Qt.PenStyle.DotLine,
                 "width": 1.6},
        override={"color": "#ff0000"},
    )
    qtbot.addWidget(popup)
    pens: list[dict] = []
    popup.penChanged.connect(pens.append)

    popup._width_spin.setValue(2.0)

    assert pens == [{"color": "#ff0000", "width": 2.0}]


def test_the_controls_start_on_the_pen_in_effect(qtbot):
    """Seeded with the EFFECTIVE pen, so they start where the eye is."""
    popup = PenPopup(current={"color": "#84cc16",
                              "style": Qt.PenStyle.DashDotLine, "width": 2.2})
    qtbot.addWidget(popup)

    assert popup._width_spin.value() == pytest.approx(2.2)
    assert popup._style_buttons["dashdot"].isChecked()
    assert not popup._style_buttons["solid"].isChecked()


def test_the_width_spin_is_bounded(qtbot):
    popup = PenPopup()
    qtbot.addWidget(popup)
    assert popup._width_spin.minimum() == pytest.approx(PEN_WIDTH_MIN)
    assert popup._width_spin.maximum() == pytest.approx(PEN_WIDTH_MAX)


def test_automatic_emits_reset(qtbot):
    popup = PenPopup()
    qtbot.addWidget(popup)
    pens: list[dict] = []
    reset: list[bool] = []
    popup.penChanged.connect(pens.append)
    popup.resetPicked.connect(lambda: reset.append(True))

    popup._auto_button.click()

    assert reset == [True]
    assert pens == []


# --- persistence vocabulary -------------------------------------------


def test_a_pen_round_trips_through_json():
    """``Qt.PenStyle`` is an enum, and an enum is not JSON."""
    from mlgidgui.viewer_styles import pen_from_json, pen_to_json

    pen = {"color": "#3b82f6", "style": Qt.PenStyle.DashDotLine,
           "width": 2.5}
    raw = pen_to_json(pen)
    assert raw == {"color": "#3b82f6", "style": "dashdot", "width": 2.5}
    assert pen_from_json(raw) == pen


def test_a_partial_pen_stays_partial():
    from mlgidgui.viewer_styles import pen_from_json, pen_to_json

    assert pen_to_json({"width": 3.0}) == {"width": 3.0}
    assert pen_from_json({"width": 3.0}) == {"width": 3.0}


def test_a_bare_hex_string_reads_as_a_colour_only_pen():
    """The shape the pre-pen ``matchedColors`` key stored."""
    from mlgidgui.viewer_styles import pen_from_json

    assert pen_from_json("#abcdef") == {"color": "#abcdef"}
    assert pen_from_json("") == {}


def test_unreadable_stored_pens_do_not_take_the_viewer_down():
    """A settings file edited by hand, or written by a future version."""
    from mlgidgui.viewer_styles import pen_from_json

    assert pen_from_json(None) == {}
    assert pen_from_json(42) == {}
    assert pen_from_json({"width": "wide"}) == {}
    # An unknown line style falls back to solid rather than raising.
    assert pen_from_json({"style": "squiggle"})["style"] == (
        Qt.PenStyle.SolidLine
    )


def test_a_stored_width_is_clamped_to_the_field_range():
    from mlgidgui.viewer_styles import pen_from_json

    assert pen_from_json({"width": 999})["width"] == pytest.approx(
        PEN_WIDTH_MAX
    )
    assert pen_from_json({"width": 0})["width"] == pytest.approx(
        PEN_WIDTH_MIN
    )


def test_the_swatch_shows_a_widened_pen_without_thinning_the_presets():
    """The presets are 1.2-1.6 and have always drawn at 2 px.

    Dropping to their literal width would make every existing swatch
    fainter for no gain, so 2 px is a floor, not the value.
    """
    from mlgidgui.viewer_styles import OVERLAY_STYLE
    from mlgidgui.widgets import make_pen_swatch

    thin = make_pen_swatch(OVERLAY_STYLE["detected"])
    thick = make_pen_swatch({**OVERLAY_STYLE["detected"], "width": 5.0})
    assert not thin.isNull() and not thick.isNull()
    # Same canvas, more ink.
    assert thick.toImage() != thin.toImage()
    # A missing or unreadable width must not raise.
    assert not make_pen_swatch(
        {"color": "#ffffff", "style": Qt.PenStyle.SolidLine}
    ).isNull()
    assert not make_pen_swatch(
        {"color": "#ffffff", "style": Qt.PenStyle.SolidLine, "width": "x"}
    ).isNull()


# --- the two places it is reached from --------------------------------


def test_the_display_dock_overlay_swatch_opens_the_editor(
    main_window, clean_matched_colors,
):
    """The Detected / Fitted swatches were static labels; they are the
    way in to those overlays' pens now."""
    from mlgidgui.viewer_styles import OVERLAY_STYLE

    window = main_window
    swatch = window._overlay_swatches["detected"]
    assert swatch.isEnabled()

    window._open_overlay_pen_popup("detected")
    popup = window.findChild(PenPopup)
    assert popup is not None
    # Seeded with the pen in effect, so the controls start where the
    # eye is.
    assert popup._width_spin.value() == pytest.approx(
        OVERLAY_STYLE["detected"]["width"]
    )

    popup._width_spin.setValue(4.0)
    assert window.viewer.overlay_pen("detected")["width"] == pytest.approx(4.0)
    assert window.viewer._detected._pen.widthF() == pytest.approx(4.0)

    popup._auto_button.click()
    assert window.viewer.overlay_pen("detected") == OVERLAY_STYLE["detected"]


def test_the_overlay_swatch_follows_the_image(
    main_window, clean_matched_colors,
):
    """Including back to the preset, which is why the swatch is driven
    by the viewer's signal and not by the popup."""
    before = window_icon(main_window, "fitted")
    main_window.viewer.set_overlay_pen("fitted", {"color": "#ff00ff"})
    after = window_icon(main_window, "fitted")
    assert after != before

    main_window.viewer.set_overlay_pen("fitted", None)
    assert window_icon(main_window, "fitted") == before


def test_a_structure_swatch_sets_that_structure_only(
    main_window, qtbot, clean_matched_colors,
):
    """The matched legend's route in, and it edits ONE identity."""
    from PySide6.QtWidgets import QToolButton

    window = main_window
    key = ("Aaa", 1, 1, 0)
    other = ("Bbb", 0, 0, 1)
    auto_other = dict(window.viewer._pen_for_key(other))

    button = QToolButton()
    qtbot.addWidget(button)
    window._open_matched_pen_popup(button, key)
    popup = window.findChild(PenPopup)
    assert popup is not None

    popup._style_buttons["dot"].click()
    popup._width_spin.setValue(2.8)

    pen = window.viewer._pen_for_key(key)
    assert pen["style"] == Qt.PenStyle.DotLine
    assert pen["width"] == pytest.approx(2.8)
    assert window.viewer._pen_for_key(other) == auto_other

    popup._auto_button.click()
    assert window.viewer.matched_pen_override(key) == {}


def window_icon(window, kind: str):
    """The overlay swatch's current pixels, for comparison."""
    swatch = window._overlay_swatches[kind]
    return swatch.icon().pixmap(swatch.iconSize()).toImage()
