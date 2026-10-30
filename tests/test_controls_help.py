"""The reworked Help > "Controls & shortcuts" (F1) reference:
data-driven inventory completeness (regressions for the controls the
old QMessageBox omitted and the two entries it described wrongly),
theme-aware rendering, live filtering, and the modeless dialog wiring.

HTML assertions run against ``build_controls_html`` directly — Qt
rewrites markup in ``QTextBrowser.toHtml()`` — and note that badge
text renders with ``&nbsp;`` between words, so multi-word pins either
target descriptions or use the escaped badge form.
"""
from __future__ import annotations

import pytest

from mlgidgui.controls_help import (
    CONTROL_SECTIONS,
    ControlsDialog,
    build_controls_html,
)

pytestmark = pytest.mark.gui


def _row(badge_start: str):
    for _title, rows in CONTROL_SECTIONS:
        for badges, kind, description in rows:
            if badges[0].startswith(badge_start):
                return badges, kind, description
    raise AssertionError(f"no row with badge starting {badge_start!r}")


def test_inventory_lists_previously_missing_controls():
    """Everything the old QMessageBox omitted is present now."""
    html = build_controls_html("dark")
    for pin in (
        "Ctrl+C", "Ctrl+V", "Ctrl+Shift+V", "Ctrl+A", "Ctrl+click",
        "F11", "F5", "Ctrl+O", "Ctrl+S", "Ctrl+Q",
        "Drag&nbsp;&amp;&nbsp;drop",       # badge form of "Drag & drop"
        "trajectory",                       # phase-views point click
        "tracking results",                 # Delete on the tracking table
        "swatch",                           # legend colour picking
        "profile",                          # profile-region resize
    ):
        assert pin in html, pin


def test_esc_row_corrected():
    """Esc deletes the selected manual peak; the old "dismiss an
    in-progress draw" description was wrong."""
    _badges, _kind, description = _row("Esc")
    assert "manual" in description
    assert "dismiss" not in description.lower()
    assert "in-progress" not in description.lower()


def test_roi_resize_row_corrected():
    """ROI dragging resizes manual/detected only; the old text claimed
    fitted was resizable."""
    _badges, _kind, description = _row("Drag ROI")
    assert "manual or detected" in description
    assert "not editable" in description


def test_html_differs_between_themes():
    dark = build_controls_html("dark")
    light = build_controls_html("light")
    assert dark != light
    assert "#2e4157" in dark and "#2e4157" not in light
    assert "#dde4ea" in light and "#dde4ea" not in dark
    # Unknown theme falls back to dark.
    assert build_controls_html("nonsense") == dark


def test_filter_narrows_and_omits_empty_sections():
    filtered = build_controls_html("dark", "ctrl+shift+v")
    assert "Ctrl+Shift+V" in filtered
    assert "F11" not in filtered
    assert "TRACKING VIEWS" not in filtered
    # Case-insensitive; empty query returns everything.
    assert "Ctrl+A" in build_controls_html("dark", "CTRL+A")
    assert build_controls_html("dark", "") == build_controls_html("dark")


def test_filter_no_match_message():
    html = build_controls_html("dark", "zzzz")
    assert "No controls match" in html and "zzzz" in html


def test_dialog_filter_widget_narrows_and_clears(qtbot):
    dlg = ControlsDialog(theme="dark")
    qtbot.addWidget(dlg)
    full = dlg._browser.toPlainText()
    assert "F11" in full and "Ctrl+Shift+V" in full
    qtbot.keyClicks(dlg._filter, "paste")
    narrowed = dlg._browser.toPlainText()
    assert "Ctrl+Shift+V" in narrowed and "F11" not in narrowed
    dlg._filter.clear()
    assert "F11" in dlg._browser.toPlainText()


def test_apply_theme_colors_maps_pg_hexes(qtbot):
    """The duck-typed hook translates the pyqtgraph background hex the
    host passes into the right rendered theme (unknown -> dark)."""
    dlg = ControlsDialog(theme="dark")
    qtbot.addWidget(dlg)
    dlg.apply_theme_colors("#fafafa", "#000000")
    assert dlg._theme == "light"
    dlg.apply_theme_colors("#19232d", "#dfe1e2")
    assert dlg._theme == "dark"
    dlg.apply_theme_colors("#123456", "#654321")
    assert dlg._theme == "dark"


def test_show_controls_modeless_and_reused(main_window):
    main_window._show_controls()
    dlg = main_window._controls_dialog
    assert dlg is not None and dlg.isVisible() and not dlg.isModal()
    main_window._show_controls()
    assert main_window._controls_dialog is dlg


def test_theme_flip_rerenders_open_dialog(main_window):
    main_window._show_controls()
    dlg = main_window._controls_dialog
    main_window._set_theme("light")
    assert dlg._theme == "light"
    assert "#dde4ea" in build_controls_html(dlg._theme)
    main_window._set_theme("dark")
    assert dlg._theme == "dark"


def test_f1_action_triggers_dialog(main_window):
    from PySide6.QtGui import QKeySequence

    assert main_window.action_controls.shortcut() == QKeySequence("F1")
    main_window.action_controls.trigger()
    assert main_window._controls_dialog.isVisible()
