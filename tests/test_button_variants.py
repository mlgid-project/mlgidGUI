"""Button hierarchy — the primary/destructive tags and what they resolve to.

Two failure modes are worth pinning. First, a tag silently disappearing:
``set_variant`` is one call per construction site, and a refactor that
rebuilds a button loses the emphasis without any test noticing. Second,
the tag being present but inert — attribute selectors are only
re-evaluated when a widget is re-polished, so a variant applied after the
first polish (or surviving a theme flip) has to be checked live, not just
read back off the property.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from mlgidgui import skin, theme_tokens
from mlgidgui.theme import apply_dark_theme, apply_light_theme
from mlgidgui.widgets import DANGER, PRIMARY, set_variant

pytestmark = pytest.mark.gui


def _variant(widget) -> str:
    return widget.property("variant")


def _text_colour(widget) -> str:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.ensurePolished()
    return widget.palette().buttonText().color().name().lower()


def test_set_variant_tags_and_returns_the_widget(qtbot):
    button = QPushButton("Run")
    qtbot.addWidget(button)
    assert set_variant(button, PRIMARY) is button
    assert _variant(button) == "primary"
    set_variant(button, "")
    assert _variant(button) == ""


@pytest.mark.parametrize("theme,apply", [("dark", apply_dark_theme),
                                         ("light", apply_light_theme)])
def test_a_primary_button_paints_the_accent_in_both_themes(qtbot, theme, apply):
    app = QApplication.instance()
    apply(app)
    primary = set_variant(QPushButton("Run"), PRIMARY)
    danger = set_variant(QPushButton("Delete"), DANGER)
    plain = QPushButton("Browse…")
    for w in (primary, danger, plain):
        qtbot.addWidget(w)

    assert _text_colour(primary) == theme_tokens.color("accent", theme)
    assert _text_colour(danger) == theme_tokens.color("danger", theme)
    assert _text_colour(plain) not in {theme_tokens.color("accent", theme),
                                       theme_tokens.color("danger", theme)}


def test_the_tag_survives_a_live_theme_flip(qtbot, main_window):
    """`_set_theme` re-polishes every widget, which is what re-evaluates
    the attribute selector against the freshly installed sheet."""
    app = QApplication.instance()
    button = set_variant(QPushButton("Run"), PRIMARY)
    qtbot.addWidget(button)

    main_window._set_theme("light")
    assert _text_colour(button) == theme_tokens.color("accent", "light")
    main_window._set_theme("dark")
    assert _text_colour(button) == theme_tokens.color("accent", "dark")


@pytest.fixture
def pipeline_panel(monkeypatch):
    """A PipelinePanel with its full widget set.

    Backend-less (CI), the panel builds a stub without the run buttons.
    ``pipeline_panel`` binds ``is_mlgidbase_available`` by name at import,
    so the patch has to target the panel module — patching
    ``mlgidgui.pipeline`` leaves the already-bound name untouched.
    """
    from mlgidgui import pipeline_panel as panel_mod

    monkeypatch.setattr(panel_mod, "is_mlgidbase_available", lambda: True)
    return panel_mod.PipelinePanel()


def test_pipeline_and_conversion_actions_are_tagged(main_window, pipeline_panel):
    from mlgidgui.conversion_panel import ConversionPanel

    for name in ("btn_run_all", "btn_detect", "btn_fit", "btn_match"):
        assert _variant(getattr(pipeline_panel, name)) == "primary", name

    conversion = ConversionPanel()
    assert _variant(conversion.btn_convert) == "primary"


def test_the_destructive_actions_are_tagged(main_window, pipeline_panel):
    """Reserved for operations that lose data: deleting a peak. The six
    "Clear" buttons stay neutral on purpose — they clear a path field."""
    assert _variant(main_window.parameter_panel.btn_delete_peak) == "danger"

    for name in ("_cif_browse_btn", "_pickle_clear_btn"):
        assert _variant(getattr(pipeline_panel, name)) in (None, "")


def test_tracking_and_export_actions_are_tagged(main_window):
    assert _variant(main_window.scan_tracking_panel.btn_track) == "primary"
    assert _variant(main_window._sim_add_btn) == "primary"


def test_the_skin_defines_a_rule_for_every_variant_in_use():
    """A tag with no matching rule is a silent no-op."""
    for theme in ("dark", "light"):
        qss = skin.build_qss(theme)
        assert f'QPushButton[variant="{PRIMARY}"]' in qss
        assert f'QPushButton[variant="{DANGER}"]' in qss
