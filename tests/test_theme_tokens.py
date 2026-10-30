"""Semantic colour tokens — completeness, provenance and contrast.

The token tables are the single source of colour for the restyle, so the
failure modes worth pinning are: a token defined in one theme but not the
other (light mode then paints an undefined colour), a value drifting away
from the qdarkstyle palette it mirrors (chrome and panels stop matching),
a value that is unreadable on its own background, and the live-theme
pointer not following ``apply_*_theme``.

No QApplication needed: ``theme_tokens`` is deliberately Qt-free.
"""

from __future__ import annotations

import re

import pytest

from mlgidgui import theme_tokens
from mlgidgui.theme_tokens import QDARKSTYLE_MIRROR, THEMES, TOKENS

HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.fixture(autouse=True)
def _restore_active_theme():
    """The live-theme pointer is module state; leave it as we found it."""
    before = theme_tokens.active_theme()
    yield
    theme_tokens.set_active_theme(before)


def test_both_themes_define_the_same_tokens():
    assert set(TOKENS) == set(THEMES)
    dark, light = set(TOKENS["dark"]), set(TOKENS["light"])
    assert dark == light, f"only in dark: {dark - light}; only in light: {light - dark}"


@pytest.mark.parametrize("theme", THEMES)
def test_every_value_is_a_lowercase_six_digit_hex(theme):
    bad = {k: v for k, v in TOKENS[theme].items() if not HEX.match(v)}
    assert not bad, f"{theme}: {bad}"


@pytest.mark.parametrize("theme", THEMES)
def test_mirrored_tokens_still_match_qdarkstyle(theme):
    """Guards against a qdarkstyle upgrade shifting its palette: the skin
    would keep painting the old values and drift off the chrome."""
    if theme == "dark":
        from qdarkstyle.dark.palette import DarkPalette as palette
    else:
        from qdarkstyle.light.palette import LightPalette as palette

    for token, attr in QDARKSTYLE_MIRROR.items():
        assert TOKENS[theme][token] == getattr(palette, attr).lower(), (
            f"{theme}.{token} no longer mirrors {attr}"
        )


def test_plot_tokens_are_the_values_theme_publishes():
    """``theme.PG_*`` is derived from these; a mismatch would put the
    plots on a different ground than the docks."""
    from mlgidgui import theme

    assert theme.PG_DARK_BACKGROUND == theme_tokens.color("plot_bg", "dark")
    assert theme.PG_DARK_FOREGROUND == theme_tokens.color("plot_fg", "dark")
    assert theme.PG_LIGHT_BACKGROUND == theme_tokens.color("plot_bg", "light")
    assert theme.PG_LIGHT_FOREGROUND == theme_tokens.color("plot_fg", "light")
    assert theme.PG_COLORS["dark"] == (theme.PG_DARK_BACKGROUND, theme.PG_DARK_FOREGROUND)


def test_controls_help_badge_values_are_preserved():
    """controls_help ships these literals today and its tests assert
    them; the token table must not change them when it takes over."""
    assert theme_tokens.color("badge_bg", "dark") == "#2e4157"
    assert theme_tokens.color("badge_bg", "light") == "#dde4ea"


def test_unknown_theme_and_token_behaviour():
    assert theme_tokens.tokens("chartreuse") is TOKENS["dark"]
    with pytest.raises(KeyError):
        theme_tokens.color("no_such_token", "dark")


def test_active_theme_tracks_the_applied_theme(main_window):
    """`_apply` records the theme, so colour lookups with no explicit
    theme follow a live switch."""
    from PySide6.QtWidgets import QApplication

    from mlgidgui.theme import apply_dark_theme, apply_light_theme

    app = QApplication.instance()

    apply_light_theme(app)
    assert theme_tokens.active_theme() == "light"
    assert theme_tokens.color("surface") == theme_tokens.color("surface", "light")

    apply_dark_theme(app)
    assert theme_tokens.active_theme() == "dark"
    assert theme_tokens.color("surface") == theme_tokens.color("surface", "dark")

    assert theme_tokens.set_active_theme("chartreuse") == "dark"


# --- contrast ----------------------------------------------------------
def _luminance(hex_color: str) -> float:
    """WCAG relative luminance."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _ratio(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("token", ["text", "text_muted"])
def test_body_text_is_readable_on_its_surface(theme, token):
    assert _ratio(theme_tokens.color(token, theme),
                  theme_tokens.color("surface", theme)) >= 4.5


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(
    "token", ["status_ok", "status_warn", "status_error", "status_info",
              "danger", "accent"])
def test_state_colours_clear_the_ui_component_threshold(theme, token):
    """3:1 is the WCAG minimum for UI components and large text, which is
    what these are used for (status lines, button borders)."""
    assert _ratio(theme_tokens.color(token, theme),
                  theme_tokens.color("surface", theme)) >= 3.0


@pytest.mark.parametrize("theme", THEMES)
def test_label_on_a_filled_accent_is_readable(theme):
    assert _ratio(theme_tokens.color("on_accent", theme),
                  theme_tokens.color("accent", theme)) >= 3.0
