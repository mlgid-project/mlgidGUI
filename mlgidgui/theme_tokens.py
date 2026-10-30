"""Semantic colour tokens, one table per theme.

Stdlib only — no Qt, no qdarkstyle, no other mlgidgui import — so any
module can ask for a colour without dragging a heavy import chain along
(``theme`` itself imports pyqtgraph and qdarkstyle at module scope, which
is why the tables do not live there).

Why tokens at all: colours used to be written literally at ~27 call
sites, each hardcoding the dark theme, so light mode was wrong wherever
one of them appeared (a ``#19232d`` panel painted inside a light window,
``#444`` separators, ``#aaa`` status text). A widget now asks for
``color("text_muted")`` and gets the value that belongs to the live
theme.

Values mirror the qdarkstyle palettes rather than being computed from
them (see ``QDARKSTYLE_MIRROR``): the harmony of derivation without the
import, and ``tests/test_theme_tokens.py`` fails loudly if a qdarkstyle
upgrade shifts a palette out from under us.

The accent family is the app's default detector colormap, "magma" —
dark theme takes its orange, light theme its magenta (orange has too
little contrast on ``#fafafa``).
"""
from __future__ import annotations

from typing import Mapping

THEMES = ("dark", "light")

#: Token -> the qdarkstyle palette attribute it mirrors. Data only; the
#: comparison lives in the test so this module stays import-free.
QDARKSTYLE_MIRROR = {
    "surface": "COLOR_BACKGROUND_1",
    "surface_raised": "COLOR_BACKGROUND_2",
    "surface_sunken": "COLOR_BACKGROUND_3",
    "border": "COLOR_BACKGROUND_4",
    "divider": "COLOR_BACKGROUND_3",
    "text": "COLOR_TEXT_1",
    "text_muted": "COLOR_TEXT_3",
    "text_disabled": "COLOR_DISABLED",
    "selection_bg": "COLOR_ACCENT_2",
}

TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        # --- surfaces and lines (qdarkstyle DarkPalette) ---------------
        "surface": "#19232d",
        "surface_raised": "#293544",
        "surface_sunken": "#37414f",
        "border": "#455364",
        "divider": "#37414f",
        # --- text ------------------------------------------------------
        "text": "#dfe1e2",
        "text_muted": "#acb1b6",
        "text_disabled": "#788d9c",
        # --- accent: magma orange on the dark ground -------------------
        "accent": "#fc8961",
        "accent_hover": "#ffa07f",
        "accent_pressed": "#e2734b",
        # accent at ~18% over `surface`; the hover fill for outlined
        # buttons, which must not lighten the panel it sits on.
        "accent_soft": "#423536",
        "on_accent": "#19232d",
        # --- brand: the magma ramp, identical in both themes -----------
        "brand_plum": "#3b0f70",
        "brand_magenta": "#b73779",
        "brand_orange": "#fc8961",
        "brand_cream": "#fcfdbf",
        # --- semantic state -------------------------------------------
        "status_ok": "#4ade80",
        "status_warn": "#ffd166",
        "status_error": "#ff6b6b",
        "status_info": "#7fb3dd",
        "danger": "#e5534b",
        "danger_hover": "#f07e76",
        # --- item views ------------------------------------------------
        "table_header": "#293544",
        "table_grid": "#37414f",
        "table_alt_row": "#1e2937",
        "selection_bg": "#346792",
        # --- chrome ----------------------------------------------------
        "banner_bg": "#2d5a88",
        "banner_fg": "#f5f7fa",
        # Controls & shortcuts dialog. `badge_*` keep the values
        # controls_help.py already ships (its tests assert the literals).
        "band_bg": "#22303f",
        "band_fg": "#7fb3dd",
        "badge_bg": "#2e4157",
        "badge_fg": "#eaf2fa",
        # --- plots (the pyqtgraph pair mlgidgui.theme publishes) -------
        "plot_bg": "#19232d",
        "plot_fg": "#dfe1e2",
        # --- overlays that are currently white-only --------------------
        "overlay_selection": "#ffffff",
        # The dashed outline of the area pygidfit fits around a selected
        # detected peak; red on both grounds so it reads as "beyond the box".
        "overlay_fit_area": "#ff1744",
        "sim_selected": "#ffffff",
        "profile_curve": "#e8e8e8",
    },
    "light": {
        "surface": "#fafafa",
        "surface_raised": "#dfe1e2",
        "surface_sunken": "#d2d5d8",
        "border": "#c0c4c8",
        "divider": "#d2d5d8",
        "text": "#19232d",
        "text_muted": "#54687a",
        "text_disabled": "#9da9b5",
        # Magma magenta: white-on-magenta clears 5:1, white-on-orange
        # would not.
        "accent": "#b73779",
        "accent_hover": "#cf4a8d",
        "accent_pressed": "#9c2a66",
        "accent_soft": "#f5e3ee",
        "on_accent": "#ffffff",
        "brand_plum": "#3b0f70",
        "brand_magenta": "#b73779",
        "brand_orange": "#fc8961",
        "brand_cream": "#fcfdbf",
        "status_ok": "#15803d",
        "status_warn": "#a16207",
        "status_error": "#b91c1c",
        "status_info": "#1c5d99",
        "danger": "#b91c1c",
        "danger_hover": "#dc2626",
        "table_header": "#dfe1e2",
        "table_grid": "#d2d5d8",
        "table_alt_row": "#f0f1f2",
        "selection_bg": "#9fcbff",
        "banner_bg": "#daedff",
        "banner_fg": "#19232d",
        "band_bg": "#e8edf2",
        "band_fg": "#1c5d99",
        "badge_bg": "#dde4ea",
        "badge_fg": "#19232d",
        "plot_bg": "#fafafa",
        "plot_fg": "#000000",
        # White is invisible on the light plot ground, so the selection
        # highlight and the profile curve flip to near-black.
        "overlay_selection": "#101418",
        "overlay_fit_area": "#c62828",
        "sim_selected": "#101418",
        "profile_curve": "#22272b",
    },
}

_active = "dark"


def tokens(theme: str | None = None) -> Mapping[str, str]:
    """The token table for ``theme`` (the live theme when omitted).

    An unknown name resolves to dark, matching ``theme.pg_colors``.
    """
    if theme is None:
        theme = _active
    return TOKENS.get(theme, TOKENS["dark"])


def color(name: str, theme: str | None = None) -> str:
    """One token's value. Raises ``KeyError`` on an unknown token, on
    purpose: a typo should fail at the call site, not paint something
    silently wrong."""
    return tokens(theme)[name]


def set_active_theme(name: str) -> str:
    """Record the theme being applied; returns the resolved name.

    Called by ``theme._apply`` so drawing code that has no widget to ask
    (overlay pens, plot curves) can still resolve a theme-correct colour.
    """
    global _active
    _active = name if name in THEMES else "dark"
    return _active


def active_theme() -> str:
    return _active
