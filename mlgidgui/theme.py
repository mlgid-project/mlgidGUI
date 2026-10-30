"""Application-wide theming.

Both themes apply a full qdarkstyle stylesheet — ``DarkPalette`` for
dark, ``LightPalette`` for light — so the look is **independent of the
host's Qt palette**. The previous light theme set an empty stylesheet
and fell back to the OS palette, which on a dark-desktop machine left
"light mode" looking dark. Each also pushes matching pyqtgraph
background / foreground defaults so plots blend with the chrome.

Switched at runtime by ``MainWindow._set_theme`` (which also forces a
live re-polish and refreshes existing plot colours — config options
below only affect *newly* created pg items).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette
from PySide6.QtWidgets import QApplication

from mlgidgui import skin, theme_tokens

# pyqtgraph background / foreground per theme. Dark matches qdarkstyle's
# panel colour (#19232d); light matches its panel colour (#fafafa) so the
# plots sit flush with the surrounding docks in either theme. The values
# live in theme_tokens (as ``plot_bg`` / ``plot_fg``); these names stay
# because callers and tests import them.
PG_DARK_BACKGROUND = theme_tokens.color("plot_bg", "dark")
PG_DARK_FOREGROUND = theme_tokens.color("plot_fg", "dark")
PG_LIGHT_BACKGROUND = theme_tokens.color("plot_bg", "light")
PG_LIGHT_FOREGROUND = theme_tokens.color("plot_fg", "light")

# Look-up used by the runtime switcher to recolour already-built plots.
PG_COLORS = {
    "dark": (PG_DARK_BACKGROUND, PG_DARK_FOREGROUND),
    "light": (PG_LIGHT_BACKGROUND, PG_LIGHT_FOREGROUND),
}


def pg_colors(theme: str) -> tuple[str, str]:
    """``(background, foreground)`` for ``"dark"`` / ``"light"``."""
    return PG_COLORS.get(theme, PG_COLORS["dark"])


def _apply(app: QApplication, *, name: str, palette,
           background: str, foreground: str) -> None:
    # Record the live theme first: drawing code with no widget to ask
    # (overlay pens, plot curves) resolves its colours through
    # theme_tokens.active_theme().
    theme_tokens.set_active_theme(name)
    pg.setConfigOption("background", background)
    pg.setConfigOption("foreground", foreground)
    pg.setConfigOption("antialias", True)
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(qt_api="pyside6", palette=palette)
        + _OVERRIDES
        # Appended last so its selectors win equal-specificity ties with
        # qdarkstyle's. Every rule is opt-in per widget — see skin.py.
        + skin.build_qss(name)
    )


def apply_dark_theme(app: QApplication) -> None:
    _apply(
        app,
        name="dark",
        palette=DarkPalette,
        background=PG_DARK_BACKGROUND,
        foreground=PG_DARK_FOREGROUND,
    )


def apply_light_theme(app: QApplication) -> None:
    """Apply qdarkstyle's **LightPalette** stylesheet + light pyqtgraph
    defaults. A real light theme, not a strip-to-OS-default fallback, so
    it reads as light on every desktop."""
    _apply(
        app,
        name="light",
        palette=LightPalette,
        background=PG_LIGHT_BACKGROUND,
        foreground=PG_LIGHT_FOREGROUND,
    )


# Qt's default QComboBox reserves an icon column (PM_SmallIconSize, ~16 px)
# on every dropdown item. Combos in this app don't use icons, so the column
# shows up as a visible empty box before each item's text. Setting
# qproperty-iconSize collapses that column for every QComboBox application-
# wide. The other rules tighten item padding and zero out the (already
# unused) check indicator that qdarkstyle reserves space for.
_OVERRIDES = """
QComboBox {
    qproperty-iconSize: 0px 0px;
}
QComboBox QAbstractItemView {
    padding: 0px;
}
QComboBox QAbstractItemView::item {
    padding-left: 6px;
    padding-right: 6px;
    min-height: 20px;
}
QComboBox QAbstractItemView::indicator {
    width: 0px;
    height: 0px;
    image: none;
}
"""
