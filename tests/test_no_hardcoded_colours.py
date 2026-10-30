"""No module may bake a colour literal into a stylesheet.

A hardcoded hex is a colour that was chosen for one theme, so it stays
wrong in the other: this is exactly how the app ended up painting a
``#19232d`` preview pane inside a light window and ``#444`` separators
across the status bar. Colours belong in ``theme_tokens``, applied either
through the skin (a ``[status=…]`` / ``[role=…]`` tag) or, where the
value is genuinely data, through an f-string that reads a token.

The scan is AST-based rather than textual so it sees only real
``setStyleSheet`` arguments — comments and docstrings that mention a hex
(there are several, explaining history) do not trip it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "mlgidgui"
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")

#: Modules allowed to name colours: the token table itself, the sheet it
#: generates, the pyqtgraph pair, and the overlay palette (whose colours
#: are data, not chrome — see viewer_styles).
COLOUR_OWNERS = {"theme_tokens.py", "skin.py", "theme.py", "viewer_styles.py"}

#: Data-driven colours: the value comes from a matched structure, not
#: from the theme, and the tests for those panels assert the literal
#: reaches the stylesheet. Listed by module with a reason.
DATA_DRIVEN = {
    "phase_views_window.py": "per-structure legend tint",
    "phase_views_dialogs.py": "per-structure export list tint",
    "parameter_panel.py": "matched-structure swatch fill",
    "pen_picker.py": "the swatch grid IS the colours",
    "controls_help.py": "rich-text renderer, migrates in a later stage",
}


def _stylesheet_literals(path: Path):
    """Every string literal handed to setStyleSheet in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setStyleSheet"):
            continue
        for arg in node.args:
            for piece in ast.walk(arg):
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    found.append((node.lineno, piece.value))
    return found


@pytest.mark.parametrize(
    "path",
    sorted(p for p in PACKAGE.glob("*.py")
           if p.name not in COLOUR_OWNERS and p.name not in DATA_DRIVEN),
    ids=lambda p: p.name,
)
def test_module_bakes_no_colour_into_a_stylesheet(path):
    offenders = [(line, text) for line, text in _stylesheet_literals(path)
                 if HEX.search(text)]
    assert not offenders, (
        f"{path.name} hardcodes a colour instead of using a token: {offenders}"
    )


def test_the_allowlists_stay_honest():
    """Both lists name real modules — a rename must not silently widen
    the exemption."""
    names = {p.name for p in PACKAGE.glob("*.py")}
    assert COLOUR_OWNERS <= names, COLOUR_OWNERS - names
    assert set(DATA_DRIVEN) <= names, set(DATA_DRIVEN) - names
