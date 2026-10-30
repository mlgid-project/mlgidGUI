"""The skin layer — scoping, provenance and application.

The load-bearing test here is ``test_every_rule_is_scoped_to_a_widget_we_tag``.
Qt matches QSS descendant selectors by walking ``parentWidget()`` and does
not stop at window boundaries, so a rule like ``ConversionPanel QPushButton``
reaches the pyFAI task widgets inside ``CalibrationDialog(self, …)``, and
anything rooted at the file-browser dock reaches silx's tree. A selector
that names a Qt class without an objectName or property qualifier is
therefore a bug, not a style choice — this test is what stops one from
being added later.
"""

from __future__ import annotations

import re

import pytest

from mlgidgui import skin, theme_tokens
from mlgidgui.theme_tokens import THEMES

# Ancestor-scoped rules that are deliberately allowed: the root is a
# widget this application tags itself AND never parents a dialog, so the
# walk cannot escape into third-party widgets. Keep this list tiny.
ALLOWED_DESCENDANT_RULES = {
    f'{skin.TABLE_SELECTOR} QHeaderView::section',
}

QT_CLASS = re.compile(r"^Q[A-Z]\w*")


def _rules(qss: str) -> list[str]:
    """Every selector in the sheet, comments stripped."""
    body = re.sub(r"/\*.*?\*/", "", qss, flags=re.S)
    selectors = []
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", body):
        for sel in block.group(1).split(","):
            sel = sel.strip()
            if sel:
                selectors.append(sel)
    return selectors


def _subject(selector: str) -> str:
    """The simple selector the rule actually paints (the last one)."""
    return re.split(r"[\s>]+", selector.strip())[-1]


@pytest.mark.parametrize("theme", THEMES)
def test_every_rule_is_scoped_to_a_widget_we_tag(theme):
    unscoped = []
    for selector in _rules(skin.build_qss(theme)):
        if selector in ALLOWED_DESCENDANT_RULES:
            continue
        subject = _subject(selector)
        if not QT_CLASS.match(subject):
            continue                      # #ObjectName-only selector: fine
        if "#" in subject or "[" in subject:
            continue                      # qualified by objectName/property
        unscoped.append(selector)
    assert not unscoped, (
        "these selectors would also style silx / pyFAI / pyqtgraph widgets: "
        f"{unscoped}"
    )


@pytest.mark.parametrize("theme", THEMES)
def test_every_colour_comes_from_the_token_table(theme):
    """No literal may sneak back into the sheet: a colour that is not a
    token is a colour that will be wrong in the other theme."""
    allowed = {v.lower() for v in theme_tokens.tokens(theme).values()}
    used = {m.lower() for m in re.findall(r"#[0-9a-fA-F]{6}\b",
                                          skin.build_qss(theme))}
    assert used <= allowed, f"non-token colours in {theme} skin: {used - allowed}"


def test_the_two_themes_differ_and_are_labelled():
    dark, light = skin.build_qss("dark"), skin.build_qss("light")
    assert dark != light
    assert dark.startswith(skin.marker("dark"))
    assert light.startswith(skin.marker("light"))
    assert skin.marker("light") not in dark


@pytest.mark.parametrize("theme", THEMES)
def test_the_tagged_widget_roles_are_all_present(theme):
    qss = skin.build_qss(theme)
    for selector in (
        f'QPushButton[variant="{skin.PRIMARY}"]',
        f'QPushButton[variant="{skin.DANGER}"]',
        f'QToolButton#{skin.SECTION_HEADER}',
        'QLabel[status="error"]',
        'QLabel[role="sb-cell"]',
        skin.TABLE_SELECTOR,
        skin.REGION_SELECTOR,
        "#UpdateBanner",
    ):
        assert selector in qss


def test_the_region_box_outranks_qdarkstyles_frame_rule():
    """It has to be an id selector, and this is why.

    qdarkstyle ships ``.QFrame[frameShape="0"] { border: 1px transparent }``
    — a class selector plus an attribute. A property-scoped rule
    (``QFrame[mlgid="region"]``, the convention everywhere else in this
    sheet) is a type selector plus an attribute, which loses, and the
    border then paints *transparent*: the rule is live, the box is
    invisible, and nothing anywhere reports a problem.
    """
    assert skin.REGION_SELECTOR.startswith("QFrame#")


@pytest.mark.gui
@pytest.mark.parametrize("theme", THEMES)
def test_a_boxed_region_actually_paints_its_border(qtbot, theme):
    """The rule is live *and* wins — asserted on pixels, not on the sheet.

    A stylesheet rule that loses a specificity contest is still in the
    sheet, so only what reaches the screen can tell the two apart.
    """
    from PySide6.QtWidgets import QApplication, QLabel

    from mlgidgui.theme import apply_dark_theme, apply_light_theme
    from mlgidgui.widgets import boxed

    app = QApplication.instance()
    (apply_dark_theme if theme == "dark" else apply_light_theme)(app)

    frame = boxed(QLabel("x"))
    qtbot.addWidget(frame)
    frame.resize(120, 60)
    frame.show()
    qtbot.waitExposed(frame)

    image = frame.grab().toImage()
    border = theme_tokens.color("border", theme).lstrip("#").lower()
    # Along the top edge, away from the rounded corners.
    row = {
        image.pixelColor(x, 0).name().lstrip("#").lower()
        for x in range(20, 100)
    }
    assert border in row, f"no {border} border on a boxed region: {row}"


def test_applying_a_theme_installs_that_theme_s_skin(main_window):
    """`theme._apply` appends the skin, and the qdarkstyle sheet it sits
    on is still intact (`test_theme_apply` pins the palette literals)."""
    from PySide6.QtWidgets import QApplication

    from mlgidgui.theme import apply_dark_theme, apply_light_theme

    app = QApplication.instance()

    apply_light_theme(app)
    sheet = app.styleSheet()
    assert skin.marker("light") in sheet
    assert skin.marker("dark") not in sheet
    assert 'QPushButton[variant="primary"]' in sheet
    assert theme_tokens.color("accent", "light") in sheet

    apply_dark_theme(app)
    sheet = app.styleSheet()
    assert skin.marker("dark") in sheet
    assert skin.marker("light") not in sheet
    assert theme_tokens.color("accent", "dark") in sheet


def test_untagged_widgets_are_untouched_by_the_skin(main_window, qtbot):
    """The regression this whole design exists to prevent: a plain button
    (a stand-in for silx's and pyFAI's) must resolve exactly as it did
    before the skin was appended."""
    import qdarkstyle
    from qdarkstyle.dark.palette import DarkPalette
    from PySide6.QtWidgets import QApplication, QPushButton

    app = QApplication.instance()
    plain = QPushButton("plain")
    qtbot.addWidget(plain)
    tagged = QPushButton("primary")
    tagged.setProperty("variant", skin.PRIMARY)
    qtbot.addWidget(tagged)

    def repolish():
        for w in (plain, tagged):
            w.style().unpolish(w)
            w.style().polish(w)
            w.ensurePolished()

    # Baseline is qdarkstyle WITHOUT the skin: the question is what the
    # skin adds, not what the theme does.
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6",
                                                 palette=DarkPalette))
    repolish()
    baseline = plain.palette().buttonText().color().name()

    from mlgidgui.theme import apply_dark_theme

    apply_dark_theme(app)
    repolish()

    assert plain.palette().buttonText().color().name() == baseline
    assert (tagged.palette().buttonText().color().name().lower()
            == theme_tokens.color("accent", "dark"))
