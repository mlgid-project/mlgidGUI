"""Progress bars carry the accent.

Two things can go wrong and neither shows up in a construction test. The
tag can be present while the sheet paints nothing — ``::chunk`` is a
sub-control, and a typo in the selector leaves qdarkstyle's blue in place
with no error anywhere. And a bar built at a call site that forgot the
tag looks fine in isolation but wrong next to the ones that have it.

So the colour check renders the widget and samples the pixels, and the
coverage check walks the bars the application actually builds.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QProgressBar

from mlgidgui import skin, theme_tokens
from mlgidgui.theme import apply_dark_theme, apply_light_theme
from mlgidgui.widgets import make_progress_dialog, skin_progress

pytestmark = pytest.mark.gui


def _fill_colours(bar: QProgressBar) -> set[str]:
    """Every colour painted along the bar's horizontal midline."""
    bar.resize(200, 20)
    bar.ensurePolished()
    image = bar.grab().toImage()
    y = image.height() // 2
    return {image.pixelColor(x, y).name().lower()
            for x in range(4, image.width() - 4)}


def _bar(value: int = 100) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(value)
    bar.setTextVisible(False)
    return bar


@pytest.mark.parametrize("theme,apply", [("dark", apply_dark_theme),
                                         ("light", apply_light_theme)])
def test_a_tagged_bar_fills_with_the_accent(qtbot, theme, apply):
    apply(QApplication.instance())
    tagged = skin_progress(_bar())
    plain = _bar()
    for w in (tagged, plain):
        qtbot.addWidget(w)

    accent = theme_tokens.color("accent", theme)
    assert accent in _fill_colours(tagged)
    assert accent not in _fill_colours(plain), "untagged bars must not change"


def test_the_fill_follows_a_live_theme_flip(qtbot, main_window):
    bar = skin_progress(_bar())
    qtbot.addWidget(bar)

    main_window._set_theme("light")
    assert theme_tokens.color("accent", "light") in _fill_colours(bar)
    main_window._set_theme("dark")
    assert theme_tokens.color("accent", "dark") in _fill_colours(bar)


def test_an_empty_bar_shows_no_accent(qtbot):
    """Guards the groove rule: the whole widget must not go orange when
    the run has not started."""
    apply_dark_theme(QApplication.instance())
    bar = skin_progress(_bar(0))
    qtbot.addWidget(bar)
    assert theme_tokens.color("accent", "dark") not in _fill_colours(bar)


def test_a_progress_dialog_tags_the_bar_qt_builds_for_it(qtbot, main_window):
    """``QProgressDialog`` owns its bar, so the tag has to be pushed down
    into a child the call site never sees."""
    dlg = make_progress_dialog(main_window, "Working…",
                               title="Test", maximum=100)
    qtbot.addWidget(dlg)
    bar = dlg.findChild(QProgressBar)
    assert bar is not None
    assert bar.property("mlgid") == "progress"
    dlg.close()


def test_every_progress_bar_the_app_builds_is_tagged(main_window,
                                                     pipeline_panel):
    """A new bar added without the tag would be the only blue one left."""
    bars = [
        main_window._sb_open_bar,
        main_window._sb_tree_bar,
        pipeline_panel._progress_bar,
        pipeline_panel._entry_progress_bar,
        pipeline_panel.cif_parse_bar,
    ]
    for bar in bars:
        assert bar.property("mlgid") == "progress"


@pytest.fixture
def pipeline_panel(monkeypatch):
    """A PipelinePanel with its full widget set.

    Backend-less (CI), the panel builds a stub without the progress rows.
    ``pipeline_panel`` binds ``is_mlgidbase_available`` by name at import,
    so the patch has to target the panel module.
    """
    from mlgidgui import pipeline_panel as panel_mod

    monkeypatch.setattr(panel_mod, "is_mlgidbase_available", lambda: True)
    return panel_mod.PipelinePanel()


def test_the_skin_defines_the_progress_rules():
    for theme in ("dark", "light"):
        qss = skin.build_qss(theme)
        assert skin.PROGRESS_SELECTOR in qss
        assert f"{skin.PROGRESS_SELECTOR}::chunk" in qss


def test_the_accent_is_the_magma_orange_in_the_default_theme():
    """The point of the change, pinned: the dark theme (the default) fills
    with magma orange, and the light theme with the magma magenta the
    buttons already use."""
    assert theme_tokens.color("accent", "dark") == "#fc8961"
    assert QColor(theme_tokens.color("accent", "light")).isValid()
