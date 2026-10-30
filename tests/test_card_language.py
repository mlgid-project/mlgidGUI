"""One container shape for every titled section.

Sections used to be built three ways — a bold ``QLabel`` with a manual
``QFont``, a ``CollapsibleSection``, and a ``QGroupBox`` — so the Display,
Pipeline and Conversion docks each read like a different program. ``Card``
is the single shape; ``CollapsibleSection`` is now one line over it.

The tests worth having here are the ones a refactor breaks silently: the
13 existing call sites still get the API they were written against, the
title really renders as a heading (a `role` that no rule matches is an
invisible no-op), and the collapse still emits exactly once.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QToolButton

from mlgidgui import skin
from mlgidgui.widgets import (
    BODY_INSET,
    Card,
    CollapsibleSection,
    section_label,
)

pytestmark = pytest.mark.gui


def test_a_plain_card_has_a_title_and_a_body(qtbot):
    card = Card("Selection")
    qtbot.addWidget(card)
    assert card.title() == "Selection"
    assert card.findChild(QToolButton) is None, "no chevron when not collapsible"

    button = QPushButton("Browse…")
    card.body_layout.addWidget(button)
    assert button.parent() is not None
    assert card.is_expanded()


def test_the_title_is_tagged_so_the_skin_can_paint_it(qtbot):
    """A `role` value with no matching rule is a silent no-op, so the tag
    and the sheet are checked against the same constant."""
    card = Card("Metadata")
    qtbot.addWidget(card)
    titles = [w for w in card.findChildren(QLabel)
              if w.property("role") == skin.CARD_TITLE]
    assert len(titles) == 1
    assert titles[0].text() == "Metadata"
    for theme in ("dark", "light"):
        assert f'QLabel[role="{skin.CARD_TITLE}"]' in skin.build_qss(theme)


def test_a_collapsible_card_toggles_its_body_once(qtbot):
    card = CollapsibleSection("Detection", expanded=True)
    qtbot.addWidget(card)
    card.show()
    seen = []
    card.expandedChanged.connect(seen.append)

    assert card.is_expanded()
    card._toggle.setChecked(False)
    assert not card.is_expanded()
    card._toggle.setChecked(True)
    assert card.is_expanded()
    assert seen == [False, True], "one emit per user toggle"


def test_a_collapsed_section_starts_closed(qtbot):
    card = CollapsibleSection("Fitting", expanded=False)
    qtbot.addWidget(card)
    card.show()
    assert not card.is_expanded()
    assert not card._toggle.isChecked()


def test_the_status_slot_is_hidden_until_it_has_text(qtbot):
    """It sits beside the title, so an empty one must not eat the width
    the header rule spans."""
    card = Card("CIF")
    qtbot.addWidget(card)
    card.show()
    assert not card._status.isVisible()

    card.set_status("12 cached", "ok")
    assert card._status.isVisible()
    assert card._status.text() == "12 cached"
    assert card._status.property("status") == "ok"

    card.set_status("")
    assert not card._status.isVisible()


def test_the_old_section_api_still_holds(qtbot):
    """13 call sites say ``CollapsibleSection(title, expanded=…)`` and
    fill ``body_layout``; that contract is what makes this refactor
    invisible to them."""
    section = CollapsibleSection("Matching", expanded=False)
    qtbot.addWidget(section)
    assert isinstance(section, Card)
    assert section.body_layout is not None
    assert tuple(
        section.body_layout.getContentsMargins()) == BODY_INSET
    assert hasattr(section, "expandedChanged")


def test_section_label_is_tagged_not_hand_bolded(qtbot):
    """The bold ``QFont`` and the ``<b>…</b>`` markup it replaces did not
    follow a theme flip."""
    label = section_label("Overlays")
    qtbot.addWidget(label)
    assert label.property("role") == "section"
    assert label.text() == "Overlays"
    assert "<b>" not in label.text()


def test_the_selected_peak_panel_is_a_card_now(main_window):
    """It was the app's only ``QGroupBox`` in a dock: a box drawn around
    one section of the Display column, which made it read as a different
    kind of thing from the sections above it."""
    panel = main_window.parameter_panel
    assert isinstance(panel, Card)
    assert not isinstance(panel, QGroupBox)
    assert panel.title() == "Selected peak"
    # Its contents survived the move onto the card's body layout.
    assert panel.btn_delete_peak.parent() is not None
    assert panel._form.rowCount() > 0


def test_the_docked_panels_agree_on_one_section_shape(main_window,
                                                      conversion_panel):
    """Every section in the three docks is a Card; a QGroupBox left in
    one of them is the inconsistency this stage removed. Modal dialogs
    keep theirs on purpose — that is native chrome, and restyling it is
    where the scoping rule gets dangerous."""
    for panel in (main_window.parameter_panel, conversion_panel):
        assert not panel.findChildren(QGroupBox), (
            f"{type(panel).__name__} still holds a QGroupBox")
    assert len(conversion_panel._sections) == 5
    for section in conversion_panel._sections:
        assert isinstance(section, Card)


@pytest.fixture
def conversion_panel(qtbot):
    from mlgidgui.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)
    return panel

def test_a_closed_section_still_reports_the_width_it_wants(qtbot):
    """The dock-sizing code asks a collapsed section how wide it would
    be if opened; a hidden widget still answers ``sizeHint``, it just
    stops contributing to its parent's."""
    card = Card("Fitting", collapsible=True, expanded=False)
    qtbot.addWidget(card)
    wide = QLabel("a label far wider than the collapsed header is")
    card.body_layout.addWidget(wide)
    assert not card.is_expanded()
    assert card.open_width_hint() > card.sizeHint().width()
    assert card.open_width_hint() >= wide.sizeHint().width()
