"""The strips above the image wrap instead of setting a width floor.

A ``QHBoxLayout``'s minimum width is the sum of everything in it and it
cannot give any of that back, so the viewer's control strip and the
workflow rail used to stop the central column from being dragged
narrower than roughly 740 px — the complaint these tests pin down.
With a ``FlowLayout`` the floor is the widest single cluster and the
controls take a second row instead.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidgui.flow_layout import ToolGroup, install_flow, wrapping_bar
from mlgidgui.image_viewer import GIWAXSImageViewer
from mlgidgui.theme import apply_dark_theme
from mlgidgui.workflow_rail import STAGES, WorkflowRail

pytestmark = pytest.mark.gui


def _fixed(width: int, text: str = "x") -> QLabel:
    label = QLabel(text)
    label.setFixedSize(width, 20)
    return label


@pytest.fixture
def bar(qtbot):
    """A flow bar with four 100 px items and no margins."""
    holder = wrapping_bar(margins=(0, 0, 0, 0))
    qtbot.addWidget(holder)
    layout = holder.layout()
    for index in range(4):
        layout.addWidget(_fixed(100, str(index)))
    return holder


def test_a_row_wraps_when_the_next_item_would_not_fit(bar):
    layout = bar.layout()
    wide = layout.heightForWidth(4 * 100 + 3 * 16)
    assert layout.heightForWidth(240) > wide, "two rows"
    assert layout.heightForWidth(120) > layout.heightForWidth(240)


def test_the_minimum_is_the_widest_item_not_the_sum(bar):
    layout = bar.layout()
    assert layout.minimumSize().width() == 100
    assert layout.sizeHint().width() == 4 * 100 + 3 * 16


def test_frame_controls_can_be_narrower_than_one_row(qtbot):
    """The point of the change: the viewer no longer imposes the sum of
    its toolbar on the central column."""
    apply_dark_theme(QApplication.instance())
    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    viewer.insert_frame_controls([
        QPushButton("prev"), QPushButton("play"), QPushButton("next"),
        QSlider(Qt.Orientation.Horizontal), QSpinBox(), QLabel("/ 602"),
    ])
    layout = viewer._toolbar_layout
    one_row = layout.sizeHint().width()
    assert layout.minimumSize().width() < one_row / 2
    assert viewer.minimumSizeHint().width() < 400
    assert layout.heightForWidth(400) > layout.heightForWidth(one_row)


def test_a_cluster_wraps_as_a_unit(qtbot):
    """A break between "Colormap:" and its combo would read as a bug, so
    the strip only ever breaks between clusters."""
    apply_dark_theme(QApplication.instance())
    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(420, 600)
    viewer.show()
    qtbot.waitExposed(viewer)
    holder = viewer._toolbar_layout.parentWidget()
    label = next(w for w in viewer.findChildren(QLabel)
                 if w.text() == "Colormap:")
    assert (label.mapTo(holder, label.rect().topLeft()).y()
            == viewer._cmap_combo.mapTo(
                holder, viewer._cmap_combo.rect().topLeft()).y())


def test_an_empty_cluster_gives_its_row_back(qtbot):
    """Regression: Qt decides whether to re-run the parent layout by
    comparing size *hints*, and a flow layout's hint is always one row
    tall. Without the nudge in ``FlowLayout.invalidate`` the strip stayed
    two rows tall after the host hid a cluster, until the next resize.
    """
    holder = QWidget()
    qtbot.addWidget(holder)
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    strip = wrapping_bar(margins=(0, 0, 0, 0))
    column.addWidget(strip)
    column.addStretch(1)

    group = ToolGroup()
    member = _fixed(200, "wide")
    group.add(member)
    strip.layout().addWidget(_fixed(100))
    strip.layout().addWidget(group)

    holder.resize(220, 400)
    holder.show()
    qtbot.waitExposed(holder)
    two_rows = strip.height()
    assert strip.layout().heightForWidth(220) == two_rows

    member.setVisible(False)
    qtbot.waitUntil(lambda: strip.height() < two_rows, timeout=2000)
    assert group.sizeHint().isEmpty(), "an empty cluster takes no room"


def test_an_expanding_item_eats_the_rows_leftover(qtbot):
    holder = wrapping_bar(margins=(0, 0, 0, 0))
    qtbot.addWidget(holder)
    fixed = _fixed(100)
    slider = QSlider(Qt.Orientation.Horizontal)
    holder.layout().addWidget(fixed)
    holder.layout().addWidget(slider)
    holder.resize(600, 40)
    holder.show()
    qtbot.waitExposed(holder)
    assert fixed.width() == 100
    assert slider.width() > 400, "the slider takes the rest of the row"


def test_the_rail_wraps_and_keeps_every_stage(qtbot):
    apply_dark_theme(QApplication.instance())
    rail = WorkflowRail()
    qtbot.addWidget(rail)
    layout = rail.layout()
    assert layout.minimumSize().width() < layout.sizeHint().width() / 3
    assert layout.heightForWidth(300) > layout.heightForWidth(1200)

    rail.resize(300, layout.heightForWidth(300))
    rail.show()
    qtbot.waitExposed(rail)
    seen = []
    rail.stageActivated.connect(seen.append)
    for key, _, _ in STAGES:
        chip = rail.chips[key]
        assert chip.isVisible(), key
        chip.button.click()
    assert seen == [key for key, _, _ in STAGES]


def test_the_rails_run_glyphs_stay_reachable_when_wrapped(qtbot):
    """Wrapping must not park a chip's run button under a sibling."""
    apply_dark_theme(QApplication.instance())
    rail = WorkflowRail()
    qtbot.addWidget(rail)
    rail.resize(320, rail.layout().heightForWidth(320))
    rail.show()
    qtbot.waitExposed(rail)
    boxes = []
    for key, _, _ in STAGES:
        button = rail.chips[key].run_button
        assert isinstance(button, QToolButton)
        top_left = button.mapTo(rail, button.rect().topLeft())
        boxes.append((key, top_left.x(), top_left.y(), button.width()))
    for index, (key, x, y, width) in enumerate(boxes):
        for other_key, ox, oy, owidth in boxes[index + 1:]:
            if y == oy:
                assert x + width <= ox or ox + owidth <= x, (key, other_key)
