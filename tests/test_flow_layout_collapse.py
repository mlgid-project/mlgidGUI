"""A flow-layout item the layout skips must not keep covering things.

The layout leaves out clusters with nothing visible in them. It never
calls setGeometry on those, so before this they kept whatever rectangle
they had — for a widget that has never been laid out, Qt's default
640x480 at the parent's origin — and went on hit-testing there.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QWidget

from mlgidgui.flow_layout import ToolGroup, install_flow

pytestmark = pytest.mark.gui


@pytest.fixture
def host(qtbot) -> QWidget:
    widget = QWidget()
    qtbot.addWidget(widget)
    install_flow(widget, margins=(4, 2, 4, 2))
    return widget


def test_an_empty_cluster_is_collapsed_to_nothing(host):
    layout = host.layout()
    full = ToolGroup()
    full.add(QLabel("kept"))
    empty = ToolGroup()
    hidden = QLabel("gone")
    empty.add(hidden)
    hidden.setVisible(False)
    layout.addWidget(full)
    layout.addWidget(empty)

    host.resize(400, 60)
    host.show()
    layout.activate()

    assert empty.size().isEmpty(), "a skipped cluster must cover nothing"
    assert not full.size().isEmpty()


def test_a_cluster_that_fills_up_again_is_laid_out(host):
    layout = host.layout()
    group = ToolGroup()
    label = QLabel("later")
    group.add(label)
    label.setVisible(False)
    layout.addWidget(group)
    host.resize(400, 60)
    host.show()
    layout.activate()
    assert group.size().isEmpty()

    label.setVisible(True)
    group.updateGeometry()
    layout.invalidate()
    layout.activate()
    assert not group.size().isEmpty(), (
        "collapsing is geometry only — it must not strand the cluster")


def test_collapsing_does_not_change_visibility(host):
    """No visibility state to get stuck in."""
    layout = host.layout()
    group = ToolGroup()
    label = QLabel("x")
    group.add(label)
    label.setVisible(False)
    layout.addWidget(group)
    host.resize(400, 60)
    host.show()
    layout.activate()
    assert not group.isHidden()
