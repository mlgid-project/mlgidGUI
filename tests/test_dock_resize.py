"""The bottom docks (Profiles / Peaks / Scan tracking) can be dragged
taller than the image.

Qt sizes a QTabWidget / QStackedWidget by the largest minimum of ALL its
pages, hidden ones included. The Structure editor (~430 px minimum) and
the welcome page (~450 px) therefore pinned the image area, and the
separator to the bottom docks hardly moved. ``widgets.
size_by_current_page`` takes hidden pages out of that minimum.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from mlgidgui.session import NexusSession
from mlgidgui.widgets import size_by_current_page

pytestmark = pytest.mark.gui


def test_only_the_current_page_sets_the_minimum(qtbot):
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    # The tall page's minimum comes from its layout (a size HINT), as
    # with the Structure editor; an explicit setMinimumHeight on the
    # page itself is always honoured by Qt and is not what this fixes.
    small, tall = QLabel("small"), QWidget()
    inner = QLabel("tall")
    inner.setMinimumHeight(600)
    QVBoxLayout(tall).addWidget(inner)
    tabs.addTab(small, "a")
    tabs.addTab(tall, "b")
    assert tabs.minimumSizeHint().height() >= 600     # Qt's default
    size_by_current_page(tabs)
    assert tabs.minimumSizeHint().height() < 200
    tabs.setCurrentIndex(1)                           # the tall page shows
    assert tabs.minimumSizeHint().height() >= 600
    assert tall.sizePolicy().verticalPolicy() != QSizePolicy.Policy.Ignored
    tabs.setCurrentIndex(0)
    assert tabs.minimumSizeHint().height() < 200


def test_bottom_docks_can_be_dragged_tall(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    w = main_window
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    qtbot.wait(100)
    central = w.centralWidget()
    # Image tab current: neither the Structure editor nor the welcome
    # page holds the image area open any more (it was ~506 px).
    assert central.minimumSizeHint().height() < 250
    w.resizeDocks([w._profile_dock], [600], Qt.Orientation.Vertical)
    qtbot.wait(50)
    assert w._profile_dock.height() > 500
