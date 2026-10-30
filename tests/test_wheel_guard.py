"""The wheel may not change a value.

Qt lets the mouse wheel step a combobox through its items and a spin
box through its range on hover. Scrolling a dock therefore rewrites
whatever the cursor passed over, which is annoying for a display
toggle and dangerous for a conversion parameter: a nudged ``dq``
produces a plausible-looking wrong result rather than an error.

``widgets.ValueWheelBlocker`` is installed on the QApplication by
``MainWindow.__init__``, so these tests need the window fixture even
when they operate on a throwaway widget.

The other half of the contract is that the page still scrolls. Eating
the wheel outright would leave the conversion panel -- a long
scrollable form, mostly spin boxes -- freezing wherever the cursor
crossed a field, so the event is forwarded to the nearest scrollable
ancestor instead.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

pytestmark = pytest.mark.gui


def _wheel(widget, step: int = -120) -> None:
    QApplication.sendEvent(widget, QWheelEvent(
        QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, step),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))


# --- the value must not move ------------------------------------------


def test_a_closed_combobox_ignores_the_wheel(main_window):
    combo = QComboBox(main_window)
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    for step in (-120, 120):
        _wheel(combo, step)
        assert combo.currentIndex() == 1


def test_a_spin_box_ignores_the_wheel(main_window):
    spin = QSpinBox(main_window)
    spin.setRange(0, 10)
    spin.setValue(5)
    for step in (-120, 120):
        _wheel(spin, step)
        assert spin.value() == 5


def test_a_double_spin_box_ignores_the_wheel(main_window):
    spin = QDoubleSpinBox(main_window)
    spin.setDecimals(4)
    spin.setRange(0.0, 10.0)
    spin.setValue(0.0125)
    for step in (-120, 120):
        _wheel(spin, step)
        assert spin.value() == pytest.approx(0.0125)


def test_the_real_conversion_dq_field_ignores_the_wheel(main_window):
    """The field the user reported. It is a `QDoubleSpinBox` deep in a
    scroll area, which is exactly the shape that made hover-scrolling
    hazardous."""
    spin = main_window.conversion_panel.dq_q_gid
    spin.setValue(0.005)
    for step in (-120, 120):
        _wheel(spin, step)
        assert spin.value() == pytest.approx(0.005)


# --- ...but the page must still scroll --------------------------------


def test_the_wheel_still_scrolls_the_form_under_a_spin_box(main_window):
    """Forwarding, not eating. Without this a long form full of spin
    boxes is barely scrollable by wheel."""
    area = QScrollArea(main_window)
    body = QWidget()
    layout = QVBoxLayout(body)
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 10.0)
    spin.setValue(1.0)
    layout.addWidget(spin)
    filler = QWidget()
    filler.setMinimumHeight(4000)
    layout.addWidget(filler)
    area.setWidget(body)
    area.resize(200, 120)
    area.show()
    QApplication.processEvents()
    bar = area.verticalScrollBar()
    assert bar.maximum() > 0, "the area must actually be scrollable"
    bar.setValue(0)

    _wheel(spin, -120)

    assert bar.value() > 0
    assert spin.value() == pytest.approx(1.0)


# --- and the deliberate exclusions ------------------------------------


def test_a_slider_still_takes_the_wheel(main_window):
    """Sliders are left alone on purpose: the frame scrubber is a place
    the wheel genuinely helps, and none of them writes to the file."""
    slider = QSlider(Qt.Orientation.Horizontal, main_window)
    slider.setRange(0, 100)
    slider.setValue(50)
    _wheel(slider, 120)
    assert slider.value() != 50


def test_the_open_popup_list_still_scrolls(main_window):
    """Only the CLOSED combobox is filtered. The popup is a separate
    QListView, so a long item list stays scrollable."""
    from mlgidgui.widgets import ValueWheelBlocker

    combo = QComboBox(main_window)
    combo.addItems([str(i) for i in range(50)])
    view = combo.view()
    assert not isinstance(view, ValueWheelBlocker._TARGETS)
