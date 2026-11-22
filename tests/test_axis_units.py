"""Axis units never get an SI prefix.

pyqtgraph rescales an axis that has ``units`` and prepends an SI prefix
to keep the numbers between 1 and 1000: a radius range below 1 read
"mA^-1" (values x 1000), one above 1000 "kA^-1", an angle range below 1
"mdeg". The q axes are in 1/A and the angle in degrees whatever the
zoom, so the prefixes were wrong. Source: viewer_items.py
``_disable_si_prefix``; applied in image_viewer.py and profile_viewer.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.profile_viewer import ProfileViewer
from mlgidgui.session import NexusSession
from mlgidgui.viewer_styles import MODE_CARTESIAN, MODE_POLAR

pytestmark = pytest.mark.gui

RANGES = [(0.0, 5.0), (0.3, 0.9), (0.05, 0.2), (0.0, 0.01), (1500.0, 3000.0)]


def _unit_text(axis) -> str:
    """The units part of the label: what sits between the parentheses."""
    s = axis.labelString()
    return s[s.rindex("(") + 1:s.rindex(")")]


def test_profile_radius_axis_stays_in_inverse_angstrom(qtbot):
    viewer = ProfileViewer()
    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitExposed(viewer)
    axis = viewer._radial_plot.getAxis("bottom")
    for rng in RANGES:
        viewer._radial_plot.setXRange(*rng, padding=0)
        qtbot.wait(10)
        assert _unit_text(axis) == "Å⁻¹", rng
        assert axis.autoSIPrefixScale == 1.0, rng


def test_profile_angle_axis_stays_in_degrees(qtbot):
    viewer = ProfileViewer()
    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitExposed(viewer)
    axis = viewer._angular_plot.getAxis("bottom")
    for rng in [(0.0, 90.0), (-180.0, 180.0), (0.5, 0.9), (0.0, 0.001)]:
        viewer._angular_plot.setXRange(*rng, padding=0)
        qtbot.wait(10)
        assert _unit_text(axis) == "deg", rng
        assert axis.autoSIPrefixScale == 1.0, rng


@pytest.mark.parametrize("mode", [MODE_POLAR, MODE_CARTESIAN])
def test_image_axes_stay_in_inverse_angstrom(
    main_window, synthetic_nexus_with_peaks, qtbot, mode,
):
    """Zoomed on a low-q region (values below 1) the image axes used
    to read "mA^-1"."""
    w = main_window
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    w._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    v = w.viewer
    v.set_mode(mode)
    qtbot.wait(50)
    vb = v._plot.getViewBox()
    for rng in [(0.2, 0.8), (1.0, 1.4)]:
        vb.setRange(xRange=rng, yRange=(0.2, 0.8), padding=0)
        qtbot.wait(20)
        for name in ("bottom", "left"):
            axis = v._plot.getAxis(name)
            unit = _unit_text(axis)
            assert unit in ("Å⁻¹", "deg"), (mode, name, rng, unit)
            assert axis.autoSIPrefixScale == 1.0, (mode, name, rng)
