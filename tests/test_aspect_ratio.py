"""Image-viewer aspect-ratio control (toolbar "Aspect:" Fit / Custom).

The viewer's plot is `setAspectLocked(False)` by default (Fit = stretch
to fill). Custom locks the image to a target on-screen **width:height**
ratio (so `2` = twice as wide as tall, the natural shape of a polar
radius×angle map). The ratio is pixel/extent-based, not q-based: the
data-unit lock handed to pyqtgraph is derived from the current image's
extent (`_data_extent`), so the same ratio means the same shape in
Cartesian and polar. Scrolling over a single axis nudges the ratio.

Mirrors `test_view_preservation.py` (drives the real viewer / ViewBox;
`vb.state['aspectLocked']` is `False` when free, else the locked
data-unit ratio) and `test_theme_persistence.py` (QSettings round-trip;
conftest redirects XDG_CONFIG_HOME so the store is throwaway).
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from mlgidgui.image_viewer import GIWAXSImageViewer
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _name_settings() -> None:
    """Mirror production QSettings resolution (set in mlgidgui.main)."""
    app = QApplication.instance()
    app.setOrganizationName("mlgidGUI")
    app.setApplicationName("mlgidGUI")


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _box_aspect(v) -> float | None:
    """The on-screen width:height the current lock produces, or None when
    free. pyqtgraph's lock value ``a`` is (px per x-unit)/(px per y-unit),
    so the displayed image box is ``box_w:box_h = (Dx/Dy) * a``; that
    should equal the requested ratio."""
    ext = v._data_extent()
    r = v._plot.getViewBox().state["aspectLocked"]
    if r is False or ext is None:
        return None
    return (ext[0] / ext[1]) * r


class _Wheel:
    """Minimal stand-in for a pyqtgraph wheel event."""

    def __init__(self, delta: int) -> None:
        self._d = delta

    def delta(self) -> int:
        return self._d

    def accept(self) -> None:
        pass


def test_startup_is_default(main_window, synthetic_nexus):
    """The viewer opens in the per-mode Default preset (polar 2:1)."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    assert v.aspect()[0] == "default"
    assert _box_aspect(v) == pytest.approx(v._default_ratio_for_mode(), rel=1e-3)


def test_double_click_sets_default(main_window, synthetic_nexus):
    """A bare LMB double-click snaps the aspect back to Default."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.set_aspect("custom", 5.0)
    assert v.aspect()[0] == "custom"
    v._label_filter.doubleClicked.emit()   # what a bare double-click fires
    assert v.aspect()[0] == "default"
    assert _box_aspect(v) == pytest.approx(v._default_ratio_for_mode(), rel=1e-3)


def test_custom_box_aspect_matches_ratio(main_window, synthetic_nexus):
    """The displayed width:height equals the requested ratio (the fix for
    the math that collapsed polar to a sliver)."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    for r in (1.0, 2.0, 4.0):
        v.set_aspect("custom", r)
        assert _box_aspect(v) == pytest.approx(r, rel=1e-3)
    v.set_aspect("fit")
    assert v._plot.getViewBox().state["aspectLocked"] is False


def test_polar_wide_ratio_does_not_compress_x(main_window, synthetic_nexus):
    """Regression for the sliver bug. In polar the angle axis (y) spans far
    more than radius (x), so a wide ratio must stretch x: the pyqtgraph
    lock value (px-per-x / px-per-y) has to be > 1, not the tiny reciprocal
    that crushed x to nothing."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.set_mode("polar")
    dx, dy = v._data_extent()
    assert dy > dx                      # angle extent >> radius extent
    v.set_aspect("custom", 2.0)
    lock = v._plot.getViewBox().state["aspectLocked"]
    assert lock > 1.0                   # x stretched, not compressed
    assert _box_aspect(v) == pytest.approx(2.0, rel=1e-3)


def test_same_ratio_same_shape_in_both_modes(main_window, synthetic_nexus):
    """A given ratio yields the same on-screen shape in Cartesian and
    polar even though the underlying data-unit lock differs."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer

    v.set_mode("polar")
    v.set_aspect("custom", 2.0)
    polar_box = _box_aspect(v)
    polar_lock = v._plot.getViewBox().state["aspectLocked"]

    v.set_mode("cartesian")
    cart_box = _box_aspect(v)
    cart_lock = v._plot.getViewBox().state["aspectLocked"]

    assert polar_box == pytest.approx(2.0, rel=1e-3)
    assert cart_box == pytest.approx(2.0, rel=1e-3)
    # The data-unit locks differ because the axes' extents differ.
    assert polar_lock != pytest.approx(cart_lock)


def test_default_preset_is_per_mode(main_window, synthetic_nexus):
    """The Default preset uses 2:1 for polar and 1:1 for Cartesian, and
    follows mode switches; the spin reflects it read-only."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer

    v.set_mode("polar")
    v.set_aspect("default")
    assert v.aspect() == ("default", pytest.approx(2.0))
    assert _box_aspect(v) == pytest.approx(2.0, rel=1e-3)
    assert v._aspect_spin.value() == pytest.approx(2.0)
    assert v._aspect_spin.isEnabled() is False   # auto, not user-editable

    v.set_mode("cartesian")   # switching mode re-applies the per-mode default
    assert v.aspect() == ("default", pytest.approx(1.0))
    assert _box_aspect(v) == pytest.approx(1.0, rel=1e-3)


def test_spin_enabled_only_in_custom(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    for mode in ("fit", "default"):
        v.set_aspect(mode)
        assert v._aspect_spin.isEnabled() is False
    v.set_aspect("custom", 1.0)
    assert v._aspect_spin.isEnabled() is True


def test_axis_scroll_autoselects_custom(main_window, synthetic_nexus):
    """Scrolling an axis from any mode switches to Custom, seeded with the
    live shown ratio, and the spin tracks it. x widens (ratio up), y
    heightens (ratio down). The live ratio is stubbed for determinism
    (offscreen `getAspectRatio` has no real geometry)."""
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v._live_box_ratio = lambda: 2.0   # deterministic seed

    v.set_aspect("fit")
    v._axis_wheel_event(_Wheel(120), axis=0)   # x-axis scroll from Fit
    assert v.aspect()[0] == "custom"
    assert v._aspect_spin.isEnabled() is True
    rx = v.aspect()[1]
    assert rx > 2.0                            # x widens
    assert v._aspect_spin.value() == pytest.approx(rx)

    v.set_aspect("fit")
    v._axis_wheel_event(_Wheel(120), axis=1)   # y-axis scroll from Fit
    assert v.aspect()[0] == "custom"
    assert v.aspect()[1] < 2.0                 # y heightens


def test_ratio_persists_and_mode_starts_default(main_window, synthetic_nexus, qtbot):
    """The Custom ratio round-trips through QSettings; a freshly built
    viewer starts in Default (the startup mode), not the persisted Custom."""
    _name_settings()
    _open(main_window, synthetic_nexus)
    main_window.viewer.set_aspect("custom", 2.0)
    s = QSettings()
    s.sync()
    assert float(s.value("viewerAspectRatio")) == pytest.approx(2.0)

    fresh = GIWAXSImageViewer()   # defaults to polar mode, no image yet
    qtbot.addWidget(fresh)
    assert fresh.aspect()[0] == "default"
    assert fresh._plot.getViewBox().state["aspectLocked"] is False  # no image
