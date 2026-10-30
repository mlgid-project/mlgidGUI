"""``show_stack(preserve_view=...)`` frame + viewbox-range behavior.

Source: ``GIWAXSImageViewer.show_stack``. ``preserve_view=True`` clamps and
keeps ``current_frame`` and restores the viewbox range via
``getViewBox().setRange(...padding=0)``; ``False`` resets to frame 0
with autorange. ``show_stack`` requires a ``_LazyImageStack``-backed
``EntryStack`` (:1065), so the already-loaded ``viewer._stack`` is
reused rather than hand-rolled.
"""

from __future__ import annotations

import pytest

from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def test_preserve_view_keeps_frame(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.set_frame(2)
    v.show_stack(v._stack, preserve_view=True)
    assert v.current_frame == 2


def test_no_preserve_resets_to_first_frame(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.set_frame(2)
    v.show_stack(v._stack, preserve_view=False)
    assert v.current_frame == 0


def test_preserve_view_keeps_viewbox_range(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    vb = v._plot.getViewBox()
    vb.setRange(xRange=(0.5, 1.5), yRange=(2.0, 3.0), padding=0)
    (x0, x1), (y0, y1) = vb.viewRange()

    v.show_stack(v._stack, preserve_view=True)

    (nx0, nx1), (ny0, ny1) = vb.viewRange()
    assert (nx0, nx1) == pytest.approx((x0, x1), rel=1e-3)
    assert (ny0, ny1) == pytest.approx((y0, y1), rel=1e-3)
