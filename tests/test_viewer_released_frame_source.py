"""Regression: the viewer must not raise ``RuntimeError("FrameSource
not acquired")`` when the user interacts with the plot during the
silx detach/reattach window (pipeline run, ROI commit, Add-to-fitted,
clear-peaks, save-as), where the FrameSource's h5py handle is closed.

Two interaction paths reach a released FrameSource:

1. Moving the cursor over the polar plot -> ``_compute_cursor_info``
   indexes the cached ``_LazyPolarStack``, which delegates into
   ``FrameSource.get_polar`` and raises on a closed handle. The
   ``_polar_cache is None`` guard alone is insufficient because a
   release path can leave the cache tuple in place while the source
   is closed; the lookup must additionally require ``is_open``.

2. Toggling the Cartesian/Polar radio -> ``set_mode`` ->
   ``_render_active_mode`` -> ``_build_*_params`` reads frame 0 through
   the FrameSource for robust levels. ``_render_active_mode`` must bail
   when the source is released and let ``acquire_frame_source``
   re-render once the handle reopens.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF

from mlgidgui.image_viewer import MODE_CARTESIAN, MODE_POLAR
from mlgidgui.session import NexusSession


def _open(window, path) -> NexusSession:
    """Open + activate, mirroring ``_on_open_finished`` (see
    ``test_smoke_silx_detach._open``)."""
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_polar_cursor_readout_survives_released_frame_source(
    main_window, synthetic_nexus
):
    """Cursor lookup over the polar plot returns a NaN intensity (not a
    crash) while the FrameSource is released but its polar cache is
    still present."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    viewer.set_mode(MODE_POLAR)
    # Rendering polar mode populates the cache the cursor path reads.
    assert viewer._polar_cache is not None

    # Reproduce the crash precondition: source closed, cache retained.
    viewer._frame_source.release()
    assert not viewer._frame_source.is_open
    assert viewer._polar_cache is not None

    info = viewer._compute_cursor_info(QPointF(1.0, 30.0))  # (radius, theta)
    assert info is not None and info["mode"] == "polar"
    assert math.isnan(info["intensity"])  # degraded, not raised

    viewer._frame_source.acquire()  # leave the viewer usable for teardown


def test_mode_toggle_during_detached_scope_does_not_raise(
    main_window, synthetic_nexus
):
    """Switching display mode inside ``_detached_silx_tree`` (FrameSource
    closed) records the new mode without reading the closed handle; the
    reattach on scope exit renders it and the viewer stays usable."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    start = viewer.mode
    other = MODE_CARTESIAN if start == MODE_POLAR else MODE_POLAR

    with main_window._detached_silx_tree():
        assert not viewer._frame_source.is_open
        viewer.set_mode(other)  # must not raise on the closed handle
        assert viewer.mode == other

    # Reattach reacquired the source; the viewer reads frames again.
    viewer.set_frame(1)
    assert viewer.current_frame == 1
