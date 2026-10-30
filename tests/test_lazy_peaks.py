"""Per-frame peaks load lazily, not all up front.

`_load_entry_into_viewer` used to loop over every frame of an entry
loading detected/fitted/matched peaks synchronously — hundreds of HDF5
opens per entry switch on a many-frame entry, a big part of the lag.
Now only the frame the viewer lands on is loaded; other frames load on
demand when navigated to (`_on_viewer_frame_changed` → `_load_frame_peaks`).
"""

from __future__ import annotations

import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def test_entry_load_loads_only_landed_frame(
    main_window, synthetic_nexus_with_peaks, monkeypatch
):
    # Peaks are read through the viewer's already-open FrameSource handle
    # (``read_peaks``), not a fresh ``load_peaks`` open — that handle reuse
    # is what stops a per-frame network open from freezing the GUI. Spy the
    # handle reader so the "only the landed frame loads, lazily" contract is
    # still asserted.
    real = file_model.read_peaks
    frames_loaded: list[int] = []

    def _spy(f, entry, frame, extra=()):
        frames_loaded.append(int(frame))
        return real(f, entry, frame, extra)

    monkeypatch.setattr(file_model, "read_peaks", _spy)

    session = NexusSession.open(synthetic_nexus_with_peaks)  # 3 frames
    main_window._set_active_session(session)

    # Only frame 0 (the landed frame) was loaded — not all 3.
    assert frames_loaded == [0]
    assert main_window._loaded_peak_frames == {0}

    # Navigating to another frame loads it lazily, once.
    main_window.viewer.set_frame(1)
    assert 1 in frames_loaded
    assert main_window._loaded_peak_frames == {0, 1}

    # Returning to a loaded frame does not re-read it.
    before = list(frames_loaded)
    main_window.viewer.set_frame(0)
    assert frames_loaded == before


def test_peaks_read_via_open_handle_not_reopen(
    main_window, synthetic_nexus_with_peaks, monkeypatch
):
    """Peaks come through the viewer's open FrameSource handle, never a
    fresh ``load_peaks(path)`` open — reopening the master is the
    multi-second network round-trip that froze the GUI per frame/entry on
    an external-link master."""
    def _boom(*a, **k):
        raise AssertionError("load_peaks (reopen-by-path) must not run "
                             "while a live FrameSource handle is available")

    monkeypatch.setattr(file_model, "load_peaks", _boom)
    monkeypatch.setattr(file_model, "load_matched_peaks", _boom)

    session = NexusSession.open(synthetic_nexus_with_peaks)
    main_window._set_active_session(session)  # must not raise

    assert main_window._loaded_peak_frames == {0}
    main_window.viewer.set_frame(1)  # frame nav also uses the handle
    assert main_window._loaded_peak_frames == {0, 1}
