"""Manual-peak box operations on a loaded file (MainWindow + viewer).

Opened through ``_set_active_session`` (the synchronous orchestrator)
exactly like the session smoke tests, so no modal copy worker is
spawned. Source: image_viewer.py manual API 1430-1591, undo helpers
1692-1732, action classes 306-393.
"""

from __future__ import annotations

import pytest

from mlgidgui.image_viewer import ManualPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _peak(temp_id=1):
    return ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0,
        temp_id=temp_id,
    )


def test_add_and_get_roundtrip(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    v.add_manual_peak(0, p)
    assert v.manual_peaks(0) == [p]


def test_frame_isolation(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    v.add_manual_peak(1, p)
    assert v.manual_peaks(1) == [p]
    assert v.manual_peaks(0) == []


def test_remove_absent_peak_is_silent_noop(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    rec = []
    v.manualPeakRemoved.connect(lambda *a: rec.append(a))
    v.remove_manual_peak(0, _peak())  # never added
    assert rec == []
    assert v.manual_peaks(0) == []


def test_undo_redo_add(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    v.add_manual_peak(0, p)
    v.undo_last_action()
    assert v.manual_peaks(0) == []
    v.redo_last_action()
    assert v.manual_peaks(0) == [p]


def test_undo_redo_remove(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    v.add_manual_peak(0, p)
    v.remove_manual_peak(0, p)
    assert v.manual_peaks(0) == []
    v.undo_last_action()  # reverses the remove
    assert v.manual_peaks(0) == [p]
    v.redo_last_action()  # re-applies the remove
    assert v.manual_peaks(0) == []


def test_clear_history_blocks_undo(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    v.add_manual_peak(0, p)
    v.clear_history()
    v.undo_last_action()  # stack empty → no-op
    assert v.manual_peaks(0) == [p]


def test_clear_all_manual_peaks(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.add_manual_peak(0, _peak(1))
    v.add_manual_peak(1, _peak(2))
    v.clear_all_manual_peaks()
    assert v.manual_peaks(0) == []
    assert v.manual_peaks(1) == []
    # History was cleared too: undo cannot resurrect anything.
    v.undo_last_action()
    assert v.manual_peaks(0) == []
    assert v.manual_peaks(1) == []


def test_signals_carry_frame_and_peak(main_window, synthetic_nexus, qtbot):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    p = _peak()
    with qtbot.waitSignal(v.manualPeakAdded, timeout=1000) as added:
        v.add_manual_peak(0, p)
    assert added.args == [0, p]
    with qtbot.waitSignal(v.manualPeakRemoved, timeout=1000) as removed:
        v.remove_manual_peak(0, p)
    assert removed.args == [0, p]
