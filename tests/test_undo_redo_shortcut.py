"""Undo/redo keyboard chords must fire even when another widget claims the
same standard shortcut ("ambiguous shortcut overload").

The embedded pyFAI calibration dialog pulls in silx mask-tool / pyFAI
peak-picking widgets that bind Undo/Redo to the standard sequences and
outlive the dialog (pyFAI's CalibrationContext singleton). Afterwards Qt
sees two actions on the redo chord and fires neither via the keyboard
(the Edit menu still works). ``MainWindow.eventFilter`` intercepts the
chord at the ShortcutOverride stage and drives undo/redo itself.

Here we simulate the lingering competitor with a rogue ApplicationShortcut
action bound to the redo chords, then confirm the keyboard still redoes.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QMenu, QMessageBox

from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))


def _open(window, path):
    window._set_active_session(NexusSession.open(path))


def _n_det(path, entry, frame):
    t = file_model.load_peaks(path, entry, frame)["detected"]
    return 0 if t is None else len(t)


def _sel_first_detected(window, frame):
    det = (window.viewer._frame_peaks.get(frame) or {})["detected"]
    return SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[0]),
        radius=float(det.radius[0]), angle=float(det.angle[0]),
        radius_width=float(det.radius_width[0]),
        angle_width=float(det.angle_width[0]),
        is_ring=bool(det.is_ring[0]), score=float(det.score[0]),
        amplitude=float(det.amplitude[0]),
    )


def _add_rogue_redo(window):
    """A second ApplicationShortcut action on the redo chords -> ambiguity."""
    rogue = QAction("rogue redo", window)
    rogue.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
    rogue.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
    window.addAction(rogue)
    return rogue


def test_redo_chord_fires_despite_ambiguous_shortcut(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    w = main_window
    w.show()
    _open(w, synthetic_nexus_with_peaks)
    path = w.session.temp_path
    v = w.viewer
    _add_rogue_redo(w)

    # delete a peak, undo it -> one entry waiting on the redo stack
    w.viewer._set_selected(_sel_first_detected(w, 0))
    w._on_delete_peak_requested(v._selected)
    qtbot.wait(10)
    w.action_undo.trigger()
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 3
    assert len(v._redo_stack) == 1

    # Ctrl+Shift+Z over an ambiguous binding still redoes (via eventFilter)
    qtbot.keyClick(
        w, Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 2
    assert len(v._redo_stack) == 0


def test_ctrl_y_redo_chord_fires_despite_ambiguity(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    w = main_window
    w.show()
    _open(w, synthetic_nexus_with_peaks)
    path = w.session.temp_path
    v = w.viewer
    _add_rogue_redo(w)

    w.viewer._set_selected(_sel_first_detected(w, 0))
    w._on_delete_peak_requested(v._selected)
    qtbot.wait(10)
    w.action_undo.trigger()
    qtbot.wait(10)
    assert len(v._redo_stack) == 1

    qtbot.keyClick(w, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 2


def test_undo_chord_still_works_with_filter(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    """The filter also drives undo; Ctrl+Z must still reverse an op."""
    w = main_window
    w.show()
    _open(w, synthetic_nexus_with_peaks)
    path = w.session.temp_path
    v = w.viewer

    w.viewer._set_selected(_sel_first_detected(w, 0))
    w._on_delete_peak_requested(v._selected)
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 2

    qtbot.keyClick(w, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 3


def test_redo_chord_no_double_fire_without_competitor(
    main_window, synthetic_nexus_with_peaks, qtbot,
):
    """With no competing action, one chord press redoes exactly once
    (the filter consumes the key, so the menu action does not also fire)."""
    w = main_window
    w.show()
    _open(w, synthetic_nexus_with_peaks)
    path = w.session.temp_path
    v = w.viewer

    # two deletes so a single redo lands at 2 (a double-fire would hit 1)
    w.viewer._set_selected(_sel_first_detected(w, 0))
    w._on_delete_peak_requested(v._selected)
    qtbot.wait(10)
    w.viewer._set_selected(_sel_first_detected(w, 0))
    w._on_delete_peak_requested(v._selected)
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 1
    w.action_undo.trigger(); qtbot.wait(10)
    w.action_undo.trigger(); qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 3
    assert len(v._redo_stack) == 2

    qtbot.keyClick(
        w, Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    qtbot.wait(10)
    assert _n_det(path, "entry_0000", 0) == 2  # exactly one redo, not two


def test_event_filter_survives_an_object_that_is_already_gone(qtbot):
    """The filter sits on the whole application, so it is offered events for
    widgets in the middle of being destroyed (a QMenu's hide event). Its
    Python wrapper is already invalid then; handing it back to C++ through
    ``super().eventFilter`` raised "Internal C++ object already deleted"
    out of the filter and ended in a crash on a user's machine."""
    import shiboken6

    from mlgidgui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    menu = QMenu(w)
    shiboken6.delete(menu)
    events = [QEvent(QEvent.Type.Hide)] + [
        QKeyEvent(kind, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        for kind in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride)
    ]
    for event in events:
        assert w.eventFilter(menu, event) is False
