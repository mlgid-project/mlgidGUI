"""Bulk delete of a multi-selected set of detected peaks (Delete key).

Behaviour (handler ``_on_delete_peaks_requested`` + viewer
``deletePeaksRequested``):

* >= 2 detected peaks selected + Delete -> one confirmation, all rows
  removed in a single ``_detached_silx_tree`` scope.
* Undoable: one Ctrl+Z re-adds every row (fresh ids); redo deletes
  again. (The single-peak delete path stays non-undoable; this is a
  deliberately richer, higher-stakes action.)
* < 2 detected selected -> no-op (the single-peak path handles those).
* Cancel in the dialog -> nothing removed.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    """warning/critical exec() blocks headless -> turn into a fast,
    visible failure instead of a hang (question is patched per test)."""
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"unexpected blocking QMessageBox in test: {args[1:3]!r}"
        )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _n_detected(path, entry, frame) -> int:
    table = file_model.load_peaks(path, entry, frame)["detected"]
    return 0 if table is None else len(table)


def _n_fitted(path, entry, frame) -> int:
    table = file_model.load_peaks(path, entry, frame)["fitted"]
    return 0 if table is None else len(table)


def _select_fitted(window, frame: int, n: int) -> list[SelectedPeak]:
    """Multi-select the first ``n`` fitted rows on ``frame``."""
    tables = window.viewer._frame_peaks.get(frame) or {}
    fit = tables["fitted"]
    sels = [
        SelectedPeak(
            kind="fitted", frame=frame, peak_id=int(fit.ids[i]),
            radius=float(fit.radius[i]), angle=float(fit.angle[i]),
            radius_width=float(fit.radius_width[i]),
            angle_width=float(fit.angle_width[i]),
            is_ring=bool(fit.is_ring[i]),
            score=float(fit.score[i]),
            amplitude=float(fit.amplitude[i]),
        )
        for i in range(n)
    ]
    window.viewer._set_selected(sels[0])
    window.viewer._selected_extras = list(sels[1:])
    return sels


def _select_detected(window, frame: int, n: int) -> list[SelectedPeak]:
    """Multi-select the first ``n`` detected rows on ``frame``."""
    tables = window.viewer._frame_peaks.get(frame) or {}
    det = tables["detected"]
    sels = [
        SelectedPeak(
            kind="detected", frame=frame, peak_id=int(det.ids[i]),
            radius=float(det.radius[i]), angle=float(det.angle[i]),
            radius_width=float(det.radius_width[i]),
            angle_width=float(det.angle_width[i]),
            is_ring=bool(det.is_ring[i]),
            score=float(det.score[i]),
            amplitude=float(det.amplitude[i]),
        )
        for i in range(n)
    ]
    window.viewer._set_selected(sels[0])
    window.viewer._selected_extras = list(sels[1:])
    return sels


def _yes(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _cancel(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )


def test_bulk_delete_removes_all_selected(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """3 detected selected + confirm -> all 3 detected rows gone."""
    _open(main_window, synthetic_nexus_with_peaks)
    sels = _select_detected(main_window, 0, 3)
    _yes(monkeypatch)
    path = main_window.session.temp_path
    assert _n_detected(path, "entry_0000", 0) == 3

    main_window._on_delete_peaks_requested(sels)
    assert _n_detected(path, "entry_0000", 0) == 0


def test_bulk_delete_leaves_fitted_intact(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    """Detected bulk delete must not touch fitted rows (no cascade).

    This is the unlinked contract, which is why the fixture turns the
    fitted/detected link off: with the link on (the shipped default) a
    detected delete deliberately takes the matching fit with it, and
    that half is covered in ``test_peak_linking.py``."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    fit_before = len(file_model.load_peaks(path, "entry_0000", 0)["fitted"])
    sels = _select_detected(main_window, 0, 3)
    _yes(monkeypatch)
    main_window._on_delete_peaks_requested(sels)
    fit_after = len(file_model.load_peaks(path, "entry_0000", 0)["fitted"])
    assert fit_after == fit_before


def test_bulk_delete_undo_restores_all(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """One Ctrl+Z re-adds every deleted detected row."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_detected(main_window, 0, 3)
    _yes(monkeypatch)
    main_window._on_delete_peaks_requested(sels)
    assert _n_detected(path, "entry_0000", 0) == 0

    main_window.viewer.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 3


def test_bulk_delete_redo_deletes_again(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """undo then redo lands back at deleted."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_detected(main_window, 0, 3)
    _yes(monkeypatch)
    main_window._on_delete_peaks_requested(sels)
    main_window.viewer.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 3

    main_window.viewer.redo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 0


def test_bulk_delete_cancel_keeps_rows(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Cancel in the confirmation dialog removes nothing."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_detected(main_window, 0, 3)
    _cancel(monkeypatch)
    main_window._on_delete_peaks_requested(sels)
    assert _n_detected(path, "entry_0000", 0) == 3


def test_bulk_delete_single_selection_is_noop(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """< 2 detected selected -> handler does nothing (the single-peak
    delete path owns that case)."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_detected(main_window, 0, 1)
    # question raising would prove we wrongly prompted for a single peak.
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not prompt for < 2 peaks"))),
    )
    main_window._on_delete_peaks_requested(sels)
    assert _n_detected(path, "entry_0000", 0) == 3


def test_delete_key_emits_bulk_signal_when_multi(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """The viewer routes Delete to the bulk signal for an all-detected
    multi-selection, and to the single signal otherwise."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer

    # The bulk signal is also wired to the real handler (from __init__),
    # which would pop a confirmation; Cancel it so the keypress just
    # exercises signal routing without deleting or blocking.
    _cancel(monkeypatch)

    bulk: list = []
    single: list = []
    v.deletePeaksRequested.connect(lambda s: bulk.append(s))
    v.deletePeakRequested.connect(lambda s: single.append(s))

    _select_detected(main_window, 0, 2)
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete,
                   Qt.KeyboardModifier.NoModifier)
    v.keyPressEvent(ev)
    assert len(bulk) == 1 and len(bulk[0]) == 2
    assert single == []


# --- single-peak delete is now undoable -----------------------------

def test_single_detected_delete_is_undoable(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """One detected peak deleted via the single path, then Ctrl+Z."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_detected(main_window, 0, 1)
    _yes(monkeypatch)
    main_window._on_delete_peak_requested(sels[0])
    assert _n_detected(path, "entry_0000", 0) == 2

    main_window.viewer.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 3


def test_single_fitted_delete_is_undoable(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """One fitted peak deleted via the single path, then Ctrl+Z."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sel = _select_fitted(main_window, 0, 1)[0]
    _yes(monkeypatch)
    main_window._on_delete_peak_requested(sel)
    assert _n_fitted(path, "entry_0000", 0) == 1

    main_window.viewer.undo_last_action()
    assert _n_fitted(path, "entry_0000", 0) == 2


# --- fitted multi-select + bulk delete ------------------------------

def test_bulk_delete_fitted_removes_and_undo(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """2 fitted selected + confirm -> both gone, detected untouched,
    one Ctrl+Z restores both."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    sels = _select_fitted(main_window, 0, 2)
    _yes(monkeypatch)
    assert _n_fitted(path, "entry_0000", 0) == 2

    main_window._on_delete_peaks_requested(sels)
    assert _n_fitted(path, "entry_0000", 0) == 0
    assert _n_detected(path, "entry_0000", 0) == 3  # no cross-kind cascade

    main_window.viewer.undo_last_action()
    assert _n_fitted(path, "entry_0000", 0) == 2


def test_fitted_delete_undo_preserves_fit_params(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Undo of a fitted delete restores the 2D-fit params (theta/A/B/C),
    not just the box geometry — proving the snapshot reads the full row,
    not the geometry-only SelectedPeak."""
    # Seed a fitted row with distinctive 2D params BEFORE open:
    # add_fitted_peak_row needs an r+ handle, which can't coexist with
    # silx holding the temp copy open. Seed the original fixture; the
    # session copies it on open so the row (and id) carry over.
    seeded_id = file_model.add_fitted_peak_row(
        synthetic_nexus_with_peaks, "entry_0000", 0,
        angle=33.0, angle_width=4.0, radius=2.2, radius_width=0.25,
        amplitude=77.0, theta=0.5, A=1.1, B=0.2, C=0.9, score=0.8,
    )
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    tables = main_window.viewer._frame_peaks.get(0) or {}
    fit = tables["fitted"]
    idx = list(int(x) for x in fit.ids).index(int(seeded_id))
    sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(seeded_id),
        radius=float(fit.radius[idx]), angle=float(fit.angle[idx]),
        radius_width=float(fit.radius_width[idx]),
        angle_width=float(fit.angle_width[idx]),
        is_ring=bool(fit.is_ring[idx]), score=float(fit.score[idx]),
        amplitude=float(fit.amplitude[idx]),
    )
    main_window.viewer._set_selected(sel)
    _yes(monkeypatch)
    main_window._on_delete_peak_requested(sel)
    main_window.viewer.undo_last_action()

    # The restored row has a fresh id; find it by its distinctive angle.
    rows = file_model.load_peaks(path, "entry_0000", 0)["fitted"]
    restored = [
        i for i in range(len(rows))
        if abs(float(rows.angle[i]) - 33.0) < 1e-3
    ]
    assert restored, "restored fitted row not found"
    ri = restored[0]
    full = file_model.read_peak_rows(
        path, "entry_0000", 0, "fitted", [int(rows.ids[ri])],
    )[0]
    assert full["theta"] == pytest.approx(0.5)
    assert full["A"] == pytest.approx(1.1)
    assert full["B"] == pytest.approx(0.2)
    assert full["C"] == pytest.approx(0.9)
    assert full["amplitude"] == pytest.approx(77.0)


# --- multi-level undo / redo across consecutive ops -----------------

def test_consecutive_deletes_undo_redo_all_levels(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Two separate single deletes must both be undoable AND redoable.

    Regression: each delete used to ``clear_history()`` before pushing
    its own action, so the undo stack never held more than the latest
    op -- a second Ctrl+Z was a no-op and redo only ever replayed the
    most recent delete. Now history accumulates: undo twice restores
    both peaks, redo twice removes both again.
    """
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    v = main_window.viewer
    _yes(monkeypatch)
    assert _n_detected(path, "entry_0000", 0) == 3

    # Delete the first detected peak, then (after the reload) the next.
    # _select_detected sets the viewer's primary selection as a side effect.
    sel1 = _select_detected(main_window, 0, 1)[0]
    main_window._on_delete_peak_requested(sel1)
    assert _n_detected(path, "entry_0000", 0) == 2
    sel2 = _select_detected(main_window, 0, 1)[0]
    main_window._on_delete_peak_requested(sel2)
    assert _n_detected(path, "entry_0000", 0) == 1
    assert len(v._undo_stack) == 2  # both ops on the stack, not just one

    v.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 2
    v.undo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 3  # multi-level undo

    v.redo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 2
    v.redo_last_action()
    assert _n_detected(path, "entry_0000", 0) == 1  # multi-level redo


def test_paste_then_delete_both_undoable(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """A paste followed by a bulk delete: undoing the delete then the
    paste walks back through both, proving append-only paste no longer
    wipes the paste's own (or any prior) undo entry."""
    _open(main_window, synthetic_nexus_with_peaks)
    path = main_window.session.temp_path
    v = main_window.viewer
    _yes(monkeypatch)

    # Copy the 3 detected, paste them back onto the same frame -> 6.
    _select_detected(main_window, 0, 3)
    main_window._on_copy_peaks()
    main_window._on_paste_peaks()
    assert _n_detected(path, "entry_0000", 0) == 6

    # Bulk-delete 3 of them -> 3, with the paste still on the stack.
    dels = _select_detected(main_window, 0, 3)
    main_window._on_delete_peaks_requested(dels)
    assert _n_detected(path, "entry_0000", 0) == 3
    assert len(v._undo_stack) == 2

    v.undo_last_action()  # undo delete -> 6
    assert _n_detected(path, "entry_0000", 0) == 6
    v.undo_last_action()  # undo paste -> 3
    assert _n_detected(path, "entry_0000", 0) == 3


def test_read_peak_rows_skips_missing_ids(synthetic_nexus_with_peaks):
    """read_peak_rows returns only the ids that exist, in request order."""
    rows = file_model.read_peak_rows(
        synthetic_nexus_with_peaks, "entry_0000", 0, "detected", [2, 999, 0],
    )
    # ids 2 and 0 exist (3 detected: 0,1,2); 999 doesn't.
    assert len(rows) == 2
    assert all("radius" in r and "angle" in r for r in rows)
