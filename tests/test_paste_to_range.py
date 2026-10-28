"""Paste detected peaks to a typed frame range (Ctrl+Shift+V).

Nine scenarios, all GUI but every dialog is monkeypatched away so the
tests run headless:

* Range expands to N frames; the clipboard is written to each one.
* Duplicates in the input dedup at parse time (frame written once).
* Out-of-range frames are filtered after a Yes-confirmation.
* No-confirmation aborts the paste (no writes, clipboard intact).
* One Ctrl+Z reverses the whole batch on every touched frame.
* QProgressDialog Cancel mid-loop keeps already-written frames.
* Empty clipboard is an early-return no-op.
* Different-entry clipboard is blocked at the take_items layer.
* The current-frame's pasted rows get added to the multi-selection;
  off-frame rows do not.

The synthetic fixture only seeds an analysis group on frame 0, so a
local ``_seed_analysis_groups`` helper materialises empty
detected/fitted datasets on the other frames before paste.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox, QProgressDialog

from mlgidlab import file_model, peak_clipboard
from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui


# Mirrors the conftest constant; redeclared so the test file is
# self-contained and doesn't depend on conftest being importable as
# a module.
_PYGID_PEAK_DTYPE = [
    ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
    ("radius", "f4"), ("radius_width", "f4"),
    ("q_z", "f4"), ("q_xy", "f4"), ("theta", "f4"),
    ("score", "f4"), ("A", "f4"), ("B", "f4"), ("C", "f4"),
    ("is_ring", "bool"), ("is_cut_qz", "bool"), ("is_cut_qxy", "bool"),
    ("visibility", "i4"), ("id", "i4"),
]


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _seed_analysis_groups(path, entry, frames) -> None:
    """Create empty detected/fitted datasets on each target frame so
    ``add_detected_peak_row`` doesn't KeyError on a missing group."""
    import h5py
    import numpy as np

    dt = np.dtype(_PYGID_PEAK_DTYPE)
    empty = np.zeros(0, dtype=dt)
    with h5py.File(path, "r+") as f:
        analysis = f[f"{entry}/data/analysis"]
        for frame in frames:
            key = f"frame{int(frame):05d}"
            if key in analysis:
                continue
            g = analysis.create_group(key, track_order=True)
            g.create_dataset("detected_peaks", data=empty)
            g.create_dataset("fitted_peaks", data=empty)
            g.create_dataset("fitted_peaks_errors", data=empty)


def _select_first_detected(window, frame: int) -> SelectedPeak:
    tables = window.viewer._frame_peaks.get(frame) or {}
    det = tables["detected"]
    sel = SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[0]),
        radius=float(det.radius[0]), angle=float(det.angle[0]),
        radius_width=float(det.radius_width[0]),
        angle_width=float(det.angle_width[0]),
        is_ring=bool(det.is_ring[0]),
        score=float(det.score[0]),
        amplitude=float(det.amplitude[0]),
    )
    window.viewer._set_selected(sel)
    return sel


def _copy_n_detected(window, frame: int, n: int) -> list[int]:
    """Multi-select the first N detected rows on ``frame`` and copy.
    Returns the source peak ids."""
    tables = window.viewer._frame_peaks.get(frame) or {}
    det = tables["detected"]
    primary = SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[0]),
        radius=float(det.radius[0]), angle=float(det.angle[0]),
        radius_width=float(det.radius_width[0]),
        angle_width=float(det.angle_width[0]),
        is_ring=bool(det.is_ring[0]),
        score=float(det.score[0]),
        amplitude=float(det.amplitude[0]),
    )
    window.viewer._set_selected(primary)
    extras = []
    for i in range(1, n):
        extras.append(SelectedPeak(
            kind="detected", frame=frame, peak_id=int(det.ids[i]),
            radius=float(det.radius[i]), angle=float(det.angle[i]),
            radius_width=float(det.radius_width[i]),
            angle_width=float(det.angle_width[i]),
            is_ring=bool(det.is_ring[i]),
            score=float(det.score[i]),
            amplitude=float(det.amplitude[i]),
        ))
    window.viewer._selected_extras = extras
    window._on_copy_peaks()
    return [int(det.ids[i]) for i in range(n)]


@pytest.fixture(autouse=True)
def _reset_clipboard():
    peak_clipboard.clear()
    yield
    peak_clipboard.clear()


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    """A modal ``QMessageBox.warning/critical`` calls ``exec()``, which
    spins a local event loop and waits for a button click. In headless
    / offscreen mode (local ``-q`` runs and CI alike) there is no
    clicker, so it blocks forever — the test never finishes and CI
    hangs. Convert the two fire-and-forget modals the paste handler can
    raise into loud failures so a mis-set test fails fast and visibly
    instead of hanging. (``question`` is patched per-test where a
    confirmation is expected.)"""
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"unexpected blocking QMessageBox in test: {args[1:3]!r}"
        )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _n_detected(path, entry, frame) -> int:
    table = file_model.load_peaks(path, entry, frame)["detected"]
    return 0 if table is None else len(table)


def test_paste_to_range_writes_one_clipboard_per_frame(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Range '0-2' writes the clipboard to each of frames 0, 1, 2."""
    # Seed BEFORE open: NexusSession.open copies the file to a temp
    # path and the GUI only touches that copy, so groups must exist in
    # the original before the copy is made. Seeding after open would
    # leave the temp copy's frames empty -> add_detected_peak_row
    # KeyError -> a modal warning that hangs headless.
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1, 2],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=2)
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)
    n1_before = _n_detected(main_window.session.temp_path, "entry_0000", 1)
    n2_before = _n_detected(main_window.session.temp_path, "entry_0000", 2)

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-2", True)),
    )
    main_window._on_paste_peaks_to_range()

    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before + 2
    assert _n_detected(main_window.session.temp_path, "entry_0000", 1) == n1_before + 2
    assert _n_detected(main_window.session.temp_path, "entry_0000", 2) == n2_before + 2


def test_paste_to_range_dedup(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Range '0,0-1' dedups to {0, 1} — frame 0 grows by exactly
    len(items), not 2 * len(items)."""
    # Seed BEFORE open (see note in the per-frame test above).
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=2)
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)
    n1_before = _n_detected(main_window.session.temp_path, "entry_0000", 1)

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0,0-1", True)),
    )
    main_window._on_paste_peaks_to_range()

    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before + 2
    assert _n_detected(main_window.session.temp_path, "entry_0000", 1) == n1_before + 2


def test_paste_to_range_filters_out_of_range(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Range '0-100' on a 3-frame stack: confirmation Yes filters to
    frames 0, 1, 2 only. Out-of-range frames neither write nor crash."""
    # Seed BEFORE open: NexusSession.open copies the file to a temp
    # path and the GUI only touches that copy, so groups must exist in
    # the original before the copy is made. Seeding after open would
    # leave the temp copy's frames empty -> add_detected_peak_row
    # KeyError -> a modal warning that hangs headless.
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1, 2],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=1)
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-100", True)),
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes),
    )
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)
    main_window._on_paste_peaks_to_range()
    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before + 1
    assert _n_detected(main_window.session.temp_path, "entry_0000", 1) == 1
    assert _n_detected(main_window.session.temp_path, "entry_0000", 2) == 1


def test_paste_to_range_cancel_user_no(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Out-of-range No keeps every frame untouched and leaves the
    clipboard intact (paste-twice still works)."""
    _open(main_window, synthetic_nexus_with_peaks)
    _copy_n_detected(main_window, 0, n=1)
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-100", True)),
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No),
    )
    main_window._on_paste_peaks_to_range()
    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before
    assert peak_clipboard.has_items() is True


def test_paste_to_range_undo_removes_all_appended(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """One Ctrl+Z reverses the whole range paste across every touched
    frame."""
    # Seed BEFORE open: NexusSession.open copies the file to a temp
    # path and the GUI only touches that copy, so groups must exist in
    # the original before the copy is made. Seeding after open would
    # leave the temp copy's frames empty -> add_detected_peak_row
    # KeyError -> a modal warning that hangs headless.
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1, 2],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=1)
    n_before = {
        f: _n_detected(main_window.session.temp_path, "entry_0000", f)
        for f in (0, 1, 2)
    }
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-2", True)),
    )
    main_window._on_paste_peaks_to_range()
    for f in (0, 1, 2):
        assert _n_detected(
            main_window.session.temp_path, "entry_0000", f,
        ) == n_before[f] + 1

    main_window.viewer.undo_last_action()
    for f in (0, 1, 2):
        assert _n_detected(
            main_window.session.temp_path, "entry_0000", f,
        ) == n_before[f]


def test_paste_to_range_progress_cancel_keeps_partial(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Cancel after the first frame's setValue keeps frame 0's rows
    but not frames 1 or 2."""
    # Seed BEFORE open: NexusSession.open copies the file to a temp
    # path and the GUI only touches that copy, so groups must exist in
    # the original before the copy is made. Seeding after open would
    # leave the temp copy's frames empty -> add_detected_peak_row
    # KeyError -> a modal warning that hangs headless.
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1, 2],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=1)
    n_before = {
        f: _n_detected(main_window.session.temp_path, "entry_0000", f)
        for f in (0, 1, 2)
    }
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-2", True)),
    )

    state = {"steps_done": 0}
    original_setValue = QProgressDialog.setValue

    def _setValue(self, v):
        state["steps_done"] = v
        original_setValue(self, v)

    def _wasCanceled(self):
        return state["steps_done"] >= 1

    monkeypatch.setattr(QProgressDialog, "setValue", _setValue)
    monkeypatch.setattr(QProgressDialog, "wasCanceled", _wasCanceled)

    main_window._on_paste_peaks_to_range()

    # Frame 0 written before the cancel kicked in; frames 1, 2 untouched.
    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n_before[0] + 1
    assert _n_detected(main_window.session.temp_path, "entry_0000", 1) == n_before[1]
    assert _n_detected(main_window.session.temp_path, "entry_0000", 2) == n_before[2]


def test_paste_to_range_blocked_when_clipboard_empty(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Empty clipboard is an early-return: no dialog, no writes."""
    _open(main_window, synthetic_nexus_with_peaks)
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)

    called = {"input_dialog": False}

    def _fail_dialog(*a, **kw):
        called["input_dialog"] = True
        return ("0-2", True)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(_fail_dialog),
    )
    main_window._on_paste_peaks_to_range()
    assert called["input_dialog"] is False
    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before


def test_paste_to_range_blocked_on_different_entry(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Clipboard scoped to another entry is invisible to the current
    entry's paste — handler hits the empty-take_items branch."""
    _open(main_window, synthetic_nexus_with_peaks)
    peak_clipboard.set_items(
        [peak_clipboard.ClipboardItem(
            radius=1.0, angle=20.0, radius_width=0.2, angle_width=5.0,
            is_ring=False, source_frame=0, source_peak_id=0,
        )],
        entry="some_other_entry",
    )
    n0_before = _n_detected(main_window.session.temp_path, "entry_0000", 0)

    called = {"input_dialog": False}

    def _fail_dialog(*a, **kw):
        called["input_dialog"] = True
        return ("0-2", True)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(_fail_dialog),
    )
    main_window._on_paste_peaks_to_range()
    assert called["input_dialog"] is False
    assert _n_detected(main_window.session.temp_path, "entry_0000", 0) == n0_before


def test_paste_to_range_selection_only_for_current_frame(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """Current frame = 2; paste range '0-2' adds only frame 2's pasted
    rows to the multi-selection. The off-frame rows on 0 and 1 are
    not in selected_peaks()."""
    # Seed BEFORE open: NexusSession.open copies the file to a temp
    # path and the GUI only touches that copy, so groups must exist in
    # the original before the copy is made. Seeding after open would
    # leave the temp copy's frames empty -> add_detected_peak_row
    # KeyError -> a modal warning that hangs headless.
    _seed_analysis_groups(
        synthetic_nexus_with_peaks, "entry_0000", [1, 2],
    )
    _open(main_window, synthetic_nexus_with_peaks)

    _copy_n_detected(main_window, 0, n=1)
    # Clear selection so post-paste sel count reflects only the new
    # rows landing on the current frame.
    main_window.viewer.clear_selection()
    main_window.viewer.set_frame(2)
    assert int(main_window.viewer.current_frame) == 2

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-2", True)),
    )
    main_window._on_paste_peaks_to_range()

    sels = main_window.viewer.selected_peaks()
    # Exactly the 1 row pasted to frame 2.
    assert len(sels) == 1
    assert sels[0].kind == "detected"
    assert int(sels[0].frame) == 2


def test_paste_to_range_leaves_no_progress_dialog_behind(
    main_window, synthetic_nexus_with_peaks, monkeypatch,
):
    """The progress dialog is parented to the window; once the paste is
    done it has to go, or every paste leaves one behind."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QProgressDialog

    _seed_analysis_groups(synthetic_nexus_with_peaks, "entry_0000", [1])
    _open(main_window, synthetic_nexus_with_peaks)
    _copy_n_detected(main_window, 0, n=2)
    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *a, **kw: ("0-1", True)),
    )
    before = len(main_window.findChildren(QProgressDialog))
    main_window._on_paste_peaks_to_range()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(main_window.findChildren(QProgressDialog)) == before
