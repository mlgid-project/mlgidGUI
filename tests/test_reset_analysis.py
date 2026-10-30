"""Regression: Tools -> Reset all peaks must not hang on a multi-frame file.

History: an early build hung when Reset -> Active frame / All entries
was clicked on a multi-frame file (flagged in presentation/demo_script.md
as a landmine to avoid). The cause was the background prefetch worker
holding its thread's event loop in a long blocking prefetch loop, so the
``BlockingQueuedConnection`` release in ``_detach_silx_tree`` could never
be serviced -> GUI thread blocked forever. The worker was later rewritten
to prefetch ONE frame per timer tick, yielding to its event loop between
frames (workers.py:_tick), which lets the blocking release complete
promptly. The hang no longer reproduces.

These tests lock that in: with the prefetch worker spawned AND active
(the multi-frame condition the old hang needed), every reset scope
completes and actually clears the peaks. A faulthandler watchdog turns
a future regression into a fast, stack-dumping failure instead of a
silent CI hang.
"""
from __future__ import annotations

import faulthandler

import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _hang_watchdog():
    """Dump every thread's stack and abort if a test runs > 30 s.

    A reset deadlock would otherwise hang the whole suite (and CI)
    indefinitely; this converts it into a visible, debuggable failure.
    """
    faulthandler.dump_traceback_later(30, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _n_detected(path, entry, frame) -> int:
    table = file_model.load_peaks(path, entry, frame)["detected"]
    return 0 if table is None else len(table)


def _activate_prefetcher(window, qtbot) -> None:
    """Spawn + activate the prefetch worker so it is actually ticking.

    On a multi-frame load the worker is created but paused; driving it
    active reproduces the exact multi-frame condition the old hang
    needed (a live worker thread the blocking release must coordinate
    with)."""
    assert window.viewer.n_frames > 1
    window._ensure_prefetch_worker()
    # Configure for the active entry, then mark active so the internal
    # QTimer starts ticking on the worker's thread.
    window._configure_prefetch_for_active_entry()
    window._prefetchUpdate.emit(0, True, 1)
    qtbot.wait(80)  # let the worker tick a few frames on its own thread


@pytest.mark.parametrize("scope", ["frame", "entry", "all"])
def test_reset_does_not_hang_and_clears(
    main_window, synthetic_nexus_with_peaks, monkeypatch, qtbot, scope,
):
    _open(main_window, synthetic_nexus_with_peaks)
    _activate_prefetcher(main_window, qtbot)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    path = main_window.session.temp_path
    assert _n_detected(path, "entry_0000", 0) > 0

    # Would block forever here if the prefetch/detach deadlock regressed;
    # the watchdog fixture bounds it to 30 s.
    main_window._action_reset_analysis(scope)

    assert _n_detected(path, "entry_0000", 0) == 0


def test_clear_dialogs_default_to_yes(
    main_window, synthetic_nexus_with_peaks, monkeypatch
):
    """The Clear-peaks / Reset-all confirmations default to Yes, so a
    single Enter confirms. Captures the ``defaultButton`` passed to
    ``QMessageBox.question`` (signature: parent, title, text, buttons,
    defaultButton) and returns Cancel so the action no-ops after."""
    _open(main_window, synthetic_nexus_with_peaks)
    captured: list = []

    def _rec(*a, **k):
        captured.append(a[4] if len(a) > 4 else k.get("defaultButton"))
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_rec))

    main_window._action_reset_analysis("all")     # Reset all peaks
    main_window._confirm_clear("detected", "")     # Clear detected

    assert captured == [
        QMessageBox.StandardButton.Yes,
        QMessageBox.StandardButton.Yes,
    ]


def test_delete_dialog_defaults_to_yes(
    main_window, synthetic_nexus_with_peaks, monkeypatch
):
    """The peak-delete confirmation defaults to Yes too (one Enter
    confirms). Drives ``_delete_peaks_scoped`` with a detected selection
    on frame 0 and returns Cancel so nothing is actually removed."""
    from mlgidgui.image_viewer import SelectedPeak

    _open(main_window, synthetic_nexus_with_peaks)
    captured: list = []

    def _rec(*a, **k):
        captured.append(a[4] if len(a) > 4 else k.get("defaultButton"))
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_rec))

    sel = SelectedPeak(
        kind="detected", frame=0, peak_id=0,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    )
    main_window._delete_peaks_scoped([sel], "entry_0000")

    assert captured == [QMessageBox.StandardButton.Yes]
