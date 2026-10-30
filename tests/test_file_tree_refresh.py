"""File-browser Refresh (button / F5): re-sync open sessions with disk.

Deleted originals close their session (temp cleaned up) unless dirty —
dirty sessions stay open with a warning so Save As can rescue the
edits. Changed-on-disk originals reload into the temp copy when clean;
dirty ones are left untouched and reported as conflicts. Raw sessions
close when every input file is gone. Source: session.py
``_disk_signature`` / ``disk_changed`` / ``reload_from_disk``;
main_window.py ``_refresh_file_tree`` / ``_reload_session_from_disk``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui.session import NexusSession, RawSession

pytestmark = pytest.mark.gui


def _nexus_file(path, n_entries):
    with h5py.File(path, "w", track_order=True) as f:
        for i in range(n_entries):
            data = f.create_group(f"entry_{i:04d}/data", track_order=True)
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q", data=np.full((2, 8, 8), float(i), np.float32)
            )
            data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
            data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
    return path


def _open(window, path):
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def _silence_warnings(monkeypatch):
    captured: list[tuple] = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: captured.append(a))
    )
    return captured


# -- session-level primitives -------------------------------------------


def test_disk_changed_and_reload(tmp_path):
    original = _nexus_file(tmp_path / "a.h5", 2)
    session = NexusSession.open(original)
    try:
        assert not session.disk_changed()

        _nexus_file(original, 3)  # rewrite with different size
        assert session.disk_changed()

        session.reload_from_disk()
        assert not session.disk_changed()
        assert session.temp_path.read_bytes() == original.read_bytes()
    finally:
        session.close()


def test_save_refreshes_disk_baseline(tmp_path):
    original = _nexus_file(tmp_path / "a.h5", 2)
    session = NexusSession.open(original)
    try:
        with h5py.File(session.temp_path, "r+") as f:
            f.attrs["edited"] = 1
        session.mark_dirty()
        session.save()
        # Save rewrote the original itself — that's not "changed on disk".
        assert not session.disk_changed()
    finally:
        session.close()


# -- window-level refresh ------------------------------------------------


def test_refresh_closes_deleted_clean_session(main_window, tmp_path, monkeypatch):
    _silence_warnings(monkeypatch)
    original = _nexus_file(tmp_path / "a.h5", 2)
    session = _open(main_window, original)
    temp = session.temp_path

    original.unlink()
    main_window._refresh_file_tree()

    assert session not in main_window._sessions
    assert not temp.exists()  # temp copy cleaned up


def test_refresh_keeps_deleted_dirty_session_with_warning(
    main_window, tmp_path, monkeypatch
):
    captured = _silence_warnings(monkeypatch)
    original = _nexus_file(tmp_path / "a.h5", 2)
    session = _open(main_window, original)
    session.mark_dirty()

    original.unlink()
    main_window._refresh_file_tree()

    assert session in main_window._sessions  # kept for Save As rescue
    assert captured and "a.h5" in captured[0][2]
    # Clean up: a dirty session left behind would block fixture teardown
    # on the modal save-confirm dialog.
    session.dirty = False


def test_refresh_reloads_changed_clean_session(main_window, tmp_path, monkeypatch):
    _silence_warnings(monkeypatch)
    original = _nexus_file(tmp_path / "a.h5", 3)
    session = _open(main_window, original)
    assert main_window.entry_combo.count() == 3

    _nexus_file(original, 1)  # shrink to one entry on disk
    main_window._refresh_file_tree()

    assert session in main_window._sessions
    assert not session.disk_changed()
    assert main_window.entry_combo.count() == 1  # viewer state rebuilt
    assert session.temp_path.read_bytes() == original.read_bytes()


def test_refresh_leaves_changed_dirty_session_alone(
    main_window, tmp_path, monkeypatch
):
    captured = _silence_warnings(monkeypatch)
    original = _nexus_file(tmp_path / "a.h5", 3)
    session = _open(main_window, original)
    session.mark_dirty()
    before = session.temp_path.read_bytes()

    _nexus_file(original, 1)
    main_window._refresh_file_tree()

    assert session in main_window._sessions
    assert session.temp_path.read_bytes() == before  # working copy untouched
    assert captured and "a.h5" in captured[0][2]
    assert main_window.entry_combo.count() == 3
    # Clean up: a dirty session left behind would block fixture teardown
    # on the modal save-confirm dialog.
    session.dirty = False


def test_refresh_closes_raw_session_when_all_inputs_gone(
    main_window, synthetic_raw, monkeypatch
):
    _silence_warnings(monkeypatch)
    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)

    synthetic_raw.unlink()
    main_window._refresh_file_tree()

    assert session not in main_window._sessions
