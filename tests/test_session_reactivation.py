"""Switching between open files restores the previous file instantly.

``_set_active_session`` stashes the outgoing NeXus session's live
``FrameSource`` (entry + warm handle) as that session's prewarm instead
of releasing it; re-activation reinstalls it from memory — no file
re-open, no frame re-read on the GUI thread (the old behaviour froze on
big/remote masters) — and lands on the entry the user was on, not back
at the first one. Stashes are released when their session closes.
Source: main_window.py ``_set_active_session`` / ``_populate_entries``
/ ``_close_session`` / ``closeEvent``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _nexus_file(tmp_path, name: str, n_entries: int):
    """Each entry's pixels are all float(i) so loads are identifiable."""
    path = tmp_path / name
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


def _viewer_mean(window) -> float:
    return float(np.mean(window.viewer._frame_source.get_cartesian(0)))


def test_switch_away_stashes_live_source(main_window, qtbot, tmp_path):
    a = _open(main_window, _nexus_file(tmp_path, "a.h5", 3))
    # Move A to a non-first entry so the stash records the user's place.
    main_window._ensure_entry_load_worker()
    with qtbot.waitSignal(main_window._entry_load_worker.loaded, timeout=30000):
        main_window.entry_combo.setCurrentText("entry_0002")
    qtbot.waitUntil(lambda: _viewer_mean(main_window) == 2.0, timeout=30000)
    src_a = main_window.viewer._frame_source

    _open(main_window, _nexus_file(tmp_path, "b.h5", 1))

    stash = getattr(a, "_prewarm", None)
    assert stash is not None
    assert stash[0] == "entry_0002" and stash[1] is src_a
    assert src_a.is_open  # parked, not released
    assert _viewer_mean(main_window) == 0.0  # B's entry_0000 is showing


def test_reactivation_restores_place_without_file_io(
    main_window, qtbot, tmp_path, monkeypatch
):
    a = _open(main_window, _nexus_file(tmp_path, "a.h5", 3))
    main_window._ensure_entry_load_worker()
    with qtbot.waitSignal(main_window._entry_load_worker.loaded, timeout=30000):
        main_window.entry_combo.setCurrentText("entry_0002")
    qtbot.waitUntil(lambda: _viewer_mean(main_window) == 2.0, timeout=30000)
    _open(main_window, _nexus_file(tmp_path, "b.h5", 1))

    def _boom(*args, **kwargs):
        raise AssertionError("re-activation must not re-open the file")

    monkeypatch.setattr(file_model, "load_entry", _boom)
    monkeypatch.setattr(file_model.FrameSource, "acquire", _boom)

    main_window._set_active_session(a)  # synchronous, from the stash

    assert main_window.entry_combo.currentText() == "entry_0002"
    assert _viewer_mean(main_window) == 2.0
    assert getattr(a, "_prewarm", None) is None  # consumed


def test_close_inactive_session_releases_stash(main_window, qtbot, tmp_path):
    a = _open(main_window, _nexus_file(tmp_path, "a.h5", 3))
    qtbot.waitUntil(
        lambda: main_window.viewer._frame_source is not None, timeout=30000
    )
    src_a = main_window.viewer._frame_source
    _open(main_window, _nexus_file(tmp_path, "b.h5", 1))
    assert src_a.is_open  # stashed on A

    main_window._close_session(a)

    assert not src_a.is_open
    assert getattr(a, "_prewarm", None) is None
