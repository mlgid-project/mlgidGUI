"""Async entry switching: the entry's frame is read off the GUI thread.

Switching entries (combo or file-browser click) on a master that links
external scans must not block the GUI on a slow network frame read.
``EntryLoadWorker`` opens + warms the entry's first frame on a worker
thread and hands the GUI a ready ``FrameSource``; ``_on_entry_loaded``
installs it and drops stale results from superseded switches. Source:
workers.py ``EntryLoadWorker``; main_window.py ``_load_entry_async`` /
``_on_entry_loaded`` / ``_install_stack_into_viewer``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession
from mlgidgui.workers import EntryLoadWorker

pytestmark = pytest.mark.gui


def _multi_entry(tmp_path, n=3):
    """A single NeXus file with ``n`` q-image entries; entry i's pixels are
    all ``float(i)`` so a load can be identified by its mean."""
    path = tmp_path / "multi.h5"
    with h5py.File(path, "w", track_order=True) as f:
        for i in range(n):
            data = f.create_group(f"entry_{i:04d}/data", track_order=True)
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q", data=np.full((2, 8, 8), float(i), np.float32)
            )
            data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
            data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
    return path


def _open(window, tmp_path):
    session = NexusSession.open(_multi_entry(tmp_path))
    window._on_open_finished(
        {
            "path": session.temp_path,
            "kind": "nexus",
            "session": session,
            "prewarm": None,
            "entries": [f"entry_{i:04d}" for i in range(3)],
            "error": None,
        }
    )
    return session


def test_entry_load_worker_emits_ready_source(qtbot, tmp_path):
    """The worker opens the requested entry and hands back a FrameSource
    with frame 0 already readable; the request_id round-trips."""
    worker = EntryLoadWorker()
    got: dict = {}
    worker.loaded.connect(
        lambda rid, e, s, o: got.update(rid=rid, entry=e, source=s, overlays=o)
    )
    worker.load(str(_multi_entry(tmp_path)), "entry_0001", 7)  # direct call, no thread

    assert got["rid"] == 7 and got["entry"] == "entry_0001"
    src = got["source"]
    assert src is not None and src.is_open
    assert float(np.mean(src.get_cartesian(0))) == 1.0
    # Overlays for frame 0 were read off-thread too (no peaks here → empties).
    assert got["overlays"][0] == 0
    src.release()


def test_entry_load_worker_bad_entry_emits_none(qtbot, tmp_path):
    """A missing / unreadable entry resolves to ``source=None`` (the GUI
    warns) rather than raising on the worker thread."""
    worker = EntryLoadWorker()
    got: dict = {}
    worker.loaded.connect(lambda rid, e, s, o: got.update(source=s, overlays=o))
    worker.load(str(_multi_entry(tmp_path)), "entry_9999", 1)
    assert got["source"] is None
    assert got["overlays"] is None


def test_on_entry_loaded_drops_stale_result(main_window, tmp_path):
    """A result whose request_id is older than the current one (the user
    switched again) is released, not installed — the viewer stays put."""
    session = _open(main_window, tmp_path)
    assert main_window.entry_combo.currentText() == "entry_0000"

    src = file_model.FrameSource(file_path=session.temp_path, entry="entry_0002")
    src.acquire()
    src.get_cartesian(0)
    stale = main_window._entry_req_id - 1

    main_window._on_entry_loaded(stale, "entry_0002", src)

    assert not src.is_open  # released
    assert main_window.entry_combo.currentText() == "entry_0000"  # unchanged


def test_on_entry_loaded_installs_overlays_without_gui_read(
    main_window, tmp_path, monkeypatch
):
    """When the worker supplies the frame's overlays, installing the entry
    does NO peak read on the GUI thread — that read is an SFTP round-trip
    on a remote master and was the residual per-switch freeze."""
    session = _open(main_window, tmp_path)
    src = file_model.FrameSource(file_path=session.temp_path, entry="entry_0002")
    src.acquire()
    src.get_cartesian(0)
    overlays = (0, {"detected": None, "fitted": None}, [])

    def _boom(*a, **k):
        raise AssertionError("no GUI-thread peak read when overlays provided")

    monkeypatch.setattr(file_model, "load_peaks", _boom)
    monkeypatch.setattr(file_model, "read_peaks", _boom)

    main_window._on_entry_loaded(
        main_window._entry_req_id, "entry_0002", src, overlays
    )  # must not raise

    assert main_window._loaded_peak_frames == {0}
    assert float(np.mean(main_window.viewer._frame_source.get_cartesian(0))) == 2.0


def test_combo_switch_renders_via_worker(main_window, qtbot, tmp_path):
    """An interactive combo switch dispatches to the worker and, once it
    returns, the viewer shows the new entry (mean 2.0 for entry_0002)."""
    _open(main_window, tmp_path)
    main_window._ensure_entry_load_worker()

    with qtbot.waitSignal(main_window._entry_load_worker.loaded, timeout=30000):
        main_window.entry_combo.setCurrentText("entry_0002")  # -> async load

    qtbot.waitUntil(
        lambda: main_window.viewer._frame_source is not None
        and float(np.mean(main_window.viewer._frame_source.get_cartesian(0))) == 2.0,
        timeout=30000,
    )
