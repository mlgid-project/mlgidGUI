"""``_pipeline_queue`` must carry an explicit per-command file_path
snapshotted at enqueue time, so a mid-queue active-session switch
can't dispatch later commands against the wrong file.

Regression: with two files loaded, hitting "All entries" → Run on
file A expanded into N commands but each command read
``self.session.temp_path`` fresh at dispatch. If the active session
flipped to file B between commands, later commands dispatched
against B, and the entry pre-flight tripped with a misleading
message naming B's available entries.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mlgidgui.pipeline import PipelineCommand
from mlgidgui.session import NexusSession


def _make_file(path: Path, entries: list[str], n_frames: int = 2) -> Path:
    """Write a minimal valid NeXus file with the listed ``img_gid_q``
    entries."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        for entry_name in entries:
            d = f.create_group(f"{entry_name}/data", track_order=True)
            d.attrs["signal"] = "img_gid_q"
            d.create_dataset(
                "img_gid_q",
                data=rng.random((n_frames, 8, 8), dtype=np.float32),
            )
            d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
            d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
    return path


def _open(window, path) -> NexusSession:
    """Mirror ``_on_open_finished``: append the session and activate it."""
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_queue_snapshots_path_per_command(main_window, tmp_path):
    """The queue must store ``(file_path, command)`` tuples whose
    path is fixed at expansion time. Switching the active session
    after expansion does not change the snapshotted paths."""
    file_a = _make_file(
        tmp_path / "file_a.h5", ["entry_4P", "entry_C60", "entry_ZnPc"]
    )
    file_b = _make_file(tmp_path / "file_b.h5", ["entry"])
    session_a = _open(main_window, file_a)
    session_b = _open(main_window, file_b)
    # session_b is active after the second _open — but the user is
    # about to expand "All entries" on session_a, which they would
    # do by first re-activating it.
    main_window._set_active_session(session_a)
    assert main_window.session is session_a

    # Fire the panel's "All entries" runRequested by hand: a
    # PipelineCommand with no entry kwarg.
    cmd = PipelineCommand("run_detection", {})

    # Capture the queue before the first dequeue fires so we can
    # inspect every tuple. The first command will pop immediately
    # via _enqueue_pipeline's auto-start, but only after all N
    # have been appended — main_window._pipe_thread is None at the
    # top of _on_run_requested but the dispatcher only starts the
    # first one once the queue is non-empty. To inspect, prevent
    # the auto-dispatch by parking _pipe_thread to a sentinel.
    main_window._pipe_thread = object()  # block _run_next_pipeline_command
    try:
        main_window._on_run_requested(cmd)
        queue = list(main_window._pipeline_queue)
    finally:
        main_window._pipe_thread = None
        main_window._pipeline_queue.clear()

    # Every queued tuple should carry session_a's temp_path.
    assert len(queue) == 3
    for fp, queued_cmd in queue:
        assert fp == session_a.temp_path, (
            f"queued command {queued_cmd.kwargs!r} captured {fp!r}, "
            f"expected {session_a.temp_path!r}"
        )
        assert queued_cmd.kwargs["entry"] in {"entry_4P", "entry_C60", "entry_ZnPc"}

    # Now flip the active session. The snapshotted paths must still
    # point at session_a (the file we expanded against), not at the
    # newly-active session_b.
    main_window._set_active_session(session_b)
    assert main_window.session is session_b
    # Reuse the inspection: re-enqueue (a real run on the now-active
    # session_b should target session_b, of course — distinct from
    # the previous queue which we cleared above).
    main_window._pipe_thread = object()
    try:
        main_window._on_run_requested(PipelineCommand("run_detection", {}))
        queue_b = list(main_window._pipeline_queue)
    finally:
        main_window._pipe_thread = None
        main_window._pipeline_queue.clear()

    # File B has only one img_gid_q entry → exactly one tuple, pointing at file_b.
    assert len(queue_b) == 1
    fp_b, cmd_b = queue_b[0]
    assert fp_b == session_b.temp_path
    assert cmd_b.kwargs.get("entry") == "entry"
