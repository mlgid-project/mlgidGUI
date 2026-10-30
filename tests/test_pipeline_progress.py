"""Per-frame progress wiring for ``PipelineWorker`` + ``PipelinePanel``.

mlgidbase emits ``Saved <kind> peaks to file: ..., entry: <entry>,
frame: <N>`` once per frame at completion time — or a skip line
("No solutions for (...)", "No peaks detected for file: ...",
"Detection failed. ...") when the frame produced nothing. The worker
turns both into structured ``frameProgress(done, total, op, entry)``
emits via ``_FrameProgressHandler``; the panel paints a determinate
``QProgressBar`` for multi-frame runs and stays hidden otherwise.
These tests exercise the contract without driving a real mlgidbase
pipeline run.
"""
from __future__ import annotations

import logging

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal

from mlgidgui import file_model
from mlgidgui.workers import _FRAME_DONE_RE, _FrameProgressHandler


class _ProgressSink(QObject):
    """Minimal QObject whose Signal can be driven from a handler.

    Mirrors the contract of ``PipelineWorker.frameProgress`` so the
    handler can emit into something we can spy on without standing
    up the whole worker.
    """

    frameProgress = Signal(int, int, str, str)


def _record(msg: str) -> logging.LogRecord:
    """Build a vanilla INFO log record carrying ``msg``."""
    return logging.LogRecord(
        name="mlgidBASE",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_frame_done_regex_matches_each_kind():
    """The regex must catch detection / fitting / matching shapes —
    all three follow the same string template in mlgidbase."""
    for kind in ("detected", "fitted", "matched"):
        msg = f"Saved {kind} peaks to file: /tmp/foo.h5, entry: entry_0000, frame: 3"
        m = _FRAME_DONE_RE.search(msg)
        assert m is not None, f"{kind} line should match"
        assert m.group("kind") == kind
        assert m.group("entry") == "entry_0000"
        assert m.group("frame") == "3"


def test_progress_handler_counts_each_completion(qtbot):
    """Each matching log line bumps ``done`` by 1 and emits
    ``frameProgress``. Non-matching records are ignored."""
    sink = _ProgressSink()
    handler = _FrameProgressHandler(sink.frameProgress, total=5, op_name="run_detection")

    received: list[tuple[int, int, str, str]] = []
    sink.frameProgress.connect(lambda d, t, o, e: received.append((d, t, o, e)))

    # Three frame-completion records and one unrelated one in the middle.
    handler.emit(_record("Saved detected peaks to file: /tmp/a.h5, entry: entry_0, frame: 0"))
    handler.emit(_record("Something unrelated logged from pygid"))
    handler.emit(_record("Saved detected peaks to file: /tmp/a.h5, entry: entry_0, frame: 1"))
    handler.emit(_record("Saved detected peaks to file: /tmp/a.h5, entry: entry_0, frame: 2"))

    assert received == [
        (1, 5, "run_detection", "entry_0"),
        (2, 5, "run_detection", "entry_0"),
        (3, 5, "run_detection", "entry_0"),
    ]


def test_progress_handler_counts_matching_no_solution_frames(qtbot):
    """A matching frame with no solutions logs "No solutions for (…)"
    instead of "Saved matched peaks" and must still tick the bar —
    regression for the bar lagging the log and then jumping to 100%
    at the end of a run with no-solution frames."""
    sink = _ProgressSink()
    handler = _FrameProgressHandler(sink.frameProgress, total=4, op_name="run_matching")

    received: list[tuple[int, int, str, str]] = []
    sink.frameProgress.connect(lambda d, t, o, e: received.append((d, t, o, e)))

    handler.emit(_record("Saved matched peaks to file: /tmp/a.h5, entry: entry_0, frame: 0"))
    handler.emit(_record(
        "No solutions for (/tmp/a.h5, entry: entry_0, frame: 1) was found. "
        "Try to decrease threshold"
    ))
    handler.emit(_record("Saved matched peaks to file: /tmp/a.h5, entry: entry_0, frame: 2"))
    handler.emit(_record(
        "No solutions for (/tmp/a.h5, entry: entry_0, frame: 3) was found. "
        "Try to decrease threshold"
    ))

    assert received == [
        (1, 4, "run_matching", "entry_0"),
        (2, 4, "run_matching", "entry_0"),
        (3, 4, "run_matching", "entry_0"),
        (4, 4, "run_matching", "entry_0"),
    ]


def test_progress_handler_counts_detection_skip_frames(qtbot):
    """Detection's two skip shapes both tick: "No peaks detected"
    (carries entry/frame) and the bare "Detection failed. …" warning
    (no entry in the text — the last-seen entry is reused for the
    label)."""
    sink = _ProgressSink()
    handler = _FrameProgressHandler(sink.frameProgress, total=3, op_name="run_detection")

    received: list[tuple[int, int, str, str]] = []
    sink.frameProgress.connect(lambda d, t, o, e: received.append((d, t, o, e)))

    handler.emit(_record("Saved detected peaks to file: /tmp/a.h5, entry: entry_0, frame: 0"))
    handler.emit(_record("No peaks detected for file: /tmp/a.h5, entry: entry_0, frame: 1"))
    handler.emit(_record("Detection failed. Couldn't detect any peaks"))

    assert received == [
        (1, 3, "run_detection", "entry_0"),
        (2, 3, "run_detection", "entry_0"),
        (3, 3, "run_detection", "entry_0"),
    ]


def test_progress_handler_ignores_skip_lookalikes(qtbot):
    """Log lines that resemble skip lines but do not mark a frame
    boundary must not tick: matching's from-memory variant (no
    "for (" — the GUI never drives that path but the record could
    appear) and mlgidbase's "No fitted peaks" hint."""
    sink = _ProgressSink()
    handler = _FrameProgressHandler(sink.frameProgress, total=3, op_name="run_matching")

    received: list[int] = []
    sink.frameProgress.connect(lambda d, t, o, e: received.append(d))

    handler.emit(_record("No solutions was found. Try to decrease the threshold"))
    handler.emit(_record("No fitted peaks. Use run_fitting() first."))
    handler.emit(_record("No fitted peak 7 for entry entry_0; frame_num 2"))

    assert received == []


def test_progress_handler_caps_at_total(qtbot):
    """If mlgidbase logs more completion lines than we pre-counted
    (multi-entry scope where the pre-count missed an entry, etc.),
    the emitted ``done`` is clamped at ``total`` so the panel UI
    never has to deal with a value past max."""
    sink = _ProgressSink()
    handler = _FrameProgressHandler(sink.frameProgress, total=2, op_name="run_fitting")

    received: list[int] = []
    sink.frameProgress.connect(lambda d, t, o, e: received.append(d))

    for i in range(4):
        handler.emit(_record(f"Saved fitted peaks to file: /tmp/a.h5, entry: e, frame: {i}"))

    # Internal counter went to 4 but the emitted ``done`` never exceeds 2.
    assert received == [1, 2, 2, 2]


def test_count_frames_returns_zero_on_missing_entry(synthetic_nexus):
    """``count_frames`` is best-effort: it must return 0 when the
    requested entry isn't present rather than raising."""
    assert file_model.count_frames(synthetic_nexus, "not_a_real_entry") == 0


def test_count_frames_returns_shape_for_valid_entry(synthetic_nexus):
    """For an entry the fixture seeds with 3 frames, ``count_frames``
    returns 3 without loading pixel data — used by the worker to
    size the progress bar before invoking mlgidBASE."""
    assert file_model.count_frames(synthetic_nexus, "entry_0000") == 3


def test_count_frames_sums_across_multi_entry_file(tmp_path):
    """Multi-entry scope sums n_frames per entry — synthesise a file
    with two img_gid_q entries of size 4 and 2 and confirm the
    worker-side total resolver would yield 6."""
    path = tmp_path / "multi.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        for entry_name, n_frames in (("entry_a", 4), ("entry_b", 2)):
            d = f.create_group(f"{entry_name}/data", track_order=True)
            d.attrs["signal"] = "img_gid_q"
            d.create_dataset(
                "img_gid_q",
                data=rng.random((n_frames, 8, 8), dtype=np.float32),
            )
            d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
            d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
    entries = file_model.list_entries(path)
    total = sum(file_model.count_frames(path, e) for e in entries)
    assert total == 6


def test_panel_hides_progress_on_single_frame_run(main_window, qtbot):
    """The panel slot's policy: ``total <= 1`` keeps the bar
    hidden. Single-frame runs should contribute zero UI noise."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    # Hand-drive the slot the worker would normally invoke.
    panel.on_frame_progress(done=0, total=1, op_name="run_detection", entry="entry_0000")
    assert not panel._progress_bar.isVisible()
    assert not panel._progress_label.isVisible()


def test_panel_shows_progress_on_multi_frame_run(main_window, qtbot):
    """Multi-frame runs paint the bar and the labelled counter."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    # The panel must be visible for child visibility to evaluate
    # correctly — show it explicitly since the harness uses qtbot.addWidget
    # which doesn't auto-show child widgets nested in scroll areas.
    main_window.show()
    qtbot.waitExposed(main_window)
    panel.on_frame_progress(done=3, total=12, op_name="run_detection", entry="entry_0000")
    assert panel._progress_bar.isVisible()
    assert panel._progress_label.isVisible()
    assert panel._progress_bar.maximum() == 12
    assert panel._progress_bar.value() == 3
    assert "Detection" in panel._progress_label.text()
    assert "entry_0000" in panel._progress_label.text()
    assert "3/12" in panel._progress_label.text()


def test_panel_shows_busy_bar_for_track_peaks(main_window, qtbot):
    """track_peaks is one opaque call (file open + IoU/graph dominate,
    per-frame reads are ~3%), so its progress row is an indeterminate
    busy bar (range 0..0) rather than a misleading frame counter. A
    later determinate run restores the normal bar."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    main_window.show()
    qtbot.waitExposed(main_window)
    # total is passed (frame count) but ignored — the bar goes busy.
    panel.on_frame_progress(done=0, total=310, op_name="track_peaks", entry="scan")
    assert panel._progress_bar.isVisible()
    assert panel._progress_bar.minimum() == 0 and panel._progress_bar.maximum() == 0
    assert "Tracking" in panel._progress_label.text()
    # Later ticks don't disturb the busy bar.
    panel.on_frame_progress(done=200, total=310, op_name="track_peaks", entry="scan")
    assert panel._progress_bar.maximum() == 0
    # A subsequent determinate run restores a normal counter.
    panel.set_running(False)
    panel.on_frame_progress(done=3, total=12, op_name="run_detection", entry="e")
    assert panel._progress_bar.maximum() == 12
    assert panel._progress_bar.value() == 3


def test_panel_hides_progress_when_set_running_false(main_window, qtbot):
    """``set_running(False)`` is the cleanup hook the host calls when
    the queue drains; the progress row must hide regardless of the
    last ``on_frame_progress`` state."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    main_window.show()
    qtbot.waitExposed(main_window)
    panel.on_frame_progress(done=5, total=10, op_name="run_fitting", entry="entry_0000")
    assert panel._progress_bar.isVisible()
    panel.set_running(False)
    assert not panel._progress_bar.isVisible()
    assert not panel._progress_label.isVisible()


def test_panel_shows_entry_progress_on_multi_entry_queue(main_window, qtbot):
    """``on_queue_progress(current, total > 1, …)`` paints the
    entry-level bar above the frame bar."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    main_window.show()
    qtbot.waitExposed(main_window)
    panel.on_queue_progress(
        current=3, total=8, entry="entry_DBTTF", op_name="run_detection",
    )
    assert panel._entry_progress_bar.isVisible()
    assert panel._entry_progress_label.isVisible()
    assert panel._entry_progress_bar.maximum() == 8
    assert panel._entry_progress_bar.value() == 3
    label = panel._entry_progress_label.text()
    assert "Detection" in label
    assert "entry 3/8" in label
    assert "entry_DBTTF" in label


def test_panel_hides_entry_progress_on_single_entry_run(main_window, qtbot):
    """Single-entry runs (``total <= 1``) must keep the entry bar
    hidden — same policy as the frame bar's single-frame guard."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    main_window.show()
    qtbot.waitExposed(main_window)
    panel.on_queue_progress(
        current=1, total=1, entry="entry_0000", op_name="run_detection",
    )
    assert not panel._entry_progress_bar.isVisible()
    assert not panel._entry_progress_label.isVisible()


def test_panel_hides_both_bars_when_set_running_false(main_window, qtbot):
    """A queue-drained ``set_running(False)`` must hide the entry
    bar too — regression against a half-cleaned state where the
    entry counter would haunt the panel after a multi-entry run."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    main_window.show()
    qtbot.waitExposed(main_window)
    panel.on_frame_progress(done=5, total=10, op_name="run_fitting", entry="entry_DBTTF")
    panel.on_queue_progress(current=2, total=8, entry="entry_DBTTF", op_name="run_fitting")
    assert panel._progress_bar.isVisible()
    assert panel._entry_progress_bar.isVisible()
    panel.set_running(False)
    assert not panel._progress_bar.isVisible()
    assert not panel._progress_label.isVisible()
    assert not panel._entry_progress_bar.isVisible()
    assert not panel._entry_progress_label.isVisible()


def test_cif_parse_busy_bar_lifecycle(main_window, qtbot):
    """"Parse CIFs" shows a compact indeterminate busy bar while the
    parse worker runs and hides it when the result (or error) lands —
    CifPattern has no per-CIF callback, so busy is the honest mode."""
    panel = main_window.pipeline_panel
    if not getattr(panel, "_available", True):
        pytest.skip("mlgidbase not installed in this env")
    # isHidden() (the widget's OWN state) rather than isVisible():
    # the Pipeline dock may be tabbed away in the fixture window, which
    # zeroes isVisible() for every child regardless of show()/hide().
    assert panel.cif_parse_bar.isHidden()
    # Swallow the emitted request so no real parse worker starts.
    panel.parseCifsRequested.disconnect()
    panel.cif_path.setText("some/folder")
    panel._on_parse_cifs()
    assert not panel.cif_parse_bar.isHidden()
    assert panel.cif_parse_bar.minimum() == 0 and panel.cif_parse_bar.maximum() == 0
    # Success hides it...
    class _Fake:
        cifs = ["A.cif"]
    panel.set_cif_pattern(_Fake(), None)
    assert panel.cif_parse_bar.isHidden()
    # ...and so does a parse failure.
    panel._on_parse_cifs()
    assert not panel.cif_parse_bar.isHidden()
    panel.set_cif_pattern(None, RuntimeError("boom"))
    assert panel.cif_parse_bar.isHidden()
