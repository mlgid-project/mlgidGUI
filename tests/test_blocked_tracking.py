"""Tracking at scale: the bounded-memory IoU is upstream's now.

Until mlgidbase 0.1.11 the dense ``calculate_iou_matrix`` needed
64 * N**2 bytes, and the GUI carried its own blocked equivalent
(``phase_tracking.track_peaks_blocked``) for big scans. 0.1.11 streams
the IoU in row blocks itself, so that code is gone and this file now
guards what the GUI still owns: that a run on a large scan goes through
upstream with no memory routing, and that the payload built from the
saved table matches the tracks upstream reports.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import phase_tracking
from mlgidgui.pipeline import PipelineCommand, execute

ENTRY = "entry_0000"


def test_the_gui_no_longer_carries_its_own_tracker():
    for gone in (
        "track_peaks_blocked", "_blocked_iou_edges",
        "estimate_tracking_memory", "available_memory_bytes",
        "capture_tracking", "build_payload",
        "amplitudes_from_track_result", "UpstreamContractError",
    ):
        assert not hasattr(phase_tracking, gone), gone


def test_payload_matches_upstreams_return_value(synthetic_fitted_scan):
    """The payload's tracks are exactly the tracks upstream returned:
    same count, and per track the same member count and frame set."""
    pytest.importorskip("mlgidbase")
    from mlgidbase import mlgidBASE

    payload = execute(
        synthetic_fitted_scan,
        PipelineCommand("track_peaks", {
            "entry": ENTRY, "threshold": 0.5, "length": 3,
        }),
    )
    analysis = mlgidBASE(filename=str(synthetic_fitted_scan))
    table = analysis.track_peaks(entry=ENTRY, threshold=0.5, length=3)
    assert len(table) == payload.n_tracks
    for k, row in enumerate(table):
        members = payload.track_members(k)
        assert sorted(payload.frame_num[members].tolist()) == sorted(
            np.asarray(row["frame_list"]).tolist()
        )
        assert sorted(payload.peak_id[members].tolist()) == sorted(
            np.asarray(row["peak_list"]).tolist()
        )


def test_a_scan_that_would_have_needed_the_blocked_path_just_runs(
    synthetic_fitted_scan, monkeypatch,
):
    """Whatever the machine reports as free memory, the run goes
    straight to upstream (no estimate, no routing)."""
    pytest.importorskip("mlgidbase")
    import mlgidbase

    calls: list = []
    real = mlgidbase.mlgidBASE.track_peaks

    def _spy(self, *a, **kw):
        calls.append(kw)
        return real(self, *a, **kw)

    monkeypatch.setattr(mlgidbase.mlgidBASE, "track_peaks", _spy)
    payload = execute(
        synthetic_fitted_scan,
        PipelineCommand("track_peaks", {
            "entry": ENTRY, "threshold": 0.5, "length": 3,
        }),
    )
    assert len(calls) == 1 and calls[0]["window_size"] is None
    assert payload.n_tracks == 1


def test_an_entry_without_fitted_rows_is_a_named_error(tmp_path):
    path = tmp_path / "nofit.h5"
    with h5py.File(path, "w") as f:
        d = f.create_group(f"{ENTRY}/data")
        d.create_dataset("img_gid_q", data=np.zeros((2, 4, 4), np.float32))
    with pytest.raises(ValueError):
        phase_tracking.read_fitted_members(path, ENTRY)
