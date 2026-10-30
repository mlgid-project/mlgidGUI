"""Lazy, per-entry pygid normalization (deferred off the open path).

Opening a master file that links hundreds of external scans used to
freeze: ``normalize_for_pygid`` read every entry's signal shape (one
external-scan open each) and created per-frame analysis groups for every
frame of every entry, on the GUI thread. It is now scoped to one entry
and run lazily, just before that entry's first pipeline run
(``MainWindow._ensure_entry_normalized``). These cover the scoping and
the once-per-(file, entry) guard.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model


def _two_entry_file(path):
    """A master with two q-entries, each 3 frames, each with a *scalar*
    angle_of_incidence (to exercise the 0-D → 1-D patch)."""
    with h5py.File(path, "w", track_order=True) as f:
        for name in ("entry_0000", "entry_0001"):
            data = f.create_group(f"{name}/data", track_order=True)
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset("img_gid_q", data=np.zeros((3, 4, 5), dtype="f4"))
            f.create_dataset(
                f"{name}/instrument/angle_of_incidence", data=np.float64(0.3)
            )
    return path


def _frame_groups(path, entry):
    with h5py.File(path, "r") as f:
        ana = f.get(f"{entry}/data/analysis")
        return sorted(ana.keys()) if ana is not None else []


def _ai_ndim(path, entry):
    with h5py.File(path, "r") as f:
        return f[f"{entry}/instrument/angle_of_incidence"].ndim


def test_normalize_scoped_to_one_entry(tmp_path):
    p = _two_entry_file(tmp_path / "m.h5")
    out = file_model.normalize_for_pygid(p, entry="entry_0000")

    assert out["frames"] == ["entry_0000"]
    assert out["angle"] == ["entry_0000"]
    # entry_0000 normalized: 3 frame groups + 1-D angle_of_incidence
    assert _frame_groups(p, "entry_0000") == [
        "frame00000", "frame00001", "frame00002",
    ]
    assert _ai_ndim(p, "entry_0000") == 1
    # entry_0001 left completely untouched (the whole point — no scan of
    # entries the caller didn't ask for).
    assert _frame_groups(p, "entry_0001") == []
    assert _ai_ndim(p, "entry_0001") == 0


def test_normalize_all_entries_when_unscoped(tmp_path):
    """entry=None keeps the original all-entries behaviour (tests, any
    future eager caller)."""
    p = _two_entry_file(tmp_path / "m.h5")
    file_model.normalize_for_pygid(p)
    for name in ("entry_0000", "entry_0001"):
        assert len(_frame_groups(p, name)) == 3
        assert _ai_ndim(p, name) == 1


def test_normalize_missing_entry_is_noop(tmp_path):
    p = _two_entry_file(tmp_path / "m.h5")
    assert file_model.normalize_for_pygid(p, entry="entry_9999") == {
        "angle": [], "frames": [],
    }


@pytest.mark.gui
def test_ensure_entry_normalized_runs_once(main_window, tmp_path, monkeypatch):
    """``_ensure_entry_normalized`` calls the scoped normalize at most once
    per (file, entry) and no-ops for a missing entry name."""
    p = _two_entry_file(tmp_path / "m.h5")
    calls: list = []

    def _spy(file_path, entry=None):
        calls.append(entry)
        return {"angle": [], "frames": []}

    monkeypatch.setattr(file_model, "normalize_for_pygid", _spy)

    main_window._ensure_entry_normalized(p, "entry_0000")
    main_window._ensure_entry_normalized(p, "entry_0000")  # cached → no 2nd call
    main_window._ensure_entry_normalized(p, None)           # no-op
    assert calls == ["entry_0000"]
