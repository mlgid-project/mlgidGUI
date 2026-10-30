"""Duplicate matched-structure consolidation (write side + read side).

mlgidmatch emits one ``matched_<type>_NNNN`` dataset per alternative
multi-phase solution, so the same (CIF, h, k, l) structure appears in
many datasets — usually with a different peak subset each time. Both
sides must collapse that to one row/structure with the peak lists
UNIONED and the maximum probability:

* write side — ``pipeline._dedupe_matched_groups`` (runs after every
  ``run_matching``);
* read side — ``file_model.read_matched_peaks`` (also covers files
  written before the fix or by external tooling).

Backend-less safe: pure h5py/numpy against the peaks fixture.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.pipeline import _dedupe_matched_groups

ENTRY = "entry_0000"


def _write_solutions(path, name, rows):
    """rows: (cif, hkl, probability, peak_list)."""
    import h5py

    dt = np.dtype([
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"),
        ("peak_list", h5py.vlen_dtype(np.int32)),
    ])
    arr = np.zeros(len(rows), dtype=dt)
    for i, (cif, hkl, prob, peak_list) in enumerate(rows):
        arr["CIF"][i] = cif.encode()
        arr["h"][i], arr["k"][i], arr["l"][i] = hkl
        arr["probability"][i] = prob
        arr["peak_list"][i] = np.asarray(peak_list, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f[f"{ENTRY}/data/analysis/frame00000"]
        if name in g:
            del g[name]
        g.create_dataset(name, data=arr)


@pytest.fixture
def duplicated(synthetic_nexus_with_peaks):
    """Two segment 'solutions' sharing In2O3 (1,1,1) with different
    peak subsets, plus an untouched rings dataset of the same
    structure (must never merge across types)."""
    path = synthetic_nexus_with_peaks
    _write_solutions(path, "matched_segments_0000", [
        ("In2O3", (1, 1, 1), 0.9, [0]),
    ])
    _write_solutions(path, "matched_segments_0001", [
        ("In2O3", (1, 1, 1), 0.7, [1]),
        ("PbI2", (1, 0, 0), 0.8, [0]),
    ])
    _write_solutions(path, "matched_rings_0000", [
        ("In2O3", (1, 1, 1), 0.6, [1]),
    ])
    return path


def test_write_side_consolidation_unions_peak_lists(duplicated):
    import h5py

    _dedupe_matched_groups(
        duplicated, ENTRY, [0], "segments", logging.getLogger("test"),
    )

    with h5py.File(duplicated, "r") as f:
        g = f[f"{ENTRY}/data/analysis/frame00000"]
        seg_keys = sorted(k for k in g if k.startswith("matched_segments"))
        assert seg_keys == ["matched_segments_0000"]
        rows = g["matched_segments_0000"][()]
        got = {
            (row["CIF"].decode(), int(row["h"]), int(row["k"]),
             int(row["l"])): (
                float(row["probability"]),
                np.asarray(row["peak_list"]).tolist(),
            )
            for row in rows
        }
        assert got == {
            ("In2O3", 1, 1, 1): (pytest.approx(0.9), [0, 1]),
            ("PbI2", 1, 0, 0): (pytest.approx(0.8), [0]),
        }
        # Rows come out probability-descending.
        assert rows["probability"][0] >= rows["probability"][1]
        # The rings dataset of the same structure is untouched.
        assert "matched_rings_0000" in g


def test_read_side_merge_and_type_isolation(duplicated):
    tables = file_model.load_peaks(duplicated, ENTRY, 0)
    structures = file_model.load_matched_peaks(
        duplicated, ENTRY, 0, tables["fitted"]
    )
    got = {
        (s.solution_field.rsplit("_", 1)[0], s.cif, s.h, s.k, s.l): (
            s.probability, list(s.peak_list), sorted(int(i) for i in s.peaks.ids),
        )
        for s in structures
    }
    # One entry per (type, structure): the segments duplicates merged
    # (peak union, max probability), the rings identification stays a
    # separate structure.
    assert got == {
        ("matched_segments", "In2O3", 1, 1, 1): (
            pytest.approx(0.9), [0, 1], [0, 1],
        ),
        ("matched_segments", "PbI2", 1, 0, 0): (
            pytest.approx(0.8), [0], [0],
        ),
        ("matched_rings", "In2O3", 1, 1, 1): (
            pytest.approx(0.6), [1], [1],
        ),
    }
