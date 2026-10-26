"""``file_model.delete_peak_row`` — kind-scoped peak removal that does
NOT cascade across detected / fitted / matched (the way
``mlgidbase.delete_peak`` does). Used by the GUI's "Delete peak"
flow so removing a fitted row leaves the detected row intact, and
vice versa.

The fitted delete also takes the row's entry in ``fitted_peaks_errors``
(paired by id) with it, and the host drops the deleted ids from
``matched_*`` ``peak_list`` via ``prepare_fitted_delete`` /
``remap_matched_peak_lists`` (see ``test_matched_remap_on_fitted_delete``).
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidlab import file_model

# Minimal peak dtype — only the fields the row-delete helper needs
# (``id`` for matching, plus a couple of geometry fields so the
# remaining rows are visibly identifiable in assertions). Mirrors
# the structured dtype layout pygid / mlgidbase write to detected /
# fitted datasets; full per-field listing lives in
# ``tests/conftest.py::PYGID_PEAK_DTYPE``.
_TEST_PEAK_DTYPE = [
    ("id", "i4"),
    ("radius", "f4"),
    ("angle", "f4"),
    ("radius_width", "f4"),
    ("angle_width", "f4"),
    ("amplitude", "f4"),
]


def _seed_file_with_three_kinds(path, *, with_matched: bool = True):
    """A frame with 3 detected, 3 fitted, and 2 matched solution
    rows. Ids are 0/1/2 in both detected and fitted (mlgidbase's
    natural convention)."""
    rng = np.random.default_rng(0)
    dt = np.dtype(_TEST_PEAK_DTYPE)
    detected = np.zeros(3, dtype=dt)
    detected["id"] = [0, 1, 2]
    detected["radius"] = [1.0, 2.0, 3.0]
    detected["angle"] = [10.0, 30.0, 80.0]
    fitted = np.zeros(3, dtype=dt)
    fitted["id"] = [0, 1, 2]
    fitted["radius"] = [1.1, 2.1, 3.1]
    fitted["angle"] = [11.0, 31.0, 81.0]
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=rng.random((1, 8, 8), dtype=np.float32),
        )
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("detected_peaks", data=detected)
        g.create_dataset("fitted_peaks", data=fitted)
        if with_matched:
            mdtype = np.dtype(
                [("CIF", "S32"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
                 ("peak_list", "S16"), ("probability", "f4")]
            )
            mrows = np.zeros(2, dtype=mdtype)
            mrows["CIF"] = [b"a.cif", b"b.cif"]
            mrows["peak_list"] = [b"[0, 1]", b"[1, 2]"]
            mrows["probability"] = [0.9, 0.7]
            g.create_dataset("matched_segments_0000", data=mrows)
    return path


def test_delete_fitted_leaves_detected_intact(tmp_path):
    """The user's specific complaint: deleting a fitted peak via the
    GUI was wiping the detected peak with the same id (mlgidbase
    cascade). With the kind-scoped helper, only fitted_peaks is
    touched."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped.h5")
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    assert removed == 1
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    # detected is untouched — all three rows still there.
    assert sorted(det["id"].tolist()) == [0, 1, 2]
    # fitted lost row id=1 only.
    assert sorted(fit["id"].tolist()) == [0, 2]


def test_delete_detected_leaves_fitted_intact(tmp_path):
    """Symmetric: deleting a detected peak doesn't wipe the fitted."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped2.h5")
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="detected", peak_id=2,
    )
    assert removed == 1
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    assert sorted(det["id"].tolist()) == [0, 1]
    assert sorted(fit["id"].tolist()) == [0, 1, 2]


def test_delete_preserves_unrelated_row_ids(tmp_path):
    """Unlike mlgidbase's _delete_*, this helper does NOT re-index
    ids on the surviving rows. A row with id=2 stays id=2 even when
    the row with id=1 was deleted in front of it. Keeps matched_*
    peak_list references stable (their indices into the kept rows
    are unchanged, modulo positional shifts which the host's
    matched_* remap handles via remap_matched_peak_lists)."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped3.h5", with_matched=False)
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=0,
    )
    with h5py.File(path, "r") as f:
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    # id=0 gone; the remaining rows kept their original ids 1, 2 —
    # NOT reindexed to 0, 1 (mlgidbase <= 0.1.8 did that; 0.1.11 and
    # this helper do not).
    assert sorted(fit["id"].tolist()) == [1, 2]


def test_delete_missing_id_is_noop(tmp_path):
    """Removing an id that isn't present returns 0 and leaves the
    file untouched (no row count change, no error)."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped4.h5", with_matched=False)
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="detected", peak_id=99,
    )
    assert removed == 0
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
    assert sorted(det["id"].tolist()) == [0, 1, 2]


def test_delete_rejects_unsupported_kind(tmp_path):
    """Matched live in per-solution datasets; the row-delete helper
    only handles detected / fitted. ``"matched"`` raises so the
    host knows to take a different path."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped5.h5")
    with pytest.raises(ValueError, match="detected or fitted"):
        file_model.delete_peak_row(
            path, "entry_0000", frame=0, kind="matched", peak_id=0,
        )


def test_delete_preserves_dataset_attrs(tmp_path):
    """Dataset is recreated under the same name with the same dtype
    and attrs; the in-place delete + recreate pattern (same as
    ``clear_peaks`` and ``add_fitted_peak_row``) must carry attrs
    forward so a downstream reader doesn't see a half-stripped
    dataset."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped6.h5")
    # Stamp a marker attr.
    with h5py.File(path, "r+") as f:
        ds = f["entry_0000/data/analysis/frame00000/fitted_peaks"]
        ds.attrs["mlgid_marker"] = "kept"
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    with h5py.File(path, "r") as f:
        ds = f["entry_0000/data/analysis/frame00000/fitted_peaks"]
        assert ds.attrs.get("mlgid_marker") == "kept"


def test_delete_nonexistent_dataset_is_noop(tmp_path):
    """When the frame's analysis group doesn't carry a fitted_peaks
    dataset at all (e.g. detection never ran), the helper returns
    0 without raising."""
    path = tmp_path / "empty_analysis.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        d = f.create_group("entry_0000/data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((1, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        d.create_group("analysis/frame00000")  # no peaks datasets
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=0,
    )
    assert removed == 0


# -- fitted_peaks_errors is paired with fitted_peaks by id ------------------


def _add_errors(path, ids=(0, 1, 2)):
    """A ``fitted_peaks_errors`` twin for ``_seed_file_with_three_kinds``:
    one row per id, ``radius`` carrying a recognisable error value."""
    dt = np.dtype(_TEST_PEAK_DTYPE)
    err = np.zeros(len(ids), dtype=dt)
    err["id"] = list(ids)
    err["radius"] = [0.01 * (i + 1) for i in ids]
    with h5py.File(path, "r+") as f:
        f["entry_0000/data/analysis/frame00000"].create_dataset(
            "fitted_peaks_errors", data=err,
        )
    return path


def _errors(path):
    with h5py.File(path, "r") as f:
        return f["entry_0000/data/analysis/frame00000/fitted_peaks_errors"][()]


def test_deleting_a_fitted_row_removes_its_errors_row(tmp_path):
    """The two tables are paired by id. A left-over errors row would keep
    a peak that no longer exists and let the next peak to reuse the id
    inherit its fit errors."""
    path = _add_errors(_seed_file_with_three_kinds(tmp_path / "err1.h5"))
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    assert _errors(path)["id"].tolist() == [0, 2]
    assert np.isclose(_errors(path)["radius"], [0.01, 0.03]).all()


def test_deleting_a_detected_row_leaves_the_errors_table_alone(tmp_path):
    path = _add_errors(_seed_file_with_three_kinds(tmp_path / "err2.h5"))
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="detected", peak_id=1,
    )
    assert _errors(path)["id"].tolist() == [0, 1, 2]


def test_a_missing_errors_table_or_row_is_not_an_error(tmp_path):
    path = _seed_file_with_three_kinds(tmp_path / "err3.h5")
    assert file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=0,
    ) == 1
    path = _add_errors(
        _seed_file_with_three_kinds(tmp_path / "err4.h5"), ids=(0, 2),
    )
    # id 1 has no errors row (a GUI-added peak): the delete still works.
    assert file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    ) == 1
    assert _errors(path)["id"].tolist() == [0, 2]


def test_the_errors_row_comes_back_with_the_snapshot_on_undo(tmp_path):
    path = _add_errors(_seed_file_with_three_kinds(tmp_path / "err5.h5"))
    (snap,) = file_model.read_peak_rows(
        path, "entry_0000", 0, "fitted", [1], with_ids=True, with_errors=True,
    )
    assert np.isclose(snap["errors"]["radius"], 0.02)
    # Without the flag the dict is exactly what it always was.
    (plain,) = file_model.read_peak_rows(
        path, "entry_0000", 0, "fitted", [1],
    )
    assert "errors" not in plain

    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    assert file_model.restore_fitted_error_row(
        path, "entry_0000", 0, 1, snap["errors"],
    )
    err = _errors(path)
    assert sorted(err["id"].tolist()) == [0, 1, 2]
    assert np.isclose(err["radius"][err["id"] == 1], 0.02)
    # Restoring twice replaces rather than duplicates.
    file_model.restore_fitted_error_row(
        path, "entry_0000", 0, 1, snap["errors"],
    )
    assert sorted(_errors(path)["id"].tolist()) == [0, 1, 2]


def test_restore_without_an_errors_table_is_a_quiet_no(tmp_path):
    path = _seed_file_with_three_kinds(tmp_path / "err6.h5")
    assert file_model.restore_fitted_error_row(
        path, "entry_0000", 0, 1, {"radius": 0.5},
    ) is False


def test_fitted_csv_pairs_errors_by_id_not_by_position(tmp_path):
    """Row 0 of fitted is id 0 but row 0 of the errors table is id 2:
    a positional join would attach the wrong errors to every peak."""
    path = _seed_file_with_three_kinds(tmp_path / "err7.h5")
    dt = np.dtype(_TEST_PEAK_DTYPE)
    err = np.zeros(2, dtype=dt)
    err["id"] = [2, 0]
    err["radius"] = [0.30, 0.10]
    with h5py.File(path, "r+") as f:
        f["entry_0000/data/analysis/frame00000"].create_dataset(
            "fitted_peaks_errors", data=err,
        )
    out = tmp_path / "fit.csv"
    assert file_model.export_peaks_csv(path, [("entry_0000", 0)], "fitted", out) == 3
    import csv
    with open(out) as fp:
        rows = {int(r["id"]): r for r in csv.DictReader(fp)}
    assert float(rows[0]["radius_err"]) == pytest.approx(0.10)
    assert float(rows[2]["radius_err"]) == pytest.approx(0.30)
    # id 1 has no errors row: the cells are left empty, not borrowed.
    assert rows[1].get("radius_err", "") in ("", None)


# -- the saved tracking table (mlgidbase >= 0.1.11) --------------------------


_TRACK_DTYPE = np.dtype([
    ("id", "i4"), ("CIF", "S64"), ("h", "f4"), ("k", "f4"), ("l", "f4"),
    ("frame_list", h5py.vlen_dtype(np.int32)),
    ("peak_list", h5py.vlen_dtype(np.int32)),
])


def _add_tracked(path, tracks, extra_general: bool = False):
    arr = np.empty(len(tracks), dtype=_TRACK_DTYPE)
    for i, (frames, ids) in enumerate(tracks):
        arr["id"][i] = i
        arr["CIF"][i] = b"unmatched"
        arr["h"][i] = arr["k"][i] = arr["l"][i] = np.nan
        arr["frame_list"][i] = np.asarray(frames, dtype=np.int32)
        arr["peak_list"][i] = np.asarray(ids, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f.require_group("entry_0000/data/analysis/general")
        g.create_dataset("tracked_peaks", data=arr)
        if extra_general:
            g.create_dataset("other", data=np.zeros(1))
        f.require_group("entry_0000/process/peak_tracking").attrs["program"] = "mlgidbase"
    return path


def _tracks(path):
    with h5py.File(path, "r") as f:
        t = f["entry_0000/data/analysis/general/tracked_peaks"][()]
    return [
        (list(map(int, r["frame_list"])), list(map(int, r["peak_list"])))
        for r in t
    ]


def test_drop_ids_from_tracked_removes_only_that_frames_pairs(tmp_path):
    path = _add_tracked(
        _seed_file_with_three_kinds(tmp_path / "trk1.h5"),
        [([0, 1, 2], [1, 1, 2]), ([0, 1], [0, 0])],
    )
    # id 1 on frame 1 leaves the first track; id 1 on frame 0 is a
    # different pair and stays.
    assert file_model.drop_ids_from_tracked(path, "entry_0000", 1, [1]) == 1
    assert _tracks(path) == [([0, 2], [1, 2]), ([0, 1], [0, 0])]
    assert file_model.drop_ids_from_tracked(path, "entry_0000", 1, [1]) == 0


def test_drop_ids_from_tracked_ignores_a_missing_or_old_table(tmp_path):
    path = _seed_file_with_three_kinds(tmp_path / "trk2.h5")
    assert file_model.drop_ids_from_tracked(path, "entry_0000", 0, [0]) == 0
    old = np.empty(1, dtype=np.dtype([
        ("id", "i4"), ("frames", h5py.vlen_dtype(np.int32)),
        ("peak_ids", h5py.vlen_dtype(np.int32)),
    ]))
    old["frames"][0] = np.asarray([0], dtype=np.int32)
    old["peak_ids"][0] = np.asarray([0], dtype=np.int32)
    with h5py.File(path, "r+") as f:
        f.require_group("entry_0000/data/analysis/general").create_dataset(
            "tracked_peaks", data=old,
        )
    assert file_model.drop_ids_from_tracked(path, "entry_0000", 0, [0]) == 0


def test_a_fitted_delete_prepare_step_drops_the_id_from_the_tracks(tmp_path):
    path = _add_tracked(
        _seed_file_with_three_kinds(tmp_path / "trk3.h5", with_matched=False),
        [([0, 0], [0, 1])],
    )
    file_model.prepare_fitted_delete(path, "entry_0000", 0, [1])
    assert _tracks(path) == [([0], [0])]
    # No fitted row matched: nothing is touched.
    file_model.prepare_fitted_delete(path, "entry_0000", 0, [99])
    assert _tracks(path) == [([0], [0])]


def test_clear_tracked_peaks_removes_table_metadata_and_empty_group(tmp_path):
    path = _add_tracked(
        _seed_file_with_three_kinds(tmp_path / "trk4.h5"), [([0], [0])],
    )
    assert file_model.clear_tracked_peaks(path, "entry_0000") is True
    with h5py.File(path, "r") as f:
        ana = f["entry_0000/data/analysis"]
        # An empty 'general' would be counted as a frame by pygid.
        assert "general" not in ana
        assert "entry_0000/process/peak_tracking" not in f
    assert file_model.clear_tracked_peaks(path, "entry_0000") is False


def test_clear_tracked_peaks_keeps_a_general_group_with_other_content(tmp_path):
    path = _add_tracked(
        _seed_file_with_three_kinds(tmp_path / "trk5.h5"), [([0], [0])],
        extra_general=True,
    )
    assert file_model.clear_tracked_peaks(path, "entry_0000") is True
    with h5py.File(path, "r") as f:
        g = f["entry_0000/data/analysis/general"]
        assert list(g.keys()) == ["other"]


def test_appending_frames_removes_the_stale_tracking_table(tmp_path):
    """pygid numbers new frames by counting every group under analysis;
    a leftover 'general' group would shift them by one."""
    from pathlib import Path

    from mlgidlab import conversion

    path = _add_tracked(
        _seed_file_with_three_kinds(tmp_path / "trk6.h5"), [([0], [0])],
    )

    class _Cfg:
        append_entry = "entry_0000"

    conversion._drop_stale_tracking(_Cfg(), {Path("raw.h5"): path})
    with h5py.File(path, "r") as f:
        assert "general" not in f["entry_0000/data/analysis"]
