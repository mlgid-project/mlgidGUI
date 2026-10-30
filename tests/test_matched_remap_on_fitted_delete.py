"""Deleting a fitted peak must keep matched structures, not wipe them.

Before: ``_delete_peaks_scoped`` cleared every ``matched_*`` row on the
frame whenever any fitted peak was deleted, because the matched
``peak_list`` stored *positions* into ``fitted_peaks`` and deleting a row
shifts them. So removing one fitted prediction made every matched
structure on the frame disappear.

Now a fitted delete is two steps (``file_model.prepare_fitted_delete``
before, ``file_model.remap_matched_peak_lists`` after):
- the frame's matched datasets are first put in ID form (mlgidbase
  >= 0.1.11 already writes ids; older files hold row positions, which
  only the still-intact fitted table can translate) and stamped with
  the ``peak_list_kind`` attribute,
- the delete then drops the deleted ids from every structure that
  referenced them and shifts nothing, and
- structures that referenced none of the deleted peaks keep their full
  membership, and no structure row is removed, even if its
  ``peak_list`` becomes empty.

These tests drive ``file_model`` directly (no Qt): they build a NeXus
file whose ``fitted_peaks`` ids are deliberately *not* equal to their
positions (mlgidGUI keeps ids stable across deletes), and
``matched_segments_*`` datasets in the real on-disk format (``peak_list``
= vlen int32, positions for old files, ids for new ones, see
``mlgidbase.mlgidmatch_functions``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mlgidgui import file_model

# fitted_peaks dtype as written by the pipeline (mirrors conftest's
# PYGID_PEAK_DTYPE; only the fields the readers touch matter here).
_FIT_DTYPE = np.dtype(
    [
        ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
        ("radius", "f4"), ("radius_width", "f4"), ("q_z", "f4"),
        ("q_xy", "f4"), ("theta", "f4"), ("score", "f4"),
        ("A", "f4"), ("B", "f4"), ("C", "f4"),
        ("is_ring", "bool"), ("is_cut_qz", "bool"), ("is_cut_qxy", "bool"),
        ("visibility", "i4"), ("id", "i4"),
    ]
)
_VLEN_INT = h5py.vlen_dtype(np.int32)
_MATCHED_DTYPE = np.dtype(
    [
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"), ("peak_list", _VLEN_INT),
    ]
)
ENTRY = "entry_0000"


def _build(
    tmp_path: Path,
    fitted_ids: list[int],
    solutions: list[tuple[str, list[int]]],
    kind: str | None = None,
    extra: dict[str, tuple[list[tuple[str, list[int]]], str | None]] | None = None,
) -> Path:
    """A NeXus file with one frame: ``fitted_peaks`` carrying ``fitted_ids``
    (radius encodes the original position so peaks stay identifiable by id),
    and one ``matched_segments_0000`` dataset whose rows are ``solutions``
    as ``(cif, peak_list)``. ``peak_list`` holds positions (the legacy
    format, the default) or ids, per ``kind`` (None = no stamp). ``extra``
    adds more datasets, ``{name: (solutions, kind)}``."""
    path = tmp_path / "matched.h5"
    n = len(fitted_ids)
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group(f"{ENTRY}/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset("img_gid_q", data=np.zeros((1, 8, 8), np.float32))
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))

        fit = np.zeros(n, dtype=_FIT_DTYPE)
        fit["id"] = fitted_ids
        fit["radius"] = [float(i + 1) for i in range(n)]  # position marker
        fit["angle"] = 45.0
        fit["angle_width"] = 5.0
        fit["radius_width"] = 0.2
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("fitted_peaks", data=fit)

        def _write(name, sols, stamp):
            rows = np.empty(len(sols), dtype=_MATCHED_DTYPE)
            for i, (cif, plist) in enumerate(sols):
                rows["CIF"][i] = cif.encode()
                rows["h"][i] = i
                rows["k"][i] = 0
                rows["l"][i] = 0
                rows["probability"][i] = 0.5
                rows["peak_list"][i] = np.asarray(plist, dtype=np.int32)
            ds = g.create_dataset(name, data=rows)
            if stamp is not None:
                ds.attrs[file_model.MATCHED_KIND_ATTR] = stamp

        _write("matched_segments_0000", solutions, kind)
        for name, (sols, stamp) in (extra or {}).items():
            _write(name, sols, stamp)
    return path


def _ids_per_solution(path: Path) -> list[set[int]]:
    """Fitted ids each rendered matched structure references, in file order."""
    fitted = file_model.load_peaks(path, ENTRY, 0)["fitted"]
    structs = file_model.load_matched_peaks(path, ENTRY, 0, fitted)
    return [set(int(x) for x in s.peaks.ids) for s in structs]


def _raw_peak_lists(
    path: Path, name: str = "matched_segments_0000",
) -> list[list[int]]:
    with h5py.File(path, "r") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/{name}"]
        arr = ds[()]
        return [list(np.asarray(arr["peak_list"][i], dtype=int))
                for i in range(len(arr))]


def _stamp(path: Path, name: str = "matched_segments_0000"):
    with h5py.File(path, "r") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/{name}"]
        return file_model.stamped_peak_list_kind(ds)


def test_positions_are_by_position_not_id(tmp_path):
    # ids deliberately != positions; id 2 sits at position 1, id 7 at 3.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1, 3])])
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [2]) == [1]
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [7, 5]) == [0, 3]
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [999]) == []


def test_delete_member_drops_only_that_peak_keeps_structure(tmp_path):
    # Structures over fitted ids [5,2,9,7] at positions [0,1,2,3]:
    #   A -> positions 0,1,3  (ids 5,2,7)
    #   B -> position  1      (id 2)            -> becomes empty
    #   C -> positions 2,3    (ids 9,7)
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [0, 1, 3]), ("B", [1]), ("C", [2, 3])],
    )
    # Delete fitted id 2 (position 1), mirroring _delete_peaks_scoped.
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [2])
    assert positions == [1]
    assert file_model.delete_peak_row(
        path, ENTRY, frame=0, kind="fitted", peak_id=2
    ) == 1
    changed = file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    # Two rows lose a member: A drops id 2, B empties; C is untouched.
    assert changed == 2

    # Raw peak_list after the delete is in id form: A = 5,7; B empty; C = 9,7.
    assert _raw_peak_lists(path) == [[5, 7], [], [9, 7]]
    assert _stamp(path) == "id"

    # All three rows survive (B kept though empty).
    assert len(_raw_peak_lists(path)) == 3

    ids = _ids_per_solution(path)
    # B (empty) is skipped by load_matched_peaks, so two render:
    assert ids == [{5, 7}, {9, 7}]  # A lost id 2; C unchanged membership


def test_delete_non_member_leaves_membership(tmp_path):
    # Delete a peak no structure references, sitting BELOW the referenced
    # ones: membership is unchanged, and (ids being stable) nothing in the
    # stored lists moves either.
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [2, 3])],  # ids 9,7 — does NOT include position 0 (id 5)
    )
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [5])
    assert positions == [0]
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=5)
    changed = file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert changed == 0
    assert _raw_peak_lists(path) == [[9, 7]]
    assert _ids_per_solution(path) == [{9, 7}]


def test_delete_between_structures_keeps_both_intact(tmp_path):
    # A structure entirely ABOVE the deleted position and one entirely
    # BELOW both keep their members; in id form neither list changes.
    path = _build(
        tmp_path, [10, 11, 12, 13, 14],
        [("low", [0, 1]), ("high", [3, 4])],
    )
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [12])  # pos 2
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=12)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert _raw_peak_lists(path) == [[10, 11], [13, 14]]
    assert _ids_per_solution(path) == [{10, 11}, {13, 14}]


def test_bulk_delete_two_positions(tmp_path):
    # Delete two fitted peaks at once (positions 1 and 3).
    path = _build(
        tmp_path, [5, 2, 9, 7, 8],
        [("A", [0, 1, 2, 3, 4])],  # all five
    )
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [2, 7])
    assert positions == [1, 3]
    for pid in (2, 7):
        file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=pid)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    # ids 5, 9, 8 survive, in their original order.
    assert _raw_peak_lists(path) == [[5, 9, 8]]
    assert _ids_per_solution(path) == [{5, 9, 8}]


def test_remap_no_op_without_matches(tmp_path):
    # No matched datasets at all: remap is a clean 0-change no-op.
    path = _build(tmp_path, [5, 2, 9], [])
    with h5py.File(path, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"]
    assert file_model.remap_matched_peak_lists(path, ENTRY, 0, [1]) == 0


def test_remap_empty_positions_is_no_op(tmp_path):
    path = _build(tmp_path, [5, 2, 9], [("A", [0, 2])])
    before = _raw_peak_lists(path)
    assert file_model.remap_matched_peak_lists(path, ENTRY, 0, []) == 0
    assert _raw_peak_lists(path) == before


# -- peak_list kinds: ids (mlgidbase >= 0.1.11) vs positions (older) --------


def test_classify_peak_list_kind_table():
    ids = [5, 2, 9, 7]  # n = 4
    classify = file_model.classify_peak_list_kind
    assert classify([0, 1, 3], ids) == "position"   # no value is an id
    assert classify([9, 7], ids) == "id"            # 9 and 7 exceed n
    assert classify([5, 2], ids) == "id"            # ids, valid positions? 5 is not
    assert classify([2], ids) is None               # valid both ways
    assert classify([], ids) is None                # nothing to go on
    # ids 0..n-1: the two readings coincide, so it can never tell.
    assert classify([0, 1, 2], [0, 1, 2, 3]) is None
    # A stale value counts for neither reading, the majority decides.
    assert classify([0, 1, 40], ids) == "position"


def test_resolve_matched_kinds_stamp_wins_and_majority_breaks_ties():
    ids = [5, 2, 9, 7]
    datasets = {
        "matched_rings_0000": ("id", np.asarray([2])),        # stamped
        "matched_segments_0000": (None, np.asarray([9, 7])),  # id by values
        "matched_segments_0001": (None, np.asarray([2])),     # cannot tell
    }
    kinds = file_model.resolve_matched_kinds(datasets, ids)
    assert kinds == {
        "matched_rings_0000": "id",
        "matched_segments_0000": "id",
        "matched_segments_0001": "id",   # majority of the decided ones
    }
    # Nothing decided anywhere: the legacy positional format.
    only_unknown = {"matched_segments_0000": (None, np.asarray([2]))}
    assert file_model.resolve_matched_kinds(only_unknown, ids) == {
        "matched_segments_0000": "position",
    }


def test_id_form_with_gaps_renders_the_right_peaks(tmp_path):
    # Ids [5,2,9,7]: n = 4, so ids 5, 9 and 7 are all >= n. Read as
    # positions these would be dropped or hit the wrong rows; they are ids.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [5, 9, 7])], kind="id")
    assert _ids_per_solution(path) == [{5, 9, 7}]
    # Same data without the stamp: the values give it away.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [9, 7])])
    assert _ids_per_solution(path) == [{9, 7}]


def test_a_mixed_frame_reads_each_dataset_in_its_own_kind(tmp_path):
    # Rings written by the new backend (ids), segments by the old one
    # (positions), in the same frame.
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("seg", [0, 1])],                                  # ids 5, 2
        extra={"matched_rings_0000": ([("ring", [9, 7])], "id")},
    )
    fitted = file_model.load_peaks(path, ENTRY, 0)["fitted"]
    structs = {
        s.cif: set(int(x) for x in s.peaks.ids)
        for s in file_model.load_matched_peaks(path, ENTRY, 0, fitted)
    }
    assert structs == {"seg": {5, 2}, "ring": {9, 7}}
    # MatchedStructure.peak_list stays POSITIONAL for downstream users.
    by_cif = {
        s.cif: sorted(int(x) for x in s.peak_list)
        for s in file_model.load_matched_peaks(path, ENTRY, 0, fitted)
    }
    assert by_cif == {"seg": [0, 1], "ring": [2, 3]}


def test_delete_in_id_form_shifts_nothing(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [5, 2, 7]), ("B", [9])], kind="id",
    )
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [2])
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=2)
    changed = file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert changed == 1
    assert _raw_peak_lists(path) == [[5, 7], [9]]
    assert _stamp(path) == "id"
    assert _ids_per_solution(path) == [{5, 7}, {9}]


def test_a_delete_migrates_a_mixed_frame_in_place(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("seg", [0, 1, 3])],                               # positions
        extra={"matched_rings_0000": ([("ring", [9, 7])], "id")},
    )
    positions = file_model.prepare_fitted_delete(path, ENTRY, 0, [7])
    # Both datasets are id form and stamped BEFORE anything is deleted.
    assert _raw_peak_lists(path) == [[5, 2, 7]]
    assert _stamp(path) == "id"
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=7)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert _raw_peak_lists(path) == [[5, 2]]
    assert _raw_peak_lists(path, "matched_rings_0000") == [[9]]


def test_remap_without_prepare_still_migrates_a_legacy_dataset(tmp_path):
    # The two-call shape older code and tests use: positions captured with
    # the pure helper, no prepare step. The dataset is unstamped, so it is
    # taken as positional, shifted, then translated to ids and stamped.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1, 3])])
    positions = file_model.fitted_positions_for_ids(path, ENTRY, 0, [2])
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=2)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert _raw_peak_lists(path) == [[5, 7]]
    assert _stamp(path) == "id"


def test_prepare_writes_nothing_when_no_row_matches(tmp_path):
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1])])
    assert file_model.prepare_fitted_delete(path, ENTRY, 0, [999]) == []
    assert _stamp(path) is None
    assert _raw_peak_lists(path) == [[0, 1]]


def test_normalise_leaves_a_frame_without_fitted_rows_alone(tmp_path):
    # Positions cannot be translated without fitted rows; emptying them
    # would destroy the data.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1])])
    with h5py.File(path, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
    assert file_model.normalise_matched_to_ids(path, ENTRY, 0) == 0
    assert _raw_peak_lists(path) == [[0, 1]]


def test_stamp_matched_kind_only_touches_unstamped_datasets(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7], [("seg", [0, 1])],
        extra={"matched_rings_0000": ([("ring", [9])], "position")},
    )
    assert file_model.stamp_matched_kind(path, ENTRY, 0, "id") == 1
    assert _stamp(path) == "id"
    assert _stamp(path, "matched_rings_0000") == "position"
    assert file_model.stamp_matched_kind(path, ENTRY, None, "id") == 0


def test_raw_snapshot_is_in_id_form_and_merge_writes_id_form(tmp_path):
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1])])  # positions
    snap = file_model.read_matched_raw(path, ENTRY, [0])
    (arr,) = snap[0].values()
    assert [int(v) for v in arr["peak_list"][0]] == [5, 2]
    # The re-match writes a different row; merging restores A as well.
    with h5py.File(path, "r+") as f:
        g = f[f"{ENTRY}/data/analysis/frame00000"]
        rows = np.empty(1, dtype=_MATCHED_DTYPE)
        rows["CIF"][0], rows["h"][0], rows["k"][0], rows["l"][0] = b"B", 9, 0, 0
        rows["probability"][0] = 0.9
        rows["peak_list"][0] = np.asarray([9, 7], dtype=np.int32)
        del g["matched_segments_0000"]
        ds = g.create_dataset("matched_segments_0000", data=rows)
        ds.attrs[file_model.MATCHED_KIND_ATTR] = "id"
    assert file_model.merge_matched_rows(path, ENTRY, 0, snap[0]) == 1
    assert _stamp(path) == "id"
    assert sorted(
        tuple(sorted(r)) for r in _raw_peak_lists(path)
    ) == [(2, 5), (7, 9)]


def test_csv_export_lists_fitted_ids_for_a_positional_file(tmp_path):
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1, 3])])
    out = tmp_path / "m.csv"
    n = file_model.export_matched_csv(path, [(ENTRY, 0)], out)
    assert n == 1
    assert "[5, 2, 7]" in out.read_text()


# -- matched-only delete (replaces the mlgidbase.delete_peak route) ---------


def test_drop_ids_from_matched_touches_only_matched(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [0, 1, 3]), ("B", [1]), ("C", [2, 3])],   # positional
    )
    changed = file_model.drop_ids_from_matched(path, ENTRY, 0, [2])
    assert changed == 2                         # A and B lose id 2
    assert _raw_peak_lists(path) == [[5, 7], [], [9, 7]]
    assert _stamp(path) == "id"
    # The fitted table is exactly as it was.
    assert _table_ids_of(path) == [5, 2, 9, 7]
    # Nothing references id 2 any more: a second drop is a no-op.
    assert file_model.drop_ids_from_matched(path, ENTRY, 0, [2]) == 0


def test_drop_ids_from_matched_on_id_form_with_gaps(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7], [("A", [5, 9, 7]), ("B", [9])], kind="id",
    )
    assert file_model.drop_ids_from_matched(path, ENTRY, 0, [9]) == 2
    assert _raw_peak_lists(path) == [[5, 7], []]
    assert _ids_per_solution(path) == [{5, 7}]   # the emptied B stops rendering


def test_matched_snapshot_restores_values_and_stamp(tmp_path):
    path = _build(
        tmp_path, [5, 2, 9, 7], [("A", [0, 1, 3])],
        extra={"matched_rings_0000": ([("ring", [9, 7])], "id")},
    )
    snap = file_model.snapshot_matched(path, ENTRY, 0)
    assert set(snap) == {"matched_segments_0000", "matched_rings_0000"}
    file_model.drop_ids_from_matched(path, ENTRY, 0, [7, 5])
    assert _stamp(path) == "id"                 # migrated by the drop
    assert file_model.restore_matched(path, ENTRY, 0, snap) == 2
    assert _raw_peak_lists(path) == [[0, 1, 3]]
    assert _stamp(path) is None                 # the legacy state, verbatim
    assert _raw_peak_lists(path, "matched_rings_0000") == [[9, 7]]
    assert _stamp(path, "matched_rings_0000") == "id"


def _table_ids_of(path):
    return [int(v) for v in file_model.load_peaks(path, ENTRY, 0)["fitted"].ids]
