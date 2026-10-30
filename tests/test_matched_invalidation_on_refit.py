"""F-04 closure: ``pipeline.execute`` must clear every ``matched_*``
row on the scope being refit, before mlgidbase / pygidFIT rewrites
``fitted_peaks``. Stale matched solutions referencing fitted
positions by integer index can otherwise silently mis-render
against the re-ordered fitted array.

The audit's preferred remediation was "re-key by fitted id or
invalidate on refit." Invalidation is the path taken; this test
locks in that the invalidation actually fires on the write side
rather than relying on the read-side clamp at
``file_model.load_matched_peaks`` to drop dangling indices.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mlgidgui.pipeline import PipelineCommand, execute


def _seed_file_with_matched(path: Path, entry: str = "entry_0000") -> Path:
    """Build a minimal NeXus carrying one entry with two frames, plus
    a populated ``matched_*`` group on one of them. ``execute`` should
    wipe the matched_* dataset before forwarding to mlgidbase.

    The matched dataset's row dtype is whatever h5py infers from the
    seed array — we only care that the dataset exists with non-zero
    length before ``execute`` runs and exists with zero length after."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry_g = f.create_group(entry, track_order=True)
        d = entry_g.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        # Seed a matched_* solution on frame 0. Real solutions are
        # written by mlgidmatch; we just need a dataset with rows so
        # we can check it ends up empty.
        analysis = d.create_group("analysis", track_order=True)
        frame0 = analysis.create_group("frame00000", track_order=True)
        dtype = np.dtype(
            [("CIF", "S32"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
             ("peak_list", "S16"), ("probability", "f4")]
        )
        rows = np.zeros(3, dtype=dtype)
        rows["CIF"] = [b"cif_a.cif", b"cif_b.cif", b"cif_c.cif"]
        rows["h"] = [1, 0, 1]
        rows["k"] = [1, 1, 0]
        rows["l"] = [0, 0, 1]
        rows["peak_list"] = [b"[0, 1]", b"[1]", b"[0, 2]"]
        rows["probability"] = [0.9, 0.7, 0.6]
        frame0.create_dataset("matched_segments_0000", data=rows)
    return path


def _matched_row_count(path: Path, entry: str, frame_idx: int) -> int:
    """Sum of all ``matched_*`` row counts under one frame group."""
    frame_key = f"frame{frame_idx:05d}"
    total = 0
    with h5py.File(path, "r") as f:
        grp = f.get(f"{entry}/data/analysis/{frame_key}")
        if grp is None:
            return 0
        for name in grp.keys():
            if name.startswith("matched_"):
                total += int(grp[name].shape[0])
    return total


def test_run_fitting_clears_matched_on_pinned_entry(tmp_path, monkeypatch):
    """A ``run_fitting`` command pinned to one entry / all frames must
    drop every matched_* row on that entry before mlgidbase runs.

    Monkeypatch mlgidbase.mlgidBASE so the test doesn't depend on
    the private backend — we only want to exercise the pre-flight."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            pass

        def run_fitting(self, **kwargs):
            return None

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)

    path = _seed_file_with_matched(tmp_path / "fit_clears.h5")
    assert _matched_row_count(path, "entry_0000", 0) == 3, (
        "fixture should have seeded 3 matched rows on frame 0"
    )

    execute(path, PipelineCommand("run_fitting", {"entry": "entry_0000"}))

    assert _matched_row_count(path, "entry_0000", 0) == 0, (
        "matched rows on frame 0 must be cleared before run_fitting"
    )


def test_run_fitting_single_frame_only_clears_that_frame(tmp_path, monkeypatch):
    """When ``frame_num`` pins a single frame, matched rows on other
    frames in the same entry must NOT be touched. Mirrors the scope
    semantics of the Tools → Reset → Active-frame action."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            pass

        def run_fitting(self, **kwargs):
            return None

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)

    path = tmp_path / "frame_scoped.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry = f.create_group("entry_0000", track_order=True)
        d = entry.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((3, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        analysis = d.create_group("analysis", track_order=True)
        dtype = np.dtype(
            [("CIF", "S32"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
             ("peak_list", "S16"), ("probability", "f4")]
        )
        rows = np.zeros(2, dtype=dtype)
        rows["CIF"] = [b"a.cif", b"b.cif"]
        rows["peak_list"] = [b"[0]", b"[1]"]
        # Frame 0 + frame 1 both have matched rows seeded.
        for frame_idx in (0, 1):
            frame_g = analysis.create_group(
                f"frame{frame_idx:05d}", track_order=True
            )
            frame_g.create_dataset("matched_segments_0000", data=rows)

    assert _matched_row_count(path, "entry_0000", 0) == 2
    assert _matched_row_count(path, "entry_0000", 1) == 2

    # Refit only frame 1.
    execute(
        path,
        PipelineCommand(
            "run_fitting", {"entry": "entry_0000", "frame_num": 1}
        ),
    )

    # Frame 1 wiped; frame 0 untouched (different frame, not in scope).
    assert _matched_row_count(path, "entry_0000", 0) == 2
    assert _matched_row_count(path, "entry_0000", 1) == 0


def test_run_fitting_all_entries_clears_all(tmp_path, monkeypatch):
    """When ``entry`` is not pinned, matched_* on every q-image entry
    must be invalidated — the "All entries" scope in the GUI expands
    to per-entry commands, but a command that never had ``entry``
    pinned should still clear every entry's matched rows."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            pass

        def run_fitting(self, **kwargs):
            return None

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)

    path = tmp_path / "all_entries.h5"
    rng = np.random.default_rng(0)
    dtype = np.dtype(
        [("CIF", "S32"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
         ("peak_list", "S16"), ("probability", "f4")]
    )
    rows = np.zeros(1, dtype=dtype)
    rows["CIF"] = [b"a.cif"]
    rows["peak_list"] = [b"[0]"]
    with h5py.File(path, "w", track_order=True) as f:
        for entry_name in ("entry_alpha", "entry_beta"):
            entry = f.create_group(entry_name, track_order=True)
            d = entry.create_group("data", track_order=True)
            d.attrs["signal"] = "img_gid_q"
            d.create_dataset(
                "img_gid_q", data=rng.random((1, 8, 8), dtype=np.float32)
            )
            d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
            d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
            analysis = d.create_group("analysis", track_order=True)
            frame0 = analysis.create_group("frame00000", track_order=True)
            frame0.create_dataset("matched_segments_0000", data=rows)

    for name in ("entry_alpha", "entry_beta"):
        assert _matched_row_count(path, name, 0) == 1

    execute(path, PipelineCommand("run_fitting", {}))  # all entries scope

    for name in ("entry_alpha", "entry_beta"):
        assert _matched_row_count(path, name, 0) == 0, (
            f"matched on {name} should have been invalidated"
        )


def test_run_matching_does_not_touch_matched(tmp_path, monkeypatch):
    """The invalidation is gated on ``run_fitting`` only. A
    ``run_matching`` command must NOT clear matched_* on entry —
    matching writes new matched_* rows, but the existing ones from
    a prior run shouldn't be pre-emptively wiped by our pre-flight."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            pass

        def run_matching(self, **kwargs):
            return None

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)

    path = _seed_file_with_matched(tmp_path / "match_keeps.h5")
    assert _matched_row_count(path, "entry_0000", 0) == 3

    execute(
        path,
        PipelineCommand(
            "run_matching",
            {
                "entry": "entry_0000",
                "cif_prepr": None,
                "peaks_type": "segments",
            },
        ),
    )

    # run_matching's own write would replace matched_*; here we
    # stubbed it out, so the seeded rows should be untouched.
    assert _matched_row_count(path, "entry_0000", 0) == 3


# -- matched peak_list kinds around a matching run --------------------------


_FIT_DTYPE = np.dtype([
    ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
    ("radius", "f4"), ("radius_width", "f4"), ("q_z", "f4"), ("q_xy", "f4"),
    ("is_ring", "bool"), ("id", "i4"),
])
_MATCHED_DTYPE = np.dtype([
    ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
    ("probability", "f4"), ("peak_list", h5py.vlen_dtype(np.int32)),
])


def _matched_rows(plists, cif=b"A"):
    rows = np.empty(len(plists), dtype=_MATCHED_DTYPE)
    for i, plist in enumerate(plists):
        rows["CIF"][i], rows["h"][i], rows["k"][i], rows["l"][i] = cif, i, 0, 0
        rows["probability"][i] = 0.5
        rows["peak_list"][i] = np.asarray(plist, dtype=np.int32)
    return rows


def _seed_fitted_with_old_rings(path: Path) -> Path:
    """One frame, fitted ids [5, 2, 9, 7], and an OLD positional, unstamped
    ``matched_rings_0000`` (positions 0 and 3 = ids 5 and 7)."""
    with h5py.File(path, "w", track_order=True) as f:
        d = f.create_group("entry_0000/data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset("img_gid_q", data=np.zeros((1, 8, 8), np.float32))
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        g = d.create_group("analysis/frame00000", track_order=True)
        fit = np.zeros(4, dtype=_FIT_DTYPE)
        fit["id"] = [5, 2, 9, 7]
        g.create_dataset("fitted_peaks", data=fit)
        g.create_dataset("matched_rings_0000", data=_matched_rows([[0, 3]]))
    return path


def _group(path: Path):
    return h5py.File(path, "r")


def _install_matching_stub(monkeypatch, writes):
    """Stand in for mlgidbase's run_matching: ``writes`` is a list of
    ``(dataset name, plists)`` it creates (fresh, so unstamped) on frame 0,
    the way pygid's saver does."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            self.filename = filename

        def run_matching(self, **kwargs):
            with h5py.File(self.filename, "r+") as f:
                g = f["entry_0000/data/analysis/frame00000"]
                for name, plists in writes:
                    if name in g:
                        del g[name]
                    g.create_dataset(name, data=_matched_rows(plists))

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)


def test_run_matching_stamps_new_datasets_and_migrates_old_ones(
    tmp_path, monkeypatch,
):
    """A segments run next to old positional rings: the new segments
    (ids, as mlgidbase 0.1.11 writes them) get the id stamp unchanged,
    and the old rings are translated to ids BEFORE the run so the stamp
    cannot mislabel them."""
    path = _seed_fitted_with_old_rings(tmp_path / "match_stamp.h5")
    _install_matching_stub(monkeypatch, [("matched_segments_0000", [[9, 2]])])
    monkeypatch.setattr(
        "mlgidgui.pipeline.backend_peak_list_kind", lambda: "id",
    )
    execute(path, PipelineCommand(
        "run_matching", {"entry": "entry_0000", "frame_num": 0},
    ))
    with _group(path) as f:
        g = f["entry_0000/data/analysis/frame00000"]
        seg, rings = g["matched_segments_0000"], g["matched_rings_0000"]
        assert seg.attrs["peak_list_kind"] == "id"
        assert [int(v) for v in seg[()]["peak_list"][0]] == [9, 2]
        assert rings.attrs["peak_list_kind"] == "id"
        assert [int(v) for v in rings[()]["peak_list"][0]] == [5, 7]


def test_run_matching_that_finds_nothing_still_stamps_the_old_data(
    tmp_path, monkeypatch,
):
    path = _seed_fitted_with_old_rings(tmp_path / "match_nothing.h5")
    _install_matching_stub(monkeypatch, [])
    monkeypatch.setattr(
        "mlgidgui.pipeline.backend_peak_list_kind", lambda: "id",
    )
    execute(path, PipelineCommand("run_matching", {"entry": "entry_0000"}))
    with _group(path) as f:
        rings = f["entry_0000/data/analysis/frame00000/matched_rings_0000"]
        assert rings.attrs["peak_list_kind"] == "id"
        assert [int(v) for v in rings[()]["peak_list"][0]] == [5, 7]


def test_run_matching_with_an_older_backend_stamps_positions(
    tmp_path, monkeypatch,
):
    path = _seed_fitted_with_old_rings(tmp_path / "match_old_backend.h5")
    _install_matching_stub(monkeypatch, [("matched_segments_0000", [[0, 1]])])
    monkeypatch.setattr(
        "mlgidgui.pipeline.backend_peak_list_kind", lambda: "position",
    )
    execute(path, PipelineCommand("run_matching", {"entry": "entry_0000"}))
    with _group(path) as f:
        g = f["entry_0000/data/analysis/frame00000"]
        assert g["matched_segments_0000"].attrs["peak_list_kind"] == "position"
        # The old dataset is left exactly as it was; the reader decides.
        assert [int(v) for v in g["matched_rings_0000"][()]["peak_list"][0]] == [0, 3]


def test_backend_peak_list_kind_follows_the_installed_mlgidbase(monkeypatch):
    from importlib import metadata

    from mlgidgui.pipeline import backend_peak_list_kind

    real = metadata.version
    for installed, expected in (
        ("0.1.8", "position"), ("0.1.10", "position"),
        ("0.1.11", "id"), ("0.2.0", "id"), ("0.1.12.dev3", "id"),
    ):
        monkeypatch.setattr(
            metadata, "version",
            lambda name, _v=installed: _v if name == "mlgidbase" else real(name),
        )
        assert backend_peak_list_kind() == expected, installed


def test_run_fitting_also_drops_the_saved_tracking_table(tmp_path, monkeypatch):
    """A tracking table describes the fits a refit replaces."""
    mlgidbase = pytest.importorskip("mlgidbase")

    class _StubAnalysis:
        def __init__(self, filename):
            pass

        def run_fitting(self, **kwargs):
            return None

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)
    path = _seed_file_with_matched(tmp_path / "fit_clears_tracks.h5")
    with h5py.File(path, "r+") as f:
        g = f.require_group("entry_0000/data/analysis/general")
        g.create_dataset("tracked_peaks", data=np.zeros(1))
    execute(path, PipelineCommand("run_fitting", {"entry": "entry_0000"}))
    with h5py.File(path, "r") as f:
        assert "general" not in f["entry_0000/data/analysis"]
