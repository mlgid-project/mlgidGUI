"""Host flow of "Find all matched structures (all frames)" (Expected
pattern sweep).

Environment-independent, same seams as test_add_predicted_gui: panel
kwargs/source stubbed on the instance, the confirmation dialog stubbed
to a fixed answer, ``_enqueue_pipeline`` capturing commands. Matched
solutions are written into the H5 file directly — both the combo
collector and the planner read the FILE.

Source: main_window.py ``_on_fill_all_matched`` /
``_collect_matched_combos`` / ``_plan_matched_sweep`` /
``_confirm_matched_sweep``; simulation_pattern.py ``specs_to_boxes``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui

ENTRY = "entry_0000"


def _fake_cifpattern():
    """Two CIFs whose oriented patterns SHARE the (0.5, 0.5) position —
    the sweep must not queue that box twice."""
    return SimpleNamespace(
        cifs=["StructA.cif", "StructB.cif"],
        pattern_3d=SimpleNamespace(orientations=[
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ]),
        all_patterns_q2d=[
            [np.array([[0.5, 0.5], [1.0, 0.2]])],
            [np.array([[0.5, 0.5]])],
        ],
        all_patterns_int2d=[
            [np.array([40.0, 100.0])],
            [np.array([9.0])],
        ],
        all_patterns_q1d=[np.array([0.8]), np.array([1.1])],
        all_patterns_int1d=[np.array([2.0]), np.array([3.0])],
    )


def _write_matched(path, rows, frame: int = 0):
    """Write matched_segments solution rows; each row is
    (cif, hkl, peak_list)."""
    import h5py

    dt = np.dtype([
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"),
        ("peak_list", h5py.vlen_dtype(np.int32)),
    ])
    arr = np.zeros(len(rows), dtype=dt)
    for i, (cif, hkl, peak_list) in enumerate(rows):
        arr["CIF"][i] = cif.encode()
        arr["h"][i], arr["k"][i], arr["l"][i] = hkl
        arr["probability"][i] = 0.9
        arr["peak_list"][i] = np.asarray(peak_list, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f[f"{ENTRY}/data/analysis/frame{frame:05d}"]
        name = "matched_segments_0000"
        if name in g:
            del g[name]
        g.create_dataset(name, data=arr)


@pytest.fixture
def harness(main_window, synthetic_nexus_with_peaks, tmp_path, monkeypatch):
    """Window on the peaks fixture (3 frames, analysis on frame 0 only)
    with every external seam stubbed. Yields (window, enqueued, state)."""
    mw = main_window
    mw._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)

    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for name in ("StructA.cif", "StructB.cif"):
        (cif_dir / name).write_text("# stub\n")

    state = {"confirm": True, "confirm_calls": [], "warnings": []}
    fake_kwargs = {
        "cif_prepr": object(),
        "peaks_type": "segments", "threshold": 0.5,
        "intensity_threshold": 0.1, "device": "cpu",
    }
    monkeypatch.setattr(
        mw.pipeline_panel, "matching_kwargs", lambda: dict(fake_kwargs)
    )
    monkeypatch.setattr(
        mw.pipeline_panel, "cif_source_text", lambda: str(cif_dir)
    )
    def fake_confirm(targets, uncomputable, n_frames, n_ready):
        state["confirm_calls"].append(
            (targets, uncomputable, n_frames, n_ready)
        )
        if not state["confirm"]:
            return None
        caps = state.get("caps")
        return list(caps) if caps is not None else [10] * len(targets)

    monkeypatch.setattr(mw, "_confirm_matched_sweep", fake_confirm)
    monkeypatch.setattr(mw, "_view_worker_blocks_pipeline", lambda: False)

    from mlgidgui import main_window as mw_mod
    monkeypatch.setattr(
        mw_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: state["warnings"].append(a)),
    )
    monkeypatch.setattr(
        mw_mod.file_model, "read_geometry_for_entry",
        lambda path, entry, frame=0: {
            "wavelength_angstrom": 1.0, "q_xy_max": 3.0,
            "q_z_max": 4.0, "ai_deg": 0.3,
            "q_z_axis": np.linspace(0.0, 4.0, 16),
        },
    )

    enqueued: list = []
    monkeypatch.setattr(
        mw, "_enqueue_pipeline", lambda path, cmd: enqueued.append(cmd)
    )
    yield mw, enqueued, state
    mw._close_tracking_progress()
    mw._interp_chain = None
    mw._predicted_fill = None


def test_sweep_combos_dedupe_and_custom_match(harness):
    mw, enqueued, state = harness
    # Two structures matched on frame 0 (their boxes are the baseline
    # fitted rows at (1.5, 20) / (2.5, 60) — neither explains any
    # predicted position).
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, [
            ("StructA", (1, 1, 1), [0]),
            ("StructB", (1, 0, 0), [1]),
        ])
    logs: list = []
    orig_log = mw.pipeline_panel.append_log
    mw.pipeline_panel.append_log = lambda m: (logs.append(m), orig_log(m))

    mw._on_fill_all_matched()

    targets, uncomputable, n_frames, n_ready = state["confirm_calls"][0]
    assert [t["label"] for t in targets] == [
        "structa (1 1 1)", "structb (1 0 0)"
    ]
    assert uncomputable == [] and n_frames == 3
    # Only frame 0 carries detected+fitted datasets — the dialog's
    # workload estimate scales with that count.
    assert n_ready == 1

    assert [c.op_name for c in enqueued] == [
        "inject_fitted_peaks", "run_matching",
    ]
    inject = enqueued[0].kwargs
    assert sorted(inject["plan"]) == [0]
    # StructA contributes (0.707, 45) and (1.02, 11.3); StructB's only
    # reflection sits at (0.707, 45) too — deduped against the queued
    # spec, so exactly two boxes.
    specs = inject["plan"][0]
    assert len(specs) == 2
    assert sorted(round(s["radius"], 3) for s in specs) == [0.707, 1.02]
    match = enqueued[1].kwargs
    assert match["frame_num"] == [0]
    assert match["peaks_type"] == "segments"
    assert isinstance(match["cif_prepr"], str)
    assert "StructA.cif" in match["cif_prepr"]
    assert "StructB.cif" in match["cif_prepr"]
    # Validation accepts BOTH swept structures; frames 1-2 lack
    # datasets and are skipped with a log note.
    assert mw._predicted_fill["accept"] == {"structa", "structb"}
    assert mw._interp_chain["label"] == "Expected pattern"
    assert mw._interp_chain["total"] == 2
    assert any("skipping frame(s)" in m and "[1, 2]" in m for m in logs)
    # The pre-run solutions are snapshotted for the finalize merge-back.
    snap = mw._predicted_fill["snapshot"]
    assert set(snap) == {0}
    assert set(snap[0]) == {"matched_segments_0000"}
    assert len(snap[0]["matched_segments_0000"]) == 2


def test_sweep_cap_keeps_strongest_reflections(harness):
    mw, enqueued, state = harness
    state["caps"] = [1]  # the dialog's per-structure spinbox answer
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, [
            ("StructA", (1, 1, 1), [0]),
        ])

    mw._on_fill_all_matched()

    # StructA predicts (0.5, 0.5) rel 0.4 and (1.0, 0.2) rel 1.0 —
    # candidates are sorted strongest-first, so a cap of 1 keeps the
    # rel-1.0 reflection (the captured target dict is truncated in
    # place after the dialog returns).
    targets = state["confirm_calls"][0][0]
    assert [
        round(r.rel_intensity, 2) for r in targets[0]["reflections"]
    ] == [1.0]
    specs = enqueued[0].kwargs["plan"][0]
    assert len(specs) == 1
    assert specs[0]["radius"] == pytest.approx(
        float(np.hypot(1.0, 0.2)), abs=1e-3
    )


def test_finalize_restores_structures_dropped_by_rematch(harness):
    """A re-match rewrites a frame's solutions from scratch; anything
    it fails to reproduce is merged back from the pre-run snapshot —
    an Add/sweep must never lose existing identifications."""
    mw, enqueued, state = harness
    path = mw.session.temp_path
    with mw._detached_silx_tree():
        _write_matched(path, [
            ("StructA", (1, 1, 1), [0, 1]),
        ])
        snapshot = file_model.read_matched_raw(path, ENTRY, [0])
        # Simulate the re-match replacing the solutions with a subset
        # attribution of a different structure.
        _write_matched(path, [
            ("StructB", (1, 0, 0), [1]),
        ])
    mw._predicted_fill = {
        "entry": ENTRY, "cif": "structa", "accept": {"structa"},
        "records": None, "match_ok": True, "snapshot": snapshot,
    }

    with mw._detached_silx_tree():
        mw._finalize_predicted_fill()

    tables = file_model.load_peaks(path, ENTRY, 0)
    structures = file_model.load_matched_peaks(
        path, ENTRY, 0, tables["fitted"]
    )
    got = {(s.cif, s.h, s.k, s.l): list(s.peak_list) for s in structures}
    assert got == {
        ("StructA", 1, 1, 1): [0, 1],
        ("StructB", 1, 0, 0): [1],
    }


def test_sweep_powder_combo_injects_rings(harness):
    mw, enqueued, state = harness
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, [
            ("StructB", (0, 0, 0), [0]),
        ])

    mw._on_fill_all_matched()

    targets = state["confirm_calls"][0][0]
    assert [t["label"] for t in targets] == ["structb powder"]
    inject = enqueued[0].kwargs
    specs = inject["plan"][0]
    assert len(specs) == 1
    assert specs[0]["is_ring"] is True
    assert specs[0]["radius"] == pytest.approx(1.1)
    assert enqueued[1].kwargs["peaks_type"] == "rings"


def test_sweep_rematch_only_when_fitted_covered(harness):
    """Predicted positions already holding fitted-but-unmatched peaks:
    no boxes injected, the frame is only re-matched."""
    mw, enqueued, state = harness
    with mw._detached_silx_tree():
        for radius, angle in ((0.707, 45.0), (1.02, 11.3)):
            file_model.add_fitted_peak_row(
                mw.session.temp_path, ENTRY, 0,
                radius=radius, radius_width=0.3, angle=angle,
                angle_width=10.0, amplitude=5.0,
            )
        _write_matched(mw.session.temp_path, [
            ("StructA", (1, 1, 1), [0]),
        ])
    logs: list = []
    orig_log = mw.pipeline_panel.append_log
    mw.pipeline_panel.append_log = lambda m: (logs.append(m), orig_log(m))

    mw._on_fill_all_matched()

    assert [c.op_name for c in enqueued] == ["run_matching"]
    assert enqueued[0].kwargs["frame_num"] == [0]
    assert any("re-match only" in m and "[0]" in m for m in logs)
    assert mw._predicted_fill["records"] is None


def test_sweep_cancel_and_no_matches(harness):
    mw, enqueued, state = harness
    # No matched structures anywhere: status message, no confirm, no
    # enqueue.
    mw._on_fill_all_matched()
    assert enqueued == [] and state["confirm_calls"] == []
    assert mw._interp_chain is None and mw._predicted_fill is None

    # Cancelled confirmation: nothing runs either.
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, [
            ("StructA", (1, 1, 1), [0]),
        ])
    state["confirm"] = False
    mw._on_fill_all_matched()
    assert enqueued == [] and len(state["confirm_calls"]) == 1
    assert mw._interp_chain is None and mw._predicted_fill is None


def test_sweep_uncomputable_combo_logged_and_skipped(harness):
    mw, enqueued, state = harness
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, [
            ("Unknown", (1, 1, 1), [0]),
            ("StructA", (1, 1, 1), [1]),
        ])
    logs: list = []
    orig_log = mw.pipeline_panel.append_log
    mw.pipeline_panel.append_log = lambda m: (logs.append(m), orig_log(m))

    mw._on_fill_all_matched()

    targets, uncomputable = state["confirm_calls"][0][:2]
    assert [t["stem"] for t in targets] == ["structa"]
    assert len(uncomputable) == 1
    assert "unknown" in uncomputable[0]
    assert any("cannot simulate" in m for m in logs)
    assert [c.op_name for c in enqueued] == [
        "inject_fitted_peaks", "run_matching",
    ]
    assert mw._predicted_fill["accept"] == {"structa"}
