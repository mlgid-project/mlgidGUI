"""Host flow of "Add selected peaks (fit + match)" (Expected pattern).

Environment-independent: ``matching_kwargs`` / ``cif_source_text`` are
stubbed on the panel instance (the real widgets don't exist
backend-less), the options dialog is stubbed to a fixed answer, and
``_enqueue_pipeline`` captures commands instead of running them.
Matched solutions are written into the H5 file directly — the planner
and the restricted re-match read the FILE, not the viewer.

Source: main_window.py ``_on_add_predicted_peaks`` /
``_plan_predicted_fills`` / ``_build_predicted_match_commands`` /
``_finalize_predicted_fill`` / ``_update_sim_add_button``.
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
    return SimpleNamespace(
        cifs=["In2O3.cif", "PbI2.cif"],
        pattern_3d=SimpleNamespace(orientations=[
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ]),
        all_patterns_q2d=[
            [np.array([[0.5, 0.5], [1.0, 0.2], [0.0, 1.2]])],
            [np.array([[0.7, 0.7]])],
        ],
        all_patterns_int2d=[
            [np.array([40.0, 100.0, 5.0])],
            [np.array([9.0])],
        ],
        all_patterns_q1d=[np.array([0.8, 0.4, 1.6]), np.array([1.1])],
        all_patterns_int1d=[np.array([2.0, 8.0, 4.0]), np.array([3.0])],
    )


def _write_matched(path, cif: str, hkl: tuple, peak_list, frame: int = 0):
    """Write one matched_segments solution row (the pygid schema)."""
    import h5py

    dt = np.dtype([
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"),
        ("peak_list", h5py.vlen_dtype(np.int32)),
    ])
    arr = np.zeros(1, dtype=dt)
    arr["CIF"][0] = cif.encode()
    arr["h"][0], arr["k"][0], arr["l"][0] = hkl
    arr["probability"][0] = 0.9
    arr["peak_list"][0] = np.asarray(peak_list, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f[f"{ENTRY}/data/analysis/frame{frame:05d}"]
        name = "matched_segments_0000"
        if name in g:
            del g[name]
        g.create_dataset(name, data=arr)


@pytest.fixture
def harness(main_window, synthetic_nexus_with_peaks, tmp_path, monkeypatch):
    """Window on the peaks fixture with the overlay armed and every
    external seam stubbed. Yields (window, enqueued, state)."""
    mw = main_window
    # The Expected-pattern section container is gated on the matching
    # backend (`_update_sim_section_state`); stub availability BEFORE
    # the session/cache installs below re-evaluate it, so the widget
    # enable-state assertions hold on backend-less boxes (CI) too.
    from mlgidgui import pipeline as _pipeline_mod
    monkeypatch.setattr(_pipeline_mod, "is_mlgidbase_available", lambda: True)
    mw._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)

    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for name in ("In2O3.cif", "PbI2.cif"):
        (cif_dir / name).write_text("# stub\n")

    state = {"scope": "restricted", "frames": (0, 0), "warnings": []}
    fake_kwargs = {
        "cif_prepr": object(),  # pre-parsed-object sentinel (not str)
        "peaks_type": "segments", "threshold": 0.5,
        "intensity_threshold": 0.1, "device": "cpu",
    }
    state["match_kwargs"] = fake_kwargs
    monkeypatch.setattr(
        mw.pipeline_panel, "matching_kwargs", lambda: dict(fake_kwargs)
    )
    monkeypatch.setattr(
        mw.pipeline_panel, "cif_source_text", lambda: str(cif_dir)
    )
    monkeypatch.setattr(
        mw, "_ask_predicted_add_options",
        lambda pattern, frame, n_frames: (
            None if state["scope"] is None else
            {
                "scope": state["scope"],
                "frames": state["frames"],
                "cifs": state.get("cifs"),
                "cap": state.get("cap"),
            }
        ),
    )
    monkeypatch.setattr(mw, "_view_worker_blocks_pipeline", lambda: False)

    from mlgidgui import main_window as mw_mod
    monkeypatch.setattr(
        mw_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: state["warnings"].append(a)),
    )
    # The synthetic fixture has no instrument metadata; the geometry
    # guard would otherwise veto every add.
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


def _select_all(mw) -> list[int]:
    return mw.viewer.select_missed_simulated(mw.viewer.current_frame)


def _pick_oriented(mw) -> None:
    """Select the (1 1 1) orientation through the user-specified mode
    (no matched entry exists for it in this harness)."""
    mw._sim_orient_mode.setCurrentIndex(
        mw._combo_data_index(mw._sim_orient_mode, "user")
    )
    mw._sim_hkl_edit.setText("1 1 1")
    mw._on_sim_hkl_edited()


def test_add_spots_restricted_scope(harness):
    mw, enqueued, state = harness
    # A matched PbI2 solution in the FILE joins the restricted set
    # (the planner reads matched state from disk, not the viewer).
    # Writes to the temp file need the app's own detach dance — the
    # session holds read handles otherwise.
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, "PbI2", (1, 0, 0), [0])
    _pick_oriented(mw)
    assert _select_all(mw) == [0, 1, 2]
    assert mw._sim_add_btn.isEnabled()

    mw._on_add_predicted_peaks()

    assert [c.op_name for c in enqueued] == [
        "inject_fitted_peaks", "run_matching",
    ]
    inject = enqueued[0].kwargs
    assert inject["entry"] == ENTRY
    assert sorted(inject["plan"]) == [0]
    specs = inject["plan"][0]
    assert len(specs) == 3
    refl = mw.viewer.simulation_pattern().reflections
    # Box sizes seed from the frame's median fitted widths (0.3 / 4.0).
    for spec, r in zip(specs, refl):
        assert spec["radius"] == pytest.approx(r.radius)
        assert spec["angle"] == pytest.approx(r.angle)
        assert spec["radius_width"] == pytest.approx(0.3)
        assert spec["angle_width"] == pytest.approx(4.0)
        assert spec["is_ring"] is False
    match = enqueued[1].kwargs
    assert match["frame_num"] == [0]
    assert match["peaks_type"] == "segments"
    # Restricted: exactly the overlay CIF + the frame's matched CIF.
    assert isinstance(match["cif_prepr"], str)
    assert "In2O3.cif" in match["cif_prepr"]
    assert "PbI2.cif" in match["cif_prepr"]
    # Chain bookkeeping under the Expected-pattern label, plus the
    # validation stash awaiting the fill records.
    chain = mw._interp_chain
    assert chain is not None
    assert chain["label"] == "Expected pattern"
    assert chain["total"] == 2
    assert mw._interp_fill_result is None
    stash = dict(mw._predicted_fill)
    # The pre-run matched snapshot (merged back at finalize) covers
    # every frame queued for re-matching.
    snapshot = stash.pop("snapshot")
    assert set(snapshot) == {0}
    assert set(snapshot[0]) == {"matched_segments_0000"}
    assert stash == {
        "entry": ENTRY, "cif": "in2o3", "accept": {"in2o3"},
        "records": None, "match_ok": True,
    }
    # The progress dialog reads the new label.
    mw._pipe_command = enqueued[0]
    mw._on_pipeline_frame_progress(1, 1, "inject_fitted_peaks", ENTRY)
    assert mw._track_progress_dialog.labelText().startswith(
        "Expected pattern: fitting injected boxes"
    )
    assert mw._track_progress_dialog.value() == 50


def test_add_powder_rings_matches_rings(harness):
    mw, enqueued, state = harness
    # No matched orientations -> the auto mode default is the random
    # (powder) pattern.
    assert mw._sim_orient_mode.currentData() == "random"
    assert mw._sim_selected_hkl() == (0, 0, 0)
    assert _select_all(mw) == [0, 1, 2]

    mw._on_add_predicted_peaks()

    assert [c.op_name for c in enqueued] == [
        "inject_fitted_peaks", "run_matching",
    ]
    specs = enqueued[0].kwargs["plan"][0]
    assert all(s["is_ring"] for s in specs)
    # Ring boxes use the finite quadrant-spanning fit geometry.
    assert all(s["angle_width"] == pytest.approx(90.0) for s in specs)
    assert enqueued[1].kwargs["peaks_type"] == "rings"


def test_range_skips_covered_and_dataset_less_frames(harness):
    mw, enqueued, state = harness
    _pick_oriented(mw)
    # A fitted row covering reflection 0 (radius ~0.707, angle 45):
    # that reflection must be skipped on frame 0.
    with mw._detached_silx_tree():
        file_model.add_fitted_peak_row(
            mw.session.temp_path, ENTRY, 0,
            radius=0.707, radius_width=0.3, angle=45.0, angle_width=10.0,
            amplitude=5.0,
        )
    state["frames"] = (0, 2)  # frames 1 & 2 have no analysis datasets
    logs: list = []
    orig_log = mw.pipeline_panel.append_log
    mw.pipeline_panel.append_log = lambda m: (logs.append(m), orig_log(m))
    _select_all(mw)

    mw._on_add_predicted_peaks()

    inject = enqueued[0].kwargs
    assert sorted(inject["plan"]) == [0]
    # Reflection 0 is covered on frame 0 -> only two boxes remain.
    assert len(inject["plan"][0]) == 2
    assert enqueued[1].kwargs["frame_num"] == [0]
    assert any("skipping frame(s)" in m and "[1, 2]" in m for m in logs)
    # Inject command ticks once per planned frame.
    assert mw._interp_chain["ticks"][id(enqueued[0])] == 1


def test_add_cap_keeps_strongest_selected(harness):
    """The options dialog's reflection cap trims the selection to its
    strongest members before planning."""
    mw, enqueued, state = harness
    state["cap"] = 1
    _pick_oriented(mw)
    assert _select_all(mw) == [0, 1, 2]

    mw._on_add_predicted_peaks()

    # Pattern intensities are 40/100/5 — the cap keeps the rel-1.0
    # reflection at (1.0, 0.2), radius ~1.020.
    specs = enqueued[0].kwargs["plan"][0]
    assert len(specs) == 1
    assert specs[0]["radius"] == pytest.approx(
        float(np.hypot(1.0, 0.2)), abs=1e-3
    )


def test_rematch_only_when_fitted_but_unclaimed(harness):
    """Predicted positions already holding fitted-but-unmatched peaks
    get NO duplicate boxes — the frame is only re-matched so the
    matcher can claim the existing peaks."""
    mw, enqueued, state = harness
    _pick_oriented(mw)
    # Cover all three reflections (r~0.707/a45, r~1.020/a11.3,
    # r1.2/a90) with plain fitted rows, none attributed to a structure.
    with mw._detached_silx_tree():
        for radius, angle in ((0.707, 45.0), (1.02, 11.3), (1.2, 90.0)):
            file_model.add_fitted_peak_row(
                mw.session.temp_path, ENTRY, 0,
                radius=radius, radius_width=0.3, angle=angle,
                angle_width=10.0, amplitude=5.0,
            )
    logs: list = []
    orig_log = mw.pipeline_panel.append_log
    mw.pipeline_panel.append_log = lambda m: (logs.append(m), orig_log(m))
    # Select missed still selects them all: fitted != matched.
    assert _select_all(mw) == [0, 1, 2]

    mw._on_add_predicted_peaks()

    assert [c.op_name for c in enqueued] == ["run_matching"]
    assert enqueued[0].kwargs["frame_num"] == [0]
    assert enqueued[0].kwargs["peaks_type"] == "segments"
    assert any("re-match only" in m and "[0]" in m for m in logs)
    # Nothing injected -> nothing to validate or roll back.
    assert mw._predicted_fill["records"] is None
    assert mw._interp_chain["ticks"] == {id(enqueued[0]): 1}
    mw._finalize_predicted_fill()  # records None -> clean no-op


def test_custom_scope_uses_chosen_cifs(harness):
    mw, enqueued, state = harness
    state["scope"] = "custom"
    state["cifs"] = {"pbi2"}
    # A matched In2O3 solution exists on the frame — the custom scope
    # must NOT pull it into the match source (the snapshot merge-back
    # keeps the identification alive instead).
    with mw._detached_silx_tree():
        _write_matched(mw.session.temp_path, "In2O3", (1, 1, 1), [0])
    _pick_oriented(mw)
    _select_all(mw)

    mw._on_add_predicted_peaks()

    match = enqueued[1].kwargs
    # EXACTLY the chosen CIF is in the restricted source, and it
    # becomes the accept-set.
    assert isinstance(match["cif_prepr"], str)
    assert "PbI2.cif" in match["cif_prepr"]
    assert "In2O3.cif" not in match["cif_prepr"]
    assert mw._predicted_fill["accept"] == {"pbi2"}
    # The frame's existing solution is snapshotted for the merge-back.
    assert set(mw._predicted_fill["snapshot"][0]) == {
        "matched_segments_0000"
    }


def test_full_scope_keeps_base_source(harness):
    mw, enqueued, state = harness
    state["scope"] = "full"
    _select_all(mw)

    mw._on_add_predicted_peaks()

    match = enqueued[1].kwargs
    assert match["cif_prepr"] is state["match_kwargs"]["cif_prepr"]


def test_restricted_falls_back_to_full_when_unsubsettable(
    harness, monkeypatch,
):
    mw, enqueued, state = harness
    # No raw source text and a non-string cif_prepr: nothing to subset.
    monkeypatch.setattr(mw.pipeline_panel, "cif_source_text", lambda: "")
    logs: list = []
    monkeypatch.setattr(mw.pipeline_panel, "append_log", logs.append)
    _select_all(mw)

    mw._on_add_predicted_peaks()

    match = enqueued[1].kwargs
    assert match["cif_prepr"] is state["match_kwargs"]["cif_prepr"]
    assert any("FULL source" in line for line in logs)


def test_cancel_enqueues_nothing(harness):
    mw, enqueued, state = harness
    state["scope"] = None
    _select_all(mw)

    mw._on_add_predicted_peaks()

    assert enqueued == []
    assert mw._interp_chain is None
    assert mw._predicted_fill is None


def test_guards_busy_and_empty_plan(harness):
    mw, enqueued, state = harness
    _select_all(mw)

    # Busy pipeline: silently refuses.
    mw._pipe_thread = object()
    mw._on_add_predicted_peaks()
    assert enqueued == [] and not state["warnings"]
    mw._pipe_thread = None

    # A range consisting only of dataset-less frames plans nothing:
    # status message, no enqueue, no chain.
    state["frames"] = (1, 2)
    mw._on_add_predicted_peaks()
    assert enqueued == []
    assert mw._interp_chain is None
    assert not state["warnings"]

    # The Add button reads disabled on such a frame too.
    mw.viewer._frame_index = 1
    mw._refresh_sim_matched_entries(1, [])
    assert not mw._sim_add_btn.isEnabled()
    assert "detection + fitting" in mw._sim_add_btn.toolTip().lower()


def test_no_match_source_warns(harness, monkeypatch):
    mw, enqueued, state = harness
    monkeypatch.setattr(
        mw.pipeline_panel, "matching_kwargs", lambda: None
    )
    _select_all(mw)

    mw._on_add_predicted_peaks()

    assert enqueued == []
    assert len(state["warnings"]) == 1


def test_add_button_disabled_without_selection(harness):
    mw, enqueued, state = harness
    assert not mw._sim_add_btn.isEnabled()
    _select_all(mw)
    assert mw._sim_add_btn.isEnabled()
    mw.viewer.clear_simulation_selection()
    assert not mw._sim_add_btn.isEnabled()


def _inject_rows(path):
    """Two detected+fitted pairs, as the executor would write them."""
    pairs = []
    for radius, angle in ((0.707, 45.0), (1.02, 11.3)):
        det_id = file_model.add_detected_peak_row(
            path, ENTRY, 0, angle=angle, angle_width=4.0,
            radius=radius, radius_width=0.3, score=1.0,
        )
        fit_id = file_model.add_fitted_peak_row(
            path, ENTRY, 0, radius=radius, radius_width=0.3,
            angle=angle, angle_width=4.0, amplitude=9.0,
        )
        pairs.append({
            "frame": 0, "detected_id": int(det_id),
            "fitted_id": int(fit_id), "is_ring": False,
        })
    return pairs


def test_finalize_keeps_matched_discards_rest(harness):
    mw, enqueued, state = harness
    path = mw.session.temp_path
    with mw._detached_silx_tree():
        records = _inject_rows(path)
        # Baseline fitted rows occupy positions 0-1; the injected pair
        # sits at positions 2-3. The target claims only the FIRST.
        _write_matched(path, "In2O3", (1, 1, 1), [2])
    mw._predicted_fill = {
        "entry": ENTRY, "cif": "in2o3",
        "records": records, "match_ok": True,
    }

    # In the app this runs at chain completion, while silx is still
    # detached — reproduce that state for the r+ rollback writes.
    with mw._detached_silx_tree():
        mw._finalize_predicted_fill()

    assert mw._predicted_fill is None
    tables = file_model.load_peaks(path, ENTRY, 0)
    fit_ids = set(int(i) for i in tables["fitted"].ids)
    det_ids = set(int(i) for i in tables["detected"].ids)
    # Kept: the matched pair. Discarded: fitted AND detected of the
    # unmatched one.
    assert records[0]["fitted_id"] in fit_ids
    assert records[0]["detected_id"] in det_ids
    assert records[1]["fitted_id"] not in fit_ids
    assert records[1]["detected_id"] not in det_ids
    # The solution still resolves to the kept peak after the remap.
    structures = file_model.load_matched_peaks(
        path, ENTRY, 0, tables["fitted"]
    )
    assert len(structures) == 1
    assert [int(i) for i in structures[0].peaks.ids] == [
        records[0]["fitted_id"]
    ]


def test_finalize_match_error_keeps_everything(harness):
    mw, enqueued, state = harness
    path = mw.session.temp_path
    with mw._detached_silx_tree():
        records = _inject_rows(path)
    mw._predicted_fill = {
        "entry": ENTRY, "cif": "in2o3",
        "records": records, "match_ok": False,
    }

    mw._finalize_predicted_fill()

    tables = file_model.load_peaks(path, ENTRY, 0)
    fit_ids = set(int(i) for i in tables["fitted"].ids)
    assert {r["fitted_id"] for r in records} <= fit_ids
