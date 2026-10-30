"""Scan-tracking dock + phase views on the mlgidBASE tracking engine.

Wiring-level tests run with a FAKE ``TrackingPayload`` (CI-safe, no
backend); the end-to-end test drives the real ``track_peaks`` through
the pipeline queue and skips without ``mlgidbase``/``networkx``.

The fake payload mirrors ``synthetic_fitted_scan`` exactly (persistent
peak on frames 0-5 with fitted id 0 each; the frame-2 blip is not a
surviving track), so id-based selection against the real file works.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.phase_tracking import TrackingPayload
from mlgidgui.pipeline import PipelineCommand
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui

ENTRY = "entry_0000"


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"unexpected blocking QMessageBox in test: {args[1:3]!r}"
        )
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _fake_payload() -> TrackingPayload:
    """The payload the real run over ``synthetic_fitted_scan`` yields:
    one surviving track of the six persistent-peak members (blip cut)."""
    frames = np.arange(6)
    radius = 1.0 + 0.002 * frames
    angle = np.full(6, 45.0)
    return TrackingPayload(
        entry=ENTRY,
        threshold=0.5,
        length=3,
        q_xy=radius * np.cos(np.deg2rad(angle)),
        q_z=radius * np.sin(np.deg2rad(angle)),
        frame_num=frames,
        amplitude=100.0 + frames,
        components=[list(range(6))],
    )


def _install_fake(window, payload: TrackingPayload | None = None) -> TrackingPayload:
    """Install a fake result exactly as ``_on_phase_track_result`` would."""
    payload = payload or _fake_payload()
    ids = [(int(f), 0) for f in payload.frame_num]
    window._scan_payload = payload
    window._scan_member_ids = ids
    window._scan_track_entry = ENTRY
    window._scan_fitted_tables = {
        f: file_model.load_peaks(window.session.temp_path, ENTRY, f)["fitted"]
        for f in range(6)
    }
    window.scan_tracking_panel.set_payload(payload, ids)
    return payload


def test_track_run_end_to_end(main_window, synthetic_fitted_scan, qtbot):
    """Real upstream run through the pipeline queue: payload installed,
    member ids taken from upstream's saved table, panel populated."""
    pytest.importorskip("mlgidbase")
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel
    assert panel.btn_track.isEnabled()

    main_window._on_track_scan_requested(0.5, 3)
    # A loading dialog appears immediately, as a busy marquee (the run
    # is one opaque call — an honest indeterminate bar, range 0..0).
    assert main_window._track_progress_dialog is not None
    assert main_window._track_progress_dialog.maximum() == 0
    qtbot.waitUntil(
        lambda: main_window._pipe_thread is None
        and not main_window._pipeline_queue
        and main_window._scan_payload is not None,
        timeout=60000,
    )
    # ...and it is torn down when the run finishes.
    assert main_window._track_progress_dialog is None
    payload = main_window._scan_payload
    assert payload.n_tracks == 1
    assert payload.track_span(0) == (0, 5, 6, 6)
    # 7 members total (6 persistent + the frame-2 blip, whose 1-member
    # component didn't survive but which is still a payload member).
    assert payload.n_members == 7
    ids = main_window._scan_member_ids
    # Every member of the SURVIVING track matched back to fitted id 0
    # on its frame; the blip matched its own row (frame 2, id 1).
    members = payload.track_members(0)
    assert [ids[int(i)] for i in members] == [(f, 0) for f in range(6)]
    assert (2, 1) in ids
    assert panel._model.rowCount() == 1
    assert "1 tracks" in panel.lbl_status.text()
    assert panel.btn_views.isEnabled()


def test_row_click_jumps_and_selects_fitted(
    main_window, synthetic_fitted_scan,
):
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    panel = main_window.scan_tracking_panel

    panel._table.selectRow(0)
    v = main_window.viewer
    assert v.current_frame == 0
    assert v._selected is not None
    assert v._selected.kind == "fitted"
    assert v._selected.peak_id == 0


def test_reverse_sync_highlights_row_without_bounce(
    main_window, synthetic_fitted_scan,
):
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    panel = main_window.scan_tracking_panel
    v = main_window.viewer

    # Select a MID-track member (frame 3) from the image side.
    v.set_frame(3)
    table = v._frame_peaks[3]["fitted"]
    main_window._select_table_row(3, "fitted", table, 0)
    rows = panel._table.selectionModel().selectedRows()
    assert len(rows) == 1
    assert v.current_frame == 3  # no bounce to the track's first frame

    # Deselect clears the highlight; a detected selection would too.
    v.clear_selection()
    assert panel._table.selectionModel().selectedRows() == []


def test_run_fitting_finish_invalidates_results(
    main_window, synthetic_fitted_scan,
):
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    panel = main_window.scan_tracking_panel
    assert panel._model.rowCount() == 1

    # Simulate a finished run_fitting on the tracked entry (the queue
    # teardown path with no live worker thread).
    main_window._pipe_command = PipelineCommand(
        "run_fitting", {"entry": ENTRY}
    )
    main_window._on_pipeline_finished(None, None)
    assert main_window._scan_payload is None
    assert panel._model.rowCount() == 0
    assert not panel.btn_views.isEnabled()


def test_views_window_renders_fake_payload(
    main_window, synthetic_fitted_scan,
):
    _open(main_window, synthetic_fitted_scan)
    payload = _install_fake(main_window)
    main_window._on_show_phase_views()
    w = main_window._phase_views_window
    assert w is not None

    # Trajectories: one curve per track; axis switch changes the data.
    curves = w.traj_plot.getPlotItem().listDataItems()
    assert len(curves) == payload.n_tracks
    y_radius = np.array(curves[0].yData)
    w.axis_combo.setCurrentIndex(
        [w.axis_combo.itemData(i) for i in range(w.axis_combo.count())]
        .index("amplitude")
    )
    curves = w.traj_plot.getPlotItem().listDataItems()
    y_amp = np.array(curves[0].yData)
    assert not np.allclose(y_radius, y_amp)
    np.testing.assert_allclose(y_amp, payload.amplitude)

    # Amplitude tab: window=1 + no normalize == raw amplitudes.
    w.amp_normalize.setChecked(False)
    w.amp_window.setValue(1)
    amp_curves = w.amp_plot.getPlotItem().listDataItems()
    np.testing.assert_allclose(
        np.array(amp_curves[0].yData), payload.amplitude
    )
    # Normalizing scales the max to 1.
    w.amp_normalize.setChecked(True)
    amp_curves = w.amp_plot.getPlotItem().listDataItems()
    assert np.nanmax(np.array(amp_curves[0].yData)) == pytest.approx(1.0)

    # q-map: per track a trajectory curve + a mean marker.
    qmap_items = w.qmap_plot.getPlotItem().listDataItems()
    assert len(qmap_items) == 2 * payload.n_tracks

    # Worker-shaped results install into the two image views.
    w._on_worker_finished(
        {"kind": "waterfall",
         "data": np.random.default_rng(0).random((6, 32)).astype("f4"),
         "radius": np.linspace(0.0, 4.0, 32), "entry": ENTRY},
        None,
    )
    assert w.waterfall_image.isVisible()
    w._on_worker_finished(
        {"kind": "mean_image",
         "data": np.random.default_rng(1).random((16, 24)).astype("f4"),
         "q_xy": np.linspace(-1, 3, 24), "q_z": np.linspace(0, 4, 16),
         "lo": 0, "hi": 5, "entry": ENTRY},
        None,
    )
    assert w.qmap_image.isVisible()


def test_views_workers_compute_real_data(
    main_window, synthetic_fitted_scan, qtbot,
):
    """ScanProfileWorker end-to-end through the views window (pure
    h5py + mlgidgui.polar — no backend needed)."""
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    main_window._on_show_phase_views()
    w = main_window._phase_views_window

    w._on_compute_waterfall()
    qtbot.waitUntil(lambda: w._worker_thread is None, timeout=30000)
    assert w._waterfall is not None
    assert w._waterfall["data"].shape[0] == 6  # one row per frame

    w.qmap_lo.setValue(1)
    w.qmap_hi.setValue(3)
    w._on_compute_mean_image()
    qtbot.waitUntil(lambda: w._worker_thread is None, timeout=30000)
    assert w._mean_image is not None
    assert w._mean_image["data"].shape == (16, 24)
    assert (w._mean_image["lo"], w._mean_image["hi"]) == (1, 3)


def test_only_tracked_filter_subsets_fitted_overlay(
    main_window, synthetic_fitted_scan,
):
    """"Show only tracked peaks": frame 2 carries the tracked member
    (id 0) AND the untracked blip (id 1) — with the toggle on, only
    the member renders; off restores both. Nothing in memory or on
    disk changes."""
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    panel = main_window.scan_tracking_panel
    v = main_window.viewer
    assert panel.chk_only_tracked.isEnabled()

    v.set_frame(2)
    full = v._fitted._path.elementCount()
    assert full > 0

    panel.chk_only_tracked.setChecked(True)
    assert v._fitted_visible_only == {f: {0} for f in range(6)}
    only = v._fitted._path.elementCount()
    assert 0 < only < full          # blip dropped, member kept
    # In-memory table untouched — render-time subset only.
    assert len(v._frame_peaks[2]["fitted"]) == 2

    panel.chk_only_tracked.setChecked(False)
    assert v._fitted_visible_only is None
    assert v._fitted._path.elementCount() == full


def test_only_tracked_filter_lifts_on_invalidation(
    main_window, synthetic_fitted_scan,
):
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    panel = main_window.scan_tracking_panel
    panel.chk_only_tracked.setChecked(True)
    assert main_window.viewer._fitted_visible_only is not None

    main_window._pipe_command = PipelineCommand(
        "run_fitting", {"entry": ENTRY}
    )
    main_window._on_pipeline_finished(None, None)
    assert main_window.viewer._fitted_visible_only is None
    assert not panel.chk_only_tracked.isChecked()
    assert not panel.chk_only_tracked.isEnabled()


def test_set_fitted_visible_only_unit(main_window, synthetic_fitted_scan):
    """Viewer-level semantics: mapping subsets per frame, a frame
    absent from an ACTIVE mapping shows no fitted rows, None disables."""
    _open(main_window, synthetic_fitted_scan)
    v = main_window.viewer
    v.set_frame(2)                        # ids {0, 1} on this frame
    full = v._fitted._path.elementCount()
    assert full > 0

    v.set_fitted_visible_only({2: {0}})
    assert 0 < v._fitted._path.elementCount() < full
    # Active filter, current frame not in the mapping -> nothing drawn.
    v.set_fitted_visible_only({5: {0}})
    assert v._fitted._path.isEmpty()
    v.set_fitted_visible_only(None)
    assert v._fitted._path.elementCount() == full


def test_hidden_fitted_peak_is_not_selectable(
    main_window, synthetic_nexus_with_peaks,
):
    """A fitted peak hidden by "show only tracked peaks" cannot be
    selected by clicking, Ctrl+clicking, or Ctrl+A — what you can't
    see, you can't select. Lifting the filter restores selectability."""
    from PySide6.QtCore import QPointF, Qt

    _open(main_window, synthetic_nexus_with_peaks)
    v = main_window.viewer
    fit = (v._frame_peaks.get(0) or {})["fitted"]
    # Whitelist only fitted id 1; id 0 becomes invisible.
    v.set_fitted_visible_only({0: {1}})
    r0, a0 = float(fit.radius[0]), float(fit.angle[0])   # hidden id 0
    r1, a1 = float(fit.radius[1]), float(fit.angle[1])   # visible id 1

    # Bare click on the hidden peak -> nothing selected...
    v._on_select_at(QPointF(r0, a0))
    assert v._selected is None
    # ...but the visible one still selects.
    v._on_select_at(QPointF(r1, a1))
    assert v._selected is not None and v._selected.peak_id == 1

    # Ctrl+click on the hidden peak -> no-op.
    v.clear_selection()
    v._on_select_at(QPointF(r0, a0), Qt.KeyboardModifier.ControlModifier)
    assert v.selected_peaks() == []

    # Ctrl+A over fitted grabs only the visible row.
    v._select_all_of_kind_on_frame("fitted")
    assert [s.peak_id for s in v.selected_peaks()] == [1]

    # Lifting the filter makes the hidden peak clickable again.
    v.clear_selection()
    v.set_fitted_visible_only(None)
    v._on_select_at(QPointF(r0, a0))
    assert v._selected is not None and v._selected.peak_id == 0


def _gap_payload() -> TrackingPayload:
    """One track with members on frames 0,1,2,4,5 — frame 3 is a gap."""
    frames = np.array([0, 1, 2, 4, 5])
    radius = 1.0 + 0.002 * frames
    angle = np.full(5, 45.0)
    return TrackingPayload(
        entry=ENTRY, threshold=0.5, length=3,
        q_xy=radius * np.cos(np.deg2rad(angle)),
        q_z=radius * np.sin(np.deg2rad(angle)),
        frame_num=frames, amplitude=100.0 + frames,
        components=[list(range(5))],
    )


def test_interpolate_button_plans_and_enqueues(
    main_window, synthetic_fitted_scan, monkeypatch,
):
    """The Interpolate-track BUTTON plans the gap fills and enqueues an
    ``interpolate_tracks`` command followed by a ``run_matching`` scoped
    to the affected frames, with the panel's matching settings."""
    import h5py

    # Carve a gap: frame 3 loses its fitted rows before the session copy.
    with h5py.File(synthetic_fitted_scan, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00003/fitted_peaks"]
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel

    assert not panel.btn_interp_track.isEnabled()   # no tracks yet
    _install_fake(main_window, _gap_payload())
    assert panel.btn_interp_track.isEnabled()

    fake_match = {
        "cif_prepr": "structures.cif", "peaks_type": "segments",
        "threshold": 0.5, "intensity_threshold": 0.0, "device": "cpu",
    }
    monkeypatch.setattr(
        main_window.pipeline_panel, "matching_kwargs", lambda: dict(fake_match)
    )
    enqueued: list = []
    monkeypatch.setattr(
        main_window, "_enqueue_pipeline",
        lambda path, cmd: enqueued.append(cmd),
    )
    panel.btn_interp_track.click()

    # Nothing was ever matched in this scan -> NO re-match is enqueued
    # (no new structures may appear from a gap fill); only the fill runs.
    assert [c.op_name for c in enqueued] == ["interpolate_tracks"]
    fill = enqueued[0].kwargs
    assert fill["entry"] == ENTRY
    # The plan covers exactly the gap frame, at the interpolated
    # position with the mean bracketing box size.
    assert sorted(fill["plan"]) == [3]
    spec = fill["plan"][3][0]
    assert spec["track"] == 0
    assert spec["radius"] == pytest.approx((1.004 + 1.008) / 2)
    assert spec["angle"] == pytest.approx(45.0)
    assert spec["is_ring"] is False
    if main_window.pipeline_panel._available:
        # Backends installed: the panel's fit config rides along.
        assert set(fill["fit_params"]) >= {"crit_angle", "theta_fixed"}
    else:
        # Backend-less (CI): the fit-config widgets don't exist, so the
        # handler's documented fallback sends {} -> op defaults.
        assert fill["fit_params"] == {}
    # Chain bookkeeping: one tick for the fill's single frame, and the
    # dialog up in manual (real-progress) mode.
    chain = main_window._interp_chain
    assert chain is not None
    assert chain["total"] == 1 and chain["base"] == 0
    assert main_window._track_progress_dialog is not None
    # Manual mode keeps the determinate 0..100 bar (no busy marquee).
    assert main_window._track_progress_dialog.maximum() == 100
    # Real per-frame progress drives the dialog: the fill command's
    # only frame done -> the whole chain -> 100%.
    main_window._pipe_command = enqueued[0]
    main_window._on_pipeline_frame_progress(1, 1, "interpolate_tracks", ENTRY)
    assert main_window._track_progress_dialog.value() == 100
    main_window._close_tracking_progress()
    main_window._interp_chain = None


def test_interp_progress_survives_reentrant_finish(main_window):
    """Field crash regression: ``QProgressDialog.setValue`` on the
    WINDOW-MODAL tracking dialog runs ``processEvents()``, which can
    deliver the running command's queued ``opFinished`` re-entrantly —
    completing the chain and tearing the dialog down MID-setValue. The
    frame-progress handler must not touch the dialog afterwards
    (previously: ``AttributeError: 'NoneType' ... setLabelText``)."""
    cmd = PipelineCommand("interpolate_tracks", {"entry": ENTRY})
    main_window._interp_chain = {
        "commands": [cmd], "ticks": {id(cmd): 1}, "total": 1, "base": 0,
    }
    main_window._pipe_command = cmd
    main_window._show_tracking_progress("Interpolate…", manual=True)
    dlg = main_window._track_progress_dialog
    real_set_value = dlg.setValue
    fired = {"done": False}

    def _reentrant_set_value(value):
        # Emulate processEvents() delivering opFinished inside the
        # setValue call: chain completed, dialog torn down, queue
        # drained and the status bar set back to "idle".
        if not fired["done"]:
            fired["done"] = True
            main_window._interp_chain = None
            main_window._close_tracking_progress()
            main_window._update_status_pipeline(running=False)
        real_set_value(value)

    dlg.setValue = _reentrant_set_value
    # Must not raise.
    main_window._on_pipeline_frame_progress(1, 1, "interpolate_tracks", ENTRY)
    assert main_window._track_progress_dialog is None
    # ...and must not resurrect a stale "running: …" status line for
    # the already-finished command (the handler bails out once it sees
    # the queue drained).
    assert main_window._sb_pipeline.text() == "idle"
    main_window._pipe_command = None


def test_interpolate_button_without_gaps(
    main_window, synthetic_fitted_scan, monkeypatch,
):
    """No gaps (every track frame has a fitted member) -> status
    message, nothing enqueued, no dialog."""
    enqueued: list = []
    monkeypatch.setattr(
        main_window, "_enqueue_pipeline",
        lambda path, cmd: enqueued.append(cmd),
    )
    # Gapless payload (the default fake covers every frame 0-5).
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    main_window._on_interpolate_tracks_requested()
    assert enqueued == []
    assert "nothing to fill" in main_window.statusBar().currentMessage()
    assert main_window._track_progress_dialog is None


def test_interpolate_button_without_cif(
    main_window, synthetic_fitted_scan, monkeypatch,
):
    """Gaps to fill but no CIF source configured -> a warning naming the
    Pipeline panel, nothing enqueued."""
    import h5py

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00003/fitted_peaks"]
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window, _gap_payload())
    enqueued: list = []
    monkeypatch.setattr(
        main_window, "_enqueue_pipeline",
        lambda path, cmd: enqueued.append(cmd),
    )
    monkeypatch.setattr(
        main_window.pipeline_panel, "matching_kwargs", lambda: None
    )
    warnings: list = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    main_window._on_interpolate_tracks_requested()
    assert enqueued == []
    assert warnings and "CIF" in warnings[0][2]


def test_interp_matching_restricted_to_tracked_cifs(
    main_window, synthetic_fitted_scan, monkeypatch, tmp_path,
):
    """The chained re-match is CIF-restricted to the structures the
    filled track was tracked in: one tracked structure -> only its .cif
    goes into the matching command; the untracked rest of the folder is
    skipped."""
    import h5py

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00003/fitted_peaks"]
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window, _gap_payload())
    # The track was matched as PbI2 (and only PbI2).
    main_window._scan_track_phases = {0: ["PbI2"]}
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "PbI2.cif").write_text("data_PbI2\n")
    (cif_dir / "Other.cif").write_text("data_Other\n")
    monkeypatch.setattr(
        main_window.pipeline_panel, "matching_kwargs",
        lambda: {"cif_prepr": str(cif_dir), "peaks_type": "segments",
                 "threshold": 0.5, "intensity_threshold": 0.0,
                 "device": "cpu"},
    )
    enqueued: list = []
    monkeypatch.setattr(
        main_window, "_enqueue_pipeline",
        lambda path, cmd: enqueued.append(cmd),
    )
    main_window._on_interpolate_tracks_requested()
    match = enqueued[1].kwargs
    assert match["peaks_type"] == "segments"
    assert match["cif_prepr"] == str(cif_dir / "PbI2.cif")   # Other.cif skipped
    main_window._close_tracking_progress()
    main_window._interp_chain = None


def test_interp_match_commands_split_segments_and_rings(
    main_window, synthetic_fitted_scan, monkeypatch, tmp_path,
):
    """A plan with segment AND ring fills yields one matching command
    per peaks_type, each pinned to its own frames and restricted to its
    own tracked CIF; an unsubsettable source SKIPS matching instead of
    brute-forcing the full set."""
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    main_window._scan_track_phases = {0: ["A"], 1: ["B"]}
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for name in ("A.cif", "B.cif", "C.cif"):
        (cif_dir / name).write_text("x")
    # Patch the panel's raw-source ACCESSOR, not its cif_path widget:
    # the widget only exists when the pipeline backends are installed,
    # and this test must also run on the backend-less CI.
    monkeypatch.setattr(
        main_window.pipeline_panel, "cif_source_text",
        lambda: str(cif_dir),
    )
    plan = {
        3: [
            {"track": 0, "radius": 1.0, "angle": 45.0,
             "radius_width": 0.1, "angle_width": 5.0, "is_ring": False},
            {"track": 1, "radius": 2.0, "angle": 45.0,
             "radius_width": 0.1, "angle_width": 90.0, "is_ring": True},
        ],
        4: [
            {"track": 1, "radius": 2.0, "angle": 45.0,
             "radius_width": 0.1, "angle_width": 90.0, "is_ring": True},
        ],
    }
    cmds = main_window._build_interp_match_commands(
        plan, ENTRY, {"cif_prepr": str(cif_dir), "threshold": 0.5},
    )
    by_type = {c.kwargs["peaks_type"]: c.kwargs for c in cmds}
    assert set(by_type) == {"segments", "rings"}
    assert by_type["segments"]["frame_num"] == [3]
    assert by_type["rings"]["frame_num"] == [3, 4]
    # Each pass restricted to ITS track's structure; C.cif never used.
    assert by_type["segments"]["cif_prepr"] == str(cif_dir / "A.cif")
    assert by_type["rings"]["cif_prepr"] == str(cif_dir / "B.cif")

    # An unsubsettable source (single foreign .cif file) -> matching
    # SKIPPED entirely rather than run against everything.
    monkeypatch.setattr(
        main_window.pipeline_panel, "cif_source_text",
        lambda: "structures.cif",
    )
    cmds = main_window._build_interp_match_commands(
        plan, ENTRY, {"cif_prepr": "structures.cif", "threshold": 0.5},
    )
    assert cmds == []


def test_interp_unmatched_track_falls_back_to_scan_cifs(
    main_window, synthetic_fitted_scan, monkeypatch, tmp_path,
):
    """A filled track that was never matched re-matches against the
    structures identified elsewhere in the SCAN — never the full panel
    folder."""
    # PbI2 identified on frame 2 of the scan (existing helper).
    _write_matched(synthetic_fitted_scan, 2, [("PbI2", [0])])
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window, _gap_payload())
    main_window._scan_track_phases = {}          # the track itself: unmatched
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "PbI2.cif").write_text("x")
    (cif_dir / "Other.cif").write_text("x")
    # cif_source_text instead of the cif_path widget — see the
    # segments-and-rings test above (backend-less CI has no widget).
    monkeypatch.setattr(
        main_window.pipeline_panel, "cif_source_text",
        lambda: str(cif_dir),
    )
    plan = {3: [{"track": 0, "radius": 1.0, "angle": 45.0,
                 "radius_width": 0.1, "angle_width": 5.0,
                 "is_ring": False}]}
    cmds = main_window._build_interp_match_commands(
        plan, ENTRY, {"cif_prepr": str(cif_dir), "threshold": 0.5},
    )
    assert len(cmds) == 1
    # Scan-wide identified structure only — Other.cif is never tried.
    assert cmds[0].kwargs["cif_prepr"] == str(cif_dir / "PbI2.cif")


def test_restrict_cif_source_forms(main_window, tmp_path):
    """`_restrict_cif_source` subsets folders and .cif lists, and
    refuses pickles / missing files (caller keeps the full source)."""
    d = tmp_path / "cifs"
    d.mkdir()
    (d / "PbI2.cif").write_text("x")
    (d / "MAPbI3.cif").write_text("x")
    w = main_window._restrict_cif_source
    # Folder: subset by stem, case-insensitive.
    assert w(str(d), {"pbi2"}) == str(d / "PbI2.cif")
    assert w(str(d), {"PbI2", "MAPbI3"}) == ";".join(
        [str(d / "MAPbI3.cif"), str(d / "PbI2.cif")]
    )
    # Explicit .cif list: subset kept in the same form.
    listed = ";".join([str(d / "PbI2.cif"), str(d / "MAPbI3.cif")])
    assert w(listed, {"MAPbI3"}) == str(d / "MAPbI3.cif")
    # A tracked CIF with no file -> None (full source fallback).
    assert w(str(d), {"PbI2", "Missing"}) is None
    # Pickle / empty / non-string sources -> None.
    assert w("pattern.pickle", {"PbI2"}) is None
    assert w("", {"PbI2"}) is None
    assert w(object(), {"PbI2"}) is None


@pytest.mark.parametrize("with_peak_ids", [False, True])
def test_interp_fill_result_appends_members(
    main_window, synthetic_fitted_scan, with_peak_ids,
):
    """Applying a gap-fill result makes the new fitted peak a member of
    its origin track: payload arrays grow, the component gains the new
    index, member ids resolve to the new row, and the only-tracked
    whitelist now covers the gap frame. A payload from a real run
    carries ``peak_id``, which has to grow with the other arrays."""
    # Carve the gap by emptying frame 3's fitted rows (dataset stays, as
    # after a real fitting run) so the fill can be written back.
    file_model.delete_peak_row(synthetic_fitted_scan, ENTRY, 3, "fitted", 0)
    _open(main_window, synthetic_fitted_scan)
    payload = _install_fake(main_window, _gap_payload())
    if with_peak_ids:
        payload.peak_id = np.zeros(payload.n_members, dtype=int)
    panel = main_window.scan_tracking_panel
    panel.chk_only_tracked.setChecked(True)
    assert 3 not in main_window.viewer._fitted_visible_only

    # Simulate what the interpolate_tracks op persisted for the gap
    # (write under the detach dance, as the pipeline machinery would).
    with main_window._detached_silx_tree():
        new_id = file_model.add_fitted_peak_row(
            main_window.session.temp_path, ENTRY, 3,
            radius=1.006, radius_width=0.05, angle=45.0, angle_width=5.0,
            amplitude=103.0, is_ring=False,
        )
    main_window._interp_fill_result = [{
        "track": 0, "frame": 3,
        "detected_id": 99, "fitted_id": int(new_id),
    }]
    main_window._apply_interp_fill_result()

    assert payload.n_members == 6
    assert 5 in payload.components[0]                    # new member index
    if with_peak_ids:
        assert payload.peak_id.size == 6
        assert int(payload.peak_id[5]) == int(new_id)
    assert main_window._scan_member_ids[5] == (3, int(new_id))
    # The former gap frame is now whitelisted with the new fitted id.
    assert main_window.viewer._fitted_visible_only[3] == {int(new_id)}
    # Panel reflects the grown track (span 0-5, 6 frames, 6 members).
    assert payload.track_span(0) == (0, 5, 6, 6)
    # Pending state consumed.
    assert main_window._interp_fill_result is None


def test_rings_tracked_gui_side_and_untracked_hidden(
    main_window, synthetic_fitted_scan,
):
    """Upstream cannot track rings, so the GUI tracks them by radial
    IoU (`_on_phase_track_result` -> `track_rings`): a ring persisting
    across every frame becomes a track and stays visible under the
    only-tracked filter, while a one-off ring is hidden.

    Drives the real result-handler (no backend needed — the payload is
    supplied, and member-id reconstruction + ring tracking are pure)."""
    import h5py

    from mlgidgui.pipeline import PipelineCommand

    # Add a persistent ring (id 9, r=1.8, amp 30) to every frame, plus a
    # one-off ring (id 10, r=2.6, amp 40) on frame 2 only. Distinct
    # amplitudes keep the two same-frame rings disambiguable during
    # member-id reconstruction.
    with h5py.File(synthetic_fitted_scan, "r+") as f:
        for frame in range(6):
            g = f[f"{ENTRY}/data/analysis/frame{frame:05d}"]
            arr = g["fitted_peaks"][()]
            dt = arr.dtype
            rows = [arr]
            ring = np.zeros(1, dtype=dt)
            ring["id"] = [9]; ring["radius"] = [1.8]
            ring["radius_width"] = [0.2]; ring["angle"] = [45.0]
            ring["angle_width"] = [np.inf]; ring["is_ring"] = [True]
            ring["amplitude"] = [30.0]
            rows.append(ring)
            if frame == 2:
                oneoff = np.zeros(1, dtype=dt)
                oneoff["id"] = [10]; oneoff["radius"] = [2.6]
                oneoff["radius_width"] = [0.2]; oneoff["angle"] = [45.0]
                oneoff["angle_width"] = [np.inf]; oneoff["is_ring"] = [True]
                oneoff["amplitude"] = [40.0]
                rows.append(oneoff)
            del g["fitted_peaks"]
            g.create_dataset("fitted_peaks", data=np.concatenate(rows))
    _open(main_window, synthetic_fitted_scan)

    # Payload mirroring what upstream emits: all fitted-peak instances
    # in frame order, coordinates/amplitude taken straight from the
    # fitted rows (as upstream does) so member-id reconstruction
    # resolves them; only the spot track is in components (upstream
    # tracks no rings).
    member_frames, pids, mq_xy, mq_z, mamp = [], [], [], [], []
    for frame in range(6):
        t = file_model.load_peaks(
            main_window.session.temp_path, ENTRY, frame
        )["fitted"]
        for j in range(len(t)):
            member_frames.append(frame)
            pids.append(int(t.ids[j]))
            mq_xy.append(float(t.q_xy[j]))
            mq_z.append(float(t.q_z[j]))
            mamp.append(float(t.amplitude[j]))
    spot_members = [i for i, p in enumerate(pids) if p == 0]
    payload = TrackingPayload(
        entry=ENTRY, threshold=0.5, length=3,
        q_xy=np.array(mq_xy), q_z=np.array(mq_z),
        frame_num=np.array(member_frames), amplitude=np.array(mamp),
        components=[spot_members],
    )
    main_window._on_phase_track_result(
        payload, PipelineCommand("track_peaks", {"entry": ENTRY}),
    )
    panel = main_window.scan_tracking_panel
    v = main_window.viewer

    # Two tracks now: the spot track + the GUI-tracked persistent ring.
    assert payload.n_tracks == 2
    assert panel._model.rowCount() == 2

    panel.chk_only_tracked.setChecked(True)
    # Frame 2 whitelist: tracked spot (0) + tracked persistent ring (9);
    # the untracked blip (1) and the one-off ring (10) are hidden.
    assert v._fitted_visible_only[2] == {0, 9}


def test_native_ring_components_not_double_counted(
    main_window, synthetic_fitted_scan,
):
    """mlgidbase 0.1.5 clamps ring angle_width inf->45 inside
    track_peaks, so rings stored with a FINITE angle (the GUI's
    injected rings persist angle=45) can arrive as NATIVE components.
    The handler must drop those before appending its own radial-IoU
    ring tracks — the same physical ring must yield exactly one
    track."""
    import h5py

    from mlgidgui.pipeline import PipelineCommand

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        for frame in range(6):
            g = f[f"{ENTRY}/data/analysis/frame{frame:05d}"]
            arr = g["fitted_peaks"][()]
            dt = arr.dtype
            ring = np.zeros(1, dtype=dt)
            ring["id"] = [9]; ring["radius"] = [1.8]
            ring["radius_width"] = [0.2]; ring["angle"] = [45.0]
            ring["angle_width"] = [np.inf]; ring["is_ring"] = [True]
            ring["amplitude"] = [30.0]
            del g["fitted_peaks"]
            g.create_dataset(
                "fitted_peaks", data=np.concatenate([arr, ring])
            )
    _open(main_window, synthetic_fitted_scan)

    member_frames, pids, mq_xy, mq_z, mamp = [], [], [], [], []
    for frame in range(6):
        t = file_model.load_peaks(
            main_window.session.temp_path, ENTRY, frame
        )["fitted"]
        for j in range(len(t)):
            member_frames.append(frame)
            pids.append(int(t.ids[j]))
            mq_xy.append(float(t.q_xy[j]))
            mq_z.append(float(t.q_z[j]))
            mamp.append(float(t.amplitude[j]))
    spot_members = [i for i, p in enumerate(pids) if p == 0]
    ring_members = [i for i, p in enumerate(pids) if p == 9]
    # The ring arrives ALSO as a native component (what 0.1.5 emits for
    # finite-angle rings) alongside the spot track.
    payload = TrackingPayload(
        entry=ENTRY, threshold=0.5, length=3,
        q_xy=np.array(mq_xy), q_z=np.array(mq_z),
        frame_num=np.array(member_frames), amplitude=np.array(mamp),
        components=[spot_members, ring_members],
    )
    main_window._on_phase_track_result(
        payload, PipelineCommand("track_peaks", {"entry": ENTRY}),
    )

    # Exactly one spot track + one ring track — the native ring
    # component was dropped, the GUI-side ring track appended.
    assert payload.n_tracks == 2
    assert main_window.scan_tracking_panel._model.rowCount() == 2
    ring_tracks = main_window._scan_ring_tracks
    assert len(ring_tracks) == 1
    (ring_idx,) = ring_tracks
    assert sorted(payload.components[ring_idx]) == sorted(ring_members)


def test_pipeline_and_view_workers_mutually_gated(
    main_window, synthetic_fitted_scan,
):
    """pygid opens the file r+ even for reads, so a phase-views worker
    (long-lived "r" handle) and a pipeline run must never overlap:
    each side refuses to start while the other holds the file."""
    from PySide6.QtCore import QThread

    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    main_window._on_show_phase_views()
    w = main_window._phase_views_window

    # View worker active -> pipeline entry points refuse (nothing
    # enqueued, no worker spawned).
    marker = QThread()  # never started — a liveness marker only
    w._worker_thread = marker
    main_window._on_track_scan_requested(0.5, 3)
    assert main_window._pipeline_queue == []
    assert main_window._pipe_thread is None
    main_window._on_run_requested(
        PipelineCommand("run_fitting", {"entry": ENTRY})
    )
    assert main_window._pipeline_queue == []
    assert main_window._pipe_thread is None
    w._worker_thread = None
    marker.deleteLater()

    # Pipeline busy -> the views window refuses to start its worker.
    pipe_marker = QThread()
    main_window._pipe_thread = pipe_marker
    w._on_compute_waterfall()
    assert w._worker_thread is None
    main_window._pipe_thread = None
    pipe_marker.deleteLater()


def test_tracking_progress_dialog_lifecycle(main_window, synthetic_fitted_scan):
    """The loading dialog shows on demand (immediately, as a busy
    marquee) and tears down cleanly — backend-free unit of the
    helpers. Manual mode keeps the determinate bar for real ticks."""
    _open(main_window, synthetic_fitted_scan)
    main_window._show_tracking_progress()
    dlg = main_window._track_progress_dialog
    assert dlg is not None
    # Busy marquee (range 0..0): honest motion for one opaque call —
    # regression against the fake timer that stalled at 95%.
    assert dlg.minimum() == 0 and dlg.maximum() == 0
    # A finished track_peaks command closes it (both success and error).
    main_window._pipe_command = PipelineCommand("track_peaks", {"entry": ENTRY})
    main_window._on_pipeline_finished(None, None)
    assert main_window._track_progress_dialog is None
    # Re-showing replaces cleanly (no leak of the previous dialog),
    # and manual mode is determinate.
    main_window._show_tracking_progress("Interpolate…", manual=True)
    assert main_window._track_progress_dialog.maximum() == 100
    main_window._close_tracking_progress()
    assert main_window._track_progress_dialog is None


def test_the_frame_gap_control_travels_with_the_run(
    main_window, synthetic_fitted_scan, monkeypatch,
):
    """'Max frame gap' (mlgidbase window_size): off = None, a number is
    sent as is, and the panel names it in the result line."""
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel
    sent: list = []
    monkeypatch.setattr(
        main_window, "_enqueue_pipeline",
        lambda path, cmd: sent.append(cmd),
    )
    monkeypatch.setattr(main_window, "_show_tracking_progress", lambda *a, **k: None)
    assert panel.window_size == 0 and panel.spin_window.specialValueText() == "off"
    main_window._on_track_scan_requested(0.5, 3)
    panel.spin_window.setValue(25)
    main_window._on_track_scan_requested(0.5, 3)
    assert [c.kwargs["window_size"] for c in sent] == [None, 25]

    payload = _fake_payload()
    payload.window_size = 25
    panel.set_payload(payload, [(int(f), 0) for f in payload.frame_num])
    assert "gap ≤ 25 frames" in panel.lbl_status.text()


def test_save_official_figure_writes_upstreams_own_plot(
    main_window, synthetic_fitted_scan, tmp_path,
):
    """'Save official figures' reruns track_peaks with the payload's
    settings and saves plot_tracked_peaks to exactly the chosen path."""
    pytest.importorskip("mlgidbase")
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    out = tmp_path / "tracks.png"
    main_window._on_save_official_figures(str(out), "radius")
    assert out.is_file() and out.stat().st_size > 0
    assert "Saved the mlgidBASE tracking figure" in (
        main_window.statusBar().currentMessage()
    )


def test_save_official_figure_amplitude_axis_falls_back_to_radius(
    main_window, synthetic_fitted_scan, tmp_path,
):
    pytest.importorskip("mlgidbase")
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window)
    out = tmp_path / "amp.png"
    main_window._on_save_official_figures(str(out), "amplitude")
    assert out.is_file()
    assert "plotted radius" in main_window.statusBar().currentMessage()


def test_delete_key_removes_track(
    main_window, synthetic_fitted_scan, qtbot,
):
    """Delete on a selected table row removes the track from the
    RESULTS only: table row gone, payload shrunk, track-indexed ring
    and phase maps remapped to the shifted indices, fitted rows in
    the file untouched."""
    _open(main_window, synthetic_fitted_scan)
    payload = _fake_payload()
    # Two 3-member tracks so something is left after the delete.
    payload.components = [[0, 1, 2], [3, 4, 5]]
    _install_fake(main_window, payload)
    main_window._scan_ring_tracks = {1}
    main_window._scan_track_phases = {1: ["PbI2"]}
    panel = main_window.scan_tracking_panel
    assert panel._model.rowCount() == 2
    panel._table.setFocus()
    panel._table.selectRow(0)
    qtbot.keyClick(panel._table, Qt.Key.Key_Delete)
    assert panel._model.rowCount() == 1
    assert main_window._scan_payload.n_tracks == 1
    assert main_window._scan_payload.components == [[3, 4, 5]]
    # Index-keyed derived state shifted down with the removal.
    assert main_window._scan_ring_tracks == {0}
    assert main_window._scan_track_phases == {0: ["PbI2"]}
    # Display-only: the file still carries its fitted rows.
    tab = file_model.load_peaks(
        main_window.session.temp_path, ENTRY, 0
    )["fitted"]
    assert len(tab) > 0
    assert "removed" in main_window.statusBar().currentMessage()


def test_structure_column_shows_matched_phases(
    main_window, synthetic_fitted_scan,
):
    """The structure column lists each track's matched CIFs (dominant
    first, comma-joined); tracks without a matched member show a
    dash placeholder."""
    _open(main_window, synthetic_fitted_scan)
    payload = _fake_payload()
    payload.components = [[0, 1, 2], [3, 4, 5]]
    ids = [(int(f), 0) for f in payload.frame_num]
    panel = main_window.scan_tracking_panel
    panel.set_payload(payload, ids, {0: ["PbI2", "Other"]})
    col = panel._model.columnCount() - 1
    assert panel._model.item(0, col).text() == "PbI2, Other"
    assert panel._model.item(1, col).text() == "—"


def test_track_button_gated_without_backend(
    main_window, synthetic_fitted_scan, monkeypatch,
):
    import mlgidgui.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "is_mlgidbase_available", lambda: False)
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel
    assert not panel.btn_track.isEnabled()
    assert "pipeline" in panel.btn_track.toolTip()


# --- matched-peak filtering + q-map phase identity ---

def _write_matched(path, frame, structures):
    """Write a matched_segments dataset. structures: [(cif, [positions])],
    where positions are POSITIONAL indices into the frame's fitted_peaks."""
    import h5py

    mdt = np.dtype([
        ("CIF", h5py.string_dtype()), ("h", "i4"), ("k", "i4"),
        ("l", "i4"), ("probability", "f8"),
        ("peak_list", h5py.vlen_dtype(np.int32)),
    ])
    arr = np.zeros(len(structures), dtype=mdt)
    for i, (cif, pos) in enumerate(structures):
        arr["CIF"][i] = cif
        arr["h"][i], arr["k"][i], arr["l"][i] = 1, 1, 0
        arr["probability"][i] = 0.9
        arr["peak_list"][i] = np.array(pos, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f[f"{ENTRY}/data/analysis/frame{frame:05d}"]
        if "matched_segments_0000" in g:
            del g["matched_segments_0000"]
        g.create_dataset("matched_segments_0000", data=arr)


def _payload_from_file(window):
    """Payload mirroring upstream: all fitted instances in frame order,
    coordinates from the fitted rows; only the spot (id 0) is tracked."""
    mf, pid, qx, qz, am = [], [], [], [], []
    for fr in range(6):
        t = file_model.load_peaks(window.session.temp_path, ENTRY, fr)["fitted"]
        if t is None:                 # gap frame (fitted rows deleted)
            continue
        for j in range(len(t)):
            mf.append(fr); pid.append(int(t.ids[j]))
            qx.append(float(t.q_xy[j])); qz.append(float(t.q_z[j]))
            am.append(float(t.amplitude[j]))
    spot = [i for i, p in enumerate(pid) if p == 0]
    return TrackingPayload(
        entry=ENTRY, threshold=0.5, length=3,
        q_xy=np.array(qx), q_z=np.array(qz),
        frame_num=np.array(mf), amplitude=np.array(am), components=[spot],
    )


def _run_result(window, path):
    payload = _payload_from_file(window)
    window._on_phase_track_result(
        payload, PipelineCommand("track_peaks", {"entry": ENTRY}),
    )
    return payload


def test_matched_overlay_filtered_by_tracked_peaks(
    main_window, synthetic_fitted_scan,
):
    """Under 'show only tracked peaks', a matched structure draws only
    the boxes around its TRACKED fitted peaks; a structure whose peaks
    are all untracked disappears."""
    # frame 2 fitted = [spot id0 @pos0 (tracked), blip id1 @pos1 (untracked)].
    # A covers both; B covers only the untracked blip.
    _write_matched(synthetic_fitted_scan, 2, [("A", [0, 1]), ("B", [1])])
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)
    v = main_window.viewer
    panel = main_window.scan_tracking_panel
    a_uid, b_uid = "matched_segments_0000/0", "matched_segments_0000/1"

    v.set_frame(2)
    full = {uid: it._path.elementCount() for uid, it in v._matched_items}
    assert full[a_uid] > 0 and full[b_uid] > 0   # both drawn, filter off

    panel.chk_only_tracked.setChecked(True)
    v.set_frame(2)
    items = {uid: it for uid, it in v._matched_items}
    # A keeps only its tracked peak -> fewer path elements than before.
    assert 0 < items[a_uid]._path.elementCount() < full[a_uid]
    # B is all-untracked -> hidden.
    assert not items[b_uid].isVisible()


def test_hidden_matched_peak_is_not_selectable(
    main_window, synthetic_fitted_scan,
):
    """With the filter on and the fitted overlay hidden (so clicks reach
    matched), an untracked matched peak is a no-op while the tracked one
    selects with the structure highlight scoped to visible peaks."""
    from PySide6.QtCore import QPointF

    _write_matched(synthetic_fitted_scan, 2, [("A", [0, 1])])
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)
    v = main_window.viewer
    main_window.scan_tracking_panel.chk_only_tracked.setChecked(True)
    v.set_frame(2)
    v._visibility["fitted"] = False   # let clicks fall through to matched

    t = v._frame_peaks[2]["fitted"]
    i0 = next(j for j in range(len(t)) if int(t.ids[j]) == 0)  # tracked
    i1 = next(j for j in range(len(t)) if int(t.ids[j]) == 1)  # untracked blip

    # Untracked matched peak -> nothing selected.
    v._on_select_at(QPointF(float(t.radius[i1]), float(t.angle[i1])))
    assert v._selected is None
    # Tracked matched peak -> matched selection, highlight only the
    # visible (tracked) peak.
    v._on_select_at(QPointF(float(t.radius[i0]), float(t.angle[i0])))
    assert v._selected is not None and v._selected.kind == "matched"
    assert v._selected.peak_id == 0
    assert v._selected.multi_peak_ids == [0]


def test_qmap_colors_by_matched_phase(main_window, synthetic_fitted_scan):
    """The q-map 'Color by matched phase' toggle recolors tracks by CIF
    (with a legend) once Matching has produced a phase map."""
    from mlgidgui.phase_views_window import _track_pen

    _write_matched(synthetic_fitted_scan, 2, [("PbI2", [0])])  # tracked id 0
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)
    assert main_window._scan_track_phases == {0: ["PbI2"]}

    main_window._on_show_phase_views()
    w = main_window._phase_views_window
    assert w._track_phases == {0: ["PbI2"]}
    assert w.qmap_phase.isEnabled()

    # Off by default -> per-track color; on -> the CIF's phase color.
    assert w._qmap_track_color(0, 1, phase_mode=False) == _track_pen(0, 1)
    assert w._qmap_track_color(0, 1, phase_mode=True) == w._phase_colors["PbI2"]

    w.qmap_phase.setChecked(True)
    assert w._qmap_legend.isVisible()
    # Per-frame attribution: matching claimed only the frame-2 member,
    # so the legend carries PbI2 AND "unmatched" (the other frames'
    # members render grey — previously the whole track was painted
    # with the dominant phase and the legend had one entry).
    assert len(w._qmap_legend.items) == 2


def test_qmap_phase_toggle_disabled_without_matching(
    main_window, synthetic_fitted_scan,
):
    """No matched data -> the phase toggle is disabled and the legend
    stays hidden."""
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)   # no matched written
    assert main_window._scan_track_phases == {}
    main_window._on_show_phase_views()
    w = main_window._phase_views_window
    assert not w.qmap_phase.isEnabled()
    assert not w._qmap_legend.isVisible()


def test_phase_color_overrides_recolor_views(phase_window):
    """A pushed per-CIF colour replaces the automatic hue everywhere the
    phase colour shows (q-map track colour, structure toggles); other
    CIFs keep their automatic colour, and an empty push restores it."""
    w = phase_window
    auto_a = w._phase_colors["A"].name()
    auto_b = w._phase_colors["B"].name()

    w.set_phase_color_overrides({"A": "#123456"})
    assert w._phase_colors["A"].name() == "#123456"
    assert w._phase_colors["B"].name() == auto_b
    assert w._qmap_track_color(0, 3, phase_mode=True).name() == "#123456"
    assert "#123456" in w._struct_checks["A"].styleSheet()

    w.set_phase_color_overrides({})
    assert w._phase_colors["A"].name() == auto_a


def test_matched_color_pick_reaches_phase_views(
    main_window, synthetic_fitted_scan, clean_matched_colors,
):
    """A colour picked in the matched-peaks legend lands in the phase
    views: at window open (persisted override) and live while the
    window is showing."""
    _write_matched(synthetic_fitted_scan, 2, [("PbI2", [0])])
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)

    # Picked before the window exists -> honoured on open.
    main_window._on_matched_pen_picked(("PbI2", 0, 0, 1), {"color": "#123456"})
    main_window._on_show_phase_views()
    w = main_window._phase_views_window
    assert w._phase_colors["PbI2"].name() == "#123456"

    # Picked while the window is open -> re-rendered live.
    main_window._on_matched_pen_picked(("PbI2", 0, 0, 1), {"color": "#654321"})
    assert w._phase_colors["PbI2"].name() == "#654321"
    assert w._qmap_track_color(0, 1, phase_mode=True).name() == "#654321"


def test_display_palette_colors_reach_phase_views(
    main_window, synthetic_fitted_scan, clean_matched_colors,
):
    """WITHOUT any hand-picked colour, the phase views use the Display
    legend's ACTUAL automatic palette colour for a matched structure
    (not the views' own hue wheel) — in the colour map, the q-map
    track pen and the structure toggle — at window open and re-pushed
    on a fresh tracking install while the window is showing."""
    import pyqtgraph as pg

    from mlgidgui.image_viewer import matched_pen_for

    _write_matched(synthetic_fitted_scan, 2, [("PbI2", [0])])
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)

    viewer = main_window.viewer
    assert viewer.cif_color_overrides() == {}     # nothing picked
    display = viewer.cif_effective_colors()["PbI2"]
    assert display == matched_pen_for(
        viewer._color_index_for_key(("PbI2", 1, 1, 0))
    )["color"]
    expected = pg.mkColor(display).name()

    main_window._on_show_phase_views()
    w = main_window._phase_views_window
    assert w._phase_colors["PbI2"].name() == expected
    assert w._qmap_track_color(0, 1, phase_mode=True).name() == expected
    assert expected in w._struct_checks["PbI2"].styleSheet()

    # A fresh tracking install while the window is open re-pushes the
    # (re-seeded) palette before the phase mapping.
    w.set_phase_color_overrides({})               # wipe to prove the push
    _run_result(main_window, synthetic_fitted_scan)
    assert w._phase_colors["PbI2"].name() == expected


def test_frame_interval_filters_trajectories_and_amplitude(phase_window):
    """The window-wide "Frames: from..to" interval narrows the
    member-based plots to those frames; the q-map stays
    frame-complete. Bounds follow the payload and default to the full
    range."""
    w = phase_window
    assert (w.frame_lo.value(), w.frame_hi.value()) == (0, 1)
    assert w.frame_lo.isEnabled() and w.frame_hi.isEnabled()
    traj = w.traj_plot.getPlotItem().listDataItems()
    assert [list(c.xData) for c in traj] == [[0.0, 1.0]] * 3

    w.frame_lo.setValue(1)
    traj = w.traj_plot.getPlotItem().listDataItems()
    assert [list(c.xData) for c in traj] == [[1.0]] * 3
    amp = w.amp_plot.getPlotItem().listDataItems()
    assert amp and all(list(c.xData) == [1.0] for c in amp)
    # q-map member curves keep both frames (2 points).
    qmap_curves = [
        it for it in w.qmap_plot.getPlotItem().listDataItems()
        if it.xData is not None and len(it.xData) == 2
    ]
    assert len(qmap_curves) == 3
    # A crossed pair still reads as a valid interval.
    assert w._frame_interval() == (1, 1)


def test_frame_interval_resets_on_new_payload(phase_window):
    w = phase_window
    w.frame_lo.setValue(1)
    fn = np.array([0, 1, 2, 3])
    q = np.zeros(4)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([1.0, 2.0, 3.0, 4.0]),
        components=[[0, 1, 2, 3]],
    )
    w.set_payload(payload)
    assert (w.frame_lo.value(), w.frame_hi.value()) == (0, 3)
    assert (w.frame_lo.minimum(), w.frame_hi.maximum()) == (0, 3)


def _late_track_window(qtbot, tmp_path):
    """6-frame scan (context set), track 0 on frames 2..3 and track 1 on
    frames 0 and 3 (a gap at 1..2) — the zero-padding test bed."""
    from mlgidgui.phase_views_window import PhaseViewsWindow

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    w.set_context(str(tmp_path / "x.h5"), "e", 6)
    fn = np.array([2, 3, 0, 3])
    q = np.zeros(4)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([7.0, 9.0, 4.0, 2.0]),
        components=[[0, 1], [2, 3]],
    )
    w.set_payload(payload)
    w.amp_window.setValue(1)
    w.amp_normalize.setChecked(False)
    return w


def test_amp_zero_pads_median_result_only(qtbot, tmp_path):
    """"Zeros at start/end" is a MEDIAN-curve feature: individual track
    curves never get zeros (and the box is disabled outside median
    mode), the per-frame statistics run over present tracks only (an
    absent track cannot drag the median down), and the finished median
    series is padded with lead-in + tail zeros over the visible range.
    With scan context set the interval bounds cover the whole scan."""
    w = _late_track_window(qtbot, tmp_path)
    assert (w.frame_lo.value(), w.frame_hi.value()) == (0, 5)
    # Outside median mode the toggle is disabled and checking it must
    # not touch the per-track curves.
    assert not w.amp_zero.isEnabled()
    w.amp_zero.setChecked(True)
    curves = w.amp_plot.getPlotItem().listDataItems()
    assert sorted(tuple(c.xData) for c in curves) == [
        (0.0, 3.0), (2.0, 3.0),
    ]

    w.set_track_phases({0: ["A"], 1: ["A"]})
    w.amp_median.setChecked(True)
    assert w.amp_zero.isEnabled()
    xs, med, q25, q75 = w._amp_median_series(
        [w._member_series(w._interval_members(0)),
         w._member_series(w._interval_members(1))], 1, False
    )
    # Present-only statistics: frame 0 is track 1's value alone (4.0),
    # NOT median(0, 4); frame 1 stays a hole (gap in track 1, track 0
    # absent); tail frames 4..5 pad to zero, IQR bounds included.
    assert list(xs) == [0.0, 2.0, 3.0, 4.0, 5.0]
    assert list(med) == [4.0, 7.0, 5.5, 0.0, 0.0]
    assert q25[-1] == 0.0 and q75[-1] == 0.0

    # Narrowing the interval clips the padding; unticking removes it.
    w.frame_hi.setValue(3)
    xs, _med, _q25, _q75 = w._amp_median_series(
        [w._member_series(w._interval_members(0)),
         w._member_series(w._interval_members(1))], 1, False
    )
    assert list(xs) == [0.0, 2.0, 3.0]
    w.frame_hi.setValue(5)
    w.amp_zero.setChecked(False)
    xs, _med, _q25, _q75 = w._amp_median_series(
        [w._member_series(w._interval_members(0)),
         w._member_series(w._interval_members(1))], 1, False
    )
    assert list(xs) == [0.0, 2.0, 3.0]


def test_amp_track_exclusion_filters_structure(phase_window):
    """Tracks unticked in the Select-tracks dialog drop out of their
    structure's grouped band and median; the selection resets when the
    payload SHAPE changes but survives a plain re-push."""
    w = phase_window
    w.set_track_phases({0: ["A"], 1: ["A"], 2: ["A"]})
    w._amp_excluded = {"A": {2}}
    assert [list(m) for m in w._amp_groups()["A"]] == [[0, 1], [2, 3]]

    w.amp_group.setChecked(True)
    curves, _lines = _split_curves_extreme_lines(
        w.amp_plot.getPlotItem().listDataItems()
    )
    assert len(curves) == 2          # excluded track dropped
    w._amp_excluded = {}
    w._refresh_amplitude()
    curves, _lines = _split_curves_extreme_lines(
        w.amp_plot.getPlotItem().listDataItems()
    )
    assert len(curves) == 3

    # Same payload object re-pushed -> selection kept; a new payload
    # object -> reset.
    w._amp_excluded = {"A": {2}}
    w.set_payload(w._payload)
    assert w._amp_excluded == {"A": {2}}
    fn = np.array([0, 1])
    q = np.zeros(2)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([1.0, 2.0]),
        components=[[0, 1]],
    ))
    assert w._amp_excluded == {}


def test_amp_track_select_dialog(phase_window):
    """The Select-tracks dialog: one sortable table per structure with
    a mean-amplitude column; unchecking applies live (checks keyed to
    the track via UserRole, so sorting cannot desync them); All/None
    batch-toggle; the button label shows the off-count."""
    w = phase_window
    w.set_track_phases({0: ["A"], 1: ["A"], 2: ["A"]})
    assert w.btn_amp_tracks.isEnabled()
    w.amp_group.setChecked(True)
    w._on_select_amp_tracks()
    dlg = w._amp_track_dialog
    assert dlg is not None and not dlg.isModal()
    assert dlg._tabs.count() == 1 and dlg._tabs.tabText(0) == "A (3)"
    table = dlg._tables["A"]
    assert table.rowCount() == 3
    amp_col = [
        table.item(r, 6).data(Qt.ItemDataRole.EditRole) for r in range(3)
    ]
    assert amp_col == [9.0, 5.5, 3.5]

    # Sort ascending by mean amplitude, then untick the faintest row:
    # the RIGHT track (index 2) is excluded despite the reorder.
    table.sortItems(6, Qt.SortOrder.AscendingOrder)
    assert table.item(0, 0).data(Qt.ItemDataRole.UserRole) == 2
    table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert w._amp_excluded == {"A": {2}}
    assert "(1 off)" in w.btn_amp_tracks.text()
    curves, _lines = _split_curves_extreme_lines(
        w.amp_plot.getPlotItem().listDataItems()
    )
    assert len(curves) == 2

    dlg._set_all("A", False)
    assert w._amp_excluded == {"A": {0, 1, 2}}
    assert w._amp_groups().get("A", []) == []
    dlg._set_all("A", True)
    assert w._amp_excluded == {"A": set()}
    assert w.btn_amp_tracks.text() == "Select tracks…"
    # Context-less host (this fixture): the dialog builds WITHOUT the
    # preview panel and all preview paths are inert.
    assert dlg._preview is None


def _preview_scan_file(tmp_path):
    """A minimal file matching the preview's read path: entry group
    "e/data" with a signal attr, a (4, 8, 12) stack and q axes."""
    import h5py

    path = tmp_path / "scan.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("e/data")
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q", data=rng.random((4, 8, 12)).astype(np.float32)
        )
        data.create_dataset("q_xy", data=np.linspace(-1.0, 2.0, 12))
        data.create_dataset("q_z", data=np.linspace(0.0, 3.0, 8))
    return path


def test_track_select_preview_highlights_selected_track(qtbot, tmp_path):
    """The Select-tracks preview: frame image from the file, slider
    spanning the scan, selected row ringed on its frames and jumped-to."""
    from mlgidgui.phase_views_window import PhaseViewsWindow

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    w.set_context(str(_preview_scan_file(tmp_path)), "e", 4)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.array([0.5, 0.6, 1.0, 1.1]),
        q_z=np.array([1.0, 1.1, 2.0, 2.1]),
        frame_num=np.array([0, 1, 2, 3]),
        amplitude=np.array([5.0, 6.0, 7.0, 8.0]),
        components=[[0, 1], [2, 3]],  # track 0: frames 0-1, track 1: 2-3
    )
    w.set_payload(payload)
    w.set_track_phases({0: ["A"], 1: ["A"]})
    w._on_select_amp_tracks()
    dlg = w._amp_track_dialog
    assert dlg._preview is not None
    assert dlg.preview_slider.maximum() == 3
    # Opening auto-selects the first row (track 0, default Track sort):
    # slider on its first frame, marker on its frame-0 member.
    assert dlg._selected_track == 0
    assert dlg.preview_slider.value() == 0
    dlg._do_load_preview_frame()  # bypass the debounce timer
    assert dlg._preview_image.isVisible()
    assert not dlg._preview_missing.isVisible()
    x, y = dlg._preview_marker.getData()
    assert list(x) == [0.5] and list(y) == [1.0]
    # Selecting the other track jumps the slider to ITS first frame and
    # moves the ring; the faint trajectory holds the whole track.
    table = dlg._tables["A"]
    row1 = next(
        r for r in range(table.rowCount())
        if int(table.item(r, 0).data(Qt.ItemDataRole.UserRole)) == 1
    )
    table.selectRow(row1)
    assert dlg._selected_track == 1
    assert dlg.preview_slider.value() == 2
    x, y = dlg._preview_marker.getData()
    assert list(x) == [1.0] and list(y) == [2.0]
    tx, ty = dlg._preview_traj.getData()
    assert list(tx) == [1.0, 1.1]
    # A frame without members of the selected track: ring gone,
    # trajectory kept, image still loads.
    dlg.preview_slider.setValue(0)
    x, y = dlg._preview_marker.getData()
    assert x is None or len(x) == 0
    tx, ty = dlg._preview_traj.getData()
    assert list(tx) == [1.0, 1.1]
    dlg._do_load_preview_frame()
    assert dlg._preview_image.isVisible()


def test_track_select_preview_jumps_to_subset_first_frame(qtbot, tmp_path):
    """Regression: the row-click jump must target the first frame of
    the row's STRUCTURE SUBSET (the "First" column), not of the whole
    track. A track fitted on frames 0..3 but claimed by matching only
    on 2..3 jumped to frame 0, where the tab's ring cannot appear."""
    from mlgidgui.phase_views_window import PhaseViewsWindow

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    w.set_context(str(_preview_scan_file(tmp_path)), "e", 4)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.array([0.5, 0.6, 0.7, 0.8]),
        q_z=np.array([1.0, 1.1, 1.2, 1.3]),
        frame_num=np.array([0, 1, 2, 3]),
        amplitude=np.array([5.0, 6.0, 7.0, 8.0]),
        components=[[0, 1, 2, 3]],
    )
    w.set_payload(payload)
    w.set_track_phases({0: ["A"]})
    w.set_member_phases({2: ["A"], 3: ["A"]})  # claimed from frame 2 only
    w._on_select_amp_tracks()
    dlg = w._amp_track_dialog
    # Tab "A" holds the claimed subset (First column = 2): the auto-
    # selected row must land the slider on 2 with the frame-2 member
    # ringed; the trajectory holds only the subset's two members.
    table = dlg._tables["A"]
    assert int(table.item(0, 3).data(Qt.ItemDataRole.EditRole)) == 2
    assert dlg.preview_slider.value() == 2
    x, y = dlg._preview_marker.getData()
    assert list(x) == [0.7] and list(y) == [1.2]
    tx, _ty = dlg._preview_traj.getData()
    assert list(tx) == [0.7, 0.8]
    # The unmatched tab's subset starts at frame 0: switching to it and
    # selecting its row jumps there.
    from mlgidgui.phase_views_window import _UNMATCHED

    idx = dlg._tab_keys.index(_UNMATCHED)
    dlg._tabs.setCurrentIndex(idx)
    dlg._tables[_UNMATCHED].selectRow(0)
    assert dlg.preview_slider.value() == 0
    x, y = dlg._preview_marker.getData()
    assert list(x) == [0.5] and list(y) == [1.0]


def test_track_select_preview_survives_missing_file(qtbot, tmp_path):
    """Context set but the file does not exist (and could vanish in real
    use): the preview degrades to a placeholder, never raises, and row
    selection still drives the slider."""
    w = _late_track_window(qtbot, tmp_path)  # x.h5 is never written
    w.set_track_phases({0: ["A"], 1: ["A"]})
    w._on_select_amp_tracks()
    dlg = w._amp_track_dialog
    assert dlg._preview is not None
    # Auto-selection picked track 0 (frames 2..3) and jumped the slider.
    assert dlg._selected_track == 0
    assert dlg.preview_slider.value() == 2
    dlg._do_load_preview_frame()  # must not raise
    assert not dlg._preview_image.isVisible()
    assert dlg._preview_missing.isVisible()
    assert "no image" in dlg._preview_missing.toPlainText()
    table = dlg._tables["A"]
    row1 = next(
        r for r in range(table.rowCount())
        if int(table.item(r, 0).data(Qt.ItemDataRole.UserRole)) == 1
    )
    table.selectRow(row1)
    assert dlg.preview_slider.value() == 0  # track 1 starts at frame 0


def test_amp_export_mirrors_zero_and_top(qtbot, tmp_path):
    """The amplitude CSV exports the numbers AS DISPLAYED: per-track
    rows stay unpadded even with the toggle checked (median-only
    feature, effective state recorded), and the median CSV carries the
    padded zero rows."""
    w = _late_track_window(qtbot, tmp_path)
    w.amp_zero.setChecked(True)
    # Per-track mode: toggle ineffective (disabled) -> no zero rows,
    # but EVERY frame of the exported range gets a row (frames without
    # a fitted value carry an explicit nan, never a skipped row).
    w._export_data_files(["amplitude"], str(tmp_path / "raw"))
    lines = (tmp_path / "raw_amplitude.csv").read_text().splitlines()
    assert "metric=amplitude" in lines[0]
    assert "zeros=off" in lines[0] and "tracks=all" in lines[0]
    rows = [line.split(",") for line in lines[2:]]
    track0 = dict(
        (int(r[2]), float(r[3])) for r in rows if r[0] == "0"
    )
    assert sorted(track0) == [0, 1, 2, 3, 4, 5]
    assert (track0[2], track0[3]) == (7.0, 9.0)
    assert all(np.isnan(track0[f]) for f in (0, 1, 4, 5))

    # Median mode: the padded structure median lands in the CSV, and a
    # track unticked in the Select-tracks dialog drops out of it.
    w.set_track_phases({0: ["A"], 1: ["A"]})
    w.amp_median.setChecked(True)
    w._amp_excluded = {"A": {1}}
    w._export_data_files(["amplitude"], str(tmp_path / "med"))
    lines = (tmp_path / "med_amplitude.csv").read_text().splitlines()
    assert "zeros=lead+tail" in lines[0] and "tracks=custom" in lines[0]
    rows = [line.split(",") for line in lines[2:]]
    a_rows = [(int(r[1]), float(r[2])) for r in rows if r[0] == "A"]
    # Track 1 (frames 0 and 3) excluded: only track 0's 2..3 values
    # remain, zero-padded lead (0..1) and tail (4..5).
    assert a_rows == [
        (0, 0.0), (1, 0.0), (2, 7.0), (3, 9.0), (4, 0.0), (5, 0.0),
    ]


def test_member_phases_split_track_at_matching_start(qtbot):
    """Per-frame phase attribution: a track fitted on frames 0..3 but
    matched only on 2..3 renders 2..3 in the phase colour and 0..1 as
    unmatched — in the trajectories (matched colouring), the grouped
    amplitude bands and the q-map legend — instead of painting the
    whole span with the dominant phase."""
    from mlgidgui.phase_views_window import PhaseViewsWindow, _UNMATCHED

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    fn = np.array([0, 1, 2, 3])
    q = np.zeros(4)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([1.0, 2.0, 3.0, 4.0]),
        components=[[0, 1, 2, 3]],
    )
    w.set_payload(payload)
    w.set_track_phases({0: ["A"]})
    w.set_member_phases({2: ["A"], 3: ["A"]})   # claimed on 2..3 only

    assert w._member_structure_key(0, 0) == _UNMATCHED
    assert w._member_structure_key(2, 0) == "A"
    assert {
        k: list(v) for k, v in w._member_subsets(0).items()
    } == {_UNMATCHED: [0, 1], "A": [2, 3]}
    # The toggles gain an "unmatched" entry although every TRACK has
    # a phase.
    assert set(w._struct_checks) == {"A", _UNMATCHED}

    # Trajectories, matched colouring: the track splits into two curves
    # at the matching start.
    w.traj_color.setCurrentIndex(1)
    curves = w.traj_plot.getPlotItem().listDataItems()
    assert sorted(tuple(c.xData) for c in curves) == [
        (0.0, 1.0), (2.0, 3.0),
    ]
    # q-map phase mode: legend lists the phase AND unmatched.
    w.qmap_phase.setChecked(True)
    assert len(w._qmap_legend.items) == 2
    # Grouped amplitude: the claimed members build the A band, the
    # unclaimed ones the unmatched band above it.
    w.amp_group.setChecked(True)
    assert [l.toPlainText() for l in w._amp_labels] == ["A", "unmatched"]
    bands = w.amp_plot.getPlotItem().listDataItems()
    xdatas = sorted(tuple(c.xData) for c in bands if len(c.xData) == 2)
    assert (2.0, 3.0) in xdatas and (0.0, 1.0) in xdatas
    # Unticking "unmatched" hides exactly the unclaimed portion.
    w._struct_checks[_UNMATCHED].setChecked(False)
    curves = w.traj_plot.getPlotItem().listDataItems()
    assert [tuple(c.xData) for c in curves] == [(2.0, 3.0)]


def test_member_phases_pushed_from_matching(
    main_window, synthetic_fitted_scan,
):
    """End to end: after tracking a scan whose matched data covers only
    frame 2, the member-phase map holds exactly that member and reaches
    the phase views."""
    _write_matched(synthetic_fitted_scan, 2, [("PbI2", [0])])
    _open(main_window, synthetic_fitted_scan)
    _run_result(main_window, synthetic_fitted_scan)
    mp = main_window._scan_member_phases
    assert list(mp.values()) == [["PbI2"]]
    (idx,) = mp.keys()
    assert main_window._scan_member_ids[idx][0] == 2   # the claimed frame
    main_window._on_show_phase_views()
    assert main_window._phase_views_window._member_phases == mp


# --- amplitude grouping by structure + per-structure toggles ---

@pytest.fixture
def phase_window(qtbot):
    """A PhaseViewsWindow with 3 tracks (2 matched: A, B; 1 unmatched)
    and a phase map installed — the amplitude-grouping test bed."""
    from mlgidgui.phase_views_window import PhaseViewsWindow
    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    fn = np.array([0, 1, 0, 1, 0, 1])
    q = np.zeros(6)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([10., 8., 5., 6., 3., 4.]),
        components=[[0, 1], [2, 3], [4, 5]],
    )
    w.set_payload(payload)
    w.set_track_phases({0: ["A"], 1: ["B"]})   # track 2 is unmatched
    return w


def test_amp_group_and_toggles_built(phase_window):
    from mlgidgui.phase_views_window import _UNMATCHED
    w = phase_window
    assert w.amp_group.isEnabled()
    assert set(w._struct_checks) == {"A", "B", _UNMATCHED}
    assert not w._struct_toggle_widget.isHidden()


def test_amp_grouped_stacks_structures(phase_window):
    w = phase_window
    w.amp_group.setChecked(True)
    items = w.amp_plot.getPlotItem().listDataItems()
    curves, lines = _split_curves_extreme_lines(items)
    assert len(curves) == 3          # (+ 6 dashed max/min lines)
    assert len(lines) == 6
    # Each track normalizes to max 1.0, offset by 1.3 per band ->
    # curve maxima at 1.0 / 2.3 / 3.6 (A / B / unmatched).
    assert sorted(round(float(max(it.yData)), 2) for it in curves) == [
        1.0, 2.3, 3.6,
    ]
    assert [l.toPlainText() for l in w._amp_labels] == ["A", "B", "unmatched"]
    # Labels sit in the inter-band gap ABOVE their band (bottom-left
    # anchored), not on the curves: y just above baseline + 1.0.
    assert [round(float(l.pos().y()), 2) for l in w._amp_labels] == [
        1.02, 2.32, 3.62,
    ]
    assert all(tuple(l.anchor) == (0.0, 1.0) for l in w._amp_labels)
    # Grouped mode always normalizes -> the manual toggle is disabled.
    assert not w.amp_normalize.isEnabled()


def test_amp_toggle_hides_structure_and_repacks(phase_window):
    w = phase_window
    w.amp_group.setChecked(True)
    w._struct_checks["B"].setChecked(False)
    items = w.amp_plot.getPlotItem().listDataItems()
    curves, lines = _split_curves_extreme_lines(items)
    assert len(curves) == 2   # A + unmatched only (+ their 4 lines)
    assert len(lines) == 4
    # Bands re-pack: no gap left where B was.
    assert sorted(round(float(max(it.yData)), 2) for it in curves) == [1.0, 2.3]
    assert [l.toPlainText() for l in w._amp_labels] == ["A", "unmatched"]


def test_amp_toggle_filters_ungrouped(phase_window):
    w = phase_window
    assert len(w.amp_plot.getPlotItem().listDataItems()) == 3
    w._struct_checks["A"].setChecked(False)   # ungrouped: hide track 0
    assert len(w.amp_plot.getPlotItem().listDataItems()) == 2


def test_amp_median_per_structure_with_iqr_band(qtbot):
    """Median mode collapses a structure's tracks into ONE per-frame
    median curve with a transparent interquartile (25th-75th
    percentile) band around it."""
    from mlgidgui.phase_views_window import PhaseViewsWindow
    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    # Two tracks, BOTH structure A: amplitudes 10/8 and 6/4 on frames
    # 0/1 -> median [8, 6]; quartiles of [10, 6] are 7/9 and of
    # [8, 4] are 5/7 (linear interpolation between two values).
    fn = np.array([0, 1, 0, 1])
    q = np.zeros(4)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=fn, amplitude=np.array([10., 8., 6., 4.]),
        components=[[0, 1], [2, 3]],
    ))
    w.set_track_phases({0: ["A"], 1: ["A"]})
    w.amp_window.setValue(1)
    w.amp_normalize.setChecked(False)
    assert w.amp_median.isEnabled()
    w.amp_median.setChecked(True)
    items = w.amp_plot.getPlotItem().listDataItems()
    # One structure -> lower bound + upper bound + median = 3 curves,
    # plus exactly one fill item for the band (no extreme markers in
    # the ungrouped view).
    assert len(items) == 3
    assert len(w._amp_fills) == 1
    ys = sorted(tuple(np.round(np.asarray(it.yData), 6)) for it in items)
    assert ys == [
        (7.0, 5.0),    # 25th percentile
        (8.0, 6.0),    # median
        (9.0, 7.0),    # 75th percentile
    ]
    # Off again: back to the individual per-track curves, band gone.
    w.amp_median.setChecked(False)
    assert len(w.amp_plot.getPlotItem().listDataItems()) == 2
    assert w._amp_fills == []


def _split_curves_extreme_lines(items):
    """Grouped-view items -> (data/band curves, dashed max/min
    reference lines). The extreme lines are the only DASHED items in
    the amplitude plot."""
    lines = [
        it for it in items
        if it.opts.get("pen") is not None
        and it.opts["pen"].style() == Qt.PenStyle.DashLine
    ]
    curves = [it for it in items if it not in lines]
    return curves, lines


def test_amp_median_combines_with_grouping(phase_window):
    """Grouped + median: one median band per structure at its offset
    (single-track structures collapse to the track itself, IQR width
    zero) plus thin dashed max/min reference lines per structure."""
    w = phase_window
    w.amp_window.setValue(1)          # raw values -> exact expectations
    w.amp_group.setChecked(True)
    w.amp_median.setChecked(True)
    curves, lines = _split_curves_extreme_lines(
        w.amp_plot.getPlotItem().listDataItems()
    )
    assert len(curves) == 9                   # 3 structures x (lo+hi+median)
    assert len(lines) == 6                    # max + min line per structure
    assert len(w._amp_fills) == 3
    # Per-track normalized medians keep the stacked maxima at the
    # band offsets (IQR collapses for single-track structures).
    maxima = sorted(round(float(max(it.yData)), 2) for it in curves)
    assert maxima == [1.0, 1.0, 1.0, 2.3, 2.3, 2.3, 3.6, 3.6, 3.6]
    assert [l.toPlainText() for l in w._amp_labels] == ["A", "B", "unmatched"]
    # Horizontal reference lines at each band's median max and min
    # (A dips to 0.8, B to 5/6+1.3, unmatched to 0.75+2.6).
    assert all(it.yData[0] == it.yData[1] for it in lines)
    line_ys = sorted(round(float(it.yData[0]), 4) for it in lines)
    assert line_ys == [0.8, 1.0, 2.1333, 2.3, 3.35, 3.6]


def test_amp_grouped_extreme_lines_without_median(phase_window):
    """Grouped view draws each structure's median-series extremes as
    thin dashed horizontal lines spanning the structure's frame range,
    even with median mode off — every point reads off its distance to
    the structure's own max/min."""
    w = phase_window
    w.amp_window.setValue(1)          # raw values -> exact expectations
    w.amp_group.setChecked(True)
    curves, lines = _split_curves_extreme_lines(
        w.amp_plot.getPlotItem().listDataItems()
    )
    assert len(curves) == 3 and len(lines) == 6
    # Horizontal, spanning the full frame range of each structure.
    for it in lines:
        assert it.yData[0] == it.yData[1]
        assert list(it.xData) == [0.0, 1.0]
    # A (amps 10, 8) -> max 1.0 / min 0.8; B (5, 6) -> 2.3 / 5/6+1.3;
    # unmatched (3, 4) -> 3.6 / 0.75+2.6.
    line_ys = sorted(round(float(it.yData[0]), 4) for it in lines)
    assert line_ys == [0.8, 1.0, 2.1333, 2.3, 3.35, 3.6]


def test_structure_toggle_filters_qmap_and_trajectories(phase_window):
    """The window-wide per-structure toggles govern the q-map and the
    trajectories too, not just the amplitude tab."""
    w = phase_window
    # 3 tracks -> q-map draws 2 items each (trajectory + mean marker).
    assert len(w.qmap_plot.getPlotItem().listDataItems()) == 6
    assert len(w.traj_plot.getPlotItem().listDataItems()) == 3
    w._struct_checks["B"].setChecked(False)
    assert len(w.qmap_plot.getPlotItem().listDataItems()) == 4   # B gone
    assert len(w.traj_plot.getPlotItem().listDataItems()) == 2


def test_qmap_legend_drops_hidden_structure(phase_window):
    w = phase_window
    w.qmap_phase.setChecked(True)
    # A / B / unmatched (track 2 has no phase) -> 3 legend rows.
    assert len(w._qmap_legend.items) == 3
    w._struct_checks["B"].setChecked(False)
    assert len(w._qmap_legend.items) == 2   # B dropped from the legend


def test_heatmaps_use_gui_colormap(phase_window):
    """Waterfall + q-map mean image render with the GUI's GIWAXS
    colormap (magma), not raw greyscale."""
    w = phase_window
    w._on_worker_finished(
        {"kind": "waterfall",
         "data": np.random.default_rng(0).random((6, 32)).astype("f4"),
         "radius": np.linspace(0.0, 4.0, 32), "entry": "e"},
        None,
    )
    w._on_worker_finished(
        {"kind": "mean_image",
         "data": np.random.default_rng(1).random((16, 24)).astype("f4"),
         "q_xy": np.linspace(-1, 3, 24), "q_z": np.linspace(0, 4, 16),
         "lo": 0, "hi": 5, "entry": "e"},
        None,
    )
    assert w.waterfall_image._colorMap is not None
    assert w.qmap_image._colorMap is not None


def test_amp_group_disabled_without_phases(qtbot):
    from mlgidgui.phase_views_window import PhaseViewsWindow
    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    q = np.zeros(2)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=np.array([0, 1]), amplitude=np.array([1., 2.]),
        components=[[0, 1]],
    ))
    w.set_track_phases({})
    assert not w.amp_group.isEnabled()
    assert not w.amp_median.isEnabled()
    assert w._struct_toggle_widget.isHidden()


# --- ROI-intensity metric (integrated intensity, image-based) ---

def _roi_scan_window(qtbot, tmp_path):
    """A views window over a real 4-frame file: constant pedestal 1.0
    with a bright block at q = (0.4..0.6, 0.4..0.6) whose excess grows
    10-per-frame, and ONE track on the block with fitted members only
    on frames 0 and 3 (a fit gap at 1..2)."""
    import h5py

    from mlgidgui.phase_views_window import PhaseViewsWindow
    from mlgidgui.roi_intensity import axis_slice

    n_f, n_z, n_xy = 4, 40, 50
    q_xy_axis = np.linspace(0.0, 1.0, n_xy)
    q_z_axis = np.linspace(0.0, 1.0, n_z)
    z0, z1 = axis_slice(q_z_axis, 0.4, 0.6)
    x0, x1 = axis_slice(q_xy_axis, 0.4, 0.6)
    stack = np.ones((n_f, n_z, n_xy), dtype=np.float32)
    for i in range(n_f):
        stack[i, z0:z1, x0:x1] += 10.0 * (i + 1)
    path = tmp_path / "roi_scan.h5"
    with h5py.File(path, "w") as f:
        data = f.create_group("e/data")
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset("img_gid_q", data=stack)
        data.create_dataset("q_xy", data=q_xy_axis)
        data.create_dataset("q_z", data=q_z_axis)

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    w.set_context(str(path), "e", n_f)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.array([0.5, 0.5]), q_z=np.array([0.5, 0.5]),
        frame_num=np.array([0, 3]), amplitude=np.array([3.0, 9.0]),
        components=[[0, 1]],
    ))
    w.amp_window.setValue(1)
    w.amp_normalize.setChecked(False)
    return w, stack, q_xy_axis, q_z_axis


def test_amp_metric_roi_computes_gapfree_trace(qtbot, tmp_path):
    """Switching the metric to "integrated intensity (ROI)" computes
    per-track traces from the image data in a background worker: the
    curve covers EVERY frame (the fit gap at 1..2 and the member-less
    frames borrow the nearest fitted position), and the values match
    the unit-tested ``integrate_roi`` on the same frames."""
    from mlgidgui.roi_intensity import integrate_roi

    w, stack, q_xy_axis, q_z_axis = _roi_scan_window(qtbot, tmp_path)
    assert w.amp_metric.currentData() == "amplitude"   # default
    assert w._roi_controls.isHidden()

    # Wider box than the default so the whole block integrates.
    w.roi_dqxy.setValue(0.12)
    w.roi_dqz.setValue(0.12)
    w.amp_metric.setCurrentIndex(1)
    assert not w._roi_controls.isHidden()
    qtbot.waitUntil(lambda: w._roi_cache_valid(), timeout=30000)
    qtbot.waitUntil(lambda: w._worker_thread is None, timeout=30000)

    curves = w.amp_plot.getPlotItem().listDataItems()
    assert len(curves) == 1
    xs = np.asarray(curves[0].xData)
    ys = np.asarray(curves[0].yData)
    assert list(xs) == [0, 1, 2, 3]           # gap-free
    expected = [
        integrate_roi(
            stack[i].astype(float), q_xy_axis, q_z_axis,
            0.5, 0.5, 0.12, 0.12, 2, 4,
        )
        for i in range(4)
    ]
    np.testing.assert_allclose(ys, expected, rtol=1e-5)
    assert np.all(np.diff(ys) > 0)            # grows 10/frame

    # CSV export: metric + ROI geometry recorded, value column renamed,
    # one row per frame of the range.
    w._export_data_files(["amplitude"], str(tmp_path / "roi"))
    lines = (tmp_path / "roi_amplitude.csv").read_text().splitlines()
    assert "metric=integrated intensity (ROI)" in lines[0]
    assert "roi=+-0.12/+-0.12 1/A bg gap 2 px strips 4 px" in lines[0]
    assert lines[1].split(",")[-1] == "roi_intensity"
    rows = [line.split(",") for line in lines[2:]]
    assert [int(r[2]) for r in rows] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [float(r[3]) for r in rows], expected, rtol=1e-5,
    )

    # Back to the fitted-amplitude metric: member-frame curves return.
    w.amp_metric.setCurrentIndex(0)
    assert w._roi_controls.isHidden()
    curves = w.amp_plot.getPlotItem().listDataItems()
    assert list(curves[0].xData) == [0, 3]


def test_amp_metric_roi_without_context_stays_empty(phase_window):
    """A tracking result without scan context (no file to integrate)
    leaves the ROI tab empty instead of crashing or spawning a worker;
    switching back restores the amplitude curves."""
    w = phase_window
    w.amp_metric.setCurrentIndex(1)
    assert w.amp_plot.getPlotItem().listDataItems() == []
    assert w._worker_thread is None
    w.amp_metric.setCurrentIndex(0)
    assert len(w.amp_plot.getPlotItem().listDataItems()) == 3


def test_roi_key_frames_partition_by_nearest_member(qtbot, tmp_path):
    """Per-frame phase attribution for ROI traces: each frame of the
    interval follows the key of the nearest member frame, so a track
    matched only from frame 2 splits its trace into an unmatched head
    and a matched tail (synthesized tail frames included)."""
    from mlgidgui.phase_views_window import PhaseViewsWindow, _UNMATCHED

    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    w.set_context(str(tmp_path / "x.h5"), "e", 6)
    q = np.zeros(4)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=np.array([0, 1, 2, 3]),
        amplitude=np.array([1.0, 2.0, 3.0, 4.0]),
        components=[[0, 1, 2, 3]],
    ))
    w.set_track_phases({0: ["A"]})
    w.set_member_phases({2: ["A"], 3: ["A"]})   # claimed on 2..3 only
    parts = w._roi_key_frames(0)
    assert {k: list(v) for k, v in parts.items()} == {
        _UNMATCHED: [0, 1], "A": [2, 3, 4, 5],
    }
    # Single-key track: the whole interval in one part.
    w.set_member_phases({})
    assert list(w._roi_key_frames(0)["A"]) == [0, 1, 2, 3, 4, 5]


def test_amp_xrange_pinned_to_frame_interval(phase_window):
    """The amplitude x-axis fits the tracked frame interval — the
    grouped view's structure labels no longer feed the auto-range
    (which used to run away to thousands of frames)."""
    w = phase_window
    w.amp_group.setChecked(True)
    lo, hi = w._frame_interval()
    span = hi - lo
    (x0, x1), _y = w.amp_plot.getPlotItem().getViewBox().viewRange()
    assert x0 == pytest.approx(lo - 0.02 * span, abs=1e-9)
    assert x1 == pytest.approx(hi + 0.02 * span, abs=1e-9)


def _ring_window(qtbot):
    """A views window with 2 spot tracks + 1 ring track (index 2), real
    (q_xy, q_z), and phases A/B on the spots."""
    from mlgidgui.phase_views_window import PhaseViewsWindow
    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    fn = np.array([0, 1, 0, 1, 0, 1])
    qx = np.array([0.7, 0.71, 1.4, 1.41, 1.27, 1.27])
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=qx, q_z=qx.copy(),
        frame_num=fn, amplitude=np.array([10., 8., 5., 6., 3., 4.]),
        components=[[0, 1], [2, 3], [4, 5]],
    )
    w.set_payload(payload)
    w.set_ring_tracks({2})
    w.set_track_phases({0: ["A"], 1: ["B"]})
    return w


def test_qmap_ring_drawn_as_dashed_arc(qtbot):
    from PySide6.QtCore import Qt
    import pyqtgraph as pg
    w = _ring_window(qtbot)
    items = w.qmap_plot.getPlotItem().listDataItems()
    # 2 spot tracks -> traj + marker each (4); 1 ring -> one dashed arc.
    assert len(items) == 5
    dashed = [
        it for it in items
        if it.opts.get("pen") is not None
        and pg.mkPen(it.opts["pen"]).style() == Qt.PenStyle.DashLine
    ]
    assert len(dashed) == 1
    arc = dashed[0]
    r = np.hypot(np.asarray(arc.xData), np.asarray(arc.yData))
    # Constant radius = the ring track's mean |q|.
    assert np.allclose(r, w._payload.track_mean("radius", 2), atol=1e-6)
    # Quarter circle: starts on the q_xy axis, ends on the q_z axis.
    assert arc.yData[0] == pytest.approx(0.0, abs=1e-6)
    assert arc.xData[-1] == pytest.approx(0.0, abs=1e-6)


def test_trajectories_color_by_matched(qtbot):
    import pyqtgraph as pg
    from PySide6.QtGui import QColor
    w = _ring_window(qtbot)
    # 'matched' option enabled with phases; default is 'fitted'.
    assert w.traj_color.model().item(1).isEnabled()
    assert w.traj_color.currentData() == "fitted"
    w.traj_color.setCurrentIndex(1)   # matched

    def _rgb(c):
        q = QColor(c)
        return (q.red(), q.green(), q.blue())

    pens = [
        _rgb(pg.mkPen(it.opts["pen"]).color())
        for it in w.traj_plot.getPlotItem().listDataItems()
    ]
    assert _rgb(w._phase_colors["A"]) in pens
    assert _rgb(w._phase_colors["B"]) in pens


def test_trajectories_matched_disabled_without_phases(qtbot):
    from mlgidgui.phase_views_window import PhaseViewsWindow
    w = PhaseViewsWindow()
    qtbot.addWidget(w)
    q = np.zeros(2)
    w.set_payload(TrackingPayload(
        entry="e", threshold=0.5, length=1, q_xy=q, q_z=q.copy(),
        frame_num=np.array([0, 1]), amplitude=np.array([1., 2.]),
        components=[[0, 1]],
    ))
    w.set_track_phases({})
    assert not w.traj_color.model().item(1).isEnabled()


# --- plot data / image export ---

def test_export_data_trajectories_csv(phase_window, tmp_path):
    """The trajectories CSV carries EVERY track member with all axis
    values plus track/structure identity — enough to redo any of the
    plots externally."""
    w = phase_window
    written = w._export_data_files(["trajectories"], str(tmp_path / "out"))
    assert written == [str(tmp_path / "out_trajectories.csv")]
    lines = (tmp_path / "out_trajectories.csv").read_text().splitlines()
    assert lines[0].startswith("#")
    assert lines[1].split(",") == [
        "track", "structure", "all_structures", "is_ring", "frame",
        "radius", "angle", "amplitude", "q_xy", "q_z",
    ]
    rows = [line.split(",") for line in lines[2:]]
    assert len(rows) == 6
    # Track 0 = structure A, raw amplitude 10 on frame 0.
    assert rows[0][:5] == ["0", "A", "A", "0", "0"]
    assert float(rows[0][7]) == 10.0
    # Track 2 has no phase.
    assert rows[4][1] == "unmatched" and rows[4][2] == ""


def test_export_data_amplitude_modes(phase_window, tmp_path):
    """The amplitude CSV mirrors the tab's current mode: per-track
    processed values normally, per-structure median/quartile columns in
    median mode."""
    w = phase_window
    w.amp_window.setValue(1)
    w.amp_normalize.setChecked(False)
    w._export_data_files(["amplitude"], str(tmp_path / "raw"))
    lines = (tmp_path / "raw_amplitude.csv").read_text().splitlines()
    assert lines[1].split(",") == ["track", "structure", "frame", "amplitude"]
    rows = [line.split(",") for line in lines[2:]]
    assert [float(r[3]) for r in rows] == [10.0, 8.0, 5.0, 6.0, 3.0, 4.0]
    # Median mode: one row per (structure, frame); single-track
    # structures collapse the quartiles onto the median.
    w.amp_median.setChecked(True)
    w._export_data_files(["amplitude"], str(tmp_path / "med"))
    lines = (tmp_path / "med_amplitude.csv").read_text().splitlines()
    assert lines[1].split(",") == ["structure", "frame", "median", "q25", "q75"]
    rows = [line.split(",") for line in lines[2:]]
    assert [(r[0], r[1]) for r in rows] == [
        ("A", "0"), ("A", "1"), ("B", "0"), ("B", "1"),
        ("unmatched", "0"), ("unmatched", "1"),
    ]
    assert [float(r[2]) for r in rows] == [10.0, 8.0, 5.0, 6.0, 3.0, 4.0]
    assert all(r[2] == r[3] == r[4] for r in rows)


def test_export_data_qmap_and_waterfall_matrices(phase_window, tmp_path):
    """q-map export = member table with track means (+ the mean-image
    matrix when computed); waterfall export = frame x |q| matrix with
    the axes on the edges."""
    w = phase_window
    w._waterfall = {
        "kind": "waterfall", "data": np.arange(6.0).reshape(2, 3),
        "radius": [1.0, 2.0, 3.0],
    }
    w._mean_image = {
        "kind": "mean_image", "data": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "q_xy": [0.0, 0.5], "q_z": [0.0, 1.0],
    }
    written = w._export_data_files(["qmap", "waterfall"], str(tmp_path / "o"))
    assert [os.path.basename(p) for p in written] == [
        "o_qmap_tracks.csv", "o_qmap_mean_image.csv", "o_waterfall.csv",
    ]
    qm = (tmp_path / "o_qmap_tracks.csv").read_text().splitlines()
    assert qm[1].split(",") == [
        "track", "structure", "all_structures", "is_ring", "frame",
        "q_xy", "q_z", "track_mean_q_xy", "track_mean_q_z",
        "track_mean_radius",
    ]
    assert len(qm) == 2 + 6
    wf = (tmp_path / "o_waterfall.csv").read_text().splitlines()
    assert wf[1].split(",") == ["frame\\|q|", "1", "2", "3"]
    assert wf[2].split(",") == ["0", "0", "1", "2"]
    assert wf[3].split(",") == ["1", "3", "4", "5"]
    mi = (tmp_path / "o_qmap_mean_image.csv").read_text().splitlines()
    assert mi[1].split(",") == ["q_z\\q_xy", "0", "0.5"]
    assert mi[2].split(",") == ["0", "1", "2"]
    assert mi[3].split(",") == ["1", "3", "4"]


def test_export_style_overrides_render(phase_window):
    """The image-export overrides restyle the live render (structure
    subset, pen width/style, marker hiding) and clearing them restores
    the historical look."""
    w = phase_window
    w._export_overrides = {
        "structures": {"A"}, "line_width": 3.0,
        "line_style": Qt.PenStyle.DotLine, "marker_size": 0,
    }
    w._refresh_trajectories()
    items = w.traj_plot.getPlotItem().listDataItems()
    assert len(items) == 1            # only structure A survives
    pen = items[0].opts["pen"]
    assert pen.widthF() == 3.0
    assert pen.style() == Qt.PenStyle.DotLine
    assert items[0].opts["symbol"] is None
    w._export_overrides = {}
    w._refresh_trajectories()
    items = w.traj_plot.getPlotItem().listDataItems()
    assert len(items) == 3
    assert items[0].opts["symbol"] == "o"
    assert items[0].opts["pen"].widthF() == 1.0


def test_export_images_write_and_restore(phase_window, tmp_path):
    """Image export renders each selected view to file with the chosen
    styling, then restores the overrides, the theme and the on-screen
    plots — even for a white-background export."""
    w = phase_window
    w._waterfall = {
        "kind": "waterfall", "data": np.arange(6.0).reshape(2, 3),
        "radius": [1.0, 2.0, 3.0],
    }
    w._refresh_waterfall()
    written = w._export_image_files(
        ["trajectories", "qmap", "amplitude", "waterfall"],
        str(tmp_path / "img"), ".png",
        structures={"A", "B"}, line_width=2.0,
        line_style=Qt.PenStyle.SolidLine, marker_size=6,
        width_px=640, white_bg=True,
    )
    assert [os.path.basename(p) for p in written] == [
        "img_trajectories.png", "img_qmap.png",
        "img_amplitude.png", "img_waterfall.png",
    ]
    assert all(os.path.getsize(p) > 0 for p in written)
    # Overrides cleared, screen rendering back to all 3 tracks.
    assert w._export_overrides == {}
    assert len(w.traj_plot.getPlotItem().listDataItems()) == 3
    assert len(w.amp_plot.getPlotItem().listDataItems()) == 3


def test_export_dialog_modes(qtbot):
    """Data mode: view checkboxes only. Image mode adds the structure
    subset + style controls; OK gates on at least one selected view."""
    from mlgidgui.phase_views_window import _ExportDialog
    views = [
        ("trajectories", "Trajectories", True, ""),
        ("qmap", "q-map overlay", True, ""),
        ("amplitude", "Amplitude evolution", True, ""),
        ("waterfall", "Radial waterfall", False, "Compute first."),
    ]
    structures = [("A", "A", "#ff0000", True), ("B", "B", "#00ff00", False)]
    dlg = _ExportDialog(
        None, "image", views, structures=structures, current_key="amplitude",
    )
    qtbot.addWidget(dlg)
    assert dlg.selected_views() == ["amplitude"]   # current tab pre-checked
    assert not dlg._view_checks["waterfall"].isEnabled()
    assert dlg.selected_structures() == {"A"}      # mirrors window toggles
    style = dlg.style()
    assert style["line_width"] == 1.0
    assert style["line_style"] == Qt.PenStyle.SolidLine
    assert style["marker_size"] == 4
    assert style["width_px"] == 1600
    assert style["white_bg"] is False
    assert dlg._ok.isEnabled()
    dlg._view_checks["amplitude"].setChecked(False)
    assert not dlg._ok.isEnabled()                 # nothing selected
    data_dlg = _ExportDialog(None, "data", views, current_key="waterfall")
    qtbot.addWidget(data_dlg)
    assert data_dlg.selected_structures() is None  # no structure section
    assert not hasattr(data_dlg, "line_width")     # no style section
    assert data_dlg.selected_views() == []         # current tab disabled


# -- single-frame data has nothing to track --------------------------------


def _tracking_shown(window) -> bool:
    return (not window._scan_tracking_dock.isHidden()
            and window.workflow_rail.is_stage_visible("track"))


def test_tracking_is_hidden_for_a_single_frame(
        main_window, synthetic_nexus_single):
    _open(main_window, synthetic_nexus_single)
    assert main_window.viewer.n_frames == 1
    dock = main_window._scan_tracking_dock
    assert dock.isHidden()
    assert not main_window.workflow_rail.is_stage_visible("track")
    assert main_window._tracking_hidden_by_rule
    # The View menu cannot bring it back.
    action = dock.toggleViewAction()
    assert not action.isEnabled()
    action.trigger()
    assert dock.isHidden()


def test_tracking_is_shown_for_many_frames(main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    assert main_window.viewer.n_frames == 3
    assert _tracking_shown(main_window)
    assert main_window._scan_tracking_dock.toggleViewAction().isEnabled()


def test_switching_sessions_hides_and_restores_tracking(
        main_window, synthetic_nexus, synthetic_nexus_single):
    many = _open(main_window, synthetic_nexus)
    _open(main_window, synthetic_nexus_single)
    assert not _tracking_shown(main_window)
    main_window._set_active_session(many)
    assert _tracking_shown(main_window)
    assert main_window._scan_tracking_dock.toggleViewAction().isEnabled()


@pytest.fixture
def mixed_entries(tmp_path):
    """entry_0000 with one frame, entry_0001 with three."""
    import h5py

    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w", track_order=True) as f:
        for name, n_frames in (("entry_0000", 1), ("entry_0001", 3)):
            data = f.create_group(f"{name}/data")
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q", data=np.ones((n_frames, 16, 24), np.float32))
            data.create_dataset(
                "q_xy", data=np.linspace(-1.0, 3.0, 24, dtype=np.float32))
            data.create_dataset(
                "q_z", data=np.linspace(0.0, 4.0, 16, dtype=np.float32))
    return path


def test_a_dock_the_user_closed_stays_closed(main_window, mixed_entries):
    # Shown, so the View-menu action tracks the dock's visibility.
    main_window.show()
    _open(main_window, mixed_entries)
    dock = main_window._scan_tracking_dock
    assert dock.isHidden()                      # entry_0000: one frame
    main_window._load_entry_into_viewer("entry_0001")
    assert _tracking_shown(main_window)         # the rule hid it: back
    dock.toggleViewAction().trigger()           # the user closes it
    assert dock.isHidden()
    main_window._load_entry_into_viewer("entry_0000")
    assert not main_window._tracking_hidden_by_rule
    main_window._load_entry_into_viewer("entry_0001")
    assert dock.isHidden()                      # still the user's choice
    assert main_window.workflow_rail.is_stage_visible("track")
    assert dock.toggleViewAction().isEnabled()


def test_the_structure_tab_does_not_bring_tracking_back(
        main_window, synthetic_nexus_single):
    _open(main_window, synthetic_nexus_single)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    try:
        main_window.tabs.setCurrentWidget(main_window.structure_panel)
        main_window.tabs.setCurrentWidget(main_window.viewer)
        assert not main_window._scan_tracking_dock.isVisible()
        assert not main_window._profile_dock.isHidden()
        assert not main_window._scan_tracking_dock.toggleViewAction().isEnabled()
    finally:
        if main_window._h5_edit_handle is not None:
            main_window._h5_edit_handle.release()


def test_the_structure_tab_round_trip_keeps_tracking_for_many_frames(
        main_window, synthetic_nexus):
    _open(main_window, synthetic_nexus)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    try:
        main_window.tabs.setCurrentWidget(main_window.structure_panel)
        assert main_window._scan_tracking_dock.isHidden()
        main_window.tabs.setCurrentWidget(main_window.viewer)
        assert _tracking_shown(main_window)
    finally:
        if main_window._h5_edit_handle is not None:
            main_window._h5_edit_handle.release()


def test_a_raw_session_after_a_single_frame_re_enables_the_action(
        main_window, synthetic_nexus_single, synthetic_raw):
    from mlgidgui.session import RawSession

    _open(main_window, synthetic_nexus_single)
    raw = RawSession.open([synthetic_raw])
    main_window._sessions.append(raw)
    main_window._set_active_session(raw)
    dock = main_window._scan_tracking_dock
    assert dock.isHidden()                      # raw mode hides it
    assert dock.toggleViewAction().isEnabled()
    assert main_window.workflow_rail.is_stage_visible("track")
    assert not main_window._tracking_hidden_by_rule


def test_rings_stored_the_0_1_8_way_yield_one_ring_track_beside_the_spots(
    main_window, synthetic_fitted_scan, qtbot,
):
    """pygidfit 0.1.8 stores a ring with angle 45 (not NaN) and an
    infinite angular width, so upstream now links rings natively: it
    clamps the width to 45 and the boxes overlap frame to frame. The GUI
    drops those native ring components and tracks rings itself, so each
    physical ring is still exactly ONE track and the spot track is
    untouched."""
    pytest.importorskip("mlgidbase")
    import h5py

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        for frame in range(6):
            ds = f[f"{ENTRY}/data/analysis/frame{frame:05d}/fitted_peaks"]
            rows = ds[()]
            ring = np.zeros(1, dtype=rows.dtype)
            ring["id"] = int(rows["id"].max()) + 1
            ring["radius"] = 2.5
            ring["radius_width"] = 0.2
            ring["angle"] = 45.0
            ring["angle_width"] = np.inf
            ring["is_ring"] = True
            ring["amplitude"] = 30.0
            ring["q_xy"] = 2.5 * np.cos(np.deg2rad(45.0))
            ring["q_z"] = 2.5 * np.sin(np.deg2rad(45.0))
            data = np.concatenate([rows, ring])
            parent = ds.parent
            del parent["fitted_peaks"]
            parent.create_dataset("fitted_peaks", data=data)

    _open(main_window, synthetic_fitted_scan)
    main_window._on_track_scan_requested(0.5, 3)
    qtbot.waitUntil(
        lambda: main_window._pipe_thread is None
        and not main_window._pipeline_queue
        and main_window._scan_payload is not None,
        timeout=60000,
    )
    payload = main_window._scan_payload
    ids = main_window._scan_member_ids
    rings = sorted(main_window._scan_ring_tracks)
    assert len(rings) == 1
    ring_members = payload.track_members(rings[0])
    assert len(ring_members) == 6
    assert all(ids[int(i)][1] == 2 or ids[int(i)][1] == 1 for i in ring_members)
    spot_tracks = [k for k in range(payload.n_tracks) if k not in rings]
    assert len(spot_tracks) == 1
    assert payload.track_span(spot_tracks[0])[:2] == (0, 5)


# -- tracking the file already carries ------------------------------------
# mlgidbase >= 0.1.11 saves its tracking table into the file; opening an
# entry shows it without running anything (built off the GUI thread).


_SAVED_DTYPE = np.dtype([
    ("id", "i4"), ("CIF", "S64"), ("h", "f4"), ("k", "f4"), ("l", "f4"),
    ("frame_list", __import__("h5py").vlen_dtype(np.int32)),
    ("peak_list", __import__("h5py").vlen_dtype(np.int32)),
])


def _save_tracking(path, tracks, *, meta=True, old_layout=False):
    """Write a tracking table the way mlgidbase 0.1.11 does. ``tracks`` is
    a list of ``(frames, ids)``; ``old_layout`` uses the pre-0.1.11 field
    names."""
    import h5py

    vlen = h5py.vlen_dtype(np.int32)
    if old_layout:
        dt = np.dtype([("id", "i4"), ("frames", vlen), ("peak_ids", vlen)])
    else:
        dt = _SAVED_DTYPE
    arr = np.empty(len(tracks), dtype=dt)
    for i, (frames, ids) in enumerate(tracks):
        arr["id"][i] = i
        if not old_layout:
            arr["CIF"][i] = b"unmatched"
            arr["h"][i] = arr["k"][i] = arr["l"][i] = np.nan
        arr["frame_list" if not old_layout else "frames"][i] = np.asarray(
            frames, dtype=np.int32)
        arr["peak_list" if not old_layout else "peak_ids"][i] = np.asarray(
            ids, dtype=np.int32)
    with h5py.File(path, "r+") as f:
        g = f.require_group(f"{ENTRY}/data/analysis/general")
        g.create_dataset("tracked_peaks", data=arr)
        if meta:
            p = f.require_group(f"{ENTRY}/process/peak_tracking")
            p.create_dataset("program", data=np.bytes_("mlgidbase"))
            p.create_dataset("version", data=np.bytes_("0.1.11"))
            p.create_dataset("date", data=np.bytes_("2026-10-23T12:18:55"))
            p.create_dataset("threshold", data=0.4)
            p.create_dataset("length", data=3)
            p.create_dataset("window_size", data=4)
    return path


_PERSISTENT = ([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])


def _wait_loaded(window, qtbot, timeout=20000):
    qtbot.waitUntil(lambda: window._scan_payload is not None, timeout=timeout)


def test_a_saved_tracking_table_shows_up_when_the_file_opens(
    main_window, synthetic_fitted_scan, qtbot,
):
    _save_tracking(synthetic_fitted_scan, [_PERSISTENT])
    _open(main_window, synthetic_fitted_scan)
    _wait_loaded(main_window, qtbot)
    panel = main_window.scan_tracking_panel
    payload = main_window._scan_payload
    assert payload.n_tracks == 1
    assert payload.track_span(0) == (0, 5, 6, 6)
    assert panel._model.rowCount() == 1
    assert main_window._scan_track_entry == ENTRY
    # The settings the table was made with are shown, so a re-run starts
    # from them, and the panel says where the tracks came from.
    assert (panel.threshold, panel.min_length, panel.window_size) == (0.4, 3, 4)
    assert not panel.lbl_saved.isHidden()
    assert "saved by mlgidbase 0.1.11 on 2026-10-23" in panel.lbl_saved.text()
    assert panel.btn_load_saved.isHidden()
    assert panel.btn_views.isEnabled()
    # Members are the exact (frame, id) pairs of the table.
    members = payload.track_members(0)
    assert [main_window._scan_member_ids[int(i)] for i in members] == [
        (f, 0) for f in range(6)
    ]


def test_no_saved_table_means_no_tracks_and_no_note(
    main_window, synthetic_fitted_scan, qtbot,
):
    _open(main_window, synthetic_fitted_scan)
    qtbot.wait(300)
    assert main_window._scan_payload is None
    panel = main_window.scan_tracking_panel
    assert panel.lbl_saved.isHidden() and panel.btn_load_saved.isHidden()


def test_a_saved_table_that_no_longer_matches_waits_for_a_click(
    main_window, synthetic_fitted_scan, qtbot,
):
    """Members naming fitted peaks that are gone mean the table was made
    for other fits: it is not loaded on its own, the panel says so and
    offers 'Load saved tracks', which loads it with those members left
    out."""
    _save_tracking(synthetic_fitted_scan, [(
        [0, 1, 2, 3, 4, 5, 2], [0, 0, 0, 0, 0, 0, 99],
    )])
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel
    qtbot.waitUntil(lambda: not panel.btn_load_saved.isHidden(), timeout=20000)
    assert main_window._scan_payload is None
    assert "does not match" in panel.lbl_saved.text()
    assert "1 of 7" in panel.lbl_saved.text()

    qtbot.mouseClick(panel.btn_load_saved, Qt.MouseButton.LeftButton)
    _wait_loaded(main_window, qtbot)
    assert main_window._scan_payload.n_tracks == 1
    assert main_window._scan_payload.track_span(0) == (0, 5, 6, 6)
    assert "1 member(s) no longer match" in panel.lbl_saved.text()
    assert panel.btn_load_saved.isHidden()


def test_a_table_from_an_older_mlgidbase_is_named_not_loaded(
    main_window, synthetic_fitted_scan, qtbot,
):
    _save_tracking(synthetic_fitted_scan, [_PERSISTENT], old_layout=True)
    _open(main_window, synthetic_fitted_scan)
    panel = main_window.scan_tracking_panel
    qtbot.waitUntil(lambda: not panel.lbl_saved.isHidden(), timeout=20000)
    assert main_window._scan_payload is None
    assert "older mlgidbase" in panel.lbl_saved.text()
    assert panel.btn_load_saved.isHidden()


def test_a_reload_of_the_same_entry_does_not_look_again(
    main_window, synthetic_fitted_scan, qtbot, monkeypatch,
):
    """Overlay reloads (peak edits, a pipeline run finishing) go through
    the same install as an entry switch; the file is only looked at once
    per entry shown."""
    _open(main_window, synthetic_fitted_scan)
    qtbot.wait(300)
    calls: list = []
    real = main_window._request_saved_tracking
    monkeypatch.setattr(
        main_window, "_request_saved_tracking",
        lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1],
    )
    main_window._load_entry_into_viewer(ENTRY, preserve_view=True)
    main_window._load_entry_into_viewer(ENTRY, preserve_view=True)
    assert calls == []


def test_a_fresh_run_result_is_not_overwritten_by_a_late_load(
    main_window, synthetic_fitted_scan, qtbot,
):
    _save_tracking(synthetic_fitted_scan, [_PERSISTENT])
    _open(main_window, synthetic_fitted_scan)
    _wait_loaded(main_window, qtbot)
    mine = main_window._scan_payload
    # A late/duplicate answer for the same entry changes nothing.
    main_window._on_saved_tracking_loaded(
        main_window._track_load_req_id, ENTRY, None, {"status": "stale",
        "message": "x", "saved": None},
    )
    assert main_window._scan_payload is mine
    assert main_window.scan_tracking_panel.btn_load_saved.isHidden()


def test_refitting_forgets_the_saved_tracking_and_looks_again(
    main_window, synthetic_fitted_scan, qtbot,
):
    _save_tracking(synthetic_fitted_scan, [_PERSISTENT])
    _open(main_window, synthetic_fitted_scan)
    _wait_loaded(main_window, qtbot)
    main_window._invalidate_scan_tracks()
    assert main_window._scan_payload is None
    assert main_window._track_autoload_key is None
    assert main_window.scan_tracking_panel.lbl_saved.isHidden()


# -- edits to the tracks are edits of the saved table too -----------------


def _saved_tracks(path):
    import h5py

    with h5py.File(path, "r") as f:
        t = f[f"{ENTRY}/data/analysis/general/tracked_peaks"][()]
        meta = f[f"{ENTRY}/process/peak_tracking"]
        edited = meta.attrs.get("edited_by")
    return [
        (int(r["id"]), list(map(int, r["frame_list"])), list(map(int, r["peak_list"])))
        for r in t
    ], edited


def test_deleting_a_track_rewrites_the_saved_table(
    main_window, synthetic_fitted_scan, qtbot,
):
    """The edit is an edit of the file: the other tracks stay, are
    renumbered 0..n-1 like upstream numbers them, and the session is
    dirty so Save writes it."""
    _save_tracking(synthetic_fitted_scan, [
        ([0, 1, 2, 3, 4, 5], [0] * 6),
        ([2], [1]),                       # the blip: a second, short track
    ])
    _open(main_window, synthetic_fitted_scan)
    _wait_loaded(main_window, qtbot)
    assert main_window._scan_payload.n_tracks == 2
    # Loading what the file carries must not make the session dirty...
    assert main_window.session.dirty is False
    panel = main_window.scan_tracking_panel
    panel._table.setFocus()
    panel._table.selectRow(0)
    qtbot.keyClick(panel._table, Qt.Key.Key_Delete)
    assert main_window._scan_payload.n_tracks == 1
    # ...but the edit does.
    assert main_window.session.dirty is True
    tracks, edited = _saved_tracks(main_window.session.temp_path)
    assert tracks == [(0, [2], [1])]
    assert str(edited).startswith("mlgidGUI")
    assert "saved in the file" in main_window.statusBar().currentMessage()
    # Reloading the entry from the file shows the edited result.
    main_window._invalidate_scan_tracks()
    main_window._request_saved_tracking(ENTRY)
    _wait_loaded(main_window, qtbot)
    assert main_window._scan_payload.n_tracks == 1


def test_a_track_cannot_be_deleted_while_a_step_runs(
    main_window, synthetic_fitted_scan, qtbot, monkeypatch,
):
    _save_tracking(synthetic_fitted_scan, [_PERSISTENT])
    _open(main_window, synthetic_fitted_scan)
    _wait_loaded(main_window, qtbot)
    monkeypatch.setattr(main_window, "_is_busy", lambda: True)
    main_window._on_scan_track_deleted(0)
    assert main_window._scan_payload.n_tracks == 1
    assert "Wait for the running step" in main_window.statusBar().currentMessage()
    tracks, _ = _saved_tracks(main_window.session.temp_path)
    assert len(tracks) == 1


def test_gap_fills_are_written_into_the_saved_table(
    main_window, synthetic_fitted_scan,
):
    file_model.delete_peak_row(synthetic_fitted_scan, ENTRY, 3, "fitted", 0)
    _open(main_window, synthetic_fitted_scan)
    _install_fake(main_window, _gap_payload())
    with main_window._detached_silx_tree():
        new_id = file_model.add_fitted_peak_row(
            main_window.session.temp_path, ENTRY, 3,
            radius=1.006, radius_width=0.05, angle=45.0, angle_width=5.0,
            amplitude=103.0, is_ring=False,
        )
        main_window._interp_fill_result = [{
            "track": 0, "frame": 3,
            "detected_id": 99, "fitted_id": int(new_id),
        }]
        # As in production, applied while the tree is still detached.
        main_window._apply_interp_fill_result()
    tracks, edited = _saved_tracks(main_window.session.temp_path)
    assert len(tracks) == 1
    frames, ids = tracks[0][1], tracks[0][2]
    assert (3, int(new_id)) in list(zip(frames, ids))      # the filled peak
    assert frames == sorted(frames)
    assert str(edited).startswith("mlgidGUI")


def test_a_track_run_leaves_the_shown_tracks_in_the_working_file(
    main_window, synthetic_fitted_scan, qtbot,
):
    """A (re)run of Track peaks replaces the table in the working file
    with the result as shown and leaves the session dirty, so Save writes
    it - not just whatever upstream saved during the run."""
    pytest.importorskip("mlgidbase")
    import h5py

    # An older table from "a previous session" that the run replaces.
    _save_tracking(synthetic_fitted_scan, [([0], [0]), ([1], [0]), ([2], [1])])
    _open(main_window, synthetic_fitted_scan)
    qtbot.waitUntil(lambda: main_window._scan_payload is not None, timeout=20000)
    main_window._invalidate_scan_tracks()
    main_window.session.dirty = False

    main_window._on_track_scan_requested(0.5, 3)
    qtbot.waitUntil(
        lambda: main_window._pipe_thread is None
        and not main_window._pipeline_queue
        and main_window._scan_payload is not None,
        timeout=60000,
    )
    assert main_window.session.dirty is True
    tracks, edited = _saved_tracks(main_window.session.temp_path)
    payload = main_window._scan_payload
    assert len(tracks) == payload.n_tracks == 1
    members = payload.track_members(0)
    assert tracks[0][1] == [int(f) for f in payload.frame_num[members]]
    assert tracks[0][2] == [int(i) for i in payload.peak_id[members]]
    assert str(edited).startswith("mlgidGUI")
    # Saving carries it to the user's file.
    main_window.session.save()
    with h5py.File(main_window.session.original_path, "r") as f:
        saved = f[f"{ENTRY}/data/analysis/general/tracked_peaks"][()]
    assert len(saved) == 1


def test_a_track_run_writes_the_gui_ring_tracks_as_rows_once(
    main_window, synthetic_fitted_scan, qtbot,
):
    """Upstream links the 0.1.8-style rings natively; the GUI replaces
    those with its own ring tracks. The file must end up with exactly the
    tracks on screen: one ring row, no duplicate from upstream."""
    pytest.importorskip("mlgidbase")
    import h5py

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        for frame in range(6):
            ds = f[f"{ENTRY}/data/analysis/frame{frame:05d}/fitted_peaks"]
            rows = ds[()]
            ring = np.zeros(1, dtype=rows.dtype)
            ring["id"] = int(rows["id"].max()) + 1
            ring["radius"], ring["radius_width"] = 2.5, 0.2
            ring["angle"], ring["angle_width"] = 45.0, np.inf
            ring["is_ring"] = True
            ring["amplitude"] = 30.0
            ring["q_xy"] = 2.5 * np.cos(np.deg2rad(45.0))
            ring["q_z"] = 2.5 * np.sin(np.deg2rad(45.0))
            parent = ds.parent
            del parent["fitted_peaks"]
            parent.create_dataset("fitted_peaks", data=np.concatenate([rows, ring]))

    _open(main_window, synthetic_fitted_scan)
    main_window._on_track_scan_requested(0.5, 3)
    qtbot.waitUntil(
        lambda: main_window._pipe_thread is None
        and not main_window._pipeline_queue
        and main_window._scan_payload is not None,
        timeout=60000,
    )
    payload = main_window._scan_payload
    tracks, _ = _saved_tracks(main_window.session.temp_path)
    assert len(tracks) == payload.n_tracks == 2
    assert len(main_window._scan_ring_tracks) == 1
