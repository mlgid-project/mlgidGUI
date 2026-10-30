"""Cross-frame peak tracking: track-scan runs, tracking progress dialogs, phase-track result install, ring tracks, interpolation fills, phase-views launch and the tracked-only filter.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from mlgidgui import file_model, peak_lists, phase_tracking
from mlgidgui.phase_views_window import PhaseViewsWindow
from mlgidgui.peak_link import link_enabled
from mlgidgui.pipeline import PipelineCommand
from mlgidgui.widgets import skin_progress
from mlgidgui.workers import TrackingLoadWorker

import logging

logger = logging.getLogger(__name__)


class TrackingMixin:
    # -- Phase tracking (mlgidBASE track_peaks) --

    def _on_track_scan_requested(self, threshold: float, length: int) -> None:
        """Enqueue an mlgidBASE ``track_peaks`` run for the active entry.

        Rides the ordinary pipeline queue (silx detach, busy gating,
        logging, entry snapshot all come for free); the
        ``TrackingPayload`` built from upstream's saved table comes back
        through ``_on_pipeline_finished`` -> ``_on_phase_track_result``.
        """
        if self.session is None:
            return
        if self._view_worker_blocks_pipeline():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        # No memory pre-flight here: upstream's tracking (mlgidbase
        # >= 0.1.11) streams the IoU in row blocks, so it is safe at any
        # scan size.
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        self._show_tracking_progress()
        self._enqueue_pipeline(
            self.session.temp_path,
            PipelineCommand("track_peaks", {
                "entry": entry,
                "threshold": float(threshold),
                "length": int(length),
                "window_size": (
                    self.scan_tracking_panel.window_size or None
                ),
            }),
        )

    def _show_tracking_progress(
        self, label: str = "Tracking peaks across the scan…",
        manual: bool = False,
    ) -> None:
        """Show a centred, always-visible loading dialog for the run.

        track_peaks is one opaque mlgidBASE call with no clean per-frame
        progress (its cost is the file open + IoU/graph, not the reads),
        so the default mode is an indeterminate busy marquee — honest
        motion for the whole run. (An earlier fake timer crept to 95%
        within seconds and then sat there for the rest of the run,
        which read as a frozen bar on long scans.)
        ``setMinimumDuration(0)`` + explicit ``show`` defeat
        QProgressDialog's default 4 s show delay.

        ``manual=True`` keeps the determinate 0..100 bar: the caller
        drives it with REAL progress values (the Interpolate-track
        chain feeds it the workers' per-frame ``frameProgress`` ticks
        via ``_on_pipeline_frame_progress``, so the dialog and the
        Pipeline panel's bar agree)."""
        self._close_tracking_progress()
        dlg = skin_progress(QProgressDialog(self))
        dlg.setWindowTitle("Peak tracking")
        dlg.setLabelText(label)
        dlg.setCancelButton(None)          # no clean way to interrupt mlgidBASE
        dlg.setRange(0, 100)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        if not manual:
            dlg.setRange(0, 0)             # 0..0 == busy marquee
        dlg.show()
        self._track_progress_dialog = dlg

    def _close_tracking_progress(self) -> None:
        # Detach the attribute FIRST: setValue on a window-modal
        # QProgressDialog runs processEvents(), so re-entrant handlers
        # (frameProgress, opFinished) must already see "no dialog".
        dlg = self._track_progress_dialog
        self._track_progress_dialog = None
        if dlg is not None:
            if dlg.maximum() > 0:
                dlg.setValue(100)
            dlg.close()
            dlg.deleteLater()

    def _on_phase_track_result(self, payload: object, command) -> None:
        """Install a finished ``track_peaks`` payload (panel + views)."""
        if self.session is None or not isinstance(
            payload, phase_tracking.TrackingPayload
        ):
            return
        entry = command.kwargs.get("entry") or payload.entry
        if self.entry_combo.currentText() != entry:
            self.statusBar().showMessage(
                "Peak tracking finished for a no-longer-active entry; "
                "results discarded.",
                4000,
            )
            return
        state = phase_tracking.build_tracking_state(
            self.session.temp_path, entry, payload,
        )
        self._apply_tracking_state(state, entry)
        # What is on screen is what the file keeps. Upstream saved its own
        # table into the working file during the run; rewrite it from the
        # result as shown (the GUI's ring tracks replace upstream's
        # ring components) so Save writes this, not the raw table. The
        # tree is still detached here (the queue has not drained).
        self._write_tracking_to_file(detach=False)

    def _apply_tracking_state(self, state, entry: str) -> None:
        """Install a ``phase_tracking.TrackingState`` into the panel, the
        viewer and the views window. GUI thread; no file reads of its
        own beyond the views window's context."""
        payload = state.payload
        ids = state.ids
        tables = state.tables
        self._scan_payload = payload
        self._scan_member_ids = ids
        self._scan_fitted_tables = tables
        self._scan_track_entry = entry
        # Which tracks are rings (any ring member; rings and spots never
        # mix in a component). The q-map draws these as dashed arcs.
        self._scan_ring_tracks = state.ring_tracks
        # Map each track to the matched crystal phases its tracked peaks
        # belong to (for the q-map phase-identity overlay). Empty when
        # Matching hasn't run.
        matched_tables = state.matched_tables
        self._scan_matched_tables = matched_tables
        self._scan_track_phases = state.track_phases
        self._scan_member_phases = state.member_phases
        # Pre-assign overlay colours for every matched identity (sorted,
        # so the assignment is deterministic no matter which frame is
        # rendered first).
        all_keys = sorted({
            s.color_key
            for structs in matched_tables.values() if structs
            for s in structs
        })
        self.viewer.seed_matched_colors(all_keys)
        self.scan_tracking_panel.set_payload(
            payload, ids, self._scan_track_phases
        )
        # Re-apply the show-only-tracked overlay filter against the
        # fresh membership (or lift it if the box is unticked).
        self._push_tracked_filter()
        if self._phase_views_window is not None:
            self._phase_views_window.set_context(
                self.session.temp_path, entry, self.viewer.n_frames
            )
            # Colours BEFORE the phase mapping (same order as window
            # open): the freshly seeded palette must be in place when
            # set_track_phases builds the colour map.
            self._push_phase_color_overrides()
            self._phase_views_window.set_payload(payload)
            self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
            self._phase_views_window.set_track_phases(self._scan_track_phases)
            self._phase_views_window.set_member_phases(self._scan_member_phases)

    def _compute_ring_tracks(self, payload, ids, tables) -> set:
        """Set of track indices whose members are rings (checked via the
        member ids against the fitted tables' ``is_ring``)."""
        return phase_tracking.ring_track_set(payload, ids, tables)

    def _read_fitted_tables(self, payload, entry: str) -> dict:
        """One-handle read of every payload frame's fitted table (kept
        on ``_scan_fitted_tables`` for the only-tracked display options:
        gap interpolation anchors, GUI-side ring tracking)."""
        return phase_tracking.read_fitted_tables(
            self.session.temp_path, entry, payload,
        )

    def _read_matched_tables(self, payload, entry: str, fitted_tables: dict) -> dict:
        """One-handle read of every payload frame's matched structures;
        empty on any error or when Matching hasn't run."""
        return phase_tracking.read_matched_tables(
            self.session.temp_path, entry, payload, fitted_tables,
        )

    # -- Tracking saved in the file (loaded without running anything) --

    def _ensure_track_load_worker(self) -> None:
        """Spawn the persistent TrackingLoadWorker + thread on first use."""
        if self._track_load_worker is not None:
            return
        self._track_load_thread = QThread(self)
        self._track_load_worker = TrackingLoadWorker()
        self._track_load_worker.moveToThread(self._track_load_thread)
        self._trackLoadRequest.connect(
            self._track_load_worker.load, Qt.ConnectionType.QueuedConnection,
        )
        self._track_load_worker.loaded.connect(
            self._on_saved_tracking_loaded, Qt.ConnectionType.QueuedConnection,
        )
        self._track_load_thread.start()

    def _maybe_autoload_saved_tracking(self, entry: str) -> None:
        """Look for tracking the entry already carries, once per (file,
        entry) shown. Runs from the entry install, so it covers opening a
        file and switching entries; later overlay reloads of the same
        entry (peak edits, a pipeline run finishing) do not repeat it.
        Skipped for a single-frame entry (nothing to track), while a
        pipeline step is running (it may be writing the file) and when
        this entry's tracks are already on screen."""
        session = self.session
        if session is None or getattr(session, "kind", None) != "nexus":
            return
        if self.viewer.n_frames <= 1 or self._pipe_thread is not None:
            return
        if self._scan_track_entry == entry:
            return
        key = (id(session), entry)
        if self._track_autoload_key == key:
            return
        self._track_autoload_key = key
        self._request_saved_tracking(entry)

    def _request_saved_tracking(self, entry: str | None = None, *, force: bool = False) -> None:
        """Ask the worker to build the Scan-tracking state from the table
        saved in the active entry. ``force`` also loads a table whose
        members no longer all match a fitted peak."""
        if self.session is None:
            return
        entry = entry or self.entry_combo.currentText()
        if not entry:
            return
        self._track_load_req_id += 1
        self._ensure_track_load_worker()
        self._trackLoadRequest.emit(
            str(self.session.temp_path), entry, self._track_load_req_id,
            bool(force),
        )

    def _on_saved_tracking_loaded(
        self, request_id: int, entry: str, state: object, info: object,
    ) -> None:
        """Show what the worker found: the saved tracks, or why not."""
        if request_id != self._track_load_req_id:
            return
        if self.session is None or self.entry_combo.currentText() != entry:
            return
        if self._scan_payload is not None and self._scan_track_entry == entry:
            return  # a run of its own got here first
        panel = self.scan_tracking_panel
        status = (info or {}).get("status", "none")
        saved = (info or {}).get("saved")
        if state is not None:
            self._apply_tracking_state(state, entry)
            panel.apply_settings(
                saved.threshold, saved.length, saved.window_size,
            )
            made = " ".join(
                part for part in (
                    f"saved by {saved.program} {saved.version}".strip(),
                    f"on {saved.date[:10]}" if saved.date else "",
                ) if part
            )
            note = f"Loaded from the file ({made})." if made else (
                "Loaded from the file."
            )
            skipped = int(info.get("unresolved", 0))
            if skipped:
                note += (
                    f" {skipped} member(s) no longer match a fitted peak and "
                    "were skipped."
                )
            panel.set_saved_note(note)
            self.statusBar().showMessage(
                f"Loaded {state.payload.n_tracks} saved track(s) for {entry}.",
                4000,
            )
        elif status == "stale":
            panel.set_saved_note(
                "The file carries a tracking table, but it does not match the "
                f"current fits: {info.get('message', '')} Re-run tracking, or "
                "load it anyway.",
                can_load=True,
            )
        elif status == "old_layout":
            panel.set_saved_note(
                "The file carries a tracking table written by an older "
                "mlgidbase (before 0.1.11), which cannot be read. Run "
                "tracking again to replace it."
            )
        elif status == "error":
            logger.warning(
                "could not read the saved tracking of %s: %s",
                entry, (info or {}).get("message"),
            )

    def _invalidate_scan_tracks(self) -> None:
        """Drop phase-tracking results and clear panel + views.

        Called whenever the fitted tables the tracks reference may have
        changed identity: entry switch, ``run_fitting`` finish,
        Tools > Clear fitted / Reset.
        """
        self._scan_payload = None
        # A look at the file's saved table is no longer current either;
        # drop in-flight loads and let the next entry install look again.
        self._track_load_req_id += 1
        self._track_autoload_key = None
        self._scan_member_ids = None
        self._scan_fitted_tables = {}
        self._scan_track_phases = {}
        self._scan_member_phases = {}
        self._scan_ring_tracks = set()
        self._scan_track_entry = None
        self._scan_matched_tables = {}
        # A gap-fill run in flight references the payload being dropped
        # — its result must not be applied against fresh tracks.
        self._interp_fill_result = None
        self._interp_chain = None
        # panel.clear() unticks "Show only tracked peaks", whose toggled
        # emit routes back through _push_tracked_filter and lifts the
        # viewer filter; the explicit lift below is belt-and-braces for
        # the box-already-unticked path.
        self.scan_tracking_panel.clear()
        self.viewer.set_fitted_visible_only(None)
        if self._phase_views_window is not None:
            self._phase_views_window.clear()

    def _sync_scan_tracking_visibility(self, *, from_scratch: bool = False) -> None:
        """Take Scan tracking away for single-frame NeXus data.

        Tracking follows peaks across frames, so one frame has nothing to
        track: the dock and the workflow strip's Track stage go, and the
        dock's View-menu action is disabled so it cannot be brought back.
        When the rule stops holding, only a dock the rule hid comes back;
        one the user closed by hand stays closed.

        ``from_scratch`` is the session-mode path, which decides dock
        visibility afresh: the dock is shown unless the rule hides it.
        Raw sessions hide the dock in ``_apply_session_mode``.
        """
        dock = getattr(self, "_scan_tracking_dock", None)
        if dock is None:
            return
        session = self.session
        single = (session is not None and session.kind == "nexus"
                  and self.viewer.n_frames <= 1)
        rail = getattr(self, "workflow_rail", None)
        if rail is not None:
            rail.set_stage_visible("track", not single)
        dock.toggleViewAction().setEnabled(not single)
        name = "_scan_tracking_dock"
        # Docks folded away for the Structure tab come back from this set
        # when the tab is left.
        folded = self._structure_folded_docks
        if single:
            if from_scratch or not dock.isHidden() or name in folded:
                self._tracking_hidden_by_rule = True
            folded.discard(name)
            dock.hide()
            return
        if from_scratch or self._tracking_hidden_by_rule:
            self._tracking_hidden_by_rule = False
            structure = getattr(self, "structure_panel", None)
            if structure is not None and self.tabs.currentWidget() is structure:
                folded.add(name)
            else:
                dock.show()

    def _on_show_phase_views(self) -> None:
        """Open (lazily building) and raise the phase-tracking views."""
        if self._phase_views_window is None:
            self._phase_views_window = PhaseViewsWindow(
                self,
                busy_probe=lambda: self._pipe_thread is not None,
            )
            self._phase_views_window.frameJumpRequested.connect(
                self._on_phase_view_frame_jump
            )
            self._phase_views_window.saveFiguresRequested.connect(
                self._on_save_official_figures
            )
        if self.session is not None:
            entry = self.entry_combo.currentText()
            if entry:
                self._phase_views_window.set_context(
                    self.session.temp_path, entry, self.viewer.n_frames
                )
        # Custom structure colours land BEFORE the phase mapping so the
        # first colour build already honours them (no palette flash).
        self._push_phase_color_overrides()
        self._phase_views_window.set_payload(self._scan_payload)
        self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
        self._phase_views_window.set_track_phases(self._scan_track_phases)
        self._phase_views_window.set_member_phases(self._scan_member_phases)
        self._phase_views_window.show()
        self._phase_views_window.raise_()
        self._phase_views_window.activateWindow()

    def _view_worker_blocks_pipeline(self) -> bool:
        """True (+ status message) while a phase-views worker holds a
        read handle on the file.

        pygid opens the NeXus file in r+ mode even for reads
        (nexus_reader.get_dataset), and HDF5 refuses an r+ open while
        another same-process handle has the file open read-only — so a
        pipeline run starting during a waterfall / mean-image
        computation dies with "Unable to synchronously open file (file
        is already open for read-only)". The silx detach dance cannot
        release that worker mid-loop; refusing to start is the honest
        fix (the views window refuses in the opposite direction via
        its busy_probe).
        """
        w = self._phase_views_window
        if w is not None and w.worker_active:
            self.statusBar().showMessage(
                "Wait for the phase-views computation (waterfall / mean "
                "image) to finish before running pipeline commands — it "
                "holds a read handle on the file.",
                6000,
            )
            return True
        return False

    def _on_only_tracked_toggled(self, _checked: bool) -> None:
        self._push_tracked_filter()

    def _push_tracked_filter(self) -> None:
        """Apply (or lift) the show-only-tracked overlay filter.

        Builds ``{frame: {fitted_id, ...}}`` from the surviving tracks'
        members via the reconstructed ids (members whose id could not
        be matched back are unavoidably hidden — best-effort, flagged
        in the checkbox tooltip) and pushes it to
        ``viewer.set_fitted_visible_only``. Anything else lifts it.

        Rings are tracked too: ``_on_phase_track_result`` appended the
        GUI-side ring tracks (``phase_tracking.track_rings``) to
        ``payload.components``, so tracked rings are whitelisted here
        like any spot track and untracked rings are hidden — no
        special-casing needed at this layer. Gap frames filled by
        "Interpolate track" join the whitelist the same way: their new
        fitted rows are appended as track members.
        """
        panel = self.scan_tracking_panel
        payload = self._scan_payload
        ids = self._scan_member_ids
        if (
            not panel.only_tracked_checked
            or payload is None
            or ids is None
        ):
            self.viewer.set_fitted_visible_only(None)
        else:
            allowed: dict[int, set[int]] = {}
            for comp in payload.components:
                for i in comp:
                    tagged = ids[int(i)]
                    if tagged is not None:
                        allowed.setdefault(int(tagged[0]), set()).add(
                            int(tagged[1])
                        )
            self.viewer.set_fitted_visible_only(allowed)
        # The "Unmatched fitted peaks" dock row exists only while the
        # frame has unmatched rows the filter lets through — rebuild the
        # matched panel so the row appears/disappears with the filter.
        self._refresh_matched_panel(
            self.viewer.current_frame,
            self.viewer.matched_structures(self.viewer.current_frame),
        )

    def _panel_fit_params(self) -> dict:
        """The Pipeline panel's fit config as ``fit_one_peak`` kwargs.

        Injected boxes (gap fills, predicted reflections) are fitted
        with the SAME settings the next ``run_fitting`` would use.
        Empty dict when the widgets are unavailable (backend-less
        stub panel) — the executor falls back to pygidfit defaults.
        """
        panel = self.pipeline_panel
        try:
            return {
                "crit_angle": float(panel.fit_crit_angle.value()),
                "clustering_distance_peaks": float(
                    panel.fit_dist_peaks.value()
                ),
                "clustering_distance_rings": float(
                    panel.fit_dist_rings.value()
                ),
                "clustering_extend": int(panel.fit_cluster_extend.value()),
                "theta_fixed": bool(panel.fit_theta_fixed.isChecked()),
            }
        except Exception:
            logger.debug(
                "suppressed exception reading the panel fit params",
                exc_info=True,
            )
            return {}

    def _on_interpolate_tracks_requested(self) -> None:
        """Fill each track's gap frames with REAL peaks (spots + rings).

        Plans the injections (``phase_tracking.plan_gap_fills``), then
        enqueues an ``interpolate_tracks`` command (inject a detected
        box/ring per gap at the interpolated position, 2D-fit it,
        persist the fitted row) followed by ``run_matching`` command(s)
        scoped to the affected frames — segments and rings separately,
        and CIF-restricted to the structures the filled tracks were
        tracked in (see ``_build_interp_match_commands``). The fill
        records come back through ``_on_pipeline_finished`` and are
        applied as new track members in ``_apply_interp_fill_result``
        once the LAST re-match lands. The loading dialog is driven by
        the workers' real per-frame progress across the whole chain.
        """
        if self.session is None or self._view_worker_blocks_pipeline():
            return
        payload = self._scan_payload
        ids = self._scan_member_ids
        entry = self.entry_combo.currentText()
        if (
            payload is None
            or ids is None
            or not entry
            or entry != self._scan_track_entry
        ):
            return
        plan = phase_tracking.plan_gap_fills(
            payload, ids, self._scan_fitted_tables, self._scan_ring_tracks
        )
        if not plan:
            self.statusBar().showMessage(
                "Interpolate track: every track already has a fitted "
                "peak on each frame of its span — nothing to fill.",
                6000,
            )
            return
        match_kwargs = self.pipeline_panel.matching_kwargs()
        if match_kwargs is None:
            QMessageBox.warning(
                self, "Interpolate track",
                "Gap filling ends with a re-match of the affected "
                "frames, but no CIF source is configured. Set the CIF "
                "file/folder (or pickle) in the Pipeline panel's "
                "Matching section first.",
            )
            return
        fit_params = self._panel_fit_params()
        n_boxes = sum(len(v) for v in plan.values())
        self.pipeline_panel.append_log(
            f"Interpolate track: filling {n_boxes} gap box(es) across "
            f"{len(plan)} frame(s) of {entry}…"
        )
        fill_cmd = PipelineCommand("interpolate_tracks", {
            "entry": entry,
            "plan": plan,
            "fit_params": fit_params,
            # Same pairing as the expected-pattern injection.
            "link_ids": link_enabled(),
        })
        match_cmds = self._build_interp_match_commands(
            plan, entry, match_kwargs
        )
        self._interp_fill_result = None
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        # Chain bookkeeping: one bar tick per gap frame (fill) + per
        # matched frame (each matching command), driven by the REAL
        # frameProgress signals in _on_pipeline_frame_progress.
        commands = [fill_cmd] + match_cmds
        ticks = {id(fill_cmd): len(plan)}
        for cmd in match_cmds:
            ticks[id(cmd)] = len(cmd.kwargs.get("frame_num") or []) or 1
        self._interp_chain = {
            "commands": commands,
            "ticks": ticks,
            "total": max(1, sum(ticks.values())),
            "base": 0,
        }
        self._show_tracking_progress(
            "Interpolate track: fitting injected boxes…", manual=True,
        )
        for cmd in commands:
            self._enqueue_pipeline(self.session.temp_path, cmd)

    def _build_interp_match_commands(
        self, plan: dict, entry: str, match_kwargs: dict
    ) -> list:
        """Build the re-match command(s) for the gap-fill chain.

        Segments and rings are matched SEPARATELY (mlgidmatch matches
        one ``peaks_type`` per run), each command pinned to the frames
        that gained fills of that type.

        The CIF source is ALWAYS restricted — no new structures can
        appear from a gap fill. Per frame + type the set is the union
        of (a) the structures the filled tracks were tracked in
        (``_scan_track_phases`` — a track matched to one structure
        re-matches against only that CIF), (b) for a never-matched
        filled track, every structure identified anywhere in this
        scan for that type (the peak may belong to one of them; the
        full panel folder is never used), and (c) the CIFs already
        matched on the frame for that type — matching REWRITES the
        frame's ``matched_<type>_*`` datasets wholesale, so dropping
        an already-identified CIF from the set would erase its
        existing solutions there. An empty set (nothing identified in
        the scan) or an unsubsettable source (pickle / missing .cif
        files) SKIPS that matching pass with a log line — the filled
        peaks stay unmatched rather than brute-forcing every loaded
        structure. Frames sharing the same (type, CIF-set) collapse
        into one command with a ``frame_num`` list.
        """
        matched_tables = self._read_matched_tables(
            self._scan_payload, entry, self._scan_fitted_tables
        )
        phases = self._scan_track_phases
        # Everything identified anywhere in this scan, split by type —
        # the fallback set for filled tracks that were never matched.
        scan_cifs = {"matched_segments": set(), "matched_rings": set()}
        for structs in matched_tables.values():
            for st in structs:
                for prefix in scan_cifs:
                    if st.solution_field.startswith(prefix):
                        scan_cifs[prefix].add(str(st.cif))
        # The raw .cif source is preferred for restriction: after a
        # panel "Parse CIFs" click, matching_kwargs() carries the
        # cached CifPattern OBJECT, which cannot be subset — the raw
        # text can.
        raw_source = self.pipeline_panel.cif_source_text()
        if not raw_source and isinstance(match_kwargs.get("cif_prepr"), str):
            raw_source = match_kwargs["cif_prepr"]

        groups: dict = {}   # (peaks_type, frozenset) -> [frames]
        skipped: list = []
        for frame, specs in plan.items():
            for ring_flag, peaks_type, prefix in (
                (False, "segments", "matched_segments"),
                (True, "rings", "matched_rings"),
            ):
                type_specs = [
                    s for s in specs
                    if bool(s.get("is_ring", False)) == ring_flag
                ]
                if not type_specs:
                    continue
                cifs: set = set()
                for s in type_specs:
                    track_cifs = phases.get(int(s["track"]))
                    if track_cifs:
                        cifs.update(str(c) for c in track_cifs)
                    else:
                        cifs.update(scan_cifs[prefix])
                for st in matched_tables.get(int(frame), []):
                    if st.solution_field.startswith(prefix):
                        cifs.add(str(st.cif))
                if not cifs:
                    skipped.append((peaks_type, int(frame), "no structures"))
                    continue
                groups.setdefault(
                    (peaks_type, frozenset(cifs)), []
                ).append(int(frame))
        cmds: list = []
        for (peaks_type, cif_key), frames in sorted(
            groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))
        ):
            restricted = self._restrict_cif_source(raw_source, set(cif_key))
            if restricted is None:
                for f in frames:
                    skipped.append((peaks_type, f, "source not subsettable"))
                continue
            cmds.append(PipelineCommand("run_matching", {
                **match_kwargs,
                "entry": entry,
                "frame_num": sorted(set(frames)),
                "peaks_type": peaks_type,
                "cif_prepr": restricted,
            }))
        if skipped:
            by_reason: dict = {}
            for peaks_type, f, reason in skipped:
                by_reason.setdefault(reason, set()).add(f)
            for reason, frames in sorted(by_reason.items()):
                self.pipeline_panel.append_log(
                    "Interpolate track: skipping the re-match on frame(s) "
                    f"{sorted(frames)} ({reason}) — filled peaks there "
                    "stay unmatched; no new structures are introduced. "
                    "Use a .cif folder/file source (not a pickle) and "
                    "keep the identified structures' files in place to "
                    "enable restricted re-matching."
                )
        return cmds

    def _restrict_cif_source(self, base, cifs: set):
        """Semicolon-joined ``.cif`` paths covering exactly ``cifs``.

        ``base`` is the panel's raw ``cif_prepr`` string (a folder, a
        single ``.cif``, or a ``;``-separated list — the forms
        ``pipeline._build_cif_pattern_from_raw`` accepts). Returns
        ``None`` when the source cannot be subset: a pickle path, a
        pre-parsed ``CifPattern`` object, or any tracked CIF whose file
        is missing from the source (restricting would silently drop
        that structure — the caller falls back to the full source).
        CIF names are matched case-insensitively against file stems
        (``MatchedStructure.cif`` is the extensionless basename).
        """
        import os

        if not cifs or not isinstance(base, str) or not base.strip():
            return None
        paths = [p.strip() for p in base.split(";") if p.strip()]
        if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
            return None
        wanted = {str(c).lower() for c in cifs}
        try:
            if len(paths) == 1 and os.path.isdir(paths[0]):
                folder = paths[0]
                hits = [
                    os.path.join(folder, f)
                    for f in sorted(os.listdir(folder))
                    if f.lower().endswith(".cif")
                    and os.path.splitext(f)[0].lower() in wanted
                ]
            else:
                hits = [
                    p for p in paths
                    if p.lower().endswith(".cif")
                    and os.path.splitext(os.path.basename(p))[0].lower()
                    in wanted
                ]
        except Exception:
            logger.debug(
                "suppressed exception restricting the CIF source",
                exc_info=True,
            )
            return None
        found = {
            os.path.splitext(os.path.basename(p))[0].lower() for p in hits
        }
        if not hits or found != wanted:
            return None
        return ";".join(hits)

    def _apply_interp_fill_result(self) -> None:
        """Append the gap-fill peaks as members of their origin tracks.

        Runs after the chained re-match finishes. Each fill record's
        fitted row is re-read from the file (fresh table → exact
        coordinates), appended to the payload's member arrays, and its
        index added to ``components[track]`` — so the track's span/count
        update everywhere (panel table, views, only-tracked whitelist)
        without a re-run of ``track_peaks``. Member ids and the q-map
        phase mapping are then rebuilt against the fresh tables.
        """
        result = self._interp_fill_result
        self._interp_fill_result = None
        payload = self._scan_payload
        entry = self._scan_track_entry
        if not result or payload is None or entry is None or self.session is None:
            return
        # Fresh fitted tables for the affected frames (they just gained
        # rows; ids beyond the old table are unknown to the cache).
        affected = sorted({int(rec["frame"]) for rec in result})
        try:
            import h5py

            with h5py.File(self.session.temp_path, "r") as f:
                for frame in affected:
                    self._scan_fitted_tables[frame] = file_model.read_peaks(
                        f, entry, frame
                    )["fitted"]
        except Exception:
            logger.debug(
                "suppressed exception re-reading fitted tables after "
                "interpolate_tracks", exc_info=True,
            )
            return
        appended = 0
        for rec in result:
            frame = int(rec["frame"])
            fid = int(rec["fitted_id"])
            track = int(rec["track"])
            table = self._scan_fitted_tables.get(frame)
            if table is None or not (0 <= track < len(payload.components)):
                continue
            hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == fid)
            if hits.size != 1:
                continue
            j = int(hits[0])
            new_index = payload.n_members
            payload.q_xy = np.append(payload.q_xy, float(table.q_xy[j]))
            payload.q_z = np.append(payload.q_z, float(table.q_z[j]))
            payload.frame_num = np.append(payload.frame_num, frame)
            payload.amplitude = np.append(
                payload.amplitude, float(table.amplitude[j])
            )
            if payload.peak_id is not None:
                payload.peak_id = np.append(payload.peak_id, fid)
            comp = list(payload.components[track])
            comp.append(new_index)
            payload.components[track] = comp
            appended += 1
        if not appended:
            return
        # Rebuild the derived state against the fresh membership.
        self._scan_member_ids = phase_tracking.member_ids(
            payload, self._scan_fitted_tables
        )
        matched_tables = self._read_matched_tables(
            payload, entry, self._scan_fitted_tables
        )
        self._scan_matched_tables = matched_tables
        self._scan_track_phases = phase_tracking.match_tracks_to_structures(
            payload, self._scan_member_ids, matched_tables
        )
        self._scan_member_phases = phase_tracking.member_matched_phases(
            payload, self._scan_member_ids, matched_tables
        )
        self.scan_tracking_panel.set_payload(
            payload, self._scan_member_ids, self._scan_track_phases
        )
        self._push_tracked_filter()
        if self._phase_views_window is not None:
            self._phase_views_window.set_payload(payload)
            self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
            self._phase_views_window.set_track_phases(self._scan_track_phases)
            self._phase_views_window.set_member_phases(self._scan_member_phases)
        # The filled peaks are now members of their tracks; the saved
        # table follows (silx is still detached on this path).
        self._write_tracking_to_file(detach=False)
        self.statusBar().showMessage(
            f"Interpolate track: filled {appended} gap peak(s) "
            f"(fitted + matched) across {len(affected)} frame(s).",
            8000,
        )

    def _write_tracking_to_file(self, *, detach: bool = True) -> bool:
        """Write the tracks on screen back into the entry's saved
        ``tracked_peaks`` table, so an edit made here is also an edit of
        the file (the file is saved with the session as usual).

        ``detach``: take the silx tree away around the ``r+`` write. The
        gap-fill path calls with ``detach=False`` because the pipeline
        completion that applies it still has the tree detached (a nested
        detach would reattach it too early). Marks the session dirty.
        Returns False, with a status-bar message, when the write failed;
        the tracks on screen are unaffected either way.
        """
        payload = self._scan_payload
        entry = self._scan_track_entry
        if self.session is None or payload is None or not entry:
            return False
        try:
            if detach:
                with self._detached_silx_tree():
                    n = phase_tracking.write_saved_tracking(
                        self.session.temp_path, entry, payload,
                        self._scan_member_ids or [],
                        self._scan_matched_tables or {},
                    )
            else:
                n = phase_tracking.write_saved_tracking(
                    self.session.temp_path, entry, payload,
                    self._scan_member_ids or [],
                    self._scan_matched_tables or {},
                )
        except Exception as exc:
            logger.warning("could not update the saved tracking", exc_info=True)
            self.statusBar().showMessage(
                f"The tracks changed on screen but the table in the file "
                f"could not be updated: {exc}", 8000,
            )
            return False
        self.session.mark_dirty()
        self._update_title()
        logger.info("saved tracking of %s updated: %d track(s)", entry, n)
        return True

    def _on_scan_track_deleted(self, track: int) -> None:
        """Remove one track from the tracking results (Delete key).

        The track disappears from the table, the views, the only-tracked
        whitelist and the phase colouring, and from the ``tracked_peaks``
        table saved in the file (rewritten from the tracks on screen;
        the other tracks are renumbered). No fitted peaks are deleted.
        Track-indexed
        derived state (ring set, phase map) is remapped since indices
        above the removed track shift down by one; member-indexed
        state (member_ids, fitted tables) is untouched.
        """
        payload = self._scan_payload
        track = int(track)
        if payload is None or not (0 <= track < payload.n_tracks):
            return
        if self._is_busy():
            # The table in the file is rewritten with the edit, which
            # needs the file to itself.
            self.statusBar().showMessage(
                "Wait for the running step to finish, then delete the "
                "track.", 5000,
            )
            return
        del payload.components[track]
        self._scan_ring_tracks = {
            (k - 1 if k > track else k)
            for k in self._scan_ring_tracks if k != track
        }
        self._scan_track_phases = {
            (k - 1 if k > track else k): v
            for k, v in self._scan_track_phases.items() if k != track
        }
        self.scan_tracking_panel.set_payload(
            payload, self._scan_member_ids, self._scan_track_phases
        )
        self._push_tracked_filter()
        if self._phase_views_window is not None:
            self._phase_views_window.set_payload(payload)
            self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
            self._phase_views_window.set_track_phases(self._scan_track_phases)
        written = self._write_tracking_to_file()
        self.statusBar().showMessage(
            f"Track {track} removed from the tracking results"
            + (
                " and from the table saved in the file" if written else ""
            )
            + " (no fitted peaks were deleted).",
            6000,
        )

    def _on_phase_view_frame_jump(self, frame: int) -> None:
        if self.viewer.n_frames <= 0:
            return
        self.viewer.set_frame(
            max(0, min(int(frame), self.viewer.n_frames - 1))
        )

    def _on_save_official_figures(self, path: str, axis: str) -> None:
        """Save upstream's own tracking figure (``plot_tracked_peaks``).

        Reruns ``track_peaks`` with the payload's settings first, so the
        saved table the figure reads is current (the GUI can have
        deleted tracks or refitted since the last run), then draws it.
        One figure with two panels (q_xy vs q_z, and ``axis`` against
        frame number) is written to exactly ``path``. Upstream plots
        radius, angle, q_z or q_xy; an amplitude axis falls back to
        radius, with a note in the status bar.

        GUI thread on purpose: the run is quick (bounded-memory
        tracking + one Agg savefig) and matplotlib's pyplot state
        machine is not worker-thread-safe under a Qt backend. The silx
        tree is detached around mlgidBASE's file open, same as pipeline
        runs.
        """
        if self._is_busy():
            return
        payload = self._scan_payload
        entry = self._scan_track_entry
        if payload is None or not entry:
            return
        plot_axis = axis if axis in ("radius", "angle", "q_z", "q_xy") else "radius"
        try:
            with self._detached_silx_tree():
                from mlgidbase import mlgidBASE  # noqa: N814

                analysis = mlgidBASE(filename=str(self.session.temp_path))
                analysis.track_peaks(
                    entry=entry,
                    threshold=payload.threshold,
                    length=payload.length,
                    window_size=payload.window_size,
                )
                analysis.plot_tracked_peaks(
                    entry=entry,
                    axis=plot_axis,
                    plot_result=False,
                    save_fig=True,
                    path_to_save_fig=path,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save figures", str(exc))
            return
        self.session.mark_dirty()
        note = (
            "" if plot_axis == axis
            else f" (upstream has no {axis} axis; plotted radius)"
        )
        self.statusBar().showMessage(
            f"Saved the mlgidBASE tracking figure to {path}{note}", 6000
        )

    def _forward_selection_to_scan_tracking(self, sel) -> None:
        """Mirror a FITTED-peak selection onto the Scan-tracking table.

        The reverse of ``_on_track_row_selected``: clicking ANY member
        peak of a track (any frame, image or Peaks dock) highlights the
        track's row — whose columns carry the first/last/frames span —
        and echoes "track N: frames x-y" to the status bar. Non-fitted
        or trackless selections just clear the row highlight. The panel
        applies the highlight under its ``_applying_external`` guard,
        so this never bounces the viewer to another frame.
        """
        if self._scan_payload is None or self._scan_track_entry is None:
            return
        if self.entry_combo.currentText() != self._scan_track_entry:
            return
        if sel is None or getattr(sel, "kind", None) != "fitted":
            self.scan_tracking_panel.clear_external_selection()
            return
        track = self.scan_tracking_panel.set_external_peak(
            int(sel.frame), int(sel.peak_id)
        )
        if track is not None:
            first, last, n_frames, _members = self._scan_payload.track_span(
                track
            )
            span = (
                f"frame {first}" if first == last
                else f"frames {first}-{last}"
            )
            self.statusBar().showMessage(
                f"Fitted peak {int(sel.peak_id)} (frame {int(sel.frame)}) "
                f"belongs to track {track}: {span}, present in "
                f"{n_frames} frame(s)",
                6000,
            )

    def _on_track_row_selected(self, frame: int, peak_id: int) -> None:
        """Jump to a clicked track's representative member (+select).

        ``peak_id`` is -1 when no fitted id could be reconstructed for
        the track — then the click only jumps to the frame.
        """
        if self.session is None or self._scan_track_entry is None:
            return
        entry = self.entry_combo.currentText()
        if entry != self._scan_track_entry:
            return
        frame = int(frame)
        if int(self.viewer.current_frame) != frame:
            # frameChanged is wired synchronously to _load_frame_peaks,
            # so the frame's tables are in _frame_peaks afterwards.
            self.viewer.set_frame(frame)
        if int(peak_id) < 0:
            return
        table = self.viewer._frame_peaks.get(frame, {}).get("fitted")
        if table is None:
            try:
                table = file_model.load_peaks(
                    self.session.temp_path, entry, frame
                ).get("fitted")
            except Exception:
                logger.debug(
                    "suppressed exception in MainWindow._on_track_row_selected",
                    exc_info=True,
                )
                return
        if table is None or len(table) == 0:
            return
        ids = [int(x) for x in table.ids]
        if int(peak_id) in ids:
            self._select_table_row(
                frame, "fitted", table, ids.index(int(peak_id))
            )

    def _on_peak_row_write_requested(
        self, frame: int, kind: str, peak_id: int, polar: dict
    ) -> None:
        """Persist a peak-row edit straight to the NeXus file.

        Carries a geometry change (the four polar fields) or a score
        change, whichever the caller put in ``polar`` --
        ``update_peak_row`` writes only what it is given, so a relabel
        cannot move a box and a drag cannot rewrite a label.

        ``kind`` may name a registered extra peak list, in which case the
        write is routed to THAT dataset rather than
        ``detected_peaks`` / ``fitted_peaks``. An unregistered list kind
        is dropped: the layer is gone, and writing to a table nobody is
        showing is worse than doing nothing.

        Drops silx's read handle for the duration of the write (matching the
        pipeline-run dance in ``_on_pipeline_run``) so h5py can open r+, then
        re-attaches. On KeyError (peak vanished), the undo/redo stacks are
        cleared since they're keyed on stale ids.
        """
        if self.session is None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        dataset = None
        if peak_lists.is_list_kind(kind):
            spec = self.viewer.peak_list_spec(kind)
            if spec is None:
                return
            dataset = spec.dataset
        with self._detached_silx_tree():
            try:
                file_model.update_peak_row(
                    self.session.temp_path, entry, frame, kind, peak_id,
                    dataset=dataset, **polar
                )
            except KeyError:
                QMessageBox.warning(
                    self, "Edit failed",
                    f"Peak id={peak_id} no longer exists in the file. "
                    "Undo history has been cleared.",
                )
                self.viewer.clear_history()
            except Exception as exc:
                QMessageBox.critical(self, "Edit failed", str(exc))
                self.viewer.clear_history()
            else:
                self.session.mark_dirty()
                self._update_title()
