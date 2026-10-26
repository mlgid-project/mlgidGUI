"""CIF parsing, pipeline dispatch and queueing, geometry preflight, conversion runs and the pipeline-finished dispatch hub.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import QMessageBox
from mlgidlab import file_model, peak_lists
from mlgidlab.main_window_constants import APP_NAME
from mlgidlab.pipeline import PipelineCommand
from mlgidlab.widgets import make_progress_dialog
from mlgidlab.workers import (
    CifParseWorker,
    ConversionWorker,
    PipelineWorker,
    stop_worker_thread,
)
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class PipelineMixin:
    # -- Pipeline --

    def _on_parse_cifs_requested(self, cif_input: str) -> None:
        """Run CIF parsing on a worker thread + post the result back.

        CIF preprocessing simulates every CIF and can take several
        seconds; the worker keeps the GUI responsive. Only one parse
        runs at a time — the panel's button stays disabled until we
        post the result back via ``set_cif_pattern``.
        """
        if self._cif_parse_thread is not None:
            return
        if self.session is None:
            self.pipeline_panel.set_cif_pattern(
                None, RuntimeError("Open a NeXus file first.")
            )
            return
        nexus_file = self.session.temp_path
        # Pass the active entry through so CifPattern is simulated
        # against that entry's energy / angle of incidence — multi-
        # energy datasets need this to match correctly.
        active_entry = self.entry_combo.currentText() or None
        self._cif_parse_thread = QThread(self)
        self._cif_parse_worker = CifParseWorker(
            cif_input, nexus_file, active_entry
        )
        self._cif_parse_worker.moveToThread(self._cif_parse_thread)
        self._cif_parse_thread.started.connect(self._cif_parse_worker.run)
        self._cif_parse_worker.finished.connect(self._on_parse_cifs_finished)
        self._cif_parse_thread.start()

    def _on_parse_cifs_finished(
        self, result: object | None, error: Exception | None
    ) -> None:
        self._cif_parse_thread, self._cif_parse_worker = stop_worker_thread(
            self._cif_parse_thread, self._cif_parse_worker
        )
        self.pipeline_panel.set_cif_pattern(result, error)
        if error is not None:
            self.pipeline_panel.append_log(f"CIF parse failed: {error}")
        elif result is not None:
            n = len(getattr(result, "cifs", []) or [])
            self.pipeline_panel.append_log(
                f"CIF cache loaded ({n} CIFs) — reused across matching runs"
            )

    def _on_run_requested(self, command: PipelineCommand) -> None:
        """Dispatch a runRequested command from the pipeline panel.

        "All entries" runs are expanded into one ``PipelineCommand`` per
        q-entry and queued sequentially — the user gets per-entry log
        lines and a single bad entry doesn't strand the others. A command
        that already names an explicit ``entry`` (or runs on a file with
        a single entry) goes straight through unchanged.

        Only the run_* ops are subject to expansion; every other op
        (inject/track/interpolate) always carries a specific ``entry``.

        The file path is snapshotted at this entry point and travels
        with every enqueued tuple, so a mid-queue active-session
        switch can't dispatch later commands at a different file.
        """
        if self.session is None:
            return
        if self._view_worker_blocks_pipeline():
            return
        file_path = self.session.temp_path
        if (
            command.op_name in ("run_detection", "run_fitting", "run_matching")
            and "entry" not in command.kwargs
        ):
            try:
                entries = file_model.list_entries(file_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Pipeline", f"Could not list entries: {exc}"
                )
                return
            if not entries:
                # No q-entries to run on — fall through and let mlgidbase
                # raise its usual "no entries" message in the log.
                self._entry_queue_total = 1
                self._entry_queue_pos = 0
                self._enqueue_pipeline(file_path, command)
                return
            # Multi-entry expansion: stash the queue depth + reset the
            # position counter so the entry progress bar starts at 0/N
            # and the first ``_on_pipeline_run`` advances to 1/N.
            self._entry_queue_total = len(entries)
            self._entry_queue_pos = 0
            for entry in entries:
                self._enqueue_pipeline(
                    file_path,
                    PipelineCommand(
                        command.op_name,
                        {**command.kwargs, "entry": entry},
                    ),
                )
        else:
            # Single-entry command (user picked one explicitly, or an
            # op that always carries its fixed entry). Counter
            # collapses to 1 so the panel keeps the entry bar hidden.
            self._entry_queue_total = 1
            self._entry_queue_pos = 0
            self._enqueue_pipeline(file_path, command)

    def _entry_missing_geometry(
        self, file_path: Path, entry: str | None
    ) -> bool:
        """True for an IMPORTED entry, which never has detector geometry.

        Keyed on the ``process/mlgidlab`` provenance group that only
        ``import_converted_stack`` writes, plus the actually-missing
        ``instrument/detector/distance`` — NOT on missing geometry
        alone: minimal or foreign mlgid files (test fixtures included)
        may lack the detector group too, and those must keep the
        pre-existing behavior of letting pygid report its own error.
        With no explicit entry the first entry decides (all-entries
        commands run on files whose entries share one provenance).
        Unreadable files return False for the same reason.
        """
        import h5py

        try:
            with h5py.File(file_path, "r") as f:
                names = (
                    [entry] if entry else file_model.entry_group_names(f)[:1]
                )
                for name in names:
                    if name not in f:
                        continue
                    g = f[name]
                    if (
                        "process/mlgidlab" in g
                        and "instrument/detector/distance" not in g
                    ):
                        return True
        except Exception:
            logger.debug(
                "suppressed geometry probe in _entry_missing_geometry",
                exc_info=True,
            )
        return False

    def _enqueue_pipeline(self, file_path: Path, command: PipelineCommand) -> None:
        """Queue ``(file_path, command)`` and start it if no run is in flight.

        The single funnel every pipeline command passes through, which
        is why the primary-table plan is stamped HERE rather than at the
        eight ``PipelineCommand(...)`` construction sites: none of them
        changes, and the multi-entry expansion in ``_on_pipeline_run``
        (which rebuilds the command per entry) still lands here.

        Reading the registry on this side is deliberate. The worker has
        no business touching QSettings, so what crosses the thread
        boundary is finished plain data.
        """
        command.table_swap = peak_lists.swap_plan(
            command.op_name, peak_lists.load_specs()
        )
        self._pipeline_queue.append((file_path, command))
        if self._pipe_thread is None:
            self._run_next_pipeline_command()

    def _run_next_pipeline_command(self) -> None:
        """Pop the next queued (path, command) tuple and start it, if any."""
        if self._pipe_thread is not None or not self._pipeline_queue:
            return
        file_path, command = self._pipeline_queue.pop(0)
        self._on_pipeline_run(file_path, command)

    def _ensure_entry_normalized(self, file_path: Path, entry: str | None) -> None:
        """Run pygid normalization for ``entry`` once per (file, entry).

        Normalization is deferred off the open path (see
        ``_on_open_finished``); this runs it lazily, scoped to the one
        entry a pipeline command targets. **Must be called with the silx
        tree detached** (no read handle on ``file_path``) — it writes new
        groups, which can't race the viewer's reads. Idempotent on repeat
        calls; a failure is non-fatal (the pipeline surfaces its own
        error if the file is genuinely unusable).
        """
        if not entry:
            return
        done = self._normalized_entries.setdefault(str(file_path), set())
        if entry in done:
            return
        try:
            patched = file_model.normalize_for_pygid(file_path, entry=entry)
        except Exception:
            logger.debug("suppressed exception in MainWindow._ensure_entry_normalized", exc_info=True)
            patched = {"angle": [], "frames": []}
        if patched.get("angle"):
            self.pipeline_panel.append_log(
                "Normalized 0-D angle_of_incidence in: " + ", ".join(patched["angle"])
            )
        if patched.get("frames"):
            self.pipeline_panel.append_log(
                "Created missing per-frame analysis groups in: "
                + ", ".join(patched["frames"])
            )
        done.add(entry)

    def _on_pipeline_run(self, file_path: Path, command: PipelineCommand) -> None:
        if self._is_busy():
            return

        # Imported image scans carry no detector geometry — pygid's
        # nexus loader hard-reads instrument/detector/* and the
        # monochromator wavelength for EVERY op, so refuse with a clear
        # message instead of surfacing a KeyError traceback.
        if self._entry_missing_geometry(file_path, command.kwargs.get("entry")):
            self.pipeline_panel.append_log(
                f"--- {command.op_name} skipped: this scan was imported "
                "from pre-converted images without q ranges and a "
                "wavelength, so no usable geometry exists. Re-import "
                "the images with both filled in (File → Import images "
                "as converted scan…) to enable "
                "detection/fitting/matching. ---"
            )
            # Continue the queue without recursing (a multi-entry batch
            # would otherwise nest one stack frame per skipped entry).
            QTimer.singleShot(0, self._run_next_pipeline_command)
            return

        # Stop frame playback if it's running — the pipeline owns the
        # file r+ for the duration of the run and ticking would either
        # contend on the silx detach or read post-write data
        # mid-render.
        self._pause_playback()
        self.pipeline_panel.set_running(True)
        self.parameter_panel.set_busy(True)
        self.scan_tracking_panel.set_busy(True)
        self.viewer.set_busy(True)
        self._update_status_pipeline(command, running=True)
        # Every pipeline op can reshuffle peak ids, which invalidates
        # pending FileGeomActions — drop history and selection up front.
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Per-run header: include the entry scope when present so the
        # user can see which entry is being processed in a multi-entry
        # batch.
        entry_tag = command.kwargs.get("entry")
        if entry_tag:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} on {entry_tag} ---"
            )
        else:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} (all entries) ---"
            )

        # Release silx's read handles on every loaded temp file so mlgidbase
        # can open the active one r+. Sibling files are reattached on finish.
        self._detach_silx_tree()

        # Normalize this entry for pygid now — lazily, once per entry.
        # Done here (tree detached, before the worker's r+ open) so the
        # group-creating write can't race the viewer's reads. Replaces the
        # old normalize-every-entry-on-open that froze huge files.
        self._ensure_entry_normalized(file_path, command.kwargs.get("entry"))

        # Stash the active command so the status-bar progress mirror
        # (``_on_pipeline_frame_progress``) can rebuild the status text
        # without needing to plumb the command through every emit.
        self._pipe_command = command
        self._pipe_progress_tail = ""

        # Advance the entry-level queue position and drive the panel's
        # second progress row. Single-entry runs (total == 1) keep the
        # entry bar hidden via the panel's own guard.
        if self._entry_queue_total >= 1:
            self._entry_queue_pos += 1
        entry_for_progress = command.kwargs.get("entry") or ""
        self.pipeline_panel.on_queue_progress(
            self._entry_queue_pos,
            self._entry_queue_total,
            entry_for_progress,
            command.op_name,
        )

        self._pipe_thread = QThread(self)
        # Use the file_path snapshotted at enqueue time, NOT
        # ``self.session.temp_path`` — they can disagree if the user
        # switched the active session between clicking Run and the
        # queue actually dispatching this command. Surfaced as a
        # pre-flight failure in pipeline.execute that named the wrong
        # file's entries.
        self._pipe_worker = PipelineWorker(file_path, command)
        self._pipe_worker.moveToThread(self._pipe_thread)
        self._pipe_worker.log.connect(self.pipeline_panel.append_log)
        self._pipe_worker.frameProgress.connect(self.pipeline_panel.on_frame_progress)
        # Mirror the frame counter into the status-bar tail so the user
        # gets a glanceable counter without needing the pipeline panel
        # in view. Stored on ``self`` so ``_update_status_pipeline``
        # can fold it into the status string.
        self._pipe_worker.frameProgress.connect(self._on_pipeline_frame_progress)
        self._pipe_worker.finished.connect(self._on_pipeline_finished)
        self._pipe_thread.started.connect(self._pipe_worker.run)
        self._pipe_thread.start()

    # -- Conversion (raw → NeXus) --

    def _on_conversion_run(self, cfg, scans: list) -> None:
        """Spawn the ConversionWorker for a fresh run.

        ``cfg`` is a ``ConversionConfig``; ``scans`` is a list of
        ``RawScan``. We don't refuse on overlapping output paths here
        — pygid handles overwrite-or-append per scan via ``cfg``'s
        flags.
        """
        if self._conv_thread is not None:
            QMessageBox.information(
                self, "Conversion in progress",
                "A conversion run is already in flight; please wait for it "
                "to finish before starting another.",
            )
            return
        # Modal progress dialog — a long batch can run for minutes; the
        # user needs a way to see it's progressing without watching the
        # log pane scroll.
        self._conv_progress = make_progress_dialog(
            self, f"Running {len(scans)} scan(s)…",
            title=APP_NAME, maximum=max(len(scans), 1),
        )

        self.conversion_panel.set_running(True)
        self.conversion_panel.clear_log()
        self.conversion_panel.append_log(
            f"Starting conversion: {len(scans)} scan(s) → {cfg.output_dir}"
        )

        self._conv_thread = QThread(self)
        self._conv_worker = ConversionWorker(scans, cfg)
        self._conv_worker.moveToThread(self._conv_thread)
        self._conv_thread.started.connect(self._conv_worker.run)
        self._conv_worker.log.connect(self.conversion_panel.append_log)
        self._conv_worker.progress.connect(self._on_conversion_progress)
        self._conv_worker.finished.connect(self._on_conversion_finished)
        self._conv_thread.start()

    def _on_conversion_progress(self, done: int, total: int) -> None:
        if self._conv_progress is None:
            return
        self._conv_progress.setMaximum(max(total, 1))
        self._conv_progress.setValue(done)

    def _on_conversion_finished(
        self, output_paths: list | None, error: Exception | None
    ) -> None:
        self._conv_thread, self._conv_worker = stop_worker_thread(
            self._conv_thread, self._conv_worker
        )
        if self._conv_progress is not None:
            self._conv_progress.close()
            self._conv_progress.deleteLater()
            self._conv_progress = None

        self.conversion_panel.set_running(False)

        if error is not None:
            self.conversion_panel.append_log(f"ERROR - {error}")
            QMessageBox.critical(self, "Conversion failed", str(error))
            return

        outputs = list(output_paths or [])
        if not outputs:
            self.conversion_panel.append_log(
                "Conversion completed but produced no output paths."
            )
            return

        self.conversion_panel.append_log(
            "Conversion DONE. Output files:\n  " + "\n  ".join(str(p) for p in outputs)
        )

        # Auto-open: queue every produced file as a NeXus session. The
        # existing CopyWorker path normalizes pygid metadata and handles
        # silx-tree insertion; ``_set_active_session`` swaps focus once
        # the first file lands.
        for out_path in outputs:
            self._open_queue.append(Path(out_path))
        self._process_open_queue()

    def _flag_rings_after_fit(self, cmd) -> None:
        """Type the run's infinite-width fitted rows as rings.

        pygidfit marks a ring only by an infinite angular width, and
        ``run_fitting`` stores its rows as they come, so they would
        read as segments. Scope: the command's entry / frame and the
        run's primary fitted table. Writes directly while the silx tree
        is detached, best-effort: a failure is logged, the fits stay.
        Independent of the peak link.
        """
        entry_arg = cmd.kwargs.get("entry")
        frame_arg = cmd.kwargs.get("frame_num")
        frame = int(frame_arg) if isinstance(frame_arg, int) else None
        fitted_ds = peak_lists.swapped_name(
            getattr(cmd, "table_swap", None), "fitted_peaks",
        )
        try:
            if isinstance(entry_arg, str) and entry_arg:
                entries = [entry_arg]
            else:
                entries = file_model.list_entries(self.session.temp_path)
            changed = sum(
                file_model.flag_infinite_width_fitted_as_rings(
                    self.session.temp_path, ent, frame=frame,
                    fitted_dataset=fitted_ds,
                )
                for ent in entries
            )
        except Exception as exc:
            self.pipeline_panel.append_log(
                f"Could not type the infinite-width fits as rings: {exc}"
            )
            return
        if changed:
            self.pipeline_panel.append_log(
                f"Typed {changed} fitted peak(s) with an infinite angular "
                f"width as rings."
            )

    def _on_pipeline_finished(self, result: object, error: Exception | None) -> None:
        self._pipe_thread, self._pipe_worker = stop_worker_thread(
            self._pipe_thread, self._pipe_worker
        )

        # Close the tracking loading dialog as soon as its run finishes
        # (before any error modal), regardless of success/failure. The
        # Interpolate-track chain keeps it up across all its queued
        # commands; it closes below when the chain's LAST command lands.
        if (
            getattr(self, "_pipe_command", None) is not None
            and self._pipe_command.op_name == "track_peaks"
        ):
            self._close_tracking_progress()

        if error is not None:
            self.pipeline_panel.append_log(f"ERROR - {error}")
            # In a queued multi-entry batch, surface the error in the log
            # but only show the modal once at the *end* (otherwise the user
            # gets a dialog per entry and the run halts in front of every
            # one). For single-command runs (queue empty) keep the modal.
            if not self._pipeline_queue:
                QMessageBox.critical(self, "Pipeline error", str(error))
        else:
            self.pipeline_panel.append_log("DONE")

        if self.session is not None and error is None:
            self.session.mark_dirty()

        # pygidfit >= 0.1.8 writes each fitted row under the id of the
        # detected row it came from, so the fitted/detected pairing the
        # link relies on survives a run without any re-keying here
        # (older versions numbered fitted rows 0..N-1 by position and
        # needed a pass; the pinned version no longer does). What a run
        # still leaves to do is typing its infinite-width rows as rings.
        # Done here rather than inside ``pipeline.execute``: the silx
        # tree is still detached at this point (it reattaches only when
        # the queue drains), so the write is safe.
        cmd = getattr(self, "_pipe_command", None)
        if (
            error is None
            and self.session is not None
            and cmd is not None
            and cmd.op_name == "run_fitting"
        ):
            self._flag_rings_after_fit(cmd)

        # A re-fit rewrites the entry's fitted_peaks wholesale, so any
        # phase-tracking results for that entry are stale. Invalidate
        # when the finished command was a fitting run touching the
        # tracked entry (kwargs without an entry ran on every entry).
        # A run redirected to a primary fitted table did NOT rewrite
        # fitted_peaks — the tracks still describe the rows they were
        # built from, so invalidating them would throw away a good
        # (and expensive) result.
        if (
            error is None
            and self._scan_track_entry is not None
            and cmd is not None
            and cmd.op_name == "run_fitting"
            and cmd.kwargs.get("entry") in (None, self._scan_track_entry)
            and not peak_lists.redirects_dataset(
                cmd.table_swap, "fitted_peaks"
            )
        ):
            self._invalidate_scan_tracks()

        # track_peaks is the one pipeline op whose RESULT the GUI
        # consumes (the captured TrackingPayload) — every other op's
        # output is read back from the file.
        if error is None and cmd is not None and cmd.op_name == "track_peaks":
            self._on_phase_track_result(result, cmd)

        # interpolate_tracks also returns its result (the gap-fill
        # records) — stash it until the chained re-match lands, then
        # apply the new peaks as track members. Applied even when the
        # re-match itself errored: the fills are real fitted rows either
        # way, and the error modal above already told the user.
        if (
            error is None
            and cmd is not None
            and cmd.op_name == "interpolate_tracks"
        ):
            self._interp_fill_result = list(result) if result else None
            if not result:
                self.pipeline_panel.append_log(
                    "Interpolate track: no gap could be filled "
                    "(every fit failed or was skipped — see warnings "
                    "above)."
                )
        # Expected-pattern injection: the records are NOT track fills —
        # never route them into _interp_fill_result (its applier would
        # try to append them as track members). They land in the
        # validation stash instead, consumed at chain completion.
        if (
            error is None
            and cmd is not None
            and cmd.op_name == "inject_fitted_peaks"
        ):
            n = len(result) if result else 0
            if self._predicted_fill is not None:
                self._predicted_fill["records"] = (
                    list(result) if result else []
                )
            if n:
                self.pipeline_panel.append_log(
                    f"Expected pattern: injected and fitted {n} "
                    "predicted peak(s); validating against the "
                    "re-match…"
                )
            else:
                self.pipeline_panel.append_log(
                    "Expected pattern: no predicted peak could be "
                    "added (every fit failed — see warnings above)."
                )
        # A failed re-match must not trigger the validation rollback —
        # nothing got matched, so every injected peak would be wrongly
        # discarded for a crash that wasn't its fault.
        if (
            error is not None
            and cmd is not None
            and cmd.op_name == "run_matching"
            and self._predicted_fill is not None
        ):
            self._predicted_fill["match_ok"] = False
        # Injection-chain accounting (interpolate-track fills AND
        # expected-pattern additions share it; both are busy-gated so
        # they never interleave): advance the dialog's completed-work
        # base, and when the chain's LAST command lands (the final
        # re-match — or the fill itself when every re-match was
        # skipped), apply any stashed track fills and drop the dialog.
        # Applied even when a re-match errored: the fills are real
        # fitted rows either way, and the error modal above already
        # told the user.
        chain = self._interp_chain
        if chain is not None and cmd is not None and id(cmd) in chain["ticks"]:
            chain["base"] += chain["ticks"].pop(id(cmd))
            if not chain["ticks"]:
                self._interp_chain = None
                self._close_tracking_progress()
                self._apply_interp_fill_result()
                if chain.get("label") == "Expected pattern":
                    # Keep only the injected peaks the matcher
                    # attributed to the target structure (silx is
                    # still detached here, so the rollback's r+
                    # writes are safe), then drop the stale white
                    # selection — the queue-drain reload below
                    # re-classifies the survivors green.
                    self._finalize_predicted_fill()
                    self.viewer.clear_simulation_selection()

        # If more commands are queued, run the next one without
        # tearing down the silx tree / viewer state for the user — keep
        # the busy gating active and chain straight into the next run.
        if self._pipeline_queue:
            self._run_next_pipeline_command()
            return

        # Queue drained — final cleanup. Reattach silx, refresh the
        # viewer for the active entry, lift busy gating. Reset the
        # entry-queue counters so the next run starts from a clean
        # slate (the panel's set_running(False) below also hides both
        # progress rows).
        self._entry_queue_total = 0
        self._entry_queue_pos = 0
        self.pipeline_panel.set_running(False)
        self.parameter_panel.set_busy(False)
        self.scan_tracking_panel.set_busy(False)
        self.viewer.set_busy(False)
        self._update_sim_add_button()
        self._update_status_pipeline(running=False)
        self._reattach_silx_tree()
        if self.session is not None:
            entry = self.entry_combo.currentText()
            if entry:
                # Same entry, same axes — preserve the user's zoom and
                # frame across the overlay refresh.
                self._load_entry_into_viewer(entry, preserve_view=True)
            self._update_title()
            # The rail counts what the viewer holds, and what the viewer
            # holds only changes on the reload above — the refresh inside
            # _update_status_pipeline ran before it and therefore against
            # the previous run's tables. Without this the chip for the run
            # that just finished still reads "not run" until the user
            # steps to another frame.
            self._refresh_workflow_rail()
