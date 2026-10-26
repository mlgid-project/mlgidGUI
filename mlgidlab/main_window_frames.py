"""Entry population (NeXus and raw), frame navigation and playback, the background prefetch worker, tree-selection activation and the async entry loader.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import math
import numpy as np
import time
from PySide6.QtCore import (
    QSettings,
    QSignalBlocker,
    QThread,
    Qt,
    Slot,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMessageBox
from mlgidlab import file_model
from mlgidlab import icons
from mlgidlab import peak_lists
from mlgidlab.browser_widgets import _ImageFileNode
from mlgidlab.image_viewer import OVERLAY_KINDS
from mlgidlab.main_window_constants import (
    DEFAULT_PLAYBACK_FRAME_MS,
    DEFAULT_PLAYBACK_TOTAL_S,
    PLAYBACK_FRAME_MS_MAX,
    PLAYBACK_FRAME_MS_MIN,
    PLAYBACK_MODE_FRAME,
    PLAYBACK_MODE_TOTAL,
    PLAYBACK_TICK_FLOOR_MS,
    PLAYBACK_TOTAL_S_MAX,
    PLAYBACK_TOTAL_S_MIN,
)
from mlgidlab import pipeline
from mlgidlab.session import RawSession
from mlgidlab.workers import EntryLoadWorker, PrefetchWorker
from pathlib import Path
from silx.gui.hdf5 import Hdf5TreeModel

import logging

logger = logging.getLogger(__name__)


class FramesMixin:
    # -- Entry / viewer wiring --

    def _populate_entries(self) -> None:
        if self.session is None:
            return
        if self.session.kind == "raw":
            self._populate_raw_entries()
            return
        # Prefer the entry list the open worker already computed (consumed
        # once); otherwise read the names shallowly now. Both paths use the
        # master's link names only and never resolve the external scans —
        # resolving them is what froze the GUI when opening a big master
        # (it opens every linked scan and holds the GIL while doing so).
        entries = getattr(self.session, "_entries_cache", None)
        if entries is not None:
            self.session._entries_cache = None  # type: ignore[attr-defined]
        else:
            try:
                entries = file_model.list_entry_names(self.session.temp_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Read failed", f"Could not list entries: {exc}"
                )
                return
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(entries)
        # Land on the prewarmed entry when one is queued: on first open
        # that is entries[0] (CopyWorker's warm frame 0); on RE-activation
        # it is whatever entry the user was on when they switched away —
        # ``_set_active_session`` stashes the live FrameSource as the
        # outgoing session's prewarm, so switching back restores their
        # place from memory instead of re-reading the file.
        target = entries[0] if entries else ""
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] in entries:
            target = prewarm[0]
            self.entry_combo.setCurrentText(target)
        self.entry_combo.blockSignals(False)
        # Push the same entry list into the pipeline panel's per-section
        # scope dropdowns so the user can pick a specific entry instead
        # of being limited to ACTIVE / ALL.
        self.pipeline_panel.set_available_entries(entries)
        if entries:
            self._load_entry_into_viewer(target)
        else:
            # Empty entry combo isn't always "empty file" — it's much more
            # often "file has entries but none are 2D q-images". Tell the
            # user exactly what's in the file so they don't think the GUI
            # silently dropped a working file.
            self._warn_no_q_entries()

    def _populate_raw_entries(self) -> None:
        """Walk every raw file in the active session and populate the
        entry combo with its 3D detector-image candidates.

        Combo items are labeled ``filename::dataset/path`` so the user
        can disambiguate when the batch contains multiple files. Pipeline
        panel's per-entry scope dropdown is cleared — pipeline ops aren't
        meaningful in raw mode. The Conversion panel also receives the
        same set of (file, entries) tuples for its selection tree.
        """
        assert isinstance(self.session, RawSession)
        # Maintain a mapping from combo label → RawEntry so the change
        # handler can resolve a click without re-walking the HDF5 file.
        self._raw_entries: dict[str, file_model.RawEntry] = {}
        self._raw_entry_label_by_path = {}
        # Per-file entry lists are cached on the session: CopyWorker fills
        # the cache when the file is opened, and a GUI-thread fallback walk
        # fills it once for sessions that arrived without one. Activation
        # is on the GUI thread, so this must not re-walk a big beamtime
        # file's metadata every time the user switches back to the session.
        labels, panel_inputs = self._collect_raw_inputs(self.session.raw_paths)
        # Push the same data into the Conversion panel for its selection
        # tree. Done before populating the combo so the panel paint
        # happens once on activation.
        self.conversion_panel.set_raw_inputs(panel_inputs)
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(labels)
        self.entry_combo.blockSignals(False)
        self.pipeline_panel.set_available_entries([])
        if labels:
            # Auto-load the first candidate so the user sees something
            # immediately. The change handler handles further picks.
            self._load_raw_entry_into_viewer(labels[0])
        else:
            QMessageBox.information(
                self,
                "No raw datasets found",
                "None of the selected raw files contain a 3D detector "
                "dataset (shape (N, H, W) with H, W ≥ 32). Check the "
                "files in the tree on the left to see their structure.",
            )

    def _collect_raw_inputs(
        self, paths: list[Path]
    ) -> tuple[list[str], list[tuple[Path, list[file_model.RawEntry]]]]:
        """Entry labels and Conversion-panel rows for ``paths``.

        Registers each entry in ``self._raw_entries`` (label → RawEntry)
        and, for standalone images, in ``self._raw_entry_label_by_path``;
        existing registrations are left alone, so this serves both the
        full populate and the append path. HDF5 entry lists come from the
        session's cache, with a one-off walk for files that have none.
        """
        assert isinstance(self.session, RawSession)
        labels: list[str] = []
        panel_inputs: list[tuple[Path, list[file_model.RawEntry]]] = []
        cache: dict[str, list] = getattr(self.session, "_raw_entries_cache", None) or {}
        # Standalone image files (TIFF/CBF/EDF) are read via fabio, not the
        # h5 detector walk; they become one 1-frame entry per file below.
        # Keep the HDF5 raw-walk path exactly as before for the rest.
        fabio_paths = [p for p in paths if file_model.is_fabio_image(p)]
        h5_paths = [p for p in paths if not file_model.is_fabio_image(p)]
        for raw_path in h5_paths:
            entries = cache.get(str(raw_path))
            if entries is None:
                try:
                    entries = file_model.list_raw_entries(raw_path)
                except Exception as exc:
                    self.conversion_panel.append_log(
                        f"Could not read {raw_path.name}: {exc}"
                    )
                    panel_inputs.append((raw_path, []))
                    continue
                cache[str(raw_path)] = entries
            panel_inputs.append((raw_path, entries))
            for re in entries:
                self._raw_entries[re.label] = re
                labels.append(re.label)
        self.session._raw_entries_cache = cache  # type: ignore[attr-defined]
        # One entry PER image file (each a standalone 1-frame view), read via
        # LazyFabioStack — multiple images open separately, not as a stack.
        if fabio_paths:
            try:
                fabio_entries = file_model.list_fabio_entries(fabio_paths)
            except Exception as exc:
                self.conversion_panel.append_log(
                    f"Could not read image files: {exc}"
                )
            else:
                for entry in fabio_entries:
                    self._raw_entries[entry.label] = entry
                    labels.append(entry.label)
                    panel_inputs.append((entry.file_path, [entry]))
                    # Browser click-to-view: an _ImageFileNode row
                    # resolves to its combo label through this map.
                    self._raw_entry_label_by_path[str(entry.file_path)] = (
                        entry.label
                    )
        return labels, panel_inputs

    def _append_raw_entries(self, paths: list[Path]) -> None:
        """Add newly dropped raw files to the active raw session's views.

        The Conversion panel keeps its rows and their check state, the
        entry combo gains the new labels, and the viewer jumps to the
        first new entry, as it does when a first file is opened.
        """
        assert isinstance(self.session, RawSession)
        labels, panel_inputs = self._collect_raw_inputs(paths)
        self.conversion_panel.add_raw_inputs(panel_inputs)
        if not labels:
            return
        self.entry_combo.blockSignals(True)
        self.entry_combo.addItems(labels)
        self.entry_combo.blockSignals(False)
        # Signals are live again, so this selects and loads the entry.
        self.entry_combo.setCurrentText(labels[0])

    def _on_raw_flips_changed(
        self, flip_lr: bool, flip_ud: bool, transp: bool = False,
    ) -> None:
        """Apply the Conversion panel's Orientation row to the raw preview."""
        self.viewer.set_raw_flips(flip_lr, flip_ud, transp)

    def _load_raw_entry_into_viewer(self, label: str) -> None:
        """Load the picked raw entry into the viewer in pixel coords.

        ``label`` is the combo's display string (file::dataset/path).
        Resolved through ``self._raw_entries`` to a ``RawEntry`` and
        dispatched to the EntryLoadWorker, which opens the dataset and
        warms frame 0 off the GUI thread (``_on_raw_entry_loaded``
        installs the result). The old synchronous path materialized the
        WHOLE 3D dataset on this thread — a multi-GB read that froze
        the window for its duration on big beamtime files.
        """
        raw_entry = getattr(self, "_raw_entries", {}).get(label)
        if raw_entry is None:
            return
        self._entry_req_id += 1
        self._ensure_entry_load_worker()
        self._sb_open_label.setText(f"Loading {label}…")
        self._sb_open_label.show()
        self._sb_open_bar.setRange(0, 0)  # indeterminate: a single read, no stages
        self._sb_open_bar.show()
        self._rawLoadRequest.emit(raw_entry, self._entry_req_id)

    def _on_raw_entry_loaded(self, request_id: int, label: str, stack) -> None:
        """Install the raw stack the worker just opened, unless a newer
        switch superseded it (then its handle is released). Mirrors
        ``_on_entry_loaded``."""
        if request_id != self._entry_req_id:
            if stack is not None:
                try:
                    stack.release()
                except Exception:
                    logger.debug("suppressed stale raw-stack release", exc_info=True)
            return
        self._dismiss_open_progress()
        if stack is None:
            QMessageBox.warning(self, "Load failed", f"Could not load {label}")
            return
        # The active session may have changed between dispatch and arrival
        # (file closed / switched to a NeXus session) — never install an
        # orphaned stack.
        if self.session is None or self.session.kind != "raw":
            try:
                stack.release()
            except Exception:
                logger.debug("suppressed orphan raw-stack release", exc_info=True)
            return
        self.viewer.show_raw_stack(stack)
        self._refresh_frame_slider()
        self._update_status_frame()

    def _warn_no_q_entries(self) -> None:
        """Diagnose why the entry combo ended up empty.

        The viewer + pipeline only handle ``img_gid_q`` entries. Files
        with reduced-data entries (``horiz_cut_gid``, ``rad_cut_gid``,
        polar-only ``img_gid_pol``, etc.) load fine but produce no
        viewable entry, which previously looked like the GUI broke.
        """
        try:
            signals = file_model.list_entry_signals(self.session.temp_path)
        except Exception:
            logger.debug("suppressed exception in MainWindow._warn_no_q_entries", exc_info=True)
            signals = {}
        if not signals:
            # Truly empty — file genuinely has no entry_* groups.
            QMessageBox.information(
                self,
                "Nothing to show",
                f"{self.session.original_path.name} has no entry_* groups "
                "to display.",
            )
            return
        rows = "\n".join(
            f"  • {name} — signal = {signal!r}"
            for name, signal in signals.items()
        )
        QMessageBox.information(
            self,
            "No 2D q-image entries",
            f"{self.session.original_path.name} loaded successfully but "
            "contains no entries with the 2D q-image data the viewer + "
            "pipeline operate on (signal = 'img_gid_q').\n\n"
            f"Entries found:\n{rows}\n\n"
            "These are likely reduced data (1D cuts, polar grids, or "
            "post-processed outputs). To use the GUI's detection / "
            "fitting / matching tools, open a NeXus file produced by "
            "the pygid → mlgidDETECT pipeline that still carries the "
            "raw q-image stack.",
        )

    def _on_entry_changed(self, entry: str) -> None:
        if not entry or self.session is None:
            return
        # A pending quick-select box belongs to the entry it was drawn
        # on. Commit it before the switch, while the entry combo still
        # names that entry — the commit reads it from there.
        self.viewer.commit_pending_manual()
        if self._entry_load_is_deferred(entry):
            return
        self._structure_pending_entry = None
        if self.session.kind == "raw":
            self._load_raw_entry_into_viewer(entry)
            self._update_status_entry()
            self._update_status_frame()
        else:
            # Interactive NeXus switch (combo or file-browser click): read
            # the entry's frame OFF the GUI thread so a big detector frame
            # over a slow share never freezes the window. The status bar is
            # refreshed in ``_on_entry_loaded`` once the frame is in.
            self._load_entry_async(entry)
            self._update_status_entry()
            # Multi-energy files: the parsed CIF cache is entry-specific.
            self._sim_entry_mismatch_hint(entry)

    def _entry_load_is_deferred(self, entry: str) -> bool:
        """Whether to skip the frame load because Structure is in front.

        The read already happens off the GUI thread; the install does
        not, and on a large detector frame that install — the image, the
        autoscale, the overlays, the profile — is the freeze the user
        feels. All of it lands in widgets the Structure tab has folded
        away, so while that tab is up the entry is only remembered. It
        is reconciled from the combo when the tab is left, which is the
        first moment anyone can see the difference.

        Both attributes are read defensively: this runs during window
        construction, before the tabs exist, and a missing panel must
        read as "not on that tab" rather than as "on it".
        """
        panel = getattr(self, "structure_panel", None)
        tabs = getattr(self, "tabs", None)
        if panel is None or tabs is None or tabs.currentWidget() is not panel:
            return False
        self._structure_pending_entry = entry
        # The status bar is not folded away, so it keeps up.
        self._update_status_entry()
        return True

    def _on_frame_slider_changed(self, value: int) -> None:
        """User dragged the Display-dock slider — push to the viewer.

        ``viewer.set_frame`` is a no-op when ``value`` already matches
        the current frame, so the bidirectional sync (slider→viewer→
        slider via _on_viewer_frame_changed) doesn't recurse.
        """
        self.viewer.set_frame(value)

    def _on_frame_spin_changed(self, value: int) -> None:
        """Exact frame index typed (or stepped) — same seek path as the
        slider; the viewer's no-op on an unchanged frame breaks the
        sync loop here too."""
        self.viewer.set_frame(value)

    def _on_viewer_frame_changed(self, frame: int) -> None:
        """Viewer changed frame (timeline scrub, programmatic seek, etc.)
        — keep the Display-dock slider + label in sync without
        re-emitting valueChanged back into the viewer. Also pushes
        the new play-head into the background prefetch worker so
        its sliding window slides with the user.
        """
        # Lazily pull this frame's peaks into the viewer (entry switch
        # loads only the landed frame; navigating loads the rest here on
        # demand). Runs before the peaks-table / matched-panel frame
        # slots — they're connected after this one — so they see the
        # freshly-installed peaks.
        entry = self.entry_combo.currentText()
        if entry:
            self._load_frame_peaks(entry, int(frame))
        self.frame_label.setText(self._frame_label_text(frame))
        if self.frame_slider.value() != frame:
            self.frame_slider.blockSignals(True)
            try:
                self.frame_slider.setValue(int(frame))
            finally:
                self.frame_slider.blockSignals(False)
        if self.frame_spin.value() != frame:
            with QSignalBlocker(self.frame_spin):
                self.frame_spin.setValue(int(frame))
        self._refresh_frame_nav_enabled()
        self._update_status_frame()
        # Tell the prefetch worker where the play-head is now. The
        # ``active`` flag tracks the play-button state so a manual
        # scrub doesn't accidentally wake the worker. Pass the live
        # step so the worker walks the same stride as the player.
        if self._prefetch_worker is not None:
            active = self.play_button.isChecked()
            step = self._play_step if active else 1
            self._prefetchUpdate.emit(int(frame), active, step)
        # Reset the Detected min-score slider to the new frame's
        # min score. Matched is reseeded via _refresh_matched_panel,
        # which fires from the viewer's matchedStructuresChanged
        # signal on the same frame change.
        self._seed_detected_score_slider()

    def _on_play_toggled(self, checked: bool) -> None:
        """Start / stop the frame-playback timer.

        Press → if the current frame is already at the end, restart
        from frame 0; otherwise advance from the current frame. Press
        again to pause. The icon flips between Play and Pause.

        Refuses to start during a pipeline run (the viewer is gated
        ``busy`` during those, so frame edits would block anyway).

        Reads the current playback settings from QSettings on every
        press so a setting change applies on the next play without
        any restart machinery.
        """
        if checked:
            if self._pipe_thread is not None or self.viewer.n_frames <= 1:
                # Bail: the play button toggle was either an erroneous
                # programmatic click or fired while a pipeline run owns
                # the viewer. Unchecking re-fires this slot with
                # checked=False, which is a no-op.
                self.play_button.setChecked(False)
                return
            # Wrap around when at the end so the second click of Play
            # always plays the full sequence.
            if self.viewer.current_frame >= self.viewer.n_frames - 1:
                self.viewer.set_frame(0)
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            self.play_button.setIcon(icons.icon("pause"))
            self.play_button.setToolTip("Pause playback")
            self._play_timer.start()
            # Activate the background prefetch worker — it'll start
            # warming frames just ahead of the play-head, stepping
            # the same way the player does so prefetched frames
            # actually match the ones we'll display.
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )
        else:
            self._play_timer.stop()
            self._play_step = 1
            self.play_button.setIcon(icons.icon("play"))
            self.play_button.setToolTip(
                "Play frames from the current position to the end.\n"
                "Stops at the last frame; click again to pause."
            )
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, False, 1,
                )

    def _compute_play_schedule(self) -> tuple[int, int]:
        """Resolve ``(timer_interval_ms, frame_step)`` from QSettings.

        The user expresses a desired *per-frame* duration — either
        directly (Time-per-frame mode) or implicitly (Total-time mode
        ÷ n_frames). If that desired duration is at or above
        ``PLAYBACK_TICK_FLOOR_MS`` (≈ 20 fps), playback uses it
        directly with ``step=1``. If it's *below* that floor, the
        timer is held at the floor and ``step`` is bumped so the
        play-head jumps multiple frames per tick — i.e. we honour the
        target total time by skipping frames instead of asking Qt to
        fire faster than the display + disk can keep up. 20 fps is
        more than enough to perceive the time-series motion; the
        skipped frames are still reachable via the slider.

        Out-of-bounds / unparseable stored values fall back to the
        defaults so a corrupted QSettings entry can't soft-lock the
        Play button.
        """
        settings = QSettings()
        mode = settings.value(self._PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME)
        if mode == PLAYBACK_MODE_TOTAL:
            try:
                total_s = float(settings.value(
                    self._PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
                ))
            except (TypeError, ValueError):
                total_s = DEFAULT_PLAYBACK_TOTAL_S
            total_s = max(PLAYBACK_TOTAL_S_MIN,
                          min(PLAYBACK_TOTAL_S_MAX, total_s))
            steps = max(self.viewer.n_frames - 1, 1)
            desired_ms = total_s * 1000.0 / steps
        else:
            try:
                frame_ms = int(settings.value(
                    self._PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
                ))
            except (TypeError, ValueError):
                frame_ms = DEFAULT_PLAYBACK_FRAME_MS
            desired_ms = float(max(PLAYBACK_FRAME_MS_MIN,
                                   min(PLAYBACK_FRAME_MS_MAX, frame_ms)))

        if desired_ms < PLAYBACK_TICK_FLOOR_MS:
            # Below the 20 fps ceiling — bunch frames together per tick.
            # step = ceil(floor / desired) so the per-frame time stays
            # ≤ desired (= we never play slower than asked). Interval
            # then = desired * step, which lands at or just above the
            # floor.
            step = max(1, int(math.ceil(PLAYBACK_TICK_FLOOR_MS / desired_ms)))
            interval_ms = max(PLAYBACK_TICK_FLOOR_MS,
                              int(round(desired_ms * step)))
        else:
            step = 1
            interval_ms = int(round(desired_ms))
        return interval_ms, step

    def _on_play_tick(self) -> None:
        """One step of frame playback.

        Stops at end-of-stack. Auto-pauses if the viewer becomes busy
        (pipeline run kicked off mid-playback) or the user closed the
        file. The slider's ``valueChanged`` connection routes the
        frame change through ``viewer.set_frame`` so the existing
        sync paths fire exactly once per step.
        """
        if (
            self.viewer.n_frames <= 1
            or self._pipe_thread is not None
            or self.session is None
        ):
            self.play_button.setChecked(False)
            return
        step = max(1, self._play_step)
        next_frame = self.viewer.current_frame + step
        if next_frame >= self.viewer.n_frames:
            # Snap to the last frame so the user always sees the end
            # of the sequence even when ``step`` would overshoot — then
            # pause. Click Play again to wrap to frame 0 (the toggle
            # handler handles the wrap).
            last = self.viewer.n_frames - 1
            if self.viewer.current_frame < last:
                self.viewer.set_frame(last)
            self.play_button.setChecked(False)
            return
        self.viewer.set_frame(next_frame)

    def _pause_playback(self) -> None:
        """Stop the playback timer if it's running.

        Called from session-swap / file-close / pipeline-start paths
        so playback doesn't tick into a torn-down viewer or contend
        with a pipeline write. Safe to call when playback is already
        stopped.
        """
        if self.play_button.isChecked():
            self.play_button.setChecked(False)

    # -- Background prefetch worker ---------------------------------------

    def _ensure_prefetch_worker(self) -> None:
        """Spawn the prefetch worker + thread on first use. Idempotent.

        Lazy spawn keeps startup fast for users who only ever view
        single-frame files (no playback, no prefetch worth running).
        Once spawned, the worker survives across entry switches —
        each new entry triggers ``configure()`` rather than a
        rebuild.
        """
        if self._prefetch_worker is not None:
            return
        self._prefetch_thread = QThread(self)
        self._prefetch_worker = PrefetchWorker()
        self._prefetch_worker.moveToThread(self._prefetch_thread)
        # Cross-thread wiring. configure / update_state / release run
        # on the worker's thread via queued connections; prefetched
        # signal delivers back to the GUI thread.
        self._prefetchConfigure.connect(
            self._prefetch_worker.configure, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchUpdate.connect(
            self._prefetch_worker.update_state, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchRelease.connect(
            self._prefetch_worker.release, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_worker.prefetched.connect(
            self._on_prefetched, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_thread.start()

    def _configure_prefetch_for_active_entry(self) -> None:
        """Tell the worker about the active entry's shape + LRU size.

        Called after every successful entry load (in
        ``_load_entry_into_viewer``) and after the silx-reattach
        path completes a pipeline run. No-op for single-frame
        stacks (nothing to prefetch) and for raw sessions
        (FrameSource isn't used).
        """
        if (
            self.session is None
            or self.session.kind != "nexus"
            or self.viewer._frame_source is None
            or self.viewer.n_frames <= 1
        ):
            # Release the worker if we have one — no work on idle.
            if self._prefetch_worker is not None:
                self._prefetchRelease.emit()
            return
        self._ensure_prefetch_worker()
        fs = self.viewer._frame_source
        # Sliding-window size = LRU - 1 so the prefetcher can never
        # evict frames the play-head still needs to reach.
        window = max(1, fs.cart_lru_size - 1)
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._prefetchConfigure.emit(
            str(self.session.temp_path), entry, fs.n_frames, window,
        )
        # Start in paused state — the worker only ticks during
        # active playback. The play-button toggle (and any frame
        # change while playing) will flip ``active=True`` via
        # _prefetchUpdate.
        self._prefetchUpdate.emit(self.viewer.current_frame, False, 1)

    @Slot(int, object, object, object, object)
    def _on_prefetched(
        self,
        idx: int,
        cart: object,
        polar: object,
        radius: object,
        angle: object,
    ) -> None:
        """Deposit a prefetched frame into the active FrameSource's LRU.

        Runs on the GUI thread (queued from the worker). The
        FrameSource's LRUs are touched only here and from the
        synchronous ``get_cartesian`` / ``get_polar`` paths, both
        of which live on the GUI thread — so no locking is needed.

        Drops the result silently if the FrameSource has been
        released (post-pipeline detach), since a stale signal in
        flight should not warm a closed cache.
        """
        fs = self.viewer._frame_source
        if fs is None or not fs.is_open:
            return
        try:
            fs.warm_cartesian(int(idx), cart)        # type: ignore[arg-type]
            fs.warm_polar(int(idx), polar, radius, angle)  # type: ignore[arg-type]
        except Exception:
            # Defensive — a stale signal during teardown shouldn't
            # propagate.
            logger.debug("suppressed exception in MainWindow._on_prefetched", exc_info=True)
            pass

    def _refresh_frame_slider(self) -> None:
        """Match the slider's range + value to the active stack's
        frame count. Called after every show_stack — covers entry
        switches, file opens, and pipeline-finished reloads.
        Single-frame stacks hide the whole nav cluster.
        """
        n = self.viewer.n_frames
        cur = self.viewer.current_frame
        self.frame_slider.blockSignals(True)
        try:
            if n <= 1:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(0)
                self.frame_slider.setValue(0)
            else:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(n - 1)
                self.frame_slider.setValue(int(cur))
        finally:
            self.frame_slider.blockSignals(False)
        with QSignalBlocker(self.frame_spin):
            self.frame_spin.setRange(0, max(0, n - 1))
            self.frame_spin.setValue(int(cur) if n > 1 else 0)
        self.frame_label.setText(self._frame_label_text(cur))
        self._set_frame_slider_visible(n > 1)
        self._refresh_frame_nav_enabled()
        self._sync_scan_tracking_visibility()

    def _set_frame_slider_visible(self, visible: bool) -> None:
        """Show or hide the toolbar's frame-navigation cluster.

        With the controls living in the image-viewer toolbar (no
        form / no row container), each widget is toggled directly.

        The strip's layout is then re-run: hiding all six empties their
        whole cluster, and a wrapping layout has to be told, or it keeps
        placing the controls that are left where they used to be. See
        ``GIWAXSImageViewer.refresh_toolbar_layout``.
        """
        for w in (
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.frame_slider,
            self.frame_spin,
            self.frame_label,
        ):
            w.setVisible(visible)
        self.viewer.refresh_toolbar_layout()

    def _refresh_frame_nav_enabled(self) -> None:
        """Disable prev/next at the boundaries so the user can see
        they've hit the start / end of the stack."""
        n = self.viewer.n_frames
        cur = self.viewer.current_frame
        self.prev_frame_button.setEnabled(n > 1 and cur > 0)
        self.next_frame_button.setEnabled(n > 1 and cur < n - 1)

    # Minimum gap between successive prev/next frame steps. The OS
    # keyboard auto-repeat fires at ~30 events/sec (~33 ms apart);
    # without throttling, set_frame calls pile up faster than the
    # viewer can render and a held arrow key leaves a backlog that
    # keeps advancing after the user releases. 80 ms matches the
    # existing toolbar prev/next button autoRepeatInterval — fast
    # enough to feel responsive, slow enough to give each frame
    # room to render.
    _FRAME_STEP_THROTTLE_S = 0.08

    def _frame_step_throttle_ok(self) -> bool:
        """Time-throttle: drop step requests that arrive within
        ``_FRAME_STEP_THROTTLE_S`` of the previous one.

        Single clicks are always >80 ms apart so they're never
        affected; only OS keyboard auto-repeat (~33 ms cadence) and
        Qt toolbar auto-repeat get suppressed below the throttle.
        """
        now = time.monotonic()
        last = getattr(self, "_last_frame_step_t", 0.0)
        if (now - last) < self._FRAME_STEP_THROTTLE_S:
            return False
        self._last_frame_step_t = now
        return True

    def _step_frame(self, direction: int) -> None:
        """Shared prev/next step path with both a time-throttle and
        a queue-drain gate.

        Two complementary mechanisms:

        1. ``_frame_step_throttle_ok`` enforces a minimum gap
           between accepted steps. Protects against fast renders +
           OS autorepeat (drops the bulk of repeat events).

        2. The ``_frame_step_in_flight`` flag + ``processEvents()``
           drain protects against **slow** renders, where the
           synchronous ``set_frame`` call blocks the event loop
           long enough for the OS to enqueue multiple keypress
           events. After the render completes we explicitly drain
           the queue while the flag is still set, so the queued
           auto-repeats hit the flag, see "busy", and drop. Without
           this drain, holding a key on a slow stack continues
           advancing for ~1 s after the user releases.
        """
        if getattr(self, "_frame_step_in_flight", False):
            return
        if not self._frame_step_throttle_ok():
            return
        self._frame_step_in_flight = True
        try:
            cur = self.viewer.current_frame
            target = cur + direction
            if 0 <= target < self.viewer.n_frames:
                self.viewer.set_frame(target)
            # Drain OS auto-repeats that piled up during the
            # synchronous render. They'll recurse into this method,
            # see the in-flight flag set, and drop. This is the
            # one place in the GUI where ``processEvents`` from
            # inside a slot is required for correctness — see the
            # docstring above.
            QApplication.processEvents()
        finally:
            self._frame_step_in_flight = False

    def _on_prev_frame_clicked(self) -> None:
        self._step_frame(-1)

    def _on_next_frame_clicked(self) -> None:
        self._step_frame(+1)

    def _on_first_frame_shortcut(self) -> None:
        """Jump to frame 0. Bound to Home. Not throttled — these
        are single-target jumps, not repeated steps."""
        if self.viewer.n_frames > 1 and self.viewer.current_frame != 0:
            self.viewer.set_frame(0)

    def _on_last_frame_shortcut(self) -> None:
        """Jump to the last frame. Bound to End. Not throttled."""
        n = self.viewer.n_frames
        last = n - 1
        if n > 1 and self.viewer.current_frame != last:
            self.viewer.set_frame(last)

    def _install_frame_shortcuts(self) -> None:
        """Register window-level keyboard shortcuts for frame navigation.

        Each binding is a hidden ``QAction`` on the main window with
        ``WindowShortcut`` context. Qt's normal key-event chain means
        text-input widgets (QLineEdit, QSpinBox, QDoubleSpinBox)
        consume Left / Right / Home / End for caret navigation before
        the shortcut fires — so typing in a dock field still works.
        J/K give a Vim-style fallback that doesn't collide with text
        input either (most fields are numeric).
        """
        bindings = [
            ("Prev frame", ["Left", "J"], self._on_prev_frame_clicked),
            ("Next frame", ["Right", "K"], self._on_next_frame_clicked),
            ("First frame", ["Home"], self._on_first_frame_shortcut),
            ("Last frame", ["End"], self._on_last_frame_shortcut),
        ]
        self._frame_shortcut_actions = []
        for name, keys, slot in bindings:
            action = QAction(name, self)
            action.setShortcuts([QKeySequence(k) for k in keys])
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(slot)
            self.addAction(action)
            self._frame_shortcut_actions.append(action)

    def _frame_label_text(self, idx: int) -> str:
        """Suffix completing the frame spinbox: "/ max". The index
        itself lives in the editable ``frame_spin``; ``idx`` is kept in
        the signature for the existing call sites."""
        n = self.viewer.n_frames
        if n <= 1:
            return "—"
        return f"/ {n - 1}"

    def _safe_selected_h5_nodes(self) -> list:
        """Return ``selectedH5Nodes`` results, swallowing silx model errors.

        Under certain races (mid-pipeline detach/reattach, freshly
        inserted file with not-yet-resolved h5py state), silx's tree
        model can raise on attribute lookup deep inside the proxy
        chain. Qt then re-fires the call, producing a stack-busting
        recursion that brings down the click handler. We catch
        anything from that path here so a single bad click can't
        wedge the GUI.
        """
        try:
            return list(self.tree.selectedH5Nodes())
        except (RecursionError, RuntimeError, KeyError, OSError) as exc:
            self.pipeline_panel.append_log(
                f"WARN — silx tree query failed ({type(exc).__name__}); "
                "rebuilding the file browser"
            )
            # Drastic but reliable: tear the tree down and rebuild it
            # from the live session list. Any orphan / half-loaded
            # silx items get dropped in the process.
            self._detach_silx_tree()
            self._reattach_silx_tree()
            return []

    def _selected_image_node(self) -> _ImageFileNode | None:
        """The selected ``_ImageFileNode`` browser row, if any.

        silx's ``selectedH5Nodes`` yields only real ``Hdf5Item``s, so
        display-only image rows need this direct item-role probe.
        """
        try:
            for index in self.tree.selectedIndexes():
                if index.column() != 0:
                    continue
                item = self.tree.model().data(
                    index, Hdf5TreeModel.H5PY_ITEM_ROLE
                )
                if isinstance(item, _ImageFileNode):
                    return item
        except Exception:
            logger.debug(
                "suppressed exception in _selected_image_node", exc_info=True
            )
        return None

    def _activate_image_node(self, node: _ImageFileNode) -> None:
        """Show the clicked image row's file in the viewer.

        Mirrors what clicking an ``entry_*`` group does for NeXus files:
        promote the owning session if needed, then select the image's
        entry in the combo (the combo change triggers the async load).
        """
        session = self._session_for_path(node.image_path)
        if session is not None and session is not self._active_session:
            self._set_active_session(session)
        label = self._raw_entry_label_by_path.get(node.image_path)
        if label is None:
            return
        if self.entry_combo.currentText() != label:
            self.entry_combo.setCurrentText(label)

    def _on_tree_selection_changed(self, *_: object) -> None:
        # A selection the app just put back after rebuilding the tree is
        # not a click: reacting to it re-enters the rebuild and leaves
        # the caller holding an index into a model that no longer has
        # one. See ``_restore_tree_state``.
        if getattr(self, "_restoring_tree_selection", False):
            return
        # Several rows picked (Shift/Ctrl-click, Ctrl+A) is a batch for
        # Delete, not a click: leave the viewer where it is.
        picked = sum(
            1 for i in self.tree.selectedIndexes() if i.column() == 0
        )
        if picked > 1:
            return
        image_node = self._selected_image_node()
        if image_node is not None:
            self._activate_image_node(image_node)
            self._set_or_defer_data_node(image_node)
            self._set_or_defer_structure_node(image_node)
            return
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        # Switch active session + entry FIRST, before touching the Data
        # viewer. silx's ``DataViewerFrame.setData`` eagerly resolves the
        # clicked entry's external-linked NXdata signal to pick a view —
        # slow enough on a big multi-scan master to freeze the GUI, and
        # because it used to run first it aborted the entry switch
        # entirely (the file-browser click appeared to "do nothing").
        # Doing the cheap entry switch first makes a tree click behave
        # exactly like the Entry combo.
        #
        # Multiple files may be loaded — clicking into a different file's
        # subtree promotes that file to the active session so the entry
        # combo, image viewer, and per-file actions follow the user's
        # focus without an extra click.
        self._activate_session_for_node(node)
        # Click into entry_X anywhere → switch the image tab to that
        # entry. The entry-combo signal already triggers the viewer
        # reload, so we just push the new value here.
        self._activate_entry_for_node(node)
        # Feed the Data viewer only when its tab is actually visible;
        # otherwise stash the node and render it lazily on tab switch, so
        # a single click on the Image tab never pays the external-link
        # resolve for a dataset the user can't even see.
        self._set_or_defer_data_node(node)
        # Same deferral for the Structure tab: describing a node costs a
        # small read, and there is no reason to pay it while the tab that
        # would show the result is hidden.
        self._set_or_defer_structure_node(node)

    def _on_tree_activated(self, *_: object) -> None:
        image_node = self._selected_image_node()
        if image_node is not None:
            self._activate_image_node(image_node)
            self.tabs.setCurrentWidget(self.data_viewer)
            self._set_data_node(image_node)
            return
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        self._activate_session_for_node(node)
        self._activate_entry_for_node(node)
        # Double-click means "inspect this in the Data tab": switch to it
        # and render now — the user explicitly asked to see the data, so
        # the resolve cost is expected here (and only here).
        self.tabs.setCurrentWidget(self.data_viewer)
        self._set_data_node(node)

    def _set_or_defer_data_node(self, node) -> None:
        """Render ``node`` in the Data viewer if that tab is showing, else
        remember it and render on the next switch to the Data tab."""
        if self.tabs.currentWidget() is self.data_viewer:
            self._set_data_node(node)
        else:
            self._pending_data_node = node

    def _set_data_node(self, node) -> None:
        """Push ``node`` into the silx Data viewer, guarded so a slow / bad
        external-link resolve can't propagate out of a tree click.

        Display-only image rows carry no h5 object — decode the file
        now (one frame, on demand) and hand the array to the viewer.
        """
        self._pending_data_node = None
        try:
            if isinstance(node, _ImageFileNode):
                import fabio

                file_model._quiet_fabio()
                self.data_viewer.setData(
                    np.asarray(fabio.open(node.image_path).data)
                )
            else:
                self.data_viewer.setData(node)
        except Exception:
            logger.debug(
                "suppressed exception in MainWindow._set_data_node", exc_info=True
            )

    def _on_main_tab_changed(self, _index: int) -> None:
        """Render whatever was clicked while the newly-shown tab was hidden.

        Both the Data viewer and the Structure panel defer a tree click
        they cannot display, to keep Image-tab clicks from paying the
        external-link resolve (Data) or a metadata read (Structure)."""
        if (
            self.tabs.currentWidget() is self.data_viewer
            and self._pending_data_node is not None
        ):
            self._set_data_node(self._pending_data_node)
        elif (
            self.tabs.currentWidget() is self.structure_panel
            and self._pending_structure_node is not None
        ):
            self._render_structure_node(self._pending_structure_node)
        # The Structure tab needs the file browser and nothing else, so
        # the side and bottom docks fold away while it is up and come
        # back exactly as they were on the way out.
        self._sync_structure_docks()

    def _activate_entry_for_node(self, node) -> None:
        """If the clicked node is inside an ``entry_*`` group, switch the
        entry combo (and therefore the image viewer) to that entry.

        No-ops for clicks on the file root or on nodes outside any entry
        group (e.g. top-level metadata). Also no-ops if the entry isn't
        in the combo — that would mean it's a non-q entry filtered out
        by ``list_entries``, where the viewer can't render anything
        useful anyway.

        Raw sessions take their own resolver: their combo holds detector-
        dataset candidates (``file::dataset/path``), not ``entry_*``
        groups. ``_activate_session_for_node`` ran first, so the active
        session's kind matches the clicked file.
        """
        if self.session is not None and self.session.kind == "raw":
            self._activate_raw_entry_for_node(node)
            return
        entry = self._node_entry_name(node)
        if entry is None:
            return
        if self.entry_combo.findText(entry) < 0:
            return
        if self.entry_combo.currentText() == entry:
            return
        # Triggers _on_entry_changed → _load_entry_into_viewer.
        self.entry_combo.setCurrentText(entry)

    def _activate_raw_entry_for_node(self, node) -> None:
        """Map a clicked tree node to a raw detector dataset and select
        it in the entry combo — raw files browse from the file browser
        exactly like NeXus ``entry_*`` groups (the combo change triggers
        the async ``EntryLoadWorker.load_raw``).

        A node matches a candidate when it IS the candidate's dataset or
        an ancestor group of one (clicking a scan group like ``1.1``
        selects the first candidate inside it, mirroring how clicking
        anywhere inside an ``entry_*`` group selects that entry). Clicks
        on the file root or on non-candidate nodes (axes, metadata)
        no-op.
        """
        raw_entries = getattr(self, "_raw_entries", {})
        if not raw_entries:
            return
        fname = self._node_filename(node)
        rel = self._node_h5_path(node)
        if fname is None or not rel:
            return
        try:
            target = fname.resolve()
        except OSError:
            target = fname
        for label, raw_entry in raw_entries.items():
            if not (
                raw_entry.dataset_path == rel
                or raw_entry.dataset_path.startswith(rel + "/")
            ):
                continue
            try:
                entry_file = raw_entry.file_path.resolve()
            except OSError:
                entry_file = raw_entry.file_path
            if entry_file != target:
                continue
            if self.entry_combo.currentText() != label:
                # Triggers _on_entry_changed → _load_raw_entry_into_viewer.
                self.entry_combo.setCurrentText(label)
            return

    @staticmethod
    def _node_h5_path(node) -> str | None:
        """The node's path inside its HDF5 file ('' for the file root).

        Prefers silx's ``local_name`` (the master-side path — reading it
        never resolves an external link); falls back to the h5py
        object's name for silx versions without it.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_h5_path", exc_info=True)
                continue
            if p:
                return str(p).strip("/")
        return None

    @staticmethod
    def _node_entry_name(node) -> str | None:
        """Extract the ``entry_*`` group name from a node's HDF5 path.

        silx exposes the absolute path as ``local_name`` (e.g.
        ``/entry_0000/data/img_gid_q``); we take the first component if
        it begins with ``entry_``. Returns None for nodes outside any
        entry group.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_entry_name", exc_info=True)
                continue
            if p:
                parts = str(p).lstrip("/").split("/")
                if not parts:
                    return None
                if file_model.is_entry_group_name(parts[0]):
                    return parts[0]
                # Entry named off-convention (mlgidFIT names it <sample>):
                # confirm parts[0] is an NXentry group via the node's file.
                try:
                    if parts[0] in file_model.entry_group_names(node.h5py_object.file):
                        return parts[0]
                except Exception:
                    logger.debug("suppressed exception in MainWindow._node_entry_name (nxentry)", exc_info=True)
                return None
        return None

    def _session_for_node(self, node) -> BaseSession | None:
        """Return the session whose file ``node`` was loaded from, or None.

        silx normalizes paths through the OS, so a literal ``Path`` equality
        with ``session.temp_path`` can fail when one side has a symlink,
        dotfile component, or trailing slash that the other doesn't.
        ``Path.resolve()`` collapses both sides to a canonical absolute form
        before the comparison.
        """
        fname = self._node_filename(node)
        if fname is None:
            return None
        try:
            target = fname.resolve()
        except OSError:
            target = fname
        return self._session_for_path(str(target))

    def _session_for_path(self, target: str) -> BaseSession | None:
        """Session owning the (resolved) filesystem path ``target``.

        Raw sessions match through their cached ``raw_path_strs`` set —
        a raw batch can hold thousands of files, and the previous
        per-click loop re-resolved every one of them (thousands of
        syscall chains per browser click). NeXus sessions own exactly
        one file (the temp working copy), resolved here per session.
        """
        for s in self._sessions:
            if isinstance(s, RawSession):
                if target in s.raw_path_strs:
                    return s
            else:
                candidate = s.temp_path
                try:
                    candidate = candidate.resolve()
                except OSError:
                    pass
                if str(candidate) == target:
                    return s
        return None

    def _activate_session_for_node(self, node) -> None:
        """If ``node`` lives in a non-active session's file, swap active.

        Previously a path mismatch silently left the wrong session active
        and the pipeline ran on the most recently opened file regardless
        of which tree the user clicked. ``_session_for_node`` does the
        resolve-and-match so the swap follows the clicked tree.
        """
        s = self._session_for_node(node)
        if s is not None and s is not self._active_session:
            self._set_active_session(s)

    def _selected_file_targets(
        self,
    ) -> tuple[dict[RawSession, list[str]], list[BaseSession]]:
        """What the selected browser rows stand for.

        Returns the raw files to close, grouped by their raw session, and
        the converted (NeXus) sessions whose file a selected row belongs
        to. Several rows of one converted file count once.
        """
        raw_paths: list[str] = []
        try:
            for index in self.tree.selectedIndexes():
                if index.column() != 0:
                    continue
                item = self.tree.model().data(
                    index, Hdf5TreeModel.H5PY_ITEM_ROLE
                )
                if isinstance(item, _ImageFileNode):
                    raw_paths.append(item.image_path)
        except Exception:
            logger.debug(
                "suppressed exception in _selected_file_targets", exc_info=True
            )
        nexus: list[BaseSession] = []
        for node in self._safe_selected_h5_nodes():
            fname = self._node_filename(node)
            if fname is None:
                continue
            try:
                fname = fname.resolve()
            except OSError:
                pass
            session = self._session_for_path(str(fname))
            if isinstance(session, RawSession):
                raw_paths.append(str(fname))
            elif session is not None and session not in nexus:
                nexus.append(session)
        raw: dict[RawSession, list[str]] = {}
        for p in raw_paths:
            session = self._session_for_path(p)
            if isinstance(session, RawSession):
                paths = raw.setdefault(session, [])
                if p not in paths:
                    paths.append(p)
        return raw, nexus

    def _remove_selected_file_from_browser(self) -> None:
        """Close the file(s) the current tree selection belongs to.

        Wired to the file browser's Delete key (``deleteFileRequested``).
        Every selected file closes: a raw row closes just its own file and
        leaves the rest of the raw batch (and its Selection rows) in place;
        any row of a converted file closes that file. Unsaved work in the
        converted files is asked about first, through the same
        discard-confirm as File → Close (Ctrl+W), and Cancel on any of
        them closes nothing. The active session is closed last so the
        window does not switch through sessions about to go. No-ops if the
        selection can't be mapped to a live session (e.g. a stale or
        orphan node), mirroring how a no-active-session Close no-ops.
        """
        raw, nexus = self._selected_file_targets()
        if not raw and not nexus:
            return
        for session in nexus:
            if not self._confirm_discard_changes(session):
                return
        for session, paths in raw.items():
            self._remove_raw_files(session, paths)
        active = self._active_session
        for session in sorted(nexus, key=lambda s: s is active):
            self._close_session(session)

    @staticmethod
    def _node_filename(node) -> Path | None:
        """Resolve the filesystem path of the file ``node`` was loaded from.

        silx exposes this differently across versions — fall through the
        known accessors and give up silently if nothing answers.
        """
        for getter in (
            lambda n: getattr(n, "local_filename", None),
            lambda n: n.h5py_object.file.filename,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_filename", exc_info=True)
                continue
            if p:
                return Path(p)
        return None

    def _extra_peak_datasets(self) -> tuple[str, ...]:
        """Dataset names of the user's registered extra peak lists.

        The viewer, not QSettings, is the source of truth: it holds the
        specs that actually have overlay items, so this can never ask for
        a list that would not be rendered anyway.

        EVERY path that fills a frame's overlay dict must pass these --
        the two worker prewarms as much as the on-demand read here. A
        path that omits them produces a dict with no ``list:`` keys, and
        because the caller then marks the frame loaded, the layer stays
        blank until something invalidates the frame.
        """
        return tuple(spec.dataset for spec in self.viewer.peak_list_specs())

    def _overlays_have_registered_lists(self, peaks: dict) -> bool:
        """Whether an overlay dict covers every registered extra list.

        Read off-thread overlays are trusted only when they do. The check
        is a key comparison, not a value one: ``read_peaks`` always
        returns a key per requested list (``None`` when that frame has no
        such dataset), so a missing KEY means the read never asked for it
        -- a stale worker request, or a list registered while the read
        was in flight. Falling back to a fresh read costs one small
        HDF5 navigation and only in that case.
        """
        return all(
            f"{peak_lists.KIND_PREFIX}{name}" in peaks
            for name in self._extra_peak_datasets()
        )

    def _load_frame_peaks(self, entry: str, frame: int) -> None:
        """Load + install one frame's detected/fitted/matched peaks into
        the viewer, at most once per frame per entry load.

        Peak datasets are tiny, so loading the frame the user is actually
        looking at on demand is cheap — far cheaper than the old
        load-every-frame-up-front loop, which dominated entry-switch lag.
        Reset via ``_loaded_peak_frames`` on each entry (re)load.
        """
        if self.session is None or frame < 0:
            return
        if entry != getattr(self, "_loaded_peaks_entry", None):
            return
        if frame in self._loaded_peak_frames:
            return
        # Read through the viewer's already-open FrameSource handle when it
        # matches this entry: its master handle has the entry's external
        # link resolved, so reading the tiny peak tables is an in-file
        # navigation, NOT a fresh multi-second network ``h5py.File`` open.
        # That synchronous open per frame/entry was the freeze on big
        # external-link masters (the off-thread frame read alone wasn't
        # enough — this peak read still ran on the GUI thread). Fall back to
        # open-by-path only when no live handle is available.
        src = self.viewer._frame_source
        # The user's registered extra peak lists ride the same read, so a
        # custom layer costs no extra file open on a frame scrub.
        extra = self._extra_peak_datasets()
        try:
            if src is not None and src.is_open and src.entry == entry:
                peaks, matched = src.read_frame_overlays(frame, extra)
            else:
                peaks = file_model.load_peaks(
                    self.session.temp_path, entry, frame, extra
                )
                matched = file_model.load_matched_peaks(
                    self.session.temp_path, entry, frame, peaks.get("fitted")
                )
        except Exception:
            logger.debug("suppressed exception in MainWindow._load_frame_peaks", exc_info=True)
            peaks = {kind: None for kind in OVERLAY_KINDS}
            matched = []
        self.viewer.set_peaks(frame, peaks)
        self.viewer.set_matched_structures(frame, matched)
        self._loaded_peak_frames.add(frame)

    def _load_entry_into_viewer(
        self, entry: str, *, preserve_view: bool = False
    ) -> None:
        """Load ``entry`` into the image viewer.

        ``preserve_view``: when True, the viewer keeps its current zoom and
        frame index across the reload. Used after pipeline ops and direct
        h5py edits (the underlying stack is unchanged — only peak overlays
        are different). Switching to a different entry passes False so the
        viewer auto-ranges to the new axes.
        """
        assert self.session is not None
        # Reuse a pre-warmed FrameSource (opened + frame-0 read off the GUI
        # thread by CopyWorker) when one is queued for this entry, so the
        # first render never blocks on a slow read. Otherwise load
        # synchronously — the path tests and entry switches use.
        overlays = None
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] == entry:
            self.session._prewarm = None  # type: ignore[attr-defined]
            stack = file_model.stack_from_source(prewarm[1])
            # Consume the worker-read first-frame peaks alongside the frame.
            overlays = getattr(self.session, "_prewarm_overlays", None)
            self.session._prewarm_overlays = None  # type: ignore[attr-defined]
        else:
            try:
                stack = file_model.load_entry(self.session.temp_path, entry)
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", f"Could not load {entry}: {exc}")
                return
        self._install_stack_into_viewer(
            entry, stack, preserve_view=preserve_view, overlays=overlays
        )

    def _install_stack_into_viewer(
        self, entry: str, stack, *, preserve_view: bool = False, overlays=None
    ) -> None:
        """Render an already-built ``EntryStack`` and wire up the per-entry
        viewer state (slider, peaks, polar profile, prefetch, peaks dock).

        Split out of ``_load_entry_into_viewer`` so the slow disk read (the
        ``FrameSource`` open + frame-0 read) can be done off the GUI thread
        — by ``CopyWorker`` (prewarm) or ``EntryLoadWorker`` (interactive
        switch) — and only this fast, GUI-thread install runs here. The
        viewer's ``show_stack`` releases the previous entry's FrameSource
        before adopting the new one (image_viewer.py), so this also frees
        the old handle.
        """
        self.viewer.show_stack(stack, preserve_view=preserve_view)
        # Match the slider to the new stack's frame range. preserve_view
        # already restores the prior frame index inside show_stack; we
        # only need to repopulate the slider's bounds + label here.
        self._refresh_frame_slider()
        # Peaks load lazily, per frame, not all up front. The old
        # all-frames loop did N HDF5 opens per entry switch (hundreds on a
        # many-frame entry) — a big part of the switch lag. Load only the
        # frame the viewer landed on; other frames load on demand in
        # ``_on_viewer_frame_changed`` via ``_load_frame_peaks``. The
        # ``_peaks_loaded`` set is reset here so a post-pipeline reload
        # (preserve_view=True) re-reads fresh peaks as frames are revisited.
        self._loaded_peaks_entry = entry
        self._loaded_peak_frames = set()
        # Phase-tracking results are per-entry: another entry's tracks
        # reference different frames/ids. Same-entry overlay refreshes
        # (peak edits, pipeline reattach) keep them.
        if self._scan_track_entry is not None and entry != self._scan_track_entry:
            self._invalidate_scan_tracks()
        if not pipeline.is_mlgidbase_available():
            self.scan_tracking_panel.set_scan_available(
                False,
                reason="Peak tracking needs the [pipeline] backends "
                       "(mlgidbase) installed.",
            )
        else:
            self.scan_tracking_panel.set_scan_available(
                self.viewer.n_frames > 1,
                reason="Peak tracking needs a multi-frame entry.",
            )
        if self._phase_views_window is not None and self.session is not None:
            self._phase_views_window.set_context(
                self.session.temp_path, entry, self.viewer.n_frames
            )
        # Tracking the file already carries (saved by mlgidbase) shows up
        # without running anything; read off the GUI thread.
        self._maybe_autoload_saved_tracking(entry)
        cur = self.viewer.current_frame
        if (
            overlays is not None and overlays[0] == cur
            and self._overlays_have_registered_lists(overlays[1])
        ):
            # Peaks for the landed frame were read off-thread (worker) — install
            # them directly so the GUI does no SFTP read here. Other frames
            # still load on demand via ``_load_frame_peaks``.
            self.viewer.set_peaks(cur, overlays[1])
            self.viewer.set_matched_structures(cur, overlays[2])
            self._loaded_peak_frames.add(cur)
        else:
            self._load_frame_peaks(entry, cur)
        # Initial panel state for whichever frame the viewer is showing now.
        self._refresh_matched_panel(
            self.viewer.current_frame,
            self.viewer.matched_structures(self.viewer.current_frame),
        )
        # Hand the polar transform to the profile viewer. After the
        # lazy-loading milestone ``viewer.polar_data()`` returns a
        # ``(_LazyPolarStack, radius, angle)`` tuple — frames are
        # resampled on demand inside the FrameSource so this no longer
        # forces an eager precompute of the full polar stack. The
        # profile viewer indexes the wrapper by frame; the cursor
        # readout uses tuple indexing for single-pixel lookups.
        polar = self.viewer.polar_data()
        if polar is not None:
            self.profile_viewer.set_polar_stack(*polar)
        # Configure the prefetch worker for the new entry. Single-
        # frame stacks short-circuit inside the helper; multi-frame
        # stacks spawn the worker (if not already) and reset its
        # _done set so the next Play press starts filling from
        # frame current+1 onward.
        self._configure_prefetch_for_active_entry()
        # Repopulate the Peaks dock with the new entry's peaks. The
        # viewer's frameChanged path also drives this slot, but the
        # initial load may finish on the *same* frame index as the
        # previous session — in which case no frameChanged fires and
        # the panel would otherwise keep stale rows.
        self._refresh_peaks_table()

    # -- Async entry switching (off-GUI-thread frame read) --

    def _ensure_entry_load_worker(self) -> None:
        """Spawn the persistent EntryLoadWorker + thread on first use.

        Idempotent; mirrors ``_ensure_prefetch_worker``. The worker reads
        an entry's first frame off the GUI thread and hands back a ready
        ``FrameSource`` so a switch never blocks on a slow network read.
        """
        if self._entry_load_worker is not None:
            return
        self._entry_load_thread = QThread(self)
        self._entry_load_worker = EntryLoadWorker()
        self._entry_load_worker.moveToThread(self._entry_load_thread)
        self._entryLoadRequest.connect(
            self._entry_load_worker.load, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_worker.loaded.connect(
            self._on_entry_loaded, Qt.ConnectionType.QueuedConnection,
        )
        # Raw-mode pair: same worker/thread, separate slot + result signal
        # (a raw load hands back a LazyRawStack, not a FrameSource).
        self._rawLoadRequest.connect(
            self._entry_load_worker.load_raw, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_worker.raw_loaded.connect(
            self._on_raw_entry_loaded, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_thread.start()

    def _load_entry_async(self, entry: str) -> None:
        """Switch to ``entry`` without blocking the GUI.

        A pre-warm queued for this entry (the just-opened first entry) is
        consumed synchronously — it is already in memory, so there is
        nothing to wait for. Otherwise the read is dispatched to the
        EntryLoadWorker and the bottom-left bar marches until it returns;
        ``_entry_req_id`` is bumped so a superseded switch's late result is
        dropped.
        """
        if self.session is None:
            return
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] == entry:
            # Already warmed (open path) — install immediately, no worker.
            self._load_entry_into_viewer(entry)
            self._update_status_frame()
            return
        self._entry_req_id += 1
        self._ensure_entry_load_worker()
        self._sb_open_label.setText(f"Loading {entry}…")
        self._sb_open_label.show()
        self._sb_open_bar.setRange(0, 0)  # indeterminate: one read, no stages
        self._sb_open_bar.show()
        self._entryLoadRequest.emit(
            str(self.session.temp_path), entry, self._entry_req_id,
            self._extra_peak_datasets(),
        )

    def _on_entry_loaded(self, request_id: int, entry: str, source, overlays=None) -> None:
        """Install the entry whose frame + peaks the worker just read, unless
        a newer switch has superseded it (then the source is released)."""
        if request_id != self._entry_req_id:
            # Superseded by a later switch — drop the stale source.
            if source is not None:
                try:
                    source.release()
                except Exception:
                    logger.debug("suppressed stale-source release", exc_info=True)
            return
        self._dismiss_open_progress()
        if source is None:
            QMessageBox.warning(self, "Load failed", f"Could not load {entry}")
            return
        # The active session may have changed between dispatch and arrival
        # (file closed / switched); guard so we never install a source from
        # a stale file. ``_set_active_session`` bumps the id, so a mismatch
        # here means this result is orphaned.
        if self.session is None or self.session.kind != "nexus":
            try:
                source.release()
            except Exception:
                logger.debug("suppressed orphan-source release", exc_info=True)
            return
        stack = file_model.stack_from_source(source)
        self._install_stack_into_viewer(
            entry, stack, preserve_view=False, overlays=overlays
        )
        self._update_status_frame()
