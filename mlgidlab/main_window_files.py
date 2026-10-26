"""Open/import/save/close actions, session lifecycle and activation, the chunked tree-insert queue and the silx detach/reattach dance.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
import time
from PySide6.QtCore import (
    QEventLoop,
    QMetaObject,
    QThread,
    Qt,
)
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)
from contextlib import contextmanager
from mlgidlab import file_dialogs, file_model
from mlgidlab.browser_widgets import _MlgidHdf5TreeModel
from mlgidlab.main_window_constants import (
    APP_NAME,
    NEXUS_FILTER,
    OPEN_FILTER,
)
from mlgidlab.session import NexusSession, RawSession
from mlgidlab.widgets import make_progress_dialog
from mlgidlab.workers import CopyWorker, ImportWorker, stop_worker_thread
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def _natural_key(path: Path):
    """Sort key so ``img_2`` precedes ``img_10`` (digits compared numerically)."""
    import re

    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", path.name)
    ]


class FilesMixin:

    # -- Actions --

    def _action_open(self) -> None:
        """Unified open: pick HDF5 files and auto-classify each as NeXus or raw.

        Multi-select is supported. Each picked file is classified by
        content (not extension) inside ``_open_paths`` — NeXus files
        stream through the per-file copy worker queue, raw files are
        bundled into a single shared ``RawSession`` matching the old
        Open-raw bulk behaviour. Files that match neither classifier
        are reported in the log and skipped.

        Uses Qt's widget dialog instead of the platform-native one ON
        PURPOSE: native pickers preview/thumbnail image files, so a
        beamtime directory holding thousands of detector TIFFs takes
        ages to even become scrollable. The widget dialog with the
        constant-time ``_FastFileIconProvider`` lists such directories
        instantly (and follows the app theme). Both that and the
        remembered starting directory now come from ``file_dialogs``,
        which every picker in the app shares.
        """
        paths = self._pick_files("Open file(s)", OPEN_FILTER)
        if paths:
            self._open_paths(paths)

    def _pick_files(self, title: str, name_filter: str) -> list[Path]:
        """Multi-select file picker shared by Open and Import.

        Uses Qt's widget dialog instead of the platform-native one ON
        PURPOSE (see ``_action_open``); the starting directory is the
        one every other picker in the app uses.
        """
        return [
            Path(p) for p in file_dialogs.open_files(self, title, name_filter)
        ]

    def _action_import_converted(self) -> None:
        """File → Import images as converted scan… — explicit entry
        point (the escape hatch when the float-dtype auto-offer guessed
        wrong or the user starts from the menu)."""
        paths = self._pick_files(
            "Import images as converted scan",
            "Detector images (*.tif *.tiff *.cbf *.edf)",
        )
        if paths:
            self._run_import_dialog(paths)

    def _run_import_dialog(self, paths: list[Path]) -> None:
        """Collect import parameters, then write the stack off-thread.

        Frames are stacked in natural filename order (img_2 before
        img_10) regardless of selection order, matching the folder-open
        convention. The finished .h5 auto-opens through the normal open
        queue, so it lands as a regular NeXus session with the frame
        slider active.
        """
        from mlgidlab.import_dialog import ImportConvertedDialog

        paths = sorted(paths, key=_natural_key)
        dialog = ImportConvertedDialog(paths, parent=self)
        if not dialog.exec():
            return
        values = dialog.values()
        if self._import_thread is not None:
            QMessageBox.information(
                self, "Import in progress",
                "An image import is already running; wait for it to "
                "finish first.",
            )
            return

        self._import_progress = make_progress_dialog(
            self,
            f"Stacking {len(paths)} image(s) into "
            f"{values['out_path'].name}…",
            title=APP_NAME, maximum=len(paths),
        )

        self._import_thread = QThread(self)
        self._import_worker = ImportWorker(
            paths,
            values["out_path"],
            values["entry_name"],
            values["qxy_range"],
            values["qz_range"],
            values["ai"],
            values["flip_vertical"],
            values["wavelength_A"],
        )
        self._import_worker.moveToThread(self._import_thread)
        self._import_thread.started.connect(self._import_worker.run)
        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_thread.start()

    def _on_import_progress(self, done: int, total: int) -> None:
        if self._import_progress is None:
            return
        self._import_progress.setMaximum(max(total, 1))
        self._import_progress.setValue(done)

    def _on_import_finished(
        self, out_path: Path | None, error: Exception | None
    ) -> None:
        self._import_thread, self._import_worker = stop_worker_thread(
            self._import_thread, self._import_worker
        )
        if self._import_progress is not None:
            self._import_progress.close()
            self._import_progress.deleteLater()
            self._import_progress = None
        if error is not None:
            self.conversion_panel.append_log(f"Import failed: {error}")
            QMessageBox.critical(self, "Import failed", str(error))
            return
        if out_path is None:
            return
        self.conversion_panel.append_log(
            f"Imported converted image stack → {out_path}"
        )
        # Same auto-open flow as a finished conversion: the file opens
        # as a regular NeXus session (temp working copy + frame slider).
        self._open_queue.append(Path(out_path))
        self._process_open_queue()

    def _offer_import_for_float_images(self, paths: list[Path]) -> bool:
        """Offer the import flow when an image batch looks converted.

        Raw detector frames are integer counts; interpolated q-space
        maps are float-valued — one cheap decode of the FIRST file
        decides. Returns True when the batch was consumed here (import
        started or user cancelled), False to continue opening as raw.
        A wrong guess costs one click either way; the File menu action
        covers float-blind sources.
        """
        try:
            import fabio

            file_model._quiet_fabio()
            dtype = np.asarray(fabio.open(str(paths[0])).data).dtype
        except Exception:
            logger.debug("suppressed dtype probe in import offer", exc_info=True)
            return False
        if dtype.kind != "f":
            return False
        box = QMessageBox(self)
        box.setWindowTitle("Import as converted scan?")
        box.setText(
            f"The {len(paths)} selected image file(s) hold floating-"
            "point pixels — typically already-converted q-space maps, "
            "not raw detector counts.\n\nImport them as ONE scan "
            "(N frames, saved to a new .h5 file), or open them as raw "
            "detector images?"
        )
        import_btn = box.addButton(
            "Import as one scan…", QMessageBox.ButtonRole.AcceptRole
        )
        raw_btn = box.addButton(
            "Open as raw images", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is import_btn:
            self._run_import_dialog(paths)
            return True
        if clicked is raw_btn:
            return False
        return True

    def _open_paths(self, paths: list[Path]) -> None:
        """Open a mixed batch of NeXus + raw files, classifying each in the
        background.

        The Open click is instant: existing paths are queued and
        processed immediately (with the bottom-left loading bar).
        Standalone images classify inline (extension check, see
        ``_process_open_queue``); every other file gets a ``CopyWorker``
        that classifies it off the GUI thread and, for NeXus, copies +
        pre-warms it. NeXus files install as they finish; raw and
        unclassifiable files are collected and handled once the queue
        drains (``_finalize_open_batch`` — raw bundles into one
        ``RawSession`` so the Conversion panel applies one config to the
        batch). Used by both File → Open and the drag-and-drop handler.
        """
        queued = False
        for p in paths:
            if p.is_dir():
                # A dropped/opened folder contributes its fabio images in
                # natural filename order (so img_2 precedes img_10); they
                # bundle into one raw session with one 1-frame entry per
                # image. Folders without any image files are rejected.
                imgs = sorted(
                    (
                        q
                        for q in p.iterdir()
                        if q.is_file() and file_model.is_fabio_image(q)
                    ),
                    key=_natural_key,
                )
                if imgs:
                    self._open_queue.extend(imgs)
                    queued = True
                else:
                    self._pending_rejected.append(p)
            elif p.is_file():
                self._open_queue.append(p)
                queued = True
            else:
                self._pending_rejected.append(p)
        if queued:
            self._process_open_queue()
        elif self._pending_rejected:
            # Nothing to load (no worker will run) — flush the rejects now.
            self._finalize_open_batch()

    def _refresh_tree_raw_paths(self) -> None:
        """Push the active set of raw filesystem paths into the tree model.

        Called whenever the session list changes so the file browser's
        custom raw-icon stays accurate. NeXus sessions don't need to be
        listed — anything the model hasn't been told about as raw
        renders with the default NeXus icon.
        """
        model = self.tree.findHdf5TreeModel()
        if not isinstance(model, _MlgidHdf5TreeModel):
            return
        raw_paths: set[str] = set()
        for s in self._sessions:
            if s.kind == "raw" and isinstance(s, RawSession):
                for p in s.raw_paths:
                    raw_paths.add(str(p))
        model.set_raw_paths(raw_paths)

    def _process_open_queue(self) -> None:
        """Kick off the next queued open if no copy is in flight.

        Standalone image files (TIFF/CBF/EDF) classify right here on the
        GUI thread: the check is the same pure extension test CopyWorker
        would run (workers.py short-circuits them before any I/O), so
        spinning up a QThread per image only added churn — at 1000
        images the spawn/teardown cycles alone took over a second.
        Everything else still classifies off the GUI thread in a
        CopyWorker. Batch order is preserved: image runs drain inline,
        and each worker finish re-enters this method for the next run.

        When the queue is exhausted the batch is finalized (raw files
        bundled, rejects logged, loading bar hidden).
        """
        if self._thread is not None:
            return
        while self._open_queue and file_model.is_fabio_image(self._open_queue[0]):
            self._pending_raw_paths.append(self._open_queue.pop(0))
        if not self._open_queue:
            self._finalize_open_batch()
            return
        self._open_path(self._open_queue.pop(0))

    def _finalize_open_batch(self) -> None:
        """Once the open queue drains: bundle any raw files into one
        ``RawSession``, log unclassifiable files, and hide the loading bar.
        The detector-dataset lists CopyWorker found per file ride along on
        the session (``_raw_entries_cache``) so activation reads metadata
        the worker already has instead of re-walking the files."""
        if self._pending_rejected:
            self.pipeline_panel.append_log(
                "Could not classify (no q-signal entries, no raw 3D "
                "detector datasets): "
                + ", ".join(str(p) for p in self._pending_rejected)
            )
            self._pending_rejected = []
        # Dismiss BEFORE bundling: raw activation dispatches an async
        # first-entry load that re-shows the bar ("Loading <entry>…"),
        # and ``_on_raw_entry_loaded`` dismisses it when the frame lands.
        # Dismissing last would hide that in-flight indicator.
        self._dismiss_open_progress()
        if self._pending_raw_paths:
            raw_paths = self._pending_raw_paths
            self._pending_raw_paths = []
            entry_cache = self._pending_raw_entry_cache
            self._pending_raw_entry_cache = {}
            # A pure-image batch with float pixels is most likely
            # already-converted q-space maps — offer the import-as-scan
            # flow before committing to a raw session (user decision
            # 2026-07-30: auto-detect via dtype with a one-click out).
            if all(file_model.is_fabio_image(p) for p in raw_paths):
                if self._offer_import_for_float_images(raw_paths):
                    return
            # Raw files dropped one at a time join the existing raw batch:
            # a session per drop left the Conversion panel showing only
            # the newest file.
            target = self._raw_session_to_extend()
            if target is not None:
                self._extend_raw_session(target, raw_paths, entry_cache)
                return
            try:
                session = RawSession.open(raw_paths)
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))
            else:
                session._raw_entries_cache = entry_cache  # type: ignore[attr-defined]
                # Chunked, not a synchronous loop: rows appear in the
                # browser progressively while the first entry already
                # renders (see _queue_tree_inserts).
                self._queue_tree_inserts([str(p) for p in session.raw_paths])
                self._sessions.append(session)
                self._set_active_session(session)
                self._add_recent_files(session.raw_paths, "raw")
                self._refresh_tree_raw_paths()

    def _raw_session_to_extend(self) -> RawSession | None:
        """The raw session newly opened raw files should join, if any.

        The active one when it is raw, else the most recent raw session
        (a NeXus file may be in front while a raw batch waits behind it).
        """
        active = self._active_session
        if isinstance(active, RawSession):
            return active
        for s in reversed(self._sessions):
            if isinstance(s, RawSession):
                return s
        return None

    def _extend_raw_session(
        self,
        target: RawSession,
        raw_paths: list[Path],
        entry_cache: dict[str, list],
    ) -> None:
        """Append ``raw_paths`` to ``target`` and show them.

        Files already in the batch are skipped. An active target keeps its
        Conversion rows and check state and gains only the new rows; a
        target behind a NeXus session is activated, which rebuilds the
        views from its full file list.
        """
        try:
            added = target.add_paths(raw_paths)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        if not added:
            self.conversion_panel.append_log(
                "Already open: " + ", ".join(p.name for p in raw_paths)
            )
            return
        cache = getattr(target, "_raw_entries_cache", None) or {}
        cache.update(entry_cache)
        target._raw_entries_cache = cache  # type: ignore[attr-defined]
        self._queue_tree_inserts([str(p) for p in added])
        self._add_recent_files(added, "raw")
        self._refresh_tree_raw_paths()
        if target is self._active_session:
            self._append_raw_entries(added)
        else:
            self._set_active_session(target)

    def _shown_raw_file(self) -> str | None:
        """Path of the raw file whose entry the viewer is showing, if any."""
        entry = getattr(self, "_raw_entries", {}).get(
            self.entry_combo.currentText()
        )
        if entry is None:
            return None
        try:
            return str(Path(entry.file_path).resolve())
        except OSError:
            return str(entry.file_path)

    def _remove_raw_files(self, session: RawSession, paths: list[str]) -> None:
        """Close single files out of an open raw batch.

        The files leave the file browser, the entry dropdown and the
        Conversion Selection together; every other file keeps its row and
        check state. Removing the last file closes the session. When the
        file on screen goes, the viewer moves to the first remaining entry.
        """
        drop = set(paths) & set(session.raw_path_strs)
        if not drop:
            return
        if drop == set(session.raw_path_strs):
            self._close_session(session)
            return
        was_active = session is self._active_session
        if was_active:
            self.viewer.commit_pending_manual()
            self._pause_playback()
        session.remove_paths(sorted(drop))
        cache = getattr(session, "_raw_entries_cache", None)
        if cache:
            for p in drop:
                cache.pop(p, None)
        # Files still waiting for their browser row never get one.
        queued = len(self._tree_insert_queue)
        self._tree_insert_queue[:] = [
            p for p in self._tree_insert_queue if p not in drop
        ]
        self._tree_insert_total -= queued - len(self._tree_insert_queue)
        self._update_tree_insert_progress()
        model = self.tree.findHdf5TreeModel()
        if isinstance(model, _MlgidHdf5TreeModel):
            model.remove_file_rows(drop)
        self._refresh_tree_raw_paths()
        self.conversion_panel.remove_raw_inputs(drop)
        entries = getattr(self, "_raw_entries", {})
        gone = [
            label for label, e in entries.items() if str(e.file_path) in drop
        ]
        for label in gone:
            entries.pop(label, None)
        for p in drop:
            self._raw_entry_label_by_path.pop(p, None)
        if was_active:
            self._drop_raw_combo_items(gone)
        self._update_title()
        self._update_actions()

    def _drop_raw_combo_items(self, labels: list[str]) -> None:
        """Remove ``labels`` from the entry dropdown; reload if the shown
        one was among them."""
        shown_gone = self.entry_combo.currentText() in labels
        self.entry_combo.blockSignals(True)
        try:
            for label in labels:
                i = self.entry_combo.findText(label)
                if i >= 0:
                    self.entry_combo.removeItem(i)
            if shown_gone and self.entry_combo.count():
                self.entry_combo.setCurrentIndex(0)
        finally:
            self.entry_combo.blockSignals(False)
        if shown_gone:
            if self.entry_combo.count():
                self._load_raw_entry_into_viewer(self.entry_combo.currentText())
            else:
                self.viewer.clear()
                self.viewer.clear_history()
        self._update_status_entry()
        self._update_status_frame()

    def _show_open_progress(self, name: str) -> None:
        """Show the bottom-left open indicator and force it to paint NOW.

        The synchronous ``repaint`` matters on a high-latency (SFTP) mount:
        the file stat + worker startup that follow can briefly starve the
        event loop, so a plain ``show()`` (which only schedules a paint for
        the next event-loop pass) would leave the user staring at an
        unchanged window — "no progress bar". Painting the widgets here,
        while the GUI thread is still free, guarantees the indicator is
        visible the instant the open is requested.
        """
        self._sb_open_label.setText(f"Loading {name}…")
        self._sb_open_label.show()
        # Indeterminate until the worker's first real tick (a prior open
        # may have left the bar determinate at 100).
        self._sb_open_bar.setRange(0, 0)
        self._sb_open_bar.show()
        # Flush layout + paint events (NOT user input — ExcludeUserInputEvents
        # avoids re-entrancy from a stray click/key) so the just-shown
        # widgets are laid out and drawn NOW, before the blocking stat /
        # worker startup, rather than on the next event-loop pass.
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )

    def _dismiss_open_progress(self) -> None:
        """Hide the bottom-left open-progress bar once the queue drains."""
        self._sb_open_label.hide()
        self._sb_open_bar.hide()
        self._sb_open_label.setText("")

    def _action_save(self) -> None:
        self._save(confirm=True)

    def _save(self, confirm: bool, session: BaseSession | None = None) -> bool:
        """Overwrite the original from the temp. Returns True on success.

        Raw sessions have no writable temp copy — Save and Save As are
        no-ops for them. The action is also disabled in the menu, but
        guard here too in case a shortcut fires.
        """
        target = session if session is not None else self._active_session
        if target is None or target.kind != "nexus":
            return False
        assert isinstance(target, NexusSession)
        if confirm:
            reply = QMessageBox.question(
                self,
                "Save",
                f"Overwrite the original file?\n\n{target.original_path}",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Save:
                return False
        try:
            target.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self._update_title()
        return True

    def _action_save_as(self) -> None:
        if self.session is None or self.session.kind != "nexus":
            return
        assert isinstance(self.session, NexusSession)
        path, _ = file_dialogs.save_file(
            self,
            "Save As",
            NEXUS_FILTER,
            suggested_name=self.session.original_path.name,
            default_suffix="h5",
        )
        if not path:
            return
        # Release the viewer's FrameSource handle *before* the rename
        # so the file isn't open when shutil.copy2 + rename runs (matters
        # on Windows; harmless on Linux). silx is also detached so the
        # tree can be rebuilt at the new basename.
        with self._detached_silx_tree():
            try:
                self.session.save_as(Path(path))
            except Exception as exc:
                QMessageBox.critical(self, "Save As failed", str(exc))
                # The context manager still reattaches on this early
                # return, so the viewer keeps reading rather than being
                # stranded with the file detached and nothing written.
                return
            # Save As renamed the temp file to match the new basename.
            # The FrameSource was created against the old basename and
            # would otherwise fail to reopen — point it at the new path
            # before the context manager's reattach reacquires it.
            if (
                self.viewer._frame_source is not None
                and isinstance(self.session, NexusSession)
            ):
                self.viewer._frame_source.relocate(self.session.temp_path)
        self._update_title()
        # The user just wrote a new file at ``path``; surface it in the
        # recent menu so it's reopenable from the next session.
        self._add_recent_file(path, "nexus")

    def _action_close_file(self) -> None:
        """Close just the currently-active file. Other files stay open."""
        active = self._active_session
        if active is None:
            return
        if isinstance(active, RawSession):
            # A raw batch is several files: close the one on screen and
            # leave the rest (and their Selection rows) alone.
            shown = self._shown_raw_file()
            if shown is not None:
                self._remove_raw_files(active, [shown])
                return
        if not self._confirm_discard_changes(active):
            return
        self._close_session(active)

    # -- Session lifecycle --

    def _open_path(self, path: Path) -> None:
        # Additive: never tear down existing sessions. The new file simply
        # gets appended to the file browser when the copy completes.
        self._thread = QThread(self)
        self._worker = CopyWorker(path, self._extra_peak_datasets())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_open_finished)
        # Queued (cross-thread) stage ticks update the bar's LABEL while the
        # indeterminate bar marches; the worker phase is GUI-responsive so
        # the march stays smooth.
        self._worker.progress.connect(self._on_open_progress)

        # Non-modal progress: a small bar + label in the bottom-left status
        # bar (NOT a window-modal dialog). The window stays fully usable
        # while the file scans + copies and its first frame warms off the
        # GUI thread; the file appears in the browser only once the load
        # finishes (``_on_open_finished``). Replaces the old WindowModal
        # ``QProgressDialog`` that dimmed + froze the whole window — for a
        # huge external-link master that froze long enough to trip the OS
        # "force quit or wait" prompt.
        self._show_open_progress(path.name)

        self._thread.start()

    def _on_open_progress(self, percent: int, label: str) -> None:
        """Update the bottom-left open bar from a ``CopyWorker`` progress
        tick (queued from the worker thread).

        The bar starts indeterminate (``_show_open_progress``) and flips
        determinate on the first worker tick — the worker reports real
        fractions (copy bytes, raw-scan groups), so the user sees actual
        progress instead of an endless march.
        """
        if label:
            self._sb_open_label.setText(label)
        if 0 <= percent <= 100:
            if self._sb_open_bar.maximum() == 0:
                self._sb_open_bar.setRange(0, 100)
            self._sb_open_bar.setValue(percent)

    def _on_open_finished(self, result: dict) -> None:
        """Handle one ``CopyWorker`` result (classify + open done off the
        GUI thread). NeXus files install + render now (from the warm
        pre-load); raw and unclassifiable files are collected for
        ``_finalize_open_batch`` when the queue drains."""
        self._thread, self._worker = stop_worker_thread(
            self._thread, self._worker
        )
        # The loading bar spans the whole queue; it's hidden in
        # ``_finalize_open_batch`` once the queue empties.

        error = result.get("error")
        kind = result.get("kind")
        session = result.get("session")
        prewarm = result.get("prewarm")
        path = result.get("path")

        if error is not None:
            QMessageBox.critical(self, "Open failed", str(error))
            if prewarm is not None:  # opened-but-unused source would leak
                self._release_prewarm(prewarm)
        elif kind == "nexus" and session is not None:
            # ``prewarm`` is ``(first_entry, FrameSource)`` warmed off the
            # GUI thread by CopyWorker (or None). Stash it on the session so
            # the first ``_load_entry_into_viewer`` renders from the warm
            # handle instead of re-reading frame 0 on the GUI thread.
            session._prewarm = prewarm  # type: ignore[attr-defined]
            # The worker already listed the q-entries (off the GUI thread,
            # the same external-link scan that drives the bar). Stash them
            # so ``_populate_entries`` fills the combo without re-scanning
            # every external scan on the GUI thread — that GUI-thread
            # re-scan was the residual open freeze on big masters.
            entries = result.get("entries")
            if entries is not None:
                session._entries_cache = entries  # type: ignore[attr-defined]
            # First frame's peaks, read off-thread by the worker, so the
            # initial render installs overlays without an SFTP read on the
            # GUI thread (consumed in ``_load_entry_into_viewer``).
            session._prewarm_overlays = result.get("prewarm_overlays")  # type: ignore[attr-defined]
            # pygid normalization (0-D angle_of_incidence → 1-D, per-frame
            # analysis groups) is NOT done here — it's deferred and run
            # lazily for one entry just before its first pipeline run (see
            # ``_ensure_entry_normalized``), so opening never touches every
            # linked scan.
            #
            # A file whose last pipeline run died mid-swap (see
            # ``peak_lists`` / ``pipeline.recover_swaps``) still has a
            # primary table sitting under a standard name. Put it back
            # before anything reads the file, so the viewer never shows
            # the wrong table as "detected peaks". A no-op, and no
            # write at all, on the overwhelmingly normal file.
            try:
                from mlgidlab.pipeline import recover_swaps
                recover_swaps(session.temp_path)
            except Exception:
                logger.debug("suppressed swap recovery on open", exc_info=True)
            self._sessions.append(session)
            self.tree.findHdf5TreeModel().insertFile(str(session.temp_path))
            # Re-opening a path that's already open REPLACES the old
            # instance: its temp copy predates the new bytes (the
            # conversion append/overwrite-then-auto-open case), and two
            # instances of one file only confuse. A dirty old instance
            # gets the usual save prompt first; cancelling keeps both.
            self._close_duplicate_sessions(session)
            # Newly-opened file becomes the active one — the user almost
            # always wants to inspect what they just opened.
            self._set_active_session(session)
            # Remember the original (not the temp) so the recent menu
            # reopens the file at its real location next session.
            self._add_recent_file(session.original_path, "nexus")
            self._refresh_tree_raw_paths()
            # If the pre-warm wasn't consumed (e.g. no q-entries to land
            # on), release its handle so it doesn't leak.
            leftover = getattr(session, "_prewarm", None)
            if leftover is not None:
                self._release_prewarm(leftover)
                session._prewarm = None  # type: ignore[attr-defined]
        elif kind == "raw" and path is not None:
            # Bundled into one RawSession when the batch finishes. The
            # worker already walked the file for its detector datasets —
            # keep that list so activation never re-walks on the GUI
            # thread (``_populate_raw_entries`` consumes it).
            self._pending_raw_paths.append(path)
            raw_entries = result.get("raw_entries")
            if raw_entries is not None:
                self._pending_raw_entry_cache[str(path)] = raw_entries
        elif path is not None:
            self._pending_rejected.append(path)

        # Keep draining the queue regardless of this open's outcome so a
        # single bad file in a batch doesn't strand the rest.
        self._process_open_queue()

    def _close_duplicate_sessions(self, session: BaseSession) -> None:
        """Close any OLDER NeXus session open on ``session``'s original path.

        Re-opening a file (typically: the conversion just appended an
        entry/frames to, or replaced, an output file that was already
        open, and the auto-open re-opened it) must not leave two
        instances — the old one's working copy is stale. Dirty old
        instances get the standard save/discard/cancel prompt; on
        cancel the old instance is kept alongside the new one.
        """
        try:
            target = session.original_path.resolve()
        except OSError:
            return
        for old in list(self._sessions):
            if old is session or old.kind != "nexus":
                continue
            try:
                old_path = old.display_path.resolve()
            except OSError:
                continue
            if old_path != target:
                continue
            if not self._confirm_discard_changes(old):
                continue  # user cancelled — keep the old instance too
            self._close_session(old)

    @staticmethod
    def _release_prewarm(prewarm: object) -> None:
        """Release a ``(entry, FrameSource)`` pre-warm that won't be used."""
        try:
            prewarm[1].release()  # type: ignore[index]
        except Exception:
            logger.debug("suppressed exception releasing prewarm", exc_info=True)

    def _close_session(self, session: BaseSession) -> None:
        """Remove ``session`` from the window: tear down its tree entry,
        delete its temp dir, and pick a new active if it was the active one.
        """
        was_active = session is self._active_session
        if session not in self._sessions:
            return
        # Last chance for a pending quick-select box: after this the
        # file it belongs to is gone. Runs before the save prompt the
        # caller may still show, so the box counts as unsaved work.
        if was_active:
            self.viewer.commit_pending_manual()
        # Stop playback before pulling the file out from under the
        # viewer — the timer's next tick would otherwise read from a
        # released FrameSource.
        if was_active:
            self._pause_playback()
        self._sessions.remove(session)
        # An inactive session may hold a stashed FrameSource (parked for
        # instant re-activation) — release it before the temp dir goes.
        stash = getattr(session, "_prewarm", None)
        if stash is not None:
            self._release_prewarm(stash)
            session._prewarm = None  # type: ignore[attr-defined]
        # Drop this file's lazy-normalization record (keyed by temp_path).
        self._normalized_entries.pop(str(session.temp_path), None)
        # silx exposes no "remove single file" API on Hdf5TreeModel, so we
        # rebuild the tree from the remaining sessions. Cheap — sessions
        # are typically <5 and the model just re-opens HDF5 files.
        with self._detached_silx_tree():
            if was_active:
                # Active state is tied to viewer/entry_combo content — drop
                # it before swapping so we don't leak the old session's
                # overlays.
                self.viewer.clear()
                self.viewer.clear_history()
                self.profile_viewer.clear()
                self.peaks_table_panel.clear()
                # The Structure panel describes a node in the file we are
                # about to delete the working copy of; empty it before the
                # path goes away.
                self._clear_structure_panel()
                self.entry_combo.blockSignals(True)
                self.entry_combo.clear()
                self.entry_combo.blockSignals(False)
                self._active_session = None
            session.close()
        # Closed session may have been raw — rebuild the tree's raw set.
        self._refresh_tree_raw_paths()
        # The Conversion panel lists the files of the raw session that was
        # last shown. Once that session is closed its rows must go too;
        # another raw session coming to the front repopulates them below.
        if isinstance(session, RawSession) and (
            was_active
            or not any(isinstance(s, RawSession) for s in self._sessions)
        ):
            self.conversion_panel.set_raw_inputs([])
            self._raw_entries = {}
            self._raw_entry_label_by_path = {}
        if was_active:
            new_active = self._sessions[-1] if self._sessions else None
            if new_active is not None:
                self._set_active_session(new_active)
            else:
                self._update_title()
                self._update_actions()

    def _refresh_file_tree(self) -> None:
        """Manually re-sync every open session with the filesystem (the
        file-browser Refresh button / F5).

        Per session: a DELETED original closes the session (temp copy
        cleaned up) unless it has unsaved changes — those stay open, with
        a warning, so the user can Save As to rescue the edits. An
        original that CHANGED on disk is reloaded into the temp copy when
        the session is clean; with unsaved changes it is left untouched
        and the conflict is reported. Raw sessions close when every one
        of their input files is gone (partially-missing batches are
        reported but kept). Blocked while a pipeline run is in flight —
        reloading the active temp file under the worker would corrupt
        the run.
        """
        if self._pipe_thread is not None:
            self.statusBar().showMessage(
                "Refresh skipped — a pipeline run is in progress.", 5000
            )
            return
        closed: list[str] = []
        reloaded: list[str] = []
        deleted_dirty: list[str] = []
        conflicts: list[str] = []
        partial_raw: list[str] = []
        for session in list(self._sessions):
            if session.kind == "raw":
                missing = [p for p in session.raw_paths if not p.exists()]  # type: ignore[attr-defined]
                if missing and len(missing) == len(session.raw_paths):  # type: ignore[attr-defined]
                    self._close_session(session)
                    closed.append(session.display_path.name)
                elif missing:
                    partial_raw.extend(p.name for p in missing)
                continue
            original = session.display_path
            if not original.exists():
                if session.dirty:
                    deleted_dirty.append(original.name)
                else:
                    self._close_session(session)
                    closed.append(original.name)
                continue
            if session.disk_changed():  # type: ignore[attr-defined]
                if session.dirty:
                    conflicts.append(original.name)
                else:
                    self._reload_session_from_disk(session)
                    reloaded.append(original.name)
        parts: list[str] = []
        if closed:
            parts.append(f"closed (deleted on disk): {', '.join(closed)}")
        if reloaded:
            parts.append(f"reloaded from disk: {', '.join(reloaded)}")
        if deleted_dirty or conflicts or partial_raw:
            lines: list[str] = []
            if deleted_dirty:
                lines.append(
                    "Deleted on disk but kept open (unsaved changes — "
                    "use Save As to keep them):\n  "
                    + "\n  ".join(deleted_dirty)
                )
            if conflicts:
                lines.append(
                    "Changed on disk but NOT reloaded (your unsaved "
                    "changes were kept):\n  " + "\n  ".join(conflicts)
                )
            if partial_raw:
                lines.append(
                    "Raw inputs missing from disk (session kept open):"
                    "\n  " + "\n  ".join(partial_raw)
                )
            if parts:
                lines.append("Also: " + "; ".join(parts) + ".")
            QMessageBox.warning(self, "Refresh", "\n\n".join(lines))
        elif parts:
            self.statusBar().showMessage(
                "Refresh: " + "; ".join(parts) + ".", 8000
            )
        else:
            self.statusBar().showMessage(
                "Refresh: all files are up to date.", 5000
            )

    def _reload_session_from_disk(self, session: BaseSession) -> None:
        """Re-copy a clean session's original over its temp working copy
        and rebuild every cache/handle that pointed at the old bytes."""
        assert isinstance(session, NexusSession)
        # Parked re-activation handle (inactive sessions) targets the temp
        # file we're about to overwrite — release before the copy.
        stash = getattr(session, "_prewarm", None)
        if stash is not None:
            self._release_prewarm(stash)
            session._prewarm = None  # type: ignore[attr-defined]
        session._prewarm_overlays = None  # type: ignore[attr-defined]
        session._entries_cache = None  # type: ignore[attr-defined]
        is_active = session is self._active_session
        if is_active:
            self._pause_playback()
            self._entry_req_id += 1  # drop in-flight async loads of old bytes
            self.viewer.clear()
            self.viewer.clear_history()
            self.profile_viewer.clear()
            self.peaks_table_panel.clear()
        # The detached scope releases the silx model + viewer/prefetch
        # handles on the temp file so the overwrite never races a reader;
        # reattach re-inserts the files.
        with self._detached_silx_tree():
            session.reload_from_disk()
        # The old normalization record described the overwritten bytes.
        self._normalized_entries.pop(str(session.temp_path), None)
        if is_active:
            self._populate_entries()
            self._update_title()

    def _set_active_session(self, session: BaseSession | None) -> None:
        """Make ``session`` the active one and reload viewer-side state.

        No-op when ``session`` is already active. Blocked while a pipeline
        run is in flight — the worker captured the active temp_path at run
        time and ``_on_pipeline_finished`` reaches for ``self.session``,
        so swapping mid-flight would corrupt that path.
        """
        if session is self._active_session:
            return
        if self._pipe_thread is not None:
            return
        # Invalidate any in-flight async entry load: its result targets the
        # session we're leaving, so a late arrival must not install into the
        # new one (``_on_entry_loaded`` drops superseded request ids).
        self._entry_req_id += 1
        # Stop playback before swapping so the timer doesn't tick into
        # the new session's viewer state mid-construction.
        self._pause_playback()
        # Stash the outgoing NeXus session's live FrameSource as that
        # session's prewarm (entry name + warm handle): switching back
        # reinstalls it from memory, with the user's entry preserved.
        # Releasing it here (the old behaviour) made every re-activation
        # re-open the file and re-read frame 0 + peaks on the GUI thread,
        # which froze the window on big/remote masters. The handle is
        # read-only and writes only ever target the ACTIVE session, so a
        # parked handle on an inactive session's file is safe.
        outgoing = self._active_session
        src = self.viewer._frame_source
        if (
            outgoing is not None
            and outgoing.kind == "nexus"
            and src is not None
            and src.is_open
        ):
            outgoing._prewarm = (src.entry, src)  # type: ignore[attr-defined]
            # Overlays belong to the frame shown at stash time; the restore
            # lands on frame 0 and reloads peaks through the open handle.
            outgoing._prewarm_overlays = None  # type: ignore[attr-defined]
            self.viewer._frame_source = None  # clear() must not release it
        # Tear down viewer-side state belonging to the prior active session
        # before swapping — the new session's overlays must replace, not
        # accumulate on top of, whatever was previously shown.
        self.viewer.clear()
        self.viewer.clear_history()
        self.profile_viewer.clear()
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.blockSignals(False)
        self._active_session = session
        # ExpParameters are derived from the active NeXus metadata, so a
        # CIF cache built against the prior session may be misleading
        # for the new one. Forget it; the user can re-Parse when ready.
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel.clear_cif_cache()
        if session is not None:
            self._populate_entries()
        else:
            # No active session → wipe the per-entry options on the
            # pipeline panel so they don't reference a closed file.
            self.pipeline_panel.set_available_entries([])
        self._apply_session_mode(session)
        self._update_title()
        self._update_actions()
        # Status bar reflects the active session's entry / frame; the
        # entry change handler will fire shortly when the entry combo
        # repopulates, but pushing the file label now keeps the bar
        # consistent with the title bar even before that fires.
        self._update_status_entry()
        self._update_status_frame()
        # If the Figure Export window is open, drop its cached
        # mlgidbase handle and re-seed its basics pane from the new
        # session so its next render targets the right file.
        if self._figure_export_window is not None and self._figure_export_window.isVisible():
            self._figure_export_window.refresh_for_session()

    def _apply_session_mode(self, session: BaseSession | None) -> None:
        """Toggle dock visibility + viewer affordances for the session kind.

        Pipeline and Conversion docks are mode-exclusive: only one is ever
        visible at a time. Switching between a NeXus and a Raw session
        flips them in lockstep. With no active session, default to the
        Pipeline dock visible — that matches the cold-start UI.

        Raw mode also hides everything that doesn't apply to a raw
        detector frame: peak overlays / matched structures / profile
        viewer / parameter panel / Cartesian-Polar radios / Tools >
        Clear-peaks submenu. The user gets a clean canvas focused on
        the conversion workflow.
        """
        # Central area: the welcome page whenever nothing is loaded, the
        # Image / Data tabs otherwise. Refreshed on the way in so the
        # recent list reflects the file that was just closed.
        stack = getattr(self, "_central_stack", None)
        if stack is not None:
            if session is None:
                self._refresh_welcome_view()
                stack.setCurrentWidget(self.welcome_view)
            else:
                stack.setCurrentWidget(self.tabs)

        is_raw = session is not None and session.kind == "raw"
        # Read before the docks are shown, hidden and re-tabified below:
        # each of those can bring another tab forward for a moment.
        front = getattr(self, "_right_front_dock", None)
        self._pipeline_dock.setVisible(not is_raw)
        self._conversion_dock.setVisible(is_raw)
        # Re-tabify the right-dock chain per mode so the tab bar
        # order matches the active workflow:
        #   Raw mode:   Display | Conversion | Logs
        #   NeXus mode: Display | Pipeline | Logs
        # Peaks is on the bottom (tabified with Profiles) and isn't
        # part of the right-side chain. ``tabifyDockWidget``
        # repositions an already-tabified dock, so calling these
        # every mode-switch is cheap and idempotent.
        if is_raw:
            self.tabifyDockWidget(self._display_dock, self._conversion_dock)
            self.tabifyDockWidget(self._conversion_dock, self._logs_dock)
            self._conversion_dock.raise_()
            self._right_front_dock = self._conversion_dock
        else:
            self.tabifyDockWidget(self._display_dock, self._pipeline_dock)
            self.tabifyDockWidget(self._pipeline_dock, self._logs_dock)
            # Keep whichever tab the user had in front. Pipeline when
            # there is none yet, or when it was Conversion (raw to NeXus).
            if front is None or front is self._conversion_dock:
                front = self._pipeline_dock
            front.raise_()
            self._right_front_dock = front
        # Hide NeXus-mode-only widgets in raw mode. Peaks is hidden
        # alongside Profiles since both depend on peak tables that
        # only exist after conversion.
        self._profile_dock.setVisible(not is_raw)
        if hasattr(self, "_peaks_dock"):
            self._peaks_dock.setVisible(not is_raw)
        if hasattr(self, "_scan_tracking_dock"):
            if is_raw:
                self._tracking_hidden_by_rule = False
                self._scan_tracking_dock.setVisible(False)
                self._sync_scan_tracking_visibility()
            else:
                # Hidden instead when the data has a single frame.
                self._sync_scan_tracking_visibility(from_scratch=True)
        if hasattr(self, "parameter_panel"):
            self.parameter_panel.setVisible(not is_raw)
        # Cartesian / Polar radios — meaningless before conversion.
        self.viewer.set_mode_radios_visible(not is_raw)
        # Tools > Clear peaks submenu has nothing to clear in raw mode.
        # Each kind is now a scope submenu, so disable the menu via its
        # menuAction (greys out the whole hover-target).
        for kind_menu in (
            getattr(self, "_clear_detected_menu", None),
            getattr(self, "_clear_fitted_menu", None),
            getattr(self, "_clear_matched_menu", None),
        ):
            if kind_menu is not None:
                kind_menu.menuAction().setEnabled(not is_raw)
        self._refresh_workflow_rail()
        # A mode change decides dock visibility from scratch, which would
        # pop the side and bottom docks back up while the Structure tab
        # is in front. Re-apply that tab's own rule afterwards.
        self._refresh_structure_tree_roots()
        self._sync_structure_docks()
        self._hide_stale_dock_tab_bars()
        # Re-tabifying above rebuilds the tab entries, and a fresh tab
        # comes up without an icon (Qt reads the dock's windowIcon only
        # when it creates the tab), so the glyphs go back on here.
        self._apply_dock_tab_icons()

    def _confirm_discard_changes(self, session: BaseSession | None = None) -> bool:
        target = session if session is not None else self._active_session
        if target is None or not target.dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{target.original_path.name} has unsaved changes. "
            f"Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save(confirm=False, session=target)
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    # -- silx tree helpers --

    # Per-tick bounds for chunked silx-tree inserts. The time budget
    # covers the slow case: one ``insertFile`` costs ~13 ms for a warm
    # ~10 MB image and is I/O-bound on cold or network storage, so a
    # count alone could stall the GUI there. The count bound covers the
    # fast case: display-only ``_ImageFileNode`` rows insert in
    # microseconds, but the VIEW pays a relayout per event pass that
    # scales with how many rows arrived — thousands in one tick caused
    # a ~0.5 s hiccup that the insert loop itself never saw.
    _TREE_INSERT_BUDGET_S = 0.05
    _TREE_INSERT_MAX_PER_TICK = 250
    # Show the "Browser: n/N files" status indicator only for batches at
    # least this big — a handful of rows lands within one tick and the
    # bar would just flicker.
    _TREE_PROGRESS_MIN = 10

    def _queue_tree_inserts(self, paths: list[str]) -> None:
        """Append ``paths`` to the chunked silx-tree insert queue.

        Rows appear progressively in queue order while the event loop
        keeps breathing; see the queue's comment in ``__init__`` for why
        a synchronous loop is not an option for image batches.
        """
        if not paths:
            return
        if not self._tree_insert_queue:
            # Fresh fill — restart the progress accounting.
            self._tree_insert_total = 0
            self._tree_insert_done = 0
        self._tree_insert_queue.extend(paths)
        self._tree_insert_total += len(paths)
        self._update_tree_insert_progress()
        if not self._tree_insert_timer.isActive():
            self._tree_insert_timer.start()

    def _update_tree_insert_progress(self) -> None:
        """Sync the status-bar browser-fill indicator with the counters."""
        if not hasattr(self, "_sb_tree_label"):
            # Status bar not built yet (init-order safety).
            return
        total = self._tree_insert_total
        if not self._tree_insert_queue or total < self._TREE_PROGRESS_MIN:
            self._sb_tree_label.hide()
            self._sb_tree_bar.hide()
            self._sb_tree_label.setText("")
            return
        done = self._tree_insert_done
        self._sb_tree_label.setText(f"Browser: {done}/{total} files")
        self._sb_tree_bar.setValue(int(done * 100 / total))
        if not self._sb_tree_bar.isVisible():
            self._sb_tree_label.show()
            self._sb_tree_bar.show()

    def _drain_tree_insert_queue(self) -> None:
        """Insert queued files into the silx tree for one time budget.

        Standalone images become display-only ``_ImageFileNode`` rows
        (never decoded); everything else goes through silx's real
        ``insertFile``. At least one file per tick (guaranteed
        progress); each insert is independent — one unreadable file is
        logged and skipped so it doesn't strand the rest of the batch.
        """
        model = self.tree.findHdf5TreeModel()
        deadline = time.monotonic() + self._TREE_INSERT_BUDGET_S
        inserted = 0
        while self._tree_insert_queue:
            p = self._tree_insert_queue.pop(0)
            try:
                if file_model.is_fabio_image(p):
                    model.insertImageRow(p)
                else:
                    model.insertFile(p)
            except Exception:
                logger.debug(
                    "suppressed insert in _drain_tree_insert_queue",
                    exc_info=True,
                )
            inserted += 1
            self._tree_insert_done += 1
            if (
                inserted >= self._TREE_INSERT_MAX_PER_TICK
                or time.monotonic() >= deadline
            ):
                break
        self._update_tree_insert_progress()
        if not self._tree_insert_queue:
            self._tree_insert_timer.stop()

    def _detach_silx_tree(self) -> None:
        """Release silx's read handles + the viewer's FrameSource handle.

        Required before any code path opens an HDF5 file ``r+`` (pipeline
        runs, direct h5py edits) since open read handles would otherwise
        block the writer. After the lazy-loading milestone the viewer
        also holds a long-lived h5py handle through its FrameSource —
        that handle must be released here in addition to silx's.

        Both calls are wrapped in try/except: silx's ``clear()`` walks
        every Hdf5Item to close its owned file via ``obj.filename``,
        and on a stale ``obj`` that raises ``ValueError: Not a file or
        file object``. We swallow such errors so a partial-clear
        doesn't strand the detach half-done — the next reattach
        rebuilds the model from scratch anyway.
        """
        # Pending chunked inserts target the model being cleared; drop
        # them (the reattach re-queues every session's files anyway).
        self._tree_insert_queue.clear()
        self._tree_insert_timer.stop()
        self._update_tree_insert_progress()
        try:
            self.tree.findHdf5TreeModel().clear()
        except Exception:
            # silx's clear() can blow up when an Hdf5Item references a
            # closed h5py file. Swallow it; the reattach rebuilds the
            # whole tree from the live session list.
            logger.debug("suppressed exception in MainWindow._detach_silx_tree", exc_info=True)
            pass
        try:
            self.data_viewer.setData(None)
        except Exception:
            logger.debug("suppressed exception in MainWindow._detach_silx_tree", exc_info=True)
            pass
        # Drop any deferred tree node too — it belongs to the tree we're
        # tearing down, so rendering it later would reference a stale /
        # closed silx item.
        self._pending_data_node = None
        self.viewer.release_frame_source()
        # The Structure editor's r+ handle is the third long-lived handle
        # on the working copy. It is released here for the same reason as
        # the other two: whatever runs next opens the file itself, and
        # HDF5 refuses r+ behind any open handle. The editor re-acquires
        # on the user's next edit.
        self._release_edit_handle()
        # Tell the background prefetch worker to drop its own h5py
        # handle too. mlgidbase opens the same file r+ in the worker
        # we're about to spawn; an outstanding read handle from the
        # prefetcher would either contend (Windows), trip HDF5 file
        # locking ("Unable to synchronously open file"), or silently
        # serve pre-write data into the LRU mid-pipeline (Linux).
        #
        # Must be synchronous: a queued emit returns before the
        # worker has actually closed its handle, so the immediate
        # r+ open downstream of every caller (clear_peaks, the
        # pipeline run, save-as, peak-CSV export) would race the
        # release. BlockingQueuedConnection blocks the GUI thread
        # until the worker's release() slot has returned and the
        # handle is provably closed.
        if self._prefetch_worker is not None:
            QMetaObject.invokeMethod(
                self._prefetch_worker,
                "release",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

    def _reattach_silx_tree(self) -> None:
        """Re-insert every session's files + reopen the viewer's
        FrameSource handle.

        NeXus sessions contribute one file (the temp working copy); raw
        sessions contribute every selected raw input so the user can keep
        browsing all of them while configuring conversion. The custom
        per-file icon set is repushed afterwards because clear() emptied
        the model. The viewer's FrameSource is reopened so subsequent
        frame reads can stream from disk again.

        NeXus temp copies insert synchronously (cheap shallow h5 open,
        and callers may select their nodes right after the reattach).
        Raw inputs go through the chunked queue instead: ``insertFile``
        decodes a standalone image fully on the GUI thread, so a raw
        session holding a big image batch would otherwise freeze the
        window on EVERY reattach (each pipeline run / save detaches).

        Each ``insertFile`` is independent — one bad path doesn't
        strand the rest. silx returns a node reference on success and
        raises ``OSError`` on a missing/corrupt file; either way we
        continue with the next session.
        """
        model = self.tree.findHdf5TreeModel()
        raw_batch: list[str] = []
        for s in self._sessions:
            try:
                if isinstance(s, RawSession):
                    raw_batch.extend(str(p) for p in s.raw_paths)
                else:
                    model.insertFile(str(s.temp_path))
            except Exception:
                # One bad session shouldn't strand the rest. The user
                # will see the missing entry; rebuild at next detach.
                logger.debug("suppressed exception in MainWindow._reattach_silx_tree", exc_info=True)
                pass
        if raw_batch:
            self._queue_tree_inserts(raw_batch)
        self._refresh_tree_raw_paths()
        self.viewer.acquire_frame_source()

    @contextmanager
    def _detached_silx_tree(self):
        """Scoped silx detach/reattach for a *synchronous* critical
        section that needs the HDF5 file free of read handles.

        Detaches on entry and guarantees re-attachment on exit via
        ``finally`` — whether the block falls off the end, ``return``s
        early, or raises. This replaces the hand-paired
        ``_detach_silx_tree()`` / ``_reattach_silx_tree()`` calls whose
        reattach had to be duplicated on the happy path *and* every
        except/early-return branch (easy to forget one half).

        Deliberately NOT used at three sites, which keep explicit
        calls because the scoped semantics don't fit:

        * the pipeline run: detach spans a worker thread and the
          reattach happens later in ``_on_pipeline_finished``;
        * ``_safe_selected_h5_nodes``: a 2-line tear-down + rebuild
          recovery, not a "do work while detached" scope;
        * ``closeEvent``: a one-way detach on teardown — reattaching
          would be wrong.
        """
        self._detach_silx_tree()
        try:
            yield
        finally:
            self._reattach_silx_tree()
