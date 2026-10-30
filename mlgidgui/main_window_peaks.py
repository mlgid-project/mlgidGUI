"""Peak editing: add to detected/fitted, copy/paste (single frame and to-range), batch 2D fit, deletes, the undo-push plumbing and the Peaks-table sync.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)
from mlgidgui import (
    file_model,
    frame_range,
    peak_clipboard,
    peak_lists,
    pipeline,
)
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.manual_fit import (
    DEFAULT_CLUSTERING_DISTANCE_PEAKS,
    DEFAULT_CLUSTERING_DISTANCE_RINGS,
    DEFAULT_CLUSTERING_EXTEND,
)
from mlgidgui.main_window_constants import (
    DEFAULT_PROFILE_MASK_NEIGHBOURS,
    PROFILE_MASK_NEIGHBOURS_KEY,
)
from mlgidgui.peak_link import link_enabled, read_bool
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui.pipeline import PipelineCommand
from mlgidgui.widgets import skin_progress

logger = logging.getLogger(__name__)


class _CallbackAction:
    """Generic ``_Action`` (per the image_viewer protocol) that
    delegates ``undo`` / ``redo`` to caller-supplied callables.

    Lets host-level operations participate in the viewer's undo
    stack without dragging file-mutation code across the viewer
    abstraction. Used by ``_on_add_to_fitted`` so Ctrl+Z reverts a
    just-committed fitted row.
    """

    def __init__(self, undo_fn, redo_fn) -> None:
        self._undo_fn = undo_fn
        self._redo_fn = redo_fn

    def undo(self, _viewer) -> None:
        self._undo_fn()

    def redo(self, _viewer) -> None:
        self._redo_fn()


class PeaksMixin:
    def _on_score_edit_requested(self, value: float) -> None:
        """Write a new confidence score to every detected-FLAVOURED peak
        selected: the built-in layer, and any registered extra list the
        user chose to treat as detected.

        The Parameter panel's Score-row editor is deliberately ignorant
        of how many peaks it covers, because relabelling a frame is
        Ctrl-click a handful and press one button. Members of the
        selection that are not score-editable are skipped rather than
        refused: a mixed selection is a normal thing to have, and
        stopping the whole write because one fitted box came along with
        it would be useless.

        A list's write lands in that list's own dataset (resolved in
        ``_on_peak_row_write_requested``), so this stays inside the
        no-spill rule: an extra layer is still readable and writable
        only as itself.

        The viewer applies them as ONE undoable gesture and fires the
        file writes through the existing ``peakRowWriteRequested``
        plumbing, so the silx detach dance and the dirty flag are
        already handled.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        targets = [
            (sel.frame, sel.kind, int(sel.peak_id), float(value))
            for sel in self.viewer.selected_peaks()
            if self._score_is_editable(sel)
        ]
        if not targets:
            return
        self.viewer.set_peak_scores(targets)

    def _score_is_editable(self, sel) -> bool:
        """Whether this selection's score may be rewritten.

        The built-in detected layer always qualifies. A registered extra
        list qualifies when it is treated as detected AND its table
        actually has a score column -- ``update_peak_row`` writes only
        the fields the dtype carries, so without that check a table
        without one would take the edit in memory and drop it on disk.
        Everything else (fitted, matched, manual, an unregistered list)
        is skipped.
        """
        if sel.kind == "detected":
            return True
        if not peak_lists.is_list_kind(sel.kind):
            return False
        spec = self.viewer.peak_list_spec(sel.kind)
        if spec is None or spec.treat_as != peak_lists.TREAT_DETECTED:
            return False
        table = (self.viewer._frame_peaks.get(sel.frame) or {}).get(sel.kind)
        return table is not None and table.has_score

    def _on_add_to_detected(self) -> None:
        """Commit the active manual box to detected_peaks at the chosen
        confidence score.

        Writes synchronously via ``file_model.add_detected_peak_row`` —
        the same path copy/paste uses — so the score picked in the
        Parameter panel's Score-row dropdown (High = 1.0 / Medium = 0.5 /
        Low = 0.1) is stored verbatim. ``mlgidBASE.add_peak`` takes no
        score argument, so it can't carry the choice; a hand-drawn box
        has no model provenance anyway, so the user's pick is the
        authority. The manual box is left in place so it can also be sent
        to fitted_peaks or tweaked further. One grouped Ctrl+Z removes the
        added row.
        """
        if self._is_busy():
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind != "manual" or sel.manual_ref is None or not entry:
            return
        self._write_detected_from_manual(
            entry, int(self.viewer.current_frame), sel.manual_ref,
        )

    def _write_detected_from_manual(
        self, entry: str, frame: int, manual_peak, *,
        select: bool = True, quiet: bool = False,
    ) -> int | None:
        """Write one manual box to ``detected_peaks``; return its id.

        Split out of ``_on_add_to_detected`` so a caller that is not
        acting on the current *selection* can use it — quick-select
        commits the box the user drew **before** the one now selected,
        which the button path could never name.

        ``select`` and ``quiet`` let the quick-select caller keep the
        selection where the user put it and stay off the modal path
        (see ``_quick_commit``). The defaults reproduce the button
        exactly.

        Returns None when the write failed; the error has already been
        surfaced.
        """
        score = float(self.parameter_panel.confidence_score())
        geom = {
            "radius": float(manual_peak.radius),
            "angle": float(manual_peak.angle),
            "radius_width": float(manual_peak.radius_width),
            "angle_width": float(manual_peak.angle_width),
            "is_ring": bool(manual_peak.is_ring),
        }
        with self._detached_silx_tree():
            try:
                new_id = int(file_model.add_detected_peak_row(
                    self.session.temp_path, entry, frame,
                    score=score, **geom,
                ))
            except Exception as exc:
                if quiet:
                    self.pipeline_panel.append_log(
                        f"Quick select: could not add a detected peak — {exc}"
                    )
                    return None
                box = (
                    QMessageBox.warning if isinstance(exc, KeyError)
                    else QMessageBox.critical
                )
                box(self, "Add to detected", str(exc))
                return None
        self.session.mark_dirty()
        self._update_title()
        self._load_entry_into_viewer(entry, preserve_view=True)
        if select:
            self._add_detected_to_selection(entry, frame, [new_id])
        self._push_add_detected_undo(
            entry=entry, frame=frame, peak_id=new_id,
            score=score, geom=geom,
        )
        msg = (
            f"Added detected peak (score {score:.2f}) on "
            f"{entry}/frame{frame:05d}"
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)
        return new_id

    # -- Quick select: drawing the next box commits the previous one --

    def _on_quick_select_toggled(self, enabled: bool) -> None:
        """Arm / disarm the viewer and repaint the status cell.

        Disarming commits whatever is still pending — the viewer does
        that inside ``set_quick_select`` — so the last box of a run is
        never lost by simply switching the mode off.
        """
        self.viewer.set_quick_select(bool(enabled))
        self._update_status_quick_select()

    def _on_quick_commit_requested(self, frame: int, peak) -> None:
        """Write one pending manual box to the file.

        The viewer asks for this from every path that would otherwise
        drop the box — the next box being drawn, the selection moving
        away, a frame change, Enter, and the mode being switched off.
        All of them land here so they cannot behave differently.

        The write and the box's retirement are bundled into **one** undo
        entry: the commit helpers push their own entries, which are
        lifted off the stack and wrapped together with the removal of
        the manual box. One Ctrl+Z therefore turns the committed peak
        back into the box it was drawn as.
        """
        entry = self.entry_combo.currentText()
        if (
            self.session is None or not entry or peak is None
            or self._is_busy()
        ):
            return
        target = self.parameter_panel.quick_select_target()
        stack = self.viewer._undo_stack
        depth = len(stack)
        written = self._quick_write(entry, int(frame), peak, target)
        if not written:
            # Nothing landed — leave the box alone so the user can retry
            # or fix it rather than losing what they drew.
            del stack[depth:]
            return
        actions = list(stack[depth:])
        del stack[depth:]
        self.viewer.retire_manual_peak(int(frame), peak)
        self.viewer._push_undo(
            _CallbackAction(
                lambda: self._quick_commit_undo(int(frame), peak, actions),
                lambda: self._quick_commit_redo(int(frame), peak, actions),
            )
        )
        self._update_status_quick_select()

    def _quick_write(self, entry: str, frame: int, peak, target: str) -> bool:
        """Write the box at the chosen target. True if anything landed.

        Detected is the cheap, cannot-fail path. Fitted runs the same
        code the Add-to-fitted button runs — including, with the peak
        link on, writing the detected partner — but **quietly**: no
        modal is allowed to interrupt a labelling run.

        A fit that will not converge therefore does not cost the user
        the box. It is committed as detected instead, with a log line
        naming what happened. That inverts the button's deliberate
        strictness (physics-audit F-06, no silent fallback), and the
        trade is different here on purpose: the button commits one peak
        the user is watching, this commits a peak they have already
        moved on from.
        """
        from mlgidgui.parameter_panel import ParameterPanel

        sel = SelectedPeak.from_manual(peak, int(frame))
        if target == ParameterPanel.QUICK_TARGET_DETECTED:
            return self._write_detected_from_manual(
                entry, frame, peak, select=False, quiet=True,
            ) is not None

        detected_id = None
        if target == ParameterPanel.QUICK_TARGET_BOTH and not self._fitted_link_on():
            # With the link on, the fitted path writes the detected
            # partner itself, so writing one here would duplicate it.
            detected_id = self._write_detected_from_manual(
                entry, frame, peak, select=False, quiet=True,
            )
        fitted_id = self._write_fitted_from_selection(
            sel, entry, frame, select=False, quiet=True,
        )
        if fitted_id is not None:
            return True
        if detected_id is not None:
            self.pipeline_panel.append_log(
                "Quick select: the fit failed — the box was kept as a "
                "detected peak."
            )
            return True
        # Fitted-only target with a failed fit: keep the box rather than
        # discard the user's work.
        fallback = self._write_detected_from_manual(
            entry, frame, peak, select=False, quiet=True,
        )
        if fallback is None:
            return False
        msg = "Quick select: the fit failed — saved as a detected peak instead."
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)
        return True

    def _quick_commit_undo(self, frame: int, peak, actions: list) -> None:
        """Reverse one quick commit: unwrite the rows, restore the box."""
        for action in reversed(actions):
            action.undo(self.viewer)
        self.viewer._undoable_add_manual(int(frame), peak)
        self._update_status_quick_select()

    def _quick_commit_redo(self, frame: int, peak, actions: list) -> None:
        self.viewer.retire_manual_peak(int(frame), peak)
        for action in actions:
            action.redo(self.viewer)
        self._update_status_quick_select()

    def _undoable_write_step(
        self, *, entry: str, frame: int | None, title: str,
        write, after=None,
    ) -> None:
        """The shared body of the undo/redo closures below.

        Bail while busy, navigate to the written entry (and frame, when
        given), run ``write`` inside the detached silx tree with
        ``QMessageBox.critical(title)`` on failure, then mark dirty,
        refresh title + viewer and hand ``write``'s return value to
        ``after`` (selection fixes, id rotation, logging). Closures
        whose sequence deviates (``_push_fitted_add_undo``'s
        removed-row bail-out, ``_push_paste_to_range_undo``'s
        no-navigation frame capture) keep their own bodies.
        """
        if self._is_busy():
            return
        if self.entry_combo.currentText() != entry:
            self.entry_combo.setCurrentText(entry)
        if frame is not None and int(self.viewer.current_frame) != int(frame):
            self.viewer.set_frame(int(frame))
        with self._detached_silx_tree():
            try:
                result = write()
            except Exception as exc:
                QMessageBox.critical(self, title, str(exc))
                return
        self.session.mark_dirty()
        self._update_title()
        self._load_entry_into_viewer(entry, preserve_view=True)
        if after is not None:
            after(result)

    def _push_add_detected_undo(
        self, *, entry: str, frame: int, peak_id: int,
        score: float, geom: dict,
    ) -> None:
        """Undo / redo for one 'Add to detected' commit.

        ``undo`` deletes the added row; ``redo`` re-adds it with the same
        geometry + score (fresh id). ``state["id"]`` rotates with each
        re-add so a follow-on undo deletes the right row — same pattern as
        ``_push_paste_undo`` / ``_push_delete_undo``.
        """
        state = {"id": int(peak_id)}
        g = dict(geom)

        def _undo() -> None:
            def write():
                file_model.delete_peak_row(
                    self.session.temp_path, entry,
                    frame=int(frame), kind="detected",
                    peak_id=int(state["id"]),
                )

            def after(_result) -> None:
                self._drop_detected_from_selection(int(frame), [state["id"]])
                self.pipeline_panel.append_log(
                    f"Undo add to detected: removed peak on "
                    f"{entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo add to detected", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                return int(file_model.add_detected_peak_row(
                    self.session.temp_path, entry, int(frame),
                    score=float(score), **g,
                ))

            def after(new_id) -> None:
                state["id"] = new_id
                self._add_detected_to_selection(entry, int(frame), [new_id])
                self.pipeline_panel.append_log(
                    f"Redo add to detected: re-added peak on "
                    f"{entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo add to detected", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Fitted rows, optionally linked to their detected peak -------

    def _fitted_link_on(self) -> bool:
        """Whether a fit should carry the id of its detected peak.

        Read at the moment of use, so flipping the setting takes effect
        on the next action with no wiring. With it off, every path below
        keeps the behaviour it had before the setting existed.
        """
        from mlgidgui import peak_link

        return peak_link.link_enabled()

    def _undo_created_detected(
        self, entry: str, frame: int,
        created: "tuple[int, dict, float] | None",
    ) -> None:
        """Remove a detected row written for a fit that then failed.

        The manual-box path writes the detected partner first, so a
        failing fit would otherwise leave a detected peak the user never
        asked for. Mirrors the injection path's "no orphan geometry"
        cleanup. Best-effort: a failure here must not mask the error
        being reported.
        """
        if created is None:
            return
        try:
            file_model.delete_peak_row(
                self.session.temp_path, entry, frame=int(frame),
                kind="detected", peak_id=int(created[0]),
            )
        except Exception:
            logger.debug(
                "suppressed exception rolling back a detected row",
                exc_info=True,
            )

    def _write_fitted_row(
        self, entry: str, frame: int, add_kwargs: dict,
        target_id: int | None,
    ) -> tuple[int, list[dict]]:
        """Write one fitted row, returning ``(new_id, replaced_rows)``.

        ``target_id`` is the detected peak's id when the link is on, and
        None otherwise — in which case this is exactly the old
        ``add_fitted_peak_row`` call and ``replaced_rows`` is empty.

        With a target, any existing fitted row under that id is removed
        first, so a detected peak can never accumulate fits. The removal
        is a normal fitted delete and gets the normal matched handling:
        ``prepare_fitted_delete``, delete, then
        ``remap_matched_peak_lists``. Matched structures lose exactly
        the replaced peak and keep everything else, which is what a
        fitted delete has always done.

        The removed row is returned so undo can put it back verbatim.
        Caller must already hold the detached silx tree.
        """
        path = self.session.temp_path
        replaced: list[dict] = []
        if target_id is not None:
            replaced = file_model.read_peak_rows(
                path, entry, int(frame), "fitted", [int(target_id)],
                with_ids=True, with_errors=True,
            )
            if replaced:
                positions = file_model.prepare_fitted_delete(
                    path, entry, int(frame), [int(target_id)],
                )
                file_model.delete_peak_row(
                    path, entry, frame=int(frame), kind="fitted",
                    peak_id=int(target_id),
                )
                file_model.remap_matched_peak_lists(
                    path, entry, int(frame), positions,
                )
        new_id = int(file_model.add_fitted_peak_row(
            path, entry, int(frame), peak_id=target_id, **add_kwargs,
        ))
        return new_id, replaced

    def _restore_peak_rows(
        self, entry: str, frame: int, kind: str, rows: list[dict],
    ) -> None:
        """Put snapshot rows back under the ids they had.

        Used by undo wherever the link is involved. Restoring under a
        fresh id would break the pair the undo is supposed to restore,
        so ``read_peak_rows(with_ids=True)`` snapshots carry the id and
        it is handed back through ``peak_id``. Caller must already hold
        the detached silx tree.
        """
        for row in rows:
            d = dict(row)
            peak_id = d.pop("id", None)
            self._readd_peak_row(entry, int(frame), kind, d, peak_id=peak_id)

    def _detected_row_from_manual(
        self, entry: str, frame: int, manual_peak,
    ) -> tuple[int, dict, float]:
        """Write a hand-drawn box to ``detected_peaks``.

        With the link on, every fitted peak must have a detected
        partner, so a manual box is committed to detected first and the
        fit pairs to that new row — the same two rows the user would get
        by pressing Add to detected and then Add to fitted. Score comes
        from the Parameter panel's dropdown, exactly as
        ``_on_add_to_detected`` reads it.

        Returns ``(new_id, geom, score)`` so the undo closure can delete
        and re-create the row. Caller must already hold the detached
        silx tree.
        """
        geom = {
            "radius": float(manual_peak.radius),
            "angle": float(manual_peak.angle),
            "radius_width": float(manual_peak.radius_width),
            "angle_width": float(manual_peak.angle_width),
            "is_ring": bool(manual_peak.is_ring),
        }
        score = float(self.parameter_panel.confidence_score())
        new_id = int(file_model.add_detected_peak_row(
            self.session.temp_path, entry, int(frame), score=score, **geom,
        ))
        return new_id, geom, score

    def _on_add_to_fitted(self) -> None:
        """Append a row to fitted_peaks for the active manual / detected box.

        Dispatches strictly on ``parameter_panel.fit_mode()``:

        * ``FIT_MODE_2D`` (pygidfit) → route through
          ``manual_fit.fit_one_peak``; the persisted row carries real
          A/B/C/theta shape coefficients — identical to what the
          pipeline ``run_fitting`` writes for the same box.
        * ``FIT_MODE_1D`` (legacy scipy) → use the profile viewer's
          cached 1D fits + zero-fill the 2D shape coefficients. Width
          convention matches the 2D path so saved boxes render
          identically: ``radius_width = angle_width = 2σ ≈ 0.849 ×
          FWHM`` (see ``_build_fitted_row_1d``).

        Ring storage forces 1D regardless of the radio (pygidfit's
        segment model can't fit rings — the ring saved row uses
        ``angle = 45°``, ``angle_width = ∞``).

        Strict failure: if the chosen mode can't produce a fit (1D
        without scipy convergence, or 2D with pygidfit raising / NaN /
        missing geometry), the user gets a clear error naming both
        the chosen mode and the reason — no silent fall-back to the
        other mode. Physics-audit finding F-06 closure.
        """
        if self._is_busy():
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind not in ("manual", "detected") or not entry:
            return
        frame = int(self.viewer.current_frame)
        if (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not self.parameter_panel.save_as_ring()
        ):
            # A 2D fit can take tens of seconds for a peak in a bad
            # peak-on-ring cluster, so it runs in the background and the
            # row is committed when it lands. The 1D and ring paths are
            # instant and stay inline.
            self._start_add_to_fitted_2d(sel, entry, frame)
            return
        self._write_fitted_from_selection(sel, entry, frame)

    def _start_add_to_fitted_2d(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> None:
        """Add to fitted (2D) without blocking the window.

        Reuses the live preview's result when the same fit (same peak
        geometry, fit settings and neighbours) is cached, or joins the
        preview fit that is already running for it, so clicking Add to
        fitted after selecting a slow peak does not start a second fit
        beside the first. Otherwise it queues the fit. Everything about
        the click is captured here: the commit happens later, and the
        user may have moved on by then.
        """
        if self._add_fitted_wait is not None:
            self.statusBar().showMessage(
                "Add to fitted: still fitting the previous peak.", 4000,
            )
            return
        fp = self._preview_fingerprint(entry, sel)
        cached = self._2d_preview_cache.get(fp)
        if cached is not None:
            self._finish_add_to_fitted_with(
                sel, entry, frame, cached, None, select=True,
            )
            return
        if not self._preview_runner.is_pending(fp):
            job, err = self._prepare_pygidfit_fit(sel, entry, frame)
            if job is None:
                QMessageBox.warning(
                    self, "Add to fitted (2D)",
                    f"pygidfit could not fit this box: {err}\n\n"
                    "Try widening the box, or switch to '1D fit (scipy)' "
                    "mode to commit with the legacy 1D Gaussian fit.",
                )
                return
            self._preview_runner.submit(fp, job)
        self._preview_runner.pin(fp)
        btn = self.parameter_panel.btn_add_fitted
        self._add_fitted_wait = {
            "key": fp, "sel": sel, "entry": entry, "frame": frame,
            "session": self.session.temp_path,
            "button_was_enabled": btn.isEnabled(),
        }
        btn.setEnabled(False)
        self.statusBar().showMessage(
            f"Add to fitted: fitting {sel.kind} peak {sel.peak_id} "
            "(the window stays usable)…"
        )

    def _finish_add_to_fitted(self, key, fit_2d, err, ran: bool = True) -> None:
        """A background fit landed; commit it if it is the one awaited."""
        wait = self._add_fitted_wait
        if wait is None or wait["key"] != key:
            return
        self._add_fitted_wait = None
        self.parameter_panel.btn_add_fitted.setEnabled(
            wait["button_was_enabled"]
        )
        self.statusBar().clearMessage()
        if not ran:
            return
        if self.session is None or self.session.temp_path != wait["session"]:
            self.pipeline_panel.append_log(
                "Add to fitted cancelled: the file was closed while the "
                "fit ran."
            )
            return
        if self.entry_combo.currentText() != wait["entry"]:
            self.pipeline_panel.append_log(
                "Add to fitted cancelled: the entry changed while the fit "
                "ran."
            )
            return
        cur = self.viewer.selected_peak
        sel = wait["sel"]
        still_selected = (
            cur is not None and cur.kind == sel.kind
            and cur.peak_id == sel.peak_id and cur.frame == sel.frame
        )
        self._finish_add_to_fitted_with(
            sel, wait["entry"], wait["frame"], fit_2d, err,
            select=still_selected,
        )

    def _finish_add_to_fitted_with(
        self, sel: SelectedPeak, entry: str, frame: int, fit_2d, err,
        *, select: bool,
    ) -> None:
        """Commit a finished 2D fit, or explain why there is none."""
        if fit_2d is None:
            QMessageBox.warning(
                self, "Add to fitted (2D)",
                f"pygidfit could not fit this box: {err}\n\n"
                "Try widening the box, or switch to '1D fit (scipy)' "
                "mode to commit with the legacy 1D Gaussian fit.",
            )
            return
        self._commit_fitted_row(
            sel, entry, frame, self._fitted_row_from_fit(fit_2d),
            False, ParameterPanel.FIT_MODE_2D, select=select, quiet=False,
        )

    def _write_fitted_from_selection(
        self, sel: SelectedPeak, entry: str, frame: int, *,
        select: bool = True, quiet: bool = False,
    ) -> int | None:
        """The body of Add-to-fitted, for an explicitly-named box.

        Split out so quick-select can commit the box the user drew
        *before* the current selection — which is exactly the box the
        button path cannot name. A manual box arrives here as
        ``SelectedPeak.from_manual(...)``, which is what
        ``_build_fitted_row_2d`` needs anyway.

        ``select`` and ``quiet`` are the same two knobs
        ``_write_detected_from_manual`` takes: keep the selection where
        the user put it, and stay off the modal path. The defaults
        reproduce the button exactly, including its strict
        no-silent-fallback failure (F-06) — the quiet path is the one
        place that trade is inverted, and ``_quick_commit`` documents
        why. The undo entry is pushed as usual; a caller that wants the
        commit bundled with something else wraps it (see
        ``_quick_commit``).

        Returns the new fitted id, or None when nothing was written.
        """
        save_as_ring = self.parameter_panel.save_as_ring()
        # ``fit_mode()`` already collapses to FIT_MODE_1D when ring is
        # checked, so the rest of this function just branches on the
        # token.
        mode = self.parameter_panel.fit_mode()

        if mode == ParameterPanel.FIT_MODE_2D:
            row = self._build_fitted_row_2d(sel, entry, frame, quiet=quiet)
        else:
            row = self._build_fitted_row_1d(sel, save_as_ring, quiet=quiet)
        if row is None:
            # The helper already showed a QMessageBox / log line.
            return None
        return self._commit_fitted_row(
            sel, entry, frame, row, save_as_ring, mode,
            select=select, quiet=quiet,
        )

    def _commit_fitted_row(
        self, sel: SelectedPeak, entry: str, frame: int, row: dict,
        save_as_ring: bool, mode: str, *, select: bool, quiet: bool,
    ) -> int | None:
        """Write ``row`` to ``fitted_peaks`` for ``sel`` and do the
        bookkeeping (reload, selection, undo, log).

        The second half of Add to fitted, split from the fit so the
        button can run a slow 2D fit in the background and commit when
        it lands (``_finish_add_to_fitted``). Everything it needs about
        the click is passed in, since the selection, frame or panel may
        have changed while the fit ran. Returns the new fitted id, or
        None when nothing was written.
        """
        # pygidfit reports a ring as an infinite angular width: such a
        # fit is stored as a ring whatever the checkbox said.
        row_angle, is_ring = file_model.ring_if_infinite_width(
            row["angle"], row["angle_width"], save_as_ring,
        )
        add_kwargs = {
            "radius": row["radius"],
            "radius_width": row["radius_width"],
            "angle": row_angle,
            "angle_width": row["angle_width"],
            "amplitude": row["amplitude"],
            "is_ring": is_ring,
            "theta": row["theta"],
            "A": row["A"], "B": row["B"], "C": row["C"],
        }
        # With the link on the row carries its detected peak's id, which
        # replaces any previous fit of that peak. A hand-drawn box has
        # no detected peak yet, so one is written first and the fit
        # pairs to it. With the link off, target_id stays None and this
        # is the plain append it always was.
        link = self._fitted_link_on()
        target_id: int | None = None
        created_detected: tuple[int, dict, float] | None = None
        with self._detached_silx_tree():
            try:
                if link and sel.kind == "detected" and sel.peak_id is not None:
                    target_id = int(sel.peak_id)
                elif link and sel.kind == "manual" and sel.manual_ref is not None:
                    created_detected = self._detected_row_from_manual(
                        entry, frame, sel.manual_ref,
                    )
                    target_id = created_detected[0]
                new_id, replaced = self._write_fitted_row(
                    entry, frame, add_kwargs, target_id,
                )
            except Exception as exc:
                # Never leave a detected row behind for a fit that did
                # not land — same rule the injection path follows.
                self._undo_created_detected(entry, frame, created_detected)
                if quiet:
                    self.pipeline_panel.append_log(
                        f"Quick select: could not add a fitted peak — {exc}"
                    )
                    return None
                box = (
                    QMessageBox.warning if isinstance(exc, KeyError)
                    else QMessageBox.critical
                )
                box(self, "Add to fitted", str(exc))
                return None

        self.session.mark_dirty()
        self._update_title()
        # Add-to-fitted appends a fitted row; with the link on it may
        # first remove the previous fit of the same detected peak, which
        # the undo entry restores. Prior undo history stays valid either
        # way (no other row's id changes).
        # Pull the fresh fitted_peaks (and matched, which references it) back
        # into the viewer — same entry, so preserve the user's zoom + frame.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # In 2D mode the user committed without seeing pygidfit's actual
        # result on the profile plots (the pink curves are hidden in
        # 2D mode — see ``_apply_fit_curve_visibility``). Auto-switch
        # the selection to the new fitted peak so the result is
        # immediately visible (cyan box + profile fits both update to
        # the fitted-kind selection). 1D mode is unchanged because the
        # pink curves already previewed what was saved.
        if select and mode == ParameterPanel.FIT_MODE_2D:
            fitted_sel = SelectedPeak(
                kind="fitted",
                frame=int(frame),
                peak_id=int(new_id),
                radius=float(row["radius"]),
                angle=float(row_angle),
                radius_width=float(row["radius_width"]),
                angle_width=float(row["angle_width"]),
                is_ring=bool(is_ring),
                amplitude=(
                    float(row["amplitude"])
                    if row.get("amplitude") is not None else None
                ),
            )
            # ``preserve_manual=True`` so the source manual peak
            # is not auto-removed by the away-from-manual side
            # effect on _set_selected. Without this, the silent
            # ManualRemoveAction it pushes would sit on the undo
            # stack alongside our FittedRowAction and the user
            # would need TWO Ctrl+Z presses to revert the commit.
            self.viewer._set_selected(fitted_sel, preserve_manual=True)
        # Ctrl+Z support for Add-to-fitted: the file mutation is
        # reversible (delete the just-added row + restore the source
        # selection). Pushed onto the existing history so the commit
        # stacks with any prior undoable edits.
        self._push_fitted_add_undo(
            entry=entry, frame=int(frame), new_id=int(new_id),
            add_kwargs=add_kwargs,
            source_sel=sel,
            target_id=target_id,
            replaced=replaced,
            created_detected=created_detected,
        )
        # Ring toggle is sticky across selections but reset on commit so
        # the next Add-to-fitted defaults back to segment unless the user
        # explicitly opts in again.
        self.parameter_panel.reset_save_as_ring()
        self.pipeline_panel.append_log(
            f"Added fitted peak id={new_id} "
            f"({'ring' if is_ring else 'segment'} from {sel.kind}) on "
            f"{entry}/frame{frame:05d}"
            + (" — replaced the previous fit of that peak" if replaced else "")
            + (
                f" (also added detected id={created_detected[0]})"
                if created_detected else ""
            )
        )
        return int(new_id)

    def _push_fitted_add_undo(
        self, *, entry: str, frame: int, new_id: int,
        add_kwargs: dict, source_sel: SelectedPeak,
        target_id: int | None = None,
        replaced: list[dict] | None = None,
        created_detected: "tuple[int, dict, float] | None" = None,
    ) -> None:
        """Push a Ctrl+Z entry for an Add-to-fitted commit.

        ``undo`` deletes the just-added fitted row (and cascades
        ``clear_peaks(matched)`` on the frame, mirroring the
        existing fitted-delete handler), then reselects the source
        peak so the user is back where they started visually. For
        a manual source, the manual peak still exists in
        ``viewer._manual_peaks`` (Add-to-fitted leaves it alone),
        so reselecting it surfaces the yellow ROI again — exactly
        the "let the manual selection reappear" behaviour from the
        feature spec. For a detected source, the detected row is
        located by id in the freshly-reloaded peaks table.

        ``redo`` re-adds the row with the original kwargs.
        ``file_model.add_fitted_peak_row`` assigns a fresh id on
        each call (``max(id)+1``), so the post-redo id may differ
        from the original; ``state["current_id"]`` is updated so
        a follow-on undo deletes the correct row. With the link on
        the id is pinned to ``target_id`` instead and does not move.

        The link adds two more things to reverse, both threaded
        through here so one Ctrl+Z still undoes the whole commit:

        * ``replaced`` — the fit of that detected peak that this
          commit displaced, restored under its own id.
        * ``created_detected`` — the detected row written for a
          hand-drawn box, removed again (and re-created under the
          same id on redo, so the pair survives the round trip).

        On either side, any file error surfaces a QMessageBox and
        leaves the stack alone — the caller's downstream slot
        (``undo_last_action`` / ``redo_last_action``) is tolerant
        of no-op actions.
        """
        # Closure state for the rotating fitted-row id across
        # repeated undo/redo cycles.
        state: dict = {
            "current_id": int(new_id),
            "replaced": [dict(d) for d in (replaced or [])],
        }
        source_kind = source_sel.kind
        source_peak_id = (
            int(source_sel.peak_id)
            if source_sel.peak_id is not None else None
        )

        def _undo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    removed = file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="fitted",
                        peak_id=int(state["current_id"]),
                    )
                    if removed > 0:
                        file_model.clear_peaks(
                            self.session.temp_path, entry,
                            kind="matched", frame=int(frame),
                        )
                        # Put back the fit this commit displaced, and
                        # drop the detected row it created for a
                        # hand-drawn box, so one Ctrl+Z returns the file
                        # to exactly what it held before.
                        self._restore_peak_rows(
                            entry, int(frame), "fitted", state["replaced"],
                        )
                        if created_detected is not None:
                            file_model.delete_peak_row(
                                self.session.temp_path, entry,
                                frame=int(frame), kind="detected",
                                peak_id=int(created_detected[0]),
                            )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo Add to fitted", str(exc)
                    )
                    return
            if removed == 0:
                self.pipeline_panel.append_log(
                    f"Undo Add-to-fitted: fitted id="
                    f"{state['current_id']} already gone on "
                    f"{entry}/frame{int(frame):05d}"
                )
                return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Reselect the source peak so the visual state matches
            # what the user had before Add-to-fitted.
            if source_kind == "manual":
                manuals = self.viewer._manual_peaks.get(int(frame), [])
                if manuals:
                    self.viewer._set_selected(
                        SelectedPeak.from_manual(manuals[0], int(frame))
                    )
            elif source_kind == "detected" and source_peak_id is not None:
                tables = self.viewer._frame_peaks.get(int(frame)) or {}
                det = tables.get("detected")
                if det is not None:
                    for i in range(len(det)):
                        if int(det.ids[i]) == source_peak_id:
                            self._select_table_row(
                                int(frame), "detected", det, i
                            )
                            break
            self.pipeline_panel.append_log(
                f"Undo Add-to-fitted: removed fitted id="
                f"{state['current_id']} on "
                f"{entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    if created_detected is not None:
                        det_id, det_geom, det_score = created_detected
                        file_model.add_detected_peak_row(
                            self.session.temp_path, entry, int(frame),
                            score=det_score, peak_id=int(det_id), **det_geom,
                        )
                    readded_id, re_replaced = self._write_fitted_row(
                        entry, int(frame), add_kwargs, target_id,
                    )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo Add to fitted", str(exc)
                    )
                    return
            state["current_id"] = int(readded_id)
            state["replaced"] = [dict(d) for d in re_replaced]
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Re-select the re-added fitted peak if the user is
            # currently in 2D mode (mirrors the original commit's
            # auto-select). In 1D mode leave selection alone, also
            # mirroring the original.
            if self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D:
                # Same ``preserve_manual=True`` rationale as the
                # original commit's auto-switch: keep the manual
                # source intact across the redo so a subsequent
                # Ctrl+Z still has the manual peak to reselect.
                self.viewer._set_selected(
                    SelectedPeak(
                        kind="fitted", frame=int(frame),
                        peak_id=int(readded_id),
                        radius=float(add_kwargs["radius"]),
                        angle=float(add_kwargs["angle"]),
                        radius_width=float(add_kwargs["radius_width"]),
                        angle_width=float(add_kwargs["angle_width"]),
                        is_ring=bool(add_kwargs["is_ring"]),
                        amplitude=(
                            float(add_kwargs["amplitude"])
                            if add_kwargs.get("amplitude") is not None
                            else None
                        ),
                    ),
                    preserve_manual=True,
                )
            self.pipeline_panel.append_log(
                f"Redo Add-to-fitted: re-added fitted id="
                f"{readded_id} on {entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Copy / paste detected peaks ---------------------------------

    def _on_copy_peaks(self) -> None:
        """Snapshot the selected detected peaks into the in-app clipboard.

        Filters the multi-selection to ``kind="detected"`` — non-detected
        kinds are deferred per user scope. No-op + status-bar hint when
        nothing copyable is selected. The clipboard remembers the source
        entry so a same-entry paste check is enforced at paste time.

        Ctrl+C is shared with the Structure tab, which copies a node
        rather than a peak. The editor answers only while its tab is in
        front or the File browser has focus; everywhere else this is
        unchanged.
        """
        if self._structure_owns_clipboard():
            self._on_structure_copy()
            return
        if self.session is None:
            return
        sels = [s for s in self.viewer.selected_peaks() if s.kind == "detected"]
        if not sels:
            self.statusBar().showMessage(
                "Copy: select one or more detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = [
            peak_clipboard.ClipboardItem(
                radius=float(s.radius),
                angle=float(s.angle),
                radius_width=float(s.radius_width),
                angle_width=float(s.angle_width),
                is_ring=bool(s.is_ring),
                source_frame=int(s.frame),
                source_peak_id=int(s.peak_id) if s.peak_id is not None else -1,
                score=float(s.score) if s.score is not None else 0.0,
            )
            for s in sels
        ]
        peak_clipboard.set_items(items, entry=entry)
        self.statusBar().showMessage(
            f"Copied {len(items)} detected peak(s)", 4000
        )

    def _on_paste_peaks(self) -> None:
        """Paste clipboard contents onto the current frame as detected rows.

        Same-entry only: ``peak_clipboard.take_items`` returns ``[]`` for
        an entry mismatch. Writes synchronously via
        ``file_model.add_detected_peak_row`` so the new ids are known
        immediately and we can multi-select the pasted peaks after the
        reload. Pushes one grouped ``_CallbackAction`` so a single
        Ctrl+Z reverses the whole paste.

        Shared with the Structure tab exactly as Ctrl+C is — see
        ``_on_copy_peaks``.
        """
        if self._structure_owns_clipboard():
            self._on_structure_paste()
            return
        if self._is_busy():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        frame = int(self.viewer.current_frame)
        added_ids: list[int] = []
        with self._detached_silx_tree():
            try:
                for item in items:
                    new_id = file_model.add_detected_peak_row(
                        self.session.temp_path, entry, frame,
                        radius=item.radius,
                        angle=item.angle,
                        radius_width=item.radius_width,
                        angle_width=item.angle_width,
                        is_ring=item.is_ring,
                        score=item.score,
                    )
                    added_ids.append(int(new_id))
            except KeyError as exc:
                QMessageBox.warning(self, "Paste detected peaks", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Paste detected peaks", str(exc))
                return
        if not added_ids:
            return
        self.session.mark_dirty()
        self._update_title()
        # Paste only appends rows (fresh ids); no existing id is
        # invalidated, so prior undo history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Add the pasted rows to the existing multi-selection so the
        # user immediately sees what landed without losing whatever
        # they had selected before. If the prior primary wasn't a
        # detected peak (or there was no selection), the first pasted
        # row becomes the new primary.
        self._add_detected_to_selection(entry, frame, added_ids)
        self._push_paste_undo(
            entry=entry, frame=frame, ids=added_ids, items=items,
        )
        self.statusBar().showMessage(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}",
            4000,
        )
        self.pipeline_panel.append_log(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}"
        )

    def _build_detected_sels(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> list[SelectedPeak]:
        """Look up the freshly-loaded ``kind`` rows with the given ids
        on ``frame`` and return them as ``SelectedPeak`` snapshots.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Silently
        skips ids that aren't present in the table (e.g. after a
        follow-on pipeline rerun). Used by the paste / bulk-delete
        handlers and their undo/redo paths to (re)build a selection."""
        if not ids:
            return []
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        det = tables.get(kind)
        if det is None:
            return []
        id_set = {int(x) for x in ids}
        sels: list[SelectedPeak] = []
        for i in range(len(det)):
            if int(det.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind=kind, frame=int(frame),
                    peak_id=int(det.ids[i]),
                    radius=float(det.radius[i]),
                    angle=float(det.angle[i]),
                    radius_width=float(det.radius_width[i]),
                    angle_width=float(det.angle_width[i]),
                    is_ring=bool(det.is_ring[i]),
                    score=float(det.score[i]),
                    amplitude=float(det.amplitude[i]),
                ))
        return sels

    def _add_detected_to_selection(
        self, entry: str, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Add the given ``kind`` rows to the existing multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Behaviour
        mirrors a sequence of Ctrl+clicks: if the current primary is the
        same kind, all ``ids`` are appended to ``_selected_extras``;
        otherwise the first id becomes the new primary (so the
        multi-selection can grow) and the rest go to extras. Lets
        paste / delete-undo grow the user's selection instead of
        wiping it.
        """
        sels = self._build_detected_sels(frame, ids, kind=kind)
        if not sels:
            return
        v = self.viewer
        if v._selected is None or v._selected.kind != kind:
            # ``preserve_manual`` so a previously-selected manual
            # peak isn't auto-removed when we promote a peak to
            # primary (matches the post-Add-to-fitted auto-select
            # pattern at ``_on_add_to_fitted``).
            v._set_selected(sels[0], preserve_manual=True)
            v._selected_extras = list(sels[1:])
        else:
            # Avoid double-adding if any id is already in the
            # selection.
            existing = {
                (s.kind, int(s.frame), int(s.peak_id))
                for s in v.selected_peaks()
            }
            for s in sels:
                key = (s.kind, int(s.frame), int(s.peak_id))
                if key not in existing:
                    v._selected_extras.append(s)
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _drop_detected_from_selection(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Remove the given ``kind`` rows from the multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Used by
        the undo/redo paths so the highlight overlay tracks the on-disk
        state — without this, just-deleted rows would keep painting
        white selection boxes at their old positions (the SelectedPeak
        holds its own geometry independent of the peak table).
        """
        v = self.viewer
        if v._selected is None and not v._selected_extras:
            return
        id_set = {(int(frame), int(i)) for i in ids}

        def _hit(s: SelectedPeak) -> bool:
            return (
                s.kind == kind
                and (int(s.frame), int(s.peak_id)) in id_set
            )

        # Filter extras first, then handle the primary.
        v._selected_extras = [
            s for s in v._selected_extras if not _hit(s)
        ]
        if v._selected is not None and _hit(v._selected):
            if v._selected_extras:
                v._selected = v._selected_extras.pop(0)
            else:
                v._selected = None
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _push_paste_undo(
        self, *, entry: str, frame: int,
        ids: list[int], items: list,
    ) -> None:
        """Undo / redo for a paste batch — one grouped action.

        ``state["ids"]`` is the rotating list of fitted ids; each redo
        re-adds rows and overwrites it with the freshly-assigned ids
        so a follow-on undo deletes the right rows. Same pattern as
        ``_push_fitted_add_undo`` extended for N rows.
        """
        state = {"ids": list(ids)}
        snapshot = list(items)

        def _undo() -> None:
            def write():
                for peak_id in state["ids"]:
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="detected",
                        peak_id=int(peak_id),
                    )

            def after(_result) -> None:
                # Drop the now-deleted rows from the multi-selection so
                # the white highlight overlay tracks the on-disk state.
                self._drop_detected_from_selection(int(frame), state["ids"])
                self.pipeline_panel.append_log(
                    f"Undo paste: removed {len(state['ids'])} detected "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo paste", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                new_ids: list[int] = []
                for item in snapshot:
                    new_id = file_model.add_detected_peak_row(
                        self.session.temp_path, entry, int(frame),
                        radius=item.radius,
                        angle=item.angle,
                        radius_width=item.radius_width,
                        angle_width=item.angle_width,
                        is_ring=item.is_ring,
                        score=item.score,
                    )
                    new_ids.append(int(new_id))
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                # Re-add to the existing selection (additive, matching
                # the original paste).
                self._add_detected_to_selection(entry, int(frame), new_ids)
                self.pipeline_panel.append_log(
                    f"Redo paste: re-added {len(new_ids)} detected "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo paste", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_paste_peaks_to_range(self) -> None:
        """Paste the clipboard onto every frame in a typed range.

        Ctrl+Shift+V entry point. Mirrors ``_on_paste_peaks`` but
        loops the frame set produced by ``frame_range.parse_frame_range``.
        ``take_items`` is non-draining, so the same snapshot is replayed
        on every target frame and remains available for further pastes.

        UX rules (confirmed with the user):
          * Out-of-range frames are filtered with a confirmation dialog
            naming the count + compact range list.
          * Duplicates in the input deduplicate silently.
          * After a successful paste, only the current frame's rows
            (if it landed in the range) get added to the multi-selection.
          * Cancel mid-loop keeps already-written frames.
          * One grouped ``_CallbackAction`` reverses the entire batch.
        """
        if self._is_busy():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        n_frames = int(self.viewer.n_frames)
        if n_frames <= 0:
            return
        text, ok = QInputDialog.getText(
            self, "Paste to frames",
            f"Frames (e.g. 0-{max(n_frames - 1, 0)},37):",
            text=str(int(self.viewer.current_frame)),
        )
        if not ok:
            return
        try:
            valid, dropped = frame_range.parse_frame_range(
                text, n_frames=n_frames,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Paste to frames", str(exc))
            return
        if dropped:
            reply = QMessageBox.question(
                self, "Paste to frames",
                f"Pasting to {len(valid)} frame(s). "
                f"{len(dropped)} frame(s) outside the {n_frames}-frame "
                f"stack will be skipped: "
                f"{frame_range.compact_repr(dropped)}.\n\nContinue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not valid:
            self.statusBar().showMessage(
                "Paste to frames: nothing to do — no frames in range",
                4000,
            )
            return

        n_frames_to_write = len(valid)
        dialog = skin_progress(QProgressDialog(
            f"Pasting to {n_frames_to_write} frame(s)…",
            "Cancel", 0, n_frames_to_write, self,
        ))
        dialog.setWindowTitle("Paste to frames")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        per_frame_ids: dict[int, list[int]] = {}
        cancelled = False
        with self._detached_silx_tree():
            try:
                for i, target_frame in enumerate(valid):
                    if dialog.wasCanceled():
                        cancelled = True
                        break
                    dialog.setLabelText(
                        f"Pasting frame {i + 1}/{n_frames_to_write} "
                        f"(frame {target_frame:05d})…"
                    )
                    QApplication.processEvents()
                    frame_ids: list[int] = []
                    for item in items:
                        new_id = file_model.add_detected_peak_row(
                            self.session.temp_path, entry,
                            int(target_frame),
                            radius=item.radius,
                            angle=item.angle,
                            radius_width=item.radius_width,
                            angle_width=item.angle_width,
                            is_ring=item.is_ring,
                            score=item.score,
                        )
                        frame_ids.append(int(new_id))
                    per_frame_ids[int(target_frame)] = frame_ids
                    dialog.setValue(i + 1)
                    QApplication.processEvents()
            except KeyError as exc:
                dialog.close()
                dialog.deleteLater()
                QMessageBox.warning(self, "Paste to frames", str(exc))
                return
            except Exception as exc:
                dialog.close()
                dialog.deleteLater()
                QMessageBox.critical(self, "Paste to frames", str(exc))
                return
        dialog.close()
        dialog.deleteLater()

        if not per_frame_ids:
            return

        self.session.mark_dirty()
        self._update_title()
        # Append-only across the range (fresh ids per frame); prior undo
        # history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Only the current frame's pasted rows get added to the
        # selection (per confirmed UX). Off-frame selections would
        # be pruned on frame switch anyway.
        current_frame = int(self.viewer.current_frame)
        if current_frame in per_frame_ids:
            self._add_detected_to_selection(
                entry, current_frame, per_frame_ids[current_frame],
            )
        self._push_paste_to_range_undo(
            entry=entry, items=items, per_frame_ids=per_frame_ids,
        )

        n_rows = sum(len(v) for v in per_frame_ids.values())
        n_frames_written = len(per_frame_ids)
        summary = (
            f"Pasted {n_rows} detected peak(s) across "
            f"{n_frames_written} frame(s) on {entry}"
        )
        if dropped:
            summary += f" ({len(dropped)} out-of-range frame(s) skipped)"
        if cancelled:
            summary += " (cancelled)"
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)

    def _push_paste_to_range_undo(
        self, *, entry: str, items: list,
        per_frame_ids: dict[int, list[int]],
    ) -> None:
        """Undo / redo for a range paste — one grouped action.

        ``state["per_frame_ids"]`` rotates on every redo so a follow-on
        undo deletes the correct (freshly-assigned) ids. Same idea as
        ``_push_paste_undo`` extended from a single frame to a frame
        dict; frames the original Cancel skipped never enter the dict
        so they're left alone on undo/redo.
        """
        state: dict[str, dict[int, list[int]]] = {
            "per_frame_ids": {
                int(f): [int(i) for i in ids]
                for f, ids in per_frame_ids.items()
            },
        }
        snapshot = list(items)

        def _undo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            with self._detached_silx_tree():
                try:
                    for f, ids in state["per_frame_ids"].items():
                        for peak_id in ids:
                            file_model.delete_peak_row(
                                self.session.temp_path, entry,
                                frame=int(f), kind="detected",
                                peak_id=int(peak_id),
                            )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo paste to frames", str(exc)
                    )
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in state["per_frame_ids"]:
                self._drop_detected_from_selection(
                    current_frame, state["per_frame_ids"][current_frame],
                )
            n_rows = sum(len(v) for v in state["per_frame_ids"].values())
            n_frames_written = len(state["per_frame_ids"])
            self.pipeline_panel.append_log(
                f"Undo paste to frames: removed {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        def _redo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            new_per_frame_ids: dict[int, list[int]] = {}
            with self._detached_silx_tree():
                try:
                    for f in sorted(state["per_frame_ids"].keys()):
                        frame_ids: list[int] = []
                        for item in snapshot:
                            new_id = file_model.add_detected_peak_row(
                                self.session.temp_path, entry, int(f),
                                radius=item.radius,
                                angle=item.angle,
                                radius_width=item.radius_width,
                                angle_width=item.angle_width,
                                is_ring=item.is_ring,
                                score=item.score,
                            )
                            frame_ids.append(int(new_id))
                        new_per_frame_ids[int(f)] = frame_ids
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo paste to frames", str(exc)
                    )
                    return
            state["per_frame_ids"] = new_per_frame_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in new_per_frame_ids:
                self._add_detected_to_selection(
                    entry, current_frame, new_per_frame_ids[current_frame],
                )
            n_rows = sum(len(v) for v in new_per_frame_ids.values())
            n_frames_written = len(new_per_frame_ids)
            self.pipeline_panel.append_log(
                f"Redo paste to frames: re-added {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Batch 2D fit ------------------------------------------------

    def _refresh_fit_buttons(self) -> None:
        """Re-evaluate visibility + enable state for the fit buttons.

        Visibility rule (per user request — only show the button that
        makes sense, don't grey them out):

        * Exactly one fittable peak selected (manual or detected) →
          'Add to fitted' visible, 'Fit selected (2D)' hidden.
        * Two or more detected peaks selected → 'Fit selected (2D)'
          visible, 'Add to fitted' hidden.
        * Otherwise (no selection, only fitted / matched, or the
          mixed-kind edge case of 1 detected + 1 manual which neither
          button handles cleanly) → both hidden.

        'Add to detected' stays in place — its enable state is driven
        by ``ParameterPanel._update_actions_enabled`` (greyed out for
        non-manual selections).

        Enable state for 'Fit selected (2D)' is still gated on
        save-as-ring being OFF (ring forces 1D).
        """
        sels = self.viewer.selected_peaks()
        n_fittable = sum(1 for s in sels if s.kind in ("manual", "detected"))
        n_detected = sum(1 for s in sels if s.kind == "detected")
        ring_on = self.parameter_panel.save_as_ring()
        show_add_fitted = (n_fittable == 1)
        show_batch_fit = (n_detected >= 2 and not ring_on)
        self.parameter_panel.set_fit_button_visibility(
            add_fitted=show_add_fitted,
            fit_selected_2d=show_batch_fit,
        )
        # Keep the enable state coherent so a click on the visible
        # batch-fit button always proceeds (otherwise a stale enable=False
        # from before the visibility change could leak through).
        self.parameter_panel.set_batch_fit_enabled(show_batch_fit)
        # ≥2 fittable selected → batch fit is the only sensible
        # action, and batch fit is 2D-only. Grey out the 1D radio
        # to make that constraint visible.
        self.parameter_panel.set_multi_select_active(n_fittable >= 2)

    def _on_batch_fit_2d(self) -> None:
        """Run pygidfit on every selected detected peak, in one batch.

        Two-phase to keep the viewer's FrameSource open while pygidfit
        reads frame data:

        1. **Fit phase** (no silx detach): loop the selections,
           calling ``_run_pygidfit_for_selection`` which needs
           ``viewer._frame_source.is_open``. The QProgressDialog
           ticks per fit and respects Cancel.
        2. **Write phase** (single ``_detached_silx_tree`` scope):
           append a ``fitted_peaks`` row for every successful fit.
           h5py detach is expensive; doing it once for the whole
           batch keeps the run snappy.
        """
        if self._is_busy():
            return
        sels = [
            s for s in self.viewer.selected_peaks() if s.kind == "detected"
        ]
        if not sels:
            self.statusBar().showMessage(
                "Fit selected (2D): select detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        n = len(sels)
        dialog = skin_progress(QProgressDialog(
            f"Fitting {n} peak(s) (2D)…", "Cancel", 0, n, self,
        ))
        dialog.setWindowTitle("Fit selected (2D)")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        # Phase 1: fits. Each ``_run_pygidfit_for_selection`` call
        # reads ``viewer._frame_source.get_cartesian(frame)``, so the
        # silx tree must stay attached here (the detach in phase 2
        # closes the frame source).
        fit_results: list[tuple[SelectedPeak, object, str | None]] = []
        cancelled = False
        for i, sel in enumerate(sels):
            if dialog.wasCanceled():
                cancelled = True
                break
            dialog.setLabelText(
                f"Fitting peak {i + 1}/{n} (id={sel.peak_id})…"
            )
            QApplication.processEvents()
            fit_2d, err = self._run_pygidfit_for_selection(
                sel, entry, int(sel.frame),
            )
            fit_results.append((sel, fit_2d, err))
            dialog.setValue(i + 1)
            QApplication.processEvents()

        # Phase 2: writes. One ``_detached_silx_tree`` scope for the
        # whole batch — h5py detach is the expensive bit.
        added_ids: list[int] = []
        added_kwargs: list[dict] = []
        target_ids: list[int | None] = []
        replaced_rows: list[dict] = []
        failures: list[tuple[int, str]] = []
        link = self._fitted_link_on()
        with self._detached_silx_tree():
            for sel, fit_2d, err in fit_results:
                if fit_2d is None:
                    failures.append((int(sel.peak_id), err or "fit failed"))
                    continue
                # An infinite angular width is how pygidfit reports a
                # ring: store it as one.
                fit_angle, fit_is_ring = file_model.ring_if_infinite_width(
                    fit_2d.angle, fit_2d.angle_width, False,
                )
                row = {
                    "radius": float(fit_2d.radius),
                    "radius_width": float(fit_2d.radius_width),
                    "angle": fit_angle,
                    "angle_width": float(fit_2d.angle_width),
                    "amplitude": float(fit_2d.amplitude),
                    "is_ring": fit_is_ring,
                    "theta": float(fit_2d.theta),
                    "A": float(fit_2d.A),
                    "B": float(fit_2d.B),
                    "C": float(fit_2d.C),
                }
                # With the link on each fit takes its detected peak's
                # id and replaces that peak's previous fit; with it off
                # this is the plain append it always was.
                target_id = int(sel.peak_id) if link else None
                try:
                    new_id, replaced = self._write_fitted_row(
                        entry, int(sel.frame), row, target_id,
                    )
                except Exception as exc:
                    failures.append((int(sel.peak_id), f"write failed: {exc}"))
                    continue
                added_ids.append(int(new_id))
                added_kwargs.append({**row, "frame": int(sel.frame)})
                target_ids.append(target_id)
                # Stamp the frame: a batch selection can span frames, and
                # undo has to put each displaced fit back on its own.
                replaced_rows.extend(
                    {**d, "frame": int(sel.frame)} for d in replaced
                )
        dialog.close()
        dialog.deleteLater()

        if not added_ids:
            QMessageBox.warning(
                self, "Fit selected (2D)",
                "No peaks were fitted successfully.\n\n"
                + "\n".join(f"id={pid}: {msg}" for pid, msg in failures),
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Batch fit appends fitted rows; with the link on each takes its
        # detected peak's id and displaces that peak's previous fit,
        # which the undo entry restores. The detected rows themselves
        # are untouched either way, so prior undo history stays valid.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Multi-select the newly-added fitted rows. They may span
        # multiple frames if the original detected selection did
        # (rare — multi-select today is per-frame — but kept correct
        # for future when cross-frame multi-select lands).
        frames_touched = sorted({kw["frame"] for kw in added_kwargs})
        if frames_touched:
            # Select on the first touched frame; the user can navigate.
            self._select_fitted_ids(
                entry, frames_touched[0],
                [
                    new_id for new_id, kw in zip(added_ids, added_kwargs)
                    if kw["frame"] == frames_touched[0]
                ],
            )
        self._push_batch_fit_undo(
            entry=entry, added_ids=added_ids, added_kwargs=added_kwargs,
            target_ids=target_ids, replaced_rows=replaced_rows,
        )

        summary = (
            f"Fit {len(added_ids)}/{n} selected peaks "
            f"({len(failures)} failed"
            + (", cancelled" if cancelled else "")
            + ")"
        )
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)
        for pid, msg in failures:
            self.pipeline_panel.append_log(f"  fit failed: id={pid} — {msg}")

    def _select_fitted_ids(
        self, entry: str, frame: int, ids: list[int],
    ) -> None:
        """Multi-select the fitted rows with the given ids on ``frame``."""
        if not ids:
            return
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        fit = tables.get("fitted")
        if fit is None:
            return
        id_set = {int(x) for x in ids}
        sels = []
        for i in range(len(fit)):
            if int(fit.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind="fitted", frame=int(frame),
                    peak_id=int(fit.ids[i]),
                    radius=float(fit.radius[i]),
                    angle=float(fit.angle[i]),
                    radius_width=float(fit.radius_width[i]),
                    angle_width=float(fit.angle_width[i]),
                    is_ring=bool(fit.is_ring[i]),
                    score=float(fit.score[i]),
                    amplitude=float(fit.amplitude[i]),
                ))
        if not sels:
            return
        # Fitted is not currently in the extras allowlist (extras are
        # detected-only this iteration), so set the primary alone here.
        # Multi-selection of fitted peaks lands when fitted copy/paste
        # is enabled in a follow-up.
        self.viewer._set_selected(sels[0], preserve_manual=True)

    def _push_batch_fit_undo(
        self, *, entry: str, added_ids: list[int], added_kwargs: list[dict],
        target_ids: "list[int | None] | None" = None,
        replaced_rows: list[dict] | None = None,
    ) -> None:
        """Undo / redo for a batch-fit run — one grouped action.

        Mirrors ``_push_fitted_add_undo`` extended to N rows. The
        rotating ``state["ids"]`` keeps undo correct after a redo
        re-assigns ids (one per redo'd row); with the link on the ids
        are pinned to the detected peaks and do not move.

        ``replaced_rows`` are the fits this run displaced (one per
        already-fitted detected peak in the selection). Undo puts them
        back under their own ids, so a batch refit is as reversible as a
        single one.
        """
        state: dict = {
            "ids": list(added_ids),
            "replaced": [dict(d) for d in (replaced_rows or [])],
        }
        snapshot_kwargs = [dict(kw) for kw in added_kwargs]
        targets = list(target_ids or [None] * len(added_kwargs))

        def _undo() -> None:
            def write():
                for kw, peak_id in zip(snapshot_kwargs, state["ids"]):
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(kw["frame"]), kind="fitted",
                        peak_id=int(peak_id),
                    )
                # Cascading clear on every touched frame mirrors
                # the single-fitted-delete behaviour.
                for f in sorted({int(kw["frame"]) for kw in snapshot_kwargs}):
                    file_model.clear_peaks(
                        self.session.temp_path, entry,
                        kind="matched", frame=f,
                    )
                # Restore the fits this run displaced, under their own
                # ids so each is paired with its detected peak again.
                for d in state["replaced"]:
                    row = dict(d)
                    peak_id = row.pop("id", None)
                    row_frame = int(row.pop("frame"))
                    self._readd_peak_row(
                        entry, row_frame, "fitted", row, peak_id=peak_id,
                    )

            def after(_result) -> None:
                self.pipeline_panel.append_log(
                    f"Undo batch fit: removed {len(state['ids'])} fitted "
                    f"row(s) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Undo batch fit", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                new_ids: list[int] = []
                re_replaced: list[dict] = []
                for kw, target in zip(snapshot_kwargs, targets):
                    write_kw = {k: v for k, v in kw.items() if k != "frame"}
                    new_id, replaced = self._write_fitted_row(
                        entry, int(kw["frame"]), write_kw, target,
                    )
                    new_ids.append(int(new_id))
                    for d in replaced:
                        re_replaced.append({**d, "frame": int(kw["frame"])})
                state["replaced"] = re_replaced
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                self.pipeline_panel.append_log(
                    f"Redo batch fit: re-added {len(new_ids)} fitted "
                    f"row(s) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Redo batch fit", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _run_pygidfit_for_selection(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> tuple["manual_fit.ManualFitResult | None", str | None]:
        """Run pygidfit on ``sel`` now and return ``(result, error)``.

        The blocking form, for the commit paths (Add to fitted, the
        batch fit, quick select), where the user asked for the fit and
        waits for it. The live preview does not call this: it takes the
        same two steps separately so the slow one can run off the GUI
        thread (``_prepare_pygidfit_fit`` here, the returned job on the
        ``PreviewFitRunner``).
        """
        job, err = self._prepare_pygidfit_fit(sel, entry, frame)
        if job is None:
            return None, err
        return job()

    def _prepare_pygidfit_fit(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> tuple["Callable[[], tuple] | None", str | None]:
        """Gather everything pygidfit needs from the GUI, return a job.

        Returns ``(job, None)``, where ``job()`` runs the fit and
        returns ``(result, error)`` using only the data captured here
        (no widget, viewer or session access), so it is safe on another
        thread; or ``(None, error)`` when the inputs are not available.
        All the reads that touch Qt objects or the open file happen in
        this call, on the thread that calls it.

        Silent — no QMessageBox, no logging at warning level. When
        ``result`` is None, ``error`` is a short human-readable string
        the caller can surface or ignore.

        Replicates ``pygidfit.ProcessDataFromFile.process_single_frame``
        — same polar resolution (``512 × 1024``), ``img_preprocessing``
        masking, ``crit_angle`` / ``theta_fixed`` / ``clustering_*``
        values pulled from the Pipeline panel — so the preview's box
        equals what the commit would save.
        """
        from mlgidgui.manual_fit import ManualFitError, fit_one_peak

        fs = getattr(self.viewer, "_frame_source", None)
        stack = getattr(self.viewer, "_stack", None)
        if fs is None or not fs.is_open or stack is None:
            return None, (
                "The active frame isn't currently available "
                "(viewer mid-pipeline or released)."
            )
        try:
            # A copy: the job may run after the viewer has moved on, so
            # it must not share a buffer the viewer can reuse.
            cartesian = np.array(fs.get_cartesian(int(frame)))
        except Exception as exc:
            return None, f"Could not load the cartesian frame: {exc}"
        q_xy = np.array(stack.q_xy)
        q_z = np.array(stack.q_z)

        if self.session is None:
            return None, "No active session."
        geom = file_model.read_geometry_for_entry(
            self.session.temp_path, entry, frame=int(frame),
        )
        if geom is None:
            return None, (
                "Instrument metadata (wavelength / angle of incidence "
                "/ q ranges) is missing or malformed for this entry."
            )
        geom = dict(geom)
        geom.pop("q_z_axis", None)

        panel = self.pipeline_panel
        try:
            crit_angle = float(panel.fit_crit_angle.value())
            cdp = float(panel.fit_dist_peaks.value())
            cdr = float(panel.fit_dist_rings.value())
            ce = int(panel.fit_cluster_extend.value())
            tf = bool(panel.fit_theta_fixed.isChecked())
        except Exception:
            crit_angle, tf = 0.0, True
            cdp = DEFAULT_CLUSTERING_DISTANCE_PEAKS
            cdr = DEFAULT_CLUSTERING_DISTANCE_RINGS
            ce = DEFAULT_CLUSTERING_EXTEND

        neighbours = self._neighbour_boxes(entry, int(frame), sel)
        radius = float(sel.radius)
        radius_width = float(sel.radius_width)
        angle = float(sel.angle)
        angle_width = float(sel.angle_width)

        def job():
            try:
                fit_2d = fit_one_peak(
                    cartesian, q_xy, q_z,
                    radius=radius,
                    radius_width=radius_width,
                    angle=angle,
                    angle_width=angle_width,
                    crit_angle=crit_angle,
                    theta_fixed=tf,
                    clustering_distance_peaks=cdp,
                    clustering_distance_rings=cdr,
                    clustering_extend=ce,
                    neighbours=neighbours,
                    **geom,
                )
            except ManualFitError as exc:
                return None, str(exc)
            except Exception as exc:
                return None, f"Unexpected 2D fit error: {exc!r}"
            return fit_2d, None

        return job, None

    def _profile_neighbour_boxes(self, frame: int, sel: SelectedPeak) -> list:
        """The peaks to mask out of ``sel``'s profiles, or [] when the
        Settings option is off. Same neighbour set the 2D fit hands to
        pygidfit (``_neighbour_boxes``); the setting is read here, at
        the moment the profiles are drawn."""
        if not read_bool(
            PROFILE_MASK_NEIGHBOURS_KEY, DEFAULT_PROFILE_MASK_NEIGHBOURS,
        ):
            return []
        entry = self.entry_combo.currentText()
        if not entry:
            return []
        return self._neighbour_boxes(entry, int(frame), sel)

    def _neighbour_boxes(
        self, entry: str, frame: int, sel: SelectedPeak,
    ) -> list:
        """The frame's other peak boxes, for pygidfit to mask or
        joint-fit alongside ``sel``.

        pygidfit only reasons about the boxes in one ``fit_data``
        call, so handing it just the user's box left neighbouring
        intensity unmasked in the ROI (see ``manual_fit``). Detected
        and fitted rows both count; ``select_neighbour_boxes`` inside
        the wrapper drops rings, duplicates and everything too far
        away to touch the target's ROI.

        Source is the viewer's per-frame peak cache — the tables the
        overlays already hold, so a click costs no I/O. Only when the
        frame is not cached (a batch fit reaching a frame the user
        never opened) does this fall back to a read, and a failed
        read means no neighbours rather than a failed fit.
        """
        from mlgidgui.manual_fit import neighbour_boxes_from_tables

        tables = (getattr(self.viewer, "_frame_peaks", None) or {}).get(frame)
        if not tables:
            if self.session is None:
                return []
            try:
                tables = file_model.load_peaks(
                    self.session.temp_path, entry, int(frame),
                )
            except Exception:
                return []
        return neighbour_boxes_from_tables(
            tables,
            exclude_kind=sel.kind,
            exclude_id=getattr(sel, "peak_id", None),
        )

    def _build_fitted_row_2d(
        self, sel: SelectedPeak, entry: str, frame: int,
        *, quiet: bool = False,
    ) -> dict | None:
        """Run pygidfit on ``sel`` and shape the result into an
        ``add_fitted_peak_row`` kwarg dict. Surfaces a QMessageBox
        on failure — strict, no silent fall-back to the 1D path.
        Delegates the actual fit to ``_run_pygidfit_for_selection``
        so the live-preview path can reuse the same code.

        ``quiet`` downgrades that dialog to a log line, for the
        quick-select commit: the user has already moved on to the next
        box, and a modal in the middle of a labelling run stops the
        run dead. The caller then decides what to do with the failure
        (``_quick_write`` keeps the box as a detected peak).
        """
        fit_2d, err = self._run_pygidfit_for_selection(sel, entry, frame)
        if fit_2d is None:
            if quiet:
                self.pipeline_panel.append_log(
                    f"Quick select: pygidfit could not fit this box — {err}"
                )
                return None
            QMessageBox.warning(
                self, "Add to fitted (2D)",
                f"pygidfit could not fit this box: {err}\n\n"
                "Try widening the box, or switch to '1D fit (scipy)' "
                "mode to commit with the legacy 1D Gaussian fit.",
            )
            return None
        return self._fitted_row_from_fit(fit_2d)

    @staticmethod
    def _fitted_row_from_fit(fit_2d) -> dict:
        """The ``add_fitted_peak_row`` kwargs for a pygidfit result."""
        return {
            "radius": fit_2d.radius,
            "radius_width": fit_2d.radius_width,
            "angle": fit_2d.angle,
            "angle_width": fit_2d.angle_width,
            "amplitude": fit_2d.amplitude,
            "theta": fit_2d.theta,
            "A": fit_2d.A, "B": fit_2d.B, "C": fit_2d.C,
        }

    def _build_fitted_row_1d(
        self, sel: SelectedPeak, save_as_ring: bool, *, quiet: bool = False,
    ) -> dict | None:
        """Build the row from the profile viewer's cached 1D scipy fits.

        Width convention matches the 2D path so the saved blue box for
        the same physical Gaussian renders identically regardless of
        which mode produced it: ``radius_width = angle_width = 2σ =
        FWHM / sqrt(2 ln 2) ≈ 0.849 × FWHM``. This is pygidfit's
        convention (see ``manual_fit.fit_one_peak``) and what the
        pipeline ``run_fitting`` writes. The only difference between
        modes is whether 2D shape coefficients (A/B/C/theta) carry
        real values — 1D zero-fills them.

        Ring storage keeps ``angle = 45°``, ``angle_width = inf`` as
        the sentinel — not a Gaussian width, so the unified
        convention doesn't apply.

        Returns ``None`` and shows a QMessageBox when the required 1D
        fit isn't available (typical cause: narrow detected box where
        scipy didn't converge). Strict — no fall-back to pygidfit.
        """
        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        missing = None
        if rfit is None:
            missing = (
                "No radial Gaussian fit is available. Drag the box "
                "until the pink fit curve appears in the radial "
                "profile, or switch to '2D fit (pygidfit)' mode."
            )
        elif not save_as_ring and afit is None:
            missing = (
                "No angular Gaussian fit is available — required for "
                "segment peaks in 1D mode. Drag the box until the "
                "pink fit curve appears in the angular profile, "
                "check 'Save fitted as ring' if this is a ring, or "
                "switch to '2D fit (pygidfit)' mode."
            )
        if missing is not None:
            if quiet:
                self.pipeline_panel.append_log(
                    f"Quick select: no 1D fit for this box — {missing}"
                )
            else:
                QMessageBox.warning(self, "Add to fitted (1D)", missing)
            return None
        fwhm_to_2sigma = 1.0 / float(np.sqrt(2.0 * np.log(2.0)))
        if save_as_ring:
            angle_to_save = 45.0
            angle_width_to_save = float("inf")
        else:
            angle_to_save = float(afit.center)
            angle_width_to_save = float(afit.fwhm) * fwhm_to_2sigma
        return {
            "radius": float(rfit.center),
            "radius_width": float(rfit.fwhm) * fwhm_to_2sigma,
            "angle": angle_to_save,
            "angle_width": angle_width_to_save,
            "amplitude": float(rfit.amplitude),
            "theta": 0.0, "A": 0.0, "B": 0.0, "C": 0.0,
        }

    def _on_delete_peak_requested(self, sel: SelectedPeak | None) -> None:
        """Confirm + delete a non-manual peak — kind-scoped, no cascade.

        Replaces the previous ``mlgidbase.delete_peak`` dispatch
        (which removed the row from detected + fitted + matched in
        one shot, leaving the user unable to delete just the fitted
        prediction without also wiping its detected source). Now:

        * ``sel.kind == "detected"`` → delete the detected row, plus
          its fit when the fitted/detected link is on (the fit belongs
          to the detection; keeping it would leave a prediction with
          nothing behind it).
        * ``sel.kind == "fitted"`` → delete only the fitted row — and
          the detected row too, but only when the user has explicitly
          asked for that second setting. Then
          drop the row from ``matched_*`` ``peak_list`` so each surviving
          structure keeps its other peaks (the deleted peak is dropped
          from any structure that referenced it; structures that didn't
          are untouched). Replaces the old blunt "clear every
          ``matched_*`` on the frame" cascade.
        * ``sel.kind == "matched"`` → removes the id from every
          ``matched_*`` ``peak_list`` on the frame
          (``file_model.drop_ids_from_matched``), undoably. Detected
          and fitted rows are not touched; no mlgidbase call.
        """
        if (
            sel is None or sel.kind == "manual"
            or self.session is None or self._pipe_thread is not None
        ):
            return
        # A registered extra peak list falls through to the ``matched``
        # branch below if it gets this far, which would hand mlgidbase a
        # kind it has never heard of. The button is disabled for it, so
        # this only catches the keyboard route -- but that is exactly
        # the one nobody would think to check.
        if peak_lists.is_list_kind(sel.kind):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return

        if sel.kind in ("detected", "fitted"):
            return self._delete_file_peak_scoped(sel, entry)

        # matched: take the peak out of the matched structures only.
        # (mlgidbase.delete_peak would delete the detected and fitted
        # rows with that id too, whatever the old dialog promised.)
        frame = int(sel.frame)
        peak_id = int(sel.peak_id)
        reply = QMessageBox.question(
            self,
            "Delete matched peak",
            (
                f"Remove peak id={peak_id} on frame {frame} from the "
                "matched structures?\n\n"
                "Every matched structure on this frame that includes it "
                "loses just this peak; the structures themselves stay. "
                "The detected and fitted peaks are not touched. This "
                "can be undone with Ctrl+Z."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        path = self.session.temp_path
        with self._detached_silx_tree():
            try:
                snapshot = file_model.snapshot_matched(path, entry, frame)
                changed = file_model.drop_ids_from_matched(
                    path, entry, frame, [peak_id],
                )
            except Exception as exc:
                QMessageBox.critical(self, "Delete matched peak", str(exc))
                return
        if changed == 0:
            self.pipeline_panel.append_log(
                f"Delete matched: no matched structure on "
                f"{entry}/frame{frame:05d} includes peak {peak_id}"
            )
            return
        self.session.mark_dirty()
        self._update_title()
        self.viewer.clear_selection()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._push_matched_delete_undo(
            entry=entry, frame=frame, peak_id=peak_id, snapshot=snapshot,
        )
        msg = (
            f"Removed peak {peak_id} from {changed} matched structure(s) "
            f"on {entry}/frame{frame:05d}"
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _push_matched_delete_undo(
        self, *, entry: str, frame: int, peak_id: int, snapshot: dict,
    ) -> None:
        """Undo / redo for a matched-only delete: ``undo`` writes the
        frame's ``matched_*`` datasets back exactly as they were,
        ``redo`` drops the id again."""

        def _undo() -> None:
            def write():
                file_model.restore_matched(
                    self.session.temp_path, entry, frame, snapshot,
                )

            def after(_result) -> None:
                self.pipeline_panel.append_log(
                    f"Undo delete matched: restored matched structures on "
                    f"{entry}/frame{frame:05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo delete matched", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                file_model.drop_ids_from_matched(
                    self.session.temp_path, entry, frame, [peak_id],
                )

            def after(_result) -> None:
                self.pipeline_panel.append_log(
                    f"Redo delete matched: removed peak {peak_id} from the "
                    f"matched structures on {entry}/frame{frame:05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo delete matched", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _delete_file_peak_scoped(
        self, sel: SelectedPeak, entry: str,
    ) -> None:
        """Delete one detected/fitted row — kind-scoped, undoable.

        Fitted deletes also drop the row from every ``matched_*``
        ``peak_list`` on the frame (stored as fitted ids, so nothing
        shifts; older positional datasets are migrated on the way),
        keeping the surviving structures. Delegates to the
        shared ``_delete_peaks_scoped`` so single and bulk delete share
        one undoable path.
        """
        self._delete_peaks_scoped([sel], entry)

    def _on_delete_peaks_requested(self, sels: list[SelectedPeak]) -> None:
        """Bulk-delete an all-detected OR all-fitted multi-selection.

        Triggered by Delete when >= 2 same-kind peaks are selected
        (``deletePeaksRequested``). One confirmation, one grouped,
        undoable action — see ``_delete_peaks_scoped``.
        """
        kinds = {s.kind for s in sels}
        if len(sels) < 2 or len(kinds) != 1:
            return
        if next(iter(kinds)) not in ("detected", "fitted"):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._delete_peaks_scoped(list(sels), entry)

    def _delete_peaks_scoped(
        self, sels: list[SelectedPeak], entry: str,
    ) -> None:
        """Shared, undoable delete for 1..N same-kind detected/fitted peaks.

        Confirms once, snapshots the full rows (so undo can re-create
        them verbatim — including the 2D-fit params PeakTable doesn't
        surface), deletes them in one ``_detached_silx_tree`` scope,
        cleans ``matched_*`` ``peak_list`` on the frame for fitted
        deletes (dropping the deleted peaks from any structure that
        referenced them, keeping every structure), and pushes one
        grouped ``_CallbackAction`` so a single Ctrl+Z restores every
        row. Undo re-adds the fitted rows with fresh ids appended at the
        end; it does **not** re-add them to the matched structures they
        were in (the structures already reference the surviving peaks
        correctly — re-run Matching if a peak's membership matters).
        """
        if self._is_busy():
            return
        if not sels or not entry:
            return
        kind = sels[0].kind
        if kind not in ("detected", "fitted"):
            return
        frame = int(self.viewer.current_frame)
        on_frame = [
            s for s in sels if s.kind == kind and int(s.frame) == frame
        ]
        if not on_frame:
            return
        ids = [int(s.peak_id) for s in on_frame]
        n = len(ids)
        kind_human = "detected" if kind == "detected" else "fitted"
        # The link pairs a fit with its detected peak by id, so a
        # delete on one side can take the other with it. Deleting a
        # detection always takes its fit; the reverse needs the second
        # setting, which is off by default. With the link off, partner
        # is None and this is the kind-scoped delete it always was.
        from mlgidgui import peak_link

        link = peak_link.link_enabled()
        partner: str | None = None
        if link and kind == "detected":
            partner = "fitted"
        elif kind == "fitted" and peak_link.reverse_delete_enabled():
            partner = "detected"
        matched_msg = (
            "Matched structures that include the deleted peak(s) lose "
            "just those peaks; structures that don't reference them are "
            "left unchanged."
        )
        if kind == "detected":
            siblings_msg = (
                "The matching fitted peak(s) go too. " + matched_msg
                if partner else "Fitted and matched rows stay intact."
            )
        else:
            siblings_msg = (
                "The matching detected peak(s) go too. " + matched_msg
                if partner
                else "Detected rows stay intact. " + matched_msg
            )
        target = (
            f"id={ids[0]} on frame {frame}" if n == 1
            else f"{n} {kind_human} peaks on frame {frame}"
        )
        reply = QMessageBox.question(
            self,
            f"Delete {kind_human} peak" + ("s" if n > 1 else ""),
            f"Delete {target}?\n\n{siblings_msg} This can be undone "
            "with Ctrl+Z.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        partner_removed = 0
        snapshot: list[dict] = []
        partner_snapshot: list[dict] = []
        matched_changed = 0
        # Whichever side is the fitted one — the kind being deleted or
        # its partner — needs the matched clean-up.
        fitted_involved = kind == "fitted" or partner == "fitted"
        with self._detached_silx_tree():
            try:
                # Snapshot full rows (incl. theta/A/B/C for fitted) so
                # undo can re-create them. Read inside the detached
                # scope so no handle contends with silx. Ids come along
                # when the link is on: a pair restored under fresh ids
                # would no longer be a pair.
                snapshot = file_model.read_peak_rows(
                    self.session.temp_path, entry, frame, kind, ids,
                    with_ids=link, with_errors=True,
                )
                if partner is not None:
                    partner_snapshot = file_model.read_peak_rows(
                        self.session.temp_path, entry, frame, partner, ids,
                        with_ids=True, with_errors=True,
                    )
                # matched_* peak_list may store positions into
                # fitted_peaks (older files) and deleting compacts the
                # array — prepare_fitted_delete captures the rows'
                # positions BEFORE deleting and puts matched into id
                # form, so the remap afterwards only drops ids.
                deleted_positions = (
                    file_model.prepare_fitted_delete(
                        self.session.temp_path, entry, frame, ids,
                    )
                    if fitted_involved else []
                )
                for peak_id in ids:
                    removed += file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=frame, kind=kind, peak_id=peak_id,
                    )
                if partner is not None:
                    for peak_id in ids:
                        partner_removed += file_model.delete_peak_row(
                            self.session.temp_path, entry,
                            frame=frame, kind=partner, peak_id=peak_id,
                        )
                if fitted_involved and (removed + partner_removed) > 0:
                    # Drop the deleted peaks from any matched structure
                    # that referenced them; structures are kept (the old
                    # code wiped every matched_* on the frame instead).
                    matched_changed = file_model.remap_matched_peak_lists(
                        self.session.temp_path, entry, frame,
                        deleted_positions,
                    )
            except Exception as exc:
                QMessageBox.critical(
                    self, f"Delete {kind_human} peak", str(exc),
                )
                return
        if removed == 0:
            self.pipeline_panel.append_log(
                f"Delete {kind_human}: no matching rows on "
                f"{entry}/frame{frame:05d} (already gone?)"
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Drop only the deleted ids' pending geom edits (they would key
        # off rows that no longer exist); prior undoable ops stay on the
        # stack so a run of deletes remains fully undoable / redoable.
        self.viewer.prune_file_geom_history(frame, kind, ids)
        if partner is not None:
            self.viewer.prune_file_geom_history(frame, partner, ids)
        self.viewer.clear_selection()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._push_delete_undo(
            entry=entry, frame=frame, kind=kind, snapshot=snapshot,
            cascade_matched=fitted_involved,
            partner_kind=partner, partner_snapshot=partner_snapshot,
            keep_ids=link,
        )
        msg = (
            f"Deleted {removed} {kind_human} peak(s) on "
            f"{entry}/frame{frame:05d}"
            + (
                f" (+{partner_removed} linked {partner} peak(s))"
                if partner_removed else ""
            )
            + (
                f" ({matched_changed} matched structure(s) updated)"
                if matched_changed else ""
            )
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _readd_peak_row(
        self, entry: str, frame: int, kind: str, d: dict,
        peak_id: int | None = None,
    ) -> int:
        """Re-create one detected/fitted row from a ``read_peak_rows``
        snapshot dict (a fitted row's ``errors`` entry, when present, is
        restored too). Returns the new id — freshly assigned unless
        ``peak_id`` names one.

        Undo passes the original id whenever the fitted/detected link is
        on: a restored pair whose ids were re-assigned would no longer
        be a pair, so the undo would not actually undo."""
        if kind == "detected":
            return file_model.add_detected_peak_row(
                self.session.temp_path, entry, int(frame),
                radius=d["radius"], angle=d["angle"],
                radius_width=d["radius_width"], angle_width=d["angle_width"],
                score=d.get("score", 0.0), is_ring=d.get("is_ring", False),
                peak_id=peak_id,
            )
        new_id = file_model.add_fitted_peak_row(
            self.session.temp_path, entry, int(frame),
            peak_id=peak_id,
            angle=d["angle"], angle_width=d["angle_width"],
            radius=d["radius"], radius_width=d["radius_width"],
            amplitude=d.get("amplitude", 0.0), is_ring=d.get("is_ring", False),
            theta=d.get("theta", 0.0), score=d.get("score", 0.0),
            A=d.get("A", 0.0), B=d.get("B", 0.0), C=d.get("C", 0.0),
            is_cut_qz=d.get("is_cut_qz", False),
            is_cut_qxy=d.get("is_cut_qxy", False),
            visibility=d.get("visibility", 0),
        )
        if d.get("errors"):
            # The delete removed the row's fit errors with it (the two
            # tables are paired by id); put them back under the id the
            # row came back with.
            file_model.restore_fitted_error_row(
                self.session.temp_path, entry, int(frame),
                int(new_id), d["errors"],
            )
        return new_id

    def _push_delete_undo(
        self, *, entry: str, frame: int, kind: str,
        snapshot: list[dict], cascade_matched: bool,
        partner_kind: str | None = None,
        partner_snapshot: list[dict] | None = None,
        keep_ids: bool = False,
    ) -> None:
        """Undo / redo for a (single or bulk) detected/fitted delete.

        The delete already happened, so ``undo`` re-adds the rows and
        ``redo`` deletes them again. ``state["ids"]`` rotates: populated
        by ``undo`` with the ids the rows came back under so a following
        ``redo`` deletes the right rows. For fitted, ``redo`` re-applies
        the ``matched_*`` clean-up (``prepare_fitted_delete`` recomputes
        the deleted rows' positions from the current ids first, since
        the re-added rows may sit at fresh positions); ``undo`` leaves
        ``matched_*`` as-is — see ``_delete_peaks_scoped``.

        ``partner_kind`` / ``partner_snapshot`` carry the other side of
        a linked delete, restored and re-deleted alongside the first.
        ``keep_ids`` restores every row under the id it had, which the
        link requires: a pair that came back under fresh ids would not
        be a pair, and the undo would not have undone. Without the link
        both are absent and the fresh-id behaviour is unchanged.
        """
        snap = [dict(d) for d in snapshot]
        partner_snap = [dict(d) for d in (partner_snapshot or [])]
        state: dict[str, list[int]] = {"ids": []}

        def _undo() -> None:
            def write():
                new_ids: list[int] = []
                for d in snap:
                    row = dict(d)
                    peak_id = row.pop("id", None) if keep_ids else None
                    new_ids.append(int(self._readd_peak_row(
                        entry, frame, kind, row, peak_id=peak_id,
                    )))
                for d in partner_snap:
                    row = dict(d)
                    self._readd_peak_row(
                        entry, frame, partner_kind, row,
                        peak_id=row.pop("id", None),
                    )
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                self._add_detected_to_selection(
                    entry, int(frame), new_ids, kind=kind,
                )
                self.pipeline_panel.append_log(
                    f"Undo delete: restored {len(new_ids)} {kind} "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo delete", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                deleted_positions = (
                    file_model.prepare_fitted_delete(
                        self.session.temp_path, entry, int(frame),
                        [int(p) for p in state["ids"]],
                    )
                    if cascade_matched else []
                )
                for peak_id in state["ids"]:
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind=kind,
                        peak_id=int(peak_id),
                    )
                    if partner_kind is not None:
                        file_model.delete_peak_row(
                            self.session.temp_path, entry,
                            frame=int(frame), kind=partner_kind,
                            peak_id=int(peak_id),
                        )
                if cascade_matched and state["ids"]:
                    file_model.remap_matched_peak_lists(
                        self.session.temp_path, entry, int(frame),
                        deleted_positions,
                    )

            def after(_result) -> None:
                self._drop_detected_from_selection(
                    int(frame), state["ids"], kind=kind,
                )
                self.pipeline_panel.append_log(
                    f"Redo delete: removed {len(state['ids'])} {kind} "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo delete", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- UI state --

    def _refresh_peaks_table(self) -> None:
        """Repopulate the Peaks dock from the viewer's current state.

        Called on entry-load completion, post-pipeline reload (via
        ``_load_entry_into_viewer``), and on every frame change (via
        ``_refresh_peaks_table_on_frame``). Empties all three tabs
        when no session is active.
        """
        if self.session is None or self.session.kind != "nexus":
            self.peaks_table_panel.clear()
            return
        frame = self.viewer.current_frame
        peaks_for_frame = self.viewer._frame_peaks.get(frame) or {}
        matched = self.viewer.matched_structures(frame)
        self.peaks_table_panel.set_frame_peaks(
            frame,
            peaks_for_frame.get("detected"),
            peaks_for_frame.get("fitted"),
            matched,
        )

    def _refresh_peaks_table_on_frame(self, _frame: int) -> None:
        """Slot for ``viewer.frameChanged`` — drops the unused frame
        index since ``_refresh_peaks_table`` re-reads it from the
        viewer."""
        self._refresh_peaks_table()

    def _on_peaks_selected_from_table(self, sels: list) -> None:
        """Route a table multi-selection into the image viewer.

        Same stamping as the single-row path — the panel has no viewer
        context, so it builds every row with ``frame=0``. The list
        order is the panel's: the row the user touched last comes
        first and becomes the primary, so the Parameter panel and the
        resize ROI follow the same peak they would after a Ctrl+click
        on the image.
        """
        if not sels:
            return
        for sel in sels:
            sel.frame = self.viewer.current_frame
        self.viewer.set_selected_peaks(list(sels))

    def _on_delete_peaks_from_table(self, sels: list) -> None:
        """Delete pressed in the Peaks dock.

        Hands straight over to the image-side handlers so the two
        routes cannot diverge: one confirmation dialog, one write
        path, one undo entry. The bulk handler wants two or more
        peaks of one kind and the single handler wants exactly one,
        which is the only reason this splits at all.
        """
        if not sels or self._is_busy():
            return
        for sel in sels:
            sel.frame = self.viewer.current_frame
        if len(sels) == 1:
            self._on_delete_peak_requested(sels[0])
        else:
            self._on_delete_peaks_requested(list(sels))

    def _install_peak_step_shortcuts(self) -> None:
        """Down / Up: select the next / previous peak of the same kind.

        Hidden window-level actions, like the frame keys. Up and Down
        also belong to spin boxes, combo boxes, sliders and item views,
        which a shortcut would otherwise steal them from, so
        ``eventFilter`` accepts the ShortcutOverride for those and lets
        the widget have the key (``_ARROW_KEY_WIDGETS``).
        """
        bindings = [
            ("Next peak", "Down", +1),
            ("Previous peak", "Up", -1),
        ]
        self._peak_step_actions = []
        for name, key, delta in bindings:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(
                lambda _checked=False, d=delta: self._on_step_peak(d)
            )
            self.addAction(action)
            self._peak_step_actions.append(action)

    def _install_add_to_fitted_shortcut(self) -> None:
        """Ctrl+Return (and the keypad Ctrl+Enter): the Add to fitted button.

        A chord rather than a bare letter because it writes to the file.
        The hidden window-level action is not tied to the button, so the
        slot checks the selection itself and says in the status bar why
        nothing happened.
        """
        self.action_add_to_fitted_shortcut = QAction("Add to fitted", self)
        self.action_add_to_fitted_shortcut.setShortcuts([
            QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter"),
        ])
        self.action_add_to_fitted_shortcut.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.action_add_to_fitted_shortcut.triggered.connect(
            self._on_add_to_fitted_shortcut
        )
        self.addAction(self.action_add_to_fitted_shortcut)

    def _on_add_to_fitted_shortcut(self) -> None:
        """Ctrl+Return: Add to fitted for the one selected manual or
        detected peak; any other state leaves the file alone and says why."""
        if self._is_busy() or self.viewer._busy:
            self.statusBar().showMessage(
                "Add to fitted: open a file and wait for the running "
                "step to finish.", 4000,
            )
            return
        sels = self.viewer.selected_peaks()
        if len(sels) > 1:
            self.statusBar().showMessage(
                "Add to fitted works on one peak; use Fit selected (2D) "
                "for several.", 4000,
            )
            return
        if not sels or sels[0].kind not in ("manual", "detected"):
            self.statusBar().showMessage(
                "Add to fitted: select a manual or detected peak first.",
                4000,
            )
            return
        self._on_add_to_fitted()

    def _on_step_peak(self, delta: int) -> None:
        """Down / Up: move a detected or fitted selection to its neighbour."""
        if self._is_busy() or self.viewer._busy:
            return
        self.viewer.step_selection(delta)

    def _on_add_detected_for_fitted(self) -> None:
        """Edit > Add detected boxes for fitted peaks.

        Old datasets can hold fitted peaks that have no detected peak of
        the same id. This adds one detected box for each, on every frame
        of the active entry, so each fitted id has a partner afterwards.
        The geometry is the fitted box's own (a fitted ring gets the
        finite quadrant-spanning form of the entry's angular axis); see
        ``file_model.add_missing_detected_for_fitted`` for the rules.
        One undo step reverses the whole batch.
        """
        if self._is_busy():
            self.statusBar().showMessage(
                "Add detected for fitted: open a file and wait for the "
                "running job first.", 4000,
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        extent = self.viewer.angular_extent()
        if extent is None:
            self.statusBar().showMessage(
                "Add detected for fitted: the entry's angular range is not "
                "available (no polar image loaded).", 5000,
            )
            return
        ring_angle = 0.5 * (float(extent[0]) + float(extent[1]))
        ring_width = abs(float(extent[1]) - float(extent[0]))
        with self._detached_silx_tree():
            try:
                added, unusable = file_model.add_missing_detected_for_fitted(
                    self.session.temp_path, entry,
                    ring_angle=ring_angle, ring_width=ring_width,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Add detected for fitted", str(exc)
                )
                return
        total = sum(len(v) for v in added.values())
        skipped = (
            f"; {unusable} fitted peak(s) with no usable radius were skipped"
            if unusable else ""
        )
        if not total:
            self.statusBar().showMessage(
                f"Every fitted peak of {entry} already has a detected "
                f"partner{skipped}.", 5000,
            )
            return
        self.session.mark_dirty()
        self._update_title()
        # Only appends rows under the fitted peaks' own ids; no id is
        # invalidated, so earlier undo history stays valid.
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._push_add_detected_for_fitted_undo(
            entry=entry, added=added,
            ring_angle=ring_angle, ring_width=ring_width,
        )
        msg = (
            f"Added {total} detected box(es) for fitted peaks on "
            f"{len(added)} frame(s) of {entry}{skipped}"
        )
        self.statusBar().showMessage(msg, 6000)
        self.pipeline_panel.append_log(msg)

    def _push_add_detected_for_fitted_undo(
        self, *, entry: str, added: dict[int, list[int]],
        ring_angle: float, ring_width: float,
    ) -> None:
        """Undo / redo for the batch above, as one grouped action.

        Undo deletes the rows that were added. Redo derives them again
        from the fitted rows, which an undo has put back to exactly the
        state the first run started from, so the same rows come back.
        """
        state = {"added": {int(k): list(v) for k, v in added.items()}}

        def _undo() -> None:
            def write():
                return file_model.delete_detected_rows(
                    self.session.temp_path, entry, state["added"],
                )

            def after(removed) -> None:
                self.pipeline_panel.append_log(
                    f"Undo add detected for fitted: removed {removed} "
                    f"detected box(es) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Undo add detected for fitted", write=write,
                after=after,
            )

        def _redo() -> None:
            def write():
                again, _unusable = file_model.add_missing_detected_for_fitted(
                    self.session.temp_path, entry,
                    ring_angle=ring_angle, ring_width=ring_width,
                )
                return again

            def after(again) -> None:
                state["added"] = {int(k): list(v) for k, v in again.items()}
                self.pipeline_panel.append_log(
                    f"Redo add detected for fitted: re-added "
                    f"{sum(len(v) for v in again.values())} detected "
                    f"box(es) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Redo add detected for fitted", write=write,
                after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_switch_detected_fitted(self) -> None:
        """T: move the selection between a detected peak and its fitted peak.

        Pairs by id, which is only meaningful with the peak link on
        (``peak_link``); with it off the two tables number their rows
        independently, so nothing is switched and the status bar says
        why. With a multi-selection every peak that has a partner is
        switched and the rest are dropped from the selection.
        """
        if self._is_busy() or self.viewer._busy:
            return
        found = self.viewer.partner_selections()
        if found is None:
            self.statusBar().showMessage(
                "Switch detected/fitted: select a detected or fitted "
                "peak first.", 4000,
            )
            return
        if not link_enabled():
            self.statusBar().showMessage(
                "Switch detected/fitted needs the peak link (Settings), "
                "which keeps detected and fitted ids paired.", 6000,
            )
            return
        partners, missing = found
        sels = self.viewer.selected_peaks()
        other = "fitted" if sels[0].kind == "detected" else "detected"
        if not partners:
            what = (
                f"{sels[0].kind} peak {sels[0].peak_id}"
                if len(sels) == 1 else f"any of the {len(sels)} selected peaks"
            )
            self.statusBar().showMessage(
                f"No {other} partner for {what}.", 4000,
            )
            return
        self.viewer.select_partners(partners)
        if missing:
            self.statusBar().showMessage(
                f"Switched {len(partners)} of {len(sels)} to {other}; "
                f"{missing} had no {other} partner.", 4000,
            )

    def _on_peak_selected_from_table(self, sel: SelectedPeak | None) -> None:
        """Route a table-row click back into the image viewer.

        The panel builds the SelectedPeak with ``frame=0`` since it
        has no viewer context; stamp the real current_frame here
        before handing it to ``_set_selected``. The viewer's
        equality guard short-circuits the round-trip if the
        selection didn't actually change.
        """
        if sel is None:
            return
        sel.frame = self.viewer.current_frame
        self.viewer._set_selected(sel)
