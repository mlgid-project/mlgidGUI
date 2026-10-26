"""Selection forwarding to the profile viewer, the debounced 2D pygidfit preview, fit-parameter dispatch, save-as-ring and manual-peak hooks.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from mlgidlab.parameter_panel import ParameterPanel


def _pygidfit_to_fit_like(fit_2d):
    """Adapter exposing ``.center`` / ``.fwhm`` / ``.amplitude`` over a
    pygidfit ``ManualFitResult`` so ``ParameterPanel.set_fits`` can
    consume it interchangeably with scipy ``GaussianFit``s.

    pygidfit stores box widths as ``2σ`` (pipeline convention; see
    ``manual_fit.fit_one_peak``), so ``fwhm = 2σ × √(2 ln 2) =
    radius_width × √(2 ln 2) ≈ 1.177 × radius_width``. Used when
    the parameter readout is showing the 2D-mode preview of what
    Add-to-fitted (2D) will save.
    """
    from types import SimpleNamespace
    width_to_fwhm = float(np.sqrt(2.0 * np.log(2.0)))
    rfit = SimpleNamespace(
        center=float(fit_2d.radius),
        fwhm=float(fit_2d.radius_width) * width_to_fwhm,
        amplitude=float(fit_2d.amplitude),
    )
    afit = SimpleNamespace(
        center=float(fit_2d.angle),
        fwhm=float(fit_2d.angle_width) * width_to_fwhm,
        amplitude=float(fit_2d.amplitude),
    )
    return rfit, afit


class PreviewMixin:
    # -- Profile viewer adapters --

    def _forward_selection_to_profile(self, sel: SelectedPeak | None) -> None:
        # Profiles render for any kind of selection; the profile viewer
        # internally makes regions non-movable for non-manual peaks since
        # those are edited through the 2D ROI.
        self.profile_viewer.set_selected_peak(sel)

    def _forward_geom_to_profile(self, sel: SelectedPeak | None) -> None:
        if sel is None:
            return
        self.profile_viewer.sync_regions_from_peak(sel)

    def _on_detected_border_commit(self, sel: SelectedPeak) -> None:
        """Persist a detected-peak border drag from the profile viewer
        to ``detected_peaks`` on disk.

        Funnels through the same ``_on_peak_row_write_requested`` slot
        the image-side ROI drag uses — it owns the silx detach /
        update_peak_row / matched cascade dance. The profile-side
        drag merely fires this commit signal at drag-end; everything
        else (live overlay sync, in-memory PeakTable mutation,
        ``q_xy``/``q_z`` recompute) has already happened during the
        live drag via ``update_detected_geometry_external``.
        """
        polar = {
            "radius": float(sel.radius),
            "angle": float(sel.angle),
            "radius_width": float(sel.radius_width),
            "angle_width": float(sel.angle_width),
        }
        self._on_peak_row_write_requested(
            int(sel.frame), "detected", int(sel.peak_id), polar,
        )

    def _on_selection_for_preview(self, sel: SelectedPeak | None) -> None:
        """Drop the fitted-preview overlay when the active selection isn't
        a candidate-for-fitted peak. Manual + detected are both candidates
        — Add-to-fitted is enabled for either — so the preview is shown
        for both kinds. Fitted / matched already have a stored box, so a
        cyan refit overlay there would be visual noise.

        Also kicks ``_refresh_2d_preview`` so the pygidfit override
        on the profile viewer follows the new selection.
        """
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)
        self._refresh_2d_preview()

    def _apply_fit_curve_visibility(self) -> None:
        """Decide whether to show the pink fit overlay on the profile plots.

        Hide only when the selection is a *source* peak about to be
        committed via Add-to-fitted (kind ``manual`` or ``detected``)
        AND the active fit mode is 2D — that is the case where the
        pink curve mismatch was misleading (pygidfit's projected 1D
        Gaussian doesn't perfectly match the integrated 1D profile,
        and the image-side cyan box already previews what gets
        saved).

        Show in every other case:
        * 1D mode (the pink curves are the scipy fits
          Add-to-fitted (1D) actually persists)
        * Fitted / matched selections (the curves represent the
          *already-saved* fit and stay informative regardless of
          the panel's mode radio — switching the radio shouldn't
          erase a fitted peak's overlay)

        Wired to ``parameter_panel.fitModeChanged``,
        ``parameter_panel.saveAsRingChanged``, and
        ``viewer.selectionChanged`` so the visibility re-evaluates
        whenever any of those three change.
        """
        sel = self.viewer.selected_peak
        is_source_kind = (
            sel is not None and sel.kind in ("manual", "detected")
        )
        is_2d = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
        )
        visible = not (is_source_kind and is_2d)
        self.profile_viewer.set_fit_curves_visible(visible)

    def _refresh_2d_preview(self) -> None:
        """Start pygidfit, in the background, on every fittable selected
        peak (primary + Ctrl+click extras) that has no result yet, and
        cache the results as they arrive so ``_update_fitted_preview``
        (primary) and ``set_fitted_preview_extras`` (extras) paint a
        cyan box for each. Returns at once; ``_on_preview_fit_done``
        repaints when a fit finishes.

        The profile viewer is intentionally NOT updated from here
        anymore: in 2D mode the pink fit curves are hidden (see
        ``_apply_fit_curve_visibility``) and the grey integrated
        trace stays sliced over the user's ROI, so the only
        consumers of the cached pygidfit result are the image-side
        cyan boxes.

        Wired to ``viewer.selectionChanged``,
        ``viewer.selectionsChanged``, ``viewer.frameChanged``,
        ``parameter_panel.fitModeChanged``,
        ``parameter_panel.saveAsRingChanged``, and
        ``viewer.peakGeometryDragFinished``. The cache is a dict
        keyed by ``(file, entry, frame, sel geometry)`` so a
        previously-fit peak that's re-selected (e.g. user toggles
        a Ctrl+click extra off and back on) reuses the cached fit
        instead of re-paying the ~100-500 ms pygidfit cost.
        """
        save_as_ring = self.parameter_panel.save_as_ring()
        is_2d = self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
        applies = (
            is_2d
            and not save_as_ring
            and self.session is not None
            and self._pipe_thread is None
        )
        entry = self.entry_combo.currentText()
        if not applies or not entry:
            # Nothing is wanted any more: queued fits are skipped.
            self._preview_runner.want(())
            self._update_preview_status()
            self._repaint_fitted_preview()
            return

        # Every fittable selection that has no result yet is handed to
        # the background runner. The fit is not waited for: it can take
        # seconds (tens, for a peak in a badly conditioned peak-on-ring
        # cluster), and on the GUI thread that froze the window. The
        # cache is preserved across calls so unchanged peaks (e.g.
        # extras whose ROI hasn't moved) don't re-fit.
        fittable = [
            s for s in self.viewer.selected_peaks()
            if s.kind in ("manual", "detected")
        ]
        keys = {self._preview_fingerprint(entry, s): s for s in fittable}
        self._preview_runner.want(keys)
        for fp, s in keys.items():
            if fp in self._2d_preview_cache or self._preview_runner.is_pending(fp):
                continue
            job, _err = self._prepare_pygidfit_fit(s, entry, int(s.frame))
            if job is None:
                # Inputs unavailable: cache the failure (None) so it is
                # not retried on every selection tick.
                self._2d_preview_cache[fp] = None
                continue
            self._preview_runner.submit(fp, job)
        self._update_preview_status()
        self._repaint_fitted_preview()

    def _on_preview_fit_done(self, key, fit_2d, _err, ran: bool) -> None:
        """A background preview fit finished (GUI thread).

        The result is kept either way (None on failure, so a failed fit
        is not retried on every tick) unless the job was skipped. It is
        painted only if it belongs to a peak that is still selected.
        """
        self._preview_runner.forget(key)
        if getattr(self, "_closed", False):
            return
        if ran and self.session is not None and key[0] == self.session.temp_path:
            self._2d_preview_cache[key] = fit_2d
            entry = self.entry_combo.currentText()
            current = {
                self._preview_fingerprint(entry, s)
                for s in self.viewer.selected_peaks()
                if s.kind in ("manual", "detected")
            }
            if key in current:
                self._repaint_fitted_preview()
        self._update_preview_status()
        # An Add to fitted click may be waiting on exactly this fit.
        self._finish_add_to_fitted(key, fit_2d, _err, ran)

    def _update_preview_status(self) -> None:
        """Status-bar note while a preview fit is still running, so a
        blank cyan box and parameter readout read as "working"."""
        busy = self._preview_runner.pending_count() > 0
        if busy:
            self.statusBar().showMessage(
                "Fitting the 2D preview (the window stays usable)…"
            )
            self._preview_status_shown = True
        elif getattr(self, "_preview_status_shown", False):
            self._preview_status_shown = False
            self.statusBar().clearMessage()

    def _preview_fingerprint(
        self, entry: str, sel: SelectedPeak,
    ) -> tuple:
        """Build the cache key for ``sel``'s pygidfit fit.

        The first five fields are the peak's identity (file, entry,
        frame, kind, id), then its geometry, then a digest of the other
        inputs the fit reads: the Pipeline panel's fit settings and the
        neighbouring boxes. Those last two are why a result can be
        reused by Add to fitted: a changed setting or an edited
        neighbour gives a different key instead of a stale hit.
        """
        return (
            self.session.temp_path if self.session is not None else None,
            entry,
            int(sel.frame),
            sel.kind,
            int(sel.peak_id) if sel.peak_id is not None else -1,
            float(sel.radius), float(sel.radius_width),
            float(sel.angle), float(sel.angle_width),
            self._pygidfit_context_key(entry, sel),
        )

    def _pygidfit_context_key(self, entry: str, sel: SelectedPeak) -> int:
        """Digest of the fit inputs that are not the peak itself."""
        panel = self.pipeline_panel
        try:
            settings = (
                float(panel.fit_crit_angle.value()),
                float(panel.fit_dist_peaks.value()),
                float(panel.fit_dist_rings.value()),
                int(panel.fit_cluster_extend.value()),
                bool(panel.fit_theta_fixed.isChecked()),
            )
        except Exception:
            settings = None
        try:
            boxes = tuple(
                (b.radius, b.radius_width, b.angle, b.angle_width)
                for b in self._neighbour_boxes(entry, int(sel.frame), sel)
            )
        except Exception:
            boxes = None
        return hash((settings, boxes))

    def _schedule_drag_2d_preview(self, _sel) -> None:
        """Debounced ``_refresh_2d_preview`` while a manual / detected
        ROI is being dragged in 2D mode.

        Per-tick ``peakGeometryChanged`` callbacks fan in here; the
        single-shot timer collapses bursts so pygidfit runs at most
        once every ~150 ms during a drag. The drag-end firing of
        ``peakGeometryDragFinished`` triggers a final synchronous
        ``_refresh_2d_preview`` that overrides whatever the timer
        was about to do.
        """
        if not self.viewer.is_dragging:
            return
        if self.parameter_panel.fit_mode() != ParameterPanel.FIT_MODE_2D:
            return
        if self.parameter_panel.save_as_ring():
            return
        # Restart the single-shot timer; bursts collapse into one
        # fire 150 ms after the last drag tick (or the timer is
        # superseded by the drag-end refresh).
        self._drag_pygidfit_timer.start()

    def _repaint_fitted_preview(self) -> None:
        """Re-run the cyan-box painter and the parameter readout
        against the freshest fit state.

        Re-reads the profile viewer's cached scipy 1D fits and
        feeds them through ``_update_fitted_preview`` (which
        prefers pygidfit cache in 2D mode) and
        ``_dispatch_fit_params`` (which routes pygidfit values to
        the parameter panel in 2D mode for the same reason). Also
        pushes the cached pygidfit boxes for the Ctrl+click extras
        into ``viewer.set_fitted_preview_extras`` so every selected
        peak gets its own cyan preview.
        """
        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        self._update_fitted_preview(rfit, afit)
        self._dispatch_fit_params(rfit, afit)
        self._push_extras_preview()

    def _push_extras_preview(self) -> None:
        """Compute the cyan preview geometries for every Ctrl+click
        extra and push them to the viewer.

        In 2D mode + non-ring + session-active: every extra (always
        detected by construction) gets a cyan box at its cached
        pygidfit-refined ``(r, dr, a, da)``. Extras with no cache
        entry (still computing, or pygidfit failed) fall back to
        the extra's own drawn geometry — better to show *something*
        than to hide the box and let the user wonder if the
        selection registered. In 1D / ring mode the extras preview
        is cleared.
        """
        v = self.viewer
        extras = v._selected_extras
        if not extras:
            v.set_fitted_preview_extras([])
            return
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not self.parameter_panel.save_as_ring()
        )
        if not is_2d_mode:
            v.set_fitted_preview_extras([])
            return
        geoms: list[tuple[float, float, float, float]] = []
        for s in extras:
            if s.kind != "detected":
                continue
            box = self._current_pygidfit_box_for_selection(s)
            if box is None:
                # Fallback: show the user's drawn geometry so the
                # box is at least visible. pygidfit may have failed
                # or just not run yet for this peak.
                geoms.append((
                    float(s.radius), float(s.radius_width),
                    float(s.angle), float(s.angle_width),
                ))
            else:
                geoms.append(box)
        v.set_fitted_preview_extras(geoms)

    def _dispatch_fit_params(self, rfit, afit) -> None:
        """Pick the right fit source for the parameter-panel readout.

        In 2D mode with a manual / detected selection that has a
        cached pygidfit result, the panel should reflect pygidfit's
        refined values (the truth Add-to-fitted (2D) will save), not
        scipy's 1D fits. Outside that case, scipy values pass
        through unchanged.

        When no pygidfit cache exists yet for a 2D-mode source
        selection, the panel is blanked rather than backfilled
        with scipy values — showing scipy numbers in 2D mode would
        misrepresent what the next commit would save.
        """
        sel = self.viewer.selected_peak
        is_source_kind = (
            sel is not None and sel.kind in ("manual", "detected")
        )
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not self.parameter_panel.save_as_ring()
        )
        if not (is_source_kind and is_2d_mode):
            self.parameter_panel.set_fits(rfit, afit)
            return
        # 2D mode + source kind: feed pygidfit-derived values when
        # the cache has them; blank otherwise.
        box = self._current_pygidfit_box_for_selection(sel)
        if box is None:
            self.parameter_panel.set_fits(None, None)
            return
        fp = self._preview_fingerprint(
            self.entry_combo.currentText(), sel,
        )
        fit_2d = self._2d_preview_cache.get(fp)
        if fit_2d is None:
            # Drag-fallback path returned a box from a stale
            # fingerprint — find any cached fit for this peak's
            # identity (file/entry/frame/kind/peak_id).
            for cached_fp, cached_fit in self._2d_preview_cache.items():
                if cached_fp[:5] == fp[:5] and cached_fit is not None:
                    fit_2d = cached_fit
                    break
        if fit_2d is None:
            self.parameter_panel.set_fits(None, None)
            return
        pyg_rfit, pyg_afit = _pygidfit_to_fit_like(fit_2d)
        self.parameter_panel.set_fits(pyg_rfit, pyg_afit)

    def _update_fitted_preview(self, rfit, afit) -> None:
        """Sync the viewer's fitted-preview box to the latest fit params.

        Relevant for manual + detected selections — both feed
        Add-to-fitted. File-resident fitted / matched peaks already
        carry their stored box and aren't previewed here.

        Mode-dependent — cyan box always matches what the
        corresponding commit path will save:

        * **2D mode**: cyan paints at pygidfit's *exact* box
          (``radius, radius_width, angle, angle_width`` straight
          from the cached ``ManualFitResult``). The pink profile
          curve is allowed to drift slightly via the anchored fit
          for visual cleanliness; that drift must not leak into
          the cyan box, which the user expects to match the saved
          blue box exactly.
        * **1D mode** (or ring, which forces 1D): cyan paints at
          ``(scipy_centre, 2σ_scipy)`` per axis — same convention
          ``_build_fitted_row_1d`` saves (``radius_width = FWHM ×
          1/√(2 ln 2)``). Per-axis fallback to the selected peak's
          drawn box when scipy hasn't converged.

        The pygidfit cache outlives a mode toggle, so the 2D-mode
        branch is gated on the live ``fit_mode()`` reading — without
        that gate, switching from 2D to 1D would briefly paint cyan
        at pygidfit's box even though the next commit would use
        scipy's geometry.
        """
        sel = self.viewer.selected_peak
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        save_as_ring = self.parameter_panel.save_as_ring()
        # Ring forces 1D in ``fit_mode()``; defensive check here keeps
        # the 2D-cyan-box branch off when the user has ring on but the
        # 2D-preview cache hasn't yet been cleared for the new state.
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not save_as_ring
        )

        fwhm_to_2sigma = 1.0 / float(np.sqrt(2.0 * np.log(2.0)))

        # 2D mode + a cached pygidfit result on the current selection:
        # paint the cyan box at pygidfit's *exact* geometry instead of
        # the anchored-fit ``rfit``/``afit``. The pink profile curve
        # was deliberately allowed to drift (centre ±0.5σ, sigma ×1.2)
        # so it sits clean on the data; that drift must NOT leak into
        # the image-side cyan box, which the user expects to match
        # the saved blue box exactly. Skipped in 1D mode because the
        # pygidfit cache can persist after a mode switch — 1D commit
        # uses scipy's fit, so cyan must follow rfit/afit there.
        if is_2d_mode:
            pygidfit_box = self._current_pygidfit_box_for_selection(sel)
            if pygidfit_box is not None:
                fr, fdr, fa, fda = pygidfit_box
                self.viewer.set_fitted_preview(
                    fr, fdr, fa, fda, is_ring=False,
                )
                return

        if rfit is not None:
            center_r = float(rfit.center)
            width_r = float(rfit.fwhm) * fwhm_to_2sigma
        else:
            center_r = float(sel.radius)
            width_r = float(sel.radius_width)

        if save_as_ring:
            # Ring sentinel — the painter ignores the angular args.
            # Ring forces 1D so the cached pygidfit result is never
            # used here; the radial width comes from scipy's rfit.
            self.viewer.set_fitted_preview(
                center_r, width_r, None, None, is_ring=True,
            )
            return

        if afit is not None:
            center_a = float(afit.center)
            width_a = float(afit.fwhm) * fwhm_to_2sigma
        else:
            center_a = float(sel.angle)
            width_a = float(sel.angle_width)

        self.viewer.set_fitted_preview(
            center_r, width_r, center_a, width_a, is_ring=False,
        )

    def _current_pygidfit_box_for_selection(
        self, sel: SelectedPeak,
    ) -> tuple[float, float, float, float] | None:
        """Return pygidfit's refined box for ``sel`` if cached, else None.

        The 2D-preview cache keyed by ``(file, entry, frame, sel
        geometry)`` is built in ``_refresh_2d_preview``. If the cache
        entry matches the current ``sel`` and holds a non-None
        ``ManualFitResult``, return its ``(r, dr, a, da)``. Otherwise
        return None — caller falls back to the rfit/afit-driven cyan
        box (the 1D-mode / no-pygidfit-yet path).

        While the viewer reports an active ROI drag, the geometry
        fields of ``sel`` change per tick faster than pygidfit can
        keep up, so this method matches on the *peak identity*
        only (file / entry / frame / kind / peak_id) and returns
        the cached refined box from the last completed pygidfit
        run. That keeps the cyan preview tracking pygidfit's
        output through the drag instead of flickering into the
        scipy 1D fallback between debounced refits.
        """
        if not self._2d_preview_cache:
            return None
        if self.session is None:
            return None
        entry = self.entry_combo.currentText()
        if not entry:
            return None
        full_fp = self._preview_fingerprint(entry, sel)
        identity = full_fp[:5]
        if self.viewer.is_dragging:
            # During drag: match on identity only so the last
            # cached refined box survives geometry mismatches.
            # Pick the first non-None fit for the matching identity
            # (typically there's just one).
            fit_2d = None
            for cache_fp, cached_fit in self._2d_preview_cache.items():
                if cache_fp[:5] == identity and cached_fit is not None:
                    fit_2d = cached_fit
                    break
            if fit_2d is None:
                return None
        else:
            fit_2d = self._2d_preview_cache.get(full_fp)
            if fit_2d is None:
                return None
        return (
            float(fit_2d.radius), float(fit_2d.radius_width),
            float(fit_2d.angle), float(fit_2d.angle_width),
        )

    def _on_save_as_ring_changed(self, is_ring: bool) -> None:
        """Toggle between segment / ring preview.

        Three coordinated effects:

        1. The profile viewer skips the angular Gaussian fit while ring
           is active — that fit wouldn't be saved by Add-to-fitted.
        2. If a manual peak is selected, its angular sweep is widened
           to span the full polar plot height (so the radial profile
           integrates over the entire angular axis, matching what the
           ring fit will eventually represent). The pre-ring geometry
           is stashed so unticking the box — including the auto-uncheck
           that fires after Add-to-fitted commits — restores the box.
        3. The fitted-preview is recomputed against the new fit cache.
        """
        sel = self.viewer.selected_peak
        manual_ref = (
            sel.manual_ref if sel is not None and sel.kind == "manual" else None
        )

        if is_ring and manual_ref is not None:
            # Stash pre-ring geometry once. If the user ticks → unticks
            # → re-ticks without committing, we keep the original stash
            # so the eventual restore returns to the very first state,
            # not the intermediate ring state.
            if self._ring_pre_geom is None:
                self._ring_pre_geom = (
                    manual_ref,
                    manual_ref.radius,
                    manual_ref.angle,
                    manual_ref.radius_width,
                    manual_ref.angle_width,
                    manual_ref.is_ring,
                )
            extent = self.viewer.angular_extent()
            if extent is not None:
                a_lo, a_hi = extent
                ring_angle = 0.5 * (a_lo + a_hi)
                ring_width = abs(a_hi - a_lo)
                self.viewer.set_manual_geometry(
                    manual_ref,
                    radius=manual_ref.radius,
                    angle=ring_angle,
                    radius_width=manual_ref.radius_width,
                    angle_width=ring_width,
                    is_ring=True,
                )
        elif not is_ring and self._ring_pre_geom is not None:
            (
                stashed_peak,
                pre_r,
                pre_a,
                pre_dr,
                pre_da,
                pre_is_ring,
            ) = self._ring_pre_geom
            self._ring_pre_geom = None
            # Only restore if the stashed peak still exists — the user
            # may have drawn a replacement (which removes the original
            # via the single-box policy) while ring was active. In that
            # case the new peak inherited the ring geometry but has no
            # captured pre-state, so leave it alone.
            for peaks in self.viewer._manual_peaks.values():
                if stashed_peak in peaks:
                    self.viewer.set_manual_geometry(
                        stashed_peak,
                        radius=pre_r,
                        angle=pre_a,
                        radius_width=pre_dr,
                        angle_width=pre_da,
                        is_ring=pre_is_ring,
                    )
                    break

        # Drop the angular fit *before* recomputing the preview so the
        # cached afit is None when _update_fitted_preview reads it.
        self.profile_viewer.set_skip_angular_fit(is_ring)
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

    def _on_fit_mode_changed(self, _mode: str) -> None:
        """Re-render the cyan preview when the user flips 1D ↔ 2D.

        The preview is only painted in 1D mode (see
        ``_update_fitted_preview``); flipping to 2D hides it,
        flipping back shows it. Without this hook the on-screen
        state wouldn't update until the next ROI / frame /
        selection change.
        """
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

    def _on_manual_peak_added(self, _frame: int, peak: ManualPeak) -> None:
        """Apply the active ring expansion to a freshly added manual peak.

        When the user draws a new manual box while the ring checkbox is
        on, the single-box-replace removes the old (with its ring stash)
        and adds the new one. Without this slot, the new box would stay
        as drawn — confusing because the checkbox is still ticked. We
        mirror what ``_on_save_as_ring_changed(True)`` would do for the
        new peak: stash its pre-ring shape, then expand to the full
        angular sweep.
        """
        if not self.parameter_panel.save_as_ring():
            return
        # Stash pre-ring state for the new peak. Any earlier stash
        # already pointed at a peak that's been removed (which our
        # manualPeakRemoved slot has already cleared).
        self._ring_pre_geom = (
            peak,
            peak.radius,
            peak.angle,
            peak.radius_width,
            peak.angle_width,
            peak.is_ring,
        )
        extent = self.viewer.angular_extent()
        if extent is None:
            return
        a_lo, a_hi = extent
        self.viewer.set_manual_geometry(
            peak,
            radius=peak.radius,
            angle=0.5 * (a_lo + a_hi),
            radius_width=peak.radius_width,
            angle_width=abs(a_hi - a_lo),
            is_ring=True,
        )

    def _on_manual_peak_removed(self, _frame: int, peak: ManualPeak) -> None:
        """Invalidate ``_ring_pre_geom`` when the peak it references goes away.

        Without this, an Esc / Delete / Add-to-detected on a ring-
        expanded peak would leave a dangling stash; later unticking
        the ring checkbox would walk the manual list looking for that
        ghost and find nothing, but the stash stays set and could
        mis-fire on a later toggle cycle.
        """
        if (
            self._ring_pre_geom is not None
            and self._ring_pre_geom[0] is peak
        ):
            self._ring_pre_geom = None
