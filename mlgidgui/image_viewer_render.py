"""Viewer rendering: mode dispatch, frame/overlay painting, log scaling, colormap plumbing and the cursor readout.

Plain mixin over ``GIWAXSImageViewer``: no __init__, no Signals; all state
lives on the combined class. Split out of ``image_viewer`` in the 2026
source split.
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from mlgidgui.file_model import PeakTable, _LazyPolarStack
from mlgidgui.main_window_constants import (
    DEFAULT_SHOW_FIT_AREA,
    SHOW_FIT_AREA_KEY,
)
from mlgidgui.manual_fit import PeakBox, fit_area
from mlgidgui.peak_link import read_bool
from mlgidgui.polar import polar_to_qxyz
from mlgidgui.viewer_items import (
    ManualPeak,
    _DisplayParams,
    _PeakShapeItem,
    _apply_raw_flips,
    _bin_index,
    _peaks_from_manual,
    _raw_pixel_index,
    _peaks_subset,
    _robust_levels,
)
from mlgidgui.viewer_styles import (
    MATCHED_LINE_WIDTH,
    MATCHED_STYLE,
    MODE_CARTESIAN,
    MODE_POLAR,
    MODE_RAW,
    UNMATCHED_COLOR,
    UNMATCHED_UID,
    resolve_colormap,
)

import logging

logger = logging.getLogger(__name__)


class ViewerRenderMixin:
    # -- Rendering --

    def _current_levels(self) -> tuple[float, float] | None:
        """The contrast levels currently on the histogram, or None when
        unreadable / degenerate. Used to snapshot the user's slider."""
        try:
            lo, hi = self._view.getHistogramWidget().getLevels()
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._current_levels", exc_info=True)
            return None
        if lo is None or hi is None:
            return None
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
        return lo, hi

    def _on_contrast_changed(self, *args) -> None:
        """Histogram drag finished — remember the user's contrast so it
        sticks across operation re-renders and frame scrubs.

        ``setImage(levels=...)`` (every render) does NOT emit
        ``sigLevelChangeFinished``, so this only fires on a genuine user
        drag, never on our own renders — no suppression flag needed.
        """
        if self._suppress_level_capture:
            return
        lv = self._current_levels()
        if lv is not None:
            self._user_levels = lv

    def _render_active_mode(self, *, auto_range: bool = True) -> None:
        """Build per-mode axis state + first-frame levels, then render.

        Called on stack swap (entry change, file open), mode toggle
        (Cartesian ↔ polar), and log/linear toggle. The actual per-
        frame pixel push happens in ``_render_frame`` which is also
        the path ``set_frame`` takes on every scrub.

        ``auto_range`` defaults to True so cold-open / mode-swap
        renders fit the new image into the viewbox. Callers that
        preserve the user's prior zoom (``show_stack(preserve_view=True)``,
        the log-scale toggle) pass ``False`` so ``setImage`` never
        clobbers the viewbox in the first place.
        """
        if self._mode == MODE_RAW:
            if self._raw_image_stack is None:
                return
            self._display_params = self._build_raw_params()
            # Re-derive the aspect lock for this mode's extents before the
            # render's autoRange so the locked shape is honoured.
            self._apply_aspect(refit=False)
            self._render_frame(self._frame_index, auto_range=auto_range)
            # No overlays in raw mode — nothing to render past _render_frame.
            return
        # ``_build_*_params`` read frame 0 through the FrameSource to
        # compute robust levels, so bail when the source is released by
        # the silx detach/reattach dance. Without this, toggling the
        # Cartesian/Polar radio during a write (pipeline run, ROI
        # commit, Add-to-fitted, clear-peaks, save-as) drives
        # ``get_cartesian(0)`` / ``get_polar(0)`` into a closed handle
        # and raises ``RuntimeError("FrameSource not acquired")``. The
        # new mode is still recorded; ``acquire_frame_source`` re-renders
        # it once the handle reopens.
        if (
            self._stack is None
            or self._frame_source is None
            or not self._frame_source.is_open
        ):
            return
        if self._mode == MODE_POLAR:
            self._display_params = self._build_polar_params()
        else:
            self._display_params = self._build_cartesian_params()
        # Re-derive the aspect lock for this mode's extents before the
        # render's autoRange so the locked shape is honoured (the same
        # ratio means the same on-screen shape in Cartesian and polar).
        self._apply_aspect(refit=False)
        self._render_frame(self._frame_index, auto_range=auto_range)
        self._render_overlays(self.current_frame)

    def _build_raw_params(self) -> _DisplayParams:
        """Pixel-coordinate axis state for a raw detector stack.

        File order is (frames, H, W). For pyqtgraph we want each frame
        displayed as (W, H) — see ``_render_frame``'s raw branch for
        the per-frame transpose. Axes are labelled in pixels; q
        coordinates aren't meaningful before conversion.
        """
        assert self._raw_image_stack is not None
        # ``image_pg`` here is just the *first* frame, used for level
        # computation. Per-frame pushes happen in ``_render_frame``.
        first = np.asarray(self._raw_image_stack[0]).T  # (W, H)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(0.0, 0.0),
            scale=(1.0, 1.0),
            levels=levels,
            x_label=("x", "px"),
            y_label=("y", "px"),
        )

    def _build_cartesian_params(self) -> _DisplayParams:
        """Cartesian axis state. Reads frame 0 via the FrameSource to
        compute robust intensity levels; per-frame rendering is lazy.
        """
        assert self._stack is not None and self._frame_source is not None
        q_xy = self._frame_source.q_xy
        q_z = self._frame_source.q_z
        x0 = float(q_xy[0]); y0 = float(q_z[0])
        sx = float(q_xy[-1] - q_xy[0]) / max(len(q_xy) - 1, 1)
        sy = float(q_z[-1] - q_z[0]) / max(len(q_z) - 1, 1)
        first = self._frame_source.get_cartesian(0).T  # (n_qxy, n_qz)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("q_xy", "Å⁻¹"),
            y_label=("q_z", "Å⁻¹"),
        )

    def _build_polar_params(self) -> _DisplayParams:
        """Polar axis state + lazy polar wrapper.

        The polar grid (``radius`` / ``angle``) is derived once from
        ``q_xy`` / ``q_z`` and reused across frames; per-frame polar
        resampling happens on demand in ``FrameSource.get_polar``.
        """
        assert self._stack is not None and self._frame_source is not None
        radius, angle = self._frame_source.polar_axes()
        # Cache a (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) share the same FrameSource-
        # backed view without re-creating wrappers.
        if self._polar_cache is None:
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        x0 = float(radius[0]); y0 = float(angle[0])
        sx = float(radius[-1] - radius[0]) / max(len(radius) - 1, 1)
        sy = float(angle[-1] - angle[0]) / max(len(angle) - 1, 1)
        # polar_stack frame layout is (n_radius, n_angle); pyqtgraph
        # wants (x=radius, y=angle) per frame, which already matches.
        first = self._frame_source.get_polar(0)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("radius", "Å⁻¹"),
            y_label=("angle", "deg"),
        )

    def _render_frame(self, idx: int, *, auto_range: bool) -> None:
        """Push the 2D frame at ``idx`` to pyqtgraph.

        Uses the cached ``_display_params`` (built once per stack/mode
        swap) so per-frame scrubs don't rebuild axes. ``auto_range`` is
        True only on the initial render after a stack/mode change;
        every subsequent frame change passes False to preserve the
        user's zoom/pan.
        """
        if self._display_params is None:
            return
        p = self._display_params
        # Fetch the actual 2D frame for ``idx``. Wrapped in
        # try/except because the FrameSource can be in a briefly-
        # released state during the silx detach/reattach dance —
        # if a play-tick or scrub fires in that window, get_polar
        # / get_cartesian raise ``RuntimeError("FrameSource not
        # acquired")``. We swallow that quietly; the reattach path
        # re-renders the active frame once acquire completes.
        try:
            if self._mode == MODE_RAW:
                if self._raw_image_stack is None:
                    return
                frame = _apply_raw_flips(
                    np.asarray(self._raw_image_stack[idx]),
                    self._raw_flip_lr, self._raw_flip_ud, self._raw_transp,
                ).T
            elif self._mode == MODE_POLAR:
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_polar(idx)
            else:  # MODE_CARTESIAN
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_cartesian(idx).T
        except (RuntimeError, ValueError, OSError, KeyError):
            return

        self._plot.setLabel("bottom", p.x_label[0], units=p.x_label[1])
        self._plot.setLabel("left", p.y_label[0], units=p.y_label[1])
        # Prefer the user's dialled-in contrast over the per-build robust
        # default so it survives re-renders; falls back to the cached
        # robust levels when the user hasn't touched the slider.
        base_levels = (
            self._user_levels if self._user_levels is not None else p.levels
        )
        image, levels = self._maybe_apply_log(frame, base_levels)
        # The histogram can emit sigLevelChangeFinished while setImage
        # re-fits the LUT to a new data range; suppress capture so that
        # render doesn't overwrite the user's sticky contrast.
        self._suppress_level_capture = True
        try:
            self._view.setImage(
                image,
                autoRange=auto_range,
                autoLevels=False,
                levels=levels,
                pos=p.pos,
                scale=p.scale,
            )
        finally:
            self._suppress_level_capture = False
        # pyqtgraph's setImage internally calls roiClicked() which
        # force-shows the bottom timeline strip whenever the image has
        # a time axis. With single 2D frames there is no time axis so
        # the strip stays hidden naturally — no extra action needed
        # since the Timeline toggle was removed in the lazy-loading
        # milestone.
        self._hide_pyqtgraph_timeline()

    def _maybe_apply_log(
        self, image: np.ndarray, levels: tuple[float, float]
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """If log-scale is on, return (log10(clip(image, floor)), levels')
        where levels' are the user's sticky contrast when set, else
        recomputed robustly on the transformed first frame.

        Deliberately NOT the tracking views' ``_log_display`` (which
        maps non-positives to NaN): the main viewer wants a continuous
        image with valid levels, not NaN holes.

        Floor is the 1st percentile of strictly-positive finite values
        (or 1e-6 fallback) so the log transform is well-defined for
        zero / negative pixels (background, masked regions). The
        original ``image`` array is not modified.
        """
        if not self._log_scale:
            return image, levels
        finite = image[np.isfinite(image)]
        pos = finite[finite > 0]
        if pos.size > 0:
            floor = float(np.percentile(pos, 1.0))
        else:
            floor = 1e-6
        if floor <= 0:
            floor = 1e-6
        transformed = np.log10(np.clip(image, floor, None))
        # Keep the user's contrast (already in the log domain — the
        # histogram shows the transformed image) when they've set one;
        # otherwise auto-contrast the transformed frame.
        if self._user_levels is not None:
            return transformed, levels
        ref = transformed[0] if transformed.ndim == 3 else transformed
        return transformed, _robust_levels(ref)

    def _on_log_toggled(self, checked: bool) -> None:
        """Re-render in the active mode with log/linear contrast.

        Saves and restores the viewbox range so toggling contrast
        doesn't reset the user's zoom or pan. Frame index is preserved
        too — pyqtgraph's setImage keeps the time-axis position when
        the stack shape is unchanged.
        """
        self._log_scale = bool(checked)
        # Linear and log levels live in different domains, so a sticky
        # linear contrast is meaningless once log is on (and vice versa);
        # drop it and let the toggle re-auto-contrast.
        self._user_levels = None
        saved: tuple[tuple[float, float], tuple[float, float]] | None = None
        try:
            xr, yr = self._plot.getViewBox().viewRange()
            saved = ((float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])))
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
            saved = None
        self._render_active_mode()
        if saved is not None:
            try:
                self._plot.getViewBox().setRange(
                    xRange=saved[0], yRange=saved[1], padding=0
                )
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
                pass

    def _hide_pyqtgraph_timeline(self) -> None:
        """Keep pyqtgraph's bottom timeline strip hidden.

        pyqtgraph's ``setImage`` internally calls ``roiClicked()``
        which re-shows the strip whenever the image has a time axis.
        After the lazy-loading milestone we only ever pass 2D frames
        (no time axis), so the strip stays hidden naturally — but
        we re-apply hidden state explicitly to defend against future
        pyqtgraph versions that might force it back on. The splitter
        handle is set to 0-width in ``__init__`` so there's no
        residual line clipping the image's x-axis label.
        """
        ui = self._view.ui
        ui.roiPlot.setVisible(False)
        ui.splitter.setSizes([1, 0])

    # NB: _on_time_changed (the old pyqtgraph sigTimeChanged slot) is
    # removed. Frame changes flow through set_frame; emissions of
    # frameChanged + matchedStructuresChanged happen there. Left as a
    # comment so future code reviews see the deliberate removal.

    def _pygidfit_q_abs_max(self) -> float | None:
        """``q_abs_max`` as pygidfit's ``ProcessDataFromFile`` computes it:
        ``sqrt(max(q_z)**2 + max(q_xy)**2)`` of the stack on display, or
        ``None`` when there is no usable stack."""
        stack = self._stack
        if stack is None:
            return None
        try:
            value = float(np.sqrt(
                np.nanmax(np.asarray(stack.q_z, dtype=float)) ** 2
                + np.nanmax(np.asarray(stack.q_xy, dtype=float)) ** 2
            ))
        except (AttributeError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def _fit_area_table(self) -> PeakTable | None:
        """One row per selected detected peak: the area pygidfit fits for it.

        That area is larger than the drawn box (``manual_fit.fit_area``:
        twice the width radially and a pixel more on every side), which is
        why a fitted peak can land outside its detected box. ``None`` when
        the "show fit area" setting is off, in raw mode, or when no
        detected peak is selected (fitted, matched and manual selections
        have no such area to show).
        """
        if self._selected is None:
            return None
        if not read_bool(SHOW_FIT_AREA_KEY, DEFAULT_SHOW_FIT_AREA):
            return None
        if self._mode not in (MODE_POLAR, MODE_CARTESIAN):
            return None
        q_abs_max = self._pygidfit_q_abs_max()
        if q_abs_max is None:
            return None
        areas: list[ManualPeak] = []
        for sel in self.selected_peaks():
            if sel.kind != "detected":
                continue
            geom = fit_area(
                PeakBox(sel.radius, sel.radius_width, sel.angle,
                        sel.angle_width),
                q_abs_max,
            )
            if geom is None:
                continue
            radius, radius_width, angle, angle_width = geom
            areas.append(ManualPeak(
                radius=radius, angle=angle,
                radius_width=radius_width, angle_width=angle_width,
                is_ring=not math.isfinite(angle_width),
                temp_id=sel.peak_id,
            ))
        return _peaks_from_manual(areas) if areas else None

    def _render_overlays(self, frame: int) -> None:
        # Raw mode has no peak data to draw — return before touching any
        # overlay path. The four _PeakShapeItems were already cleared by
        # show_raw_stack via clear(), so they have no leftover geometry.
        if self._mode == MODE_RAW:
            return
        peaks = self._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        fit = peaks.get("fitted")

        # Apply the Display-dock min-score filter to the detected
        # overlay. Epsilon is half the slider's natural resolution
        # (0.01 per step → 0.005 tolerance) so a peak whose score
        # the slider was just seeded from is guaranteed to pass,
        # regardless of FP roundoff in the underlying value.
        if det is not None and self._detected_score_cutoff > 0.0 and len(det) > 0:
            try:
                cutoff = self._detected_score_cutoff - 0.005
                mask = np.asarray(det.score) >= cutoff
                if mask.sum() < len(det):
                    keep_ids = [int(det.ids[i]) for i in np.where(mask)[0]]
                    det = _peaks_subset(det, keep_ids)
            except Exception:
                # If the table doesn't have a usable score column
                # (older files), silently skip the filter rather
                # than break the overlay.
                logger.debug("suppressed exception in GIWAXSImageViewer._render_overlays", exc_info=True)
                pass

        # "Show only tracked peaks" (Scan-tracking dock): when active,
        # render only the fitted rows that belong to a surviving track
        # on this frame; frames without members show none. Render-time
        # subset only — the in-memory tables and the file are untouched.
        if self._fitted_visible_only is not None and fit is not None and len(fit) > 0:
            allowed = self._fitted_visible_only.get(frame, set())
            keep_ids = [int(i) for i in fit.ids if int(i) in allowed]
            if len(keep_ids) < len(fit):
                fit = _peaks_subset(fit, keep_ids)

        manual_list = list(self._manual_peaks.get(frame, []))
        # Manual boxes are scratch labels — keep them invisible
        # unless they are the active selection. The ManualPeak
        # instances stay in ``_manual_peaks`` so things like the
        # post-Add-to-fitted Ctrl+Z (FittedRowAction.undo) can
        # reselect the source manual and bring its yellow ROI
        # back. Hit-testing in ``_on_select_at`` still operates on
        # ``_manual_peaks`` directly, so the user can click the
        # invisible region to re-select / reveal the peak.
        selected_is_manual = (
            self._selected is not None and self._selected.kind == "manual"
        )
        if not selected_is_manual:
            manual_list = []
        # When an ROI is active the selected peak is shown via the ROI handles —
        # exclude the manual peak from the manual-overlay path so it doesn't
        # render twice. (Detected/fitted overlays still draw the underlying
        # row; the white SELECTION_STYLE highlight is what's suppressed.)
        roi_active = self._roi_item is not None and self._selected is not None
        if roi_active and self._selected.kind == "manual":
            manual_list = [
                m for m in manual_list if m is not self._selected.manual_ref
            ]
        manual_table = _peaks_from_manual(manual_list)
        # Selection highlight: a PeakTable holding one row per selected
        # peak (primary + Ctrl+click extras). The primary peak gets
        # painted *both* the white outline and (for editable kinds)
        # the resize-ROI handles — the redundant outline is a small
        # price for the visual consistency of "every selected peak
        # looks selected", which matters in multi-select where the
        # user is scanning to confirm what's in the set.
        #
        # Matched-structure primary selections (multi_peak_ids set)
        # expand to every peak in the structure — pull the geometry
        # from the frame's fitted table so the boxes track ROI edits
        # / re-fits. Matched-multi can't coexist with extras (extras
        # only accept detected), so the two branches stay mutually
        # exclusive.
        sel_table: PeakTable | None = None
        if self._selected is not None:
            if self._selected.multi_peak_ids and fit is not None:
                sel_table = _peaks_subset(fit, self._selected.multi_peak_ids)
                # Fallback to the representative peak if every id has
                # gone stale (e.g. user re-ran fitting since the
                # selection was set).
                if len(sel_table) == 0:
                    sel_table = _peaks_from_manual([
                        ManualPeak(
                            radius=self._selected.radius,
                            angle=self._selected.angle,
                            radius_width=self._selected.radius_width,
                            angle_width=self._selected.angle_width,
                            is_ring=self._selected.is_ring,
                            temp_id=self._selected.peak_id,
                        )
                    ])
            else:
                sel_table = _peaks_from_manual([
                    ManualPeak(
                        radius=s.radius,
                        angle=s.angle,
                        radius_width=s.radius_width,
                        angle_width=s.angle_width,
                        is_ring=s.is_ring,
                        temp_id=s.peak_id,
                    )
                    for s in self.selected_peaks()
                ])

        # Fitted-preview overlay: one cyan dashed box per fittable
        # selected peak (primary + Ctrl+click extras). Only meaningful
        # for manual / detected selections — fitted / matched are
        # already on file with their own boxes.
        #
        # Layered painter contract: primary geometry comes from
        # ``_fitted_preview_geom`` (with optional ring mode);
        # extras come from ``_fitted_preview_extras_geoms`` (always
        # segments). The host populates both before invoking
        # ``_render_overlays``; this function just builds the table.
        preview_peaks: list[ManualPeak] = []
        primary_eligible = (
            self._selected is not None
            and self._selected.kind in ("manual", "detected")
        )
        if primary_eligible and self._fitted_preview_geom is not None:
            cr, wr, ca, wa = self._fitted_preview_geom
            if self._fitted_preview_is_ring:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=45.0,
                    radius_width=wr,
                    angle_width=float("inf"),
                    is_ring=True, temp_id=0,
                ))
            else:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=ca,
                    radius_width=wr,
                    angle_width=wa,
                    is_ring=False, temp_id=0,
                ))
        # Extras get a preview whenever the primary is fittable
        # (extras only carry detected, which is always fittable).
        if primary_eligible:
            for cr, wr, ca, wa in self._fitted_preview_extras_geoms:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=ca,
                    radius_width=wr,
                    angle_width=wa,
                    is_ring=False, temp_id=0,
                ))
        preview_table: PeakTable | None = (
            _peaks_from_manual(preview_peaks) if preview_peaks else None
        )

        # Live angular extent of the displayed polar stack so ring
        # overlays (and any segment whose stored angle_width spills
        # past the image bounds) clip to the data instead of the
        # global ±180° fallback.
        extent = self.angular_extent()
        fit_area_table = self._fit_area_table()
        if self._mode == MODE_POLAR:
            self._detected.set_polar(det, extent=extent)
            self._fitted.set_polar(fit, extent=extent)
            self._manual.set_polar(manual_table, extent=extent)
            for kind, item in self._list_items.items():
                item.set_polar(peaks.get(kind), extent=extent)
            if sel_table is not None:
                self._selection.set_polar(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if fit_area_table is not None:
                self._fit_area.set_polar(fit_area_table, extent=extent)
            else:
                self._fit_area.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_polar(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()
        else:
            self._detected.set_cartesian(det, extent=extent)
            self._fitted.set_cartesian(fit, extent=extent)
            self._manual.set_cartesian(manual_table, extent=extent)
            for kind, item in self._list_items.items():
                item.set_cartesian(peaks.get(kind), extent=extent)
            if sel_table is not None:
                self._selection.set_cartesian(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if fit_area_table is not None:
                self._fit_area.set_cartesian(fit_area_table, extent=extent)
            else:
                self._fit_area.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_cartesian(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()

        self._detected.setVisible(self._visibility["detected"])
        self._fitted.setVisible(self._visibility["fitted"])
        self._manual.setVisible(self._visibility["manual"])
        for kind, item in self._list_items.items():
            item.setVisible(self._visibility.get(kind, True))

        # Matched overlays: rebuild items for whatever the current frame has.
        self._render_matched_overlays(frame)
        # The hover outline is drawn outside this path (per mouse move),
        # so re-point it at whatever now sits under a stationary cursor.
        self._refresh_hover()
        # Simulated "expected pattern" overlay (frame-independent data,
        # but rebuilt here so it follows every mode/stack transition).
        self._render_simulation_overlays(frame)

    def _render_matched_overlays(self, frame: int) -> None:
        """Tear down the previous frame's matched items and rebuild for this
        frame. Each structure becomes one ``_PeakShapeItem`` in its assigned
        color, painted in the current display mode (polar / Cartesian).
        """
        self._teardown_matched_items()
        # Identities rendered with an empty tracked subset this frame —
        # kept hidden by _is_matched_item_visible against later refreshes.
        self._matched_empty_uids = set()
        structures = self._matched_per_frame.get(frame, [])
        extent = self.angular_extent()
        vb = self._plot.getViewBox()
        for i, s in enumerate(structures):
            # "Show only tracked peaks" applies per matched peak: draw
            # only the boxes around this structure's TRACKED fitted
            # peaks. A structure whose peaks are all untracked ends up
            # with an empty subset and paints nothing (hidden below).
            peaks = s.peaks
            has_peaks = len(peaks) > 0
            if self._fitted_visible_only is not None and has_peaks:
                kept = [
                    int(pid) for pid in s.peaks.ids
                    if not self._fitted_row_hidden(frame, int(pid))
                ]
                peaks = _peaks_subset(s.peaks, kept)
                has_peaks = len(peaks) > 0
            if not has_peaks:
                self._matched_empty_uids.add(s.unique_id)
            pen_info = self._pen_for_key(s.color_key)
            visible = has_peaks and self._is_matched_item_visible(s.unique_id)
            if self._matched_display_style == "markers":
                # q-map look: circles for peaks, dashed arcs for rings.
                # A structure may map to several items (one scatter + N
                # ring curves); all share the structure's unique_id so
                # visibility toggling reaches every one.
                items = self._matched_marker_items(
                    peaks, pen_info["color"], extent
                )
            else:
                box = _PeakShapeItem(**pen_info)
                if self._mode == MODE_POLAR:
                    box.set_polar(peaks, extent=extent)
                else:
                    box.set_cartesian(peaks, extent=extent)
                items = [box]
            for it in items:
                it.setVisible(visible)
                vb.addItem(it, ignoreBounds=True)
                self._matched_items.append((s.unique_id, it))

        # Unmatched-fitted pseudo-group: every fitted row no structure
        # claims (tracked filter applied), drawn in neutral grey with
        # the same display style — rendered even when the frame has no
        # matched structures at all (then EVERY fitted row is
        # unmatched). Built hidden unless its checkbox is on.
        un = self._unmatched_fitted_subset(frame)
        if un is not None:
            visible = self._is_matched_item_visible(UNMATCHED_UID)
            if self._matched_display_style == "markers":
                items = self._matched_marker_items(
                    un, UNMATCHED_COLOR, extent
                )
            else:
                box = _PeakShapeItem(
                    color=UNMATCHED_COLOR,
                    style=MATCHED_STYLE["style"],
                    width=MATCHED_LINE_WIDTH,
                )
                if self._mode == MODE_POLAR:
                    box.set_polar(un, extent=extent)
                else:
                    box.set_cartesian(un, extent=extent)
                items = [box]
            for it in items:
                it.setVisible(visible)
                vb.addItem(it, ignoreBounds=True)
                self._matched_items.append((UNMATCHED_UID, it))

    def _teardown_matched_items(self) -> None:
        if not self._matched_items:
            return
        vb = self._plot.getViewBox()
        for _uid, item in self._matched_items:
            vb.removeItem(item)
        self._matched_items.clear()

    # -- Internals --

    def _on_radio_toggled(self) -> None:
        # The Cartesian / Polar radios are meaningless in RAW mode (raw
        # frames carry no q-axes), so swallow the toggle. Step 7 hides
        # the radios entirely in raw sessions; this guard is the
        # belt-and-braces backup if the radios are still reachable.
        if self._mode == MODE_RAW:
            return
        new = MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
        if new != self._mode:
            self.set_mode(new)

    def _on_cmap_changed(self, name: str) -> None:
        self._apply_cmap(name)

    def _apply_cmap(self, name: str) -> None:
        # One resolver, shared with the dropdown's gradient swatches, so
        # the strip cannot advertise a ramp the image does not use.
        cmap = resolve_colormap(name)
        if cmap is not None:
            self._view.setColorMap(cmap)

    # -- Cursor readout (status bar) --

    def _on_cursor_pos(self, pt: QPointF) -> None:
        # Hover first: it both draws the pre-selection outline and
        # reports how many boxes are stacked here, which the status bar
        # shows so the user knows a second click has somewhere to go.
        depth = self._update_hover(float(pt.x()), float(pt.y()))
        info = self._compute_cursor_info(pt)
        if info is not None and depth > 1:
            info["overlapping"] = depth
        self.cursorMoved.emit(info)

    def _on_cursor_left(self) -> None:
        self._clear_hover()
        self.cursorMoved.emit(None)

    def _compute_cursor_info(self, pt: QPointF) -> dict | None:
        """Translate a data-space cursor point into a status-bar payload.

        Returns one of three shapes, distinguished by the ``mode`` key:

        - ``"pixel"`` — raw mode: ``row, col, intensity``.
        - ``"cartesian"`` — q-cartesian view: ``q_xy, q_z, intensity``.
        - ``"polar"`` — q-polar view: ``r, theta, intensity``.

        Polar-view axes are **x = radius, y = angle** (matches what
        ``_polar_params`` puts on the plot, NOT the math convention).
        Intensity is looked up against the polar cache when the polar
        view is active; if that returns NaN (uncovered region of the
        polar transform), we fall back to the cartesian grid via the
        derived ``(q_xy, q_z)`` so users see something useful at the
        rim of the polar image.
        """
        x, y = pt.x(), pt.y()
        frame = self.current_frame
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            stack = self._raw_image_stack
            n_fr, n_rows, n_cols = stack.shape
            # The cursor is over the *oriented* preview, so its indices
            # have to be mapped back before reading the stack, which is
            # in file order. Reported row/col are the file-order ones:
            # they name the detector pixel, which is what a readout is
            # for. Without this the intensity was simply the wrong pixel
            # whenever a flip was on, and a transpose on a non-square
            # detector would have read out of bounds.
            row, col = _raw_pixel_index(
                int(round(y)), int(round(x)), (n_rows, n_cols),
                self._raw_flip_lr, self._raw_flip_ud, self._raw_transp,
            )
            if 0 <= frame < n_fr and row >= 0:
                intensity = float(stack[frame, row, col])
            else:
                intensity = float("nan")
            return {
                "mode": "pixel",
                "row": row,
                "col": col,
                "intensity": intensity,
            }
        if self._stack is None:
            return None
        if self._mode == MODE_CARTESIAN:
            q_xy_val = float(x)
            q_z_val = float(y)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
            return {
                "mode": "cartesian",
                "q_xy": q_xy_val,
                "q_z": q_z_val,
                "intensity": intensity,
            }
        # MODE_POLAR — viewer's polar image is laid out with radius on
        # the x axis and angle on the y axis (see _polar_params).
        r_val = float(x)
        theta_deg = float(y)
        intensity = float("nan")
        # The ``_polar_cache is None`` guard fires once
        # ``release_frame_source`` runs, but the cursor can also move in
        # the window where the cache tuple still exists yet the
        # underlying FrameSource is released (e.g. ``clear()`` drops the
        # source without nulling the cache first). Indexing the stack
        # then delegates into ``FrameSource.get_polar`` and raises
        # ``RuntimeError("FrameSource not acquired")``. Require the
        # source to be open so the lookup falls through to the cartesian
        # fallback (which returns NaN safely) instead. Mirror of the
        # guard in ``_lookup_cartesian_intensity``.
        if (
            self._polar_cache is not None
            and self._frame_source is not None
            and self._frame_source.is_open
        ):
            polar_stack, radius_axis, angle_axis = self._polar_cache
            if (
                0 <= frame < polar_stack.shape[0]
                and len(radius_axis) > 0
                and len(angle_axis) > 0
            ):
                r_idx = _bin_index(radius_axis, r_val)
                a_idx = _bin_index(angle_axis, theta_deg)
                intensity = float(polar_stack[frame, r_idx, a_idx])
        # Polar transform leaves NaN in uncovered regions; fall back
        # to the cartesian grid so the readout still shows a real
        # intensity near the edge.
        if intensity != intensity:  # NaN
            q_xy_val, q_z_val = polar_to_qxyz(r_val, theta_deg)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
        return {
            "mode": "polar",
            "r": r_val,
            "theta": theta_deg,
            "intensity": intensity,
        }

    def _lookup_cartesian_intensity(
        self, frame: int, q_xy_val: float, q_z_val: float
    ) -> float:
        """Pixel-bin intensity lookup against the cartesian stack.

        Uses floor-based binning (not nearest-neighbour) so the
        returned intensity stays constant while the cursor is inside
        the same displayed pixel — matches pyqtgraph's ``pos=axis[0]``
        / ``scale=step`` image transform exactly.

        Returns NaN while the FrameSource is released by the silx
        detach/reattach dance (pipeline runs, Add-to-fitted, clear-
        peaks, save-as). Without this guard, ``stack3d[...]`` delegates
        into ``FrameSource.get_cartesian`` which raises
        ``RuntimeError("FrameSource not acquired")`` every time the
        cursor moves over the cartesian plot during a write. Polar
        mode uses the ``self._polar_cache is None`` guard for the
        same reason; this is the cartesian-side mirror.
        """
        if self._stack is None:
            return float("nan")
        if self._frame_source is None or not self._frame_source.is_open:
            return float("nan")
        stack3d = self._stack.image_stack
        qxy_axis = self._stack.q_xy
        qz_axis = self._stack.q_z
        if not (
            0 <= frame < stack3d.shape[0]
            and len(qxy_axis) > 0
            and len(qz_axis) > 0
        ):
            return float("nan")
        qxy_idx = _bin_index(qxy_axis, q_xy_val)
        qz_idx = _bin_index(qz_axis, q_z_val)
        return float(stack3d[frame, qz_idx, qxy_idx])
