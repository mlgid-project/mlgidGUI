"""Matched-structure and Expected-pattern overlay state: color index, per-CIF effective colors, simulation reflections, visibility, filters and score cutoffs.

Plain mixin over ``GIWAXSImageViewer``: no __init__, no Signals; all state
lives on the combined class. Split out of ``image_viewer`` in the 2026
source split.
"""
from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from mlgidgui import simulation_pattern
from mlgidgui.file_model import PeakTable
from mlgidgui.viewer_items import (
    _PeakShapeItem,
    _clip_angle,
    _peaks_subset,
)
from mlgidgui import peak_lists
from mlgidgui.viewer_styles import (
    MATCHED_MARKER_SIZE,
    MODE_CARTESIAN,
    MODE_POLAR,
    SIM_OVERLAY_OPACITY,
    UNMATCHED_KEY,
    UNMATCHED_UID,
    _SIM_STATE_COLORS,
    sim_state_colors,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    _save_matched_pen_overrides,
    save_overlay_pen_overrides,
    _sim_intensity_scale,
    _sim_marker_size,
    matched_pen_for,
)

logger = logging.getLogger(__name__)


class ViewerOverlaysMixin:
    # -- Matched-structure API --

    def set_matched_structures(
        self, frame: int, structures: list[MatchedStructure]
    ) -> None:
        """Replace the list of matched structures for ``frame``.

        Visibility flags for previously-seen ``unique_id``s on this frame are
        preserved; new structures default to visible. Re-renders if ``frame``
        is the one currently shown.
        """
        self._matched_per_frame[frame] = list(structures)
        # Visibility is identity-keyed and persists across frames, so
        # there is nothing to prune here; new identities default to
        # visible via the ``.get(..., True)`` lookups. Assign each a
        # stable colour index up front so the swatch/overlay agree.
        for s in structures:
            self._color_index_for(s)
        if frame == self.current_frame:
            self._render_overlays(frame)
            self.matchedStructuresChanged.emit(frame, list(structures))

    def matched_structures(self, frame: int) -> list[MatchedStructure]:
        return list(self._matched_per_frame.get(frame, []))

    def matched_color(self, structure: MatchedStructure) -> str:
        """Return the hex color assigned to a structure on the current frame.
        Deterministic per insertion order within the frame so the Display
        panel and the overlay agree without extra plumbing.
        """
        return self.matched_pen(structure)["color"]

    def _color_index_for_key(self, key: tuple) -> int:
        """Stable palette index for a structure identity (``color_key``),
        assigned on first sight and reused thereafter so the colour is
        constant across frames and shared by the real overlay, the
        interpolated overlay, and the Display-dock swatch."""
        return self._matched_color_index.setdefault(
            key, len(self._matched_color_index)
        )

    def _color_index_for(self, structure: MatchedStructure) -> int:
        return self._color_index_for_key(structure.color_key)

    def seed_matched_colors(self, keys) -> None:
        """Pre-assign palette indices for ``keys`` (an ordered iterable of
        ``color_key`` tuples) so colours are deterministic regardless of
        which frame is rendered first. Idempotent: keys already assigned
        keep their index. The host seeds this from the full set of matched
        identities (sorted) when a scan is tracked, so a structure seen
        only via interpolation gets the same colour as on its real frames.
        """
        for key in keys:
            self._color_index_for_key(tuple(key))

    def _pen_for_key(self, key: tuple) -> dict:
        """Palette pen for an identity, with the user's pen layered on top.

        The override is PARTIAL: whatever the user did not set stays
        palette-driven, so a structure whose width was raised still
        picks up a distinguishable colour and dash pattern when the
        file's structure list changes.
        """
        pen = matched_pen_for(self._color_index_for_key(key))
        pen.update(self._matched_pen_overrides.get(key) or {})
        return pen

    def matched_pen(self, structure: MatchedStructure) -> dict:
        """Return the full ``{color, style, width}`` pen for ``structure``.

        Keyed by the structure's cross-frame identity so the panel swatch
        and the overlay share one stable colour on every frame.
        """
        return self._pen_for_key(structure.color_key)

    def set_matched_pen(self, key: tuple, pen: dict | None) -> None:
        """Set or clear the user's pen for a structure identity.

        ``pen`` is a partial ``{color, style, width}``; ``None`` (or an
        empty dict) puts the identity back on the automatic palette
        entirely. Persisted app-wide; re-renders the current frame's
        overlays and re-emits ``matchedStructuresChanged`` so both
        legends redraw their swatches.
        """
        key = tuple(key)
        if pen:
            self._matched_pen_overrides[key] = dict(pen)
        else:
            self._matched_pen_overrides.pop(key, None)
        _save_matched_pen_overrides(self._matched_pen_overrides)
        self._render_overlays(self.current_frame)
        self.matchedStructuresChanged.emit(
            self.current_frame, self.matched_structures(self.current_frame)
        )

    def set_matched_color(self, key: tuple, color: str | None) -> None:
        """Colour-only shorthand for ``set_matched_pen``.

        ``None`` clears the whole pen, matching what it meant before
        line style and width were editable.
        """
        self.set_matched_pen(key, {"color": str(color)} if color else None)

    def matched_pen_override(self, key: tuple) -> dict:
        """The user's partial pen for an identity ({} when untouched)."""
        return dict(self._matched_pen_overrides.get(tuple(key)) or {})

    # --- detected / fitted overlay pens ---

    def overlay_pen(self, kind: str) -> dict:
        """The effective ``{color, style, width}`` for an overlay kind.

        A registered extra list starts from the preset of the kind it is
        treated as, so a detected-flavoured list looks like detected
        until the user says otherwise -- and the pen editor then
        overrides it exactly as it does for the built-in layers.
        """
        base = kind
        if peak_lists.is_list_kind(kind):
            spec = self.peak_list_spec(kind)
            base = spec.treat_as if spec is not None else "detected"
        pen = dict(OVERLAY_STYLE[base])
        pen.update(self._overlay_pen_overrides.get(kind) or {})
        return pen

    def overlay_pen_override(self, kind: str) -> dict:
        """The user's partial pen for an overlay kind ({} when untouched)."""
        return dict(self._overlay_pen_overrides.get(kind) or {})

    def set_overlay_pen(self, kind: str, pen: dict | None) -> None:
        """Set or clear the user's pen for the detected / fitted overlay.

        ``pen`` is a partial ``{color, style, width}``; ``None`` puts the
        kind back on its ``OVERLAY_STYLE`` preset. Persisted app-wide
        and applied to the live item, so the image follows the moment
        the user moves a control.
        """
        if kind not in OVERLAY_KINDS and self._overlay_item(kind) is None:
            raise ValueError(f"Unknown overlay kind: {kind!r}")
        if pen:
            self._overlay_pen_overrides[kind] = dict(pen)
        else:
            self._overlay_pen_overrides.pop(kind, None)
        save_overlay_pen_overrides(self._overlay_pen_overrides)
        item = self._overlay_item(kind)
        if item is not None:
            item.set_pen(**self.overlay_pen(kind))
        self.overlayPenChanged.emit(kind)

    def cif_color_overrides(self) -> dict[str, str]:
        """CIF-level view of the colour overrides, for consumers keyed by
        CIF name only (the phase views). When several hkl rows of one CIF
        are overridden, the smallest sorted ``(h, k, l)`` wins."""
        out: dict[str, str] = {}
        for key in sorted(self._matched_pen_overrides):
            color = self._matched_pen_overrides[key].get("color")
            if color:
                # A width-only or style-only override is not a colour
                # choice; the CIF keeps its palette hue.
                out.setdefault(str(key[0]), str(color))
        return out

    def cif_effective_colors(self) -> dict[str, str]:
        """Effective per-CIF colour exactly as the Display legend shows
        it: the automatic palette pen for the identity with the user's
        override layered on top (``_pen_for_key``). Where several (hkl)
        identities of one CIF have colours assigned, the smallest sorted
        identity wins — the same convention as ``cif_color_overrides``.

        Feeds the phase views' colour map so the q-map, amplitude bands
        and structure toggles agree with the Display-dock swatches even
        for structures whose colour was never hand-picked (the views'
        own hue wheel is only the fallback for CIFs this viewer has not
        assigned yet). Custom picks keep CIF-level priority: a pick on
        ANY (hkl) row of a CIF wins for that CIF, exactly as
        ``cif_color_overrides`` alone behaved before."""
        out: dict[str, str] = dict(self.cif_color_overrides())
        for key in sorted(self._matched_color_index):
            if key == UNMATCHED_KEY:
                continue
            out.setdefault(str(key[0]), str(self._pen_for_key(key)["color"]))
        return out

    def matched_visibility(self, frame: int, unique_id: str) -> bool:
        """Show/hide state for the structure with ``unique_id`` on
        ``frame`` — resolved to its cross-frame identity so the state
        follows the structure between frames."""
        for s in self._matched_per_frame.get(frame, []):
            if s.unique_id == unique_id:
                return self._matched_visibility.get(s.color_key, True)
        return True

    def unmatched_visible(self) -> bool:
        """Checkbox state of the unmatched-fitted pseudo-group (defaults
        OFF — the fitted overlay already draws every fitted row)."""
        return bool(self._matched_visibility.get(UNMATCHED_KEY, False))

    def set_unmatched_visible(self, visible: bool) -> None:
        """Show/hide the unmatched-fitted pseudo-group. Persists across
        frames like a structure checkbox (identity-keyed)."""
        self._matched_visibility[UNMATCHED_KEY] = bool(visible)
        self._apply_matched_item_visibility()

    def has_unmatched_fitted(self, frame: int) -> bool:
        """True when ``frame`` has fitted rows not claimed by any
        matched structure (tracked-peak filter applied) — drives the
        Display dock's "Unmatched fitted peaks" row."""
        return self._unmatched_fitted_subset(frame) is not None

    def _unmatched_fitted_subset(self, frame: int):
        """Fitted rows on ``frame`` outside every matched structure's
        ``peaks.ids`` (minus rows the tracked-peak filter hides), or
        ``None`` when there are none."""
        fit = self._frame_peaks.get(frame, {}).get("fitted")
        if fit is None or len(fit) == 0:
            return None
        claimed: set = set()
        for s in self._matched_per_frame.get(frame, []):
            claimed.update(
                int(p) for p in np.asarray(s.peaks.ids, dtype=int)
            )
        kept = [
            int(pid) for pid in fit.ids
            if int(pid) not in claimed
            and not self._fitted_row_hidden(frame, int(pid))
        ]
        if not kept:
            return None
        if len(kept) == len(fit):
            return fit
        return _peaks_subset(fit, kept)

    def set_matched_display_style(self, style: str) -> None:
        """Switch the matched overlay between ``"boxes"`` and
        ``"markers"`` (hollow circles for peaks, dashed arcs/lines for
        rings — the q-map look). Re-renders the current frame; no other
        overlay is affected."""
        style = "markers" if style == "markers" else "boxes"
        if style == self._matched_display_style:
            return
        self._matched_display_style = style
        self._render_overlays(self.current_frame)

    def _matched_marker_items(self, peaks, color, extent) -> list:
        """Build q-map-style items for one structure's peaks: a hollow
        circle scatter for the spots and a dashed arc (Cartesian) /
        constant-radius line (polar) per ring. Returns pg items to add
        and track alongside the box items."""
        items: list = []
        if peaks is None or len(peaks) == 0:
            return items
        is_ring = np.asarray(peaks.is_ring, dtype=bool)
        spot = np.flatnonzero(~is_ring)
        if spot.size:
            r = np.asarray(peaks.radius, dtype=float)[spot]
            ang = np.asarray(peaks.angle, dtype=float)[spot]
            if self._mode == MODE_POLAR:
                xs, ys = r, ang
            else:
                a = np.deg2rad(ang)
                xs, ys = r * np.cos(a), r * np.sin(a)
            items.append(pg.ScatterPlotItem(
                x=xs, y=ys, symbol="o", size=MATCHED_MARKER_SIZE,
                pen=pg.mkPen(QColor(color), width=1.5), brush=None,
            ))
        # Ring and peak share the structure's colour; the SHAPE tells
        # them apart (a big dashed quarter-circle arc vs small circles),
        # so one well-distinguishable colour per structure is enough and
        # the eye isn't asked to separate two shades of the same hue.
        for i in np.flatnonzero(is_ring):
            clip = _clip_angle(
                float(peaks.angle[i]), float(peaks.angle_width[i]),
                extent=extent,
            )
            if clip is None:
                continue
            a_lo, a_hi = clip
            r = float(peaks.radius[i])
            pen = pg.mkPen(QColor(color), width=1.5, style=Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            if self._mode == MODE_POLAR:
                curve = pg.PlotCurveItem(
                    x=np.array([r, r]), y=np.array([a_lo, a_hi]), pen=pen,
                )
            else:
                th = np.linspace(np.deg2rad(a_lo), np.deg2rad(a_hi), 100)
                curve = pg.PlotCurveItem(
                    x=r * np.cos(th), y=r * np.sin(th), pen=pen,
                )
            items.append(curve)
        return items

    # -- "Expected pattern" simulation overlay --

    def set_simulation_pattern(self, pattern) -> None:
        """Install (or clear, with None) the simulated-pattern overlay.

        ``pattern`` is a ``simulation_pattern.SimulatedPattern``.
        Replacing it drops any reflection selection — selection indices
        are only meaningful within one pattern.
        """
        self._sim_pattern = pattern
        if self._sim_selected:
            self._sim_selected.clear()
            self.simulationSelectionChanged.emit([])
        self._render_simulation_overlays(self.current_frame)

    def clear_simulation_pattern(self) -> None:
        self.set_simulation_pattern(None)

    def set_simulation_visible(self, visible: bool) -> None:
        self._sim_visible = bool(visible)
        self._render_simulation_overlays(self.current_frame)

    def set_simulation_min_intensity(self, cutoff: float) -> None:
        """Hide reflections below ``cutoff`` (relative intensity, 0-1).

        Selected reflections falling below the cutoff drop out of the
        selection — they are no longer visible, so keeping them queued
        for injection would act on markers the user can't see.
        """
        self._sim_min_intensity = max(0.0, min(1.0, float(cutoff)))
        if self._sim_pattern is not None and self._sim_selected:
            visible = {r.index for r in self._visible_sim_reflections()}
            pruned = self._sim_selected & visible
            if pruned != self._sim_selected:
                self._sim_selected = pruned
                self.simulationSelectionChanged.emit(sorted(pruned))
        self._render_simulation_overlays(self.current_frame)

    def simulation_pattern(self):
        return self._sim_pattern

    def simulation_selected(self) -> list[int]:
        return sorted(self._sim_selected)

    def simulated_visible_reflections(self) -> list:
        """Reflections above the intensity cutoff (the rendered set)."""
        return self._visible_sim_reflections()

    def _visible_sim_reflections(self) -> list:
        if self._sim_pattern is None:
            return []
        cut = self._sim_min_intensity
        return [
            r for r in self._sim_pattern.reflections
            if r.rel_intensity >= cut
        ]

    def _sim_q_max(self) -> float | None:
        """Largest |q| the current stack can display, or None.

        Used to drop simulated rings that lie entirely outside the
        data (the upstream powder pattern applies no q-range cut).
        """
        if self._polar_cache is not None:
            _, radius, _ = self._polar_cache
            if radius.size:
                return float(np.nanmax(radius))
        stack = self._stack
        if stack is None:
            return None
        try:
            return float(np.hypot(
                np.nanmax(np.abs(np.asarray(stack.q_xy, dtype=float))),
                np.nanmax(np.abs(np.asarray(stack.q_z, dtype=float))),
            ))
        except (AttributeError, TypeError, ValueError):
            return None

    def _render_simulation_overlays(self, frame: int) -> None:
        """Tear down + rebuild the simulated-reflection items (same
        per-frame lifecycle as the matched overlays). Follows the
        matched display style: "boxes" draws each reflection as the
        detected box an injection would create; "markers" draws the
        q-map look (diamonds + dashed arcs). Per-reflection colour
        encodes state: selected (white) > explained by a matched box
        (green) > missed (orange)."""
        self._teardown_sim_items()
        if (
            self._sim_pattern is None
            or not self._sim_visible
            or self._mode not in (MODE_POLAR, MODE_CARTESIAN)
        ):
            return
        reflections = self._visible_sim_reflections()
        q_max = self._sim_q_max()
        if q_max is not None:
            # Powder patterns carry rings beyond the data's q-range
            # (no upstream cut) — don't render or hit-test those.
            reflections = [
                r for r in reflections
                if not (r.is_ring and r.radius > q_max)
            ]
        if not reflections:
            return
        states = self._sim_reflection_states(frame, reflections)
        if self._matched_display_style == "markers":
            items = self._sim_marker_style_items(reflections, states)
        else:
            items = self._sim_box_style_items(frame, reflections, states)
        vb = self._plot.getViewBox()
        for item in items:
            item.setOpacity(SIM_OVERLAY_OPACITY)
            vb.addItem(item, ignoreBounds=True)
            self._sim_items.append(item)

    def _sim_matched_boxes(self, frame: int):
        """The SELECTED structure's matched boxes on ``frame`` (same
        CIF stem, any orientation — the matcher may attribute a peak
        to a different texture of the same phase), merged into one
        duck-typed table for ``classify_explained``. None when nothing
        of that phase is matched there. Scoped to the overlay's CIF so
        "explained" (green) means "matched to THIS phase" — a peak
        matched to a different structure does not count."""
        if self._sim_pattern is None:
            return None
        want = simulation_pattern.cif_stem(self._sim_pattern.cif)
        return simulation_pattern.merge_boxes([
            s.peaks for s in self._matched_per_frame.get(frame, [])
            if simulation_pattern.cif_stem(s.cif) == want
        ])

    def _sim_seed_widths(self, frame: int) -> tuple[float, float]:
        """The injection seed box for ``frame`` (median fitted widths)
        — also the coverage-recognition floor: fitted rows carry tight
        2σ widths around the peak's REAL position, so a peak that
        landed inside the box an injection would use must count as
        covering its reflection (see ``classify_explained``)."""
        tables = self._frame_peaks.get(frame) or {}
        return simulation_pattern.default_box_size(tables.get("fitted"))

    def simulation_explained_mask(self, frame: int) -> np.ndarray:
        """Which of the visible reflections are explained by the
        frame's matched boxes (aligned with
        ``simulated_visible_reflections()``)."""
        rw, aw = self._sim_seed_widths(frame)
        return simulation_pattern.classify_explained(
            self._visible_sim_reflections(),
            self._sim_matched_boxes(frame),
            seed_radius_width=rw, seed_angle_width=aw,
        )

    def _sim_reflection_states(self, frame: int, reflections) -> list[str]:
        """Per-reflection render state: selected > explained > missed."""
        rw, aw = self._sim_seed_widths(frame)
        explained = simulation_pattern.classify_explained(
            reflections, self._sim_matched_boxes(frame),
            seed_radius_width=rw, seed_angle_width=aw,
        )
        states = []
        for r, exp in zip(reflections, explained):
            if r.index in self._sim_selected:
                states.append("selected")
            elif exp:
                states.append("explained")
            else:
                states.append("missed")
        return states

    def _sim_marker_style_items(self, reflections, states) -> list:
        """q-map look: one hollow-diamond scatter per state bucket for
        the spots, one dashed curve per ring (state-coloured)."""
        items: list = []
        polar = self._mode == MODE_POLAR
        for state, color in sim_state_colors().items():
            spots = [
                r for r, s in zip(reflections, states)
                if s == state and not r.is_ring
            ]
            if not spots:
                continue
            selected = state == "selected"
            if polar:
                xs = np.array([r.radius for r in spots])
                ys = np.array([r.angle for r in spots])
            else:
                xs = np.array([r.q_xy for r in spots])
                ys = np.array([r.q_z for r in spots])
            items.append(pg.ScatterPlotItem(
                x=xs, y=ys, symbol="d",
                size=[
                    _sim_marker_size(r.rel_intensity) + (2 if selected else 0)
                    for r in spots
                ],
                pen=pg.mkPen(QColor(color), width=2.2 if selected else 1.5),
                brush=None,
                data=[r.index for r in spots],
            ))
        extent = self.angular_extent()
        a_lo, a_hi = extent if extent is not None else (0.0, 90.0)
        for r, state in zip(reflections, states):
            if not r.is_ring:
                continue
            selected = state == "selected"
            width = 1.0 + 1.5 * _sim_intensity_scale(r.rel_intensity)
            pen = pg.mkPen(
                QColor(sim_state_colors()[state]),
                width=width + (1.0 if selected else 0.0),
                style=Qt.PenStyle.DashLine,
            )
            pen.setCosmetic(True)
            if polar:
                curve = pg.PlotCurveItem(
                    x=np.array([r.radius, r.radius]),
                    y=np.array([a_lo, a_hi]), pen=pen,
                )
            else:
                th = np.linspace(np.deg2rad(a_lo), np.deg2rad(a_hi), 100)
                curve = pg.PlotCurveItem(
                    x=r.radius * np.cos(th), y=r.radius * np.sin(th),
                    pen=pen,
                )
            items.append(curve)
        return items

    def _sim_box_style_items(self, frame: int, reflections, states) -> list:
        """Boxes look (matches the matched 'Boxes' style): each spot is
        drawn as the detected box an injection would create — centred
        on the reflection, sized to the frame's median fitted box —
        and each ring as a full-extent radial band. One dashed
        ``_PeakShapeItem`` per state bucket carries the colour."""
        tables = self._frame_peaks.get(frame) or {}
        rw, aw = simulation_pattern.default_box_size(tables.get("fitted"))
        extent = self.angular_extent()
        items: list = []
        for state, color in sim_state_colors().items():
            rows = [
                r for r, s in zip(reflections, states) if s == state
            ]
            if not rows:
                continue
            selected = state == "selected"
            table = PeakTable(
                q_xy=np.array([r.q_xy for r in rows]),
                q_z=np.array([r.q_z for r in rows]),
                angle=np.array([
                    45.0 if r.is_ring else r.angle for r in rows
                ]),
                radius=np.array([r.radius for r in rows]),
                angle_width=np.array([
                    np.inf if r.is_ring else aw for r in rows
                ]),
                radius_width=np.full(len(rows), rw),
                is_ring=np.array([r.is_ring for r in rows], dtype=bool),
                ids=np.array([r.index for r in rows]),
                score=np.zeros(len(rows)),
                amplitude=np.zeros(len(rows)),
            )
            box = _PeakShapeItem(
                color=color,
                style=Qt.PenStyle.DashLine,
                width=2.2 if selected else 1.4,
            )
            if self._mode == MODE_POLAR:
                box.set_polar(table, extent=extent)
            else:
                box.set_cartesian(table, extent=extent)
            items.append(box)
        return items

    def _sim_hit_reflection(self, x: float, y: float):
        """Nearest visible simulated reflection within click tolerance
        of data-space point ``(x, y)``, or None. Distances are computed
        in screen pixels so the tolerance is zoom-independent."""
        if self._sim_pattern is None:
            return None
        try:
            px, py = self._plot.getViewBox().viewPixelSize()
        except Exception:
            return None
        if not (np.isfinite(px) and np.isfinite(py)) or px <= 0 or py <= 0:
            return None
        polar = self._mode == MODE_POLAR
        reflections = self._visible_sim_reflections()
        q_max = self._sim_q_max()
        best = None
        best_d = np.inf
        for r in reflections:
            if r.is_ring:
                if q_max is not None and r.radius > q_max:
                    continue
                # Radial distance to the ring's constant-|q| line/arc.
                if polar:
                    d = abs(x - r.radius) / px
                else:
                    d = abs(float(np.hypot(x, y)) - r.radius) / min(px, py)
                tol = 4.0
            else:
                if polar:
                    dx, dy = (x - r.radius) / px, (y - r.angle) / py
                else:
                    dx, dy = (x - r.q_xy) / px, (y - r.q_z) / py
                d = float(np.hypot(dx, dy))
                tol = _sim_marker_size(r.rel_intensity) / 2.0 + 2.0
            if d <= tol and d < best_d:
                best, best_d = r, d
        return best

    def _sim_toggle_at(self, x: float, y: float) -> bool:
        """Toggle the simulated reflection under the click; True when
        one was hit (the caller stops peak selection)."""
        hit = self._sim_hit_reflection(x, y)
        if hit is None:
            return False
        if hit.index in self._sim_selected:
            self._sim_selected.discard(hit.index)
        else:
            self._sim_selected.add(hit.index)
        self._render_simulation_overlays(self.current_frame)
        self.simulationSelectionChanged.emit(sorted(self._sim_selected))
        return True

    def select_missed_simulated(self, frame: int | None = None) -> list[int]:
        """Select every visible reflection not matched to the selected
        structure on ``frame`` — exactly the ones rendered orange.

        Reflections sitting on fitted-but-unmatched peaks (or on peaks
        matched to a DIFFERENT phase) are selected too: the injection
        planner skips duplicate boxes over existing fitted peaks and
        instead schedules a re-match so those peaks can be claimed.
        Returns the new selection (sorted reflection indices).
        """
        if self._sim_pattern is None:
            return []
        frame = self.current_frame if frame is None else int(frame)
        reflections = self._visible_sim_reflections()
        rw, aw = self._sim_seed_widths(frame)
        matched = simulation_pattern.classify_explained(
            reflections, self._sim_matched_boxes(frame),
            seed_radius_width=rw, seed_angle_width=aw,
        )
        q_max = self._sim_q_max()
        self._sim_selected = {
            r.index
            for r, m in zip(reflections, matched)
            if not m
            and not (r.is_ring and q_max is not None and r.radius > q_max)
        }
        self._render_simulation_overlays(frame)
        sel = sorted(self._sim_selected)
        self.simulationSelectionChanged.emit(sel)
        return sel

    def clear_simulation_selection(self) -> None:
        if not self._sim_selected:
            return
        self._sim_selected.clear()
        self._render_simulation_overlays(self.current_frame)
        self.simulationSelectionChanged.emit([])

    def _teardown_sim_items(self) -> None:
        if not self._sim_items:
            return
        vb = self._plot.getViewBox()
        for item in self._sim_items:
            vb.removeItem(item)
        self._sim_items.clear()

    def _apply_matched_item_visibility(self) -> None:
        """Refresh visibility of every current-frame matched item
        (gated by ``_is_matched_item_visible``)."""
        for uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(uid))

    def set_matched_master_visible(self, visible: bool) -> None:
        self._matched_master_visible = visible
        self._apply_matched_item_visibility()

    def set_matched_structure_visible(self, unique_id: str, visible: bool) -> None:
        # Store against the structure's cross-frame identity so the
        # choice persists as the user scrubs frames.
        key = None
        for s in self._matched_per_frame.get(self.current_frame, []):
            if s.unique_id == unique_id:
                key = s.color_key
                break
        if key is None:
            return
        self._matched_visibility[key] = visible
        # Refresh every current-frame item (cheap; small list) so any
        # item sharing this identity updates too.
        self._apply_matched_item_visibility()

    def set_detected_score_cutoff(self, value: float) -> None:
        """Hide detected-overlay rows whose score is below ``value``.

        Re-renders overlays for the current frame so the change
        takes effect immediately. The cutoff is applied with a
        small float-epsilon so a slider at exactly the maximum
        score still shows that row (otherwise FP roundoff on stored
        scores can make ``score >= cutoff`` spuriously False).
        """
        self._detected_score_cutoff = float(value)
        self._render_overlays(self.current_frame)

    def set_fitted_visible_only(
        self, mapping: dict[int, set[int]] | None
    ) -> None:
        """Render ONLY the given fitted rows (``{frame: {id, ...}}``).

        Driven by the Scan-tracking dock's "Show only tracked peaks"
        toggle: pass the per-frame member ids of the surviving tracks
        and every other fitted row disappears from the overlay (frames
        not in the mapping show none), so scrubbing the scan reveals
        whether the tracks are visually consistent. Pass ``None`` to
        disable the filter entirely. Non-destructive render-time
        subset, like the Display dock's min-score cutoff; re-renders
        the current frame immediately.
        """
        if mapping is None:
            self._fitted_visible_only = None
        else:
            self._fitted_visible_only = {
                int(f): {int(i) for i in ids}
                for f, ids in mapping.items()
            }
        self._render_overlays(self.current_frame)

    def _fitted_row_hidden(self, frame: int, peak_id: int) -> bool:
        """True when the "show only tracked peaks" filter is active and
        this fitted row is not in the visible whitelist for ``frame``.

        Such a row is not rendered, so it must not be click- or
        Ctrl+A-selectable either — what you can't see, you can't select.
        Used by every fitted hit-test path in ``_on_select_at`` and by
        ``_select_all_of_kind_on_frame``.
        """
        vo = self._fitted_visible_only
        return vo is not None and int(peak_id) not in vo.get(int(frame), set())

    def set_matched_filter_hidden(self, hidden_ids) -> None:
        """Hide overlays for structures filtered out of the
        Display-dock search.

        Independent of ``set_matched_structure_visible`` (the
        checkbox-driven path): an empty filter set restores
        whatever the user's per-structure checkboxes say, while a
        non-empty set forces those ``unique_id``s off regardless of
        checkbox state. This lets the user search by CIF substring
        without losing their checkbox selections when the filter
        clears.
        """
        self._matched_filter_hidden = set(hidden_ids)
        for uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(uid))

    def _is_matched_item_visible(self, unique_id: str) -> bool:
        if not self._matched_master_visible:
            return False
        # The unmatched-fitted pseudo-group has no per-frame structure
        # to resolve; its own identity-keyed checkbox (default OFF)
        # decides, under the master toggle above.
        if unique_id == UNMATCHED_UID:
            return bool(self._matched_visibility.get(UNMATCHED_KEY, False))
        # Structures whose tracked-peak subset is empty on this frame were
        # rendered with nothing to draw; they must stay hidden even when a
        # later visibility refresh (e.g. the Display-dock filter) re-runs.
        if unique_id in self._matched_empty_uids:
            return False
        if unique_id in self._matched_filter_hidden:
            return False
        # Resolve the current-frame item to its cross-frame identity and
        # read the identity-keyed checkbox state.
        for s in self._matched_per_frame.get(self.current_frame, []):
            if s.unique_id == unique_id:
                return self._matched_visibility.get(s.color_key, True)
        return True

    def _overlay_item(self, kind: str) -> _PeakShapeItem | None:
        if peak_lists.is_list_kind(kind):
            return self._list_items.get(kind)
        return {
            "detected": self._detected,
            "fitted":   self._fitted,
            "manual":   self._manual,
        }.get(kind)

    # --- registered extra peak lists ---

    def peak_list_specs(self) -> list:
        """The extra peak lists this viewer is currently drawing."""
        return list(self._peak_list_specs)

    def peak_list_spec(self, kind: str):
        """The spec behind a list kind, or None when it is not registered."""
        return peak_lists.spec_for_kind(self._peak_list_specs, kind)

    def set_peak_list_specs(self, specs) -> None:
        """Rebuild the extra-list overlay items from the registry.

        Called at construction and again whenever Settings changes. Items
        for lists that went away are removed from the viewbox; the ones
        that stay keep their item (and so their pen) rather than being
        rebuilt, so a Settings visit that only renames a list does not
        make the image flicker.

        A list's item is added AFTER the fixed three, so a custom layer
        paints above detected and fitted -- it is the thing the user
        went out of their way to see.
        """
        specs = list(specs)
        self._peak_list_specs = specs
        wanted = {spec.kind: spec for spec in specs}
        vb = self._plot.getViewBox()
        for kind in list(self._list_items):
            if kind not in wanted:
                item = self._list_items.pop(kind)
                try:
                    vb.removeItem(item)
                except Exception:
                    logger.debug("removing a peak-list item failed",
                                 exc_info=True)
                self._visibility.pop(kind, None)
                if (
                    self._selected is not None
                    and self._selected.kind == kind
                ):
                    self.clear_selection()
        for kind, spec in wanted.items():
            if kind in self._list_items:
                continue
            item = _PeakShapeItem(**self.overlay_pen(kind))
            item.setZValue(35)
            vb.addItem(item, ignoreBounds=True)
            self._list_items[kind] = item
            self._visibility.setdefault(kind, True)
            item.setVisible(self._visibility[kind])
        if self._mode in (MODE_CARTESIAN, MODE_POLAR):
            self._render_overlays(self.current_frame)

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_CARTESIAN, MODE_POLAR) or mode == self._mode:
            return
        self._mode = mode
        if mode == MODE_POLAR:
            self._radio_polar.setChecked(True)
        else:
            self._radio_cart.setChecked(True)
        self._sync_roi()  # ROI exists only in polar mode
        # The remembered cursor point is in the space we just left
        # (polar is (r, angle), Cartesian is (q_xy, q_z)), so re-running
        # the hover on it would outline nonsense. The next mouse move
        # re-establishes it.
        self._clear_hover()
        self._render_active_mode()

    @property
    def mode(self) -> str:
        return self._mode

    def clear(self) -> None:
        self._view.clear()
        self._detected.clear_path()
        self._fitted.clear_path()
        self._manual.clear_path()
        self._selection.clear_path()
        self._fit_area.clear_path()
        self._clear_hover()
        self._fitted_preview.clear_path()
        self._fitted_preview_geom = None
        self._fitted_preview_extras_geoms = []
        self._frame_peaks.clear()
        self._manual_peaks.clear()
        self._fitted_visible_only = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Tear down all matched items and forget per-frame state.
        self._teardown_matched_items()
        self._matched_empty_uids = set()
        self._matched_per_frame.clear()
        self._matched_visibility.clear()
        self._matched_color_index.clear()
        # Simulated pattern belongs to the closed file's parse; the
        # visibility + intensity-cutoff prefs survive (dock-mirrored).
        self._teardown_sim_items()
        self._sim_pattern = None
        if self._sim_selected:
            self._sim_selected.clear()
            self.simulationSelectionChanged.emit([])
        had_selection = self._selected is not None
        self._selected = None
        self._selected_extras = []
        self._roi_drag_before = None
        self._sync_roi()
        # Release the long-lived h5py read handle before dropping the
        # stack reference — temp-dir cleanup at session close otherwise
        # fails on Windows because the file is still open.
        if self._frame_source is not None:
            self._frame_source.release()
            self._frame_source = None
        self._stack = None
        self._drop_raw_stack()
        self._polar_cache = None
        self._display_params = None
        self._user_levels = None
        self._frame_index = 0
        if had_selection:
            self.selectionChanged.emit(None)
