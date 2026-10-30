"""The matched-structures legend of the Display dock and its Expected-pattern twin: rebuild, colour popup, probability/score sliders, filters and visibility toggles.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)
from mlgidgui.pen_picker import PenPopup
from mlgidgui.image_viewer import MATCHED_STYLE, UNMATCHED_COLOR
from mlgidgui.widgets import make_pen_swatch as _make_pen_swatch

import logging

logger = logging.getLogger(__name__)


class MatchedLegendMixin:
    def _refresh_matched_panel(self, _frame: int, structures: list) -> None:
        """Rebuild the per-structure rows under the Matched-peaks master.

        Called on every frame change and after a fresh entry load. We blow
        away the old QCheckBox widgets and create new ones — the structure
        list is small (1-N rows) so this is cheap.

        Every row is built twice: once for the Display dock and once for
        the Expected-pattern dock's twin legend. The twins share their
        toggled handlers, which mirror each change to the counterpart
        (``_on_matched_structure_toggled``, ``_on_matched_master_toggled``,
        ``_on_unmatched_row_toggled``).

        The active filter (see ``_apply_matched_filter``) is re-applied
        at the end so newly-built rows respect the current search
        text without the user needing to retype.
        """
        # Clear children of both dynamic containers.
        for lay in (
            self._matched_struct_layout, self._sim_legend_struct_layout,
        ):
            while lay.count():
                item = lay.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        self._matched_filter_empty_label = None
        self._sim_legend_filter_empty_label = None
        self._matched_struct_checkboxes.clear()
        self._matched_struct_rows.clear()
        self._matched_struct_probs.clear()
        self._sim_legend_struct_checkboxes.clear()
        self._sim_legend_struct_rows.clear()
        self._unmatched_row_checks.clear()

        # Fitted rows no loaded structure claims (tracked filter
        # applied) get their own sub-row below the structures.
        has_unmatched = self.viewer.has_unmatched_fitted(_frame)

        if not structures and not has_unmatched:
            self._add_matched_empty_label(self._matched_struct_layout)
            self._add_matched_empty_label(self._sim_legend_struct_layout)
            return

        for s in structures:
            # Stable per-identity pen (CIF + hkl) so the swatch matches
            # the overlay and stays the same colour across frames.
            pen = self.viewer.matched_pen(s)
            for checks, rows, lay in (
                (self._matched_struct_checkboxes,
                 self._matched_struct_rows,
                 self._matched_struct_layout),
                (self._sim_legend_struct_checkboxes,
                 self._sim_legend_struct_rows,
                 self._sim_legend_struct_layout),
            ):
                row_widget, chk = self._make_matched_struct_row(
                    s, pen, _frame
                )
                lay.addWidget(row_widget)
                checks[s.unique_id] = chk
                rows[s.unique_id] = row_widget
            try:
                self._matched_struct_probs[s.unique_id] = float(s.probability)
            except Exception:
                logger.debug("suppressed exception in MainWindow._refresh_matched_panel", exc_info=True)
                self._matched_struct_probs[s.unique_id] = 0.0

        # Reset the min-probability slider so each frame's first
        # render shows every structure; the user can drag up to
        # filter weak matches.
        if structures:
            self._seed_matched_prob_slider(structures)
            self._apply_matched_filter()

        # "Unmatched fitted peaks" sub-row: the grey pseudo-group of
        # fitted rows no loaded structure claims, rendered in the
        # matched display style (markers/boxes). Off by default (the
        # fitted overlay already draws these rows); the checkbox state
        # persists across frames. Deliberately NOT registered in
        # _matched_struct_checkboxes/rows so the CIF-substring and
        # min-probability filters never touch it.
        if has_unmatched:
            pen = {"color": UNMATCHED_COLOR, **MATCHED_STYLE}
            for lay in (
                self._matched_struct_layout, self._sim_legend_struct_layout,
            ):
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                swatch = QLabel()
                swatch.setPixmap(_make_pen_swatch(pen))
                row.addWidget(swatch)
                chk = QCheckBox("Unmatched fitted peaks")
                chk.setToolTip(
                    "Show every fitted peak that is NOT part of any matched "
                    "structure, in neutral grey, using the matched display "
                    "style (boxes or markers). Follows the Matched master "
                    "toggle and the \"Show only tracked peaks\" filter — "
                    "handy for viewing tracked-but-unmatched peaks as "
                    "markers alongside the structures."
                )
                chk.setChecked(self.viewer.unmatched_visible())
                chk.toggled.connect(self._on_unmatched_row_toggled)
                row.addWidget(chk)
                row.addStretch(1)
                row_widget = QWidget()
                row_widget.setLayout(row)
                lay.addWidget(row_widget)
                self._unmatched_row_checks.append(chk)

    def _add_matched_empty_label(self, layout) -> None:
        """Append the "no matched solutions" placeholder to ``layout``.
        Torn down with the rest of the rows on the next rebuild."""
        label = QLabel("<i>No matched solutions for this frame.</i>")
        label.setWordWrap(True)
        layout.addWidget(label)

    def _make_matched_struct_row(
        self, s, pen: dict, frame: int
    ) -> tuple[QWidget, QCheckBox]:
        """One legend row (pen swatch + visibility checkbox) for
        structure ``s``. Called once per legend; the checkbox routes
        through ``_on_matched_structure_toggled``, which keeps the
        Display and Expected-pattern twins in sync. The swatch is a
        button: clicking it opens the colour-grid popup to recolour
        the structure."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        swatch = QToolButton()
        swatch.setAutoRaise(True)
        # Mirror the *exact* pen used to render the structure on the
        # image so the user can map a row to its overlay shape even
        # when colour repeats — the dashed/dotted swatch flags it.
        pix = _make_pen_swatch(pen)
        swatch.setIcon(QIcon(pix))
        swatch.setIconSize(pix.size())
        swatch.setFixedSize(pix.width() + 4, pix.height() + 4)
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.setToolTip(
            "Click to set this structure's colour, line style and width"
        )
        swatch.clicked.connect(
            lambda _=False, key=s.color_key, b=swatch:
                self._open_matched_pen_popup(b, key)
        )
        row.addWidget(swatch)
        chk = QCheckBox(s.label)
        chk.setChecked(self.viewer.matched_visibility(frame, s.unique_id))
        chk.toggled.connect(
            lambda v, uid=s.unique_id: self._on_matched_structure_toggled(uid, v)
        )
        row.addWidget(chk)
        row.addStretch(1)
        row_widget = QWidget()
        row_widget.setLayout(row)
        return row_widget, chk

    def _open_matched_pen_popup(
        self, button: QToolButton, key: tuple
    ) -> None:
        """Pen editor under a legend swatch.

        Seeded with the pen the structure is actually drawn with, so the
        controls start where the eye is, and with the part of it the
        user had already set, which is what gets extended. "Automatic"
        clears the whole override.
        """
        popup = PenPopup(
            self,
            current=self.viewer._pen_for_key(tuple(key)),
            override=self.viewer.matched_pen_override(key),
            title="Structure colour",
        )
        popup.penChanged.connect(
            lambda pen, k=key: self._on_matched_pen_picked(k, pen)
        )
        popup.resetPicked.connect(
            lambda k=key: self._on_matched_pen_picked(k, None)
        )
        popup.show_under(button)

    def _on_matched_pen_picked(self, key: tuple, pen: dict | None) -> None:
        """Apply a picked pen (``None`` = back to automatic) for a
        structure identity. The viewer re-renders the overlays and
        re-emits ``matchedStructuresChanged``, which rebuilds both
        legends' swatches; the phase views follow via the CIF-level
        override push (colour only — a width is not a hue)."""
        self.viewer.set_matched_pen(key, pen)
        self._push_phase_color_overrides()

    def _push_phase_color_overrides(self) -> None:
        """Mirror the viewer's EFFECTIVE structure colours — the
        automatic Display-legend palette plus any custom picks — to the
        phase views (keyed by CIF there), so the q-map/amplitude views
        and their legend agree with the matched-peaks legend for every
        structure, not only hand-recoloured ones."""
        if self._phase_views_window is not None:
            self._phase_views_window.set_phase_color_overrides(
                self.viewer.cif_effective_colors()
            )

    def _on_unmatched_row_toggled(self, checked: bool) -> None:
        """"Unmatched fitted peaks" toggle (either legend): mirror the
        state to the twin checkbox, then forward to the viewer."""
        for chk in self._unmatched_row_checks:
            with QSignalBlocker(chk):
                chk.setChecked(checked)
        self.viewer.set_unmatched_visible(checked)

    def _on_matched_prob_changed(self, value: int) -> None:
        """Slider 0–100 → readable 0.00–1.00 in the side label, then
        re-apply the composite filter."""
        if hasattr(self, "_matched_prob_value_label"):
            self._matched_prob_value_label.setText(f"{value / 100.0:.2f}")
        self._apply_matched_filter()

    def _on_detected_score_changed(self, value: int) -> None:
        """Slider 0–100 → 0.00–1.00 cutoff forwarded to the viewer.

        Updates the side-label readout, then asks the viewer to
        re-render the detected overlay with the new threshold. The
        filter is applied in ``GIWAXSImageViewer._render_overlays``
        via a row-subset of the detected ``PeakTable``.
        """
        if hasattr(self, "_detected_score_value_label"):
            self._detected_score_value_label.setText(f"{value / 100.0:.2f}")
        if hasattr(self, "viewer"):
            self.viewer.set_detected_score_cutoff(value / 100.0)

    def _seed_detected_score_slider(self) -> None:
        """Reset the Detected min-score slider to the lowest score
        on the current frame so the default shows every detection.

        Called on every frame change and after entry load. Uses
        ``blockSignals`` so the seed doesn't trigger a redundant
        viewer re-render (we're already rendering this frame).
        """
        if not hasattr(self, "_detected_score_slider"):
            return
        frame = self.viewer.current_frame
        peaks = self.viewer._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        try:
            if det is not None and len(det) > 0:
                scores = np.asarray(det.score, dtype=float)
                if scores.size and np.all(np.isfinite(scores)):
                    lo = float(scores.min())
                else:
                    lo = 0.0
            else:
                lo = 0.0
        except Exception:
            logger.debug("suppressed exception in MainWindow._seed_detected_score_slider", exc_info=True)
            lo = 0.0
        lo = max(0.0, min(1.0, lo))
        slider_val = int(round(lo * 100))
        self._detected_score_slider.blockSignals(True)
        try:
            self._detected_score_slider.setValue(slider_val)
        finally:
            self._detected_score_slider.blockSignals(False)
        self._detected_score_value_label.setText(f"{slider_val / 100.0:.2f}")
        # Also apply the new cutoff to the viewer so its render
        # state matches the slider after the silent setValue.
        if hasattr(self, "viewer"):
            self.viewer.set_detected_score_cutoff(slider_val / 100.0)

    def _seed_matched_prob_slider(self, structures: list) -> None:
        """Reset the matched min-probability slider to the lowest
        probability on the current frame so the default shows every
        structure. Mirrors ``_seed_detected_score_slider``."""
        if not hasattr(self, "_matched_prob_slider"):
            return
        try:
            probs = [float(s.probability) for s in structures]
        except Exception:
            logger.debug("suppressed exception in MainWindow._seed_matched_prob_slider", exc_info=True)
            probs = []
        lo = min(probs) if probs else 0.0
        lo = max(0.0, min(1.0, lo))
        slider_val = int(round(lo * 100))
        self._matched_prob_slider.blockSignals(True)
        try:
            self._matched_prob_slider.setValue(slider_val)
        finally:
            self._matched_prob_slider.blockSignals(False)
        self._matched_prob_value_label.setText(f"{slider_val / 100.0:.2f}")

    def _apply_matched_filter(self, *_args) -> None:
        """Hide per-structure rows that fail either:
        (a) the substring filter — label must contain
        ``_matched_filter_edit``'s text (case-insensitive), or
        (b) the min-probability slider — structure probability
        must be ≥ ``_matched_prob_slider`` value / 100.

        Empty substring + zero cutoff = show everything. When all
        rows are hidden by the active filter a "No matches" hint
        replaces them so the empty pane doesn't look like a bug.
        """
        text = ""
        if hasattr(self, "_matched_filter_edit"):
            text = self._matched_filter_edit.text().strip().lower()
        prob_cutoff = 0.0
        if hasattr(self, "_matched_prob_slider"):
            prob_cutoff = self._matched_prob_slider.value() / 100.0

        # Drop leftover "no filter matches" hints before recomputing.
        if self._matched_filter_empty_label is not None:
            self._matched_filter_empty_label.deleteLater()
            self._matched_filter_empty_label = None
        if self._sim_legend_filter_empty_label is not None:
            self._sim_legend_filter_empty_label.deleteLater()
            self._sim_legend_filter_empty_label = None

        any_visible = False
        hidden_uids: set[str] = set()
        for uid, row_widget in self._matched_struct_rows.items():
            chk = self._matched_struct_checkboxes.get(uid)
            label = chk.text().lower() if chk is not None else ""
            substring_ok = (text == "") or (text in label)
            prob = self._matched_struct_probs.get(uid, 0.0)
            # Epsilon = half the slider's natural step (0.01 → 0.005)
            # so a structure with p=1.00 passes when the slider is at
            # 1.00, regardless of FP roundoff in the stored value.
            prob_ok = prob >= prob_cutoff - 0.005
            visible = substring_ok and prob_ok
            row_widget.setVisible(visible)
            # The Expected-pattern twin row follows: a checkbox whose
            # overlay is filter-hidden would otherwise look like a
            # no-op over there.
            twin_row = self._sim_legend_struct_rows.get(uid)
            if twin_row is not None:
                twin_row.setVisible(visible)
            if visible:
                any_visible = True
            else:
                hidden_uids.add(uid)

        # Forward the hidden-uid set to the viewer so filtered-out
        # structures also drop their overlay on the image. Independent
        # of the per-structure checkbox state — see
        # ``GIWAXSImageViewer.set_matched_filter_hidden``.
        if hasattr(self, "viewer"):
            self.viewer.set_matched_filter_hidden(hidden_uids)

        # Only show the "no matches" hint when an active filter has
        # zeroed the list — pure "no matched solutions for this
        # frame" is handled by ``_add_matched_empty_label`` in
        # ``_refresh_matched_panel`` separately.
        filter_active = bool(text) or prob_cutoff > 0.0
        if filter_active and self._matched_struct_rows and not any_visible:
            reasons = []
            if text:
                reasons.append(
                    f"CIF substring '{self._matched_filter_edit.text()}'"
                )
            if prob_cutoff > 0.0:
                reasons.append(f"p ≥ {prob_cutoff:.2f}")
            hint = f"<i>No structures match {' and '.join(reasons)}.</i>"
            self._matched_filter_empty_label = QLabel(hint)
            self._matched_filter_empty_label.setWordWrap(True)
            self._matched_struct_layout.addWidget(self._matched_filter_empty_label)
            self._sim_legend_filter_empty_label = QLabel(hint)
            self._sim_legend_filter_empty_label.setWordWrap(True)
            self._sim_legend_struct_layout.addWidget(
                self._sim_legend_filter_empty_label
            )

    def _on_matched_master_toggled(self, checked: bool) -> None:
        """Master toggles cascade to every per-structure row.

        Connected to BOTH master checkboxes (Display dock and the
        Expected-pattern twin legend); the first step mirrors the new
        state to the other master without re-entering this slot.

        Unchecking the master now also unchecks every structure
        checkbox; checking it back rechecks them all. The viewer's
        own master flag is updated either way so its hit-test gating
        stays in sync. Per-checkbox ``setChecked`` calls are blocked
        from re-emitting ``toggled`` so the structure-toggled slot
        doesn't interpret the cascade as a user-driven single-show.
        """
        for master in (
            self._matched_master_check, self._sim_legend_master_check,
        ):
            with QSignalBlocker(master):
                master.setChecked(checked)
        self.viewer.set_matched_master_visible(checked)
        for uid in list(self._matched_struct_checkboxes):
            self._set_matched_checkbox_twins(uid, checked)
            self.viewer.set_matched_structure_visible(uid, checked)

    def _set_matched_checkbox_twins(self, uid: str, checked: bool) -> None:
        """Set the ``uid`` structure checkbox in BOTH legends without
        re-emitting ``toggled`` (the caller updates the viewer state)."""
        for checks in (
            self._matched_struct_checkboxes,
            self._sim_legend_struct_checkboxes,
        ):
            chk = checks.get(uid)
            if chk is not None:
                with QSignalBlocker(chk):
                    chk.setChecked(checked)

    def _on_matched_style_changed(self, _index: int) -> None:
        """Switch the matched overlay between boxes and q-map markers."""
        self.viewer.set_matched_display_style(
            self._matched_style_combo.currentData()
        )

    def _on_matched_structure_toggled(self, uid: str, checked: bool) -> None:
        """Per-structure toggle (from either legend). Mirrors the state
        to the twin checkbox first, then promotes a ``check while master
        is off`` click into a "show only this one" view: every other
        structure is unchecked, the master is auto-ticked (without
        re-cascading), and only the freshly-checked structure ends up
        visible.
        """
        self._set_matched_checkbox_twins(uid, checked)
        self.viewer.set_matched_structure_visible(uid, checked)
        if not checked:
            return
        if self._matched_master_check.isChecked():
            return
        # Master was off → user wants to see this single structure.
        # Force the others off (both UI + viewer state) before flipping
        # the master ON, since the master toggle would otherwise
        # cascade and re-show every structure.
        for other_uid in list(self._matched_struct_checkboxes):
            if other_uid == uid:
                continue
            self._set_matched_checkbox_twins(other_uid, False)
            self.viewer.set_matched_structure_visible(other_uid, False)
        for master in (
            self._matched_master_check, self._sim_legend_master_check,
        ):
            with QSignalBlocker(master):
                master.setChecked(True)
        # blockSignals suppressed _on_matched_master_toggled, so call
        # the viewer's master flag directly.
        self.viewer.set_matched_master_visible(True)
