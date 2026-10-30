"""Live readout of the currently selected peak, plus commit/delete actions.

The buttons emit signals that MainWindow turns into ``PipelineCommand``s on
the existing worker thread. Add-to-detected only makes sense for manual
peaks (the others are already on file). Delete-peak is the inverse: only
file-resident peaks can be deleted from here — manual peaks use the
Delete shortcut.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QWidget,
)

from mlgidgui import peak_lists
from mlgidgui.fit import GaussianFit
from mlgidgui.main_window_constants import QUICK_TARGET_KEY
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.pipeline import is_mlgidbase_available
from mlgidgui import theme_tokens
from mlgidgui.widgets import (
    Card,
    DANGER as _DANGER,
    GAP,
    section_label,
    set_variant as _set_variant,
)

EMPTY = "—"

_SOURCE_LABEL = {
    "manual": "Manual",
    "detected": "Detected",
    "fitted": "Fitted",
    "matched": "Matched",
}


class ParameterPanel(Card):
    # Mode tokens for the Add-to-fitted dispatch. ``"scipy_1d"`` runs
    # the legacy 1D scipy + zero-fill code path that pre-dated the F-06
    # work; ``"pygidfit_2d"`` routes through ``manual_fit.fit_one_peak``
    # and matches what the pipeline ``run_fitting`` writes. Kept as
    # module-level string constants so callers can compare cleanly
    # (no enum import, no magic strings spread across files).
    FIT_MODE_1D = "scipy_1d"
    FIT_MODE_2D = "pygidfit_2d"

    # Confidence presets offered when adding a fresh detected peak from a
    # manual box: the "Score:" row becomes this dropdown while a manual
    # peak is selected. (label, score) — the score is written verbatim to
    # the new detected_peaks row. ``confidence_score()`` reads the choice.
    CONFIDENCE_LEVELS = (("High", 1.0), ("Medium", 0.5), ("Low", 0.1))

    # What a quick-select commit writes. Tokens, not indices, so the
    # stored preference survives a reordering of the dropdown.
    QUICK_TARGET_DETECTED = "detected"
    QUICK_TARGET_FITTED = "fitted"
    QUICK_TARGET_BOTH = "both"
    QUICK_TARGETS = (
        ("Detected", QUICK_TARGET_DETECTED),
        ("Fitted", QUICK_TARGET_FITTED),
        ("Both", QUICK_TARGET_BOTH),
    )

    addToDetectedRequested = Signal()
    addToFittedRequested = Signal()
    # Batch 2D-fit of the multi-selection. Only emitted when at least
    # one detected peak is selected and ring-storage is OFF (ring forces
    # 1D, which doesn't batch). The host loops ``_run_pygidfit_for_selection``
    # over the selected detected peaks inside a QProgressDialog.
    batchFit2DRequested = Signal()
    # Emits the new state of the "Save fitted as ring" checkbox so the host
    # can refresh the cyan fitted-preview overlay (rings render as a full
    # angular sweep) without waiting for the next selection change.
    saveAsRingChanged = Signal(bool)
    # Emits the active fit-mode token (``"scipy_1d"`` or
    # ``"pygidfit_2d"``) when the user flips the radio pair. The host
    # connects this to a preview-refresh shim so the dashed cyan
    # preview redraws immediately with the new mode's box widths
    # (radial / angular convention differs per mode — see
    # ``_update_fitted_preview`` in main_window).
    fitModeChanged = Signal(str)
    deletePeakRequested = Signal()
    # A new confidence score for the CURRENT SELECTION, from the Score
    # row's editor (a High/Medium/Low preset or the spin box). The panel
    # does not know how many peaks that is -- the host walks the
    # selection and writes each detected one.
    scoreEditRequested = Signal(float)
    # Quick-select mode on/off, and the token of the target dropdown
    # beside it. The host uses the first to arm the viewer and paint the
    # status-bar cell, and reads the second at commit time.
    quickSelectChanged = Signal(bool)
    quickTargetChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        # A Card rather than the QGroupBox this used to be: it is one
        # section of the Display dock among several, and a box drawn
        # around only this one made it read as a different kind of thing.
        # The Card supplies the layout, so the panel fills ``body_layout``
        # instead of installing one of its own.
        super().__init__("Selected peak", parent=parent)
        outer = self.body_layout
        outer.setSpacing(GAP)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        outer.addWidget(form_widget)

        self._source_label = self._make_value_label()
        self._radius_label = self._make_value_label()
        self._radius_width_label = self._make_value_label()
        self._angle_label = self._make_value_label()
        self._angle_width_label = self._make_value_label()
        self._score_label = self._make_value_label()
        self._type_label = self._make_value_label()
        self._id_label = self._make_value_label()
        # Color swatch shown next to the ID for matched selections —
        # matches the matched-overlay palette so the user can map the
        # readout back to the box on screen at a glance. Lives on the
        # ID row (not Source) because the structure ID is what the
        # colour identifies.
        self._source_swatch = QLabel()
        self._source_swatch.setFixedSize(14, 14)
        self._source_swatch.setVisible(False)
        self._id_row = QWidget()
        _id_h = QHBoxLayout(self._id_row)
        _id_h.setContentsMargins(0, 0, 0, 0)
        _id_h.setSpacing(6)
        _id_h.addWidget(self._id_label, 1)
        _id_h.addWidget(self._source_swatch)
        # Fit-derived rows. Populated from the profile viewer's last 1D
        # Gaussian fits (manual peaks: real refit; non-manual: synthetic
        # Gaussian honoring the unified ``2σ`` box convention shared by
        # the 1D and 2D Add-to-fitted code paths).
        self._fit_radius_label = self._make_value_label()
        self._fit_fwhm_r_label = self._make_value_label()
        self._fit_angle_label = self._make_value_label()
        self._fit_fwhm_a_label = self._make_value_label()
        self._fit_amp_label = self._make_value_label()

        # Source / Type / ID describe the peak itself and apply to every
        # kind, so they sit above the kind-specific Detected/Fitted blocks.
        form.addRow("Source:", self._source_label)
        form.addRow("Type:", self._type_label)
        form.addRow("ID:", self._id_row)
        # Track row indices for the section blocks so set_peak can hide
        # the irrelevant section wholesale (header + 4-5 value rows).
        # Only the section the peak's kind actually populates stays
        # visible; the other is removed from the layout flow rather
        # than just blanked.
        self._detected_section_label = self._make_section_label("Detected peak")
        # Registered extra peak lists, so a selection from one can be
        # read through its treat-as flavour and named in the header.
        self._peak_list_specs: list = []
        form.addRow(self._detected_section_label)
        self._row_detected_header = form.rowCount() - 1
        form.addRow("Radius:", self._radius_label)
        form.addRow("Δ radius:", self._radius_width_label)
        form.addRow("Angle:", self._angle_label)
        form.addRow("Δ angle:", self._angle_width_label)
        # mlgidDETECT confidence score. For detected/fitted/matched rows
        # the row shows the read-only ``_score_label``. For a manual box
        # (about to be committed via "Add to detected") the same row
        # swaps to a confidence dropdown so the user picks the score the
        # new detected peak will carry — a manual box has no model
        # provenance, so the choice is theirs. A QStackedWidget holds
        # both and shows exactly one (see set_peak).
        self._confidence_combo = QComboBox()
        for _lbl, _val in self.CONFIDENCE_LEVELS:
            self._confidence_combo.addItem(_lbl, _val)
        self._confidence_combo.setToolTip(
            "Confidence saved with a detected peak added from this manual "
            "box (High = 1.0, Medium = 0.5, Low = 0.1)."
        )
        # ...and for an EXISTING detected peak the same row becomes an
        # editor, because a model's confidence is exactly what a human
        # is there to correct when labelling validation data. Presets
        # write the same numbers the manual-commit dropdown above does,
        # so "Medium" means one thing in this app.
        self._score_editor = QWidget()
        score_edit_row = QHBoxLayout(self._score_editor)
        score_edit_row.setContentsMargins(0, 0, 0, 0)
        score_edit_row.setSpacing(4)
        self._score_preset_buttons: list[QPushButton] = []
        for label, value in self.CONFIDENCE_LEVELS:
            btn = QPushButton(label)
            btn.setToolTip(f"Set the score to {value:g}")
            btn.clicked.connect(
                lambda _=False, v=value: self._emit_score_edit(v)
            )
            score_edit_row.addWidget(btn)
            self._score_preset_buttons.append(btn)
        self._score_spin = QDoubleSpinBox()
        self._score_spin.setRange(0.0, 1.0)
        self._score_spin.setDecimals(3)
        self._score_spin.setSingleStep(0.05)
        self._score_spin.setToolTip(
            "Confidence stored with this peak. Editing it rewrites the "
            "score column and nothing else -- the box does not move."
        )
        self._score_spin.editingFinished.connect(
            lambda: self._emit_score_edit(self._score_spin.value())
        )
        score_edit_row.addWidget(self._score_spin)
        score_edit_row.addStretch(1)

        self._score_stack = QStackedWidget()
        self._score_stack.addWidget(self._score_label)       # page 0: readout
        self._score_stack.addWidget(self._confidence_combo)  # page 1: picker
        self._score_stack.addWidget(self._score_editor)      # page 2: editor
        form.addRow("Score:", self._score_stack)
        self._row_score = form.rowCount() - 1
        # How many peaks a preset click would relabel. Only shown when
        # that is more than one, so a batch write is never a surprise.
        self._selection_count = 1
        self._score_count_label = QLabel("")
        self._score_count_label.setProperty("status", "muted")
        form.addRow("", self._score_count_label)
        self._row_score_count = form.rowCount() - 1
        self._detected_rows = list(range(
            self._row_detected_header, form.rowCount()
        ))
        self._fitted_section_label = self._make_section_label("Fitted peak")
        form.addRow(self._fitted_section_label)
        self._row_fitted_header = form.rowCount() - 1
        # The row captions are swapped at selection time, because this
        # block reports two different things (see ``_set_fit_captions``):
        # the stored row for a saved fitted/matched peak, and a live
        # preview for a manual one. Held as widgets so the captions can
        # change; a plain ``addRow("text", field)`` would bury them.
        self._fit_captions = [QLabel() for _ in range(5)]
        for caption, field in zip(self._fit_captions, (
            self._fit_radius_label, self._fit_fwhm_r_label,
            self._fit_angle_label, self._fit_fwhm_a_label,
            self._fit_amp_label,
        )):
            form.addRow(caption, field)
        self._set_fit_captions(stored=False)
        self._fitted_rows = list(range(
            self._row_fitted_header, form.rowCount()
        ))
        # True while the Fitted block is showing the values stored in
        # the file. ``set_fits`` leaves the block alone in that state so
        # a live 1D profile fit cannot overwrite the saved numbers.
        self._showing_stored_fit = False
        self._form = form

        self._mlgidbase_available = is_mlgidbase_available()
        # Host-driven flag: True while the viewer has ≥2 fittable
        # peaks selected. Disables the 1D fit-mode radio because
        # batch fits are 2D-only.
        self._multi_select_active = False

        # "Add to detected" and "Add to fitted" are mutually exclusive choices
        # the user picks per manual peak — sit them side by side.
        self.btn_add_detected = QPushButton("Add to detected")
        self.btn_add_detected.clicked.connect(self.addToDetectedRequested)
        self.btn_add_fitted = QPushButton("Add to fitted")
        self.btn_add_fitted.setToolTip(
            "Append a row to fitted_peaks using the 1D Gaussian fit "
            "parameters from the radial / angular profile. "
            "Shortcut: Ctrl+Return."
        )
        self.btn_add_fitted.clicked.connect(self.addToFittedRequested)
        # Batch 2D fit: run pygidfit on every selected detected peak in
        # one go. Disabled unless multi-selection has at least one
        # detected peak AND ring storage is OFF (ring forces 1D).
        self.btn_fit_selected_2d = QPushButton("Fit selected (2D)")
        self.btn_fit_selected_2d.setToolTip(
            "Run pygidfit on every selected detected peak and append "
            "a fitted_peaks row for each. 2D only — 1D batch fits are "
            "not offered (the 1D projection doesn't generalise across "
            "peaks the way pygidfit's 2D model does)."
        )
        self.btn_fit_selected_2d.setEnabled(False)
        self.btn_fit_selected_2d.clicked.connect(self.batchFit2DRequested)
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(6)
        add_row.addWidget(self.btn_add_detected)
        add_row.addWidget(self.btn_add_fitted)
        add_row.addWidget(self.btn_fit_selected_2d)
        add_row_widget = QWidget()
        add_row_widget.setLayout(add_row)
        outer.addWidget(add_row_widget)

        # Quick select: while it is on, drawing the next box commits the
        # previous one, so a labelling run is drag-drag-drag instead of
        # drag-reach-for-the-button-drag. The target says what each
        # committed box becomes. Both live here rather than on the
        # viewer toolbar because every other commit-time flag does —
        # the score dropdown above, the fit-mode radios and the ring
        # checkbox below — and the host reads them all from this one
        # object. The host mirrors the mode into a status-bar cell so
        # it stays visible when this dock is tabbed away.
        quick_row = QHBoxLayout()
        quick_row.setContentsMargins(0, 0, 0, 0)
        quick_row.setSpacing(6)
        self.chk_quick_select = QCheckBox("Quick select")
        self.chk_quick_select.setToolTip(
            "Label without leaving the image: draw a box, and drawing "
            "the next one commits it as the peak kind chosen beside "
            "this box.\n\n"
            "A box drawn over the pending one replaces it instead — "
            "that is how you correct an attempt. The last box is "
            "committed when you click away, press Enter, change frame "
            "or turn the mode off; Esc still discards it."
        )
        self.chk_quick_select.toggled.connect(self.quickSelectChanged)
        self.chk_quick_select.toggled.connect(self._on_quick_select_toggled)
        self.combo_quick_target = QComboBox()
        for label, token in self.QUICK_TARGETS:
            self.combo_quick_target.addItem(label, token)
        self.combo_quick_target.setToolTip(
            "What each auto-committed box becomes.\n\n"
            "Detected writes the box as you drew it, at the confidence "
            "above — quick, and it cannot fail. Fitted runs the 2D fit "
            "on it. Both writes a detected row and fits it.\n\n"
            "With 'One fitted peak per detected peak' on (Settings), "
            "Fitted already writes the detected partner, so Both is "
            "the same thing."
        )
        self.combo_quick_target.currentIndexChanged.connect(
            lambda _i: self.quickTargetChanged.emit(self.quick_select_target())
        )
        quick_row.addWidget(self.chk_quick_select)
        quick_row.addWidget(self.combo_quick_target, 1)
        quick_row_widget = QWidget()
        quick_row_widget.setLayout(quick_row)
        outer.addWidget(quick_row_widget)
        self._restore_quick_target()
        self._on_quick_select_toggled(False)

        # Fit-mode selector for Add-to-fitted. Two radios, mutually
        # exclusive via a QButtonGroup. Default is 2D pygidfit (matches
        # what the pipeline run_fitting writes). 1D scipy is the legacy
        # mode that pre-dated the F-06 work — kept available because
        # narrow / off-shape peaks sometimes look better with scipy's
        # quick 1D model than with the full 2D Gaussian. Greyed out
        # when "Save fitted as ring" is on because pygidfit's segment
        # model can't fit a ring cleanly; the ring storage convention
        # bypasses both fit paths anyway.
        self.rb_fit_2d = QRadioButton("2D fit (pygidfit)")
        self.rb_fit_2d.setToolTip(
            "Save through pygidfit's 2D Gaussian fit — same model the "
            "pipeline 'run_fitting' uses. Stores real A/B/C/theta shape "
            "coefficients on the row."
        )
        self.rb_fit_1d = QRadioButton("1D fit (scipy)")
        self.rb_fit_1d.setToolTip(
            "Save through the legacy 1D scipy Gaussian fit on the radial "
            "and angular profile slices. 2D shape coefficients are "
            "zero-filled. Useful when the 2D fit doesn't converge on "
            "the active box."
        )
        self.rb_fit_2d.setChecked(True)
        self._fit_mode_group = QButtonGroup(self)
        self._fit_mode_group.setExclusive(True)
        self._fit_mode_group.addButton(self.rb_fit_2d)
        self._fit_mode_group.addButton(self.rb_fit_1d)
        # Either toggled signal fires for both buttons in an exclusive
        # group — fan in to one emit per user click.
        self._fit_mode_group.buttonToggled.connect(
            lambda *_: self.fitModeChanged.emit(self.fit_mode())
        )
        fit_mode_row = QWidget()
        fit_mode_h = QHBoxLayout(fit_mode_row)
        fit_mode_h.setContentsMargins(0, 0, 0, 0)
        fit_mode_h.setSpacing(12)
        fit_mode_h.addWidget(self.rb_fit_2d)
        fit_mode_h.addWidget(self.rb_fit_1d)
        fit_mode_h.addStretch(1)
        outer.addWidget(fit_mode_row)

        # Ring/segment toggle — applies to whichever box "Add to fitted"
        # would commit. State is sticky: once the user (un)checks it the
        # value persists across selection changes; only Add-to-fitted
        # itself resets it (back to unchecked) after a successful commit.
        # When checked, the saved row uses the canonical ring convention
        # (angle = 45°, angle_width = ∞), the angular profile fit is
        # skipped, and the cyan preview renders as a full-sweep ring.
        # Ring also forces the legacy 1D code path — pygidfit's segment
        # model has no ring analogue — so the fit-mode radios above are
        # greyed out while the ring box is checked (see
        # ``_sync_fit_mode_enabled``).
        self.chk_save_as_ring = QCheckBox("Save fitted as ring")
        self.chk_save_as_ring.toggled.connect(self.saveAsRingChanged)
        self.chk_save_as_ring.toggled.connect(self._sync_fit_mode_enabled)
        outer.addWidget(self.chk_save_as_ring)

        # Run fitting / Run matching used to live here too, but they're
        # already exposed in the Pipeline dock with their full kwarg
        # surface — duplicating them in the per-peak panel just confused
        # the user about what each call would do. Removed.
        self.btn_delete_peak = _set_variant(QPushButton("Delete peak"), _DANGER)
        self.btn_delete_peak.clicked.connect(self.deletePeakRequested)
        outer.addWidget(self.btn_delete_peak)

        if not self._mlgidbase_available:
            note = QLabel("<i>mlgidbase not installed — actions disabled.</i>")
            note.setWordWrap(True)
            outer.addWidget(note)

        self.set_peak(None)

    @staticmethod
    def _make_value_label() -> QLabel:
        lbl = QLabel(EMPTY)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lbl

    @staticmethod
    def _make_section_label(text: str) -> QLabel:
        """A heading for one block of form rows.

        Weight and colour come from the skin (``role="section"``) rather
        than a bold ``QFont`` set here, so it follows a theme flip.
        """
        lbl = section_label(text)
        lbl.setContentsMargins(0, 4, 0, 0)
        return lbl

    # Both selectionChanged and peakGeometryChanged emit SelectedPeak | None.
    # One slot handles both — when peak is None (deselect) we blank every
    # row; otherwise we populate only the section(s) that apply to the
    # peak's kind:
    #
    #   manual           → Detected + Fitted (user is choosing what to commit)
    #   detected         → Detected only
    #   fitted / matched → Fitted only
    #
    # The opposite section stays blank so the same parameters never appear
    # twice for a single peak.
    @Slot(object)
    def set_peak(self, peak: SelectedPeak | None) -> None:
        self._last_peak = peak
        self._update_actions_enabled(peak)
        if peak is None:
            for lbl in (
                self._source_label,
                self._type_label,
                self._id_label,
                self._radius_label,
                self._radius_width_label,
                self._angle_label,
                self._angle_width_label,
                self._score_label,
                self._fit_radius_label,
                self._fit_fwhm_r_label,
                self._fit_angle_label,
                self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)
            self._source_swatch.setVisible(False)
            # No selection → collapse both kind-specific sections so
            # the panel doesn't show stale section headers above empty
            # rows.
            for r in self._detected_rows:
                self._form.setRowVisible(r, False)
            for r in self._fitted_rows:
                self._form.setRowVisible(r, False)
            self._showing_stored_fit = False
            return
        source = _SOURCE_LABEL.get(peak.kind, peak.kind.capitalize())
        if peak.kind == "matched":
            # Prefer the human-readable structure label (CIF + (hkl) +
            # probability) when the viewer attached one; fall back to the
            # raw structure_uid only if the label wasn't populated.
            tag = peak.structure_label or peak.structure_uid
            if tag:
                source = f"{source} ({tag})"
            if peak.structure_color:
                # The fill is data (the matched structure's colour);
                # only the border follows the theme.
                self._source_swatch.setStyleSheet(
                    f"background-color: {peak.structure_color};"
                    f" border: 1px solid {theme_tokens.color('border')};"
                )
                self._source_swatch.setVisible(True)
            else:
                self._source_swatch.setVisible(False)
        else:
            self._source_swatch.setVisible(False)
        self._source_label.setText(source)
        self._type_label.setText("Ring" if peak.is_ring else "Segment")
        self._id_label.setText(str(peak.peak_id))

        # Ring/segment toggle is sticky across selection changes (set by
        # the user, reset only by Add-to-fitted) — see chk_save_as_ring.

        # Show only the section(s) relevant to this peak's kind:
        #   manual           → Detected + Fitted (user is choosing what to commit)
        #   detected         → Detected only
        #   fitted / matched → Fitted only
        # A registered extra peak list reads through the flavour it was
        # registered as, so a table the user calls "detected" shows the
        # radius/angle/score block and one they call "fitted" shows the
        # centre/FWHM/amplitude block. Its own kind never matches the
        # built-in tuples, which is what keeps it out of everything else.
        effective_kind = peak.kind
        if peak_lists.is_list_kind(peak.kind) and self._peak_list_specs:
            spec = peak_lists.spec_for_kind(self._peak_list_specs, peak.kind)
            if spec is not None:
                effective_kind = spec.treat_as
        show_detected = effective_kind in ("manual", "detected")
        show_fitted = effective_kind in ("manual", "fitted", "matched")
        # Name the source in the header. Two layers can look identical
        # on the image, so "Detected peak" over a Br_peaks box would be
        # a straight lie about which table is being read.
        list_spec = (
            peak_lists.spec_for_kind(self._peak_list_specs, peak.kind)
            if peak_lists.is_list_kind(peak.kind)
            else None
        )
        self._detected_section_label.setText(
            list_spec.display_label if list_spec is not None
            else "Detected peak"
        )
        # "(preview)" only for a manual peak, where the block forecasts
        # what Add-to-fitted would write and there is no stored row yet.
        # Everything else reads its own saved values, so the plain
        # heading is now the truth.
        self._fitted_section_label.setText(
            list_spec.display_label if list_spec is not None
            else ("Fitted peak (preview)" if effective_kind == "manual"
                  else "Fitted peak")
        )
        for r in self._detected_rows:
            self._form.setRowVisible(r, show_detected)
        for r in self._fitted_rows:
            self._form.setRowVisible(r, show_fitted)

        # A saved fitted / matched row reports what is IN THE FILE, in
        # the same quantities and formats the Peaks dock prints, so the
        # two readouts of one peak agree digit for digit. Only a manual
        # peak keeps the live-fit preview, where a forecast is the whole
        # point (see ``set_fits``).
        self._showing_stored_fit = show_fitted and effective_kind != "manual"
        self._set_fit_captions(stored=self._showing_stored_fit)
        if self._showing_stored_fit:
            self._show_stored_fit(peak)

        if show_detected:
            self._radius_label.setText(f"{peak.radius:.3f} Å⁻¹")
            self._radius_width_label.setText(f"{peak.radius_width:.3f} Å⁻¹")
            self._angle_label.setText(f"{peak.angle:.2f}°")
            self._angle_width_label.setText(f"{peak.angle_width:.2f}°")
        # Score row. For a manual box it is the confidence dropdown (the
        # score the new detected peak gets on "Add to detected"); for an
        # existing detected peak it is the read-only model score. Hidden
        # for fitted / matched (their Detected block is collapsed).
        # Three pages, one row:
        #   manual   -> the confidence picker (score the commit will write)
        #   detected -> the EDITOR (correcting a model's call is the point)
        #   anything else with a score -> a read-only readout
        is_manual = peak.kind == "manual"
        has_score = peak.score is not None and not is_manual
        # ``effective_kind``, so a registered list the user chose to
        # treat as detected is editable exactly like the built-in layer
        # -- that flavour is a statement about what the table IS, and
        # relabelling is the reason such a list gets registered. The
        # write still goes to that list's own dataset and nowhere else.
        # ``has_score`` keeps the editor off a table with no score
        # column: ``update_peak_row`` writes only fields the dtype has,
        # so offering it there would drop the write silently.
        is_editable_score = (
            effective_kind == "detected" and show_detected
            and not is_manual and has_score
        )
        self._form.setRowVisible(
            self._row_score, show_detected and (is_manual or has_score)
        )
        if show_detected and is_manual:
            page = self._confidence_combo
        elif is_editable_score:
            page = self._score_editor
        else:
            page = self._score_label
        self._score_stack.setCurrentWidget(page)
        if is_editable_score:
            with QSignalBlocker(self._score_spin):
                self._score_spin.setValue(float(peak.score or 0.0))
        if has_score and show_detected:
            self._score_label.setText(f"{peak.score:.3f}")
        else:
            self._score_label.setText(EMPTY)
        # Re-apply the count line against the page that is now showing.
        self.set_selection_count(getattr(self, "_selection_count", 1))
        # Detected rows are hidden when ``show_detected`` is False, so
        # we don't blank them — set_fits / next show will refresh.

        # If the new selection has no Fitted section, blank those rows
        # so a stale value from the previous selection can't linger if
        # the section is later re-shown without set_fits running.
        if not show_fitted:
            for lbl in (
                self._fit_radius_label, self._fit_fwhm_r_label,
                self._fit_angle_label, self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)

    # Captions for the Fitted block. The stored set names exactly what
    # the Peaks dock's Fitted tab names, including the ``2sigma`` width
    # convention, because the two are showing one row and any
    # difference in wording reads as a difference in value.
    _FIT_CAPTIONS_STORED = ("r:", "Δr:", "a:", "Δa:", "Amplitude:")
    _FIT_CAPTIONS_PREVIEW = (
        "Center r:", "FWHM r:", "Center a:", "FWHM a:", "Amplitude:",
    )

    # Same wording as the Peaks dock header tooltips, so the width
    # convention is discoverable from whichever dock the user is in.
    _FIT_TIPS_STORED = (
        "Radius (Å⁻¹), as stored in the file",
        "Radial width = 2σ_r (Å⁻¹). Same convention as pygidfit / "
        "pipeline; FWHM_r ≈ 1.177 × Δr.",
        "Angle (°), as stored in the file",
        "Angular width = 2σ_a (°). Same convention as pygidfit / "
        "pipeline; FWHM_a ≈ 1.177 × Δa.",
        "Peak amplitude (2D-Gaussian height), as stored in the file",
    )
    _FIT_TIPS_PREVIEW = (
        "Radial centre the next Add-to-fitted would save",
        "Radial FWHM of the preview fit (stored as 2σ_r = FWHM / 1.177)",
        "Angular centre the next Add-to-fitted would save",
        "Angular FWHM of the preview fit (stored as 2σ_a = FWHM / 1.177)",
        "Amplitude of the preview fit",
    )

    def _set_fit_captions(self, *, stored: bool) -> None:
        captions = (
            self._FIT_CAPTIONS_STORED if stored
            else self._FIT_CAPTIONS_PREVIEW
        )
        tips = self._FIT_TIPS_STORED if stored else self._FIT_TIPS_PREVIEW
        fields = (
            self._fit_radius_label, self._fit_fwhm_r_label,
            self._fit_angle_label, self._fit_fwhm_a_label,
            self._fit_amp_label,
        )
        for widget, field, text, tip in zip(
            self._fit_captions, fields, captions, tips,
        ):
            widget.setText(text)
            widget.setToolTip(tip)
            field.setToolTip(tip)

    def _show_stored_fit(self, peak: SelectedPeak) -> None:
        """Fill the Fitted block from the peak's own stored row.

        Formats mirror ``peaks_table_panel._fill_fitted`` field for
        field (``.3f`` radius and width, ``.2f`` angles, ``.3g``
        amplitude) so selecting a peak and finding it in the Peaks dock
        gives identical digits. Widths are printed as stored, pygidfit's
        ``2sigma``, NOT converted to FWHM: a conversion here is what
        made the same peak read 18% wider in one dock than the other.

        A ring's non-finite ``angle_width`` prints as ``inf``, which is
        what the Peaks dock and the Detected block above both already
        print for the same value: matching means matching. Only
        ``amplitude`` can be genuinely absent (``None`` on a hand-built
        selection), and that blanks its row.
        """
        self._fit_radius_label.setText(f"{peak.radius:.3f} Å⁻¹")
        self._fit_fwhm_r_label.setText(f"{peak.radius_width:.3f} Å⁻¹")
        self._fit_angle_label.setText(f"{peak.angle:.2f}°")
        self._fit_fwhm_a_label.setText(f"{peak.angle_width:.2f}°")
        if peak.amplitude is None or not math.isfinite(peak.amplitude):
            self._fit_amp_label.setText(EMPTY)
        else:
            self._fit_amp_label.setText(f"{peak.amplitude:.3g}")

    @Slot(object, object)
    def set_fits(
        self, rfit: GaussianFit | None, afit: GaussianFit | None,
    ) -> None:
        """Update the Fitted-peak rows from the profile viewer's 1D fits.

        Ignored while the block is showing a saved row
        (``_showing_stored_fit``): the profile viewer re-fits the live
        data for every selection, and letting that land here is what
        made a stored fitted peak read differently in the Display dock
        than in the Peaks dock. The profile curve still comes from the
        live fit; only the numbers are pinned to the file.

        Skipped (and blanked) for detected selections — those don't have a
        meaningful fitted-peak readout. Either fit may be ``None`` (no
        convergence, ring with inf width, no selection) → blank that row.
        """
        if self._showing_stored_fit:
            return
        peak = self._last_peak
        if peak is not None and peak.kind == "detected":
            for lbl in (
                self._fit_radius_label, self._fit_fwhm_r_label,
                self._fit_angle_label, self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)
            return
        if rfit is not None:
            self._fit_radius_label.setText(f"{rfit.center:.3f} Å⁻¹")
            self._fit_fwhm_r_label.setText(f"{rfit.fwhm:.3f} Å⁻¹")
            self._fit_amp_label.setText(f"{rfit.amplitude:.3g}")
        else:
            self._fit_radius_label.setText(EMPTY)
            self._fit_fwhm_r_label.setText(EMPTY)
            self._fit_amp_label.setText(EMPTY)
        if afit is not None:
            self._fit_angle_label.setText(f"{afit.center:.2f}°")
            self._fit_fwhm_a_label.setText(f"{afit.fwhm:.2f}°")
        else:
            self._fit_angle_label.setText(EMPTY)
            self._fit_fwhm_a_label.setText(EMPTY)

    def quick_select_enabled(self) -> bool:
        """Whether drawing the next box commits the previous one."""
        return self.chk_quick_select.isChecked()

    def quick_select_target(self) -> str:
        """Token of the kind a quick-select commit writes."""
        data = self.combo_quick_target.currentData()
        return str(data) if data else self.QUICK_TARGET_DETECTED

    def set_quick_select(self, enabled: bool) -> None:
        """Turn the mode on/off from outside (the host does this when a
        session closes, so a mode cannot outlive the file it labels)."""
        self.chk_quick_select.setChecked(bool(enabled))

    def _on_quick_select_toggled(self, enabled: bool) -> None:
        # The target only means anything while the mode is on, and a
        # live dropdown next to an unchecked box reads as a setting that
        # is doing something.
        self.combo_quick_target.setEnabled(bool(enabled))

    def _restore_quick_target(self) -> None:
        """Load the stored target.

        The *target* persists across restarts — a labelling run outlives
        one session and re-picking it every time is friction. The
        **mode** deliberately does not: a window that came back in a
        mode where drawing writes to the file would be a nasty
        surprise.
        """
        stored = str(QSettings().value(QUICK_TARGET_KEY, "") or "")
        idx = self.combo_quick_target.findData(stored)
        if idx >= 0:
            self.combo_quick_target.setCurrentIndex(idx)
        self.combo_quick_target.currentIndexChanged.connect(
            lambda _i: QSettings().setValue(
                QUICK_TARGET_KEY, self.quick_select_target()
            )
        )

    def _emit_score_edit(self, value: float) -> None:
        """Ask the host to write ``value`` to every detected peak selected.

        Guarded on the editor actually being the visible page: the spin
        box's ``editingFinished`` also fires when focus leaves during a
        selection change, and without this a stale value would be
        written to whatever was selected next.
        """
        if self._score_stack.currentWidget() is not self._score_editor:
            return
        value = max(0.0, min(1.0, float(value)))
        with QSignalBlocker(self._score_spin):
            self._score_spin.setValue(value)
        self.scoreEditRequested.emit(value)

    def set_peak_list_specs(self, specs) -> None:
        """The registered extra lists, so a selection from one can be read."""
        self._peak_list_specs = list(specs)

    def set_selection_count(self, count: int) -> None:
        """How many peaks a score edit would reach (0/1 hides the line)."""
        self._selection_count = int(count)
        self._score_count_label.setText(
            f"{count} peaks selected" if count > 1 else ""
        )
        self._form.setRowVisible(
            self._row_score_count,
            count > 1 and self._score_stack.currentWidget() is self._score_editor,
        )

    def save_as_ring(self) -> bool:
        """Whether the next Add-to-fitted should commit a ring row."""
        return self.chk_save_as_ring.isChecked()

    def confidence_score(self) -> float:
        """Score a freshly added detected peak should carry, per the
        Score-row confidence dropdown (High = 1.0 / Medium = 0.5 /
        Low = 0.1). Falls back to 1.0 if the combo is somehow empty."""
        data = self._confidence_combo.currentData()
        return float(data) if data is not None else 1.0

    def fit_mode(self) -> str:
        """Return the active Add-to-fitted dispatch mode.

        ``FIT_MODE_1D`` (``"scipy_1d"``) → legacy 1D scipy + zero-fill
        path. ``FIT_MODE_2D`` (``"pygidfit_2d"``) → pygidfit's 2D fit
        via ``mlgidgui.manual_fit.fit_one_peak``. Returns the 1D token
        whenever the ring toggle is on, since pygidfit's segment model
        can't fit a ring cleanly — the host should respect this and
        skip the 2D dispatch even if the radio is on.
        """
        if self.chk_save_as_ring.isChecked():
            return self.FIT_MODE_1D
        return (
            self.FIT_MODE_2D if self.rb_fit_2d.isChecked() else self.FIT_MODE_1D
        )

    def _sync_fit_mode_enabled(self, ring_checked: bool) -> None:
        """Grey out the fit-mode radios for the active constraints.

        Two gates compose:

        * **Ring storage on**: both radios disabled. pygidfit doesn't
          model rings; the ring code path uses the legacy 1D machinery
          regardless, so neither radio's choice can change the result.
        * **Multi-select active**: only the 1D radio is disabled.
          Batch fits are 2D-only (per user constraint: "should not be
          possible for 1D fits since that does not make sense
          physically"), so the 1D option would mislead about what
          'Fit selected (2D)' will do.
        """
        ring_disabled = bool(ring_checked)
        multi_disabled = self._multi_select_active
        self.rb_fit_2d.setEnabled(not ring_disabled)
        self.rb_fit_1d.setEnabled(not ring_disabled and not multi_disabled)

    def set_multi_select_active(self, active: bool) -> None:
        """Toggle the multi-select gate on the 1D fit-mode radio.

        Driven by the host from ``selectionsChanged``: when ≥2
        fittable peaks are selected, the 1D radio greys out. Pure
        UI state; ``fit_mode()`` still reports whatever the radio
        is checked on (the host's batch-fit handler ignores it and
        always runs 2D).
        """
        if self._multi_select_active == bool(active):
            return
        self._multi_select_active = bool(active)
        self._sync_fit_mode_enabled(self.chk_save_as_ring.isChecked())

    def reset_save_as_ring(self) -> None:
        """Force the ring toggle back to unchecked.

        Called by MainWindow after a successful Add-to-fitted commit so
        the user has to opt back in for each new ring row. Emits
        saveAsRingChanged via the standard toggled connection so the
        cyan preview / angular fit refresh follow.
        """
        if self.chk_save_as_ring.isChecked():
            self.chk_save_as_ring.setChecked(False)

    def set_batch_fit_enabled(self, enabled: bool) -> None:
        """Enable / disable the 'Fit selected (2D)' button.

        Driven by the host from ``selectionsChanged`` /
        ``saveAsRingChanged``. The host computes the predicate
        (≥1 detected selected AND not save-as-ring) since the panel
        has no view of the multi-selection.
        """
        self.btn_fit_selected_2d.setEnabled(enabled)

    def set_fit_button_visibility(
        self, *, add_fitted: bool, fit_selected_2d: bool,
    ) -> None:
        """Show one of {Add to fitted, Fit selected (2D)} at a time.

        Mutually exclusive visibility (mirrors the host's view of the
        multi-selection: a single fittable peak shows Add to fitted;
        ≥2 detected peaks show Fit selected (2D); neither when no
        fittable selection exists). 'Add to detected' stays in
        place and is only enable-toggled by ``_update_actions_enabled``.
        Driven by the host on every ``selectionsChanged`` /
        ``saveAsRingChanged`` tick.
        """
        self.btn_add_fitted.setVisible(add_fitted)
        self.btn_fit_selected_2d.setVisible(fit_selected_2d)

    def set_busy(self, busy: bool) -> None:
        """Disable buttons while a pipeline run is in flight."""
        if not self._mlgidbase_available:
            return
        if busy:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_fit_selected_2d,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
            self.chk_save_as_ring.setEnabled(False)
        else:
            self._update_actions_enabled(self._current_peak())

    def _update_actions_enabled(self, peak: SelectedPeak | None) -> None:
        if not self._mlgidbase_available:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
            self.chk_save_as_ring.setEnabled(False)
            return
        # Add-to-detected only makes sense for manual peaks (committing the
        # in-memory candidate). Add-to-fitted accepts manual *and* detected
        # selections — a detected box is the natural input for a fit, and
        # this lets the user promote a detected row into fitted_peaks
        # using the live 1D Gaussian fit. Delete-peak only applies to
        # non-manual peaks (manual uses the Delete shortcut).
        is_manual = peak is not None and peak.kind == "manual"
        is_addable_to_fitted = peak is not None and peak.kind in ("manual", "detected")
        # A registered extra peak list is a display layer: mlgidGUI draws
        # it and lets the user nudge a box, but deleting rows out of a
        # table it does not own is not on offer. This is the ONE place
        # the enable rule is "anything but manual" rather than an
        # explicit list, so it is the one that has to say no.
        is_file_peak = (
            peak is not None
            and peak.kind != "manual"
            and not peak_lists.is_list_kind(peak.kind)
        )
        self.btn_add_detected.setEnabled(is_manual)
        self.btn_add_fitted.setEnabled(is_addable_to_fitted)
        self.chk_save_as_ring.setEnabled(is_addable_to_fitted)
        self.btn_delete_peak.setEnabled(is_file_peak)

    def _current_peak(self) -> SelectedPeak | None:
        # Re-derive the peak the panel is currently showing (if any) so we can
        # restore button state after a busy spell.
        return getattr(self, "_last_peak", None)
