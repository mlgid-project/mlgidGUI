"""Expected-pattern overlay controls: CIF/orientation-mode combos, hkl validation, predicted-peak fills with rollback, the all-frames matched sweep and pattern application.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)
from mlgidgui import file_model, pipeline, simulation_pattern
from mlgidgui.peak_link import link_enabled
from mlgidgui.pipeline import PipelineCommand

import logging

logger = logging.getLogger(__name__)


class SimOverlayMixin:
    # -- "Expected pattern" simulation overlay (Display-dock section) --

    def _on_cif_cache_changed(self, _obj: object) -> None:
        """The Pipeline panel's parsed-CIF cache changed (parse result,
        input edit, session swap). Refresh the section and force a
        pattern re-apply — even an unchanged (CIF, hkl) selection must
        re-extract against the new cache object."""
        self._sim_applied = None
        self._update_sim_section_state()
        self._populate_sim_combos()

    def _update_sim_section_state(self) -> None:
        """Enable the Expected-pattern section only when it can work:
        matching backend installed, a CifPattern parsed, and a NeXus
        session active."""
        panel = getattr(self, "pipeline_panel", None)
        if panel is None:
            return
        if not pipeline.is_mlgidbase_available():
            enabled = False
            tip = "Matching backend (mlgidbase) is not installed."
        elif panel.cached_cif_pattern() is None:
            enabled = False
            tip = "Parse CIFs in the Pipeline tab first."
        elif self.session is None or self.session.kind == "raw":
            enabled = False
            tip = "Open a NeXus file first."
        else:
            enabled = True
            tip = ""
        self._sim_section_container.setEnabled(enabled)
        self._sim_section_container.setToolTip(tip)

    def _sim_matched_structures(self) -> list:
        """Current frame's matched structures (empty when none)."""
        try:
            return list(
                self.viewer.matched_structures(self.viewer.current_frame)
                or []
            )
        except (AttributeError, KeyError):
            return []

    @staticmethod
    def _combo_data_index(combo: QComboBox, value) -> int:
        """Index of ``value`` among the combo's userData, or -1.

        ``QComboBox.findData`` compares wrapped Python objects by
        identity, so a freshly-built tuple never matches the stored
        one — compare by equality instead.
        """
        if value is None:
            return -1
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                return i
        return -1

    def _populate_sim_combos(self) -> None:
        """Rebuild the CIF combo from the parse cache (preserving the
        selection by userData), then the orientation combo."""
        panel = getattr(self, "pipeline_panel", None)
        cache = panel.cached_cif_pattern() if panel is not None else None
        prev = self._sim_cif_combo.currentData()
        matched_stems = {
            simulation_pattern.cif_stem(s.cif)
            for s in self._sim_matched_structures()
        }
        with QSignalBlocker(self._sim_cif_combo):
            self._sim_cif_combo.clear()
            if cache is not None:
                for name in getattr(cache, "cifs", []) or []:
                    stem = simulation_pattern.cif_stem(name)
                    label = stem + (
                        " — matched" if stem in matched_stems else ""
                    )
                    self._sim_cif_combo.addItem(label, userData=stem)
                idx = self._combo_data_index(self._sim_cif_combo, prev)
                if idx >= 0:
                    self._sim_cif_combo.setCurrentIndex(idx)
        self._populate_sim_orientations()

    def _populate_sim_orientations(self) -> None:
        """Rebuild the matched-orientation combo for the selected CIF
        (the current frame's matched (cif, hkl) combos, labelled with
        their probability), steer the auto mode default (matched when
        matches exist, random/powder otherwise), re-validate a typed
        hkl against the selected CIF and end by (re-)applying the
        pattern — a no-op when the effective selection is unchanged."""
        panel = getattr(self, "pipeline_panel", None)
        cache = panel.cached_cif_pattern() if panel is not None else None
        prev = self._sim_matched_combo.currentData()
        with QSignalBlocker(self._sim_matched_combo):
            self._sim_matched_combo.clear()
            stem = self._sim_cif_combo.currentData()
            if cache is not None and stem is not None:
                seen: set[tuple] = set()
                for s in self._sim_matched_structures():
                    if simulation_pattern.cif_stem(s.cif) != stem:
                        continue
                    hkl = (int(s.h), int(s.k), int(s.l))
                    if hkl in seen:
                        continue
                    seen.add(hkl)
                    if hkl == simulation_pattern.POWDER_HKL:
                        base = "Powder (rings)"
                    else:
                        base = f"({hkl[0]} {hkl[1]} {hkl[2]})"
                    self._sim_matched_combo.addItem(
                        f"{base} — p={float(s.probability):.2f}",
                        userData=hkl,
                    )
                idx = self._combo_data_index(self._sim_matched_combo, prev)
                if idx >= 0:
                    self._sim_matched_combo.setCurrentIndex(idx)
        if self._sim_mode_auto:
            want = (
                "matched" if self._sim_matched_combo.count() else "random"
            )
            with QSignalBlocker(self._sim_orient_mode):
                self._sim_orient_mode.setCurrentIndex(
                    self._combo_data_index(self._sim_orient_mode, want)
                )
        self._validate_sim_hkl(announce=False)
        self._update_sim_orient_visibility()
        self._apply_sim_pattern()

    def _sim_selected_hkl(self) -> tuple | None:
        """Effective orientation of the current mode: the matched
        combo's choice, ``POWDER_HKL`` for random, or the validated
        user-typed triple. None = nothing to render (matched mode with
        no match on this frame / empty or invalid typed hkl)."""
        mode = self._sim_orient_mode.currentData()
        if mode == "random":
            return simulation_pattern.POWDER_HKL
        if mode == "matched":
            data = self._sim_matched_combo.currentData()
            return tuple(int(v) for v in data) if data is not None else None
        return self._sim_user_hkl

    def _set_sim_hint(self, text: str, error: bool = True) -> None:
        self._sim_orient_hint.setText(text)
        self._sim_orient_hint.setProperty("status", "error" if error else "")
        style = self._sim_orient_hint.style()
        style.unpolish(self._sim_orient_hint)
        style.polish(self._sim_orient_hint)
        self._sim_orient_hint.setVisible(bool(text))

    def _update_sim_orient_visibility(self) -> None:
        """Show the mode's input widget and the state hint: the matched
        combo in matched mode (with a hint when the frame has no
        matched orientation), the hkl field in user mode (its hint is
        owned by ``_validate_sim_hkl``), nothing extra for random."""
        mode = self._sim_orient_mode.currentData()
        self._sim_matched_combo.setVisible(mode == "matched")
        self._sim_hkl_edit.setVisible(mode == "user")
        if mode == "matched" and self._sim_matched_combo.count() == 0:
            self._set_sim_hint(
                "No matched orientation for this structure on the "
                "current frame — run Matching, change frame, or pick "
                "another mode.", error=False,
            )
        elif mode != "user":
            self._set_sim_hint("")

    def _validate_sim_hkl(self, announce: bool = True) -> None:
        """Parse + validate the typed hkl against the selected CIF's
        precomputed orientations; sets ``_sim_user_hkl`` and the hint.
        ``announce=False`` (repopulation paths) keeps the status bar
        quiet so frame changes never spam it."""
        self._sim_user_hkl = None
        if self._sim_orient_mode.currentData() != "user":
            return
        text = self._sim_hkl_edit.text().strip()
        if not text:
            self._set_sim_hint("")
            return
        panel = getattr(self, "pipeline_panel", None)
        cache = panel.cached_cif_pattern() if panel is not None else None
        stem = self._sim_cif_combo.currentData()
        if cache is None or stem is None:
            self._set_sim_hint("")
            return
        try:
            hkl = simulation_pattern.parse_hkl(text)
        except ValueError as exc:
            self._set_sim_hint(f"Invalid hkl: {exc}")
            if announce:
                self.statusBar().showMessage(
                    f"Expected pattern: invalid hkl — {exc}", 6000
                )
            return
        ci = simulation_pattern.cif_index(cache, stem)
        resolved = (
            simulation_pattern.resolve_orientation(cache, ci, hkl)
            if ci is not None else None
        )
        if resolved is None:
            try:
                n = len(simulation_pattern.list_orientations(cache, ci))
            except (AttributeError, IndexError, TypeError):
                n = 0
            msg = (
                f"({hkl[0]} {hkl[1]} {hkl[2]}) is not a simulated "
                f"orientation of {stem} ({n} precomputed)."
            )
            self._set_sim_hint(msg)
            if announce:
                self.statusBar().showMessage(
                    f"Expected pattern: {msg}", 6000
                )
            return
        self._sim_user_hkl = resolved
        if resolved != hkl:
            self._set_sim_hint(
                f"Rendering the equivalent simulated orientation "
                f"({resolved[0]} {resolved[1]} {resolved[2]}).",
                error=False,
            )
        else:
            self._set_sim_hint("")

    def _refresh_sim_matched_entries(self, _frame: int, _structures: list) -> None:
        """The frame's matched set changed (frame change, re-match,
        fresh load) — refresh the '— matched' labels and ordering.
        Selection is preserved by userData and ``_apply_sim_pattern``
        no-ops on an unchanged choice, so plain frame changes never
        drop the overlay or its reflection selection."""
        # Frame changes alter the Add button's dataset guard even when
        # no CIF cache exists yet.
        self._update_sim_add_button()
        panel = getattr(self, "pipeline_panel", None)
        if panel is None or panel.cached_cif_pattern() is None:
            return
        self._populate_sim_combos()

    def _on_sim_master_toggled(self, checked: bool) -> None:
        self.viewer.set_simulation_visible(checked)
        # Extract lazily on first enable; drop the pattern when the
        # section is unticked so nothing stale lingers invisibly.
        self._apply_sim_pattern()

    def _on_sim_cif_changed(self, _index: int) -> None:
        self._populate_sim_orientations()

    def _on_sim_orient_mode_changed(self, _index: int) -> None:
        # An explicit mode choice ends the data-driven auto default.
        self._sim_mode_auto = False
        self._validate_sim_hkl(announce=False)
        self._update_sim_orient_visibility()
        self._apply_sim_pattern()

    def _on_sim_matched_changed(self, _index: int) -> None:
        self._apply_sim_pattern()

    def _on_sim_hkl_edited(self) -> None:
        self._validate_sim_hkl()
        self._apply_sim_pattern()

    def _on_sim_int_spin_changed(self, value: float) -> None:
        """Cutoff edited (typed or stepped): apply it as a fraction."""
        self.viewer.set_simulation_min_intensity(float(value) / 100.0)

    def _on_sim_selection_changed(self, indices: list) -> None:
        self._sim_sel_count_label.setText(f"{len(indices)} selected")
        self._update_sim_add_button()

    def _update_sim_add_button(self) -> None:
        """Enable state + tooltip of "Add selected peaks (fit + match)".

        Cheap by construction: reads the viewer's already-loaded frame
        tables instead of the file, so it can run on every selection /
        frame change.
        """
        btn = getattr(self, "_sim_add_btn", None)
        if btn is None:
            return
        selected = self.viewer.simulation_selected()
        tables = (
            self.viewer._frame_peaks.get(self.viewer.current_frame) or {}
        )
        has_datasets = (
            tables.get("detected") is not None
            and tables.get("fitted") is not None
        )
        busy = (
            getattr(self, "_pipe_thread", None) is not None
            or bool(getattr(self, "_pipeline_queue", None))
        )
        if not selected:
            enabled = False
            tip = (
                "Select predicted reflections first — click markers on "
                "the image or use Select missed."
            )
        elif not has_datasets:
            enabled = False
            tip = "Run detection + fitting on this frame first."
        elif busy:
            enabled = False
            tip = "The pipeline is busy — wait for the current run."
        else:
            enabled = True
            tip = (
                "Inject a detected box at each selected predicted "
                "position, 2D-fit it, then re-match this frame so the "
                "new peaks join the matched solutions. Not undoable — "
                "re-run Detection/Fitting/Matching to reset."
            )
        btn.setEnabled(enabled)
        btn.setToolTip(tip)
        # The all-frames sweep needs no selection and no per-frame
        # datasets up front (it plans per frame itself) — only an idle
        # pipeline.
        sweep = getattr(self, "_sim_sweep_btn", None)
        if sweep is not None:
            sweep.setEnabled(not busy)
            sweep.setToolTip(
                "The pipeline is busy — wait for the current run."
                if busy else
                "Collect every matched structure/orientation found "
                "anywhere in this entry, then try to find its expected "
                "peaks on EVERY frame: missing ones are injected as "
                "detected boxes, 2D-fitted and re-matched; injected "
                "peaks the matcher does not attribute to one of those "
                "structures are discarded again. Not undoable — re-run "
                "Detection/Fitting/Matching to reset."
            )

    def _ask_predicted_add_options(
        self, pattern, frame: int, n_frames: int,
    ) -> dict | None:
        """Options dialog for adding predicted peaks: re-match scope,
        frame range and a reflection cap. Returns
        ``{"scope": "restricted"|"full"|"custom",
        "cifs": set[str] | None, "frames": (lo, hi), "cap": int}`` or
        None (cancelled). ``cifs`` carries the chosen stems for the
        custom scope — they double as the validation accept-set (a
        peak kept only when matched by one of them); ``cap`` keeps at
        most the N strongest selected reflections as the template."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Expected pattern")
        v = QVBoxLayout(dlg)
        intro = QLabel(
            "The selected reflections are injected as detected boxes, "
            "2D-fitted, then each affected frame is re-matched. Peaks "
            "the matcher does not attribute to the target structure "
            f"('{pattern.cif}' — or any chosen CIF below) are "
            "discarded again afterwards; matched ones keep their "
            "detected + fitted rows for review.\n\n"
            "Positions already holding a fitted peak get no duplicate "
            "box — the frame is only re-matched so that peak can be "
            "claimed; positions already matched to the target "
            "structure(s) are skipped. The written peaks are not "
            "undoable — re-run Detection/Fitting/Matching to reset."
        )
        intro.setWordWrap(True)
        v.addWidget(intro)

        v.addWidget(QLabel("Re-match scope:"))
        rb_restricted = QRadioButton(
            f"Matched CIFs + '{pattern.cif}' (fast, keeps existing "
            "solutions)"
        )
        rb_full = QRadioButton(
            "Full CIF source (can discover new structures, slower)"
        )
        rb_custom = QRadioButton(
            "Specific CIFs (match ONLY the checked structures; "
            "existing identifications are kept):"
        )
        rb_restricted.setChecked(True)
        v.addWidget(rb_restricted)
        v.addWidget(rb_full)
        v.addWidget(rb_custom)
        cif_list = QListWidget(dlg)
        cif_list.setMaximumHeight(140)
        cache = self.pipeline_panel.cached_cif_pattern()
        for name in getattr(cache, "cifs", []) or []:
            stem = simulation_pattern.cif_stem(name)
            item = QListWidgetItem(stem)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if stem == pattern.cif
                else Qt.CheckState.Unchecked
            )
            cif_list.addItem(item)
        cif_list.setEnabled(False)
        rb_custom.toggled.connect(cif_list.setEnabled)
        v.addWidget(cif_list)

        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Frames:"))
        spin_from = QSpinBox(dlg)
        spin_from.setRange(0, max(0, n_frames - 1))
        spin_from.setValue(int(frame))
        frames_row.addWidget(spin_from)
        frames_row.addWidget(QLabel("to"))
        spin_to = QSpinBox(dlg)
        spin_to.setRange(0, max(0, n_frames - 1))
        spin_to.setValue(int(frame))
        frames_row.addWidget(spin_to)
        btn_active = QPushButton("Active frame")
        btn_active.setToolTip("Only the currently shown frame.")
        frames_row.addWidget(btn_active)
        btn_all = QPushButton("All")
        btn_all.setToolTip("Every frame of the scan.")
        frames_row.addWidget(btn_all)
        frames_row.addStretch(1)
        v.addLayout(frames_row)

        def _set_range(lo: int, hi: int) -> None:
            spin_from.setValue(lo)
            spin_to.setValue(hi)

        btn_active.clicked.connect(lambda: _set_range(frame, frame))
        btn_all.clicked.connect(lambda: _set_range(0, max(0, n_frames - 1)))

        # Scale limiter, same idea as the sweep dialog's per-structure
        # caps: only the N strongest of the selected reflections are
        # used as the template (each injected box costs a 2D fit per
        # frame and every extra peak slows the re-match).
        n_sel = len(self.viewer.simulation_selected())
        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("Add at most"))
        spin_cap = QSpinBox(dlg)
        spin_cap.setRange(1, max(1, n_sel))
        spin_cap.setValue(min(10, max(1, n_sel)))
        cap_row.addWidget(spin_cap)
        cap_row.addWidget(QLabel(
            f"of the {n_sel} selected reflection(s), strongest first"
        ))
        cap_row.addStretch(1)
        v.addLayout(cap_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)

        while True:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            if not rb_custom.isChecked():
                cifs = None
                scope = "restricted" if rb_restricted.isChecked() else "full"
                break
            cifs = {
                cif_list.item(i).text()
                for i in range(cif_list.count())
                if cif_list.item(i).checkState() == Qt.CheckState.Checked
            }
            if cifs:
                scope = "custom"
                break
            QMessageBox.information(
                dlg, "Expected pattern",
                "Check at least one CIF for the specific-CIFs scope.",
            )
        lo, hi = sorted((int(spin_from.value()), int(spin_to.value())))
        return {
            "scope": scope,
            "cifs": cifs,
            "frames": (lo, hi),
            "cap": int(spin_cap.value()),
        }

    def _plan_predicted_fills(
        self, entry: str, pattern, reflections, f_lo: int, f_hi: int,
        accept_stems: set,
    ) -> tuple[dict, dict, dict, list]:
        """Per-frame injection specs for the template ``reflections``.

        For every frame in ``[f_lo, f_hi]`` each reflection is judged
        twice (matched boxes are always a SUBSET of fitted boxes, so
        explained implies covered):

        - not covered by ANY fitted box → inject a detected box there
          (this is the only duplicate-avoidance gate);
        - covered but not explained by a matched box of an
          ``accept_stems`` structure → no new box, but the frame gets
          a re-match pass so the unclaimed fitted peak sitting at the
          predicted position can be attributed.

        Frames without detected/fitted datasets are skipped whole. Box
        sizes seed from each frame's own median fitted box. Returns
        ``(plan, match_need, frame_cifs, skipped_frames)`` where
        ``match_need`` maps each frame needing a re-match to the peaks
        types involved ("segments"/"rings") and ``frame_cifs`` maps
        those frames to the CIF stems already matched there (the
        restricted re-match set must carry them — matching rewrites
        the frame's solutions wholesale).
        """
        import h5py

        accept = {simulation_pattern.cif_stem(c) for c in accept_stems}
        plan: dict[int, list] = {}
        match_need: dict[int, set] = {}
        frame_cifs: dict[int, set] = {}
        skipped: list[int] = []
        with h5py.File(self.session.temp_path, "r") as f:
            for fr in range(int(f_lo), int(f_hi) + 1):
                tables = file_model.read_peaks(f, entry, fr)
                fitted = tables.get("fitted")
                if tables.get("detected") is None or fitted is None:
                    skipped.append(fr)
                    continue
                structures = file_model.read_matched_peaks(
                    f, entry, fr, fitted
                )
                matched_boxes = simulation_pattern.merge_boxes([
                    s.peaks for s in structures
                    if simulation_pattern.cif_stem(s.cif) in accept
                ])
                # The seed size doubles as the coverage-recognition
                # floor: a peak inside the box we WOULD inject means
                # the reflection is already accounted for there.
                rw, aw = simulation_pattern.default_box_size(fitted)
                explained = simulation_pattern.classify_explained(
                    reflections, matched_boxes,
                    seed_radius_width=rw, seed_angle_width=aw,
                )
                covered = simulation_pattern.classify_explained(
                    reflections, fitted,
                    seed_radius_width=rw, seed_angle_width=aw,
                )
                todo = []
                need: set[str] = set()
                for r, e, c in zip(reflections, explained, covered):
                    if e:
                        continue
                    if not c:
                        todo.append(r)
                    need.add("rings" if r.is_ring else "segments")
                if todo:
                    plan[fr] = simulation_pattern.build_injection_specs(
                        pattern, [r.index for r in todo], rw, aw
                    )
                if need:
                    match_need[fr] = need
                    frame_cifs[fr] = {
                        simulation_pattern.cif_stem(s.cif)
                        for s in structures
                    }
        return plan, match_need, frame_cifs, skipped

    def _build_predicted_match_commands(
        self, match_need: dict, frame_cifs: dict, entry: str,
        match_kwargs: dict, pattern, scope: str,
        custom_cifs: set | None = None,
    ) -> list:
        """Re-match command(s) chained after ``inject_fitted_peaks``.

        ``match_need`` maps frame → peaks types to re-match there
        (segments and rings run separately — mlgidmatch matches one
        ``peaks_type`` per run); it covers both freshly injected
        boxes and predicted positions already holding an unclaimed
        fitted peak. ``scope == "restricted"`` subsets the raw CIF
        source per frame to its already-matched CIFs plus the
        overlay's CIF; ``scope == "custom"`` subsets it to EXACTLY
        the user-chosen ``custom_cifs`` — no frame CIFs ride along
        (``pattern`` may be None then — the all-frames sweep has no
        single overlay structure). Matching still rewrites the
        frame's ``matched_<type>_*`` wholesale, but identifications
        a narrow source cannot reproduce are restored by the
        finalize snapshot merge-back, so a custom scope never loses
        them. Frames sharing the same (type, CIF-set) collapse into
        one command with a ``frame_num`` list. Unlike the
        interpolate chain — which SKIPS unsubsettable sources — an
        unsubsettable source falls back to the FULL source with a
        log note: the user explicitly asked for these peaks, so a
        match must always run.
        """
        raw_source = self.pipeline_panel.cif_source_text()
        if not raw_source and isinstance(match_kwargs.get("cif_prepr"), str):
            raw_source = match_kwargs["cif_prepr"]

        groups: dict = {}  # (peaks_type, frozenset | None) -> [frames]
        for fr, types in match_need.items():
            for peaks_type in types:
                key = None
                if scope == "restricted":
                    key = frozenset(
                        {pattern.cif} | frame_cifs.get(int(fr), set())
                    )
                elif scope == "custom":
                    key = frozenset(custom_cifs or {pattern.cif})
                groups.setdefault((peaks_type, key), []).append(int(fr))

        cmds: list = []
        fell_back = False
        for (peaks_type, key), frames in sorted(
            groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))
        ):
            restricted_value = None
            if key is not None:
                restricted_value = self._restrict_cif_source(
                    raw_source, set(key)
                )
                if restricted_value is None:
                    fell_back = True
            kw = {
                **match_kwargs,
                "entry": entry,
                "frame_num": sorted(set(frames)),
                "peaks_type": peaks_type,
            }
            if restricted_value is not None:
                kw["cif_prepr"] = restricted_value
            cmds.append(PipelineCommand("run_matching", kw))
        if fell_back:
            self.pipeline_panel.append_log(
                "Expected pattern: the CIF source cannot be subset "
                "(pickle / pre-parsed object / missing .cif files) "
                "— re-matching with the FULL source instead."
            )
        return cmds

    def _on_add_predicted_peaks(self) -> None:
        """Turn the selected predicted reflections into real peaks.

        The selection acts as a TEMPLATE over the frame range chosen
        in the options dialog: per frame, only the reflections not
        already covered by a fitted peak there are injected; positions
        holding an unclaimed fitted peak get a re-match pass instead
        of a duplicate box (``_plan_predicted_fills``). Enqueues one
        ``inject_fitted_peaks`` command (inject detected box, 2D-fit,
        persist fitted row — per box) when anything needs injecting,
        followed by ``run_matching`` command(s) grouped per
        (peaks type, CIF set). When the chain completes,
        ``_finalize_predicted_fill`` keeps only the injected peaks the
        matcher attributed to the target structure — the rest lose
        their fitted AND detected rows again.
        """
        if (
            self.session is None
            or self._pipe_thread is not None
            or self._pipeline_queue
            or self._view_worker_blocks_pipeline()
        ):
            return
        entry = self.entry_combo.currentText()
        pattern = self.viewer.simulation_pattern()
        selected = self.viewer.simulation_selected()
        if not entry or pattern is None or not selected:
            return
        frame = int(self.viewer.current_frame)
        if file_model.read_geometry_for_entry(
            self.session.temp_path, entry, frame
        ) is None:
            QMessageBox.warning(
                self, "Expected pattern",
                "This entry has no instrument geometry (wavelength / "
                "incidence angle) — the 2D fit cannot run.",
            )
            return
        match_kwargs = self.pipeline_panel.matching_kwargs()
        if match_kwargs is None:
            QMessageBox.warning(
                self, "Expected pattern",
                "Adding predicted peaks ends with a re-match of the "
                "affected frames, but no CIF source is configured. Set "
                "the CIF file/folder (or pickle) in the Pipeline "
                "panel's Matching section first.",
            )
            return
        options = self._ask_predicted_add_options(
            pattern, frame, self.viewer.n_frames
        )
        if options is None:
            return
        f_lo, f_hi = options["frames"]
        reflections = [
            pattern.reflections[i] for i in selected
            if 0 <= i < len(pattern.reflections)
        ]
        # The dialog's cap: keep the N strongest selected reflections
        # (stubbed dialogs may omit it — then everything selected runs).
        cap = options.get("cap")
        if cap is not None and len(reflections) > int(cap):
            reflections = sorted(
                reflections, key=lambda r: r.rel_intensity, reverse=True,
            )[: int(cap)]
        # "accept" is the set of CIF stems whose solutions validate an
        # injected peak (and whose matched boxes count as "explained"
        # during planning): the overlay's structure, or the user's
        # picks for the specific-CIFs scope.
        accept = (
            set(options["cifs"]) if options.get("cifs")
            else {pattern.cif}
        )
        plan, match_need, frame_cifs, skipped = self._plan_predicted_fills(
            entry, pattern, reflections, f_lo, f_hi, accept
        )
        if skipped:
            self.pipeline_panel.append_log(
                "Expected pattern: skipping frame(s) without "
                f"detected/fitted datasets: {skipped} — run Detection "
                "and Fitting there first."
            )
        if not match_need:
            self.statusBar().showMessage(
                "Expected pattern: nothing to do — every selected "
                "reflection is already matched to the target "
                "structure(s) on the chosen frames (or the frames "
                "lack detection/fitting).", 8000,
            )
            return
        commands = []
        ticks = {}
        if plan:
            inject_cmd = PipelineCommand("inject_fitted_peaks", {
                "entry": entry,
                "plan": plan,
                "fit_params": self._panel_fit_params(),
                # The op writes a detected row and its fit as a
                # pair; with the link on they share an id. Read
                # here, not in the worker.
                "link_ids": link_enabled(),
            })
            commands.append(inject_cmd)
            ticks[id(inject_cmd)] = len(plan)
        match_cmds = self._build_predicted_match_commands(
            match_need, frame_cifs, entry, match_kwargs, pattern,
            options["scope"], options.get("cifs"),
        )
        self._interp_fill_result = None
        # Validation stash consumed by _finalize_predicted_fill on
        # chain completion; "records" lands when the inject command
        # finishes (stays None on a re-match-only run — pre-existing
        # peaks are never rolled back), "match_ok" drops on any
        # re-match error (then the rollback is skipped — better
        # unvalidated peaks than peaks discarded for a crash that
        # wasn't theirs). "snapshot" holds each affected frame's
        # matched solutions as they are NOW: the re-match rewrites
        # them from scratch and may fail to reproduce an existing
        # identification — those are merged back at finalize.
        self._predicted_fill = {
            "entry": entry,
            "cif": pattern.cif,
            "accept": accept,
            "records": None,
            "match_ok": True,
            "snapshot": self._matched_snapshot(entry, match_need),
        }
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        commands += match_cmds
        for cmd in match_cmds:
            ticks[id(cmd)] = len(cmd.kwargs.get("frame_num") or []) or 1
        self._interp_chain = {
            "commands": commands,
            "ticks": ticks,
            "total": max(1, sum(ticks.values())),
            "base": 0,
            "label": "Expected pattern",
        }
        n_boxes = sum(len(v) for v in plan.values())
        rematch_only = sorted(set(match_need) - set(plan))
        self.pipeline_panel.append_log(
            f"Expected pattern: adding {n_boxes} predicted box(es) of "
            f"{pattern.cif} across {len(plan)} frame(s) "
            f"[{f_lo}–{f_hi}] of {entry}…"
        )
        if rematch_only:
            self.pipeline_panel.append_log(
                "Expected pattern: re-match only on frame(s) "
                f"{rematch_only} — the predicted positions already "
                "hold fitted peaks not yet attributed to the target "
                "structure(s); no duplicate boxes injected."
            )
        self._show_tracking_progress(
            "Expected pattern: fitting injected boxes…", manual=True,
        )
        for cmd in commands:
            self._enqueue_pipeline(self.session.temp_path, cmd)
        self._update_sim_add_button()

    def _matched_snapshot(self, entry: str, frames) -> dict:
        """Raw matched datasets of ``frames``, captured before their
        re-match (merged back at finalize — see
        ``_finalize_predicted_fill``). Never raises: a snapshot
        failure must not block the add itself."""
        try:
            return file_model.read_matched_raw(
                self.session.temp_path, entry, sorted(frames)
            )
        except Exception:
            logger.debug(
                "suppressed exception snapshotting matched data",
                exc_info=True,
            )
            return {}

    def _collect_matched_combos(
        self, entry: str,
    ) -> tuple[list, list, int]:
        """Every matched (CIF, orientation) combo anywhere in ``entry``,
        resolved against the parsed CIF cache.

        Returns ``(targets, uncomputable, n_ready)``: each target is a
        dict with ``stem`` / ``hkl`` / ``label`` / ``pattern`` /
        ``reflections`` (the expected pattern cut to the section's
        min-intensity threshold and the data's q-range, sorted by
        relative intensity descending — the confirmation dialog's
        per-structure caps truncate it); combos whose CIF is missing
        from the cache or whose orientation was not precomputed land in
        ``uncomputable`` as human-readable strings; ``n_ready`` counts
        the frames with detected+fitted datasets (the ones the sweep
        can plan on — used for the dialog's workload estimate).
        """
        import h5py

        cache = self.pipeline_panel.cached_cif_pattern()
        combos: dict[tuple, None] = {}  # insertion-ordered dedupe
        n_ready = 0
        with h5py.File(self.session.temp_path, "r") as f:
            for fr in range(int(self.viewer.n_frames)):
                tables = file_model.read_peaks(f, entry, fr)
                fitted = tables.get("fitted")
                if fitted is None:
                    continue
                if tables.get("detected") is not None:
                    n_ready += 1
                for s in file_model.read_matched_peaks(f, entry, fr, fitted):
                    stem = simulation_pattern.cif_stem(s.cif)
                    hkl = (int(s.h), int(s.k), int(s.l))
                    combos.setdefault((stem, hkl))
        targets: list[dict] = []
        uncomputable: list[str] = []
        cutoff = float(self._sim_int_spin.value()) / 100.0
        q_max = self.viewer._sim_q_max()
        for stem, hkl in combos:
            label = f"{stem} " + (
                "powder" if hkl == simulation_pattern.POWDER_HKL
                else "({} {} {})".format(*hkl)
            )
            idx = simulation_pattern.cif_index(cache, stem)
            if idx is None:
                uncomputable.append(
                    f"{label} — not in the parsed CIF cache"
                )
                continue
            try:
                pattern = simulation_pattern.extract_pattern(
                    cache, idx,
                    None if hkl == simulation_pattern.POWDER_HKL else hkl,
                )
            except ValueError as exc:
                uncomputable.append(f"{label} — {exc}")
                continue
            reflections = [
                r for r in pattern.reflections
                if r.rel_intensity >= cutoff
                and not (
                    r.is_ring and q_max is not None and r.radius > q_max
                )
            ]
            if not reflections:
                uncomputable.append(
                    f"{label} — no reflections above the intensity cutoff"
                )
                continue
            # Strongest first, so a per-structure cap is a simple
            # truncation (chosen in the sweep's confirmation dialog).
            reflections = sorted(
                reflections, key=lambda r: r.rel_intensity, reverse=True,
            )
            targets.append({
                "stem": stem, "hkl": hkl, "label": label,
                "pattern": pattern, "reflections": reflections,
            })
        return targets, uncomputable, n_ready

    def _confirm_matched_sweep(
        self, targets: list, uncomputable: list, n_frames: int,
        n_ready: int,
    ) -> list[int] | None:
        """Configuration/confirmation dialog for the all-frames sweep.

        One spinbox per found structure chooses how many of ITS
        strongest predicted reflections (candidate count shown, the
        spinbox upper bound) to look for on every frame, with a live
        upper-bound estimate of the injected-box workload. Returns the
        caps aligned with ``targets``, or None (cancelled). Separate
        method so tests can stub the answer.
        """
        cutoff = float(self._sim_int_spin.value())
        dlg = QDialog(self)
        dlg.setWindowTitle("Expected pattern")
        v = QVBoxLayout(dlg)
        intro = QLabel(
            "Try to find the expected peaks of every matched structure "
            f"on all {n_frames} frames of this entry. Per structure, "
            "choose how many of its strongest predicted reflections "
            f"(above {cutoff:g} % relative intensity) to look for; "
            "each missing one is injected as a detected box and "
            "2D-fitted, then every affected frame is re-matched. "
            "Injected peaks the matcher does not attribute to one of "
            "these structures are discarded again afterwards, and "
            "structures already matched are kept even when the "
            "re-match fails to reproduce them. The written peaks are "
            "not undoable — re-run Detection/Fitting/Matching to "
            "reset."
        )
        intro.setWordWrap(True)
        v.addWidget(intro)

        spins: list[QSpinBox] = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 6, 0, 6)
        grid.setHorizontalSpacing(8)
        for row, t in enumerate(targets):
            n_cand = len(t["reflections"])
            grid.addWidget(
                QLabel(f"{t['label']} — {n_cand} candidate(s)"), row, 0,
            )
            spin = QSpinBox(dlg)
            spin.setRange(1, n_cand)
            spin.setValue(min(10, n_cand))
            spin.setSuffix(" refl")
            spins.append(spin)
            grid.addWidget(spin, row, 1)
        grid.setColumnStretch(0, 1)
        v.addLayout(grid)

        estimate = QLabel()
        estimate.setWordWrap(True)

        def _update_estimate() -> None:
            total = sum(s.value() for s in spins) * max(1, n_ready)
            estimate.setText(
                f"Workload upper bound: {total} injected box(es) — one "
                f"2D fit each — across {n_ready} frame(s) with "
                "detection + fitting, plus their re-match. Positions "
                "already fitted are skipped, so the real count is "
                "usually lower; matching can still take a while on "
                "large scans."
            )

        for s in spins:
            s.valueChanged.connect(_update_estimate)
        _update_estimate()
        v.addWidget(estimate)

        if uncomputable:
            note = QLabel(
                "Skipped (cannot simulate):\n- " + "\n- ".join(uncomputable)
            )
            note.setWordWrap(True)
            v.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        return [int(s.value()) for s in spins]

    def _plan_matched_sweep(
        self, entry: str, targets: list,
    ) -> tuple[dict, dict, dict, list]:
        """Sweep planner: apply every target combo to every frame.

        Same per-reflection rules as ``_plan_predicted_fills`` — inject
        when no fitted box covers the position, re-match when a fitted
        box covers it but no matched box of the combo's OWN CIF does —
        except that each combo is judged against its own phase and
        boxes already queued for the frame by earlier combos count as
        coverage (two structures predicting the same position must not
        produce two boxes). Returns
        ``(plan, match_need, frame_cifs, skipped_frames)``.
        """
        import h5py

        plan: dict[int, list] = {}
        match_need: dict[int, set] = {}
        frame_cifs: dict[int, set] = {}
        skipped: list[int] = []
        with h5py.File(self.session.temp_path, "r") as f:
            for fr in range(int(self.viewer.n_frames)):
                tables = file_model.read_peaks(f, entry, fr)
                fitted = tables.get("fitted")
                if tables.get("detected") is None or fitted is None:
                    skipped.append(fr)
                    continue
                structures = file_model.read_matched_peaks(
                    f, entry, fr, fitted
                )
                rw, aw = simulation_pattern.default_box_size(fitted)
                specs: list[dict] = []
                need: set[str] = set()
                for t in targets:
                    matched_boxes = simulation_pattern.merge_boxes([
                        s.peaks for s in structures
                        if simulation_pattern.cif_stem(s.cif) == t["stem"]
                    ])
                    explained = simulation_pattern.classify_explained(
                        t["reflections"], matched_boxes,
                        seed_radius_width=rw, seed_angle_width=aw,
                    )
                    covered = simulation_pattern.classify_explained(
                        t["reflections"], fitted,
                        seed_radius_width=rw, seed_angle_width=aw,
                    ) | simulation_pattern.classify_explained(
                        t["reflections"],
                        simulation_pattern.specs_to_boxes(specs),
                        seed_radius_width=rw, seed_angle_width=aw,
                    )
                    todo = []
                    for r, e, c in zip(t["reflections"], explained, covered):
                        if e:
                            continue
                        if not c:
                            todo.append(r)
                        need.add("rings" if r.is_ring else "segments")
                    if todo:
                        specs.extend(
                            simulation_pattern.build_injection_specs(
                                t["pattern"], [r.index for r in todo],
                                rw, aw,
                            )
                        )
                if specs:
                    plan[fr] = specs
                if need:
                    match_need[fr] = need
                    frame_cifs[fr] = {
                        simulation_pattern.cif_stem(s.cif)
                        for s in structures
                    }
        return plan, match_need, frame_cifs, skipped

    def _on_fill_all_matched(self) -> None:
        """All-frames structure sweep: collect every matched
        (CIF, orientation) combo in the entry and try to find its
        expected peaks on EVERY frame — inject the missing ones as
        detected boxes, 2D-fit, re-match. The standard validation
        applies at chain completion: injected peaks not matched to one
        of the swept structures lose their fitted and detected rows
        again; matched ones keep both for review.
        """
        if (
            self.session is None
            or self._pipe_thread is not None
            or self._pipeline_queue
            or self._view_worker_blocks_pipeline()
        ):
            return
        entry = self.entry_combo.currentText()
        panel = self.pipeline_panel
        if not entry or panel.cached_cif_pattern() is None:
            return
        if file_model.read_geometry_for_entry(
            self.session.temp_path, entry, int(self.viewer.current_frame)
        ) is None:
            QMessageBox.warning(
                self, "Expected pattern",
                "This entry has no instrument geometry (wavelength / "
                "incidence angle) — the 2D fit cannot run.",
            )
            return
        match_kwargs = panel.matching_kwargs()
        if match_kwargs is None:
            QMessageBox.warning(
                self, "Expected pattern",
                "The structure sweep ends with a re-match of the "
                "affected frames, but no CIF source is configured. Set "
                "the CIF file/folder (or pickle) in the Pipeline "
                "panel's Matching section first.",
            )
            return
        targets, uncomputable, n_ready = self._collect_matched_combos(entry)
        if uncomputable:
            panel.append_log(
                "Expected pattern: cannot simulate matched "
                "structure(s): " + "; ".join(uncomputable)
            )
        if not targets:
            self.statusBar().showMessage(
                "Expected pattern: no matched structures found in this "
                "entry (or none can be simulated from the parsed CIF "
                "cache).", 8000,
            )
            return
        n_frames = int(self.viewer.n_frames)
        caps = self._confirm_matched_sweep(
            targets, uncomputable, n_frames, n_ready
        )
        if caps is None:
            return
        # Per-structure caps: candidates are pre-sorted strongest-first.
        for t, cap in zip(targets, caps):
            t["reflections"] = t["reflections"][: max(1, int(cap))]
        plan, match_need, frame_cifs, skipped = self._plan_matched_sweep(
            entry, targets
        )
        if skipped:
            panel.append_log(
                "Expected pattern: skipping frame(s) without "
                f"detected/fitted datasets: {skipped} — run Detection "
                "and Fitting there first."
            )
        if not match_need:
            self.statusBar().showMessage(
                "Expected pattern: nothing to do — every expected peak "
                "of the matched structures is already matched on its "
                "frame.", 8000,
            )
            return
        n_boxes = sum(len(v) for v in plan.values())
        rematch_only = sorted(set(match_need) - set(plan))
        accept = {t["stem"] for t in targets}
        commands: list = []
        ticks: dict = {}
        if plan:
            inject_cmd = PipelineCommand("inject_fitted_peaks", {
                "entry": entry,
                "plan": plan,
                "fit_params": self._panel_fit_params(),
                # The op writes a detected row and its fit as a
                # pair; with the link on they share an id. Read
                # here, not in the worker.
                "link_ids": link_enabled(),
            })
            commands.append(inject_cmd)
            ticks[id(inject_cmd)] = len(plan)
        match_cmds = self._build_predicted_match_commands(
            match_need, frame_cifs, entry, match_kwargs,
            None, "custom", accept,
        )
        self._interp_fill_result = None
        self._predicted_fill = {
            "entry": entry,
            "cif": "/".join(sorted(accept)),
            "accept": accept,
            "records": None,
            "match_ok": True,
            "snapshot": self._matched_snapshot(entry, match_need),
        }
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        commands += match_cmds
        for cmd in match_cmds:
            ticks[id(cmd)] = len(cmd.kwargs.get("frame_num") or []) or 1
        self._interp_chain = {
            "commands": commands,
            "ticks": ticks,
            "total": max(1, sum(ticks.values())),
            "base": 0,
            "label": "Expected pattern",
        }
        panel.append_log(
            f"Expected pattern: sweeping {len(targets)} matched "
            f"structure(s) ({', '.join(t['label'] for t in targets)}) "
            f"over {n_frames} frame(s) of {entry} — adding {n_boxes} "
            f"predicted box(es) on {len(plan)} frame(s)…"
        )
        if rematch_only:
            panel.append_log(
                "Expected pattern: re-match only on frame(s) "
                f"{rematch_only} — the predicted positions already "
                "hold fitted peaks not yet attributed to the swept "
                "structure(s); no duplicate boxes injected."
            )
        self._show_tracking_progress(
            "Expected pattern: fitting injected boxes…", manual=True,
        )
        for cmd in commands:
            self._enqueue_pipeline(self.session.temp_path, cmd)
        self._update_sim_add_button()

    def _finalize_predicted_fill(self) -> None:
        """Post-match validation of injected predicted peaks.

        A peak earns its keep by appearing in a matched solution of
        an ACCEPTED structure on its frame (same CIF, any orientation
        — the matcher may attribute the peak to a different texture
        of the same phase). The accept-set is the overlay's target
        CIF, the user's picks under the specific-CIFs scope, or every
        swept structure for the all-frames sweep. Everything else is
        rolled back: fitted row deleted, ``matched_*`` peak_list
        entries for it dropped (``prepare_fitted_delete`` /
        ``remap_matched_peak_lists``), detected row deleted.
        Afterwards the pre-run matched snapshot is union-merged back so
        identifications the re-match failed to reproduce are never
        lost. Runs at chain completion, before the
        queue-drain reload — silx is still detached, so the r+ writes
        are safe.
        """
        stash = getattr(self, "_predicted_fill", None)
        self._predicted_fill = None
        if stash is None or self.session is None:
            return
        entry = stash["entry"]
        path = self.session.temp_path
        records = stash.get("records") or []
        if records and not stash.get("match_ok", True):
            self.pipeline_panel.append_log(
                "Expected pattern: a re-match failed — keeping all "
                f"{len(records)} injected peak(s) UNVALIDATED; review "
                "or re-run matching manually."
            )
            records = []
        if records:
            self._rollback_unvalidated_fills(stash, records)
        # Merge-back pass, regardless of records: the re-match rewrote
        # each frame's solutions from scratch, and the (temporarily)
        # larger peak set can push a previously identified structure
        # below the matcher's threshold — the user's existing results
        # must survive an Add/sweep. The snapshot's peak_lists are
        # fitted ids (``read_matched_raw``), which stay valid: injected
        # rows are appended under fresh ids and only appended rows are
        # rolled back.
        snapshot = stash.get("snapshot") or {}
        restored = 0
        for fr, snap in sorted(snapshot.items()):
            if not snap:
                continue
            try:
                restored += file_model.merge_matched_rows(
                    path, entry, int(fr), snap
                )
            except Exception:
                logger.debug(
                    "suppressed exception merging matched snapshot on "
                    "frame %s", fr, exc_info=True,
                )
        if restored:
            self.pipeline_panel.append_log(
                f"Expected pattern: restored {restored} previously "
                "matched structure(s) the re-match had dropped (peak "
                "lists merged)."
            )

    def _rollback_unvalidated_fills(self, stash: dict, records: list) -> None:
        """Drop injected peaks the matcher did not attribute to an
        accepted structure (see ``_finalize_predicted_fill``)."""
        entry = stash["entry"]
        accept = {
            str(c) for c in stash.get("accept") or {stash["cif"]}
        }
        path = self.session.temp_path
        by_frame: dict[int, list] = {}
        for rec in records:
            by_frame.setdefault(int(rec["frame"]), []).append(rec)
        kept = 0
        discarded = 0
        for fr, recs in sorted(by_frame.items()):
            try:
                tables = file_model.load_peaks(path, entry, fr)
                structures = file_model.load_matched_peaks(
                    path, entry, fr, tables.get("fitted")
                )
            except Exception:
                logger.debug(
                    "suppressed exception validating predicted fills "
                    "on frame %d", fr, exc_info=True,
                )
                kept += len(recs)
                continue
            validated: set[int] = set()
            for s in structures:
                if simulation_pattern.cif_stem(s.cif) not in accept:
                    continue
                validated.update(
                    int(i) for i in np.asarray(s.peaks.ids, dtype=int)
                )
            drop = [
                r for r in recs if int(r["fitted_id"]) not in validated
            ]
            kept += len(recs) - len(drop)
            if not drop:
                continue
            # Positions BEFORE deleting (this also puts the frame's
            # matched peak_lists into id form, which needs the
            # pre-delete fitted table).
            positions = file_model.prepare_fitted_delete(
                path, entry, fr, [int(r["fitted_id"]) for r in drop]
            )
            for r in drop:
                file_model.delete_peak_row(
                    path, entry, fr, "fitted", int(r["fitted_id"])
                )
                file_model.delete_peak_row(
                    path, entry, fr, "detected", int(r["detected_id"])
                )
            file_model.remap_matched_peak_lists(path, entry, fr, positions)
            discarded += len(drop)
        msg = (
            f"Expected pattern: kept {kept} peak(s) matched by "
            f"{'/'.join(sorted(accept))}, discarded {discarded} "
            "(fitted but not matched by the target structure(s))."
        )
        self.pipeline_panel.append_log(msg)
        self.statusBar().showMessage(msg, 8000)

    def _apply_sim_pattern(self) -> None:
        """Install the selected (CIF, orientation) pattern in the
        viewer, or clear the overlay when the selection is incomplete
        or the section is off. No-ops when the effective selection is
        unchanged — a re-install would drop the reflection selection."""
        panel = getattr(self, "pipeline_panel", None)
        cache = panel.cached_cif_pattern() if panel is not None else None
        desired: tuple | None = None
        if (
            cache is not None
            and self._sim_master_check.isChecked()
            and self.session is not None
            and self.session.kind != "raw"
        ):
            stem = self._sim_cif_combo.currentData()
            hkl = self._sim_selected_hkl()
            if stem is not None and hkl is not None:
                desired = (stem, tuple(int(v) for v in hkl))
        # No-op on an unchanged choice — but only while the viewer's
        # actual state agrees with the bookkeeping (viewer.clear() on a
        # session reload drops the pattern behind our back, and
        # _on_cif_cache_changed resets _sim_applied before the combos
        # repopulate).
        viewer_has = self.viewer.simulation_pattern() is not None
        if desired == self._sim_applied and viewer_has == (desired is not None):
            return
        if desired is None:
            self._sim_applied = None
            self.viewer.clear_simulation_pattern()
            return
        ci = simulation_pattern.cif_index(cache, desired[0])
        if ci is None:
            self._sim_applied = None
            self.viewer.clear_simulation_pattern()
            self.statusBar().showMessage(
                f"Expected pattern: CIF '{desired[0]}' is not in the "
                "parsed cache — re-parse the CIFs.", 6000,
            )
            return
        try:
            pattern = simulation_pattern.extract_pattern(
                cache, ci, desired[1]
            )
        except (ValueError, IndexError, TypeError) as exc:
            self._sim_applied = None
            self.viewer.clear_simulation_pattern()
            self.statusBar().showMessage(f"Expected pattern: {exc}", 6000)
            return
        self._sim_applied = desired
        self.viewer.set_simulation_pattern(pattern)

    def _sim_entry_mismatch_hint(self, entry: str) -> None:
        """Status-bar warning when the cached CifPattern was parsed
        against a different energy / incidence angle than ``entry``.

        Mirrors ``pipeline._warn_if_cif_params_mismatch`` thresholds
        (0.5% relative on energy; ~5% relative on incidence angle, or
        1e-3 absolute near zero) but derives the entry's values
        backend-free from the file geometry: en[eV] = 12398 /
        wavelength[Å], the same conversion ExpParameters uses.
        """
        panel = getattr(self, "pipeline_panel", None)
        if panel is None or self.session is None:
            return
        params = getattr(panel.cached_cif_pattern(), "params", None)
        cached_en = getattr(params, "en", None)
        cached_ai = getattr(params, "ai", None)
        if cached_en is None or cached_ai is None:
            return
        geom = file_model.read_geometry_for_entry(
            self.session.temp_path, entry
        )
        if geom is None:
            return
        wl = float(geom.get("wavelength_angstrom") or 0.0)
        if wl <= 0:
            return
        file_en = 12398.0 / wl
        file_ai = float(geom.get("ai_deg") or 0.0)
        en_off = abs(file_en - float(cached_en)) > 0.005 * abs(file_en)
        ai_scale = max(abs(file_ai), abs(float(cached_ai)))
        ai_off = abs(file_ai - float(cached_ai)) > (
            0.05 * ai_scale if ai_scale > 1e-3 else 1e-3
        )
        if en_off or ai_off:
            self.statusBar().showMessage(
                "Expected pattern: the parsed CIF cache was built for "
                "a different energy / incidence angle than this entry "
                "— re-parse CIFs for accurate positions.", 8000,
            )
