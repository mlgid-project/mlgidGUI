"""The tracking views' dialogs: figure/data export picker and the
track-select dialog with its scan preview, plus the track-pen and
log-display helpers they share with the window.
Moved out of ``phase_views_window`` in the 2026 source split; the
window re-exports every name so tests resolve unchanged.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlgidgui.widgets import make_debounced_timer


_UNMATCHED = "— unmatched —"


def _track_pen(k: int, n_tracks: int):
    """Distinct per-track color, cycling hues (tab20-flavored)."""
    return pg.intColor(k, hues=max(9, min(n_tracks, 20)), maxValue=255)


def _log_display(data: np.ndarray) -> np.ndarray:
    """log10 view of an intensity array; non-positive/NaN -> NaN.

    Deliberately NOT the viewer's ``_maybe_apply_log`` (which clips
    non-positives to a percentile floor and recomputes levels): the
    tracking views want gaps to render as gaps, not as floor-value
    pixels.
    """
    out = np.full(data.shape, np.nan, dtype=np.float32)
    positive = np.isfinite(data) & (data > 0)
    out[positive] = np.log10(data[positive], dtype=np.float32)
    return out


class _ExportDialog(QDialog):
    """Pick which views to export and — for images — how to style them.

    ``views`` is ``[(key, label, enabled, tooltip), ...]``; disabled
    entries stay listed (with the tooltip saying what is missing) so
    the user learns what CAN be exported. ``structures`` (image mode)
    is ``[(key, label, css_color, checked), ...]`` — the initial check
    state mirrors the window's structure toggles, but the dialog's
    selection is authoritative for the export (WYSIWYG start, then
    override). OK is enabled only while at least one view is ticked.
    """

    _PEN_STYLES = (
        ("Solid", Qt.PenStyle.SolidLine),
        ("Dashed", Qt.PenStyle.DashLine),
        ("Dotted", Qt.PenStyle.DotLine),
        ("Dash-dot", Qt.PenStyle.DashDotLine),
    )

    def __init__(
        self, parent, mode: str, views, structures=(), current_key=None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self.setWindowTitle(
            "Export plot images" if mode == "image"
            else "Export plot data (CSV)"
        )
        layout = QVBoxLayout(self)
        views_box = QGroupBox("Plots (one file each)", self)
        views_lay = QVBoxLayout(views_box)
        self._view_checks: dict = {}
        for key, label, enabled, tip in views:
            chk = QCheckBox(label, views_box)
            chk.setEnabled(enabled)
            if tip:
                chk.setToolTip(tip)
            chk.setChecked(enabled and key == current_key)
            chk.toggled.connect(self._update_ok)
            views_lay.addWidget(chk)
            self._view_checks[key] = chk
        layout.addWidget(views_box)
        note = QLabel(
            "Files are written next to the chosen name, suffixed per "
            "plot (e.g. <name>_trajectories). Existing files are "
            "overwritten.", self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._struct_checks: dict = {}
        if mode == "image":
            if structures:
                struct_box = QGroupBox("Structures to include", self)
                struct_lay = QVBoxLayout(struct_box)
                for key, label, color, checked in structures:
                    chk = QCheckBox(label, struct_box)
                    chk.setChecked(bool(checked))
                    chk.setStyleSheet(f"color: {color};")
                    struct_lay.addWidget(chk)
                    self._struct_checks[key] = chk
                layout.addWidget(struct_box)
            style_box = QGroupBox("Style", self)
            form = QFormLayout(style_box)
            self.line_width = QDoubleSpinBox(style_box)
            self.line_width.setRange(0.5, 10.0)
            self.line_width.setSingleStep(0.5)
            self.line_width.setValue(1.0)
            form.addRow("Line width", self.line_width)
            self.line_style = QComboBox(style_box)
            for name, pen_style in self._PEN_STYLES:
                self.line_style.addItem(name, userData=pen_style)
            form.addRow("Line style", self.line_style)
            self.marker_size = QSpinBox(style_box)
            self.marker_size.setRange(0, 20)
            self.marker_size.setValue(4)
            self.marker_size.setSpecialValueText("hidden")
            form.addRow("Marker size", self.marker_size)
            self.image_width = QSpinBox(style_box)
            self.image_width.setRange(400, 8000)
            self.image_width.setSingleStep(200)
            self.image_width.setValue(1600)
            self.image_width.setSuffix(" px")
            self.image_width.setToolTip(
                "PNG output width in pixels (height keeps the plot's "
                "aspect ratio). Ignored for SVG, which is scale-free."
            )
            form.addRow("Image width", self.image_width)
            self.white_bg = QCheckBox("White background", style_box)
            self.white_bg.setToolTip(
                "Render on white with dark axes (publication style) "
                "instead of the current theme."
            )
            form.addRow("", self.white_bg)
            style_note = QLabel(
                "Line/marker styling applies to the curve plots; the "
                "waterfall and mean-image heatmaps are unaffected.",
                style_box,
            )
            style_note.setWordWrap(True)
            form.addRow(style_note)
            layout.addWidget(style_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._update_ok()

    def _update_ok(self) -> None:
        self._ok.setEnabled(
            any(c.isChecked() for c in self._view_checks.values())
        )

    def selected_views(self) -> list:
        return [k for k, c in self._view_checks.items() if c.isChecked()]

    def selected_structures(self):
        """Structure keys to draw, or None when there is no structure
        group (no matching yet -> nothing to filter)."""
        if not self._struct_checks:
            return None
        return {k for k, c in self._struct_checks.items() if c.isChecked()}

    def style(self) -> dict:
        return {
            "line_width": float(self.line_width.value()),
            "line_style": self.line_style.currentData(),
            "marker_size": int(self.marker_size.value()),
            "width_px": int(self.image_width.value()),
            "white_bg": self.white_bg.isChecked(),
        }


class _TrackSelectDialog(QDialog):
    """Per-structure track picker for the amplitude evolution.

    Modeless: one sortable table per structure (tabs), a checkbox per
    track, applied to the host window LIVE on every toggle so the bands
    and medians update next to the dialog. Rebuilt fresh on every open;
    the host closes it when the payload shape changes.

    When the host knows its scan (``set_context`` ran), a non-interactive
    preview panel sits next to the tables: the scan frame under a frame
    slider, with the SELECTED row's track ringed on the current frame and
    its full trajectory drawn faintly. Clicking a row jumps the slider to
    that track's first frame, so judging faint/spurious tracks is a
    matter of looking, not of reading numbers.
    """

    _COLUMNS = (
        "Use", "Track", "Frames", "First", "Last",
        "Mean |q|", "Mean amplitude",
    )

    def __init__(self, host: "PhaseViewsWindow") -> None:
        super().__init__(host)
        self._host = host
        self.setWindowTitle("Amplitude evolution: select tracks")
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        outer.addLayout(left, 1)
        note = QLabel(
            "Untick tracks to drop them from their structure's grouped "
            "band and median (amplitude display + CSV export only). "
            "Click a column header to sort.", self,
        )
        note.setWordWrap(True)
        left.addWidget(note)
        self._tabs = QTabWidget(self)
        left.addWidget(self._tabs, 1)
        self._tables: dict = {}
        self._tab_keys: list = []
        # Preview state. ``_preview`` stays None on a context-less host
        # (bare test windows) and every preview code path guards on it.
        self._preview = None
        self._preview_axes = None
        self._selected_track: int | None = None
        self._selected_key = None
        for key in host._structure_keys():
            self._add_structure_tab(key)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        left.addWidget(buttons)
        if host._n_frames > 0:
            self._build_preview(outer)
            self.resize(1000, 460)
            self._tabs.currentChanged.connect(self._on_tab_changed)
            # Never open onto an empty preview: pre-select the first
            # row of the first tab (jumps the slider to its track).
            if self._tab_keys:
                table = self._tables[self._tab_keys[0]]
                if table.rowCount():
                    table.selectRow(0)
        else:
            self.resize(560, 420)

    # ---------------------- preview panel ---------------------- #
    def _build_preview(self, outer: QHBoxLayout) -> None:
        host = self._host
        panel = QVBoxLayout()
        outer.addLayout(panel, 1)
        cap = QLabel("Selected track on the scan:", self)
        panel.addWidget(cap)
        plot = pg.PlotWidget(self)
        plot.setLabel("bottom", "q_xy")
        plot.setLabel("left", "q_z")
        pi = plot.getPlotItem()
        vb = pi.getViewBox()
        vb.setAspectLocked(True)
        # Display-only: no pan/zoom/menu — the slider is the only input.
        vb.setMouseEnabled(False, False)
        pi.setMenuEnabled(False)
        pi.hideButtons()
        self._preview_plot = plot
        img = pg.ImageItem()
        img.setZValue(-10)
        host._apply_image_cmap(img)
        plot.addItem(img)
        self._preview_image = img
        missing = pg.TextItem(
            "(no image)", color=(128, 128, 128), anchor=(0.5, 0.5)
        )
        missing.setVisible(False)
        plot.addItem(missing)
        self._preview_missing = missing
        # Trajectory under the markers: the whole track as a faint line
        # so the path stays visible while scrubbing frames.
        self._preview_traj = plot.plot([], [])
        self._preview_marker = plot.plot([], [])
        panel.addWidget(plot, 1)
        row = QHBoxLayout()
        self.preview_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.preview_slider.setRange(0, max(0, host._n_frames - 1))
        self.preview_slider.valueChanged.connect(
            self._on_preview_frame_changed
        )
        row.addWidget(self.preview_slider, 1)
        self._preview_frame_label = QLabel("", self)
        row.addWidget(self._preview_frame_label)
        panel.addLayout(row)
        # Debounce the actual file read: scrubbing updates the label and
        # the overlay immediately, the frame image follows 150 ms after
        # the slider settles.
        self._preview_timer = make_debounced_timer(
            self, 150, self._do_load_preview_frame
        )
        self._preview = panel
        self._update_preview_frame_label()
        self._preview_timer.start()  # initial frame 0 load

    def _update_preview_frame_label(self) -> None:
        self._preview_frame_label.setText(
            f"Frame {self.preview_slider.value()} / "
            f"{max(0, self._host._n_frames - 1)}"
        )

    def _on_preview_frame_changed(self, _value: int) -> None:
        self._update_preview_frame_label()
        self._update_preview_overlay()
        self._preview_timer.start()

    def _do_load_preview_frame(self) -> None:
        """Read the slider's frame from the host's file and show it.

        Any failure (missing file, pipeline holding the file, bad
        entry) degrades to a "(no image)" placeholder — the overlay and
        the tables keep working without the backdrop.
        """
        if self._preview is None:
            return
        host = self._host
        if host._busy_probe is not None:
            try:
                if host._busy_probe():
                    return  # pipeline writing; keep the last image
            except Exception:
                pass
        i = int(self.preview_slider.value())
        try:
            import h5py

            from mlgidgui import file_model

            with h5py.File(host._file_path, "r") as f:
                data = f[host._entry]["data"]
                signal = file_model.read_signal_attr(data)
                frame = np.asarray(data[signal][i], dtype=np.float32)
                if self._preview_axes is None:
                    self._preview_axes = file_model.read_q_axes(data)
        except Exception as exc:
            self._preview_image.setVisible(False)
            self._show_preview_placeholder(type(exc).__name__)
            return
        self._preview_missing.setVisible(False)
        self._preview_image.setImage(_log_display(frame).T, autoLevels=True)
        q_xy, q_z = self._preview_axes
        x0, x1 = float(np.min(q_xy)), float(np.max(q_xy))
        y0, y1 = float(np.min(q_z)), float(np.max(q_z))
        self._preview_image.setRect(x0, y0, x1 - x0, y1 - y0)
        self._preview_image.setVisible(True)

    def _show_preview_placeholder(self, reason: str) -> None:
        vb = self._preview_plot.getPlotItem().getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        self._preview_missing.setText(f"(no image: {reason})")
        self._preview_missing.setPos((x0 + x1) / 2, (y0 + y1) / 2)
        self._preview_missing.setVisible(True)

    def _on_tab_changed(self, index: int) -> None:
        if 0 <= index < len(self._tab_keys):
            self._on_row_selected(self._tab_keys[index])

    def _on_row_selected(self, key) -> None:
        """Highlight the active tab's selected track and jump to its
        first frame. Bound per-table; only the ACTIVE tab drives the
        preview."""
        if self._preview is None:
            return
        index = self._tabs.currentIndex()
        if not (0 <= index < len(self._tab_keys)) or (
            self._tab_keys[index] != key
        ):
            return
        table = self._tables[key]
        row = table.currentRow()
        if row < 0 or table.item(row, 0) is None:
            self._selected_track = None
            self._selected_key = None
            self._update_preview_overlay()
            return
        k = int(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        self._selected_track = k
        self._selected_key = key
        # Jump to the first frame of the row's STRUCTURE SUBSET, not of
        # the whole track: with per-frame phase attribution a track is
        # often fitted frames before matching claims it, and the whole-
        # track first frame would land where this tab's ring cannot show
        # (and disagree with the row's "First" column).
        mem = self._selected_members()
        first = int(np.min(self._host._payload.frame_num[mem]))
        # setValue triggers the overlay + debounced image load; when the
        # slider is already there the explicit call below covers it.
        self.preview_slider.setValue(first)
        self._update_preview_overlay()

    def _selected_members(self) -> np.ndarray:
        """Member indices behind the selected row: the track's subset
        for the active tab's structure, whole track as a fallback."""
        payload = self._host._payload
        mem = self._host._member_subsets(
            self._selected_track, interval=False
        ).get(self._selected_key)
        if mem is None or not len(mem):
            mem = payload.track_members(self._selected_track)
        return np.asarray(mem)

    def _update_preview_overlay(self) -> None:
        if self._preview is None:
            return
        payload = self._host._payload
        k = self._selected_track
        if k is None or payload is None:
            self._preview_traj.setData([], [])
            self._preview_marker.setData([], [])
            return
        mem = self._selected_members()
        color = self._host._structure_color(self._selected_key)
        faint = pg.mkColor(color)
        faint.setAlpha(110)
        order = np.argsort(payload.frame_num[mem], kind="stable")
        tr = mem[order]
        self._preview_traj.setData(
            payload.q_xy[tr], payload.q_z[tr],
            pen=pg.mkPen(faint, width=1),
        )
        frame = int(self.preview_slider.value())
        on_frame = mem[payload.frame_num[mem] == frame]
        self._preview_marker.setData(
            payload.q_xy[on_frame], payload.q_z[on_frame],
            pen=None, symbol="o", symbolSize=14,
            symbolPen=pg.mkPen(color, width=2), symbolBrush=None,
        )

    def _add_structure_tab(self, key) -> None:
        host = self._host
        payload = host._payload
        radius = payload.axis_values("radius")
        rows = []
        for k in range(payload.n_tracks):
            mem = host._member_subsets(k, interval=False).get(key)
            if mem is None or not len(mem):
                continue
            frames = payload.frame_num[mem]
            rows.append((
                k, len(mem), int(frames.min()), int(frames.max()),
                float(np.nanmean(radius[mem])),
                float(np.nanmean(payload.amplitude[mem])),
            ))
        page = QWidget(self)
        box = QVBoxLayout(page)
        table = QTableWidget(len(rows), len(self._COLUMNS), page)
        table.setHorizontalHeaderLabels(list(self._COLUMNS))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        flags = (
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        )
        excluded = host._amp_excluded.get(key, set())
        for r, (k, n, first, last, mean_q, mean_amp) in enumerate(rows):
            use = QTableWidgetItem()
            use.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(
                Qt.CheckState.Unchecked if k in excluded
                else Qt.CheckState.Checked
            )
            # The track index rides the item so sorting can never
            # desync a checkbox from its track.
            use.setData(Qt.ItemDataRole.UserRole, int(k))
            table.setItem(r, 0, use)
            numeric = (
                int(k), int(n), int(first), int(last),
                round(mean_q, 4), round(mean_amp, 4),
            )
            for c, value in enumerate(numeric, start=1):
                item = QTableWidgetItem()
                # EditRole data -> numeric (not lexicographic) sorting.
                item.setData(Qt.ItemDataRole.EditRole, value)
                item.setFlags(flags)
                table.setItem(r, c, item)
        table.setSortingEnabled(True)
        table.sortItems(1)
        table.resizeColumnsToContents()
        table.itemChanged.connect(
            lambda item, key=key: self._on_item_changed(key, item)
        )
        table.itemSelectionChanged.connect(
            lambda key=key: self._on_row_selected(key)
        )
        box.addWidget(table)
        btns = QHBoxLayout()
        all_btn = QPushButton("All", page)
        all_btn.clicked.connect(
            lambda _=False, key=key: self._set_all(key, True)
        )
        btns.addWidget(all_btn)
        none_btn = QPushButton("None", page)
        none_btn.clicked.connect(
            lambda _=False, key=key: self._set_all(key, False)
        )
        btns.addWidget(none_btn)
        btns.addStretch(1)
        box.addLayout(btns)
        label = "unmatched" if key == _UNMATCHED else str(key)
        self._tabs.addTab(page, f"{label} ({len(rows)})")
        self._tables[key] = table
        self._tab_keys.append(key)

    def _on_item_changed(self, key, item) -> None:
        if item.column() != 0:
            return
        track = item.data(Qt.ItemDataRole.UserRole)
        if track is None:
            return
        excluded = self._host._amp_excluded.setdefault(key, set())
        if item.checkState() == Qt.CheckState.Checked:
            excluded.discard(int(track))
        else:
            excluded.add(int(track))
        self._host._refresh_amplitude()
        self._host._update_amp_tracks_button()

    def _set_all(self, key, checked: bool) -> None:
        table = self._tables[key]
        # Batch: one refresh at the end, not one per row.
        table.blockSignals(True)
        state = (
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        excluded = self._host._amp_excluded.setdefault(key, set())
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            item.setCheckState(state)
            track = int(item.data(Qt.ItemDataRole.UserRole))
            (excluded.discard if checked else excluded.add)(track)
        table.blockSignals(False)
        self._host._refresh_amplitude()
        self._host._update_amp_tracks_button()
