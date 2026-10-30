"""Small dialogs owned by the main window: peak export and app settings.
Moved out of ``main_window`` in the 2026 source split.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from mlgidgui import file_model, peak_lists
from mlgidgui.widgets import skin_item_view

import logging

logger = logging.getLogger(__name__)

from mlgidgui.main_window_constants import (
    DEFAULT_PROFILE_MASK_NEIGHBOURS,
    DEFAULT_SHOW_FIT_AREA,
    DEFAULT_ZOOM_TO_SELECTED_PEAK,
    SHOW_FIT_AREA_KEY,
    ZOOM_TO_SELECTED_PEAK_KEY,
    PEAK_LINK_KEY,
    PROFILE_MASK_NEIGHBOURS_KEY,
    PEAK_LINK_REVERSE_KEY,
    PLAYBACK_FRAME_MS_KEY,
    PLAYBACK_MODE_KEY,
    PLAYBACK_TOTAL_S_KEY,
    DEFAULT_PLAYBACK_FRAME_MS,
    DEFAULT_PLAYBACK_TOTAL_S,
    PLAYBACK_FRAME_MS_MAX,
    PLAYBACK_FRAME_MS_MIN,
    PLAYBACK_MODE_FRAME,
    PLAYBACK_MODE_TOTAL,
    PLAYBACK_TOTAL_S_MAX,
    PLAYBACK_TOTAL_S_MIN,
)
from mlgidgui.main_window_build import RAIL_RUN_OPENS_DOCK_KEY
from mlgidgui.peak_link import link_enabled, read_bool


class _ExportPeaksDialog(QDialog):
    """Modal kind/scope picker for Tools → Export peaks as CSV.

    Two QButtonGroups hold the kind (Detected/Fitted/Matched) and the
    scope (Active frame / Active entry / All entries) respectively;
    Active-frame is greyed when the active stack has only one frame
    so the option doesn't masquerade as different from Active-entry.
    """

    def __init__(self, parent: QWidget, *, has_multiple_frames: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export peaks as CSV")
        layout = QVBoxLayout(self)

        kind_box = QGroupBox("Peak kind")
        kind_layout = QVBoxLayout(kind_box)
        self._rb_detected = QRadioButton("Detected")
        self._rb_fitted = QRadioButton("Fitted (with fit errors)")
        self._rb_matched = QRadioButton("Matched (flattened: one row per peak)")
        self._rb_fitted.setChecked(True)
        for rb in (self._rb_detected, self._rb_fitted, self._rb_matched):
            kind_layout.addWidget(rb)
        layout.addWidget(kind_box)

        scope_box = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_box)
        self._rb_frame = QRadioButton("Active frame")
        self._rb_entry = QRadioButton("Active entry (all frames)")
        self._rb_all = QRadioButton("All entries (one combined CSV)")
        self._rb_entry.setChecked(True)
        if not has_multiple_frames:
            self._rb_frame.setEnabled(False)
            self._rb_frame.setToolTip(
                "Available only when the active entry has more than one frame."
            )
        for rb in (self._rb_frame, self._rb_entry, self._rb_all):
            scope_layout.addWidget(rb)
        layout.addWidget(scope_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_kind(self) -> str:
        if self._rb_detected.isChecked():
            return "detected"
        if self._rb_matched.isChecked():
            return "matched"
        return "fitted"

    def selected_scope(self) -> str:
        if self._rb_frame.isChecked():
            return "frame"
        if self._rb_all.isChecked():
            return "all"
        return "entry"


# Playback settings persisted via QSettings under the keys below. The
# defaults give a 2× speed-up over the previous fixed 100 ms interval
# while still leaving headroom for cold-cache disk reads (~70-100 ms
# per fresh frame on local SSD). Users who want true frame-by-frame
# stepping can dial Frame interval up; users who want a fixed total
# duration (e.g. 5 s overview regardless of frame count) can flip to
# Total play time.


def _describe(text: str) -> QLabel:
    """One muted explanatory line under an option, so what it does is
    readable without hovering for the tooltip. Indented to sit under the
    checkbox text rather than the checkbox."""
    label = QLabel(text)
    label.setProperty("status", "muted")
    label.setWordWrap(True)
    label.setContentsMargins(24, 0, 0, 0)
    return label


def _option(layout: QVBoxLayout, box: QCheckBox, description: str) -> None:
    layout.addWidget(box)
    layout.addWidget(_describe(description))


class _SettingsDialog(QDialog):
    """Application-wide settings dialog.

    Three tabs so nothing needs scrolling: **Display** (image viewer,
    profiles, frame playback), **Peaks** (peak editing, extra peak
    lists) and **Interface** (workflow strip). Adding a section is
    appending another ``QGroupBox`` to the right tab's layout; each
    option is a checkbox with a muted one-line description under it
    (``_option``), the longer explanation staying in the tooltip.

    On accept, every changed value is written back to QSettings and
    the host MainWindow is told to re-apply (so an in-flight
    playback timer picks up the new interval immediately).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)

        outer = QVBoxLayout(self)
        outer.setSpacing(12)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        def _page(title: str) -> QVBoxLayout:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(14)
            self._tabs.addTab(page, title)
            return layout

        display_page = _page("Display")
        peaks_page = _page("Peaks")
        interface_page = _page("Interface")

        # --- Playback section -------------------------------------------------
        playback_box = QGroupBox("Frame playback")
        playback_layout = QVBoxLayout(playback_box)

        hint = _describe("Controls the speed of the Display-dock Play button.")
        hint.setContentsMargins(0, 0, 0, 0)
        playback_layout.addWidget(hint)

        # Two mutually-exclusive modes. The active radio's spinbox is
        # the one that takes effect; the other stays editable so the
        # user can flip between modes without losing their values.
        mode_box = QButtonGroup(self)
        mode_box.setExclusive(True)
        self._rb_frame = QRadioButton("Time per frame")
        self._rb_total = QRadioButton("Total play time")
        mode_box.addButton(self._rb_frame)
        mode_box.addButton(self._rb_total)
        self._rb_frame.toggled.connect(self._refresh_enabled)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._spin_frame_ms = QSpinBox()
        self._spin_frame_ms.setRange(
            PLAYBACK_FRAME_MS_MIN, PLAYBACK_FRAME_MS_MAX
        )
        self._spin_frame_ms.setSingleStep(10)
        self._spin_frame_ms.setSuffix(" ms")
        self._spin_frame_ms.setToolTip(
            "Time spent on each frame. Lower = faster playback. The 10 ms "
            "lower bound caps playback at 100 fps; cold-cache disk reads "
            "may stutter below ~50 ms on large files."
        )

        self._spin_total_s = QDoubleSpinBox()
        self._spin_total_s.setRange(
            PLAYBACK_TOTAL_S_MIN, PLAYBACK_TOTAL_S_MAX
        )
        self._spin_total_s.setSingleStep(0.5)
        self._spin_total_s.setDecimals(2)
        self._spin_total_s.setSuffix(" s")
        self._spin_total_s.setToolTip(
            "Total time to traverse the whole stack (first frame → last "
            "frame). The per-frame interval is computed at play-start "
            "from the active entry's frame count, so swapping entries "
            "automatically adjusts the speed."
        )

        form.addRow(self._rb_frame, self._spin_frame_ms)
        form.addRow(self._rb_total, self._spin_total_s)
        playback_layout.addLayout(form)

        # --- Peak editing section ---------------------------------------------
        peaks_box = QGroupBox("Peak editing")
        peaks_layout = QVBoxLayout(peaks_box)

        peaks_hint = _describe(
            "How fitted peaks relate to the detected peaks they come from."
        )
        peaks_hint.setContentsMargins(0, 0, 0, 0)
        peaks_layout.addWidget(peaks_hint)

        self._chk_peak_link = QCheckBox(
            "One fitted peak per detected peak"
        )
        self._chk_peak_link.setToolTip(
            "A fit made from a detected peak is stored under that peak's "
            "id, so fitting it again replaces the fit instead of adding a "
            "second one, and deleting the detected peak deletes its fit. "
            "A hand-drawn box is added to the detected peaks first, so "
            "every fitted peak has a detected partner.\n\n"
            "Turn this off to keep the two tables fully independent, as "
            "they were before this option existed."
        )
        self._chk_peak_link_reverse = QCheckBox(
            "Deleting a fitted peak also deletes its detected peak"
        )
        self._chk_peak_link_reverse.setToolTip(
            "Off by default: discarding a fit usually means the "
            "prediction was wrong, not that the detection was. Needs the "
            "option above, since without paired ids there is no partner "
            "to delete."
        )
        self._chk_peak_link.toggled.connect(self._refresh_enabled)
        _option(
            peaks_layout, self._chk_peak_link,
            "Fitting a detected peak again replaces its fit instead of "
            "adding a second one.",
        )
        _option(
            peaks_layout, self._chk_peak_link_reverse,
            "Off by default. Needs the option above.",
        )

        # --- Extra peak lists -------------------------------------------------
        lists_box = QGroupBox("Extra peak lists")
        lists_layout = QVBoxLayout(lists_box)
        for paragraph in (
            "Show another peak table from a file's analysis group as its "
            "own layer in the Display dock. A display layer only: you can "
            "see it, pick a box and nudge it, and nothing else in the app "
            "reads it.",
            "<b>Pipeline primary</b> is the one exception, and it is "
            "niche. Tick it and Detection and Fitting read and write that "
            "table instead of the standard one for its flavour, so you can "
            "build a second analysis without touching the first. One "
            "table per flavour.",
            "Everything else still uses the standard tables: the Peaks "
            "table, CSV export, peak tracking, and matching. With a "
            "primary Fitted table, matching keeps matching a table the "
            "pipeline is no longer updating.",
        ):
            lists_hint = _describe(paragraph)
            lists_hint.setContentsMargins(0, 0, 0, 0)
            lists_layout.addWidget(lists_hint)

        self._lists_table = QTableWidget(0, 4)
        self._lists_table.setHorizontalHeaderLabels(
            ["Dataset", "Shown as", "Treat as", "Pipeline primary"]
        )
        self._lists_table.verticalHeader().setVisible(False)
        self._lists_table.horizontalHeader().setStretchLastSection(True)
        self._lists_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        skin_item_view(self._lists_table)
        self._lists_table.setMinimumHeight(90)
        lists_layout.addWidget(self._lists_table)

        # What the OPEN FILE actually has, so the dataset is picked
        # rather than typed -- a name that does not exist would register
        # a layer that never appears, with nothing to say why.
        self._available_datasets = self._scan_available_datasets(parent)
        lists_btns = QHBoxLayout()
        self._btn_add_list = QPushButton("Add\u2026")
        self._btn_add_list.clicked.connect(self._add_list_row)
        self._btn_remove_list = QPushButton("Remove")
        self._btn_remove_list.clicked.connect(self._remove_list_row)
        lists_btns.addWidget(self._btn_add_list)
        lists_btns.addWidget(self._btn_remove_list)
        lists_btns.addStretch(1)
        lists_layout.addLayout(lists_btns)

        self._lists_empty_hint = QLabel("")
        self._lists_empty_hint.setProperty("status", "muted")
        self._lists_empty_hint.setWordWrap(True)
        lists_layout.addWidget(self._lists_empty_hint)

        # --- Workflow strip ---------------------------------------------------
        rail_box = QGroupBox("Workflow strip")
        rail_layout = QVBoxLayout(rail_box)
        self._chk_rail_run_opens_dock = QCheckBox(
            "Open a stage's dock when its play button is pressed"
        )
        self._chk_rail_run_opens_dock.setToolTip(
            "The play button next to a stage on the strip above the image "
            "runs that stage. With this on it also brings the dock that "
            "owns the stage forward, so the run's settings and progress "
            "are in view. Clicking a stage's name always does."
        )
        _option(
            rail_layout, self._chk_rail_run_opens_dock,
            "The play button next to a stage also brings that stage's "
            "dock forward. Clicking a stage's name always does.",
        )

        # --- Profiles ---------------------------------------------------------
        profile_box = QGroupBox("Profiles")
        profile_layout = QVBoxLayout(profile_box)
        self._chk_profile_mask = QCheckBox(
            "Mask the other peaks out of the selected peak's profiles"
        )
        self._chk_profile_mask.setToolTip(
            "Sets the boxes of the frame's other detected and fitted peaks "
            "to NaN before the radial and angular profiles are averaged, "
            "so the profiles show the selected peak and not its "
            "neighbours. This is what pygidfit does to the other boxes "
            "when it fits a peak: a neighbour masks its radius +- "
            "radius_width by angle +- angle_width / 2. Neighbours "
            "within pygidfit's clustering distance (the Pipeline panel's "
            "Cluster dist (peaks), 2 px of its polar grid by default) "
            "are left in, as pygidfit fits those jointly, and "
            "rings are not masked. The profile titles say when masking "
            "is active."
        )
        _option(
            profile_layout, self._chk_profile_mask,
            "The profiles show the selected peak and not its neighbours, "
            "as pygidfit sees it when it fits. The profile titles say when "
            "this is active.",
        )

        # --- Image viewer -----------------------------------------------------
        view_box = QGroupBox("Image viewer")
        view_layout = QVBoxLayout(view_box)
        self._chk_zoom_peak = QCheckBox("Zoom to a peak when it is selected")
        self._chk_zoom_peak.setToolTip(
            "Selecting a detected or fitted peak (in the image or in the "
            "Peaks table) zooms the image onto its box, so the box fills "
            "about the middle third of the view. Rings are never zoomed, "
            "and neither are manual boxes you are still drawing. Reset "
            "zoom brings the full image back."
        )
        _option(
            view_layout, self._chk_zoom_peak,
            "Detected and fitted peaks only; rings stay where they are. "
            "Reset zoom returns to the full image.",
        )
        self._chk_fit_area = QCheckBox(
            "Outline the area pygidFIT fits around a selected detected peak"
        )
        self._chk_fit_area.setToolTip(
            "A dashed red box around the selected detected peak shows the "
            "region pygidfit really fits for it. That region is larger than "
            "the box drawn: radially twice as wide (pygidfit uses radius "
            "+- radius_width where the box is drawn as radius +- "
            "radius_width / 2), the same angular range, and one pixel of "
            "its polar grid more on every side. It is why a fitted peak "
            "can land outside its detected box. Fitted, matched and "
            "manual selections do not show it."
        )
        _option(
            view_layout, self._chk_fit_area,
            "A dashed red box, larger than the drawn one: the area that is "
            "actually fitted. Detected peaks only.",
        )

        # --- Lay the sections out on their tabs --------------------------------
        for box in (view_box, profile_box, playback_box):
            display_page.addWidget(box)
        display_page.addStretch(1)
        for box in (peaks_box, lists_box):
            peaks_page.addWidget(box)
        peaks_page.addStretch(1)
        interface_page.addWidget(rail_box)
        interface_page.addStretch(1)

        # --- Buttons + outer wiring -------------------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Load current values from QSettings (with defaults).
        settings = QSettings()
        mode = settings.value(
            PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME
        )
        # QSettings returns strings on Linux but raw types on macOS;
        # coerce defensively.
        try:
            frame_ms = int(settings.value(
                PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
            ))
        except (TypeError, ValueError):
            frame_ms = DEFAULT_PLAYBACK_FRAME_MS
        try:
            total_s = float(settings.value(
                PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
            ))
        except (TypeError, ValueError):
            total_s = DEFAULT_PLAYBACK_TOTAL_S
        # Clamp into the spinbox range so out-of-bounds stored values
        # don't silently revert to the spinbox minimum.
        frame_ms = max(PLAYBACK_FRAME_MS_MIN,
                       min(PLAYBACK_FRAME_MS_MAX, frame_ms))
        total_s = max(PLAYBACK_TOTAL_S_MIN,
                      min(PLAYBACK_TOTAL_S_MAX, total_s))
        self._spin_frame_ms.setValue(frame_ms)
        self._spin_total_s.setValue(total_s)
        if mode == PLAYBACK_MODE_TOTAL:
            self._rb_total.setChecked(True)
        else:
            self._rb_frame.setChecked(True)
        # ``reverse_delete_enabled()`` folds in the link, which would
        # show the box unticked whenever the link is off even though the
        # user's own choice was to tick it. Read the raw key so the
        # dialog reflects what is stored, and let the enabled-state
        # carry the dependency.
        self._chk_peak_link.setChecked(link_enabled())
        self._chk_peak_link_reverse.setChecked(
            read_bool(PEAK_LINK_REVERSE_KEY, False)
        )
        self._chk_rail_run_opens_dock.setChecked(
            read_bool(RAIL_RUN_OPENS_DOCK_KEY, True)
        )
        self._chk_zoom_peak.setChecked(
            read_bool(
                ZOOM_TO_SELECTED_PEAK_KEY, DEFAULT_ZOOM_TO_SELECTED_PEAK,
            )
        )
        self._chk_profile_mask.setChecked(
            read_bool(
                PROFILE_MASK_NEIGHBOURS_KEY, DEFAULT_PROFILE_MASK_NEIGHBOURS,
            )
        )
        self._chk_fit_area.setChecked(
            read_bool(SHOW_FIT_AREA_KEY, DEFAULT_SHOW_FIT_AREA)
        )
        self._load_list_rows()
        self._refresh_enabled()

    # --- Layout ---

    def _fit_wrapped_labels(self) -> None:
        """Give every word-wrapped label the height its text needs at its
        current width.

        A wrapped ``QLabel`` inside a ``QTabWidget`` page reports a
        height computed before the width is known, so a long hint is
        cut off after its first lines. Once the dialog has a width the
        real height is known; this applies it.
        """
        for label in self.findChildren(QLabel):
            if label.wordWrap() and label.width() > 0:
                label.setMinimumHeight(label.heightForWidth(label.width()))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_wrapped_labels()
        self.adjustSize()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_wrapped_labels()

    # --- Extra peak lists ---

    def _scan_available_datasets(self, parent) -> list[str]:
        """Peak-shaped tables in the open file's current frame.

        Read once when the dialog opens: it is a handful of names from
        an already-open handle, and re-scanning per keystroke would put
        file I/O behind a combo box. Any failure yields an empty list --
        the section then explains itself and Add stays disabled, which
        is better than a dialog that will not open.
        """
        try:
            session = getattr(parent, "session", None)
            if session is None:
                return []
            entry = parent.entry_combo.currentText()
            if not entry:
                return []
            frame = int(parent.viewer.current_frame)
            source = parent.viewer._frame_source
            if source is not None and source.is_open and source.entry == entry:
                return file_model.list_peak_tables(
                    source._file, entry, frame
                )
            import h5py

            with h5py.File(session.temp_path, "r") as f:
                return file_model.list_peak_tables(f, entry, frame)
        except Exception:
            logger.debug("could not list this frame's peak tables",
                         exc_info=True)
            return []

    def _load_list_rows(self) -> None:
        self._lists_table.setRowCount(0)
        for spec in peak_lists.load_specs():
            self._append_list_row(
                spec.dataset, spec.label, spec.treat_as, spec.primary,
            )
        self._refresh_lists_hint()

    def _primary_box(self, row: int):
        """The Pipeline-primary checkbox on ``row`` (None if absent)."""
        widget = self._lists_table.cellWidget(row, 3)
        return widget if isinstance(widget, QCheckBox) else None

    def _treat_of_row(self, row: int) -> str:
        widget = self._lists_table.cellWidget(row, 2)
        if isinstance(widget, QComboBox):
            return widget.currentData() or peak_lists.TREAT_DETECTED
        return peak_lists.TREAT_DETECTED

    def _enforce_single_primary(self, row: int) -> None:
        """Keep at most one primary per flavour.

        Ticking a row unticks every other row of the same flavour: two
        primaries for one standard table would make the swap ambiguous,
        and a radio-like tick says that better than a validation error
        on OK would.
        """
        box = self._primary_box(row)
        if box is None or not box.isChecked():
            return
        flavour = self._treat_of_row(row)
        for other in range(self._lists_table.rowCount()):
            if other == row:
                continue
            other_box = self._primary_box(other)
            if (
                other_box is not None
                and other_box.isChecked()
                and self._treat_of_row(other) == flavour
            ):
                other_box.setChecked(False)

    def _on_treat_changed(self, row: int) -> None:
        """Changing a row's flavour drops its tick.

        The tick means "primary for THIS flavour"; carrying it across a
        flavour change could silently produce a second primary for the
        new one. Clearing is the rule with no surprising case.
        """
        box = self._primary_box(row)
        if box is not None and box.isChecked():
            box.setChecked(False)

    def _refresh_lists_hint(self) -> None:
        """Explain an empty Add rather than just greying it out."""
        registered = {
            self._dataset_of_row(r) for r in range(self._lists_table.rowCount())
        }
        remaining = [
            name for name in self._available_datasets
            if name not in registered
        ]
        self._btn_add_list.setEnabled(bool(remaining))
        if not self._available_datasets:
            self._lists_empty_hint.setText(
                "Open a file whose current frame has an extra peak table "
                "to add one."
            )
        elif not remaining:
            self._lists_empty_hint.setText(
                "Every extra table in this frame is already listed."
            )
        else:
            self._lists_empty_hint.setText("")

    def _dataset_of_row(self, row: int) -> str:
        widget = self._lists_table.cellWidget(row, 0)
        if isinstance(widget, QComboBox):
            return widget.currentText()
        item = self._lists_table.item(row, 0)
        return item.text() if item is not None else ""

    def _append_list_row(
        self, dataset: str, label: str, treat_as: str, primary: bool = False,
    ) -> None:
        row = self._lists_table.rowCount()
        self._lists_table.insertRow(row)
        # A dataset already registered keeps a plain (read-only) cell:
        # the file it came from may not be the one open now, and turning
        # it into a combo would offer to silently repoint the layer.
        item = QTableWidgetItem(dataset)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._lists_table.setItem(row, 0, item)
        self._lists_table.setCellWidget(row, 1, QLineEdit(label))
        combo = QComboBox()
        for value in peak_lists.TREAT_AS:
            combo.addItem(value.capitalize(), value)
        index = combo.findData(treat_as)
        combo.setCurrentIndex(max(index, 0))
        self._lists_table.setCellWidget(row, 2, combo)
        self._install_primary_cell(row, combo, primary)

    def _install_primary_cell(self, row: int, treat_combo, primary: bool) -> None:
        """The Pipeline-primary tick, plus the two rules that keep it sane.

        Both connections resolve the row through the widget rather than
        capturing ``row``: rows shift when one above is removed, and a
        captured index would then point at the wrong table.
        """
        box = QCheckBox()
        box.setChecked(bool(primary))
        box.setToolTip(
            "Detection and Fitting read and write this table instead of "
            "the standard one for its flavour. Nothing else in the app "
            "changes."
        )
        box.toggled.connect(
            lambda _on, w=box: self._enforce_single_primary(
                self._row_of_widget(w, 3)
            )
        )
        treat_combo.currentIndexChanged.connect(
            lambda _i, w=treat_combo: self._on_treat_changed(
                self._row_of_widget(w, 2)
            )
        )
        self._lists_table.setCellWidget(row, 3, box)

    def _row_of_widget(self, widget, column: int) -> int:
        for row in range(self._lists_table.rowCount()):
            if self._lists_table.cellWidget(row, column) is widget:
                return row
        return -1

    def _add_list_row(self) -> None:
        registered = {
            self._dataset_of_row(r) for r in range(self._lists_table.rowCount())
        }
        remaining = [
            name for name in self._available_datasets
            if name not in registered
        ]
        if not remaining:
            return
        row = self._lists_table.rowCount()
        self._lists_table.insertRow(row)
        combo = QComboBox()
        combo.addItems(remaining)
        combo.currentTextChanged.connect(
            lambda _t: self._refresh_lists_hint()
        )
        self._lists_table.setCellWidget(row, 0, combo)
        self._lists_table.setCellWidget(row, 1, QLineEdit(remaining[0]))
        treat = QComboBox()
        for value in peak_lists.TREAT_AS:
            treat.addItem(value.capitalize(), value)
        self._lists_table.setCellWidget(row, 2, treat)
        self._install_primary_cell(row, treat, False)
        self._refresh_lists_hint()

    def _remove_list_row(self) -> None:
        rows = sorted(
            {i.row() for i in self._lists_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self._lists_table.removeRow(row)
        self._refresh_lists_hint()

    def peak_list_specs(self) -> list:
        """The registry as the dialog currently shows it."""
        out = []
        seen: set[str] = set()
        for row in range(self._lists_table.rowCount()):
            dataset = self._dataset_of_row(row).strip()
            if not dataset or dataset in seen:
                continue
            label_widget = self._lists_table.cellWidget(row, 1)
            treat_widget = self._lists_table.cellWidget(row, 2)
            primary_box = self._primary_box(row)
            out.append(peak_lists.PeakListSpec(
                dataset=dataset,
                label=(
                    label_widget.text().strip()
                    if isinstance(label_widget, QLineEdit) else ""
                ),
                treat_as=(
                    treat_widget.currentData()
                    if isinstance(treat_widget, QComboBox)
                    else peak_lists.TREAT_DETECTED
                ),
                primary=(
                    primary_box.isChecked() if primary_box is not None
                    else False
                ),
            ))
            seen.add(dataset)
        return out

    def primary_name_conflict(self) -> str:
        """A registered list a fitted primary's errors table would eat.

        A fitted run writes two datasets, so a primary ``trial2`` also
        claims ``trial2_errors``. If some other row registers exactly
        that name, the first fitting run would silently overwrite it.
        Returns the offending name, or "" when there is no clash.
        """
        specs = self.peak_list_specs()
        primary = peak_lists.primary_for(specs, peak_lists.TREAT_FITTED)
        if primary is None:
            return ""
        errors = peak_lists.errors_name(primary.dataset)
        for spec in specs:
            if spec.dataset == errors:
                return errors
        return ""

    def _on_accept(self) -> None:
        """Refuse the one combination that would eat a registered table."""
        clash = self.primary_name_conflict()
        if clash:
            QMessageBox.warning(
                self, "Extra peak lists",
                f"A fitted primary table also writes its errors table, "
                f"so it needs the name {clash!r} — which is registered "
                f"as a list of its own. Rename or remove that list, or "
                f"pick a different primary.",
            )
            return
        self.accept()

    def _refresh_enabled(self) -> None:
        frame_active = self._rb_frame.isChecked()
        self._spin_frame_ms.setEnabled(frame_active)
        self._spin_total_s.setEnabled(not frame_active)
        # The reverse cascade needs paired ids to have anything to
        # cascade to, so it greys out with the link rather than lying
        # about what it would do.
        self._chk_peak_link_reverse.setEnabled(
            self._chk_peak_link.isChecked()
        )

    def save_to_qsettings(self) -> None:
        """Write the dialog's current values to QSettings.

        Called by the host on accept. Stores both spinbox values so a
        later mode-flip preserves the user's last value in each mode.
        """
        settings = QSettings()
        mode = PLAYBACK_MODE_FRAME if self._rb_frame.isChecked() else PLAYBACK_MODE_TOTAL
        settings.setValue(PLAYBACK_MODE_KEY, mode)
        settings.setValue(
            PLAYBACK_FRAME_MS_KEY, int(self._spin_frame_ms.value())
        )
        settings.setValue(
            PLAYBACK_TOTAL_S_KEY, float(self._spin_total_s.value())
        )
        peak_lists.save_specs(self.peak_list_specs())
        settings.setValue(PEAK_LINK_KEY, bool(self._chk_peak_link.isChecked()))
        settings.setValue(
            PEAK_LINK_REVERSE_KEY,
            bool(self._chk_peak_link_reverse.isChecked()),
        )
        settings.setValue(
            RAIL_RUN_OPENS_DOCK_KEY,
            bool(self._chk_rail_run_opens_dock.isChecked()),
        )
        settings.setValue(
            ZOOM_TO_SELECTED_PEAK_KEY,
            bool(self._chk_zoom_peak.isChecked()),
        )
        settings.setValue(
            PROFILE_MASK_NEIGHBOURS_KEY,
            bool(self._chk_profile_mask.isChecked()),
        )
        settings.setValue(
            SHOW_FIT_AREA_KEY,
            bool(self._chk_fit_area.isChecked()),
        )
