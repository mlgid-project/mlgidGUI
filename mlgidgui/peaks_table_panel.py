"""Tabbed peak-table dock with bidirectional click-sync to the image.

Three tabs — Detected / Fitted / Matched — each backed by a
``QStandardItemModel`` wrapped in a ``QSortFilterProxyModel`` so
header clicks sort the rows without us writing any sort logic.
Frame scope is **current frame only**; the host calls
``set_frame_peaks(...)`` on every frame change + after pipeline
runs. The user's chosen sort column / order persists across these
refreshes (the proxy just re-sorts the new rows).

Click-sync is bidirectional:

- **Row click → image**: a row's underlying ``(kind, peak_id)``
  is stored on ``Qt.UserRole`` of the row's items; the click
  handler reconstructs a ``SelectedPeak`` from the live source
  table the row was built from and emits
  ``peakSelectedFromTable``. The host wires that to the viewer's
  selection setter.
- **Image click → row**: ``set_external_selection(sel)`` looks
  up the ``(kind, peak_id)`` in each tab's index, switches to
  the relevant tab, selects the row, scrolls into view. The
  panel blocks its own row-selection signal while applying the
  external one so the host's wiring can't bounce.
"""

from __future__ import annotations


import numpy as np
from PySide6.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QHeaderView,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlgidgui.file_model import MatchedStructure, PeakTable
from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.widgets import skin_item_view as _skin_item_view
from mlgidgui.widgets import attach_empty_hint as _attach_empty_hint


# Tab index constants. Order matches the order tabs are added in
# ``__init__`` and the kind strings used by SelectedPeak.
_TAB_DETECTED = 0
_TAB_FITTED = 1
_TAB_MATCHED = 2

_KIND_BY_TAB = {
    _TAB_DETECTED: "detected",
    _TAB_FITTED: "fitted",
    _TAB_MATCHED: "matched",
}
_TAB_BY_KIND = {v: k for k, v in _KIND_BY_TAB.items()}


# Roles on the item used to round-trip the peak identity.
_ROLE_KIND = Qt.ItemDataRole.UserRole + 1
_ROLE_PEAK_ID = Qt.ItemDataRole.UserRole + 2
_ROLE_STRUCTURE_UID = Qt.ItemDataRole.UserRole + 3
# Stored as a float on the item so QSortFilterProxyModel can sort
# numerically without us implementing a custom less-than. Display
# text is set separately via setData(role=DisplayRole).
_ROLE_NUMERIC = Qt.ItemDataRole.UserRole + 4


class _NumericItem(QStandardItem):
    """Item that sorts by its numeric ``UserRole+4`` payload.

    Default ``QStandardItem`` sorts by ``Qt.DisplayRole`` which
    compares as strings — `"10.0"` would sort before `"2.0"`. We
    stash the raw float and compare it directly.
    """

    def __lt__(self, other) -> bool:  # type: ignore[override]
        a = self.data(_ROLE_NUMERIC)
        b = other.data(_ROLE_NUMERIC)
        if a is None or b is None:
            return super().__lt__(other)
        return float(a) < float(b)


def _num_item(value: float, fmt: str = "{:.3f}") -> _NumericItem:
    """Read-only numeric cell. Display text uses ``fmt``; sort key
    is the raw float."""
    item = _NumericItem(fmt.format(float(value)))
    item.setData(float(value), _ROLE_NUMERIC)
    item.setEditable(False)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _text_item(value: str) -> QStandardItem:
    item = QStandardItem(value)
    item.setEditable(False)
    return item


def _int_item(value: int) -> _NumericItem:
    item = _NumericItem(str(int(value)))
    item.setData(float(int(value)), _ROLE_NUMERIC)
    item.setEditable(False)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _type_str(is_ring: bool) -> str:
    return "ring" if is_ring else "segment"


def _set_header(
    model: QStandardItemModel, columns: list[tuple[str, str]],
) -> None:
    """Set horizontal header labels + per-column tooltip text.

    ``columns`` is an ordered list of ``(short_label, tooltip)``
    pairs; the short label is what shows in the header, the
    tooltip surfaces on hover so the user can recover the full
    meaning of an abbreviated header.
    """
    model.setColumnCount(len(columns))
    for i, (label, tip) in enumerate(columns):
        model.setHeaderData(i, Qt.Orientation.Horizontal, label,
                            Qt.ItemDataRole.DisplayRole)
        model.setHeaderData(i, Qt.Orientation.Horizontal, tip,
                            Qt.ItemDataRole.ToolTipRole)


class _PeakTableView(QTableView):
    """Table view that reports Delete instead of swallowing it.

    ``QTableView`` consumes key presses it does not handle rather than
    letting them bubble to the dock, so a Delete pressed with the
    table focused would do nothing at all. Re-emitting it lets the
    panel run the same delete the image viewer runs for the same
    selection.
    """

    deleteRequested = Signal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.key() == Qt.Key.Key_Delete
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            self.deleteRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _PeakProxy(QSortFilterProxyModel):
    """Mixed numeric / text sort.

    ``QSortFilterProxyModel`` doesn't call the source item's ``__lt__``
    — it reads the configured sort role and compares the values
    directly. With a single SortRole pointed at ``_ROLE_NUMERIC``
    text columns (CIF, hkl, type, peak-ids list) would sort as
    invalid QVariants. Override ``lessThan`` to prefer the numeric
    payload when both items carry one, otherwise fall back to the
    text DisplayRole. This gives proper numeric sorting on radius /
    angle / score / probability columns and natural lexicographic
    sorting on text columns.
    """

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        l_num = left.data(_ROLE_NUMERIC)
        r_num = right.data(_ROLE_NUMERIC)
        if l_num is not None and r_num is not None:
            return float(l_num) < float(r_num)
        l_text = left.data(Qt.ItemDataRole.DisplayRole)
        r_text = right.data(Qt.ItemDataRole.DisplayRole)
        if l_text is None or r_text is None:
            return False
        return str(l_text) < str(r_text)


class PeaksTablePanel(QWidget):
    """Tabbed sortable peak-table dock for the current frame."""

    # Row click — emits the SelectedPeak the host should set on the
    # image viewer. Suppressed while ``set_external_selection`` is
    # applying an external change so the round-trip can't bounce.
    peakSelectedFromTable = Signal(object)
    # The full multi-selection, whenever it holds two or more rows.
    # Single-row changes keep using the signal above so the existing
    # round-trip is untouched.
    peaksSelectedFromTable = Signal(list)
    # Delete pressed with rows selected on the Detected / Fitted tab.
    # Carries the same SelectedPeak list the host's image-side delete
    # handlers take, so the panel does no deleting of its own.
    deletePeaksRequested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._tabs = QTabWidget(self)
        layout.addWidget(self._tabs)

        # Headers are kept tight (single character / abbreviated) so
        # ``resizeColumnsToContents`` ends up packing each column to
        # the width of its numeric values rather than the longer
        # English word. Full names are attached as tooltips on the
        # header sections so hovering still reveals what each column
        # is.
        # --- Detected tab ---
        self._detected_model = QStandardItemModel(self)
        _set_header(
            self._detected_model,
            [("ID",     "Peak id (sortable)"),
             ("r",      "Radius (Å⁻¹)"),
             ("Δr",     "Radius width (Å⁻¹)"),
             ("a",      "Angle (°)"),
             ("Δa",     "Angle width (°)"),
             ("score",  "mlgidDETECT confidence score"),
             ("type",   "Ring vs segment")],
        )
        self._detected_table, self._detected_proxy = self._build_table(
            self._detected_model
        )
        self._tabs.addTab(self._detected_table, "Detected")

        # --- Fitted tab ---
        self._fitted_model = QStandardItemModel(self)
        _set_header(
            self._fitted_model,
            [("ID",    "Peak id (sortable)"),
             ("r",     "Radius (Å⁻¹)"),
             ("Δr",    "Radial width = 2σ_r (Å⁻¹). Same convention as "
                      "pygidfit / pipeline; FWHM_r ≈ 1.177 × Δr."),
             ("a",     "Angle (°)"),
             ("Δa",    "Angular width = 2σ_a (°). Same convention as "
                      "pygidfit / pipeline; FWHM_a ≈ 1.177 × Δa."),
             ("amp",   "Peak amplitude (2D-Gaussian height)"),
             ("score", "mlgidDETECT confidence score"),
             ("type",  "Ring vs segment")],
        )
        self._fitted_table, self._fitted_proxy = self._build_table(
            self._fitted_model
        )
        self._tabs.addTab(self._fitted_table, "Fitted")

        # --- Matched tab ---
        # One row per *structure* (not per matched peak): a structure
        # is a (CIF, hkl) solution that explains a subset of fitted
        # peaks, and the row's "IDs" column lists every fitted-peak
        # id in that subset. Selecting the row highlights every peak
        # of the structure on the image at once.
        self._matched_model = QStandardItemModel(self)
        _set_header(
            self._matched_model,
            [("CIF",  "CIF file (no extension)"),
             ("hkl",  "Miller indices of the matched plane"),
             ("prob", "Match probability"),
             ("#",    "Number of fitted peaks in this structure"),
             ("IDs",  "Comma-separated fitted-peak ids in this structure")],
        )
        self._matched_table, self._matched_proxy = self._build_table(
            self._matched_model
        )
        # A matched row is a whole structure, not a peak: multi-select
        # and Delete both mean something different there, so the tab
        # keeps the single-select behaviour it has always had.
        self._matched_table.setSelectionMode(
            QTableView.SelectionMode.SingleSelection
        )
        self._tabs.addTab(self._matched_table, "Matched")

        # Source PeakTables for the most recent ``set_frame_peaks``
        # call. Row clicks pull live geometry from these (not from
        # the QStandardItem text) so format precision can't lose
        # information.
        self._detected_src: PeakTable | None = None
        self._fitted_src: PeakTable | None = None
        # Matched is keyed by structure_uid → (MatchedStructure, row_idx-in-its-PeakTable)
        # because the same fitted-peak id can appear in multiple
        # matched structures with different (CIF, hkl) contexts.
        self._matched_src: list[MatchedStructure] = []

        # Suppress the row-selection signal while we apply an
        # external (image-driven) selection. Otherwise the host's
        # wiring would round-trip the same selection back through
        # the viewer.
        self._applying_external_selection = False
        # And the other direction: while the panel is announcing its
        # own selection, ignore the mirror coming back. The viewer
        # emits synchronously, so without this the table would be
        # cleared and re-selected from inside its own mouse event —
        # which a Ctrl+click survives least of all, because Qt applies
        # its Toggle command on release and would toggle straight back
        # off the row we had just re-selected underneath it.
        self._emitting_own_selection = False

        # Per-tab row-selection wiring. Done once; selection-model
        # is re-fetched via ``selectionModel()`` because it changes
        # whenever the table's model is reset (it isn't here, but
        # be safe).
        for table, kind in (
            (self._detected_table, "detected"),
            (self._fitted_table, "fitted"),
            (self._matched_table, "matched"),
        ):
            table.selectionModel().currentRowChanged.connect(
                lambda cur, _prev, k=kind: self._on_row_changed(k, cur)
            )
            # currentRowChanged carries only the row the keyboard focus
            # moved to, which is the whole story for a single-select
            # tab but not for a Ctrl+click that adds a second row
            # without moving the current index. selectionChanged fills
            # that gap; it fires after currentRowChanged, so the
            # multi-selection is the last word.
            table.selectionModel().selectionChanged.connect(
                lambda _sel, _desel, k=kind: self._on_selection_changed(k)
            )
        for table, kind in (
            (self._detected_table, "detected"),
            (self._fitted_table, "fitted"),
        ):
            table.deleteRequested.connect(
                lambda k=kind: self._on_delete_requested(k)
            )

        # Default sort orders. These persist across refresh because
        # the proxy keeps its sortColumn / sortOrder between
        # repopulations — the user can override by clicking any
        # header, and that override sticks too.
        #
        # Detected / Fitted: column 0 = ID, ascending (ID 0 at top).
        # Matched: column 2 = probability, descending (best match first).
        self._detected_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._fitted_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._matched_table.sortByColumn(2, Qt.SortOrder.DescendingOrder)

    # --- Public API ---

    def set_frame_peaks(
        self,
        frame: int,
        detected: PeakTable | None,
        fitted: PeakTable | None,
        matched: list[MatchedStructure],
    ) -> None:
        """Repopulate all three tabs with the current frame's peaks.

        Sort column / order on each tab is preserved by the proxy —
        the underlying source model is repopulated, the proxy
        re-sorts the new rows automatically. Sort-state survives
        across frame changes.
        """
        self._detected_src = detected
        self._fitted_src = fitted
        self._matched_src = list(matched)

        # Repopulate inside a single block so the proxy / view only
        # re-sorts once at the end (Qt batches the dataChanged calls
        # under the hood, but we still avoid emitting per-row).
        self._fill_detected(detected)
        self._fill_fitted(fitted)
        self._fill_matched(self._matched_src)

    def set_external_selection(self, sel: SelectedPeak | None) -> None:
        """Mirror an image-driven selection onto the tables.

        - Auto-switches to the relevant tab (Detected / Fitted /
          Matched) so the user sees the corresponding row.
        - Selects + scrolls-into-view the matching row.
        - Clears the other tabs' selection so only one tab is
          ever "live".
        - Manual selections clear every tab (manual peaks have no
          table representation).
        """
        if self._emitting_own_selection:
            return
        self._applying_external_selection = True
        try:
            # Always clear all three first so a fresh selection
            # never leaves a stale highlight behind. ``current`` /
            # ``selection`` both cleared so the next row click
            # registers as a fresh change.
            for table in (
                self._detected_table, self._fitted_table, self._matched_table,
            ):
                table.selectionModel().clearCurrentIndex()
                table.selectionModel().clearSelection()

            if sel is None or sel.kind == "manual":
                return
            if sel.kind not in _TAB_BY_KIND:
                return
            tab_idx = _TAB_BY_KIND[sel.kind]
            self._tabs.setCurrentIndex(tab_idx)

            if sel.kind == "matched":
                self._select_matched_row(sel)
            else:
                self._select_simple_row(sel)
        finally:
            self._applying_external_selection = False

    def set_external_selections(self, sels: list[SelectedPeak]) -> None:
        """Mirror an image-driven *multi*-selection onto the tables.

        Same contract as :meth:`set_external_selection`, extended to the
        Ctrl+click multi-selection the image viewer supports for
        detected and fitted peaks: without this the table would show
        only the primary while the image showed five boxes, and a
        Delete pressed in the table would then act on a different set
        than the one the user can see highlighted.

        Zero or one peak defers to the single-selection path, so the
        two entry points cannot drift.
        """
        if self._emitting_own_selection:
            return
        if len(sels) < 2:
            self.set_external_selection(sels[0] if sels else None)
            return
        kind = sels[0].kind
        if kind not in ("detected", "fitted"):
            self.set_external_selection(sels[0])
            return
        self._applying_external_selection = True
        try:
            for table in (
                self._detected_table, self._fitted_table, self._matched_table,
            ):
                table.selectionModel().clearCurrentIndex()
                table.selectionModel().clearSelection()
            self._tabs.setCurrentIndex(_TAB_BY_KIND[kind])
            table = self._table_for(kind)
            model = (
                self._detected_model if kind == "detected"
                else self._fitted_model
            )
            proxy = (
                self._detected_proxy if kind == "detected"
                else self._fitted_proxy
            )
            selection_model = table.selectionModel()
            flags = (
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows
            )
            first_idx = None
            # One pass over the model and one select() call, not one of
            # each per selected peak: Ctrl+A hands over every peak on the
            # frame, and both made this quadratic (2000 peaks took
            # seconds).
            rows_by_id = _rows_by_id_in_model(model)
            selection = QItemSelection()
            for sel in sels:
                src_row = rows_by_id.get(int(sel.peak_id))
                if src_row is None:
                    continue
                proxy_idx = proxy.mapFromSource(model.index(src_row, 0))
                selection.select(proxy_idx, proxy_idx)
                if first_idx is None:
                    first_idx = proxy_idx
            if first_idx is not None:
                selection_model.select(selection, flags)
                # The primary drives the scroll and the current index,
                # matching what the image calls the primary.
                selection_model.setCurrentIndex(
                    first_idx,
                    QItemSelectionModel.SelectionFlag.NoUpdate,
                )
                table.scrollTo(
                    first_idx, QTableView.ScrollHint.PositionAtCenter,
                )
        finally:
            self._applying_external_selection = False

    def clear(self) -> None:
        """Empty all three tabs and drop source references."""
        self._detected_src = None
        self._fitted_src = None
        self._matched_src = []
        self._detected_model.removeRows(0, self._detected_model.rowCount())
        self._fitted_model.removeRows(0, self._fitted_model.rowCount())
        self._matched_model.removeRows(0, self._matched_model.rowCount())

    # --- Internals ---

    def _build_table(
        self, source_model: QStandardItemModel,
    ) -> tuple[QTableView, QSortFilterProxyModel]:
        # Custom proxy handles mixed numeric / text columns — see
        # ``_PeakProxy.lessThan``.
        proxy = _PeakProxy(self)
        proxy.setSourceModel(source_model)

        table = _PeakTableView(self)
        table.setModel(proxy)
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        # Extended = Ctrl+click to add/remove one row, Shift+click for a
        # run, Ctrl+A for all. The image viewer already multi-selects
        # detected and fitted peaks the same way, and the bulk actions
        # behind it (delete, batch fit) are kind-scoped — which is why
        # the Matched tab, whose rows are structures rather than peaks,
        # stays single-select (see ``_build_table`` callers).
        table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        _skin_item_view(table)
        table.verticalHeader().setVisible(False)
        # Pack columns to their narrowest content for max visibility
        # in a narrow dock. ``ResizeToContents`` is applied after
        # every fill so newly-added rows don't push columns wider
        # than needed; the user can still drag headers to resize
        # because we don't lock the header mode here.
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        # Reduce the Qt-style minimum so very short content (a
        # single-character header like "a" / "#") can shrink to the
        # actual text width rather than the platform-default 26 px
        # floor. 18 px leaves room for the sort-arrow indicator.
        table.horizontalHeader().setMinimumSectionSize(18)
        # Compact cell + header padding (~6 px reclaimed per column in
        # a narrow dock) now comes from the skin's QTableView[mlgid]
        # rules, applied by _skin_item_view above.
        _attach_empty_hint(
            table,
            "No peaks on this frame.\n"
            "Run detection from the Pipeline dock, or draw a box on the "
            "image and add it."
        )
        return table, proxy

    # Per-column upper bound on width after pack. Long string
    # columns (matched "IDs" cell that can hold dozens of peak ids)
    # would otherwise dominate the dock width; cells elide
    # automatically and the full text is always available as a
    # tooltip — see ``_fill_matched``.
    _MAX_COL_WIDTH = 220

    def _pack_columns(self, table: QTableView) -> None:
        """Shrink every column to fit the widest entry in the
        current proxy rows, then clamp any column wider than
        ``_MAX_COL_WIDTH`` so a single long cell can't blow out
        the dock. The user can still drag the header to expand."""
        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        for c in range(table.model().columnCount()):
            if table.columnWidth(c) > self._MAX_COL_WIDTH:
                header.resizeSection(c, self._MAX_COL_WIDTH)
        # Word-elide is the default on QTableView so the over-wide
        # text gets rendered as "13, 17, 26, …" automatically.

    def _fill_detected(self, table: PeakTable | None) -> None:
        self._detected_model.removeRows(0, self._detected_model.rowCount())
        if table is None or len(table) == 0:
            self._pack_columns(self._detected_table)
            return
        for i in range(len(table)):
            row = [
                _make_id_item(int(table.ids[i]), "detected"),
                _num_item(float(table.radius[i])),
                _num_item(float(table.radius_width[i])),
                _num_item(float(table.angle[i]), fmt="{:.2f}"),
                _num_item(float(table.angle_width[i]), fmt="{:.2f}"),
                _num_item(float(table.score[i])),
                _text_item(_type_str(bool(table.is_ring[i]))),
            ]
            self._detected_model.appendRow(row)
        self._pack_columns(self._detected_table)

    def _fill_fitted(self, table: PeakTable | None) -> None:
        self._fitted_model.removeRows(0, self._fitted_model.rowCount())
        if table is None or len(table) == 0:
            self._pack_columns(self._fitted_table)
            return
        for i in range(len(table)):
            row = [
                _make_id_item(int(table.ids[i]), "fitted"),
                _num_item(float(table.radius[i])),
                _num_item(float(table.radius_width[i])),
                _num_item(float(table.angle[i]), fmt="{:.2f}"),
                _num_item(float(table.angle_width[i]), fmt="{:.2f}"),
                _num_item(float(table.amplitude[i]), fmt="{:.3g}"),
                _num_item(float(table.score[i])),
                _text_item(_type_str(bool(table.is_ring[i]))),
            ]
            self._fitted_model.appendRow(row)
        self._pack_columns(self._fitted_table)

    def _fill_matched(self, structures: list[MatchedStructure]) -> None:
        """One row per structure. The CIF cell (column 0) stores the
        ``structure_uid`` on ``_ROLE_STRUCTURE_UID`` and the full list
        of fitted-peak ids on ``_ROLE_PEAK_ID`` (as a list, not a
        scalar) so the row-click handler can rebuild the matched
        SelectedPeak with ``multi_peak_ids`` populated."""
        self._matched_model.removeRows(0, self._matched_model.rowCount())
        for s in structures:
            tbl = s.peaks
            hkl = f"({s.h}{s.k}{s.l})" if (s.h, s.k, s.l) != (0, 0, 0) else "rand"
            ids = [int(x) for x in tbl.ids]
            ids_text = ", ".join(str(x) for x in ids) if ids else "—"
            cif_item = _text_item(s.cif)
            cif_item.setData("matched", _ROLE_KIND)
            cif_item.setData(ids, _ROLE_PEAK_ID)
            cif_item.setData(s.unique_id, _ROLE_STRUCTURE_UID)
            # IDs cell is capped width-wise by ``_pack_columns``; the
            # display elides with "…", so attach the full list as a
            # tooltip so the user can hover to see every id.
            ids_item = _text_item(ids_text)
            ids_item.setToolTip(ids_text)
            row = [
                cif_item,
                _text_item(hkl),
                _num_item(float(s.probability), fmt="{:.3f}"),
                _int_item(len(ids)),
                ids_item,
            ]
            self._matched_model.appendRow(row)
        self._pack_columns(self._matched_table)

    def _on_row_changed(self, kind: str, current: QModelIndex) -> None:
        if self._applying_external_selection:
            return
        if not current.isValid():
            return
        # A Ctrl+click moves the current row *and* grows the selection,
        # and the two signals do not arrive in a fixed order across
        # input paths (mouse, Ctrl+Space, programmatic). Whenever the
        # tab holds more than one row the multi-selection is the
        # answer, whichever signal noticed first — otherwise the single
        # emit would land last and collapse the selection to one peak.
        if len(self._selected_rows(kind)) >= 2:
            self._on_selection_changed(kind)
            return
        # The signal carries the proxy index; the items live in the
        # source model. Map back to source, then read the row's
        # column-0 item — for Detected / Fitted it's the ID item, for
        # Matched it's the CIF item (which carries the structure_uid
        # + the structure's full peak-id list).
        proxy = current.model()
        source_idx = proxy.mapToSource(current)
        source_model = source_idx.model()
        id_item = source_model.item(source_idx.row(), 0)
        if id_item is None:
            return
        peak_id_payload = id_item.data(_ROLE_PEAK_ID)
        if peak_id_payload is None:
            return
        # Matched stores a list under _ROLE_PEAK_ID; pass it through
        # untouched. Detected / Fitted store an int.
        if kind == "matched":
            sel = self._build_selection(kind, 0, id_item)
        else:
            sel = self._build_selection(kind, int(peak_id_payload), id_item)
        if sel is None:
            return
        self._emit_own(self.peakSelectedFromTable, sel)

    def _emit_own(self, signal, payload) -> None:
        """Emit one of the panel's own selection signals.

        The flag tells ``set_external_selection(s)`` that whatever
        comes back is this panel's own change echoing through the
        viewer, and must not be re-applied to the table mid-event.
        """
        self._emitting_own_selection = True
        try:
            signal.emit(payload)
        finally:
            self._emitting_own_selection = False

    def _table_for(self, kind: str):
        return {
            "detected": self._detected_table,
            "fitted": self._fitted_table,
            "matched": self._matched_table,
        }.get(kind)

    def _selected_rows(self, kind: str) -> list[SelectedPeak]:
        """Every selected row of one tab, as ``SelectedPeak``s.

        The row under the current index comes first, so the peak the
        user touched last is the one that becomes the primary
        selection — the one the Parameter panel and the resize ROI
        follow. The rest keep the view's own top-to-bottom order,
        which is the sort order the user is looking at.

        Rows are rebuilt from the live source table, not from the cell
        text, so display rounding never leaks into the geometry (same
        rule as ``_on_row_changed``).
        """
        table = self._table_for(kind)
        if table is None:
            return []
        model = table.selectionModel()
        if model is None:
            return []
        rows = list(model.selectedRows())
        current = model.currentIndex()
        if current.isValid():
            rows.sort(key=lambda idx: idx.row() != current.row())
        out: list[SelectedPeak] = []
        for proxy_idx in rows:
            source_idx = proxy_idx.model().mapToSource(proxy_idx)
            id_item = source_idx.model().item(source_idx.row(), 0)
            if id_item is None:
                continue
            payload = id_item.data(_ROLE_PEAK_ID)
            if payload is None:
                continue
            sel = self._build_selection(
                kind, 0 if kind == "matched" else int(payload), id_item,
            )
            if sel is not None:
                out.append(sel)
        return out

    def _on_selection_changed(self, kind: str) -> None:
        """Emit the multi-selection when a tab holds more than one row.

        A single row already went out through ``peakSelectedFromTable``
        on ``currentRowChanged``; re-emitting it here would only make
        the viewer do the same work twice.
        """
        if self._applying_external_selection:
            return
        sels = self._selected_rows(kind)
        if len(sels) < 2:
            return
        self._emit_own(self.peaksSelectedFromTable, sels)

    def _on_delete_requested(self, kind: str) -> None:
        """Delete pressed on the Detected / Fitted tab.

        The panel does not delete anything itself: it hands the host
        the same ``SelectedPeak`` list the image-side Delete produces,
        so both routes share one confirmation, one write path and one
        undo entry.
        """
        sels = self._selected_rows(kind)
        if sels:
            self._emit_own(self.deletePeaksRequested, sels)

    def _build_selection(
        self,
        kind: str,
        peak_id: int,
        id_item: QStandardItem,
    ) -> SelectedPeak | None:
        """Reconstruct a SelectedPeak from the live source table.

        Pulls geometry from ``self._detected_src`` / ``_fitted_src``
        / ``_matched_src`` so the SelectedPeak carries the full
        per-row precision (the cell display strings are formatted /
        truncated for readability).
        """
        if kind == "detected":
            src = self._detected_src
            if src is None:
                return None
            row_idx = _find_row_by_id(src, peak_id)
            if row_idx is None:
                return None
            return _selected_from_table(src, row_idx, kind, peak_id)
        if kind == "fitted":
            src = self._fitted_src
            if src is None:
                return None
            row_idx = _find_row_by_id(src, peak_id)
            if row_idx is None:
                return None
            return _selected_from_table(src, row_idx, kind, peak_id)
        if kind == "matched":
            # Matched rows are per-structure; ``id_item`` here is the
            # CIF column (col 0) which carries the structure_uid + the
            # full peak-id list as a Python list under _ROLE_PEAK_ID.
            structure_uid = id_item.data(_ROLE_STRUCTURE_UID)
            if structure_uid is None:
                return None
            ids = id_item.data(_ROLE_PEAK_ID)
            if not isinstance(ids, list) or not ids:
                return None
            for s in self._matched_src:
                if s.unique_id != structure_uid:
                    continue
                # Representative peak — first in the structure. Used
                # for the ParameterPanel's per-peak readout while the
                # overlay highlight covers every peak via
                # multi_peak_ids.
                rep_row = _find_row_by_id(s.peaks, ids[0])
                if rep_row is None:
                    return None
                base = _selected_from_table(s.peaks, rep_row, "matched", ids[0])
                base.structure_uid = s.unique_id
                base.structure_label = s.label
                base.multi_peak_ids = list(ids)
                return base
            return None
        return None

    def _select_simple_row(self, sel: SelectedPeak) -> None:
        """Highlight the row matching ``sel`` in Detected / Fitted."""
        if sel.kind == "detected":
            model = self._detected_model
            table = self._detected_table
            proxy = self._detected_proxy
        elif sel.kind == "fitted":
            model = self._fitted_model
            table = self._fitted_table
            proxy = self._fitted_proxy
        else:
            return
        src_row = _find_row_for_id_in_model(model, sel.peak_id)
        if src_row is None:
            return
        proxy_idx = proxy.mapFromSource(model.index(src_row, 0))
        table.selectRow(proxy_idx.row())
        table.scrollTo(proxy_idx, QTableView.ScrollHint.PositionAtCenter)

    def _select_matched_row(self, sel: SelectedPeak) -> None:
        """Highlight the matched-structure row whose ``structure_uid``
        matches the SelectedPeak. Falls back to finding any
        structure that contains the SelectedPeak's representative
        peak_id when structure_uid isn't set (e.g. matched peak
        selected via the image click in a frame with overlapping
        structures)."""
        model = self._matched_model
        target_uid = sel.structure_uid
        target_id = int(sel.peak_id) if sel.peak_id is not None else None
        for r in range(model.rowCount()):
            cif_item = model.item(r, 0)
            if cif_item is None:
                continue
            row_uid = cif_item.data(_ROLE_STRUCTURE_UID)
            row_ids = cif_item.data(_ROLE_PEAK_ID)
            uid_match = target_uid is not None and row_uid == target_uid
            id_match = (
                target_id is not None
                and isinstance(row_ids, list)
                and target_id in row_ids
            )
            if not (uid_match or (target_uid is None and id_match)):
                continue
            proxy_idx = self._matched_proxy.mapFromSource(model.index(r, 0))
            self._matched_table.selectRow(proxy_idx.row())
            self._matched_table.scrollTo(
                proxy_idx, QTableView.ScrollHint.PositionAtCenter,
            )
            return


# --- Module-level helpers ---


def _make_id_item(peak_id: int, kind: str) -> _NumericItem:
    item = _NumericItem(str(int(peak_id)))
    item.setData(float(int(peak_id)), _ROLE_NUMERIC)
    item.setEditable(False)
    item.setTextAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    item.setData(kind, _ROLE_KIND)
    item.setData(int(peak_id), _ROLE_PEAK_ID)
    return item


def _find_row_by_id(table: PeakTable, peak_id: int) -> int | None:
    """First row in ``table`` whose ``ids[i] == peak_id``, or None."""
    if table is None or len(table) == 0:
        return None
    matches = np.where(table.ids == int(peak_id))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _find_row_for_id_in_model(
    model: QStandardItemModel, peak_id: int,
) -> int | None:
    """Source-model row index where the ID-column item carries
    ``peak_id`` as its ``_ROLE_PEAK_ID``."""
    for r in range(model.rowCount()):
        item = model.item(r, 0)
        if item is None:
            continue
        v = item.data(_ROLE_PEAK_ID)
        if v is not None and int(v) == int(peak_id):
            return r
    return None


def _rows_by_id_in_model(model: QStandardItemModel) -> dict[int, int]:
    """``{peak_id: source-model row}`` for every ID-column item, the
    first row winning, as ``_find_row_for_id_in_model`` would."""
    rows: dict[int, int] = {}
    for r in range(model.rowCount()):
        item = model.item(r, 0)
        if item is None:
            continue
        v = item.data(_ROLE_PEAK_ID)
        if v is not None:
            rows.setdefault(int(v), r)
    return rows


def _selected_from_table(
    table: PeakTable, row_idx: int, kind: str, peak_id: int,
) -> SelectedPeak:
    """Build a SelectedPeak from one row of a PeakTable.

    ``frame`` is left at 0 because the host's ``_set_selected``
    only consumes ``kind`` + ``peak_id`` for equality / overlay
    purposes (the active frame is implied by what's on screen).
    The host wires this through ``viewer._set_selected`` which
    will set ``frame=self.current_frame`` on the actual stored
    snapshot if it cares.
    """
    return SelectedPeak(
        kind=kind,
        frame=0,
        peak_id=int(peak_id),
        radius=float(table.radius[row_idx]),
        angle=float(table.angle[row_idx]),
        radius_width=float(table.radius_width[row_idx]),
        angle_width=float(table.angle_width[row_idx]),
        is_ring=bool(table.is_ring[row_idx]),
        score=float(table.score[row_idx]),
    )
