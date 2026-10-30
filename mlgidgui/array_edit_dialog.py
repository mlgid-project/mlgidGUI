"""The value grid: editing what a dataset actually holds.

Two widgets behind one dialog, because the two shapes of data want
genuinely different tables.

* **Numeric arrays** reuse silx's ``ArrayTableWidget``, which already
  ships the thing that is tedious to write: axis selectors that pick a
  2-D slice out of an N-d stack, with a browser per remaining axis.
* **Compound datasets** — the peak tables, and anything else with named
  fields — get a table of their own, one column per field, because a
  record is a row of separate quantities and silx renders it as one
  opaque cell. Row insert, duplicate and delete live here too: a
  compound dataset is a table, and tables grow.

Nothing here opens a file. The dialog is handed an array, and hands one
back; ``MainWindow`` decides what that means for the file.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from silx.gui.data.ArrayTableWidget import ArrayTableWidget

from mlgidgui.h5_edit import EditError, parse_scalar
from mlgidgui.widgets import GAP, PAD, PRIMARY, set_variant, skin_item_view

import logging

logger = logging.getLogger(__name__)


class CompoundTableModel(QAbstractTableModel):
    """A structured array as a table: one column per field.

    Edits are parsed into the field's own dtype, so typing ``2`` into a
    float column stores a float and typing nonsense is refused rather
    than silently stored as zero.
    """

    def __init__(self, array: np.ndarray, *, editable: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._array = np.array(array, copy=True)
        self._fields = list(self._array.dtype.names or ())
        self._editable = editable

    # -- Qt model contract
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(self._array.shape[0])

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._fields)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._fields[section]
        return section

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        value = self._array[index.row()][self._fields[index.column()]]
        return str(value)

    def flags(self, index):
        base = super().flags(index)
        if self._editable and index.isValid():
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        field = self._fields[index.column()]
        dtype = self._array.dtype[field]
        try:
            parsed = (
                bool(parse_scalar(str(value), "bool"))
                if dtype.kind == "b"
                else parse_scalar(str(value), str(dtype))
            )
        except EditError:
            # Refusing is the honest answer: storing a silent 0 for
            # "abc" would look like a successful edit.
            return False
        self._array[index.row()][field] = parsed
        self.dataChanged.emit(index, index, [role])
        return True

    # -- rows
    def insert_rows(self, at: int, count: int = 1) -> None:
        """Insert rows copying the one above, or zeros at the top."""
        at = max(0, min(int(at), self._array.shape[0]))
        if at > 0:
            filler = np.repeat(self._array[at - 1: at], count, axis=0)
        else:
            filler = np.zeros(count, dtype=self._array.dtype)
        self.beginInsertRows(QModelIndex(), at, at + count - 1)
        self._array = np.concatenate(
            [self._array[:at], filler, self._array[at:]])
        self.endInsertRows()

    def delete_rows(self, rows) -> None:
        wanted = sorted({int(r) for r in rows
                         if 0 <= int(r) < self._array.shape[0]}, reverse=True)
        for row in wanted:
            self.beginRemoveRows(QModelIndex(), row, row)
            self._array = np.delete(self._array, row, axis=0)
            self.endRemoveRows()

    def array(self) -> np.ndarray:
        return self._array


class ArrayEditDialog(QDialog):
    """Show — and optionally edit — one dataset's values.

    ``editable`` is decided by the caller from the dataset's size: a
    grid over a detector image would be a way to make a mistake, not a
    way to fix one, so past the threshold it opens read-only with an
    explicit way in.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        path: str = "",
        array: np.ndarray | None = None,
        editable: bool = True,
        read_only_reason: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Values — {path}" if path else "Values")
        self.resize(720, 520)
        self._array = np.asarray(array)
        self._editable = bool(editable)
        self._is_compound = bool(self._array.dtype.names)

        column = QVBoxLayout(self)
        column.setContentsMargins(PAD, PAD, PAD, PAD)
        column.setSpacing(GAP)

        self._header = QLabel(self._describe(), self)
        self._header.setProperty("status", "muted")
        self._header.setWordWrap(True)
        column.addWidget(self._header)

        self._reason = QLabel(read_only_reason, self)
        self._reason.setProperty("status", "warn")
        self._reason.setWordWrap(True)
        self._reason.setVisible(bool(read_only_reason) and not self._editable)
        column.addWidget(self._reason)

        if self._is_compound:
            self._model = CompoundTableModel(
                self._array, editable=self._editable, parent=self)
            self._table = QTableView(self)
            skin_item_view(self._table)
            self._table.setModel(self._model)
            self._table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows)
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents)
            column.addWidget(self._table, 1)
            column.addLayout(self._build_row_actions())
            self._grid = None
        else:
            self._model = None
            self._grid = ArrayTableWidget(self)
            self._grid.setArrayData(
                self._array, copy=True, editable=self._editable)
            column.addWidget(self._grid, 1)
            self._row_actions = None

        buttons = QDialogButtonBox(self)
        self._unlock_btn = None
        if not self._editable:
            self._unlock_btn = buttons.addButton(
                "Edit anyway", QDialogButtonBox.ButtonRole.ResetRole)
            self._unlock_btn.clicked.connect(self._unlock)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        set_variant(ok, PRIMARY)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

    def _describe(self) -> str:
        shape = " × ".join(str(n) for n in self._array.shape) or "scalar"
        if self._is_compound:
            fields = ", ".join(self._array.dtype.names)
            return f"{self._array.shape[0]} rows · {fields}"
        return f"{self._array.dtype} · {shape}"

    def _build_row_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._insert_btn = QPushButton("Insert row", self)
        self._insert_btn.setToolTip(
            "Insert a row above the selection, copying it.")
        self._insert_btn.clicked.connect(self._on_insert)
        row.addWidget(self._insert_btn)
        self._delete_btn = QPushButton("Delete rows", self)
        self._delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self._delete_btn)
        row.addStretch(1)
        for button in (self._insert_btn, self._delete_btn):
            button.setEnabled(self._editable)
        self._row_actions = row
        return row

    def _selected_rows(self) -> list[int]:
        model = self._table.selectionModel()
        if model is None:
            return []
        return sorted({index.row() for index in model.selectedRows()})

    def _on_insert(self) -> None:
        rows = self._selected_rows()
        self._model.insert_rows(rows[0] if rows else self._model.rowCount())

    def _on_delete(self) -> None:
        rows = self._selected_rows()
        if rows:
            self._model.delete_rows(rows)

    def _unlock(self) -> None:
        """Turn a read-only grid editable, on an explicit press."""
        self._editable = True
        self._reason.hide()
        if self._unlock_btn is not None:
            self._unlock_btn.hide()
        if self._model is not None:
            self._model.beginResetModel()
            self._model._editable = True
            self._model.endResetModel()
            for button in (self._insert_btn, self._delete_btn):
                button.setEnabled(True)
        elif self._grid is not None:
            self._grid.setArrayData(self._array, copy=True, editable=True)

    # -- results
    @property
    def is_compound(self) -> bool:
        return self._is_compound

    def result_array(self) -> np.ndarray:
        """The values as they now stand."""
        if self._model is not None:
            return self._model.array()
        return np.asarray(self._grid.getData(copy=True))

    def rows_changed(self) -> bool:
        """Whether the row count differs from what was handed in."""
        return bool(self.result_array().shape[0] != self._array.shape[0]) if (
            self._array.ndim and self.result_array().ndim) else False
