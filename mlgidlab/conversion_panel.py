"""Conversion dock — UI for running pygid raw → NeXus conversion.

Mirrors ``pipeline_panel`` in style: collapsible sections, log pane, single
Run button. Visible only when the active session is a ``RawSession``.

Section state is collected into a ``ConversionConfig`` + list of
``RawScan`` and emitted on ``conversionRunRequested`` for MainWindow to
hand off to the worker. Wiring of the emit path lives in Step 5.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import ai_values
from mlgidlab import file_dialogs
from mlgidlab.file_model import RawEntry, list_entry_names

import logging
logger = logging.getLogger(__name__)


# Shared vocabulary + widgets: canonical homes moved in the 2026 source
# split (conversion_config.py, widgets.py); re-imported here so panel
# internals, tests and older references resolve unchanged.
from mlgidlab.conversion_config import (
    CONV_DET2POL,
    CONV_DET2POL_GID,
    CONV_DET2Q,
    CONV_DET2Q_GID,
    FRAME_ALL,
    FRAME_LIST,
    FRAME_SINGLE,
    GEOM_GID,
    GEOM_TRANSMISSION,
    OUTPUT_SEPARATE_DATASETS,
    OUTPUT_SEPARATE_FILES,
    OUTPUT_SINGLE_ENTRY,
    ConversionConfig,
    RawScan,
    _expand_fabio_scans,
    parse_poni_overrides,
)
from mlgidlab.widgets import CollapsibleSection as _CollapsibleSection
from mlgidlab.widgets import make_form as _make_form
from mlgidlab.widgets import skin_item_view as _skin_item_view
from mlgidlab.widgets import section_label as _section_label
from mlgidlab.widgets import DANGER as _DANGER, PRIMARY as _PRIMARY, set_variant as _set_variant



# -------------- module-level helpers --------------


def _row(*widgets: QWidget) -> QWidget:
    """Pack widgets in a horizontal row with no margins. Convenience for
    QFormLayout entries that combine a line edit + side buttons.
    """
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    for i, child in enumerate(widgets):
        # Stretch the first widget (typically the line edit) so the
        # buttons stay at their natural width.
        h.addWidget(child, 1 if i == 0 else 0)
    return w


def _spin_or_none(spin: QDoubleSpinBox) -> float | None:
    """Read a value from an ``_opt_spin`` box, returning None for "(unset)".

    Mirrors the special-value sentinel used in ``_opt_spin``: a value at
    or below the minimum (-1.0 by construction) means the user left the
    field unset, so we return None and let pygid pick its default.
    """
    v = spin.value()
    if v <= spin.minimum() + 1e-12:
        return None
    return float(v)


def _range_or_none(
    lo: QDoubleSpinBox, hi: QDoubleSpinBox
) -> tuple[float, float] | None:
    """Read a (min, max) pair from two ``_opt_spin`` boxes.

    Returns None when either bound is "(unset)" — both ends of a range
    have to be specified for pygid to honour it; partial ranges revert
    to the default (auto).
    """
    lo_v = _spin_or_none(lo)
    hi_v = _spin_or_none(hi)
    if lo_v is None or hi_v is None:
        return None
    return (lo_v, hi_v)


class _AutoSelectDoubleSpinBox(QDoubleSpinBox):
    """A QDoubleSpinBox that selects all of its text on focus-in.

    Without this, focusing a spinbox showing ``setSpecialValueText``
    placeholder ("(none)" / "(unset)") parks the caret inside the
    placeholder and the user has to manually select + delete the
    string before they can type a real number. Selecting on focus
    means the next keystroke replaces the placeholder so the user
    can just click → type.

    The select-all is wrapped in ``QTimer.singleShot(0, …)`` so it
    runs *after* Qt's default focus handling — otherwise Qt's own
    cursor placement runs after our selectAll and wipes it out.
    """

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        line = self.lineEdit()
        if line is not None:
            QTimer.singleShot(0, line.selectAll)


def _opt_spin(
    *,
    decimals: int = 2,
    max_v: float = 1e6,
    suffix: str = "",
) -> QDoubleSpinBox:
    """A QDoubleSpinBox configured for "leave blank → use PONI default".

    The minimum is set to a sentinel just below 0 and the special value
    text is shown there so the user can dial down to "(unset)" without
    typing 0.
    """
    box = _AutoSelectDoubleSpinBox()
    # Sentinel below 0 acts as "unset"; pygid never sees a negative SDD
    # / wavelength in practice so this is safe.
    box.setMinimum(-1.0)
    box.setMaximum(max_v)
    box.setDecimals(decimals)
    box.setSingleStep(10 ** (-decimals))
    box.setSpecialValueText("(unset)")
    box.setSuffix(suffix)
    box.setValue(-1.0)
    return box


def _build_q_gid_params(panel: ConversionPanel) -> QWidget:
    """Sub-form for ``det2q_gid``: dq, q_xy_range, q_z_range."""
    w = QWidget()
    form = _make_form(w)
    form.setContentsMargins(0, 0, 0, 0)
    panel.dq_q_gid = _opt_spin(decimals=4, max_v=10.0, suffix=" Å⁻¹")
    panel.q_xy_min = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_xy_max = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_z_min = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_z_max = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    form.addRow("dq:", panel.dq_q_gid)
    form.addRow("q_xy min:", panel.q_xy_min)
    form.addRow("q_xy max:", panel.q_xy_max)
    form.addRow("q_z min:", panel.q_z_min)
    form.addRow("q_z max:", panel.q_z_max)
    return w


def _build_q_trans_params(panel: ConversionPanel) -> QWidget:
    """Sub-form for transmission ``det2q``: dq, q_x_range, q_y_range."""
    w = QWidget()
    form = _make_form(w)
    form.setContentsMargins(0, 0, 0, 0)
    panel.dq_q_trans = _opt_spin(decimals=4, max_v=10.0, suffix=" Å⁻¹")
    panel.q_x_min = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_x_max = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_y_min = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    panel.q_y_max = _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹")
    form.addRow("dq:", panel.dq_q_trans)
    form.addRow("q_x min:", panel.q_x_min)
    form.addRow("q_x max:", panel.q_x_max)
    form.addRow("q_y min:", panel.q_y_min)
    form.addRow("q_y max:", panel.q_y_max)
    return w


class _Hdf5MetaPicker(QDialog):
    """Modal silx tree picker for adding HDF5 datasets as metadata.

    Used by the Conversion panel's "Add from HDF5…" button. Returns a
    tuple ``(suggested_key, value, source_path)`` on accept, where:

    - ``suggested_key`` is the basename of the dataset (e.g. ``temperature``
      for ``measurement/temperature``) — the user can edit it after the
      row is added.
    - ``value`` is the first element of the dataset coerced to a string.
    - ``source_path`` is ``filename:/path/inside/file`` so the metadata
      row carries provenance for the value.
    """

    def __init__(
        self, parent: QWidget | None, files: list[Path]
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick HDF5 dataset")
        self.setMinimumSize(640, 480)
        self._files = list(files)
        self._result: tuple[str, str, str] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        hint = QLabel(
            "<i>Select an HDF5 dataset to add as a metadata key. The "
            "value is read once at this moment and stored verbatim.</i>"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # silx tree — same widget the main browser uses, just in a
        # modal dialog scope. Importing inside the constructor keeps
        # silx out of the import path when this dialog isn't reached.
        from silx.gui.hdf5 import Hdf5TreeView

        self._tree = Hdf5TreeView(self)
        self._tree.setSortingEnabled(True)
        for fp in self._files:
            self._tree.findHdf5TreeModel().insertFile(str(fp))
        self._tree.activated.connect(self._on_activated)
        self._tree.selectionModel().selectionChanged.connect(self._on_selection)
        layout.addWidget(self._tree, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(False)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _on_selection(self, *_: object) -> None:
        nodes = list(self._tree.selectedH5Nodes())
        ok = bool(nodes) and self._is_dataset(nodes[0])
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def _on_activated(self, *_: object) -> None:
        # Double-click on a dataset accepts the dialog directly.
        nodes = list(self._tree.selectedH5Nodes())
        if nodes and self._is_dataset(nodes[0]):
            self._on_accept()

    @staticmethod
    def _is_dataset(node) -> bool:
        try:
            import h5py
            return isinstance(node.h5py_object, h5py.Dataset)
        except Exception:
            logger.debug("suppressed exception in _Hdf5MetaPicker._is_dataset", exc_info=True)
            return False

    def _on_accept(self) -> None:
        nodes = list(self._tree.selectedH5Nodes())
        if not nodes or not self._is_dataset(nodes[0]):
            return
        node = nodes[0]
        try:
            ds_path = node.h5py_object.name
            file_name = Path(node.h5py_object.file.filename).name
            data = node.h5py_object[()]
        except Exception as exc:
            logger.debug("suppressed exception in _Hdf5MetaPicker._on_accept", exc_info=True)
            self._result = (str(node.local_name or "value"),
                            f"<read error: {exc}>", "")
            self.accept()
            return
        # Coerce to a single string value. Prefer the scalar form if the
        # dataset is 0D; otherwise show the first element with a hint.
        value = self._coerce_scalar(data)
        suggested_key = ds_path.rsplit("/", 1)[-1] or "value"
        source = f"{file_name}:{ds_path}"
        self._result = (suggested_key, value, source)
        self.accept()

    @staticmethod
    def _coerce_scalar(data) -> str:
        try:
            import numpy as np
        except ImportError:
            return repr(data)
        arr = np.asarray(data)
        if arr.ndim == 0:
            v = arr[()]
        elif arr.size == 1:
            v = arr.flat[0]
        else:
            v = arr.flat[0]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        if arr.size > 1:
            return f"{v} (first of {arr.size}; full array not stored)"
        return str(v)

    @classmethod
    def pick(
        cls, parent: QWidget | None, files: list[Path]
    ) -> tuple[str, str, str] | None:
        dlg = cls(parent, files)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._result
        return None


def _build_pol_params(panel: ConversionPanel, *, gid: bool) -> QWidget:
    """Sub-form shared by ``det2pol`` and ``det2pol_gid``: dang, dq,
    radial_range, angular_range. The two variants share the same
    parameter set; ``gid`` only affects the suffix label.
    """
    w = QWidget()
    form = _make_form(w)
    form.setContentsMargins(0, 0, 0, 0)
    suffix = "_gid" if gid else ""
    dang_attr = f"dang_pol{suffix}"
    dq_attr = f"dq_pol{suffix}"
    rad_min_attr = f"radial_min{suffix}"
    rad_max_attr = f"radial_max{suffix}"
    ang_min_attr = f"angular_min{suffix}"
    ang_max_attr = f"angular_max{suffix}"
    setattr(panel, dang_attr, _opt_spin(decimals=3, max_v=180.0, suffix=" °"))
    setattr(panel, dq_attr, _opt_spin(decimals=4, max_v=10.0, suffix=" Å⁻¹"))
    setattr(panel, rad_min_attr, _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹"))
    setattr(panel, rad_max_attr, _opt_spin(decimals=3, max_v=10.0, suffix=" Å⁻¹"))
    setattr(panel, ang_min_attr, _opt_spin(decimals=2, max_v=360.0, suffix=" °"))
    setattr(panel, ang_max_attr, _opt_spin(decimals=2, max_v=360.0, suffix=" °"))
    form.addRow("dang:", getattr(panel, dang_attr))
    form.addRow("dq:", getattr(panel, dq_attr))
    form.addRow("radial min:", getattr(panel, rad_min_attr))
    form.addRow("radial max:", getattr(panel, rad_max_attr))
    form.addRow("angular min:", getattr(panel, ang_min_attr))
    form.addRow("angular max:", getattr(panel, ang_max_attr))
    return w


class ConversionPanel(QWidget):
    """Top-level widget for the Conversion dock.

    Public surface mirrors the ``PipelinePanel`` slots that MainWindow
    uses (``append_log``, ``clear_log``, ``set_running``) so the host
    can wire either panel uniformly. Run wiring (``conversionRunRequested``
    emit) lands in Step 5.
    """

    # Emitted when the user clicks Convert. Args: (ConversionConfig,
    # list[RawScan]). MainWindow runs the worker and handles results.
    conversionRunRequested = Signal(object, list)
    # Log routing: the panel emits messages and the host forwards them to
    # the shared Logs dock. Public ``append_log`` / ``clear_log`` API is
    # preserved so existing call sites keep working.
    logMessage = Signal(str)
    logCleared = Signal()
    # Emitted (fliplr, flipud, transp) when any Orientation checkbox
    # toggles, so the host can reorient the live raw preview to match
    # the conversion output.
    rawFlipsChanged = Signal(bool, bool, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # File/entry inputs are populated by ``set_raw_inputs`` from
        # MainWindow when a raw session is activated. Empty until then.
        self._raw_inputs: list[tuple[Path, list[RawEntry]]] = []
        # Resolver wired by MainWindow that returns a 2D numpy array
        # of the currently displayed raw frame (or None if no raw
        # session is active). Used to pre-load the in-GUI
        # calibration dialog so the user doesn't have to re-browse
        # to the same image they're already looking at. Wired in
        # ``set_active_raw_frame_resolver``.
        self._get_active_raw_frame: Callable[[], object] | None = None
        # {field: value} last written into the override spinboxes by
        # ``_autofill_overrides_from_poni``. A field still holding its
        # autofilled value is a readout of the PONI, not a user override,
        # and ``_build_config`` drops it — see ``_unedited_autofill``.
        self._poni_autofill: dict[str, float] = {}
        self._build_ui()

    # ---------------- Public surface ----------------

    def append_log(self, msg: str) -> None:
        """Forward ``msg`` to the shared Logs dock via ``logMessage``."""
        self.logMessage.emit(msg)

    def clear_log(self) -> None:
        """Ask the shared Logs dock to wipe its contents."""
        self.logCleared.emit()

    def set_running(self, running: bool) -> None:
        """Disable / re-enable interactive widgets while a run is in flight."""
        # Just gate the Convert button — every parameter widget remains
        # readable so the user can review what's running.
        if hasattr(self, "btn_convert"):
            self.btn_convert.setEnabled(not running and self._is_runnable())

    def set_raw_inputs(
        self, inputs: list[tuple[Path, list[RawEntry]]]
    ) -> None:
        """Populate the file/entry tree from the active raw session.

        ``inputs`` is a list of ``(file_path, entries)`` tuples — one
        entry per (file, dataset) pair found by ``list_raw_entries``.
        Existing user check-state is dropped on each call; the typical
        caller activates one raw session per call.
        """
        self._raw_inputs = list(inputs)
        self._refresh_selection_tree()
        self._refresh_runnable()

    def add_raw_inputs(
        self, inputs: list[tuple[Path, list[RawEntry]]]
    ) -> None:
        """Append files to the selection tree, keeping what is already there.

        Used when raw files are dropped one at a time onto an open raw
        batch: existing rows and their check state stay untouched and the
        new files arrive unchecked below them.
        """
        if not inputs:
            return
        self._raw_inputs.extend(inputs)
        tree = self.selection_tree
        tree.blockSignals(True)
        try:
            for file_path, entries in inputs:
                file_item = self._make_file_item(file_path, entries)
                tree.addTopLevelItem(file_item)
                file_item.setExpanded(True)
        finally:
            tree.blockSignals(False)
        self.select_all_box.setEnabled(True)
        self._sync_select_all_state()
        self._refresh_runnable()

    def remove_raw_inputs(self, paths) -> None:
        """Drop the rows of ``paths`` from the selection tree.

        The remaining rows keep their check state. Used when single files
        are closed out of an open raw batch.
        """
        gone = {str(p) for p in paths}
        if not gone:
            return
        self._raw_inputs = [
            (fp, entries) for fp, entries in self._raw_inputs
            if str(fp) not in gone
        ]
        tree = self.selection_tree
        tree.blockSignals(True)
        try:
            for i in reversed(range(tree.topLevelItemCount())):
                if tree.topLevelItem(i).toolTip(0) in gone:
                    tree.takeTopLevelItem(i)
        finally:
            tree.blockSignals(False)
        self.select_all_box.setEnabled(bool(self._raw_inputs))
        self._sync_select_all_state()
        self._refresh_runnable()

    # ---------------- UI construction ----------------

    def _build_ui(self) -> None:
        # Outer layout owns only the scroll area + the always-visible
        # Convert button; the inner content widget owns section margins.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Vertical scroll fires only when content overflows; horizontal
        # is hard-locked off so a narrow dock collapses form rows
        # (labels wrap above fields, see ``_make_form``) instead of
        # introducing an x-axis scrollbar.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)

        # Sections are independent — any combination can be open at once.
        # Selection and Output start open: what to convert and where it
        # goes are what the user sets first. The rest stay collapsed to
        # keep the initial UI compact.
        self._sections: list[_CollapsibleSection] = [
            self._build_selection_section(),
            self._build_output_section(),
            self._build_exp_params_section(),
            self._build_metadata_section(),
            self._build_conversion_config_section(),
        ]
        for s in self._sections:
            inner.addWidget(s)

        # Trailing stretch keeps the sections top-anchored when they
        # don't fill the visible scroll height.
        inner.addStretch(1)
        scroll.setWidget(content)

        # Convert button lives *outside* the scroll area so it's always
        # reachable without scrolling, regardless of how many sections
        # the user has expanded.
        button_row = QWidget()
        button_layout = QVBoxLayout(button_row)
        button_layout.setContentsMargins(8, 4, 8, 8)
        button_layout.setSpacing(0)
        self.btn_convert = _set_variant(QPushButton("Convert"), _PRIMARY)
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        button_layout.addWidget(self.btn_convert)
        outer.addWidget(button_row)

    # ---------------- Section: Selection ----------------

    def _build_selection_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Selection", expanded=True)

        hint = QLabel(
            "<i>Tick the entries to convert. Frame mode applies to every "
            "selected entry.</i>"
        )
        hint.setWordWrap(True)
        section.body_layout.addWidget(hint)

        # Whole-batch toggle — with thousands of image files per raw
        # batch, per-file ticking is impractical. Tristate is display
        # only: a click never lands on "partial" (see
        # _on_select_all_clicked), it just mirrors a mixed tree.
        self.select_all_box = QCheckBox("Select all")
        self.select_all_box.setTristate(True)
        # Enabled once set_raw_inputs delivers something to select.
        self.select_all_box.setEnabled(False)
        self.select_all_box.clicked.connect(self._on_select_all_clicked)
        section.body_layout.addWidget(self.select_all_box)

        self.selection_tree = QTreeWidget()
        self.selection_tree.setColumnCount(3)
        self.selection_tree.setHeaderLabels(["Source", "Shape", "Dtype"])
        self.selection_tree.setRootIsDecorated(True)
        self.selection_tree.setUniformRowHeights(True)
        # Two-stage column widths: the entry column gets the bulk of
        # available space so nested dataset paths stay readable.
        header = self.selection_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.selection_tree.itemChanged.connect(self._on_selection_changed)
        section.body_layout.addWidget(self.selection_tree, 1)

        # Frame mode picker — one config applies to every checked entry.
        frame_form = _make_form()
        frame_form.setContentsMargins(0, 0, 0, 0)
        self.frame_mode = QComboBox()
        self.frame_mode.addItems([FRAME_ALL, FRAME_SINGLE, FRAME_LIST])
        self.frame_mode.currentTextChanged.connect(self._on_frame_mode_changed)
        frame_form.addRow("Frame mode:", self.frame_mode)
        # Stack swaps the input widget by mode: Single → spinbox-style int;
        # List → comma-separated text. ``_on_frame_mode_changed`` toggles
        # visibility.
        self.frame_single = QLineEdit()
        self.frame_single.setPlaceholderText("frame index (e.g. 0)")
        self.frame_single.setVisible(False)
        frame_form.addRow("", self.frame_single)
        self.frame_list = QLineEdit()
        self.frame_list.setPlaceholderText("comma-separated indices, e.g. 0,3,7")
        self.frame_list.setVisible(False)
        frame_form.addRow("", self.frame_list)
        section.body_layout.addLayout(frame_form)

        return section

    def _on_frame_mode_changed(self, mode: str) -> None:
        self.frame_single.setVisible(mode == FRAME_SINGLE)
        self.frame_list.setVisible(mode == FRAME_LIST)

    def _all_top_level_checked(self) -> bool:
        tree = self.selection_tree
        n = tree.topLevelItemCount()
        return n > 0 and all(
            tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
            for i in range(n)
        )

    def _on_select_all_clicked(self) -> None:
        """Check or uncheck the whole tree in one pass.

        Anything not fully checked becomes checked; fully checked
        becomes unchecked (Qt's native tristate click-cycle would force
        the user through the partial state — the target is computed
        from the TREE, then stamped on the box). Tree signals are
        blocked while stamping: letting the per-item cascade run would
        fire ``_on_selection_changed`` (parent-state recompute + a
        full-tree runnability walk) once per file — quadratic on a
        several-thousand-image batch.
        """
        target = (
            Qt.CheckState.Unchecked
            if self._all_top_level_checked()
            else Qt.CheckState.Checked
        )
        self.select_all_box.setCheckState(target)
        tree = self.selection_tree
        tree.blockSignals(True)
        try:
            for i in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(i)
                item.setCheckState(0, target)
                for j in range(item.childCount()):
                    item.child(j).setCheckState(0, target)
        finally:
            tree.blockSignals(False)
        self._refresh_runnable()

    def _sync_select_all_state(self) -> None:
        """Mirror the tree's aggregate check state on the box.

        Programmatic ``setCheckState`` does not emit ``clicked``, so
        this never re-enters ``_on_select_all_clicked``.
        """
        tree = self.selection_tree
        n = tree.topLevelItemCount()
        checked = sum(
            1
            for i in range(n)
            if tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        )
        partial = any(
            tree.topLevelItem(i).checkState(0)
            == Qt.CheckState.PartiallyChecked
            for i in range(n)
        )
        if n == 0 or checked == 0 and not partial:
            state = Qt.CheckState.Unchecked
        elif checked == n:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        self.select_all_box.setCheckState(state)

    def _on_selection_changed(self, item: QTreeWidgetItem, col: int) -> None:
        # Only react to checkbox changes; column edits are not editable
        # in the selection tree.
        if col != 0:
            return
        # Cascading top-level → children selection: when the user toggles
        # a file's box, cascade to all its entries unless they were
        # already individually toggled.
        if item.parent() is None:
            state = item.checkState(0)
            if state == Qt.CheckState.PartiallyChecked:
                return
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
        else:
            self._refresh_parent_check_state(item.parent())
        self._sync_select_all_state()
        self._refresh_runnable()
        # The frame axis just changed, and it decides both how three
        # angles are read and whether a list is the right length.
        self._refresh_ai_hint()

    def _refresh_parent_check_state(self, parent: QTreeWidgetItem) -> None:
        """Set parent to checked / unchecked / partial based on children."""
        n = parent.childCount()
        if n == 0:
            return
        checked = sum(
            1
            for i in range(n)
            if parent.child(i).checkState(0) == Qt.CheckState.Checked
        )
        parent.treeWidget().blockSignals(True)
        try:
            if checked == 0:
                parent.setCheckState(0, Qt.CheckState.Unchecked)
            elif checked == n:
                parent.setCheckState(0, Qt.CheckState.Checked)
            else:
                parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
        finally:
            parent.treeWidget().blockSignals(False)

    def _refresh_selection_tree(self) -> None:
        # Fresh tree, fresh box: check-state was dropped with the items.
        self.select_all_box.setCheckState(Qt.CheckState.Unchecked)
        self.select_all_box.setEnabled(bool(self._raw_inputs))
        self.selection_tree.clear()
        for file_path, entries in self._raw_inputs:
            file_item = self._make_file_item(file_path, entries)
            self.selection_tree.addTopLevelItem(file_item)
            file_item.setExpanded(True)

    def _make_file_item(
        self, file_path: Path, entries: list[RawEntry]
    ) -> QTreeWidgetItem:
        """One unchecked file row with a checkable child per raw entry."""
        file_item = QTreeWidgetItem([file_path.name, "", ""])
        file_item.setFlags(
            file_item.flags()
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsAutoTristate
        )
        file_item.setCheckState(0, Qt.CheckState.Unchecked)
        file_item.setToolTip(0, str(file_path))
        for re in entries:
            shape = "×".join(str(s) for s in re.shape)
            # Fabio images have no internal dataset path: label the child
            # by frame count instead of the empty string.
            if getattr(re, "frame_map", None) is not None:
                n = len(re.frame_map)
                child_text = f"image ({n} frame{'s' if n != 1 else ''})"
            else:
                child_text = re.dataset_path
            child = QTreeWidgetItem([
                child_text, shape, re.dtype,
            ])
            child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            child.setCheckState(0, Qt.CheckState.Unchecked)
            # Stash the RawEntry on the item so collection back into a
            # ConversionConfig is a single ``data()`` lookup.
            child.setData(0, Qt.ItemDataRole.UserRole, re)
            file_item.addChild(child)
        return file_item

    # ---------------- Section: Experimental parameters ----------------

    def _build_exp_params_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Experimental parameters", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.poni_path = QLineEdit()
        self.poni_path.setPlaceholderText("Path to pyFAI PONI file (required)")
        poni_browse = QPushButton("Browse…")
        poni_browse.clicked.connect(self._browse_poni)
        poni_create = QPushButton("Create…")
        poni_create.setToolTip(
            "Calibrate a new PONI inside mlgidLAB. Opens pyFAI's "
            "calibration workflow (experiment → mask → peak picking → "
            "geometry refinement) and auto-populates this field with "
            "the saved file."
        )
        poni_create.clicked.connect(self._create_poni)
        poni_clear = QPushButton("Clear")
        poni_clear.clicked.connect(lambda: self.poni_path.setText(""))
        form.addRow("PONI:", _row(
            self.poni_path, poni_browse, poni_create, poni_clear,
        ))
        self.poni_path.textChanged.connect(self._refresh_runnable)
        # A hand-typed path autofills the override fields on focus-out /
        # Enter; the Browse / Create flows call the autofill directly.
        self.poni_path.editingFinished.connect(self._autofill_overrides_from_poni)

        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("Optional .npy / .tif / .edf mask")
        mask_browse = QPushButton("Browse…")
        mask_browse.clicked.connect(self._browse_mask)
        mask_create = QPushButton("Create…")
        mask_create.setToolTip(
            "Draw a mask interactively. Opens pyFAI's calibration "
            "workflow on the Mask task; on save, the path lands in "
            "this field automatically."
        )
        mask_create.clicked.connect(self._create_mask)
        mask_clear = QPushButton("Clear")
        mask_clear.clicked.connect(lambda: self.mask_path.setText(""))
        form.addRow("Mask:", _row(
            self.mask_path, mask_browse, mask_create, mask_clear,
        ))

        # Angle of incidence — one value for the whole scan, or one per
        # frame. A line edit rather than a spinbox because a spinbox
        # cannot hold a list; the grammar lives in ``ai_values`` and the
        # hint below spells out what the typed text resolved to, which
        # is how the ramp form's off-by-one stays visible. Empty means
        # "not set" — the old spinbox used 0.0 for that, which is also
        # a legal angle.
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText(
            "0.2   or   0.1,0.3,0.5   or   (0.1,1.5,13)"
        )
        self.ai_input.setToolTip(
            "One angle in degrees for every frame, or one per frame:\n"
            "  0.2                 the same angle for all frames\n"
            "  0.1,0.3,0.5         an explicit angle per frame\n"
            "  (0.1,1.5,13)        a ramp: start, end, STEPS\n"
            "A ramp gives one more angle than its step count (13 steps "
            "= 14 angles), matching pygid's own scan convention. The "
            "line below always shows what your text resolved to."
        )
        self.ai_input.textChanged.connect(self._refresh_ai_hint)
        form.addRow("Angle of incidence:", self.ai_input)
        self.ai_hint = QLabel("")
        self.ai_hint.setProperty("status", "muted")
        self.ai_hint.setWordWrap(True)
        form.addRow("", self.ai_hint)

        # Detector orientation. These live in the main form (not the
        # Manual-overrides subsection) because they're routine
        # per-beamline settings used on most conversions; they're always
        # honoured when checked. Transpose sits with the flips because it
        # is the same kind of decision and the same three end up in the
        # same ExpParams -- it used to be buried under Manual overrides,
        # where it read as a rarely-needed correction and did nothing to
        # the preview.
        self.flip_lr = QCheckBox("Flip horizontally (fliplr)")
        self.flip_ud = QCheckBox("Flip vertically (flipud)")
        self.transp = QCheckBox("Transpose (transp)")
        # Re-emit the combined orientation so the host can reorient the
        # raw preview.
        self.flip_lr.toggled.connect(self._emit_raw_flips)
        self.flip_ud.toggled.connect(self._emit_raw_flips)
        self.transp.toggled.connect(self._emit_raw_flips)
        flips = QHBoxLayout()
        flips.setContentsMargins(0, 0, 0, 0)
        flips.addWidget(self.flip_lr)
        flips.addWidget(self.flip_ud)
        flips.addWidget(self.transp)
        flips.addStretch(1)
        flips_widget = QWidget()
        flips_widget.setLayout(flips)
        form.addRow("Orientation:", flips_widget)

        section.body_layout.addLayout(form)

        # Manual override fields, in their own collapsible subsection to
        # keep the default form compact. Value-driven: any field left at
        # "(unset)" is simply not sent, so pygid reads it from the PONI
        # file; a set field overrides the PONI. Loading a PONI pre-fills
        # these with the values pygid would derive from it (see
        # ``_autofill_overrides_from_poni``), so they double as a readout
        # the user can tweak.
        override_section = _CollapsibleSection("Manual overrides", expanded=False)
        ovl = _make_form()
        ovl.setContentsMargins(0, 0, 0, 0)
        self.over_centerX = _opt_spin(decimals=2, max_v=1e9)
        self.over_centerY = _opt_spin(decimals=2, max_v=1e9)
        self.over_SDD = _opt_spin(decimals=4, max_v=1e6, suffix=" m")
        self.over_wavelength = _opt_spin(decimals=6, max_v=1e3, suffix=" Å")
        ovl.addRow("centerX (px):", self.over_centerX)
        ovl.addRow("centerY (px):", self.over_centerY)
        ovl.addRow("SDD:", self.over_SDD)
        ovl.addRow("Wavelength:", self.over_wavelength)
        override_section.body_layout.addLayout(ovl)
        section.body_layout.addWidget(override_section)

        return section

    def _emit_raw_flips(self, _checked: bool = False) -> None:
        """Broadcast the current (fliplr, flipud, transp) checkbox state."""
        self.rawFlipsChanged.emit(
            self.flip_lr.isChecked(),
            self.flip_ud.isChecked(),
            self.transp.isChecked(),
        )

    def _browse_poni(self) -> None:
        path = file_dialogs.open_file(
            self, "Select PONI file",
            "PONI calibration (*.poni);;All files (*)",
        )
        if path:
            self.poni_path.setText(path)
            self._autofill_overrides_from_poni()

    def _unedited_autofill(self, key: str, value: float) -> bool:
        """Whether ``value`` is still exactly what the PONI autofill wrote.

        Loading a PONI pre-fills the override boxes so they double as a
        readout. Sending an untouched one back looks like a no-op but is
        not: pygid only applies ``fliplr`` / ``flipud`` / ``transp`` to
        the beam center when *one* of (poni1/poni2, centerX/centerY) is
        given — ``ExpParams._exp_params_update_`` takes neither branch
        when both are set. Handing back the PONI's own centre alongside
        the PONI therefore left the centre unflipped while the image was
        flipped, putting the missing wedge on the wrong side of the
        converted frame.

        So an untouched field is not an override. A field the user
        actually changed still is, and ``conversion.run_conversion``
        makes it win (see the poni1/poni2 reset there).
        """
        prev = self._poni_autofill.get(key)
        return prev is not None and float(value) == float(prev)

    def _autofill_overrides_from_poni(self) -> None:
        """Pre-fill the override fields with the loaded PONI's values.

        The overrides exist to override the PONI, so showing the PONI's
        own values (as pygid will derive them) gives the user a readout
        to tweak instead of blank fields. Sending unedited values back
        is a no-op — pygid computes the same numbers from the file.
        Best-effort: an unreadable/foreign file leaves the fields alone.
        """
        self._poni_autofill = {}
        text = self.poni_path.text().strip()
        if not text:
            return
        path = Path(text)
        if not path.is_file():
            return
        try:
            values = parse_poni_overrides(path)
        except Exception:
            logger.debug("suppressed PONI parse in autofill", exc_info=True)
            self.append_log(f"Could not parse {path.name} for override pre-fill.")
            self._poni_autofill = {}
            return
        if not values:
            return
        fields = {
            "centerX": self.over_centerX,
            "centerY": self.over_centerY,
            "SDD": self.over_SDD,
            "wavelength": self.over_wavelength,
        }
        for key, value in values.items():
            fields[key].setValue(value)
        # Snapshot what the autofill put there, read back through the
        # spinbox so the comparison in ``_build_config`` is against the
        # rounded value the widget actually holds. A field still showing
        # this is a readout, not an override, and must not be sent — see
        # ``_unedited_autofill``.
        self._poni_autofill = {
            key: _spin_or_none(fields[key]) for key in values
        }
        self.append_log(
            f"Override fields pre-filled from {path.name}: "
            + ", ".join(f"{k}={v:.6g}" for k, v in sorted(values.items()))
        )

    def _browse_mask(self) -> None:
        path = file_dialogs.open_file(
            self, "Select mask file",
            "Mask images (*.npy *.tif *.tiff *.edf);;All files (*)",
        )
        if path:
            self.mask_path.setText(path)

    # ---------------- In-GUI calibration ----------------

    def set_active_raw_frame_resolver(
        self, fn: Callable[[], object],
    ) -> None:
        """Install a callable that returns a 2D ndarray of the raw
        frame currently on screen, or None.

        Used by the in-GUI calibration dialog to pre-load the
        user's active raw frame so they don't have to re-browse to
        the same image. ``fn`` is invoked lazily at the moment the
        dialog opens — not stored as a reference to the frame, so
        late-bound semantics (frame slider may have moved) are
        respected.
        """
        self._get_active_raw_frame = fn

    def _open_calibration_dialog(self, start_task: str):
        """Lazily import + construct the calibration dialog.

        pyFAI's Qt-heavy import chain is deferred until the user
        actually clicks ``Create…`` — so a broken pyFAI install
        only surfaces here (with a friendly message) instead of
        breaking cold startup. Returns the dialog *or* None when
        the import fails and the user has already seen the error.
        """
        try:
            from mlgidlab.calibration_dialog import CalibrationDialog
        except Exception as exc:
            QMessageBox.critical(
                self, "Calibration unavailable",
                "pyFAI's calibration widgets couldn't load:\n\n"
                f"{exc}\n\n"
                "Reinstall mlgidLAB or pyFAI to enable in-GUI "
                "calibration. You can still browse to an externally "
                "calibrated PONI / mask via the Browse… buttons.",
            )
            return None
        initial = None
        if self._get_active_raw_frame is not None:
            try:
                initial = self._get_active_raw_frame()
            except Exception:
                # The resolver shouldn't raise, but if it does the
                # dialog can still open without a pre-filled image.
                logger.debug("suppressed exception in ConversionPanel._open_calibration_dialog", exc_info=True)
                initial = None
        # If the user already has PONI / mask paths in the
        # Conversion dock, carry them into the dialog so workflows
        # like "I came here to make a mask, my PONI is fine" don't
        # require re-picking the existing file. Only forward paths
        # that point to a real file — empty or stale entries get
        # silently dropped.
        def _existing(line_edit) -> str | None:
            text = line_edit.text().strip()
            if not text:
                return None
            try:
                return text if Path(text).exists() else None
            except Exception:
                logger.debug("suppressed exception in ConversionPanel._open_calibration_dialog._existing", exc_info=True)
                return None

        dlg = CalibrationDialog(
            self,
            initial_image=initial,
            initial_poni=_existing(self.poni_path),
            initial_mask=_existing(self.mask_path),
            start_task=start_task,
        )
        # The dialog's "Add PONI / Mask to conversion" buttons
        # emit these signals; route them straight into the QLineEdits
        # so the user can apply the freshly-saved paths without
        # closing the dialog (and can iterate — produce a second
        # PONI, click Add again, etc.).
        dlg.applyPoniRequested.connect(self._apply_poni_path)
        dlg.applyMaskRequested.connect(self.mask_path.setText)
        return dlg

    def _apply_poni_path(self, path: str) -> None:
        """Set the PONI path and pre-fill the override fields from it."""
        self.poni_path.setText(path)
        self._autofill_overrides_from_poni()

    def _create_poni(self) -> None:
        """Launch the calibration dialog on the Experiment task and,
        on accept, populate the PONI path field with whatever path
        the user saved. We start at step 1 (Experiment) rather than
        jumping to Geometry because the experimental setup
        (detector, wavelength, calibrant image) feeds every later
        step — skipping it leaves the geometry refinement working
        from defaults that are almost never right."""
        dlg = self._open_calibration_dialog(start_task="experiment")
        if dlg is None:
            return
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.saved_poni_path is not None:
            self._apply_poni_path(str(dlg.saved_poni_path))
        if dlg.saved_mask_path is not None and not self.mask_path.text().strip():
            # Convenience: if the user happened to save a mask
            # while they were in the PONI dialog (the workflows
            # share the same window), pick it up too — but only
            # when the mask field is currently empty so we don't
            # overwrite something they've already chosen.
            self.mask_path.setText(str(dlg.saved_mask_path))

    def _create_mask(self) -> None:
        """Launch the calibration dialog on the Mask task and, on
        accept, populate the mask path field with whatever path
        the user saved."""
        dlg = self._open_calibration_dialog(start_task="mask")
        if dlg is None:
            return
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.saved_mask_path is not None:
            self.mask_path.setText(str(dlg.saved_mask_path))
        if dlg.saved_poni_path is not None and not self.poni_path.text().strip():
            # Same convenience as ``_create_poni``: if they also
            # produced a PONI while in this dialog, pick it up
            # provided the field is empty.
            self._apply_poni_path(str(dlg.saved_poni_path))

    # ---------------- Section: Metadata ----------------

    def _build_metadata_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Metadata", expanded=False)

        smpl_label = _section_label("Sample metadata (YAML)")
        section.body_layout.addWidget(smpl_label)

        smpl_buttons = QHBoxLayout()
        smpl_buttons.setContentsMargins(0, 0, 0, 0)
        load_btn = QPushButton("Load YAML…")
        load_btn.clicked.connect(self._load_smpl_yaml)
        save_btn = QPushButton("Save copy…")
        save_btn.clicked.connect(self._save_smpl_yaml)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.smpl_yaml.setPlainText(""))
        smpl_buttons.addWidget(load_btn)
        smpl_buttons.addWidget(save_btn)
        smpl_buttons.addWidget(clear_btn)
        smpl_buttons.addStretch(1)
        smpl_btn_widget = QWidget()
        smpl_btn_widget.setLayout(smpl_buttons)
        section.body_layout.addWidget(smpl_btn_widget)

        self.smpl_yaml = QPlainTextEdit()
        self.smpl_yaml.setFont(QFont("monospace"))
        self.smpl_yaml.setPlaceholderText(
            "data:\n  name: my_sample\n  ..."
        )
        self.smpl_yaml.setMaximumHeight(120)
        section.body_layout.addWidget(self.smpl_yaml)

        section.body_layout.addSpacing(8)
        exp_label = _section_label("Experimental metadata")
        section.body_layout.addWidget(exp_label)

        self.exp_meta_table = _skin_item_view(QTableWidget(0, 3))
        self.exp_meta_table.setHorizontalHeaderLabels(["Key", "Value", "Source"])
        self.exp_meta_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.exp_meta_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.exp_meta_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.exp_meta_table.setMaximumHeight(140)
        section.body_layout.addWidget(self.exp_meta_table)

        meta_buttons = QHBoxLayout()
        meta_buttons.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton("Add manual")
        add_btn.clicked.connect(self._add_manual_meta_row)
        from_hdf5_btn = QPushButton("Add from HDF5…")
        # Wired in Step 6 — opens a dataset picker rooted at the active
        # raw file's tree.
        from_hdf5_btn.clicked.connect(self._add_meta_from_hdf5)
        del_btn = _set_variant(QPushButton("Remove"), _DANGER)
        del_btn.clicked.connect(self._remove_meta_row)
        meta_buttons.addWidget(add_btn)
        meta_buttons.addWidget(from_hdf5_btn)
        meta_buttons.addWidget(del_btn)
        meta_buttons.addStretch(1)
        meta_btn_widget = QWidget()
        meta_btn_widget.setLayout(meta_buttons)
        section.body_layout.addWidget(meta_btn_widget)

        return section

    def _load_smpl_yaml(self) -> None:
        path = file_dialogs.open_file(
            self, "Load sample metadata",
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text()
        except OSError as exc:
            self.append_log(f"Failed to read {path}: {exc}")
            return
        self.smpl_yaml.setPlainText(text)

    def _save_smpl_yaml(self) -> None:
        path, _ = file_dialogs.save_file(
            self, "Save sample metadata",
            "YAML (*.yaml *.yml);;All files (*)",
            suggested_name="sample.yaml",
            default_suffix="yaml",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.smpl_yaml.toPlainText())
        except OSError as exc:
            self.append_log(f"Failed to write {path}: {exc}")

    def _add_manual_meta_row(self) -> None:
        row = self.exp_meta_table.rowCount()
        self.exp_meta_table.insertRow(row)
        self.exp_meta_table.setItem(row, 0, QTableWidgetItem(""))
        self.exp_meta_table.setItem(row, 1, QTableWidgetItem(""))
        src = QTableWidgetItem("manual")
        src.setFlags(src.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.exp_meta_table.setItem(row, 2, src)

    def _remove_meta_row(self) -> None:
        rows = sorted({i.row() for i in self.exp_meta_table.selectedItems()},
                      reverse=True)
        for r in rows:
            self.exp_meta_table.removeRow(r)

    def _add_meta_from_hdf5(self) -> None:
        """Open a dataset picker rooted at one of the loaded raw files.

        The picker exposes the HDF5 tree of every raw input. Selecting a
        dataset reads its first scalar/string value (or first element
        for arrays, since the user is typically pointing at a metadata
        scalar) and adds a row to the experimental metadata table with
        the dataset path as the source.
        """
        if not self._raw_inputs:
            self.append_log(
                "Add from HDF5: no raw files loaded. Open a raw session first."
            )
            return
        files = [fp for fp, _entries in self._raw_inputs]
        result = _Hdf5MetaPicker.pick(self, files)
        if result is None:
            return
        key_default, value, source = result
        row = self.exp_meta_table.rowCount()
        self.exp_meta_table.insertRow(row)
        self.exp_meta_table.setItem(row, 0, QTableWidgetItem(key_default))
        self.exp_meta_table.setItem(row, 1, QTableWidgetItem(value))
        src_item = QTableWidgetItem(source)
        src_item.setFlags(src_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.exp_meta_table.setItem(row, 2, src_item)

    # ---------------- Section: Conversion config ----------------

    def _build_conversion_config_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Conversion config", expanded=False)

        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems([GEOM_GID, GEOM_TRANSMISSION])
        self.geometry_combo.currentTextChanged.connect(self._on_geometry_changed)
        form.addRow("Geometry:", self.geometry_combo)

        self.conv_type_combo = QComboBox()
        # Initially populated for GID; ``_on_geometry_changed`` rebuilds
        # the list when transmission is chosen.
        self.conv_type_combo.addItems([CONV_DET2Q_GID, CONV_DET2POL_GID])
        self.conv_type_combo.currentTextChanged.connect(self._on_conv_type_changed)
        form.addRow("Conversion:", self.conv_type_combo)

        # Defaults match the pygid example notebook
        # (``CoordMaps(..., vert_positive=True, hor_positive=True)``) — the
        # author labels these "(optional, recommended)" because pygid's
        # bare-default (both False) often lands the converted image in
        # the negative quadrant depending on detector flips, which is
        # rarely what the user wants when reviewing a single frame.
        self.vert_positive_chk = QCheckBox("vert_positive")
        self.vert_positive_chk.setChecked(True)
        self.vert_positive_chk.setToolTip(
            "Constrain the q_z range to non-negative values during conversion. "
            "Recommended (matches the pygid example notebook). Uncheck to keep "
            "any natural negative q_z extent in the converted output."
        )
        self.hor_positive_chk = QCheckBox("hor_positive")
        self.hor_positive_chk.setChecked(True)
        self.hor_positive_chk.setToolTip(
            "Constrain the q_xy range to non-negative values during conversion. "
            "Recommended (matches the pygid example notebook). Uncheck to keep "
            "any natural negative q_xy extent in the converted output."
        )
        orient_row = QHBoxLayout()
        orient_row.setContentsMargins(0, 0, 0, 0)
        orient_row.addWidget(self.vert_positive_chk)
        orient_row.addWidget(self.hor_positive_chk)
        orient_row.addStretch(1)
        orient_widget = QWidget()
        orient_widget.setLayout(orient_row)
        form.addRow("Orientation:", orient_widget)

        section.body_layout.addLayout(form)

        # Stack of parameter sub-forms. The visible page swaps based on
        # conv_type so the user only sees parameters that apply.
        self._param_stack = QStackedWidget()
        # Page 0: det2q_gid (q_xy_range / q_z_range / dq)
        self._param_pages: dict[str, QWidget] = {}
        self._param_pages[CONV_DET2Q_GID] = _build_q_gid_params(self)
        self._param_pages[CONV_DET2Q] = _build_q_trans_params(self)
        self._param_pages[CONV_DET2POL_GID] = _build_pol_params(self, gid=True)
        self._param_pages[CONV_DET2POL] = _build_pol_params(self, gid=False)
        for page in self._param_pages.values():
            self._param_stack.addWidget(page)
        section.body_layout.addWidget(self._param_stack)

        # Initial page matches the default conv_type.
        self._show_param_page(self.conv_type_combo.currentText())

        return section

    def _on_geometry_changed(self, geom: str) -> None:
        # Re-populate conv_type combo to only show variants compatible
        # with the chosen geometry. The user picks det2q vs det2pol
        # within the right family.
        self.conv_type_combo.blockSignals(True)
        try:
            self.conv_type_combo.clear()
            if geom == GEOM_GID:
                self.conv_type_combo.addItems([CONV_DET2Q_GID, CONV_DET2POL_GID])
            else:
                self.conv_type_combo.addItems([CONV_DET2Q, CONV_DET2POL])
        finally:
            self.conv_type_combo.blockSignals(False)
        self._show_param_page(self.conv_type_combo.currentText())

    def _on_conv_type_changed(self, conv: str) -> None:
        self._show_param_page(conv)

    def _show_param_page(self, conv: str) -> None:
        page = self._param_pages.get(conv)
        if page is not None:
            self._param_stack.setCurrentWidget(page)

    # ---------------- Section: Output ----------------

    def _build_output_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Output", expanded=True)

        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)

        self.output_dir = QLineEdit()
        # The placeholder shows the live default (_refresh_output_dir_default).
        self._default_output_dir_shown: Path | None = None
        self.output_dir.setPlaceholderText(
            "Default: the folder of the first checked raw file"
        )
        self.output_dir.setToolTip(
            "Where the converted files are written. Leave it empty to "
            "write next to the raw data, in the folder of the first "
            "checked raw file. With raw files from several folders, "
            "every output goes to that first file's folder."
        )
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output_dir)
        form.addRow("Directory:", _row(self.output_dir, out_browse))
        self.output_dir.textChanged.connect(self._refresh_runnable)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(
            [OUTPUT_SEPARATE_FILES, OUTPUT_SEPARATE_DATASETS,
             OUTPUT_SINGLE_ENTRY]
        )
        self.output_mode_combo.currentTextChanged.connect(
            self._update_output_filename_placeholder
        )
        self.output_mode_combo.currentTextChanged.connect(
            self._on_output_mode_changed
        )
        form.addRow("Save as:", self.output_mode_combo)

        # Optional output filename. Behaviour depends on the mode above
        # (the placeholder text reflects the active rule):
        #   separate-files (default): blank → "{stem}_converted.h5"
        #   separate-files w/ prefix:  prefix appended with raw stem
        #   separate-datasets:         blank → "converted.h5"
        self.output_filename = QLineEdit()
        self._update_output_filename_placeholder(
            self.output_mode_combo.currentText()
        )
        form.addRow("Filename:", self.output_filename)

        self.overwrite_file_chk = QCheckBox("Overwrite existing file")
        self.overwrite_file_chk.setChecked(True)
        self.overwrite_dataset_chk = QCheckBox("Overwrite existing dataset")
        flags_row = QHBoxLayout()
        flags_row.setContentsMargins(0, 0, 0, 0)
        flags_row.addWidget(self.overwrite_file_chk)
        flags_row.addWidget(self.overwrite_dataset_chk)
        flags_row.addStretch(1)
        flags_widget = QWidget()
        flags_widget.setLayout(flags_row)
        form.addRow("Overwrite:", flags_widget)

        # Append-frames mode: instead of landing in a fresh entry_NNNN,
        # the converted frames extend an EXISTING entry's image stack
        # (pygid resizes the dataset and adds the per-frame analysis
        # groups). The combo lists the target file's entries and is
        # refreshed whenever the resolved output path can change.
        self.append_frames_chk = QCheckBox("Append frames to existing entry")
        self.append_frames_chk.setToolTip(
            "Add the converted image(s) as new frames of an entry that "
            "already exists in the output file, instead of creating a "
            "new entry. The frame shape must match the existing stack "
            "(same q-grid); on mismatch pygid falls back to a new "
            "sibling group and warns."
        )
        self.append_entry_combo = QComboBox()
        self.append_entry_combo.setEnabled(False)
        self.append_entry_combo.setMinimumWidth(140)
        append_row = QHBoxLayout()
        append_row.setContentsMargins(0, 0, 0, 0)
        append_row.addWidget(self.append_frames_chk)
        append_row.addWidget(self.append_entry_combo)
        append_row.addStretch(1)
        append_widget = QWidget()
        append_widget.setLayout(append_row)
        form.addRow("Append:", append_widget)
        self.append_frames_chk.toggled.connect(self._on_append_frames_toggled)
        # Anything that changes which file the output resolves to must
        # refresh the entry list. The line edits use editingFinished
        # (focus-out / Enter), NOT textChanged: refreshing per keystroke
        # would re-open the output HDF5 on every character — a network
        # round-trip per key when the output directory is remote.
        self.output_dir.editingFinished.connect(self._refresh_append_entries)
        self.output_filename.editingFinished.connect(self._refresh_append_entries)
        self.output_mode_combo.currentTextChanged.connect(
            self._refresh_append_entries
        )

        section.body_layout.addLayout(form)
        return section

    def _on_append_frames_toggled(self, checked: bool) -> None:
        """Append mode must not truncate: lock the overwrite boxes off
        while it's on, and populate the target-entry combo."""
        self.append_entry_combo.setEnabled(checked)
        self.overwrite_file_chk.setEnabled(not checked)
        self.overwrite_dataset_chk.setEnabled(not checked)
        if checked:
            self.overwrite_file_chk.setChecked(False)
            self.overwrite_dataset_chk.setChecked(False)
            self._refresh_append_entries()

    def _resolve_append_target(self) -> Path | None:
        """The single output file appending would target, or None.

        Reuses the engine's ``_plan_output_paths`` naming rules on the
        currently checked scans; resolvable only when every scan maps to
        ONE output path (always true in separate-datasets mode; true in
        separate-files mode for a single raw file).
        """
        out_dir = self._effective_output_dir()
        if out_dir is None:
            return None
        from mlgidlab import conversion  # lazy: defer the engine module until a run is configured

        cfg = ConversionConfig()
        cfg.output_mode = self.output_mode_combo.currentText()
        cfg.output_filename = self.output_filename.text().strip()
        try:
            scans = self._collect_scans()
        except ValueError:
            scans = []
        if not scans:
            # No selection yet — only the shared-file modes have a
            # selection-independent target.
            if cfg.output_mode not in (
                OUTPUT_SEPARATE_DATASETS, OUTPUT_SINGLE_ENTRY
            ):
                return None
            scans = [RawScan(file_path=Path("_"), entry="_")]
        try:
            outputs = conversion._plan_output_paths(
                scans, cfg, out_dir.expanduser()
            )
        except Exception:
            logger.debug("suppressed output-path plan in append refresh", exc_info=True)
            return None
        targets = set(outputs.values())
        return targets.pop() if len(targets) == 1 else None

    def _refresh_append_entries(self, *_args) -> None:
        """Repopulate the append-target entry combo from the output file.

        Cheap: ``list_entry_names`` reads only the file's top-level link
        names. Keeps the current selection when it survives the refresh;
        otherwise defaults to the LAST entry (the most recent one).
        """
        if not self.append_frames_chk.isChecked():
            return
        previous = self.append_entry_combo.currentText()
        self.append_entry_combo.clear()
        target = self._resolve_append_target()
        if target is None or not target.is_file():
            return
        try:
            names = list_entry_names(target)
        except Exception:
            logger.debug("suppressed entry listing in append refresh", exc_info=True)
            return
        self.append_entry_combo.addItems(names)
        if previous and previous in names:
            self.append_entry_combo.setCurrentText(previous)
        elif names:
            self.append_entry_combo.setCurrentIndex(len(names) - 1)

    def _default_output_dir(self) -> Path | None:
        """The folder of the first checked raw file, or None."""
        entries = self._checked_entries()
        return entries[0].file_path.parent if entries else None

    def _effective_output_dir(self) -> Path | None:
        """The typed directory, else the default next to the raw data."""
        out_text = self.output_dir.text().strip()
        return Path(out_text) if out_text else self._default_output_dir()

    def _refresh_output_dir_default(self) -> None:
        """Show the default directory in the empty field, and re-list the
        append targets when it moves with the selection."""
        default = self._default_output_dir()
        if default == self._default_output_dir_shown:
            return
        self._default_output_dir_shown = default
        self.output_dir.setPlaceholderText(
            f"Default: {default}" if default is not None
            else "Default: the folder of the first checked raw file"
        )
        if not self.output_dir.text().strip():
            self._refresh_append_entries()

    def _browse_output_dir(self) -> None:
        path = file_dialogs.existing_directory(
            self, "Select output directory"
        )
        if path:
            self.output_dir.setText(path)
            # setText does not fire editingFinished — refresh explicitly
            # so an enabled append combo tracks the new directory.
            self._refresh_append_entries()

    def _update_output_filename_placeholder(self, mode: str) -> None:
        """Adjust the filename placeholder so the user knows what blank means."""
        if mode in (OUTPUT_SEPARATE_DATASETS, OUTPUT_SINGLE_ENTRY):
            self.output_filename.setPlaceholderText(
                "Optional. Default: converted.h5"
            )
        else:
            self.output_filename.setPlaceholderText(
                "Optional. Default: {raw_stem}_converted.h5  (or prefix for batches)"
            )

    def _on_output_mode_changed(self, mode: str) -> None:
        """Single-entry mode and append-to-existing are mutually
        exclusive (one creates a fresh entry, the other requires an
        existing one) — lock the append controls out while it's active."""
        single = mode == OUTPUT_SINGLE_ENTRY
        if single and self.append_frames_chk.isChecked():
            self.append_frames_chk.setChecked(False)
        self.append_frames_chk.setEnabled(not single)

    # ---------------- Run wiring ----------------

    def _is_runnable(self) -> bool:
        """Whether the Convert button should be enabled.

        Requires: at least one entry checked and a non-empty PONI path.
        The output directory is optional (``_collect_run_inputs`` falls
        back to the raw data's folder). ``set_running`` overrides this
        when a run is in flight.
        """
        if not self.poni_path.text().strip():
            return False
        return self._has_checked_entries()

    def _has_checked_entries(self) -> bool:
        for i in range(self.selection_tree.topLevelItemCount()):
            file_item = self.selection_tree.topLevelItem(i)
            for j in range(file_item.childCount()):
                if file_item.child(j).checkState(0) == Qt.CheckState.Checked:
                    return True
        return False

    def _refresh_runnable(self) -> None:
        self.btn_convert.setEnabled(self._is_runnable())
        # Every selection change passes through here.
        self._refresh_output_dir_default()

    def _on_convert_clicked(self) -> None:
        """Collect every section's state into a config + scan list and
        emit ``conversionRunRequested``. MainWindow spawns the worker.
        """
        try:
            scans, cfg = self._collect_run_inputs()
        except ValueError as exc:
            self.append_log(f"Cannot start conversion: {exc}")
            return
        self.conversionRunRequested.emit(cfg, scans)

    # ---------------- Run-input collection ----------------

    def _collect_run_inputs(self) -> tuple[list[RawScan], ConversionConfig]:
        """Gather every panel field into ``(scans, ConversionConfig)``.

        Raises ``ValueError`` for inputs that are too malformed to send
        to the engine (e.g. an unparseable frame list). The Convert
        button is already gated on the obvious required fields, so
        these errors are typically just frame-mode parse failures or
        invalid override values.
        """
        scans = self._collect_scans()
        if not scans:
            raise ValueError("No entries selected — tick at least one in the Selection tree.")
        cfg = self._collect_config()
        if cfg.output_dir is None:
            folder = scans[0].file_path.parent
            if not os.access(folder, os.W_OK | os.X_OK):
                raise ValueError(
                    f"no output directory is set and the raw data's folder "
                    f"{folder} is not writable. Choose an output directory "
                    f"in the Output section."
                )
            cfg.output_dir = folder
            self.append_log(
                f"No output directory set: writing next to the raw data, "
                f"in {folder}"
            )
        if isinstance(cfg.ai, list):
            axis = self._selection_frame_axis()
            if len(cfg.ai) != axis:
                # Name both numbers: with the ramp form the count is one
                # more than what was typed, so "wrong length" alone
                # sends the user looking in the wrong place.
                raise ValueError(
                    f"{len(cfg.ai)} angles of incidence were given, but "
                    f"{axis} frames are selected. A ramp gives one more "
                    f"angle than its step count."
                )
            self.append_log(
                f"Angle of incidence: {ai_values.describe(cfg.ai)}"
            )
        return scans, cfg

    def _collect_scans(self) -> list[RawScan]:
        frame_num = self._resolve_frame_num()
        scans: list[RawScan] = []
        base = 0
        for re in self._checked_entries():
            if getattr(re, "frame_map", None) is not None:
                # Fabio stack: one per-file scan per selected frame.
                scans.extend(_expand_fabio_scans(re, frame_num, base))
            else:
                scans.append(
                    RawScan(
                        file_path=re.file_path,
                        entry=re.dataset_path,
                        frame_num=frame_num,
                        frame_offset=base,
                    )
                )
            base += int(re.shape[0])
        return scans

    def _checked_entries(self) -> list[RawEntry]:
        """Every ticked entry, in tree order."""
        found: list[RawEntry] = []
        for i in range(self.selection_tree.topLevelItemCount()):
            file_item = self.selection_tree.topLevelItem(i)
            for j in range(file_item.childCount()):
                child = file_item.child(j)
                if child.checkState(0) != Qt.CheckState.Checked:
                    continue
                re: RawEntry | None = child.data(0, Qt.ItemDataRole.UserRole)
                if re is not None:
                    found.append(re)
        return found

    def _selection_frame_axis(self) -> int:
        """How many frames the ticked entries hold, all told.

        This is the axis a per-frame angle list is indexed against --
        the frames of the *data*, not of the current frame-mode subset,
        because that is how pygid looks an angle up (``ai[frame +
        offset]``). Converting frames 3 and 7 of a fourteen-image stack
        still wants fourteen angles.
        """
        return sum(int(re.shape[0]) for re in self._checked_entries())

    def _refresh_ai_hint(self, *_args) -> None:
        """Echo what the angle field's text resolved to, or why it didn't.

        Worth a line of its own because the ramp form gives one MORE
        angle than its step count, and finding that out from a converted
        file is far too late.
        """
        text = self.ai_input.text().strip()
        n_frames = self._selection_frame_axis()
        if not text:
            self._set_ai_hint("", error=False)
            return
        try:
            value = ai_values.parse_ai(text, n_frames=n_frames)
        except ValueError as exc:
            self._set_ai_hint(str(exc), error=True)
            return
        described = ai_values.describe(value)
        if isinstance(value, list) and n_frames and len(value) != n_frames:
            self._set_ai_hint(
                f"{described} — but {n_frames} frames are selected",
                error=True,
            )
            return
        self._set_ai_hint(described, error=False)

    def _set_ai_hint(self, text: str, error: bool) -> None:
        self.ai_hint.setText(text)
        self.ai_hint.setProperty("status", "error" if error else "muted")
        style = self.ai_hint.style()
        style.unpolish(self.ai_hint)
        style.polish(self.ai_hint)
        self.ai_hint.setVisible(bool(text))

    def _resolve_frame_num(self) -> int | list[int] | None:
        mode = self.frame_mode.currentText()
        if mode == FRAME_ALL:
            return None
        if mode == FRAME_SINGLE:
            text = self.frame_single.text().strip()
            if not text:
                raise ValueError("Frame mode is 'Single' but no frame index was given.")
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(f"Frame index is not an integer: {text!r}") from exc
        if mode == FRAME_LIST:
            text = self.frame_list.text().strip()
            if not text:
                raise ValueError("Frame mode is 'List' but no indices were given.")
            try:
                return [int(p.strip()) for p in text.split(",") if p.strip()]
            except ValueError as exc:
                raise ValueError(f"Frame list contains a non-integer entry: {text!r}") from exc
        return None

    def _collect_config(self) -> ConversionConfig:
        cfg = ConversionConfig()
        cfg.geometry = self.geometry_combo.currentText()
        cfg.conv_type = self.conv_type_combo.currentText()
        cfg.vert_positive = self.vert_positive_chk.isChecked()
        cfg.hor_positive = self.hor_positive_chk.isChecked()

        # Range / step parameters per conversion type.
        if cfg.conv_type == CONV_DET2Q_GID:
            cfg.dq = _spin_or_none(self.dq_q_gid)
            cfg.q_xy_range = _range_or_none(self.q_xy_min, self.q_xy_max)
            cfg.q_z_range = _range_or_none(self.q_z_min, self.q_z_max)
        elif cfg.conv_type == CONV_DET2Q:
            cfg.dq = _spin_or_none(self.dq_q_trans)
            cfg.q_x_range = _range_or_none(self.q_x_min, self.q_x_max)
            cfg.q_y_range = _range_or_none(self.q_y_min, self.q_y_max)
        elif cfg.conv_type == CONV_DET2POL_GID:
            cfg.dq = _spin_or_none(self.dq_pol_gid)
            cfg.dang = _spin_or_none(self.dang_pol_gid)
            cfg.radial_range = _range_or_none(
                self.radial_min_gid, self.radial_max_gid
            )
            cfg.angular_range = _range_or_none(
                self.angular_min_gid, self.angular_max_gid
            )
        elif cfg.conv_type == CONV_DET2POL:
            cfg.dq = _spin_or_none(self.dq_pol)
            cfg.dang = _spin_or_none(self.dang_pol)
            cfg.radial_range = _range_or_none(self.radial_min, self.radial_max)
            cfg.angular_range = _range_or_none(self.angular_min, self.angular_max)

        # Experimental parameters.
        poni_text = self.poni_path.text().strip()
        cfg.poni_path = Path(poni_text) if poni_text else None
        mask_text = self.mask_path.text().strip()
        cfg.mask_path = Path(mask_text) if mask_text else None
        ai_text = self.ai_input.text().strip()
        cfg.ai = (
            ai_values.parse_ai(
                ai_text, n_frames=self._selection_frame_axis()
            )
            if ai_text
            else None
        )

        # Manual overrides — value-driven: any field left at "(unset)" is
        # skipped (pygid reads it from the PONI); a set field is forwarded
        # and wins. The orientation flips sit OUTSIDE the override
        # subsection (routine per-beamline settings) but travel in the
        # same dict because pygid takes them as ExpParams kwargs too.
        overrides: dict = {}
        for attr, key in (
            ("over_centerX", "centerX"),
            ("over_centerY", "centerY"),
            ("over_SDD", "SDD"),
            ("over_wavelength", "wavelength"),
        ):
            v = _spin_or_none(getattr(self, attr))
            if v is not None and not self._unedited_autofill(key, v):
                overrides[key] = v
        if self.transp.isChecked():
            overrides["transp"] = True
        if self.flip_lr.isChecked():
            overrides["fliplr"] = True
        if self.flip_ud.isChecked():
            overrides["flipud"] = True
        cfg.expmeta_overrides = overrides

        # Metadata.
        cfg.smplmeta_yaml = self.smpl_yaml.toPlainText()
        kv: dict[str, str] = {}
        for r in range(self.exp_meta_table.rowCount()):
            key_item = self.exp_meta_table.item(r, 0)
            val_item = self.exp_meta_table.item(r, 1)
            key = key_item.text().strip() if key_item is not None else ""
            value = val_item.text().strip() if val_item is not None else ""
            if key:
                kv[key] = value
        cfg.expmeta_kv = kv

        # Output.
        out_text = self.output_dir.text().strip()
        cfg.output_dir = Path(out_text) if out_text else None
        cfg.output_mode = self.output_mode_combo.currentText()
        cfg.output_filename = self.output_filename.text().strip()
        cfg.overwrite_file = self.overwrite_file_chk.isChecked()
        cfg.overwrite_dataset = self.overwrite_dataset_chk.isChecked()
        cfg.append_frames = self.append_frames_chk.isChecked()
        cfg.append_entry = self.append_entry_combo.currentText().strip()
        if cfg.append_frames:
            if not cfg.append_entry:
                raise ValueError(
                    "Append frames is checked but no target entry is "
                    "selected — the output file must already exist and "
                    "contain at least one entry."
                )
            # Appending must never truncate, whatever the boxes said
            # before they were locked.
            cfg.overwrite_file = False
            cfg.overwrite_dataset = False

        return cfg
