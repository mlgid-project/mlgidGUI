"""Dialogs for creating things in an HDF5 file.

Split out of ``structure_panel`` so the panel stays a panel; the same
split ``phase_views_dialogs`` makes for the phase-views window.

Both dialogs are pure input forms: they collect what the user typed and
hand it back. Nothing here opens a file, and nothing here validates
against one — a name that collides or a shape that will not allocate is
reported by ``h5_edit``, where the file is, in the same message the user
would get from any other failed edit.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlgidgui import file_dialogs, nexus_schema
from mlgidgui.h5_edit import DTYPE_CHOICES
from mlgidgui.widgets import make_form

#: The template entry that means "no template": a plain group with
#: whatever class the user types, or none at all.
PLAIN_GROUP = "Plain group"


class NewGroupDialog(QDialog):
    """Name, NeXus class and optional template for a new group.

    The template is the friendly half: picking *Sample (NXsample)* both
    stamps the class and pre-creates the fields that group nearly always
    has, so the user is editing values rather than looking up what an
    NXsample is supposed to contain.
    """

    def __init__(self, parent: QWidget | None = None, *, parent_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("New group")
        form = make_form(self)

        if parent_path:
            where = QLabel(f"Inside {parent_path}", self)
            where.setProperty("status", "muted")
            form.addRow("", where)

        self.template_combo = QComboBox(self)
        self.template_combo.addItem(PLAIN_GROUP)
        for key in nexus_schema.template_names():
            self.template_combo.addItem(nexus_schema.TEMPLATES[key].label, key)
        self.template_combo.currentIndexChanged.connect(self._on_template)
        form.addRow("Template:", self.template_combo)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("sample, instrument, notes, …")
        form.addRow("Name:", self.name_edit)

        self.class_combo = QComboBox(self)
        self.class_combo.setEditable(True)
        self.class_combo.addItem("")
        self.class_combo.addItems(nexus_schema.NX_CLASSES)
        self.class_combo.currentTextChanged.connect(self._on_class)
        form.addRow("NX_class:", self.class_combo)

        self.help = QLabel("", self)
        self.help.setProperty("status", "muted")
        self.help.setWordWrap(True)
        form.addRow("", self.help)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_template(self, _index: int) -> None:
        key = self.template_combo.currentData()
        if key is None:
            self.help.setText("")
            return
        template = nexus_schema.TEMPLATES[key]
        self.class_combo.setCurrentText(template.nx_class)
        if not self.name_edit.text().strip():
            # A sensible default name: the class without its NX prefix,
            # which is what these groups are conventionally called.
            self.name_edit.setText(template.nx_class[2:].lower())
        fields = ", ".join(f.name for f in template.fields)
        self.help.setText(
            f"{template.help}  Creates: {fields}." if fields else template.help
        )

    def _on_class(self, text: str) -> None:
        if self.template_combo.currentData() is None:
            self.help.setText(nexus_schema.class_help(text.strip()))

    def result_values(self) -> tuple[str, str, str | None]:
        """``(name, nx_class, template_key)``; the key is None for a plain group."""
        return (
            self.name_edit.text().strip(),
            self.class_combo.currentText().strip(),
            self.template_combo.currentData(),
        )


class NewDatasetDialog(QDialog):
    """Name, type, shape and fill value for a new dataset.

    Shape is typed as a comma-separated list, empty meaning a scalar,
    which is how NeXus writes most metadata fields. The units box exists
    because a NeXus number without units is an unfinished field, and
    offering the right ones up front is cheaper than a later audit.
    """

    def __init__(self, parent: QWidget | None = None, *, parent_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("New field")
        form = make_form(self)

        if parent_path:
            where = QLabel(f"Inside {parent_path}", self)
            where.setProperty("status", "muted")
            form.addRow("", where)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("wavelength, exposure_time, …")
        self.name_edit.textChanged.connect(self._on_name)
        form.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(DTYPE_CHOICES)
        self.type_combo.setCurrentText("float64")
        form.addRow("Type:", self.type_combo)

        self.shape_edit = QLineEdit(self)
        self.shape_edit.setPlaceholderText("empty for a single value, or 3, 4")
        form.addRow("Shape:", self.shape_edit)

        self.value_edit = QLineEdit(self)
        self.value_edit.setPlaceholderText("0")
        form.addRow("Value / fill:", self.value_edit)

        self.units_combo = QComboBox(self)
        self.units_combo.setEditable(True)
        self.units_combo.addItems(nexus_schema.GENERIC_UNITS)
        form.addRow("Units:", self.units_combo)

        self.resizable_check = QCheckBox("Growable along the first axis", self)
        self.resizable_check.setToolTip(
            "Allows rows to be appended later without rewriting the dataset.")
        form.addRow("", self.resizable_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_name(self, name: str) -> None:
        """Offer the units the field's name implies, keeping what was typed."""
        current = self.units_combo.currentText()
        self.units_combo.clear()
        self.units_combo.addItems(nexus_schema.suggest_units(name.strip()))
        self.units_combo.setCurrentText(current)

    def result_values(self) -> tuple[str, str, str, str, str, bool]:
        """``(name, dtype, shape_text, value_text, units, resizable)``."""
        return (
            self.name_edit.text().strip(),
            self.type_combo.currentText(),
            self.shape_edit.text().strip(),
            self.value_edit.text().strip(),
            self.units_combo.currentText().strip(),
            self.resizable_check.isChecked(),
        )


def parse_shape(text: str) -> tuple[int, ...]:
    """``"3, 4"`` -> ``(3, 4)``; empty -> ``()``, a scalar.

    Raises ``ValueError`` with a message fit for a dialog.
    """
    text = (text or "").strip().strip("()[]")
    if not text:
        return ()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            "A shape is a comma-separated list of whole numbers, "
            "for example 3, 4. Leave it empty for a single value."
        ) from None
    if any(d < 0 for d in dims):
        raise ValueError("A dimension cannot be negative.")
    return dims


#: What each link kind means, in the dialog, in plain language.
LINK_KIND_HELP = {
    "hard": (
        "A second name for the same object in this file. Deleting either "
        "name leaves the data reachable through the other."
    ),
    "soft": (
        "A pointer to another path in this file. It keeps working if the "
        "target is replaced, and dangles if the target is removed."
    ),
    "external": (
        "A pointer into another HDF5 file. This is how a Bliss master "
        "refers to its scans: the data stays where it is and is only read "
        "when something follows the link."
    ),
}


class NewLinkDialog(QDialog):
    """Kind, name and target for a new link.

    The target is typed rather than picked from a tree: a link may point
    at something that does not exist yet, which is legal HDF5 and
    occasionally what the user means (writing the link before the scan
    lands). The external file, by contrast, gets a real file picker,
    because a mistyped filename is the one failure that looks identical
    to a working link until someone follows it.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        parent_path: str = "",
        kind: str = "soft",
        name: str = "",
        target: str = "",
        filename: str = "",
        retarget: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Retarget link" if retarget else "New link")
        self._retarget = retarget
        form = make_form(self)

        if parent_path:
            where = QLabel(
                f"{'Link' if retarget else 'Inside'} {parent_path}", self)
            where.setProperty("status", "muted")
            form.addRow("", where)

        self.kind_combo = QComboBox(self)
        self.kind_combo.addItem("Soft link (a path in this file)", "soft")
        self.kind_combo.addItem("External link (another file)", "external")
        if not retarget:
            # A hard link is the object itself, so there is nothing to
            # retarget — h5_edit.retarget_link refuses it, and offering
            # it here would only produce that message.
            self.kind_combo.addItem("Hard link (a second name)", "hard")
        index = self.kind_combo.findData(kind)
        if index >= 0:
            self.kind_combo.setCurrentIndex(index)
        self.kind_combo.currentIndexChanged.connect(self._on_kind)
        form.addRow("Kind:", self.kind_combo)

        self.name_edit = QLineEdit(name, self)
        self.name_edit.setPlaceholderText("the name the link appears under")
        if retarget:
            self.name_edit.setEnabled(False)
        form.addRow("Name:", self.name_edit)

        file_row = QWidget(self)
        row = QHBoxLayout(file_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.file_edit = QLineEdit(filename, file_row)
        self.file_edit.setPlaceholderText("the HDF5 file to point into")
        row.addWidget(self.file_edit, 1)
        self.browse_btn = QPushButton("Browse…", file_row)
        self.browse_btn.clicked.connect(self._on_browse)
        row.addWidget(self.browse_btn)
        self.file_row = file_row
        form.addRow("File:", file_row)

        self.target_edit = QLineEdit(target, self)
        self.target_edit.setPlaceholderText("/entry_0000/data")
        form.addRow("Target path:", self.target_edit)

        self.help = QLabel("", self)
        self.help.setProperty("status", "muted")
        self.help.setWordWrap(True)
        form.addRow("", self.help)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._on_kind(self.kind_combo.currentIndex())

    def _on_kind(self, _index: int) -> None:
        kind = self.kind_combo.currentData()
        self.file_row.setVisible(kind == "external")
        self.help.setText(LINK_KIND_HELP.get(kind, ""))

    def _on_browse(self) -> None:
        path = file_dialogs.open_file(
            self, "Link to file",
            "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)",
            suggested_name=self.file_edit.text(),
        )
        if path:
            self.file_edit.setText(path)

    def result_values(self) -> tuple[str, str, str, str]:
        """``(kind, name, target, filename)`` as chosen."""
        return (
            str(self.kind_combo.currentData()),
            self.name_edit.text().strip(),
            self.target_edit.text().strip(),
            self.file_edit.text().strip(),
        )


class PickNodeDialog(QDialog):
    """Choose a group or dataset inside another HDF5 file.

    Backs *Paste from file…*: the source is not open in the app, so
    there is no browser row to right-click. The tree fills one level at
    a time, on expand, through a short-lived read handle.

    Soft and external links are shown as leaves and never followed. A
    master file's 226 entries therefore list instantly, and picking one
    copies the link rather than the multi-GB scan behind it — which is
    the same choice ``h5_edit.copy_node`` makes.
    """

    #: Stored on each row so a selection knows its in-file path without
    #: rebuilding it from the tree's labels.
    PATH_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, file_path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from pathlib import Path

        self.file_path = Path(file_path)
        self.setWindowTitle(f"Copy from {self.file_path.name}")
        self.resize(460, 420)

        column = QVBoxLayout(self)
        hint = QLabel(
            "Pick what to copy. Links are copied as links, so nothing "
            "behind them is opened or duplicated.",
            self,
        )
        hint.setProperty("status", "muted")
        hint.setWordWrap(True)
        column.addWidget(hint)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.itemExpanded.connect(self._on_expanded)
        column.addWidget(self.tree, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

        self._fill(None, "/")

    def _fill(self, parent_item, path: str) -> None:
        """List one group's children as rows, without following links."""
        import h5py

        from mlgidgui import h5_edit

        try:
            with h5py.File(self.file_path, "r") as f:
                group = f[path] if path != "/" else f
                if not isinstance(group, h5py.Group):
                    return
                rows = []
                for name in group.keys():
                    link = group.get(name, getlink=True)
                    child_path = h5_edit.join_path(path, name)
                    if isinstance(link, h5py.ExternalLink):
                        rows.append((name, "external link", child_path, False))
                    elif isinstance(link, h5py.SoftLink):
                        rows.append((name, "soft link", child_path, False))
                    else:
                        obj = group.get(name)
                        if isinstance(obj, h5py.Group):
                            rows.append((name, "group", child_path, len(obj) > 0))
                        elif obj is not None:
                            rows.append(
                                (name, str(obj.dtype), child_path, False))
        except OSError as exc:
            QTreeWidgetItem(self.tree, [f"cannot read: {exc}", ""])
            return

        for name, kind, child_path, expandable in rows:
            item = (QTreeWidgetItem(parent_item, [name, kind])
                    if parent_item is not None
                    else QTreeWidgetItem(self.tree, [name, kind]))
            item.setData(0, self.PATH_ROLE, child_path)
            if expandable:
                # A placeholder makes the row expandable without reading
                # the group; the real children arrive on expand.
                QTreeWidgetItem(item, ["…", ""])

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).text(0) != "…":
            return
        item.takeChildren()
        path = item.data(0, self.PATH_ROLE)
        if path:
            self._fill(item, str(path))

    def selected_path(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        path = item.data(0, self.PATH_ROLE)
        return str(path) if path else None
