"""File-browser tree widgets: the silx Hdf5 tree subclasses plus the
lightweight image-file rows and the constant-time icon provider.
Moved out of ``main_window`` in the 2026 source split; the underscore
names are kept because tests import them.
"""
from __future__ import annotations

from PySide6.QtCore import QFileInfo, Qt, Signal
from PySide6.QtWidgets import QApplication, QFileIconProvider, QStyle
from pathlib import Path
from silx.gui.hdf5 import Hdf5TreeModel, Hdf5TreeView
from silx.gui.hdf5.Hdf5Node import Hdf5Node
from silx.gui.hdf5.NexusSortFilterProxyModel import NexusSortFilterProxyModel

import logging

logger = logging.getLogger(__name__)


class _FastFileIconProvider(QFileIconProvider):
    """Constant-time icons for the Open dialog's file listing.

    The default provider inspects every file (mime detection) to pick
    an icon, and the platform-native dialogs go further and thumbnail
    image files — either way a directory holding thousands of detector
    images takes ages to become scrollable. Two cached icons keep the
    per-entry cost at a dict lookup. Used together with
    ``QFileDialog.Option.DontUseNativeDialog`` (the native dialog never
    consults a Qt icon provider).
    """

    def __init__(self) -> None:
        super().__init__()
        style = QApplication.style()
        self._dir_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self._file_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def icon(self, info):  # type: ignore[override]
        if isinstance(info, QFileInfo):
            return self._dir_icon if info.isDir() else self._file_icon
        if info == QFileIconProvider.IconType.Folder:
            return self._dir_icon
        return self._file_icon


class _ImageFileNode(Hdf5Node):
    """Display-only file-browser row for a standalone image file.

    silx's real rows fully decode the file they represent (the fabio
    wrapper keeps the pixels alive), so listing a several-thousand-image
    batch the normal way pinned tens of GB of RAM and ground the view
    down — for rows whose only GUI purpose is click-to-view. This node
    stores just the path, name and icon; ``MainWindow`` resolves clicks
    to the matching entry-dropdown item (``_activate_image_node``) and
    feeds the Data tab a freshly decoded frame on demand.

    Modeled on silx's ``Hdf5LoadingItem`` (the other obj-less node the
    model/view stack already handles): ``obj`` is ``None``, ``h5Class``
    is FILE, and only name/description answer display roles. The nexus
    sort proxy never touches ``obj`` for non-GROUP nodes, and
    ``removeIndex``/``clear`` no-op on ``obj is None``.
    """

    def __init__(self, path: str, icon, parent=None) -> None:
        super().__init__(parent, openedPath=path)
        self.image_path = path
        # What MainWindow._node_filename reads first — lets the shared
        # session-resolution helpers treat this row like any file node.
        self.local_filename = path
        self._icon = icon
        self._display_name = Path(path).name

    @property
    def obj(self):  # noqa: D401 - silx node contract
        return None

    @property
    def h5Class(self):
        import silx.io.utils

        return silx.io.utils.H5Type.FILE

    def dataName(self, role):
        if role == Qt.ItemDataRole.DecorationRole:
            return self._icon
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_name
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.image_path
        return None

    def dataDescription(self, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return "Image file"
        return None


class _MlgidHdf5TreeModel(Hdf5TreeModel):
    """Silx tree model that swaps the file-root icon for raw sessions.

    The default ``Hdf5TreeModel`` uses ``SP_FileIcon`` for every loaded
    HDF5 file. Distinguishing converted-NeXus files (the pipeline runs
    on these) from raw detector files (they need conversion first)
    helps the user spot which is which when both are open in the file
    browser dock at the same time.

    The set of "raw" filesystem paths is owned by ``MainWindow`` and
    pushed in via ``set_raw_paths``; the model emits ``dataChanged``
    so existing rows refresh without a full rebuild.

    All read-only model overrides (``data`` / ``flags`` / ``rowCount`` /
    ``columnCount`` / ``hasChildren`` / ``index``) are wrapped in
    defensive try/except blocks that swallow ``ValueError`` /
    ``RecursionError`` / ``KeyError`` / ``OSError`` / ``RuntimeError``
    and return safe defaults. silx's ``Hdf5Item`` keeps an
    ``h5py.Group`` reference that can outlive the file handle
    (post-pipeline-run detach/reattach, session swap, file close);
    silx's own model methods don't defend against that and raise
    ``ValueError: Invalid group (or file) id`` from inside
    ``len(self.obj)``. Qt's QSortFilterProxyModel then re-fires the
    failing call through every proxy layer, producing a stack-busting
    recursion (40+ frames of ``QSortFilterProxyModel::data`` →
    ``QSortFilterProxyModel::rowCount`` → ``mapToSource``). Swallowing
    the error at our layer stops the storm at its source; the view
    paints a blank row for that frame, which the next natural repaint
    (after the proxy/source rebuild that follows the silx-dance
    completes) overwrites with the correct content.
    """

    # Exceptions raised when an Hdf5Item holds a stale h5py reference
    # or when Qt re-fires a failed call recursively.
    _STALE_EXC = (ValueError, KeyError, OSError, RuntimeError, RecursionError)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._raw_paths: set[str] = set()
        from PySide6.QtWidgets import QApplication, QStyle
        style = QApplication.style()
        self._raw_icon = style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        self._nexus_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def set_raw_paths(self, paths) -> None:
        # No dataChanged.emit here on purpose. Forcing silx's tree to
        # repaint while h5py items may still be in lazy-init has
        # produced reentrancy storms in QSortFilterProxyModel under
        # PySide6. Icons just take effect on the next natural paint
        # (resize / scroll / new insert), which is good enough.
        self._raw_paths = {str(p) for p in paths}

    def insertImageRow(self, path: str) -> None:
        """Append a display-only ``_ImageFileNode`` row for ``path``.

        The node needs the model root as its construction parent
        (silx's ``insertChild`` does not re-parent); ``nodeFromIndex``
        of the invalid index is the documented way to reach it.
        """
        from PySide6.QtCore import QModelIndex

        root = self.nodeFromIndex(QModelIndex())
        self.insertNode(-1, _ImageFileNode(path, self._raw_icon, parent=root))

    def remove_file_rows(self, paths) -> int:
        """Remove the top-level rows of the files in ``paths``; return how
        many went.

        silx offers no remove-by-name, and rebuilding the whole tree to
        drop one file would collapse every other open file and re-insert
        a several-thousand-image batch. Each removal goes through silx's
        ``removeIndex``, which closes a file the model owns.
        """
        from PySide6.QtCore import QModelIndex

        wanted = {str(p) for p in paths}
        root = self.nodeFromIndex(QModelIndex())
        removed = 0
        row = 0
        while row < root.childCount():
            node = root.child(row)
            name = getattr(node, "local_filename", None)
            if name is None:
                try:
                    name = node.obj.file.filename
                except Exception:
                    name = None
            if name is not None and str(name) in wanted:
                self.removeIndex(self.index(row, 0, QModelIndex()))
                removed += 1
            else:
                row += 1
        return removed

    def clear(self) -> None:  # type: ignore[override]
        """Empty the model in ONE reset instead of silx's per-row loop.

        silx's ``clear`` removes row 0 in a loop — each removal is a
        ``beginRemoveRows`` the sort proxy and the view react to, so
        tearing down a big image batch was quadratic and froze every
        tree detach (session close, app exit, each pipeline run).
        Owned h5 handles are still released per node, exactly like
        silx's ``removeIndex`` does.
        """
        from PySide6.QtCore import QModelIndex

        root = self.nodeFromIndex(QModelIndex())
        self.beginResetModel()
        try:
            while root.childCount():
                node = root.removeChildAtIndex(root.childCount() - 1)
                try:
                    self._closeFileIfOwned(node)
                except Exception:
                    logger.debug(
                        "suppressed handle close in clear()", exc_info=True
                    )
        finally:
            self.endResetModel()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DecorationRole
            and index.column() == self.NAME_COLUMN
            and not index.parent().isValid()
        ):
            try:
                node = self.nodeFromIndex(index)
                obj = getattr(node, "obj", None)
                if obj is not None:
                    filename = getattr(obj, "filename", None)
                    if filename:
                        if str(filename) in self._raw_paths:
                            return self._raw_icon
                        return self._nexus_icon
            except self._STALE_EXC:
                # If silx / h5py is in a transient bad state, fall
                # through to super().data() rather than propagating
                # an exception that Qt would re-fire endlessly.
                pass
        try:
            return super().data(index, role)
        except self._STALE_EXC:
            # silx's Hdf5Item.dataDescription walks `len(self.obj)` on
            # a possibly-stale h5py group and propagates a ValueError
            # ("Invalid group (or file) id") through every proxy layer.
            # Returning None lets the view paint a blank cell; the next
            # repaint after the silx-dance completes shows real data.
            return None

    def flags(self, index):
        try:
            return super().flags(index)
        except self._STALE_EXC:
            return Qt.ItemFlag.NoItemFlags

    def rowCount(self, parent=None):
        try:
            if parent is None:
                return super().rowCount()
            return super().rowCount(parent)
        except self._STALE_EXC:
            return 0

    def columnCount(self, parent=None):
        try:
            if parent is None:
                return super().columnCount()
            return super().columnCount(parent)
        except self._STALE_EXC:
            return 0

    def hasChildren(self, parent=None):
        try:
            if parent is None:
                return super().hasChildren()
            return super().hasChildren(parent)
        except self._STALE_EXC:
            return False

    def index(self, row, column, parent=None):
        try:
            if parent is None:
                return super().index(row, column)
            return super().index(row, column, parent)
        except self._STALE_EXC:
            from PySide6.QtCore import QModelIndex
            return QModelIndex()


class _MlgidHdf5TreeView(Hdf5TreeView):
    """Hdf5TreeView that builds its default model from our subclass.

    Also disables silx's built-in file-drop handler so all drag-and-
    drop events fall through to ``MainWindow.dropEvent``. Silx's
    default behaviour is to accept any URL drop on the tree and call
    ``insertFileAsync`` directly — that creates an orphan tree node
    with no matching ``Session`` in our session list, and later
    queries (selection changes, pipeline detach/reattach) blow up
    against the orphan's stale h5py handle.

    Emits ``deleteFileRequested`` when the user presses Delete with a
    tree row selected, so MainWindow can close the file the selection
    belongs to. Scoped to the tree's own ``keyPressEvent`` (fires only
    when the browser holds focus), so it never collides with the image
    viewer's Delete = "delete the selected peak" binding.
    """

    deleteFileRequested = Signal()

    def createDefaultModel(self):
        model = _MlgidHdf5TreeModel(self)
        model.setFileDropEnabled(False)
        proxy = NexusSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        return proxy

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if (
            ev.key() == Qt.Key.Key_Delete
            and self.selectionModel() is not None
            and self.selectionModel().hasSelection()
        ):
            self.deleteFileRequested.emit()
            ev.accept()
            return
        super().keyPressEvent(ev)
