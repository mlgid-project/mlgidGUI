"""File browser Delete-key removal.

Covers the binding added so testers can press ``Delete`` on a
selected file-browser row to remove that file, mirroring
``File → Close`` (``Ctrl+W``). Two layers:

* ``_MlgidHdf5TreeView.keyPressEvent`` emits ``deleteFileRequested``
  *only* when the tree holds a selection (so a stray Delete with
  nothing selected falls through to silx instead of firing a no-op
  close). Tested on a standalone view to isolate the key→signal map
  from the window's heavy close machinery.
* ``MainWindow._remove_selected_file_from_browser`` resolves the
  selected node back to its session and closes exactly that file,
  leaving other open files untouched — via the same confirm +
  ``_close_session`` path as Ctrl+W.

The window-level test drives the handler with a stub node carrying a
``local_filename`` (the first accessor ``_node_filename`` tries),
which deterministically exercises the node→session resolution
without depending on silx's offscreen ``selectedH5Nodes`` behaviour.
"""

from __future__ import annotations

import shutil

from PySide6.QtCore import QEvent, QItemSelectionModel, QModelIndex, Qt
from PySide6.QtGui import QKeyEvent

from mlgidlab.session import NexusSession


def _open(window, path) -> NexusSession:
    """Open + activate, appending to ``_sessions`` first so the silx
    tree rebuild (inside ``_set_active_session``) reinserts the file —
    mirroring ``_on_open_finished`` (see test_smoke_silx_detach)."""
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def _delete_key() -> QKeyEvent:
    return QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
    )


def test_delete_emits_only_with_a_selection(qtbot, synthetic_nexus):
    """The tree view fires ``deleteFileRequested`` on Delete when a row
    is selected, and stays silent when nothing is selected."""
    from mlgidlab.main_window import _MlgidHdf5TreeView

    view = _MlgidHdf5TreeView()
    qtbot.addWidget(view)
    view.findHdf5TreeModel().insertFile(str(synthetic_nexus))
    model = view.model()
    assert model.rowCount(QModelIndex()) >= 1  # insertFile is synchronous

    received: list[int] = []
    view.deleteFileRequested.connect(lambda: received.append(1))

    # No selection → Delete falls through, no signal.
    view.keyPressEvent(_delete_key())
    assert received == []

    # Select the file root → Delete emits exactly once.
    idx = model.index(0, 0, QModelIndex())
    view.selectionModel().select(
        idx,
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    assert view.selectionModel().hasSelection()
    view.keyPressEvent(_delete_key())
    assert received == [1]


def test_handler_closes_only_the_selected_file(
    main_window, synthetic_nexus, tmp_path
):
    """``_remove_selected_file_from_browser`` closes the session the
    selected node belongs to and leaves the other open file alone."""
    s1 = _open(main_window, synthetic_nexus)
    second = tmp_path / "second.h5"
    shutil.copy2(synthetic_nexus, second)
    s2 = _open(main_window, second)
    assert len(main_window._sessions) == 2

    # Stub node resolving (via local_filename) to the *non-active* s1.
    class _Node:
        local_filename = str(s1.temp_path)

    main_window._safe_selected_h5_nodes = lambda: [_Node()]
    main_window._remove_selected_file_from_browser()

    assert s1 not in main_window._sessions
    assert s2 in main_window._sessions


def test_handler_noops_without_a_resolvable_session(
    main_window, synthetic_nexus
):
    """A selection that maps to no live session is a safe no-op — the
    open file is untouched (mirrors a no-active-session Ctrl+W)."""
    s1 = _open(main_window, synthetic_nexus)

    class _Orphan:
        local_filename = "/nonexistent/orphan.h5"

    main_window._safe_selected_h5_nodes = lambda: [_Orphan()]
    main_window._remove_selected_file_from_browser()
    assert s1 in main_window._sessions

    # Empty selection is likewise a no-op.
    main_window._safe_selected_h5_nodes = lambda: []
    main_window._remove_selected_file_from_browser()
    assert s1 in main_window._sessions


class _Node:
    def __init__(self, path) -> None:
        self.local_filename = str(path)


def test_the_browser_allows_selecting_several_rows(main_window):
    from PySide6.QtWidgets import QAbstractItemView

    assert main_window.tree.selectionMode() == (
        QAbstractItemView.SelectionMode.ExtendedSelection
    )


def test_delete_closes_every_selected_converted_file(
    main_window, synthetic_nexus, tmp_path
):
    s1 = _open(main_window, synthetic_nexus)
    second = tmp_path / "second.h5"
    third = tmp_path / "third.h5"
    shutil.copy2(synthetic_nexus, second)
    shutil.copy2(synthetic_nexus, third)
    s2 = _open(main_window, second)
    s3 = _open(main_window, third)
    # Two rows of the same file count once; the active file (s3) goes too.
    main_window._safe_selected_h5_nodes = lambda: [
        _Node(s1.temp_path), _Node(s1.temp_path), _Node(s3.temp_path),
    ]

    main_window._remove_selected_file_from_browser()

    assert main_window._sessions == [s2]
    assert main_window.session is s2


def test_cancelling_the_unsaved_prompt_closes_nothing(
    main_window, synthetic_nexus, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    s1 = _open(main_window, synthetic_nexus)
    second = tmp_path / "second.h5"
    shutil.copy2(synthetic_nexus, second)
    s2 = _open(main_window, second)
    s2.dirty = True
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )
    main_window._safe_selected_h5_nodes = lambda: [
        _Node(s1.temp_path), _Node(s2.temp_path),
    ]

    main_window._remove_selected_file_from_browser()

    assert main_window._sessions == [s1, s2]
    # The window's own teardown must not ask again.
    s2.dirty = False


def test_delete_closes_converted_and_raw_files_together(
    main_window, synthetic_nexus, synthetic_raw, tmp_path
):
    from mlgidlab.session import RawSession

    s1 = _open(main_window, synthetic_nexus)
    raw_files = []
    for name in ("a.h5", "b.h5"):
        f = tmp_path / name
        shutil.copy2(synthetic_raw, f)
        raw_files.append(f)
    raw = RawSession.open(raw_files)
    main_window._sessions.append(raw)
    main_window._set_active_session(raw)
    main_window._safe_selected_h5_nodes = lambda: [
        _Node(s1.temp_path), _Node(raw_files[0]),
    ]

    main_window._remove_selected_file_from_browser()

    assert main_window._sessions == [raw]
    assert raw.raw_paths == [raw_files[1].resolve()]
