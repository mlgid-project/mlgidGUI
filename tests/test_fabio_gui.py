"""Fabio images through the GUI: open -> separate raw entries -> view + convert.

GUI-side counterpart of ``test_fabio_stack``: a TIFF classifies as raw in the
CopyWorker, a RawSession of image files activates with one entry PER file
(built by ``_populate_raw_entries`` via ``list_fabio_entries``), the viewer
renders the selected image through a ``LazyFabioStack``, and the Conversion
panel yields one ``RawScan`` per checked image file.

Source: workers.py ``CopyWorker`` (fabio branch); main_window.py
``_populate_raw_entries`` (fabio branch); conversion_panel.py
``_refresh_selection_tree`` / ``_collect_scans`` / ``_expand_fabio_scans``.

Batch-open behavior (the 1000-image freeze fix): image paths classify
inline in ``_process_open_queue`` (no CopyWorker churn), browser rows
arrive via the chunked ``_queue_tree_inserts`` timer as display-only
``_ImageFileNode`` rows (no decode — silx's real rows pin the full
pixel data per file), clicking a row selects that image in the entry
combo, and the recent-files list updates in one write via
``_add_recent_files``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from mlgidlab import file_model
from mlgidlab.conversion_panel import FRAME_ALL
from mlgidlab.session import RawSession
from mlgidlab.workers import CopyWorker

pytestmark = pytest.mark.gui


def _write_tiffs(tmp_path, n=3, h=48, w=40) -> list[Path]:
    import fabio

    paths: list[Path] = []
    for i in range(n):
        p = tmp_path / f"scan_{i:04d}.tif"
        fabio.tifimage.TifImage(
            data=np.full((h, w), i + 1, dtype=np.int32)
        ).write(str(p))
        paths.append(p)
    return paths


def _activate_fabio(window, paths):
    """Install a fabio RawSession (no pre-cached walk — the fabio branch of
    ``_populate_raw_entries`` builds the per-file entries itself)."""
    session = RawSession.open(paths)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_copyworker_classifies_tiff_as_raw(qtbot, tmp_path):
    """A standalone TIFF classifies as raw with no pre-walked entries (the
    entry is built lazily at activation)."""
    (tiff,) = _write_tiffs(tmp_path, n=1)
    worker = CopyWorker(tiff)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    assert got["error"] is None
    assert got["kind"] == "raw"
    assert got["session"] is None
    assert got["raw_entries"] is None


def test_fabio_files_open_as_separate_entries(main_window, qtbot, tmp_path):
    tiffs = _write_tiffs(tmp_path, n=3)
    session = _activate_fabio(main_window, tiffs)

    assert session.kind == "raw"
    # One entry per file (separate, NOT a combined stack).
    assert main_window.entry_combo.count() == 3
    labels = [main_window.entry_combo.itemText(i) for i in range(3)]
    assert labels == [p.name for p in tiffs]
    # Generous timeout: the first image loads on a worker thread and
    # CI's shared 2-core runners have blown a 5 s budget before.
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
    )
    stack = main_window.viewer._raw_image_stack
    assert isinstance(stack, file_model.LazyFabioStack)
    assert stack.shape == (1, 48, 40)
    assert main_window.viewer.n_frames == 1
    assert float(stack[0].mean()) == 1.0  # first file auto-loaded


def test_fabio_conversion_collects_one_scan_per_file(main_window, qtbot, tmp_path):
    tiffs = _write_tiffs(tmp_path, n=3)
    _activate_fabio(main_window, tiffs)
    # Same slow-runner allowance as test_fabio_files_open_as_separate_entries.
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
    )

    panel = main_window.conversion_panel
    tree = panel.selection_tree
    # One top-level file group per image, each with a single "image (1 frame)".
    assert tree.topLevelItemCount() == 3
    for i in range(3):
        file_item = tree.topLevelItem(i)
        assert file_item.childCount() == 1
        child = file_item.child(0)
        assert "image (1 frame)" in child.text(0)
        child.setCheckState(0, Qt.CheckState.Checked)

    panel.frame_mode.setCurrentText(FRAME_ALL)
    scans = panel._collect_scans()
    assert [s.file_path for s in scans] == tiffs
    assert all(s.entry == "" and s.frame_num is None for s in scans)

    # Checking just one file yields only that file's scan.
    for i in range(3):
        tree.topLevelItem(i).child(0).setCheckState(
            0, Qt.CheckState.Checked if i == 1 else Qt.CheckState.Unchecked
        )
    one = panel._collect_scans()
    assert [s.file_path for s in one] == [tiffs[1]]


def test_select_all_checkbox_round_trip(main_window, qtbot, tmp_path):
    """The Selection section's select-all box: click checks every file
    (one `_collect_scans` covers the batch), click again unchecks, and
    manual per-item edits mirror back as checked/partial/unchecked."""
    tiffs = _write_tiffs(tmp_path, n=3)
    _activate_fabio(main_window, tiffs)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
    )
    panel = main_window.conversion_panel
    tree = panel.selection_tree
    box = panel.select_all_box

    assert box.isEnabled()
    assert box.checkState() == Qt.CheckState.Unchecked

    box.click()
    assert box.checkState() == Qt.CheckState.Checked
    assert all(
        tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        and tree.topLevelItem(i).child(0).checkState(0)
        == Qt.CheckState.Checked
        for i in range(3)
    )
    panel.frame_mode.setCurrentText(FRAME_ALL)
    assert [s.file_path for s in panel._collect_scans()] == tiffs

    box.click()
    assert box.checkState() == Qt.CheckState.Unchecked
    assert panel._collect_scans() == []

    # Manual edits reflect back on the box.
    tree.topLevelItem(0).child(0).setCheckState(0, Qt.CheckState.Checked)
    assert box.checkState() == Qt.CheckState.PartiallyChecked
    for i in (1, 2):
        tree.topLevelItem(i).child(0).setCheckState(
            0, Qt.CheckState.Checked
        )
    assert box.checkState() == Qt.CheckState.Checked
    tree.topLevelItem(1).child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert box.checkState() == Qt.CheckState.PartiallyChecked

    # A click from partial selects EVERYTHING (never lands on partial).
    box.click()
    assert box.checkState() == Qt.CheckState.Checked
    assert len(panel._collect_scans()) == 3


def test_batch_open_images_skips_worker_queue(main_window, qtbot, tmp_path):
    """``_open_paths`` with image files classifies inline on the GUI
    thread: no CopyWorker thread per file, the batch finalizes
    synchronously, browser rows arrive via the chunked insert queue,
    and the recents list records the batch."""
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))

    # Finalized synchronously — no worker thread was ever spawned.
    assert main_window._thread is None
    assert main_window._open_queue == []
    session = main_window.session
    assert session is not None and session.kind == "raw"
    assert session.raw_paths == [p.resolve() for p in tiffs]
    assert main_window.entry_combo.count() == 3

    # Tree rows land chunk by chunk off the insert timer.
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)

    # One batched recents write: last-opened first.
    items = main_window._load_recent_files()
    assert [i["path"] for i in items[:3]] == [
        str(p) for p in reversed(tiffs)
    ]


def test_recent_files_batch_matches_sequential_semantics(main_window, tmp_path):
    """``_add_recent_files`` produces the exact list sequential
    ``_add_recent_file`` pushes would: later items higher, capped at
    ``_MAX_RECENT_FILES``, duplicates bubble to their last occurrence."""
    window = main_window
    window._save_recent_files(
        [{"type": "raw", "path": f"/old/{i}.h5"} for i in range(4)]
    )
    batch = [tmp_path / f"b_{i:03d}.tif" for i in range(12)]
    window._add_recent_files(batch, "raw")
    items = window._load_recent_files()
    expected = [str(p) for p in reversed(batch)][: window._MAX_RECENT_FILES]
    assert [i["path"] for i in items] == expected

    window._save_recent_files([])
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    window._add_recent_files([a, b, a], "raw")
    assert [i["path"] for i in window._load_recent_files()] == [str(a), str(b)]


def test_image_rows_are_lightweight_and_click_to_view(
    main_window, qtbot, tmp_path
):
    """Browser rows for image files are display-only ``_ImageFileNode``s
    (name + icon, never a decoded silx file), and selecting one switches
    the entry combo — and therefore the viewer — to that image."""
    from PySide6.QtCore import QItemSelectionModel, QModelIndex
    from silx.gui.hdf5 import Hdf5TreeModel as _SilxModel

    from mlgidlab.main_window import _ImageFileNode

    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)

    items = [
        model.data(
            model.index(r, 0, QModelIndex()), _SilxModel.H5PY_ITEM_ROLE
        )
        for r in range(3)
    ]
    assert all(isinstance(i, _ImageFileNode) for i in items)
    assert [i._display_name for i in items] == [p.name for p in tiffs]

    # Click the middle row (through the view's proxy): the combo — the
    # actual navigation surface — follows.
    view = main_window.tree
    proxy = view.model()
    view.selectionModel().select(
        proxy.index(1, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect
    )
    assert main_window.entry_combo.currentText() == tiffs[1].name


def test_action_open_uses_fast_widget_dialog(
    main_window, qtbot, tmp_path, monkeypatch
):
    """File -> Open runs Qt's widget dialog (non-native, constant-time
    icon provider — native pickers thumbnail image files and crawl on
    big detector dirs) and feeds the picked files to ``_open_paths``."""
    from mlgidlab import main_window as mw_mod

    tiffs = _write_tiffs(tmp_path, n=3)
    seen: dict = {}

    def fake_exec(dlg):
        seen["non_native"] = dlg.testOption(
            mw_mod.QFileDialog.Option.DontUseNativeDialog
        )
        seen["provider"] = dlg.iconProvider()
        dlg.selectFile(str(tiffs[0]))
        return 1

    monkeypatch.setattr(mw_mod.QFileDialog, "exec", fake_exec)
    main_window._action_open()

    assert seen["non_native"] is True
    assert isinstance(seen["provider"], mw_mod._FastFileIconProvider)
    session = main_window.session
    assert session is not None and session.kind == "raw"
    assert session.raw_paths == [tiffs[0].resolve()]

    # The provider itself answers without touching file contents.
    provider = main_window._file_dialog_icons
    from PySide6.QtCore import QFileInfo

    assert not provider.icon(QFileInfo(str(tiffs[0]))).isNull()
    assert not provider.icon(QFileInfo(str(tmp_path))).isNull()


def test_browser_fill_progress_indicator(
    main_window, qtbot, tmp_path, monkeypatch
):
    """Image batches at or above ``_TREE_PROGRESS_MIN`` drive the
    status-bar "Browser: n/N files" indicator while rows stream in;
    it clears when the fill completes, and small batches never show
    it."""
    from mlgidlab.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_TREE_PROGRESS_MIN", 2)
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))
    # The finalize queued the rows synchronously; the drain timer has
    # not fired yet, so the indicator is up at 0/3.
    assert main_window._tree_insert_queue
    assert not main_window._sb_tree_bar.isHidden()
    assert main_window._sb_tree_label.text() == "Browser: 0/3 files"

    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    assert main_window._sb_tree_bar.isHidden()
    assert main_window._sb_tree_label.text() == ""

    # Below the threshold the indicator never appears.
    monkeypatch.setattr(MainWindow, "_TREE_PROGRESS_MIN", 10)
    sub = tmp_path / "b"
    sub.mkdir()
    more = _write_tiffs(sub, n=3)
    main_window._open_paths(list(more))
    assert main_window._tree_insert_queue
    assert main_window._sb_tree_bar.isHidden()
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)


def test_detach_drops_pending_tree_inserts(main_window, qtbot, tmp_path):
    """Detaching the silx tree mid-fill drops the queued rows (they
    target the model being cleared); the reattach re-queues every
    session's files and the rows come back."""
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))

    main_window._detach_silx_tree()
    assert main_window._tree_insert_queue == []
    assert not main_window._tree_insert_timer.isActive()

    main_window._reattach_silx_tree()
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)


def _selection_files(window) -> list[str]:
    tree = window.conversion_panel.selection_tree
    return [
        tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
    ]


def test_images_dropped_one_by_one_accumulate(main_window, qtbot, tmp_path):
    """Each single-image drop joins the open raw batch: one session, and
    the Conversion panel's Selection gains a row per drop instead of
    showing only the newest file."""
    tiffs = _write_tiffs(tmp_path, n=3)
    for i, p in enumerate(tiffs, start=1):
        main_window._open_paths([p])
        assert len(main_window._sessions) == 1
        assert _selection_files(main_window) == [t.name for t in tiffs[:i]]
        assert main_window.entry_combo.count() == i
        # The viewer follows the file that was just dropped.
        assert main_window.entry_combo.currentText() == p.name

    assert main_window.session.raw_paths == [p.resolve() for p in tiffs]
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)


def test_drop_keeps_existing_selection_state(main_window, tmp_path):
    """A later drop must not clear what is already ticked."""
    tiffs = _write_tiffs(tmp_path, n=2)
    main_window._open_paths([tiffs[0]])
    panel = main_window.conversion_panel
    first = panel.selection_tree.topLevelItem(0)
    first.setCheckState(0, Qt.CheckState.Checked)

    main_window._open_paths([tiffs[1]])

    assert panel.selection_tree.topLevelItem(0).checkState(0) == (
        Qt.CheckState.Checked
    )
    assert panel.selection_tree.topLevelItem(1).checkState(0) == (
        Qt.CheckState.Unchecked
    )
    assert panel.select_all_box.checkState() == Qt.CheckState.PartiallyChecked
    assert panel.select_all_box.isEnabled()


def test_dropping_the_same_image_twice_adds_nothing(main_window, tmp_path):
    (tiff,) = _write_tiffs(tmp_path, n=1)
    main_window._open_paths([tiff])
    main_window._open_paths([tiff])

    assert len(main_window._sessions) == 1
    assert main_window.session.raw_paths == [tiff.resolve()]
    assert _selection_files(main_window) == [tiff.name]
    assert main_window.entry_combo.count() == 1


def test_raw_drop_joins_the_raw_batch_behind_a_nexus_file(
    main_window, synthetic_nexus, tmp_path
):
    """With a NeXus session in front, a dropped raw file still extends
    the existing raw batch and brings it to the front with both files."""
    from mlgidlab.session import NexusSession

    first, second = _write_tiffs(tmp_path, n=2)
    main_window._open_paths([first])
    raw = main_window.session
    nexus = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(nexus)
    main_window._set_active_session(nexus)
    assert main_window.session is nexus

    main_window._open_paths([second])

    assert main_window.session is raw
    assert len([s for s in main_window._sessions if s.kind == "raw"]) == 1
    assert raw.raw_paths == [first.resolve(), second.resolve()]
    assert _selection_files(main_window) == [first.name, second.name]


def test_hdf5_raw_files_dropped_one_by_one_accumulate(
    main_window, qtbot, tmp_path, synthetic_raw
):
    """Same as the image case, through the CopyWorker path HDF5 files use."""
    import shutil

    files = []
    for i in range(3):
        dest = tmp_path / f"raw_{i}.h5"
        shutil.copy(synthetic_raw, dest)
        files.append(dest)

    for i, p in enumerate(files, start=1):
        main_window._open_paths([p])
        qtbot.waitUntil(
            lambda: main_window._thread is None
            and not main_window._open_queue
            and not main_window._pending_raw_paths,
            timeout=15000,
        )
        assert len(main_window._sessions) == 1
        assert _selection_files(main_window) == [f.name for f in files[:i]]
        assert main_window.entry_combo.count() == i


def test_closing_the_raw_session_empties_the_selection(main_window, tmp_path):
    """Closed raw files must leave the Conversion panel's Selection list
    (and the Convert button) instead of lingering until the next load."""
    tiffs = _write_tiffs(tmp_path, n=2)
    main_window._open_paths([tiffs[0]])
    main_window._open_paths([tiffs[1]])
    panel = main_window.conversion_panel
    panel.selection_tree.topLevelItem(0).setCheckState(0, Qt.CheckState.Checked)
    assert _selection_files(main_window) == [t.name for t in tiffs]

    main_window._close_session(main_window.session)

    assert main_window.session is None
    assert _selection_files(main_window) == []
    assert not panel.select_all_box.isEnabled()
    assert not panel.btn_convert.isEnabled()
    assert main_window._raw_entries == {}
    assert main_window._raw_entry_label_by_path == {}


def test_closing_a_raw_session_behind_a_nexus_file_empties_the_selection(
    main_window, synthetic_nexus, tmp_path
):
    from mlgidlab.session import NexusSession

    (tiff,) = _write_tiffs(tmp_path, n=1)
    main_window._open_paths([tiff])
    raw = main_window.session
    nexus = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(nexus)
    main_window._set_active_session(nexus)
    assert _selection_files(main_window) == [tiff.name]

    main_window._close_session(raw)

    assert main_window.session is nexus
    assert _selection_files(main_window) == []


def test_a_file_dropped_after_closing_starts_a_fresh_selection(
    main_window, tmp_path
):
    first, second = _write_tiffs(tmp_path, n=2)
    main_window._open_paths([first])
    main_window._close_session(main_window.session)

    main_window._open_paths([second])

    assert _selection_files(main_window) == [second.name]
    assert main_window.session.raw_paths == [second.resolve()]


def _browser_rows(window) -> list[str]:
    from PySide6.QtCore import QModelIndex

    model = window.tree.findHdf5TreeModel()
    return [
        model.data(model.index(r, 0, QModelIndex()), Qt.ItemDataRole.DisplayRole)
        for r in range(model.rowCount())
    ]


def _open_tiffs(window, qtbot, tmp_path, n=3) -> list[Path]:
    tiffs = _write_tiffs(tmp_path, n=n)
    for p in tiffs:
        window._open_paths([p])
    qtbot.waitUntil(lambda: not window._tree_insert_queue, timeout=10000)
    model = window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == n, timeout=10000)
    return tiffs


def _select_browser_rows(window, rows) -> None:
    from PySide6.QtCore import QItemSelectionModel

    view = window.tree
    flags = QItemSelectionModel.SelectionFlag.ClearAndSelect
    for r in rows:
        view.selectionModel().select(view.model().index(r, 0), flags)
        flags = QItemSelectionModel.SelectionFlag.Select


def test_closing_one_raw_file_removes_only_it_from_the_selection(
    main_window, qtbot, tmp_path
):
    """Ctrl+W on a raw batch closes the file on screen: its Selection row,
    entry and browser row go, the others (and their ticks) stay."""
    tiffs = _open_tiffs(main_window, qtbot, tmp_path)
    panel = main_window.conversion_panel
    panel.selection_tree.topLevelItem(2).setCheckState(0, Qt.CheckState.Checked)
    main_window.entry_combo.setCurrentText(tiffs[1].name)

    main_window._action_close_file()

    assert main_window.session.raw_paths == [tiffs[0].resolve(), tiffs[2].resolve()]
    assert _selection_files(main_window) == [tiffs[0].name, tiffs[2].name]
    assert panel.selection_tree.topLevelItem(1).checkState(0) == (
        Qt.CheckState.Checked
    )
    assert panel.select_all_box.checkState() == Qt.CheckState.PartiallyChecked
    assert _browser_rows(main_window) == [tiffs[0].name, tiffs[2].name]
    assert [
        main_window.entry_combo.itemText(i)
        for i in range(main_window.entry_combo.count())
    ] == [tiffs[0].name, tiffs[2].name]
    assert tiffs[1].name not in {
        e.file_path.name for e in main_window._raw_entries.values()
    }
    # The viewer moved off the closed image onto a remaining one.
    assert main_window.entry_combo.currentText() == tiffs[0].name
    assert main_window.session is not None


def test_closing_a_file_that_is_not_on_screen_keeps_the_view(
    main_window, qtbot, tmp_path
):
    tiffs = _open_tiffs(main_window, qtbot, tmp_path)
    main_window.entry_combo.setCurrentText(tiffs[2].name)

    main_window._remove_raw_files(main_window.session, [str(tiffs[0].resolve())])

    assert _selection_files(main_window) == [tiffs[1].name, tiffs[2].name]
    assert main_window.entry_combo.currentText() == tiffs[2].name


def test_delete_key_removes_only_the_selected_raw_file(
    main_window, qtbot, tmp_path
):
    tiffs = _open_tiffs(main_window, qtbot, tmp_path)
    _select_browser_rows(main_window, [1])

    main_window._remove_selected_file_from_browser()

    assert _selection_files(main_window) == [tiffs[0].name, tiffs[2].name]
    assert _browser_rows(main_window) == [tiffs[0].name, tiffs[2].name]
    assert len(main_window._sessions) == 1


def test_delete_key_removes_every_selected_raw_file(
    main_window, qtbot, tmp_path
):
    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=4)
    _select_browser_rows(main_window, [0, 2])

    main_window._remove_selected_file_from_browser()

    assert _selection_files(main_window) == [tiffs[1].name, tiffs[3].name]
    assert _browser_rows(main_window) == [tiffs[1].name, tiffs[3].name]


def test_closing_the_last_raw_file_closes_the_session(
    main_window, qtbot, tmp_path
):
    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=2)
    main_window._action_close_file()
    assert len(main_window.session.raw_paths) == 1
    main_window._action_close_file()

    assert main_window.session is None
    assert _selection_files(main_window) == []
    assert not main_window.conversion_panel.btn_convert.isEnabled()
    assert _browser_rows(main_window) == []


def test_a_closed_raw_file_can_be_dropped_again(main_window, qtbot, tmp_path):
    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=2)
    main_window._remove_raw_files(main_window.session, [str(tiffs[0].resolve())])
    assert _selection_files(main_window) == [tiffs[1].name]

    main_window._open_paths([tiffs[0]])

    assert main_window.session.raw_paths == [tiffs[1].resolve(), tiffs[0].resolve()]
    assert _selection_files(main_window) == [tiffs[1].name, tiffs[0].name]


def test_closing_a_raw_file_behind_a_nexus_file_updates_the_selection(
    main_window, qtbot, synthetic_nexus, tmp_path
):
    from mlgidlab.session import NexusSession

    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=2)
    raw = main_window.session
    nexus = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(nexus)
    main_window._set_active_session(nexus)

    main_window._remove_raw_files(raw, [str(tiffs[0].resolve())])

    assert main_window.session is nexus
    assert raw.raw_paths == [tiffs[1].resolve()]
    assert _selection_files(main_window) == [tiffs[1].name]


def test_closing_one_hdf5_raw_file_keeps_the_others(
    main_window, qtbot, tmp_path, synthetic_raw
):
    import shutil

    files = []
    for i in range(2):
        dest = tmp_path / f"raw_{i}.h5"
        shutil.copy(synthetic_raw, dest)
        files.append(dest)
        main_window._open_paths([dest])
        qtbot.waitUntil(
            lambda: main_window._thread is None
            and not main_window._open_queue
            and not main_window._pending_raw_paths,
            timeout=15000,
        )
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 2, timeout=10000)

    main_window._remove_raw_files(main_window.session, [str(files[0].resolve())])

    assert _selection_files(main_window) == [files[1].name]
    assert main_window.session.raw_paths == [files[1].resolve()]
    assert model.rowCount() == 1
    assert main_window.session.display_path == files[1].resolve()
    assert all(
        e.file_path.resolve() == files[1].resolve()
        for e in main_window._raw_entries.values()
    )


def test_shift_selecting_rows_and_pressing_delete_closes_them(
    main_window, qtbot, tmp_path
):
    """The real route: extended selection in the browser, then the Delete
    key on the tree."""
    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=5)
    view = main_window.tree
    proxy = view.model()
    from PySide6.QtCore import QItemSelection, QItemSelectionModel

    sel = QItemSelection(proxy.index(1, 0), proxy.index(3, 0))
    view.selectionModel().select(
        sel,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    assert len(view.selectionModel().selectedRows()) == 3

    qtbot.keyClick(view, Qt.Key.Key_Delete)

    assert _selection_files(main_window) == [tiffs[0].name, tiffs[4].name]
    assert _browser_rows(main_window) == [tiffs[0].name, tiffs[4].name]
    assert main_window.session.raw_paths == [
        tiffs[0].resolve(), tiffs[4].resolve()
    ]


def test_a_multi_row_selection_does_not_move_the_viewer(
    main_window, qtbot, tmp_path
):
    """A single click follows the row; adding a second row to the
    selection (Ctrl-click) leaves the viewer where it was."""
    from PySide6.QtCore import QItemSelectionModel

    tiffs = _open_tiffs(main_window, qtbot, tmp_path, n=3)
    view = main_window.tree
    proxy = view.model()
    view.selectionModel().select(
        proxy.index(0, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect
    )
    assert main_window.entry_combo.currentText() == tiffs[0].name

    view.selectionModel().select(
        proxy.index(2, 0), QItemSelectionModel.SelectionFlag.Select
    )

    picked = [i for i in view.selectedIndexes() if i.column() == 0]
    assert len(picked) == 2
    assert main_window.entry_combo.currentText() == tiffs[0].name
