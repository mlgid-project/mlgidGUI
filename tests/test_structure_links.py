"""Links: creating them, retargeting them, unlinking them, following them.

Stage five of the editor. The rule under all of it is that a link is
described, never resolved, unless the user asks — which is what keeps a
master file whose entries are external links to multi-GB scans quick to
browse. Every test here that touches an external link points it at a
file that does NOT exist, so anything that quietly resolved would fail
loudly instead of passing slowly.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QMenu, QMessageBox

from mlgidlab import h5_edit
from mlgidlab.h5_edit_ops import DeleteLinkOp, RetargetLinkOp
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


class _FakeNode:
    def __init__(self, filename, local_name: str) -> None:
        self.local_filename = str(filename)
        self.local_name = local_name


@pytest.fixture
def editing(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window._reattach_silx_tree()
    return main_window


def _target(window, h5_path: str):
    from pathlib import Path
    return (Path(window.session.temp_path), h5_path)


def _show(window, h5_path: str, monkeypatch) -> None:
    node = _FakeNode(window.session.temp_path, h5_path)
    monkeypatch.setattr(window, "_safe_selected_h5_nodes", lambda: [node])
    window._on_tree_selection_changed()


def _stub_link_dialog(monkeypatch, kind, name, target, filename=""):
    from mlgidlab import main_window_structure as mws

    class _Stub:
        def __init__(self, *a, **k):
            self.kwargs = k

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_values(self):
            return (kind, name, target, filename)

    monkeypatch.setattr(mws, "NewLinkDialog", _Stub)


def _cancel_link_dialog(monkeypatch):
    from mlgidlab import main_window_structure as mws

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def deleteLater(self):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def result_values(self):  # pragma: no cover - never reached
            return ("soft", "", "", "")

    monkeypatch.setattr(mws, "NewLinkDialog", _Stub)


def _read(window):
    """The open write handle when there is one.

    Before the first edit there is none, so fall back to naming the
    keys through a plain read — a second ``r`` open is legal either way.
    """
    handle = window._h5_edit_handle
    if handle is not None and handle.is_open:
        return handle.file
    return h5py.File(window.session.temp_path, "r")


# -- creating --------------------------------------------------------------


def test_create_a_soft_link(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "qxy_alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    link = _read(editing)["entry_0000"].get("qxy_alias", getlink=True)
    assert isinstance(link, h5py.SoftLink)
    assert link.path == "/entry_0000/data/q_xy"


def test_create_a_hard_link(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "hard", "qz_alias", "/entry_0000/data/q_z")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    f = _read(editing)
    assert isinstance(f["entry_0000"].get("qz_alias", getlink=True), h5py.HardLink)
    assert np.array_equal(
        f["entry_0000/qz_alias"][()], f["entry_0000/data/q_z"][()])


def test_create_an_external_link_to_a_file_that_is_not_there(editing, monkeypatch):
    """A link is a pointer, not a promise: it may dangle, legally."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(
        monkeypatch, "external", "scan", "/entry_0000/data",
        filename="/nowhere/absent.h5",
    )
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    link = _read(editing)["entry_0000"].get("scan", getlink=True)
    assert isinstance(link, h5py.ExternalLink)
    assert link.filename == "/nowhere/absent.h5"


def test_an_external_link_with_no_file_is_refused(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "external", "scan", "/entry_0000/data")
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    assert seen and "needs a file" in str(seen[0])
    assert "scan" not in _read(editing)["entry_0000"]


def test_a_cancelled_dialog_creates_nothing(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _cancel_link_dialog(monkeypatch)
    with h5py.File(editing.session.temp_path, "r") as f:
        before = set(f["entry_0000"].keys())
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    assert set(_read(editing)["entry_0000"].keys()) == before


def test_creating_a_link_is_undoable(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    assert "alias" in _read(editing)["entry_0000"]
    editing._structure_undo()
    assert "alias" not in _read(editing)["entry_0000"]
    editing._structure_redo()
    assert "alias" in _read(editing)["entry_0000"]


def test_the_changes_list_names_the_link_kind(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    assert editing.structure_panel.change_rows() == [
        "+ /entry_0000/alias  (soft link)"]


# -- retargeting -----------------------------------------------------------


def test_retarget_a_soft_link(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_z")
    editing._on_structure_retarget_link(_target(editing, "/entry_0000/alias"))
    link = _read(editing)["entry_0000"].get("alias", getlink=True)
    assert link.path == "/entry_0000/data/q_z"


def test_retargeting_is_undoable_both_ways(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_z")
    editing._on_structure_retarget_link(_target(editing, "/entry_0000/alias"))

    editing._structure_undo()
    assert _read(editing)["entry_0000"].get(
        "alias", getlink=True).path == "/entry_0000/data/q_xy"
    editing._structure_redo()
    assert _read(editing)["entry_0000"].get(
        "alias", getlink=True).path == "/entry_0000/data/q_z"


def test_a_soft_link_can_become_an_external_one(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    _stub_link_dialog(
        monkeypatch, "external", "alias", "/entry_0000/data",
        filename="/nowhere/absent.h5")
    editing._on_structure_retarget_link(_target(editing, "/entry_0000/alias"))
    link = _read(editing)["entry_0000"].get("alias", getlink=True)
    assert isinstance(link, h5py.ExternalLink)


def test_a_hard_link_cannot_be_retargeted(editing, monkeypatch):
    _show(editing, "/entry_0000/data/q_xy", monkeypatch)
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a[1:3])))
    editing._on_structure_retarget_link(_target(editing, "/entry_0000/data/q_xy"))
    assert seen and "hard link" in str(seen[0])


def test_the_panel_button_retargets_the_link_on_show(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    _show(editing, "/entry_0000/alias", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_z")
    editing.structure_panel.retargetLinkRequested.emit()
    assert _read(editing)["entry_0000"].get(
        "alias", getlink=True).path == "/entry_0000/data/q_z"


# -- unlinking -------------------------------------------------------------


def test_deleting_a_link_removes_only_the_link(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    editing._on_structure_delete(_target(editing, "/entry_0000/alias"))
    f = _read(editing)
    assert "alias" not in f["entry_0000"]
    assert "q_xy" in f["entry_0000/data"], "the target must be untouched"


def test_unlinking_is_undoable_without_touching_the_target(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    editing._on_structure_delete(_target(editing, "/entry_0000/alias"))
    editing._structure_undo()
    link = _read(editing)["entry_0000"].get("alias", getlink=True)
    assert isinstance(link, h5py.SoftLink)
    assert link.path == "/entry_0000/data/q_xy"


def test_a_dangling_external_link_can_be_deleted_and_restored(editing, monkeypatch):
    """The case a snapshot could never handle: the target does not exist."""
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(
        monkeypatch, "external", "scan", "/entry_0000/data",
        filename="/nowhere/absent.h5")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    editing._on_structure_delete(_target(editing, "/entry_0000/scan"))
    assert "scan" not in _read(editing)["entry_0000"].keys()

    editing._structure_undo()
    link = _read(editing)["entry_0000"].get("scan", getlink=True)
    assert isinstance(link, h5py.ExternalLink)
    assert link.filename == "/nowhere/absent.h5"


def test_deleting_a_link_is_recorded_as_a_link(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    editing._on_structure_delete(_target(editing, "/entry_0000/alias"))
    assert editing.structure_panel.change_rows()[-1] == (
        "- /entry_0000/alias  (soft link)")


# -- describing and following ---------------------------------------------


def test_an_external_link_is_described_without_being_opened(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(
        monkeypatch, "external", "scan", "/entry_0000/data",
        filename="/nowhere/absent.h5")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    _show(editing, "/entry_0000/scan", monkeypatch)
    panel = editing.structure_panel
    assert panel.current.kind == "link"
    assert panel.current.link.kind == "external"
    assert "absent.h5" in panel._link_label.text()
    assert panel._follow_btn.isVisibleTo(panel._link_card)


def test_following_a_soft_link_shows_the_target(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))

    _show(editing, "/entry_0000/alias", monkeypatch)
    # A soft link is in-file and cheap, so it is already resolved.
    assert editing.structure_panel.current.kind == "dataset"
    assert editing.structure_panel.current.shape == (24,)


def test_following_a_broken_external_link_does_not_raise(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(
        monkeypatch, "external", "scan", "/entry_0000/data",
        filename="/nowhere/absent.h5")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    _show(editing, "/entry_0000/scan", monkeypatch)

    editing._on_structure_follow_link()
    # It stays described as a link — there is nothing behind it to show.
    assert editing.structure_panel.current.kind == "link"


def test_following_a_live_external_link_shows_the_target(
    main_window, tmp_path, monkeypatch
):
    linked = tmp_path / "scan.h5"
    with h5py.File(linked, "w") as f:
        f.create_dataset("payload", data=np.zeros((3, 4), dtype="f4"))
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as f:
        f["scan"] = h5py.ExternalLink(str(linked), "/payload")

    node = _FakeNode(master, "/scan")
    main_window._render_structure_node(node)
    assert main_window.structure_panel.current.kind == "link"

    monkeypatch.setattr(main_window, "_safe_selected_h5_nodes", lambda: [node])
    main_window._on_structure_follow_link()
    followed = main_window.structure_panel.current
    assert followed.kind == "dataset" and followed.shape == (3, 4)


# -- the context menu ------------------------------------------------------


class _MenuEvent:
    def __init__(self, node, menu):
        self._node, self._menu = node, menu

    def hoveredObject(self):
        return self._node

    def menu(self):
        return self._menu


def _menu_labels(window, h5_path):
    menu = QMenu()
    node = _FakeNode(window.session.temp_path, h5_path)
    window._structure_context_actions(_MenuEvent(node, menu))
    return [a.text() for a in menu.actions() if a.text()]


def test_a_group_offers_new_link(editing):
    assert "New link…" in _menu_labels(editing, "/entry_0000")


def test_a_link_offers_retarget(editing, monkeypatch):
    _show(editing, "/entry_0000", monkeypatch)
    _stub_link_dialog(monkeypatch, "soft", "alias", "/entry_0000/data/q_xy")
    editing._on_structure_new_link(_target(editing, "/entry_0000"))
    labels = _menu_labels(editing, "/entry_0000/alias")
    assert "Retarget link…" in labels
    assert "New group…" not in labels


def test_a_plain_dataset_offers_no_retarget(editing):
    assert "Retarget link…" not in _menu_labels(editing, "/entry_0000/data/q_xy")


# -- the ops, on their own -------------------------------------------------


def test_delete_link_op_round_trips(tmp_path):
    path = tmp_path / "ops.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("real", data=[1, 2, 3])
        f["alias"] = h5py.SoftLink("/real")
    op = DeleteLinkOp("/alias", "soft", "/real")
    with h5py.File(path, "r+") as f:
        op.redo(f)
        assert "alias" not in f.keys()
        op.undo(f)
        assert f.get("alias", getlink=True).path == "/real"


def test_retarget_link_op_round_trips(tmp_path):
    path = tmp_path / "ops.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=[1])
        f.create_dataset("b", data=[2])
        f["alias"] = h5py.SoftLink("/a")
    op = RetargetLinkOp("/alias", "soft", "/a", "", "soft", "/b", "")
    with h5py.File(path, "r+") as f:
        op.redo(f)
        assert f.get("alias", getlink=True).path == "/b"
        op.undo(f)
        assert f.get("alias", getlink=True).path == "/a"
