"""File-browser (silx tree) entry selection on a multi-entry file.

Clicking an entry node in the tree must switch the displayed entry just
like the Entry combo does. On a master that links many external scans
this used to fail: ``_on_tree_selection_changed`` fed the silx Data
viewer first, and ``DataViewerFrame.setData`` eagerly resolves the
entry's external-linked NXdata signal — slow enough to freeze, and
because it ran first it aborted the entry switch. The fix switches the
entry first and defers / guards the Data-viewer render.

Source: main_window.py ``_on_tree_selection_changed``,
``_activate_entry_for_node``, ``_set_or_defer_data_node``,
``_set_data_node``, ``_on_main_tab_changed``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


class _FakeNode:
    """Stand-in for a silx h5 node: only ``local_name`` is read by
    ``_node_entry_name`` to resolve the entry group."""

    def __init__(self, path: str) -> None:
        self.local_name = path


def _two_entry_nexus(tmp_path):
    path = tmp_path / "two_entry.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        for i in range(2):
            data = f.create_group(f"entry_{i:04d}/data", track_order=True)
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
            )
            data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
            data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
    return path


def _open(window, path):
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def test_tree_node_switches_entry(main_window, tmp_path):
    """A node inside ``entry_0001`` switches the combo (and so the viewer)
    to that entry — the core file-browser behaviour."""
    _open(main_window, _two_entry_nexus(tmp_path))
    assert main_window.entry_combo.currentText() == "entry_0000"

    main_window._activate_entry_for_node(_FakeNode("/entry_0001/data/img_gid_q"))

    assert main_window.entry_combo.currentText() == "entry_0001"


def test_set_data_node_swallows_setdata_error(main_window, tmp_path, monkeypatch):
    """A slow / failing ``setData`` (huge external-link resolve) must not
    propagate out of the click — it only feeds the Data tab."""
    _open(main_window, _two_entry_nexus(tmp_path))

    def _boom(_node):
        raise RuntimeError("external link resolve blew up")

    monkeypatch.setattr(main_window.data_viewer, "setData", _boom)
    main_window._pending_data_node = object()

    main_window._set_data_node(_FakeNode("/entry_0001"))  # must not raise

    assert main_window._pending_data_node is None


def test_data_render_deferred_until_data_tab(main_window, tmp_path, monkeypatch):
    """Clicking a node while the Image tab is showing defers the Data-tab
    render; switching to the Data tab flushes it exactly once."""
    _open(main_window, _two_entry_nexus(tmp_path))
    main_window.tabs.setCurrentWidget(main_window.viewer)  # ensure Image tab

    seen: list[object] = []
    monkeypatch.setattr(main_window.data_viewer, "setData", seen.append)

    node = _FakeNode("/entry_0001")
    main_window._set_or_defer_data_node(node)

    assert seen == []  # not rendered while hidden
    assert main_window._pending_data_node is node

    main_window.tabs.setCurrentWidget(main_window.data_viewer)

    assert seen == [node]  # flushed on tab switch
    assert main_window._pending_data_node is None
