"""Shallow entry listing on the open path.

Opening a master that links many external scans must NOT resolve those
links: h5py holds the GIL across each external open, so resolving 226 of
them over a network share freezes the GUI even from a worker thread.
``list_entry_names`` reads only the master's link names (``keys()`` does
not dereference external links); ``list_entries`` / ``list_entry_signals``
do resolve them and are off the open path. Pure file_model logic — runs
in the CI subset. Source: file_model.py ``list_entry_names``.
"""

from __future__ import annotations

import h5py
import numpy as np

from mlgidgui import file_model


def test_list_entry_names_does_not_resolve_externals(tmp_path):
    """The names come back even when every external target is missing —
    proof that ``keys()`` does not open the linked scans."""
    master = tmp_path / "master.h5"
    with h5py.File(master, "w", track_order=True) as f:
        f["entry_0000"] = h5py.ExternalLink("missing_a.h5", "/")
        f["entry_0001"] = h5py.ExternalLink("missing_b.h5", "/")
        f.create_group("metadata")  # non-entry top-level group is skipped

    assert file_model.list_entry_names(master) == ["entry_0000", "entry_0001"]


def test_list_entry_names_matches_q_entries_for_a_normal_file(tmp_path):
    """For a normal (resolvable) pygid file the shallow name list equals
    the q-filtered ``list_entries`` — they only diverge on broken / mixed
    files, which the open path tolerates."""
    path = tmp_path / "normal.h5"
    with h5py.File(path, "w", track_order=True) as f:
        for i in range(3):
            data = f.create_group(f"entry_{i:04d}/data", track_order=True)
            data.attrs["signal"] = "img_gid_q"
            data.create_dataset(
                "img_gid_q", data=np.zeros((1, 4, 4), dtype=np.float32)
            )

    names = file_model.list_entry_names(path)
    assert names == ["entry_0000", "entry_0001", "entry_0002"]
    assert names == file_model.list_entries(path)
