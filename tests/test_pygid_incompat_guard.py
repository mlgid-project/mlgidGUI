"""Pre-flight guard: pipeline.execute refuses files pygid can't open.

``pygid.NexusFile.read_structure`` iterates *every* top-level group and
unconditionally indexes ``root[f"/{entry}/data"]``
(``pygid/nexus_reader.py::get_entry_type``). A single raw-style or
stray-metadata top-level group therefore brings down the whole
``mlgidBASE`` open with an opaque h5py ``KeyError``. The fix lives in
``file_model.list_pygid_incompatible_top_level`` (detection) and
``pipeline.execute`` (pre-flight RuntimeError). These tests pin the
contract: the detector flags the same group set pygid would crash on,
and the pipeline never reaches mlgidBASE when bad groups exist.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.pipeline import PipelineCommand, execute


def test_detects_raw_style_top_level(tmp_path):
    """A raw eiger-style file with ``/entry/data0/image`` (no ``/data``)
    surfaces ``entry`` as incompatible — this is the exact shape of the
    bug the guard was added for."""
    path = tmp_path / "raw_eiger.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data0")
        g.create_dataset("image", data=np.zeros((1, 64, 64), dtype=np.uint32))
    assert file_model.list_pygid_incompatible_top_level(path) == ["entry"]


def test_clean_nexus_passes(synthetic_nexus):
    """The minimal valid NeXus fixture has one entry with a populated
    ``data/img_gid_q`` — pygid would open it fine."""
    assert file_model.list_pygid_incompatible_top_level(synthetic_nexus) == []


def test_stray_top_level_group_flagged(synthetic_nexus):
    """Adding a sibling group with no ``/data`` reproduces the second
    failure mode users hit — a metadata / log group injected next to
    real entries."""
    with h5py.File(synthetic_nexus, "r+") as f:
        f.create_group("extra_metadata")
    assert file_model.list_pygid_incompatible_top_level(synthetic_nexus) == [
        "extra_metadata"
    ]


def test_signal_pointing_at_missing_dataset_flagged(tmp_path):
    """A ``/data`` group whose ``signal`` attr names a dataset that
    isn't actually there would crash pygid at the second probe
    (``root[entry/data/signal].shape``). The detector mirrors that
    probe, so this case is caught too."""
    path = tmp_path / "lying_signal.h5"
    with h5py.File(path, "w") as f:
        data = f.create_group("entry/data")
        data.attrs["signal"] = "img_gid_q"  # but no such dataset
    assert file_model.list_pygid_incompatible_top_level(path) == ["entry"]


def test_execute_raises_friendly_error_on_raw_file(tmp_path):
    """``pipeline.execute`` must short-circuit before mlgidBASE is
    imported / called, so the user sees the actionable RuntimeError
    instead of pygid's KeyError stack."""
    path = tmp_path / "raw_eiger.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data0")
        g.create_dataset("image", data=np.zeros((1, 64, 64), dtype=np.uint32))
    with pytest.raises(RuntimeError) as exc_info:
        execute(path, PipelineCommand("delete_peak",
                                      {"entry": "entry", "frame_num": 0, "peak_id": 0}))
    msg = str(exc_info.value)
    assert "delete_peak" in msg
    assert "'entry'" in msg
    assert "/data" in msg
