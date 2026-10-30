"""Fabio image-file support: TIFF/CBF/EDF open as raw-mode entries.

Pure-logic coverage of the fabio path added to ``file_model`` and
``conversion_panel`` (no Qt, no network share — TIFFs are generated into
``tmp_path`` with fabio): the ``is_fabio_image`` gate, ``list_fabio_entries``
(one entry PER image file — separate, not a combined stack), ``LazyFabioStack``
(same surface as ``LazyRawStack`` but read via fabio), the ``build_raw_stack``
factory, and the ``_expand_fabio_scans`` frame->file conversion mapping.

Source: file_model.py (``FABIO_EXTENSIONS``, ``is_fabio_image``,
``list_fabio_entries``, ``LazyFabioStack``, ``build_raw_stack``);
conversion_panel.py (``_expand_fabio_scans``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlgidgui import file_model as fm


def _write_tiffs(tmp_path, n=3, h=40, w=50) -> list[Path]:
    """Write ``n`` single-frame int32 TIFFs (constant value i+1) in order."""
    import fabio

    paths: list[Path] = []
    for i in range(n):
        p = tmp_path / f"img_{i:04d}.tif"
        fabio.tifimage.TifImage(
            data=np.full((h, w), i + 1, dtype=np.int32)
        ).write(str(p))
        paths.append(p)
    return paths


def _multiframe_entry(tiffs) -> fm.RawEntry:
    """A synthetic multi-frame fabio entry (frame i = tiffs[i]) — used to
    exercise LazyFabioStack / _expand_fabio_scans over several frames, which
    the per-file ``list_fabio_entries`` no longer produces on its own."""
    return fm.RawEntry(
        file_path=tiffs[0],
        dataset_path="",
        shape=(len(tiffs), 40, 50),
        dtype="int32",
        frame_map=[(t, 0) for t in tiffs],
    )


@pytest.fixture
def tiffs(tmp_path):
    return _write_tiffs(tmp_path)


def test_is_fabio_image():
    assert fm.is_fabio_image("a.tif")
    assert fm.is_fabio_image("A.TIFF")  # case-insensitive
    assert fm.is_fabio_image(Path("x/y.cbf"))
    assert fm.is_fabio_image("z.edf")
    assert not fm.is_fabio_image("f.h5")
    assert not fm.is_fabio_image("f.nxs")
    assert not fm.is_fabio_image("noext")


def test_list_fabio_entries_one_per_file(tiffs):
    entries = fm.list_fabio_entries(tiffs)
    # One separate 1-frame entry per file (NOT a combined stack).
    assert len(entries) == 3
    for entry, path in zip(entries, tiffs):
        assert entry.frame_map == [(path, 0)]
        assert entry.file_path == path
        assert entry.shape == (1, 40, 50)
        assert entry.dataset_path == ""  # pygid ignores dataset for fabio
        assert entry.dtype == "int32"
        assert entry.label == path.name  # labelled by filename
    # A single image opens as one entry.
    assert len(fm.list_fabio_entries(tiffs[:1])) == 1


def test_list_fabio_entries_empty_raises():
    with pytest.raises(ValueError):
        fm.list_fabio_entries([])


def test_lazy_fabio_stack_reads_frames(tiffs):
    stack = fm.LazyFabioStack(_multiframe_entry(tiffs))
    assert stack.ndim == 3 and len(stack) == 3
    assert stack.shape == (3, 40, 50)
    assert stack.dtype == np.float32  # int frames upcast, like LazyRawStack
    stack.acquire()
    for i in range(3):
        frame = stack[i]
        assert frame.shape == (40, 50)
        assert frame.dtype == np.float32
        assert float(frame.mean()) == float(i + 1)
    assert stack[2, 0, 0] == 3.0  # pixel readout
    stack.release()


def test_lazy_fabio_stack_guards(tiffs):
    stack = fm.LazyFabioStack(fm.list_fabio_entries(tiffs)[0])
    with pytest.raises(RuntimeError):  # not acquired
        stack[0]
    stack.acquire()
    with pytest.raises(TypeError):  # no whole-stack materialization
        np.asarray(stack)
    with pytest.raises(TypeError):  # slices unsupported
        stack[1:2]
    stack.release()
    with pytest.raises(RuntimeError):  # released
        stack.get_frame(0)


def test_lazy_fabio_stack_requires_frame_map(synthetic_raw):
    (h5_entry,) = fm.list_raw_entries(synthetic_raw)
    with pytest.raises(ValueError):
        fm.LazyFabioStack(h5_entry)


def test_build_raw_stack_polymorphism(tiffs, synthetic_raw):
    assert isinstance(
        fm.build_raw_stack(fm.list_fabio_entries(tiffs)[0]), fm.LazyFabioStack
    )
    (h5_entry,) = fm.list_raw_entries(synthetic_raw)
    assert isinstance(fm.build_raw_stack(h5_entry), fm.LazyRawStack)


def test_expand_fabio_scans(tiffs):
    from mlgidgui.conversion_panel import _expand_fabio_scans

    # A multi-frame entry maps each selected frame to its own image file.
    entry = _multiframe_entry(tiffs)
    all_scans = _expand_fabio_scans(entry, None)
    assert [s.file_path for s in all_scans] == tiffs
    assert all(s.entry == "" and s.frame_num is None for s in all_scans)
    one = _expand_fabio_scans(entry, 1)
    assert len(one) == 1 and one[0].file_path == tiffs[1]
    assert [s.file_path for s in _expand_fabio_scans(entry, [0, 2])] == [
        tiffs[0], tiffs[2]
    ]
    assert _expand_fabio_scans(entry, 9) == []
    # A real per-file entry has a single frame -> a single scan for its file.
    single = fm.list_fabio_entries(tiffs)[2]
    assert [s.file_path for s in _expand_fabio_scans(single, None)] == [tiffs[2]]
