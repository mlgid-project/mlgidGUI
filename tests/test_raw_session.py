"""Raw-file discovery and RawSession lifecycle.

Pure logic (no QApplication). Source: ``file_model.list_raw_entries``
and ``session.RawSession``. NeXus/raw classification of opened files is
covered by the CopyWorker tests (it classifies inline, off the GUI
thread).
"""

from __future__ import annotations

import pytest

from mlgidlab import file_model
from mlgidlab.session import RawSession


def test_list_raw_entries_applies_size_and_ndim_filter(synthetic_raw):
    entries = file_model.list_raw_entries(synthetic_raw)
    assert len(entries) == 1
    e = entries[0]
    assert e.dataset_path == "raw/data0/image"
    assert e.shape == (4, 64, 64)


def test_raw_session_open_empty_list_raises_valueerror():
    with pytest.raises(ValueError):
        RawSession.open([])


def test_raw_session_open_missing_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        RawSession.open([tmp_path / "does_not_exist.h5"])


def test_raw_session_open_happy(synthetic_raw):
    session = RawSession.open([synthetic_raw])
    assert session.kind == "raw"
    assert session.raw_paths == [synthetic_raw.resolve()]
    assert session.temp_path == synthetic_raw.resolve()


def test_add_paths_appends_and_skips_known_files(tmp_path):
    from mlgidlab.session import RawSession

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    a.write_bytes(b"")
    b.write_bytes(b"")
    session = RawSession.open([a])
    assert session.raw_path_strs == frozenset({str(a.resolve())})

    added = session.add_paths([a, b])

    assert added == [b.resolve()]
    assert session.raw_paths == [a.resolve(), b.resolve()]
    # The cached string set follows the list.
    assert session.raw_path_strs == frozenset(
        {str(a.resolve()), str(b.resolve())}
    )
    assert session.add_paths([b]) == []


def test_add_paths_rejects_a_missing_file(tmp_path):
    import pytest

    from mlgidlab.session import RawSession

    a = tmp_path / "a.h5"
    a.write_bytes(b"")
    session = RawSession.open([a])
    with pytest.raises(FileNotFoundError):
        session.add_paths([tmp_path / "missing.h5"])
    assert session.raw_paths == [a.resolve()]


def test_remove_paths_drops_files_and_moves_the_first(tmp_path):
    files = []
    for name in ("a.h5", "b.h5", "c.h5"):
        f = tmp_path / name
        f.write_bytes(b"x")
        files.append(f)
    session = RawSession.open(files)

    removed = session.remove_paths([files[0], tmp_path / "unknown.h5"])

    assert removed == [files[0].resolve()]
    assert session.raw_paths == [files[1].resolve(), files[2].resolve()]
    assert session.temp_path == files[1].resolve()
    assert str(files[0].resolve()) not in session.raw_path_strs


def test_remove_paths_refuses_to_empty_the_session(tmp_path):
    f = tmp_path / "a.h5"
    f.write_bytes(b"x")
    session = RawSession.open([f])
    with pytest.raises(ValueError):
        session.remove_paths([f])
    assert session.raw_paths == [f.resolve()]
