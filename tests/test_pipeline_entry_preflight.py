"""Pre-flight check: ``pipeline.execute`` must reject a stale or
nonexistent ``entry`` kwarg with a clear RuntimeError naming the
available entries — *before* mlgidBASE / pygidfit gets invoked and
raises its opaque ``ValueError("entry not found in the NeXus
file")`` from deep inside ``ProcessDataFromFile``.

The bug this guards against: the entry combo can carry a stale name
across a file modification (entry deleted externally, or the
selection survived a file swap). pygidfit then raises a message
naming neither the picked entry nor the actually-available ones, so
the user has to inspect the file manually to recover.
"""
from __future__ import annotations

import pytest

from mlgidgui.pipeline import PipelineCommand, execute


def test_execute_rejects_unknown_entry(synthetic_nexus):
    """A non-existent entry triggers a RuntimeError naming the
    available ones, raised by our pre-flight rather than by
    pygidfit. The fixture file carries ``entry_0000`` only, so the
    error names that as the available option."""
    cmd = PipelineCommand(
        "run_fitting",
        {"entry": "not_a_real_entry", "frame_num": 0, "crit_angle": 0.0,
         "clustering_distance_peaks": 10.0,
         "clustering_distance_rings": 10.0, "clustering_extend": 2,
         "theta_fixed": False, "use_pool": False, "debug": False},
    )
    with pytest.raises(RuntimeError, match="entry 'not_a_real_entry' is not present"):
        execute(synthetic_nexus, cmd)


def test_execute_rejects_unknown_entry_lists_available(synthetic_nexus):
    """The error string must include the actually-available entry
    name so the user can recover without inspecting the file."""
    cmd = PipelineCommand(
        "run_fitting",
        {"entry": "wrong_entry"},
    )
    with pytest.raises(RuntimeError) as exc:
        execute(synthetic_nexus, cmd)
    assert "entry_0000" in str(exc.value)


def test_execute_skips_validation_when_no_entry(synthetic_nexus, monkeypatch):
    """A command with no ``entry`` key in kwargs should NOT trigger
    the pre-flight (mlgidbase iterates all entries on its own).
    Monkeypatch mlgidBASE so we only check the pre-flight gate, not
    the heavy mlgidbase invocation that follows.

    Skipped on CI: this test specifically needs the ``mlgidbase``
    module to be importable so it can monkeypatch the ``mlgidBASE``
    symbol. The two preceding tests don't need the import (they
    raise from the pre-flight before ``execute`` reaches the lazy
    mlgidbase import) and run unconditionally."""
    mlgidbase = pytest.importorskip("mlgidbase")
    construct_calls = []

    class _StubAnalysis:
        def __init__(self, filename):
            construct_calls.append(filename)

        def run_fitting(self, **kwargs):
            return "stub-result"

    monkeypatch.setattr(mlgidbase, "mlgidBASE", _StubAnalysis)
    cmd = PipelineCommand("run_fitting", {})  # no entry key
    result = execute(synthetic_nexus, cmd)
    assert result == "stub-result"
    assert len(construct_calls) == 1
