"""Pipeline "Frame range…" scope: the Frames dropdown's third option
restricts an operation to typed frames (``frame_range`` grammar,
0-based, inclusive ranges), riding the existing ``frame_num`` kwarg
that mlgidbase accepts as a list. Bad expressions block the run with a
warning instead of enqueuing anything."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidlab.pipeline_panel import (
    FRAME_ACTIVE,
    FRAME_ALL,
    FRAME_RANGE,
    PipelinePanel,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qtbot, monkeypatch):
    # The frame-scope UI is pure Qt — no backend call happens until a
    # command actually RUNS — but the panel builds only a stub without
    # mlgidbase. Force availability so these tests cover the real
    # widgets on backend-less CI too.
    from mlgidlab import pipeline_panel as panel_mod

    monkeypatch.setattr(panel_mod, "is_mlgidbase_available", lambda: True)
    p = PipelinePanel()
    qtbot.addWidget(p)
    p.set_frame_count_resolver(lambda: 6)
    p.set_active_frame_resolver(lambda: 2)
    return p


def _commands(panel):
    got = []
    panel.runRequested.connect(got.append)
    return got


def test_frame_range_edit_shown_only_for_range_scope(panel):
    assert panel.fit_frame_scope.currentText() == FRAME_ALL
    assert panel.fit_frame_range.isHidden()
    panel.fit_frame_scope.setCurrentText(FRAME_RANGE)
    assert not panel.fit_frame_range.isHidden()
    panel.fit_frame_scope.setCurrentText(FRAME_ALL)
    assert panel.fit_frame_range.isHidden()


def test_frame_range_scope_injects_frame_list(panel):
    got = _commands(panel)
    panel.fit_frame_scope.setCurrentText(FRAME_RANGE)
    panel.fit_frame_range.setText("1-3, 5")
    panel._on_run_fitting()
    assert len(got) == 1
    assert got[0].op_name == "run_fitting"
    assert got[0].kwargs["frame_num"] == [1, 2, 3, 5]
    # The other scopes are untouched: Active frame still injects the
    # resolver's int, All frames leaves the kwarg out entirely.
    panel.det_frame_scope.setCurrentText(FRAME_ACTIVE)
    panel._on_run_detection()
    assert got[1].kwargs["frame_num"] == 2
    panel.det_frame_scope.setCurrentText(FRAME_ALL)
    panel._on_run_detection()
    assert "frame_num" not in got[2].kwargs


def test_frame_range_invalid_blocks_run(panel, monkeypatch):
    warnings: list = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    got = _commands(panel)
    panel.fit_frame_scope.setCurrentText(FRAME_RANGE)
    panel.fit_frame_range.setText("abc")
    panel._on_run_fitting()
    assert got == [] and len(warnings) == 1
    # Syntactically fine but entirely outside the 6-frame scan.
    panel.fit_frame_range.setText("10-20")
    panel._on_run_fitting()
    assert got == [] and len(warnings) == 2


def test_frame_range_partly_outside_runs_with_valid_frames(panel):
    got = _commands(panel)
    logs: list = []
    panel.logMessage.connect(logs.append)
    panel.fit_frame_scope.setCurrentText(FRAME_RANGE)
    panel.fit_frame_range.setText("4-9")
    panel._on_run_fitting()
    assert len(got) == 1
    assert got[0].kwargs["frame_num"] == [4, 5]
    assert any("ignoring" in m for m in logs)


def test_fit_form_starts_from_the_mlgidbase_defaults(panel):
    """An unedited Fitting form is a bare run_fitting(): the clustering
    distances are 2 / 2 since mlgidbase 0.1.11 (they were 10 / 10), and
    the numbers come from manual_fit so every fallback agrees."""
    from mlgidlab import manual_fit

    assert panel.fit_dist_peaks.value() == manual_fit.DEFAULT_CLUSTERING_DISTANCE_PEAKS == 2.0
    assert panel.fit_dist_rings.value() == manual_fit.DEFAULT_CLUSTERING_DISTANCE_RINGS == 2.0
    assert panel.fit_cluster_extend.value() == manual_fit.DEFAULT_CLUSTERING_EXTEND == 2
    got = _commands(panel)
    panel._on_run_fitting()
    assert len(got) == 1
    kw = got[0].kwargs
    assert kw["clustering_distance_peaks"] == 2.0
    assert kw["clustering_distance_rings"] == 2.0
    assert kw["clustering_extend"] == 2
