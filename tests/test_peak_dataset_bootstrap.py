"""Peaks can be added to a frame that has never been through the pipeline.

``detected_peaks`` / ``fitted_peaks`` only exist once a detection or
fitting run has written them, so before any pipeline stage a frame has
either a bare ``frameNNNNN`` group (what pygid's conversion and
``normalize_for_pygid`` leave behind) or no ``data/analysis`` tree at
all. Every hand-labelling path — Add to detected, Add to fitted, quick
select, paste, and the undo re-adds — goes through ``_add_peak_row``,
which used to refuse both states with "run the appropriate pipeline
stage at least once". That made manual labelling, the one job that has
no reason to need a model, depend on running the model first.

``file_model.ensure_peak_dataset`` now creates the group chain and an
empty dataset instead. An unknown *entry* is still an error: the helper
will bootstrap containers, not invent scans.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from mlgidgui import file_model
from mlgidgui.file_model import ANALYSIS_REL, FRAME_KEY_FMT, PEAK_ROW_DTYPE
from mlgidgui.image_viewer import ManualPeak, SelectedPeak
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui.session import NexusSession

ENTRY = "entry_0000"

GEOM = dict(radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0)


def _ds_path(frame: int, kind: str) -> str:
    return f"{ENTRY}/{ANALYSIS_REL}/{FRAME_KEY_FMT.format(frame)}/{kind}"


def _rows(path, frame: int, kind: str):
    with h5py.File(path, "r") as f:
        return f[_ds_path(frame, kind)][()]


# -- The file with no analysis tree at all ------------------------------

def test_detected_row_lands_on_a_file_with_no_analysis_group(synthetic_nexus):
    """``synthetic_nexus`` has three frames and not one analysis group —
    a freshly converted file the user opened and started labelling."""
    with h5py.File(synthetic_nexus, "r") as f:
        assert f"{ENTRY}/{ANALYSIS_REL}" not in f

    new_id = file_model.add_detected_peak_row(
        synthetic_nexus, ENTRY, 0, score=0.5, **GEOM,
    )

    assert new_id == 0
    arr = _rows(synthetic_nexus, 0, "detected_peaks")
    assert len(arr) == 1
    assert arr.dtype == PEAK_ROW_DTYPE
    assert float(arr["radius"][0]) == pytest.approx(2.0)
    assert float(arr["score"][0]) == pytest.approx(0.5)


def test_the_created_groups_carry_the_nexus_attrs_pygid_writes(synthetic_nexus):
    """A stricter NeXus reader must not be able to tell the difference
    between a group we made and one pygid's bulk save made."""
    file_model.add_detected_peak_row(synthetic_nexus, ENTRY, 1, **GEOM)

    with h5py.File(synthetic_nexus, "r") as f:
        for path in (
            f"{ENTRY}/{ANALYSIS_REL}",
            f"{ENTRY}/{ANALYSIS_REL}/{FRAME_KEY_FMT.format(1)}",
        ):
            attrs = dict(f[path].attrs)
            assert attrs.get("NX_class") == "NXparameters"
            assert attrs.get("EX_required") == "true"
        # The intermediate data group is NXdata and must keep saying so.
        assert dict(f[f"{ENTRY}/data"].attrs).get("NX_class") != "NXparameters"


def test_a_second_add_appends_rather_than_starting_over(synthetic_nexus):
    file_model.add_detected_peak_row(synthetic_nexus, ENTRY, 0, **GEOM)
    second = file_model.add_detected_peak_row(
        synthetic_nexus, ENTRY, 0, **{**GEOM, "radius": 3.0},
    )

    assert second == 1
    arr = _rows(synthetic_nexus, 0, "detected_peaks")
    assert [int(v) for v in arr["id"]] == [0, 1]


# -- The frame group exists, the dataset does not -----------------------

def test_fitted_row_lands_on_a_frame_whose_group_is_bare(
    synthetic_nexus_with_peaks,
):
    """``synthetic_nexus_with_peaks`` has peaks on frame 0 only — the
    common shape after a single-frame detection run."""
    new_id = file_model.add_fitted_peak_row(
        synthetic_nexus_with_peaks, ENTRY, 2, amplitude=5.0, **GEOM,
    )

    assert new_id == 0
    assert len(_rows(synthetic_nexus_with_peaks, 2, "fitted_peaks")) == 1
    # Frame 0's populated tables are none of this write's business.
    assert len(_rows(synthetic_nexus_with_peaks, 0, "detected_peaks")) == 3
    assert len(_rows(synthetic_nexus_with_peaks, 0, "fitted_peaks")) == 2


def test_creating_fitted_peaks_creates_its_errors_sibling(synthetic_nexus):
    """pygid always writes the pair, and mlgidbase's ``delete_peak``
    reads the errors dataset unconditionally, so a lone ``fitted_peaks``
    is a shape the backend does not expect. Empty is the honest value:
    mlgidGUI appends fitted rows without fabricating errors for them."""
    file_model.add_fitted_peak_row(
        synthetic_nexus, ENTRY, 0, amplitude=5.0, **GEOM,
    )

    errors = _rows(synthetic_nexus, 0, "fitted_peaks_errors")
    assert len(errors) == 0
    assert errors.dtype == PEAK_ROW_DTYPE


def test_detected_alone_does_not_create_a_fitted_table(synthetic_nexus):
    file_model.add_detected_peak_row(synthetic_nexus, ENTRY, 0, **GEOM)

    with h5py.File(synthetic_nexus, "r") as f:
        group = f[f"{ENTRY}/{ANALYSIS_REL}/{FRAME_KEY_FMT.format(0)}"]
        assert set(group.keys()) == {"detected_peaks"}


# -- What it refuses to do ----------------------------------------------

def test_an_unknown_entry_is_still_an_error(synthetic_nexus):
    """Bootstrapping a container is one thing; fabricating a whole scan
    that the caller only thinks exists is another."""
    with h5py.File(synthetic_nexus, "r") as f:
        before = sorted(f.keys())

    with pytest.raises(KeyError):
        file_model.add_detected_peak_row(synthetic_nexus, "entry_0007", 0, **GEOM)

    with h5py.File(synthetic_nexus, "r") as f:
        assert sorted(f.keys()) == before


def test_the_older_columnar_fitted_layout_is_refused_clearly(synthetic_nexus):
    """Old mlgid output stores fitted peaks as a *group* of 1-D columns.
    mlgidGUI reads that layout but cannot append to it; the error should
    say so instead of failing somewhere in numpy."""
    with h5py.File(synthetic_nexus, "r+") as f:
        g = f.create_group(f"{ENTRY}/{ANALYSIS_REL}/{FRAME_KEY_FMT.format(0)}/fitted_peaks")
        g.create_dataset("peaks_qxy", data=np.zeros(1, dtype="f4"))
        g.create_dataset("peaks_qz", data=np.zeros(1, dtype="f4"))

    with pytest.raises(KeyError, match="columnar"):
        file_model.add_fitted_peak_row(
            synthetic_nexus, ENTRY, 0, amplitude=1.0, **GEOM,
        )


# -- Layout the helper inherits rather than imposes ---------------------

def test_a_new_table_borrows_the_dtype_of_the_one_beside_it(synthetic_nexus):
    """A file written by a pygid whose row layout differs from ours must
    not end up with two peak tables of two different dtypes on one
    frame."""
    reduced = np.dtype([("radius", "f4"), ("angle", "f4"), ("id", "i4")])
    with h5py.File(synthetic_nexus, "r+") as f:
        f.create_dataset(_ds_path(0, "detected_peaks"), data=np.zeros(0, dtype=reduced))

    file_model.add_fitted_peak_row(
        synthetic_nexus, ENTRY, 0, amplitude=5.0, **GEOM,
    )

    assert _rows(synthetic_nexus, 0, "fitted_peaks").dtype == reduced


def test_a_legacy_frame_key_is_reused_not_shadowed(synthetic_nexus):
    """Older mlgid output names frame groups ``NNNNN``. Creating our
    ``frameNNNNN`` beside one would split a frame's peaks across two
    groups, and the reader (which checks ``frameNNNNN`` first) would
    show the new row and hide the old ones."""
    legacy = f"{ENTRY}/{ANALYSIS_REL}/00000"
    with h5py.File(synthetic_nexus, "r+") as f:
        f.create_group(legacy)

    file_model.add_detected_peak_row(synthetic_nexus, ENTRY, 0, **GEOM)

    with h5py.File(synthetic_nexus, "r") as f:
        analysis = f[f"{ENTRY}/{ANALYSIS_REL}"]
        assert sorted(analysis.keys()) == ["00000"]
        assert len(analysis["00000"]["detected_peaks"]) == 1


def test_the_bootstrap_dtype_still_matches_pygids(): 
    """``PEAK_ROW_DTYPE`` is a hand copy — ``file_model`` is backend-free
    by contract (see ``tests/test_hermetic_guard.py``), so it cannot
    import pygid to get this. Whenever pygid *is* installed, hold the
    copy to the original."""
    pygid = pytest.importorskip("pygid")
    from pygid.datasaver import pygid_results_dtype

    assert PEAK_ROW_DTYPE == pygid_results_dtype


# -- Through the GUI ----------------------------------------------------

gui = pytest.mark.gui


@pytest.fixture
def _no_blocking_modals(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window.session.temp_path


def _n_detected(path, frame=0) -> int:
    t = file_model.load_peaks(path, ENTRY, frame)["detected"]
    return 0 if t is None else len(t)


@gui
def test_add_to_detected_works_on_an_unanalysed_file(
    main_window, synthetic_nexus, _no_blocking_modals,
):
    path = _open(main_window, synthetic_nexus)
    peak = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    main_window.viewer.add_manual_peak(0, peak)
    main_window.viewer._set_selected(SelectedPeak.from_manual(peak, 0))

    main_window._on_add_to_detected()
    assert _n_detected(path) == 1

    main_window.viewer.undo_last_action()
    assert _n_detected(path) == 0


@gui
def test_quick_select_labels_an_unanalysed_file(
    main_window, synthetic_nexus, _no_blocking_modals,
):
    """The mode exists to label frames a model has never touched, which
    is precisely the case that used to fail — and quietly, since a quick
    commit never raises a dialog."""
    from PySide6.QtCore import QPointF

    path = _open(main_window, synthetic_nexus)
    combo = main_window.parameter_panel.combo_quick_target
    combo.setCurrentIndex(combo.findData(ParameterPanel.QUICK_TARGET_DETECTED))
    main_window.parameter_panel.chk_quick_select.setChecked(True)

    def draw(radius, angle):
        main_window.viewer._on_draw_finished(
            QPointF(radius - 0.1, angle - 3.0),
            QPointF(radius + 0.1, angle + 3.0),
        )

    logged: list[str] = []
    main_window.pipeline_panel.logMessage.connect(logged.append)

    draw(1.4, 30.0)
    assert _n_detected(path) == 0  # nothing committed yet
    draw(2.6, 70.0)

    assert _n_detected(path) == 1
    assert not [m for m in logged if "could not add" in m]
