"""A registered peak list standing in for a standard pipeline table.

Tick "Pipeline primary" on an extra peak list and Detection/Fitting
read and write THAT table instead of ``detected_peaks`` /
``fitted_peaks``, so a second analysis can be built without destroying
the first. Everything else in the app keeps using the standard tables.

mlgidbase writes fixed names (they are resolved inside pygid's
``_save_img_container_*`` and no method takes an output name), so the
redirect is a set of HDF5 link renames around the call. These tests
drive that machinery around a **stub run** that writes the standard
names the way the backend does -- which is what lets the whole feature
be covered on a box without the private backend, CI included.

The invariant behind most of these: after a redirected run the standard
tables are BYTE-IDENTICAL to what they were before it.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QSettings

from mlgidlab import file_model, peak_lists, pipeline

ENTRY = "entry_0000"
DET = "my_detected"
FIT = "my_fitted"


@pytest.fixture(autouse=True)
def _clean_registry():
    QSettings().remove(peak_lists.SETTINGS_KEY)
    yield
    QSettings().remove(peak_lists.SETTINGS_KEY)


def _specs(detected: bool = False, fitted: bool = False) -> list:
    out = []
    if detected:
        out.append(peak_lists.PeakListSpec(
            dataset=DET, label="", treat_as="detected", primary=True,
        ))
    if fitted:
        out.append(peak_lists.PeakListSpec(
            dataset=FIT, label="", treat_as="fitted", primary=True,
        ))
    return out


def _group_path(frame: int = 0) -> str:
    return (f"{ENTRY}/{file_model.ANALYSIS_REL}/"
            f"{file_model.FRAME_KEY_FMT.format(frame)}")


def _read(path, name, frame: int = 0):
    """Raw bytes of one dataset, or None when it is not there."""
    with h5py.File(path, "r") as f:
        group = f[_group_path(frame)]
        if name not in group:
            return None
        return group[name][()].tobytes()


def _rows(path, name, frame: int = 0):
    with h5py.File(path, "r") as f:
        return np.asarray(f[f"{_group_path(frame)}/{name}"][()])


def _seed(path, name, n: int, first_id: int = 500, frame: int = 0) -> None:
    """Write a peak-shaped table, standing in for what a run produces.

    The dtype comes from the shared fixture definition rather than from
    a table in the file: half of these run mid-swap, where the standard
    tables are exactly what is NOT at their usual name.
    """
    from conftest import PYGID_PEAK_DTYPE

    rows = np.zeros(n, dtype=np.dtype(PYGID_PEAK_DTYPE))
    rows["id"] = np.arange(first_id, first_id + n)
    rows["radius"] = np.arange(n, dtype=float) + 1.0
    with h5py.File(path, "r+") as f:
        group = f.require_group(_group_path(frame))
        if name in group:
            del group[name]
        group.create_dataset(name, data=rows)


# --- the plan ---------------------------------------------------------


def test_no_primary_means_no_plan_at_all():
    """The guarantee the ordinary path rests on: without a tick there
    is no plan, so not one byte is written around a normal run."""
    specs = [peak_lists.PeakListSpec(dataset="Br_peaks", label="Br")]
    assert peak_lists.swap_plan("run_detection", specs) is None
    assert peak_lists.swap_plan("run_fitting", specs) is None


def test_the_plan_matches_the_combination_table():
    det_plan = peak_lists.swap_plan("run_detection", _specs(detected=True))
    assert det_plan["stash"] == ["detected_peaks"]
    assert det_plan["swap_in"] == [(DET, "detected_peaks")]
    assert det_plan["capture"] == [("detected_peaks", DET)]

    # A detected primary alone still reaches fitting, because fitting
    # READS detected_peaks.
    fit_of_det = peak_lists.swap_plan("run_fitting", _specs(detected=True))
    assert fit_of_det["stash"] == ["detected_peaks"]

    # ...and a fitted primary claims both of pygidFIT's outputs.
    fit_plan = peak_lists.swap_plan("run_fitting", _specs(fitted=True))
    assert fit_plan["stash"] == ["fitted_peaks", "fitted_peaks_errors"]
    assert fit_plan["capture"] == [
        ("fitted_peaks", FIT), ("fitted_peaks_errors", f"{FIT}_errors"),
    ]

    both = peak_lists.swap_plan(
        "run_fitting", _specs(detected=True, fitted=True)
    )
    assert both["stash"] == [
        "detected_peaks", "fitted_peaks", "fitted_peaks_errors",
    ]


def test_a_detection_primary_never_reaches_matching_or_tracking():
    """Excluded ops carry no plan, so they cannot be redirected by
    accident -- matched_* rows hold POSITIONS into the fitted table."""
    specs = _specs(detected=True, fitted=True)
    assert peak_lists.swap_plan("run_matching", specs) is None
    assert peak_lists.swap_plan("track_peaks", specs) is None
    assert peak_lists.swap_plan("inject_fitted_peaks", specs) is None


def test_the_stash_prefix_matches_the_one_pipeline_uses():
    """``pipeline`` keeps its own copy to stay Qt-free; pin them equal."""
    assert pipeline._STASH_PREFIX == peak_lists.STASH_PREFIX


# --- the swap around a run --------------------------------------------


def test_detection_output_lands_in_the_primary_table(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    before = _read(path, "detected_peaks")
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    with pipeline._swapped_tables(path, ENTRY, 0, plan):
        _seed(path, "detected_peaks", 4)      # what the backend does

    assert len(_rows(path, DET)) == 4
    assert _read(path, "detected_peaks") == before


def test_fitting_reads_the_primary_detected_table(
    synthetic_nexus_with_peaks,
):
    """The input half, provable only from inside the run: while the body
    executes, ``detected_peaks`` must BE the primary table."""
    path = synthetic_nexus_with_peaks
    _seed(path, DET, 7, first_id=900)
    custom = _rows(path, DET).tobytes()
    standard_before = _read(path, "detected_peaks")
    plan = peak_lists.swap_plan(
        "run_fitting", _specs(detected=True, fitted=True)
    )

    with pipeline._swapped_tables(path, ENTRY, 0, plan):
        assert _read(path, "detected_peaks") == custom
        _seed(path, "fitted_peaks", 7, first_id=900)
        _seed(path, "fitted_peaks_errors", 7, first_id=900)

    assert len(_rows(path, FIT)) == 7
    assert len(_rows(path, f"{FIT}_errors")) == 7
    assert _read(path, "detected_peaks") == standard_before
    assert len(_rows(path, DET)) == 7


def test_every_standard_table_survives_a_redirected_fit(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    _seed(path, DET, 3, first_id=900)
    _seed(path, "fitted_peaks_errors", 3)
    before = {
        name: _read(path, name)
        for name in ("detected_peaks", "fitted_peaks", "fitted_peaks_errors")
    }
    plan = peak_lists.swap_plan(
        "run_fitting", _specs(detected=True, fitted=True)
    )

    with pipeline._swapped_tables(path, ENTRY, 0, plan):
        _seed(path, "fitted_peaks", 5, first_id=700)
        _seed(path, "fitted_peaks_errors", 5, first_id=700)

    for name, blob in before.items():
        assert _read(path, name) == blob, f"{name} was modified"


def test_a_primary_table_need_not_exist_yet(synthetic_nexus_with_peaks):
    """First run of a freshly registered list: nothing to swap in, and
    the capture still has to work."""
    path = synthetic_nexus_with_peaks
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))
    with pipeline._swapped_tables(path, ENTRY, 0, plan):
        _seed(path, "detected_peaks", 2)
    assert len(_rows(path, DET)) == 2


# --- scope ------------------------------------------------------------


def test_a_single_frame_run_leaves_its_neighbours_alone(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    # frame 1 has no analysis group in the fixture; give it one.
    with h5py.File(path, "r+") as f:
        src = f[_group_path(0)]
        dst = f.require_group(_group_path(1))
        src.copy("detected_peaks", dst, name="detected_peaks")
    frame1_before = _read(path, "detected_peaks", frame=1)
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    with pipeline._swapped_tables(path, ENTRY, 0, plan):
        _seed(path, "detected_peaks", 2)

    assert len(_rows(path, DET, frame=0)) == 2
    assert _read(path, "detected_peaks", frame=1) == frame1_before
    assert _read(path, DET, frame=1) is None


def test_a_whole_entry_run_covers_a_frame_the_run_created(
    synthetic_nexus_with_peaks,
):
    """Detection creates frame groups, so the post-run pass has to
    re-list rather than reuse what it saw going in."""
    path = synthetic_nexus_with_peaks
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    with pipeline._swapped_tables(path, ENTRY, None, plan):
        _seed(path, "detected_peaks", 2, frame=0)
        _seed(path, "detected_peaks", 3, frame=2)  # a brand-new group

    assert len(_rows(path, DET, frame=0)) == 2
    assert len(_rows(path, DET, frame=2)) == 3


# --- crash recovery ---------------------------------------------------


def test_an_interrupted_run_is_repaired_on_the_next_one(
    synthetic_nexus_with_peaks,
):
    """A hard kill mid-swap leaves the primary table under the standard
    name. Simulated by swapping in and never swapping out."""
    path = synthetic_nexus_with_peaks
    _seed(path, DET, 6, first_id=900)
    standard_before = _read(path, "detected_peaks")
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    pipeline._swap_in(path, ENTRY, 0, plan)
    assert _read(path, "detected_peaks") != standard_before  # mid-swap

    assert pipeline.recover_swaps(path) == 1

    assert _read(path, "detected_peaks") == standard_before
    assert len(_rows(path, DET)) == 6
    with h5py.File(path, "r") as f:
        group = f[_group_path(0)]
        assert not [k for k in group if k.startswith(peak_lists.STASH_PREFIX)]
        assert f"{ENTRY}/process/mlgidlab/pipeline_swap" not in f


def test_recovery_works_from_the_stash_names_alone(
    synthetic_nexus_with_peaks,
):
    """Marker gone too. The standard table must still come back, and
    the interrupted run's data must not be destroyed to do it."""
    path = synthetic_nexus_with_peaks
    _seed(path, DET, 6, first_id=900)
    standard_before = _read(path, "detected_peaks")
    swapped_in = _read(path, DET)
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    pipeline._swap_in(path, ENTRY, 0, plan)
    with h5py.File(path, "r+") as f:
        del f[f"{ENTRY}/process/mlgidlab/pipeline_swap"]

    assert pipeline.recover_swaps(path) == 1

    assert _read(path, "detected_peaks") == standard_before
    assert _read(path, f"{pipeline._ORPHAN_PREFIX}detected_peaks") == swapped_in


def test_an_exception_in_the_run_still_restores(synthetic_nexus_with_peaks):
    path = synthetic_nexus_with_peaks
    standard_before = _read(path, "detected_peaks")
    plan = peak_lists.swap_plan("run_detection", _specs(detected=True))

    with pytest.raises(RuntimeError):
        with pipeline._swapped_tables(path, ENTRY, 0, plan):
            raise RuntimeError("detection blew up")

    assert _read(path, "detected_peaks") == standard_before
    with h5py.File(path, "r") as f:
        assert f"{ENTRY}/process/mlgidlab/pipeline_swap" not in f


def test_recovery_is_a_no_op_on_an_untouched_file(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    before = _read(path, "detected_peaks")
    assert pipeline.recover_swaps(path) == 0
    assert _read(path, "detected_peaks") == before


# --- what the redirect must NOT invalidate ----------------------------


def test_only_a_redirected_fitted_table_answers_the_invalidation_question():
    """Two follow-on passes (the F-04 matched clear, the phase-track
    invalidation) exist because a refit rewrites fitted_peaks. A
    redirected fit does not, but a DETECTED-only primary still does."""
    det_only = peak_lists.swap_plan("run_fitting", _specs(detected=True))
    fit_too = peak_lists.swap_plan("run_fitting", _specs(fitted=True))

    assert peak_lists.redirects_dataset(det_only, "fitted_peaks") is False
    assert peak_lists.redirects_dataset(fit_too, "fitted_peaks") is True
    assert peak_lists.redirects_dataset(None, "fitted_peaks") is False


def test_post_run_passes_follow_the_tables_the_run_used():
    plan = peak_lists.swap_plan(
        "run_fitting", _specs(detected=True, fitted=True)
    )
    assert peak_lists.swapped_name(plan, "detected_peaks") == DET
    assert peak_lists.swapped_name(plan, "fitted_peaks") == FIT
    assert peak_lists.swapped_name(None, "fitted_peaks") == "fitted_peaks"


# --- the registry -----------------------------------------------------


def test_primary_round_trips_through_settings():
    peak_lists.save_specs(_specs(detected=True, fitted=True))
    loaded = peak_lists.load_specs()
    assert [s.dataset for s in loaded if s.primary] == [DET, FIT]
    assert peak_lists.primary_for(loaded, "detected").dataset == DET
    assert peak_lists.primary_for(loaded, "fitted").dataset == FIT


def test_two_primaries_of_one_flavour_load_as_one():
    """The setting is a string a user can hand-edit, and two primaries
    for one table would make the swap ambiguous. First one wins."""
    import json

    QSettings().setValue(peak_lists.SETTINGS_KEY, json.dumps([
        {"dataset": "a", "label": "", "treat_as": "detected", "primary": True},
        {"dataset": "b", "label": "", "treat_as": "detected", "primary": True},
        {"dataset": "c", "label": "", "treat_as": "fitted", "primary": True},
    ]))
    loaded = peak_lists.load_specs()
    assert [s.dataset for s in loaded if s.primary] == ["a", "c"]
    assert len(loaded) == 3  # 'b' survives, just not as primary


# --- the Settings column ----------------------------------------------


def _dialog(main_window):
    from mlgidlab.main_window_dialogs import _SettingsDialog

    return _SettingsDialog(main_window)


@pytest.mark.gui
def test_ticking_one_primary_unticks_its_sibling(main_window):
    """Radio-like within a flavour. Two primaries for one standard
    table would make the swap ambiguous, and saying so with the tick
    beats a validation error on OK."""
    peak_lists.save_specs([
        peak_lists.PeakListSpec(dataset="a", label="", treat_as="detected"),
        peak_lists.PeakListSpec(dataset="b", label="", treat_as="detected"),
        peak_lists.PeakListSpec(dataset="c", label="", treat_as="fitted"),
    ])
    dlg = _dialog(main_window)

    dlg._primary_box(0).setChecked(True)
    dlg._primary_box(2).setChecked(True)   # other flavour, coexists
    assert [s.dataset for s in dlg.peak_list_specs() if s.primary] == ["a", "c"]

    dlg._primary_box(1).setChecked(True)   # same flavour as 'a'
    assert [s.dataset for s in dlg.peak_list_specs() if s.primary] == ["b", "c"]


@pytest.mark.gui
def test_changing_the_flavour_drops_the_tick(main_window):
    """The tick means "primary for THIS flavour"; carrying it across a
    change could quietly produce a second primary for the new one."""
    peak_lists.save_specs([
        peak_lists.PeakListSpec(
            dataset="a", label="", treat_as="detected", primary=True,
        ),
    ])
    dlg = _dialog(main_window)
    assert dlg._primary_box(0).isChecked()

    treat = dlg._lists_table.cellWidget(0, 2)
    treat.setCurrentIndex(treat.findData("fitted"))

    assert dlg._primary_box(0).isChecked() is False


@pytest.mark.gui
def test_a_fitted_primary_may_not_eat_another_registered_list(main_window):
    """A fitted run writes two datasets, so a primary 'trial2' also
    claims 'trial2_errors' -- which must not already be someone's
    layer, or the first run would silently overwrite it."""
    peak_lists.save_specs([
        peak_lists.PeakListSpec(dataset="trial2", label="", treat_as="fitted"),
        peak_lists.PeakListSpec(
            dataset="trial2_errors", label="", treat_as="fitted",
        ),
    ])
    dlg = _dialog(main_window)
    assert dlg.primary_name_conflict() == ""

    dlg._primary_box(0).setChecked(True)
    assert dlg.primary_name_conflict() == "trial2_errors"


@pytest.mark.gui
def test_the_tick_survives_a_settings_round_trip(main_window):
    peak_lists.save_specs([
        peak_lists.PeakListSpec(dataset="a", label="A", treat_as="fitted"),
    ])
    dlg = _dialog(main_window)
    dlg._primary_box(0).setChecked(True)
    dlg.save_to_qsettings()

    reloaded = peak_lists.load_specs()
    assert reloaded[0].primary is True
    assert peak_lists.primary_for(reloaded, "fitted").dataset == "a"


@pytest.mark.gui
def test_the_queue_stamps_the_plan_onto_every_command(main_window):
    """The seam between Settings and the worker. The plan is resolved
    HERE, on the GUI thread, because the worker has no business reading
    QSettings -- and because the multi-entry expansion rebuilds the
    command, so stamping at construction would lose it."""
    from pathlib import Path

    from mlgidlab.pipeline import PipelineCommand

    peak_lists.save_specs(_specs(detected=True))
    main_window._pipeline_queue = []
    main_window._pipe_thread = object()   # keep the queue from draining

    cmd = PipelineCommand("run_detection", {"entry": ENTRY})
    main_window._enqueue_pipeline(Path("x.h5"), cmd)
    assert cmd.table_swap["capture"] == [("detected_peaks", DET)]

    # An excluded op is stamped with None, so the swap can never reach
    # matching even with a primary registered.
    other = PipelineCommand("run_matching", {"entry": ENTRY})
    main_window._enqueue_pipeline(Path("x.h5"), other)
    assert other.table_swap is None

    main_window._pipe_thread = None
    main_window._pipeline_queue = []
