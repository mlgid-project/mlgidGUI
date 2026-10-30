"""The ``inject_fitted_peaks`` pipeline op (shared injection executor).

Runs ``pipeline.execute`` against the synthetic NeXus fixture with the
2D fit and the geometry reader monkeypatched (both are imported inside
the executor at call time, so module-attribute patches take effect).
Verifies the op needs no matching backend, appends detected+fitted
rows, honours the ring row convention, cleans up after failed fits,
and that its records — unlike ``interpolate_tracks`` — carry no
``track`` field.

Source: pipeline.py ``_execute_peak_injection`` /
``_execute_interpolate_tracks``; workers.py ``_resolve_total_frames``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui import file_model
from mlgidgui.manual_fit import ManualFitError, ManualFitResult
from mlgidgui.pipeline import PipelineCommand, execute
from mlgidgui.workers import PipelineWorker

ENTRY = "entry_0000"


def _patch_geometry(monkeypatch):
    monkeypatch.setattr(
        file_model, "read_geometry_for_entry",
        lambda path, entry, frame=0: {
            "wavelength_angstrom": 1.0,
            "q_xy_max": 3.0,
            "q_z_max": 4.0,
            "ai_deg": 0.3,
            "q_z_axis": np.linspace(0.0, 4.0, 16),
        },
    )


def _patch_fit(monkeypatch, fn):
    import mlgidgui.manual_fit as manual_fit
    monkeypatch.setattr(manual_fit, "fit_one_peak", fn)


def _echo_fit(cart, q_xy, q_z, *, radius, radius_width, angle,
              angle_width, **kwargs) -> ManualFitResult:
    """A fit that returns the injected box, slightly refined."""
    return ManualFitResult(
        radius=radius + 0.01, radius_width=radius_width,
        angle=angle, angle_width=angle_width,
        amplitude=42.0, A=1.0, B=0.0, C=1.0, theta=0.0,
    )


def _counts(path, frame=0):
    tables = file_model.load_peaks(path, ENTRY, frame)
    det, fit = tables.get("detected"), tables.get("fitted")
    return (
        0 if det is None else len(det),
        0 if fit is None else len(fit),
    )


def test_inject_spots_appends_rows_without_track(
    synthetic_nexus_with_peaks, monkeypatch,
):
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)
    _patch_fit(monkeypatch, _echo_fit)
    assert _counts(path) == (3, 2)

    specs = [
        {"radius": 1.0, "angle": 30.0, "radius_width": 0.2,
         "angle_width": 6.0, "is_ring": False},
        {"radius": 2.0, "angle": 70.0, "radius_width": 0.2,
         "angle_width": 6.0, "is_ring": False},
    ]
    records = execute(path, PipelineCommand("inject_fitted_peaks", {
        "entry": ENTRY, "plan": {0: specs}, "fit_params": {},
    }))

    assert _counts(path) == (5, 4)
    assert [sorted(r) for r in records] == [
        ["detected_id", "fitted_id", "frame", "is_ring"],
        ["detected_id", "fitted_id", "frame", "is_ring"],
    ]
    assert [r["frame"] for r in records] == [0, 0]
    tables = file_model.load_peaks(path, ENTRY, 0)
    det, fit = tables["detected"], tables["fitted"]
    for rec, spec in zip(records, specs):
        di = int(np.flatnonzero(det.ids == rec["detected_id"])[0])
        # Injected detected rows carry score 1.0 (user-vetted box).
        assert det.score[di] == pytest.approx(1.0)
        assert det.radius[di] == pytest.approx(spec["radius"])
        fi = int(np.flatnonzero(fit.ids == rec["fitted_id"])[0])
        assert fit.radius[fi] == pytest.approx(spec["radius"] + 0.01)
        assert fit.amplitude[fi] == pytest.approx(42.0)
        assert not fit.is_ring[fi]


def test_link_ids_pairs_each_fit_with_its_injected_box(
    synthetic_nexus_with_peaks, monkeypatch,
):
    """The op writes a detected box and its fit as a pair, so with the
    fitted/detected link on the fit takes the box's id — the GUI reads
    the setting and passes the answer in, since a worker has no
    business touching QSettings."""
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)
    _patch_fit(monkeypatch, _echo_fit)
    specs = [
        {"radius": 1.0, "angle": 30.0, "radius_width": 0.2,
         "angle_width": 6.0, "is_ring": False},
        {"radius": 2.0, "angle": 70.0, "radius_width": 0.2,
         "angle_width": 6.0, "is_ring": False},
    ]
    records = execute(path, PipelineCommand("inject_fitted_peaks", {
        "entry": ENTRY, "plan": {0: specs}, "fit_params": {},
        "link_ids": True,
    }))
    assert [r["fitted_id"] for r in records] == [
        r["detected_id"] for r in records
    ]
    # The ids are the injected boxes', not a continuation of the fitted
    # table (which held 0 and 1 before this run).
    assert [r["detected_id"] for r in records] == [3, 4]
    tables = file_model.load_peaks(path, ENTRY, 0)
    assert sorted(int(v) for v in tables["fitted"].ids) == [0, 1, 3, 4]
    # Without the flag the fitted rows follow their own table, which
    # ``test_inject_spots_appends_rows_without_track`` covers.


def test_inject_ring_persists_ring_convention(
    synthetic_nexus_with_peaks, monkeypatch,
):
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)

    def ring_fit(cart, q_xy, q_z, *, radius, radius_width, **kwargs):
        # A real pygidfit ring fit returns NaN angle / inf width.
        return ManualFitResult(
            radius=radius, radius_width=radius_width,
            angle=float("nan"), angle_width=float("inf"),
            amplitude=7.0, A=float("nan"), B=0.0, C=0.0,
            theta=float("nan"),
        )

    _patch_fit(monkeypatch, ring_fit)
    records = execute(path, PipelineCommand("inject_fitted_peaks", {
        "entry": ENTRY,
        "plan": {0: [{
            "radius": 1.5, "angle": 45.0, "radius_width": 0.2,
            "angle_width": 90.0, "is_ring": True,
        }]},
        "fit_params": {},
    }))

    assert len(records) == 1 and records[0]["is_ring"] is True
    fit = file_model.load_peaks(path, ENTRY, 0)["fitted"]
    fi = int(np.flatnonzero(fit.ids == records[0]["fitted_id"])[0])
    assert bool(fit.is_ring[fi])
    assert fit.angle[fi] == pytest.approx(45.0)
    assert np.isinf(fit.angle_width[fi])
    # The injected DETECTED box keeps its finite quadrant-spanning
    # geometry (an inf width breaks pygidfit on a later refit).
    det = file_model.load_peaks(path, ENTRY, 0)["detected"]
    di = int(np.flatnonzero(det.ids == records[0]["detected_id"])[0])
    assert det.angle_width[di] == pytest.approx(90.0)


def test_failed_fit_removes_injected_detected_row(
    synthetic_nexus_with_peaks, monkeypatch,
):
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)

    def failing_fit(cart, q_xy, q_z, *, radius, **kwargs):
        if radius > 1.5:
            raise ManualFitError("no convergence")
        return _echo_fit(
            cart, q_xy, q_z, radius=radius, **{
                k: kwargs[k] for k in
                ("radius_width", "angle", "angle_width")
            },
        )

    _patch_fit(monkeypatch, failing_fit)
    records = execute(path, PipelineCommand("inject_fitted_peaks", {
        "entry": ENTRY,
        "plan": {0: [
            {"radius": 1.0, "angle": 30.0, "radius_width": 0.2,
             "angle_width": 6.0, "is_ring": False},
            {"radius": 2.0, "angle": 70.0, "radius_width": 0.2,
             "angle_width": 6.0, "is_ring": False},
        ]},
        "fit_params": {},
    }))

    # Only the succeeding box landed; the failed one's injected
    # detected row was deleted again (no orphan geometry).
    assert len(records) == 1
    assert records[0]["frame"] == 0
    assert _counts(path) == (4, 3)


def test_implausible_fits_are_discarded(
    synthetic_nexus_with_peaks, monkeypatch,
):
    """The plausibility gate: fits that escaped the injected box, blew
    up their width (background fit) or found nothing (non-positive
    amplitude) are treated like failed fits — detected row removed,
    no fitted row, no record."""
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)

    def gate_fit(cart, q_xy, q_z, *, radius, radius_width, angle,
                 angle_width, **kwargs):
        base = dict(
            radius=radius, radius_width=radius_width, angle=angle,
            angle_width=angle_width, amplitude=5.0,
            A=1.0, B=0.0, C=1.0, theta=0.0,
        )
        if radius == pytest.approx(0.5):
            base["radius"] = radius + radius_width      # escaped radially
        elif radius == pytest.approx(1.5):
            base["radius_width"] = radius_width * 5.0   # background blow-up
        elif radius == pytest.approx(2.5):
            base["amplitude"] = 0.0                     # nothing there
        elif radius == pytest.approx(2.8):
            base["angle"] = angle + angle_width         # escaped in angle
        return ManualFitResult(**base)

    _patch_fit(monkeypatch, gate_fit)
    spec = {"radius_width": 0.2, "angle_width": 6.0, "is_ring": False}
    records = execute(path, PipelineCommand("inject_fitted_peaks", {
        "entry": ENTRY,
        "plan": {0: [
            {"radius": 0.5, "angle": 30.0, **spec},
            {"radius": 1.5, "angle": 30.0, **spec},
            {"radius": 2.5, "angle": 30.0, **spec},
            {"radius": 2.8, "angle": 30.0, **spec},
            {"radius": 1.0, "angle": 30.0, **spec},   # plausible
        ]},
        "fit_params": {},
    }))

    assert len(records) == 1
    fit = file_model.load_peaks(path, ENTRY, 0)["fitted"]
    fi = int(np.flatnonzero(fit.ids == records[0]["fitted_id"])[0])
    assert fit.radius[fi] == pytest.approx(1.0)
    # Every rejected box's injected detected row was removed again.
    assert _counts(path) == (4, 3)


def test_interpolate_records_still_carry_track(
    synthetic_nexus_with_peaks, monkeypatch,
):
    path = synthetic_nexus_with_peaks
    _patch_geometry(monkeypatch)
    _patch_fit(monkeypatch, _echo_fit)
    records = execute(path, PipelineCommand("interpolate_tracks", {
        "entry": ENTRY,
        "plan": {0: [{
            "track": 7, "radius": 1.0, "angle": 30.0,
            "radius_width": 0.2, "angle_width": 6.0, "is_ring": False,
        }]},
        "fit_params": {},
    }))
    assert len(records) == 1
    assert records[0]["track"] == 7


def test_worker_total_frames_counts_plan(synthetic_nexus_with_peaks):
    worker = PipelineWorker(
        synthetic_nexus_with_peaks,
        PipelineCommand("inject_fitted_peaks", {
            "entry": ENTRY,
            "plan": {0: [{"radius": 1.0}], 2: [{"radius": 2.0}]},
        }),
    )
    assert worker._resolve_total_frames() == (2, ENTRY)
