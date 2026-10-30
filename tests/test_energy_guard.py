"""F-02 guard: ``_exp_params_from_nexus`` must raise a clear error
when the photon energy derived from
``instrument/monochromator/wavelength`` lies outside the plausible
1-200 keV X-ray range.

Two failure modes the audit calls out:
  * the wavelength datum is malformed (wrong units, e.g. Å instead
    of meters)
  * pygidsim silently changed its ``en`` contract from eV to keV,
    shrinking the computed value 1000x

Either way, the matching path must fail loud before mlgidmatch
builds a CIF pattern against a wrong wavelength and returns
plausible-but-bogus solutions.

Skipped wholesale on CI: ``_exp_params_from_nexus`` imports
``pygidsim.experiment.ExpParameters`` at the function top, and
CI does not ship the private ``pygidsim`` backend. Local dev
envs with the full mlgid stack still run all five cases.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("pygidsim")  # CI lacks the private backend; skip module.

from mlgidgui.pipeline import _EnergyOutOfRangeError, _exp_params_from_nexus  # noqa: E402


def _make_nexus_with_wavelength(path: Path, wavelength_m: float) -> Path:
    """Write a minimal valid NeXus file whose
    ``instrument/monochromator/wavelength`` is ``wavelength_m`` (in
    meters, as pygid expects)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry = f.create_group("entry_0000", track_order=True)
        d = entry.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        instr = entry.create_group("instrument", track_order=True)
        mono = instr.create_group("monochromator", track_order=True)
        mono.create_dataset("wavelength", data=np.array([wavelength_m]))
        instr.create_dataset("angle_of_incidence", data=np.array([0.3]))
    return path


def test_realistic_xray_wavelength_passes(tmp_path):
    """1 Å (≈ 12.4 keV) is the canonical synchrotron GIWAXS
    energy and must pass cleanly — confirms the guard does not
    over-fire on plausible inputs. ``ExpParameters`` stores the
    wavelength downstream of its ``en`` kwarg (en in eV converted
    to wavelength in Å for the rest of pygidsim's math), so we
    verify by round-tripping back to the seeded value."""
    path = _make_nexus_with_wavelength(tmp_path / "ok.h5", wavelength_m=1.0e-10)
    params = _exp_params_from_nexus(path, entry="entry_0000")
    # Round-trip should land within float-precision of 1 Å.
    assert params.wavelength == pytest.approx(1.0, rel=1e-3)


def test_pygidsim_kev_flip_trips_lower_bound(tmp_path):
    """Synthesise the divergence scenario: a wavelength of 12.4 nm
    (= 100 eV photon, way below the 1 keV floor) trips the guard.
    Models what would happen if pygidsim silently flipped the
    ``en`` unit to keV — the computed value would shrink by 1000x
    and land below the lower bound."""
    path = _make_nexus_with_wavelength(
        tmp_path / "soft.h5", wavelength_m=1.24e-8
    )
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def test_angstrom_units_mistake_trips_lower_bound(tmp_path):
    """If a writer stores wavelength in Å rather than meters (a
    1Å wavelength written as the literal 1.0), the derived energy
    is ~12.4 keV * 1e-10 ≈ 1.24e-6 eV. Guard must catch this."""
    path = _make_nexus_with_wavelength(tmp_path / "angstrom.h5", wavelength_m=1.0)
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def test_extreme_hard_xray_trips_upper_bound(tmp_path):
    """Wavelength of 1e-12 m (1 pm) corresponds to ~1.24 MeV,
    well past the 200 keV upper bound. Guard catches this too."""
    path = _make_nexus_with_wavelength(
        tmp_path / "gamma.h5", wavelength_m=1.0e-12
    )
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def _make_nexus(path: Path, *, include_instrument: bool = True,
                 wavelength_m: float | None = 1.0e-10,
                 ai_deg: float | None = 0.3) -> Path:
    """Flexible NeXus builder: include / omit the instrument subtree,
    or omit individual fields under it. Used by the per-field
    fallback tests below."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry = f.create_group("entry_0000", track_order=True)
        d = entry.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        if include_instrument:
            instr = entry.create_group("instrument", track_order=True)
            if wavelength_m is not None:
                mono = instr.create_group("monochromator", track_order=True)
                mono.create_dataset("wavelength", data=np.array([wavelength_m]))
            if ai_deg is not None:
                instr.create_dataset("angle_of_incidence", data=np.array([ai_deg]))
    return path


def test_full_metadata_warns_nothing(tmp_path, caplog):
    """When every field reads cleanly, no fallback warning fires —
    the matching run uses real geometry and the user sees no F-05
    notice."""
    path = _make_nexus(tmp_path / "ok.h5")
    caplog.clear()
    with caplog.at_level("WARNING"):
        _exp_params_from_nexus(path, entry="entry_0000")
    assert not any("Geometry fallback" in r.getMessage() for r in caplog.records)


def test_missing_instrument_emits_field_warning(tmp_path, caplog):
    """Without the instrument subtree, the per-field reads of
    wavelength + angle of incidence both fail; the function logs
    one WARNING naming the substituted fields. q_xy_max / q_z_max
    are still read from the data group, so the warning lists only
    the two genuinely-missing fields."""
    path = _make_nexus(tmp_path / "no_instr.h5", include_instrument=False)
    caplog.clear()
    with caplog.at_level("WARNING"):
        params = _exp_params_from_nexus(path, entry="entry_0000")
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    fallback_msgs = [m for m in msgs if "Geometry fallback" in m]
    assert len(fallback_msgs) == 1, f"expected one warning, got: {msgs}"
    text = fallback_msgs[0]
    assert "wavelength" in text
    assert "ai" in text
    assert "F-05" in text
    # The returned ExpParameters still uses the default en (18000 eV
    # ≈ 0.689 Å wavelength). q_xy / q_z reads succeeded, so those
    # land at the file's real values (≈ 1.0 on the synthetic axes).
    assert params.wavelength == pytest.approx(0.6888, rel=1e-2)


def test_only_one_field_missing_emits_targeted_warning(tmp_path, caplog):
    """A file with the wavelength dataset but no angle of incidence
    triggers a warning naming only ``ai`` — q_xy_max, q_z_max, and
    wavelength all read fine, so the warning is narrow."""
    path = _make_nexus(tmp_path / "no_ai.h5", ai_deg=None)
    caplog.clear()
    with caplog.at_level("WARNING"):
        params = _exp_params_from_nexus(path, entry="entry_0000")
    msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and "Geometry fallback" in r.getMessage()
    ]
    assert len(msgs) == 1
    text = msgs[0]
    assert "ai" in text
    # The other three fields read cleanly; the warning must not name
    # them as fallbacks.
    assert "wavelength:" not in text
    assert "q_xy_max:" not in text
    assert "q_z_max:" not in text
    # ai falls back to the 0.3° default; wavelength is the real 1Å.
    assert params.ai == pytest.approx(0.3)
    assert params.wavelength == pytest.approx(1.0, rel=1e-3)
