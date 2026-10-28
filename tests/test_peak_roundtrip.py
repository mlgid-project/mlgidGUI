"""Behavioral coverage of the peak file API (file_model only).

These run with no QApplication and operate on the fixture path
directly — never a ``NexusSession.temp_path`` held open by a
FrameSource — so they sidestep the h5py ``r+`` reopen the silx-detach
machinery exists for. Every assertion is grounded in source:
``file_model.load_peaks``, ``add_fitted_peak_row``,
``update_peak_row`` and ``clear_peaks`` (with
``_empty_dataset_in_place``).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab import file_model

ENTRY = "entry_0000"


def test_seeded_peak_counts(synthetic_nexus_with_peaks):
    """The fixture's frame 0 has 3 detected + 2 fitted rows; frames
    with no analysis group read back as ``{None, None}``."""
    peaks = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)
    assert peaks["detected"] is not None and len(peaks["detected"]) == 3
    assert peaks["fitted"] is not None and len(peaks["fitted"]) == 2
    assert sorted(peaks["detected"].ids.tolist()) == [0, 1, 2]

    bare = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 1)
    assert bare == {"detected": None, "fitted": None}


def test_add_fitted_peak_row_assigns_next_id_and_recomputes_cartesian(
    synthetic_nexus_with_peaks,
):
    """A new fitted row gets ``max(id)+1`` and q_xy/q_z recomputed from
    polar. f4 storage → tolerance compare, never exact equality."""
    angle, radius, amplitude = 30.0, 2.0, 17.0
    new_id = file_model.add_fitted_peak_row(
        synthetic_nexus_with_peaks,
        ENTRY,
        0,
        angle=angle,
        angle_width=4.0,
        radius=radius,
        radius_width=0.3,
        amplitude=amplitude,
        is_ring=True,
    )
    assert new_id == 2  # seeded ids were 0, 1

    fitted = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["fitted"]
    assert len(fitted) == 3
    row = int(np.where(fitted.ids == new_id)[0][0])
    assert fitted.angle[row] == pytest.approx(angle, abs=1e-4)
    assert fitted.radius[row] == pytest.approx(radius, abs=1e-4)
    assert fitted.amplitude[row] == pytest.approx(amplitude, abs=1e-4)
    assert fitted.q_xy[row] == pytest.approx(
        radius * np.cos(np.deg2rad(angle)), abs=1e-4
    )
    assert fitted.q_z[row] == pytest.approx(
        radius * np.sin(np.deg2rad(angle)), abs=1e-4
    )
    assert bool(fitted.is_ring[row]) is True


def test_add_fitted_peak_row_bootstraps_a_frame_with_no_analysis_group(
    synthetic_nexus_with_peaks,
):
    """Frame 1 has no analysis group, so no fitted_peaks dataset.

    This used to raise ``KeyError: run the appropriate pipeline stage at
    least once``. It no longer does: ``ensure_peak_dataset`` creates the
    group and an empty table, so a frame can be labelled by hand before
    any detection or fitting run. Full coverage of that path lives in
    ``tests/test_peak_dataset_bootstrap.py``; this case is here to pin
    that ``add_fitted_peak_row`` itself no longer refuses.
    """
    new_id = file_model.add_fitted_peak_row(
        synthetic_nexus_with_peaks,
        ENTRY,
        1,
        angle=10.0,
        angle_width=4.0,
        radius=1.0,
        radius_width=0.2,
        amplitude=5.0,
    )

    assert new_id == 0
    fitted = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 1)["fitted"]
    assert fitted is not None and len(fitted) == 1


def test_update_peak_row_roundtrip(synthetic_nexus_with_peaks):
    """Polar fields are written verbatim; q_xy/q_z recomputed."""
    file_model.update_peak_row(
        synthetic_nexus_with_peaks,
        ENTRY,
        0,
        "detected",
        1,
        angle=33.0,
        angle_width=7.0,
        radius=2.75,
        radius_width=0.11,
    )
    det = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["detected"]
    row = int(np.where(det.ids == 1)[0][0])
    assert det.angle[row] == pytest.approx(33.0, abs=1e-4)
    assert det.radius[row] == pytest.approx(2.75, abs=1e-4)
    assert det.angle_width[row] == pytest.approx(7.0, abs=1e-4)
    assert det.radius_width[row] == pytest.approx(0.11, abs=1e-4)
    assert det.q_xy[row] == pytest.approx(
        2.75 * np.cos(np.deg2rad(33.0)), abs=1e-4
    )
    assert det.q_z[row] == pytest.approx(
        2.75 * np.sin(np.deg2rad(33.0)), abs=1e-4
    )


def test_update_peak_row_bad_id_raises_keyerror(synthetic_nexus_with_peaks):
    with pytest.raises(KeyError):
        file_model.update_peak_row(
            synthetic_nexus_with_peaks,
            ENTRY,
            0,
            "detected",
            999,
            angle=1.0,
            angle_width=1.0,
            radius=1.0,
            radius_width=1.0,
        )


def test_update_peak_row_bad_kind_raises_valueerror(synthetic_nexus_with_peaks):
    with pytest.raises(ValueError):
        file_model.update_peak_row(
            synthetic_nexus_with_peaks,
            ENTRY,
            0,
            "matched",
            0,
            angle=1.0,
            angle_width=1.0,
            radius=1.0,
            radius_width=1.0,
        )


def test_clear_peaks_empties_not_deletes(synthetic_nexus_with_peaks):
    """``clear_peaks`` recreates the dataset at shape (0,); the dataset
    is still *present*, so ``load_peaks`` returns an empty PeakTable —
    ``len == 0``, NOT ``None``."""
    removed = file_model.clear_peaks(
        synthetic_nexus_with_peaks, ENTRY, "detected"
    )
    assert removed == 3
    det = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["detected"]
    assert det is not None
    assert len(det) == 0

    removed_fit = file_model.clear_peaks(
        synthetic_nexus_with_peaks, ENTRY, "fitted"
    )
    assert removed_fit == 2
    fit = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["fitted"]
    assert fit is not None and len(fit) == 0


def test_clear_peaks_bad_kind_raises_valueerror(synthetic_nexus_with_peaks):
    with pytest.raises(ValueError):
        file_model.clear_peaks(synthetic_nexus_with_peaks, ENTRY, "bogus")


@pytest.mark.parametrize(
    "angle,radius,amplitude",
    [(0.0, 1.0, 5.0), (45.0, 2.5, 12.0), (90.0, 3.0, 30.0), (135.0, 0.5, 1.0)],
)
def test_add_then_clear_fitted_roundtrip(
    synthetic_nexus_with_peaks, angle, radius, amplitude
):
    new_id = file_model.add_fitted_peak_row(
        synthetic_nexus_with_peaks,
        ENTRY,
        0,
        angle=angle,
        angle_width=3.0,
        radius=radius,
        radius_width=0.2,
        amplitude=amplitude,
    )
    fitted = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["fitted"]
    assert new_id in fitted.ids.tolist()
    assert len(fitted) == 3

    file_model.clear_peaks(synthetic_nexus_with_peaks, ENTRY, "fitted")
    fitted = file_model.load_peaks(synthetic_nexus_with_peaks, ENTRY, 0)["fitted"]
    assert len(fitted) == 0
