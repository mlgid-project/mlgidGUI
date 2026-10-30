"""``manual_fit`` neighbour selection: which of a frame's other peak
boxes travel with the user's box into ``pygidfit.fit_data``.

pygidfit only masks (other clusters NaN-ed out of the ROI) or
joint-fits (boxes within ``clustering_distance_peaks`` fitted as a sum
of Gaussians) the boxes handed to a *single* call. "Add to fitted"
used to pass one box, which disabled both and left a neighbouring
peak's intensity sitting unmasked in the target's ROI — the mechanism
behind fits landing outside the box the user drew.

Most of these cover the pure selection layer: no Qt, no pygidfit, no
I/O, so they also run in the backend-less CI environment. The last
two check where ``MainWindow`` sources the boxes from. The end-to-end
proof that the masking changes the fit needs the real pygidfit and
lives in ``test_manual_fit.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.file_model import PeakTable
from mlgidgui.manual_fit import (
    PeakBox,
    neighbour_boxes_from_tables,
    select_neighbour_boxes,
)

# The synthetic frames below use a 3 Å⁻¹ square q range, so
# ``q_abs_max = sqrt(3² + 3²)``. One polar pixel is then
# ``q_abs_max / 1024`` in q and ``90 / 512`` degrees.
Q_ABS_MAX = float(np.sqrt(3.0 ** 2 + 3.0 ** 2))
TARGET = PeakBox(radius=1.5, radius_width=0.08, angle=45.0, angle_width=7.0)


def _select(*candidates, **kwargs):
    return select_neighbour_boxes(
        TARGET, list(candidates), q_abs_max=Q_ABS_MAX, **kwargs
    )


def test_it_keeps_a_neighbour_that_can_reach_the_target_roi():
    near = PeakBox(radius=1.5, radius_width=0.08, angle=52.0, angle_width=7.0)
    assert _select(near) == [near]


def test_it_drops_a_box_too_far_away_to_touch_the_roi():
    """A box that neither clusters with the target nor intersects its
    ROI cannot change the target's fit, and passing it would cost a
    whole extra cluster fit on every click."""
    far = PeakBox(radius=2.6, radius_width=0.08, angle=20.0, angle_width=7.0)
    assert _select(far) == []


def test_the_cutoff_follows_the_clustering_settings():
    """The inflation is ``clustering_distance_peaks + 2 × extend`` in
    polar pixels, so raising the clustering distance pulls more
    neighbours in — the same knob that decides whether pygidfit would
    have clustered them together."""
    mid = PeakBox(radius=1.5, radius_width=0.08, angle=57.0, angle_width=7.0)
    assert _select(mid) == []
    assert _select(mid, clustering_distance_peaks=60.0) == [mid]


def test_it_drops_the_same_peak_under_another_name():
    """A detected box and the fitted row made from it describe one
    peak. Passing both would put two Gaussians on it."""
    duplicate = PeakBox(
        radius=1.505, radius_width=0.079, angle=45.2, angle_width=6.9,
    )
    assert _select(duplicate) == []


def test_it_drops_a_duplicate_among_the_candidates_too():
    a = PeakBox(radius=1.5, radius_width=0.08, angle=52.0, angle_width=7.0)
    b = PeakBox(radius=1.502, radius_width=0.081, angle=52.1, angle_width=7.1)
    assert _select(a, b) == [a]


def test_it_drops_rings():
    """mlgidGUI stores a ring as ``angle_width = inf``, which
    pygidfit's ``boxes_preprocessing`` cannot cast to an int pixel
    row; a ring in the call would also flip the target's cluster to
    pygidfit's peak-on-ring model."""
    ring = PeakBox(
        radius=1.5, radius_width=0.08, angle=45.0, angle_width=float("inf"),
    )
    assert _select(ring) == []


def test_it_drops_degenerate_and_malformed_boxes():
    """``boxes_preprocessing`` raises a bare ``raise`` when a box's two
    angular limits land on the same pixel row. That would fail the
    user's fit because of a *neighbour*, so such boxes never go in."""
    zero_width = PeakBox(
        radius=1.5, radius_width=0.08, angle=52.0, angle_width=0.0,
    )
    off_grid = PeakBox(
        radius=1.5, radius_width=0.08, angle=-40.0, angle_width=7.0,
    )
    nan_box = PeakBox(
        radius=float("nan"), radius_width=0.08, angle=52.0, angle_width=7.0,
    )
    assert _select(zero_width, off_grid, nan_box) == []


def test_it_returns_nothing_for_a_meaningless_q_range():
    near = PeakBox(radius=1.5, radius_width=0.08, angle=52.0, angle_width=7.0)
    assert select_neighbour_boxes(TARGET, [near], q_abs_max=0.0) == []


def _table(radius, radius_width, angle, angle_width, ids, is_ring=None):
    n = len(ids)
    zeros = np.zeros(n, dtype=float)
    return PeakTable(
        q_xy=zeros, q_z=zeros,
        angle=np.asarray(angle, dtype=float),
        radius=np.asarray(radius, dtype=float),
        angle_width=np.asarray(angle_width, dtype=float),
        radius_width=np.asarray(radius_width, dtype=float),
        is_ring=(np.zeros(n, dtype=bool) if is_ring is None
                 else np.asarray(is_ring, dtype=bool)),
        ids=np.asarray(ids, dtype=int),
        score=zeros, amplitude=zeros,
    )


def test_it_flattens_detected_before_fitted():
    """Detected first, so when a peak appears in both tables it is the
    detected geometry that survives the duplicate rule — the same
    input the pipeline ``run_fitting`` would have used."""
    tables = {
        "detected": _table([1.0], [0.1], [30.0], [6.0], [7]),
        "fitted": _table([2.0], [0.2], [60.0], [8.0], [3]),
    }
    boxes = neighbour_boxes_from_tables(tables)
    assert [b.radius for b in boxes] == [1.0, 2.0]
    assert boxes[0].angle == 30.0 and boxes[1].angle_width == 8.0


def test_it_excludes_the_row_being_fitted():
    tables = {
        "detected": _table(
            [1.0, 1.4], [0.1, 0.1], [30.0, 40.0], [6.0, 6.0], [7, 9],
        ),
        "fitted": None,
    }
    boxes = neighbour_boxes_from_tables(
        tables, exclude_kind="detected", exclude_id=7,
    )
    assert [b.radius for b in boxes] == [1.4]


def test_it_skips_ring_rows_in_the_tables():
    tables = {
        "detected": _table(
            [1.0, 1.4], [0.1, 0.1], [45.0, 40.0],
            [float("inf"), 6.0], [1, 2], is_ring=[True, False],
        ),
    }
    boxes = neighbour_boxes_from_tables(tables)
    assert [b.radius for b in boxes] == [1.4]


@pytest.mark.parametrize("tables", [None, {}, {"detected": None}])
def test_it_tolerates_a_frame_with_no_peak_tables(tables):
    assert neighbour_boxes_from_tables(tables) == []


# -- Window level: where the neighbour boxes come from ------------------

@pytest.mark.gui
def test_the_window_gathers_neighbours_from_the_frame_cache(
    main_window, synthetic_nexus_with_peaks,
):
    """``MainWindow._neighbour_boxes`` reads the viewer's per-frame
    peak cache — the tables the overlays already hold — so a click
    costs no I/O, and it drops the row being fitted."""
    from mlgidgui.image_viewer import SelectedPeak
    from mlgidgui.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    tables = main_window.viewer._frame_peaks.get(0)
    det, fit = tables["detected"], tables["fitted"]
    sel = SelectedPeak(
        kind="detected", frame=0, peak_id=int(det.ids[0]),
        radius=float(det.radius[0]), angle=float(det.angle[0]),
        radius_width=float(det.radius_width[0]),
        angle_width=float(det.angle_width[0]),
        is_ring=False, score=0.0, amplitude=0.0,
    )
    boxes = main_window._neighbour_boxes("entry_0000", 0, sel)
    # Every detected row but the selected one, plus every fitted row.
    assert len(boxes) == (len(det.ids) - 1) + len(fit.ids)
    assert all(b.angle != float(det.angle[0]) for b in boxes)


@pytest.mark.gui
def test_a_frame_without_peaks_yields_no_neighbours(
    main_window, synthetic_nexus_with_peaks,
):
    """Frames 1 and 2 of the fixture carry no analysis group. The
    fall-back read returns empty tables, and a fit there simply runs
    without neighbours instead of failing."""
    from mlgidgui.image_viewer import SelectedPeak
    from mlgidgui.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    sel = SelectedPeak(
        kind="manual", frame=1, peak_id=None,
        radius=1.5, angle=45.0, radius_width=0.2, angle_width=5.0,
        is_ring=False, score=0.0, amplitude=0.0,
    )
    assert main_window._neighbour_boxes("entry_0000", 1, sel) == []
