"""Pure-numpy unit tests for ``mlgidgui.roi_intensity`` — the
background-subtracted integrated ROI intensity behind the amplitude
tab's "integrated intensity (ROI)" metric. No Qt, no files."""
from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.phase_tracking import TrackingPayload
from mlgidgui.roi_intensity import (
    axis_slice,
    integrate_roi,
    track_frame_positions,
)


def _payload(frames, q_xy, q_z):
    n = len(frames)
    return TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.asarray(q_xy, dtype=float),
        q_z=np.asarray(q_z, dtype=float),
        frame_num=np.asarray(frames, dtype=int),
        amplitude=np.ones(n),
        components=[list(range(n))],
    )


def test_track_frame_positions_members_and_nearest_fill():
    """Member frames use the frame's (mean) fitted position; every
    other frame borrows the NEAREST member frame's position — ties go
    to the earlier frame — so the trace extends gap-free over the
    whole scan including the lead-in before the peak appeared."""
    # Members on frames 2 (twice -> averaged) and 5.
    p = _payload(
        frames=[2, 2, 5],
        q_xy=[1.0, 3.0, 10.0],
        q_z=[0.4, 0.6, 2.0],
    )
    cx, cz = track_frame_positions(p, 0, 8)
    # Frames 0..3 nearest to member frame 2 (frame 3 is nearer to 2
    # than to 5); 4..7 nearest to 5 (frame 4: |4-2|=2 > |5-4|=1).
    np.testing.assert_allclose(cx, [2, 2, 2, 2, 10, 10, 10, 10])
    np.testing.assert_allclose(
        cz, [0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2]
    )


def test_track_frame_positions_tie_prefers_earlier():
    p = _payload(frames=[2, 4], q_xy=[1.0, 9.0], q_z=[0.1, 0.9])
    cx, _cz = track_frame_positions(p, 0, 6)
    assert cx[3] == 1.0          # equidistant to 2 and 4 -> frame 2
    np.testing.assert_allclose(cx, [1, 1, 1, 1, 9, 9])


def test_axis_slice_covers_value_range():
    axis = np.linspace(0.0, 1.0, 11)          # 0, 0.1, ..., 1.0
    i0, i1 = axis_slice(axis, 0.25, 0.55)
    assert (i0, i1) == (3, 6)                 # 0.3, 0.4, 0.5
    # Exact endpoints are inclusive on both sides.
    i0, i1 = axis_slice(axis, 0.3, 0.5)
    assert (i0, i1) == (3, 6)


def _frame_and_axes(n_z=40, n_xy=50, value=2.0):
    frame = np.full((n_z, n_xy), value)
    q_xy = np.linspace(0.0, 1.0, n_xy)
    q_z = np.linspace(0.0, 1.0, n_z)
    return frame, q_xy, q_z


def test_integrate_roi_flat_background_is_zero():
    frame, q_xy, q_z = _frame_and_axes()
    val = integrate_roi(frame, q_xy, q_z, 0.5, 0.5, 0.1, 0.1, 2, 4)
    assert val == pytest.approx(0.0)


def test_integrate_roi_recovers_peak_over_background():
    """A peak sitting on a constant pedestal integrates to exactly its
    own counts — the pedestal cancels through the strip background."""
    frame, q_xy, q_z = _frame_and_axes(value=3.0)
    z0, z1 = axis_slice(q_z, 0.45, 0.55)
    x0, x1 = axis_slice(q_xy, 0.45, 0.55)
    frame[z0:z1, x0:x1] += 7.0
    peak_counts = 7.0 * (z1 - z0) * (x1 - x0)
    val = integrate_roi(frame, q_xy, q_z, 0.5, 0.5, 0.1, 0.1, 2, 4)
    assert val == pytest.approx(peak_counts)


def test_integrate_roi_flanks_along_qz_for_near_axis_peak():
    """|q_z| >= |q_xy|: the strips sit below/above the box (along
    q_z). Rows there carry the true background; the columns left and
    right are poisoned — a wrong flank direction would show."""
    frame, q_xy, q_z = _frame_and_axes(value=0.0)
    z0, z1 = axis_slice(q_z, 0.5 - 0.1, 0.5 + 0.1)
    x0, x1 = axis_slice(q_xy, 0.05 - 0.04, 0.05 + 0.04)
    frame[:, :x0] = 999.0                     # left of the box
    frame[:, x1:] = 999.0                     # right of the box
    frame[z0 - 6:z0 - 2, x0:x1] = 2.0         # gap 2, strip 4 below
    frame[z1 + 2:z1 + 6, x0:x1] = 2.0         # ... and above
    frame[z0:z1, x0:x1] = 5.0
    val = integrate_roi(frame, q_xy, q_z, 0.05, 0.5, 0.04, 0.1, 2, 4)
    expected = (5.0 - 2.0) * (z1 - z0) * (x1 - x0)
    assert val == pytest.approx(expected)


def test_integrate_roi_flanks_along_qxy_off_axis():
    """|q_xy| > |q_z|: the strips sit left/right of the box (along
    q_xy, ~ the radial direction); rows above/below are poisoned."""
    frame, q_xy, q_z = _frame_and_axes(value=0.0)
    z0, z1 = axis_slice(q_z, 0.05 - 0.04, 0.05 + 0.04)
    x0, x1 = axis_slice(q_xy, 0.5 - 0.1, 0.5 + 0.1)
    frame[:z0, :] = 999.0
    frame[z1:, :] = 999.0
    frame[z0:z1, x0 - 6:x0 - 2] = 2.0
    frame[z0:z1, x1 + 2:x1 + 6] = 2.0
    frame[z0:z1, x0:x1] = 5.0
    val = integrate_roi(frame, q_xy, q_z, 0.5, 0.05, 0.1, 0.04, 2, 4)
    expected = (5.0 - 2.0) * (z1 - z0) * (x1 - x0)
    assert val == pytest.approx(expected)


def test_integrate_roi_pools_both_flanks():
    """The background is the POOLED median of both strips: one strip
    fully on a neighbor peak's tail (value 4) and one on true
    background (value 0) subtract the pooled midpoint, not the
    contaminated strip alone. This is the observed real-data failure a
    single background box produces (a whole trace pushed negative)."""
    frame, q_xy, q_z = _frame_and_axes(value=0.0)
    z0, z1 = axis_slice(q_z, 0.5 - 0.1, 0.5 + 0.1)
    x0, x1 = axis_slice(q_xy, 0.05 - 0.04, 0.05 + 0.04)
    frame[z0 - 6:z0 - 2, x0:x1] = 4.0         # contaminated flank
    frame[z1 + 2:z1 + 6, x0:x1] = 0.0         # clean flank
    frame[z0:z1, x0:x1] = 5.0
    val = integrate_roi(frame, q_xy, q_z, 0.05, 0.5, 0.04, 0.1, 2, 4)
    # Equal-size flanks at 0 and 4 -> pooled median = midpoint 2.
    expected = (5.0 - 2.0) * (z1 - z0) * (x1 - x0)
    assert val == pytest.approx(expected)


def test_integrate_roi_median_ignores_minority_contamination():
    """A bright feature under LESS than half of the pooled strip
    pixels (a neighbor's tail clipping one strip's corner) does not
    inflate the background at all — the median stays on the clean
    level. With a mean this over-subtraction made a monotonically
    rising real trace dip."""
    frame, q_xy, q_z = _frame_and_axes(value=0.0)
    z0, z1 = axis_slice(q_z, 0.5 - 0.1, 0.5 + 0.1)
    x0, x1 = axis_slice(q_xy, 0.05 - 0.04, 0.05 + 0.04)
    frame[z0:z1, x0:x1] = 5.0
    # Contaminate half of one strip (a quarter of the pooled pixels).
    frame[z0 - 6:z0 - 4, x0:x1] = 1000.0
    val = integrate_roi(frame, q_xy, q_z, 0.05, 0.5, 0.04, 0.1, 2, 4)
    expected = 5.0 * (z1 - z0) * (x1 - x0)    # background stays 0
    assert val == pytest.approx(expected)


def test_integrate_roi_nan_pixels_ignored():
    """NaN pixels count neither into the box sum nor the background,
    and the background scales with the box's FINITE pixel count."""
    frame, q_xy, q_z = _frame_and_axes(value=2.0)
    z0, z1 = axis_slice(q_z, 0.5 - 0.1, 0.5 + 0.1)
    x0, x1 = axis_slice(q_xy, 0.5 - 0.1, 0.5 + 0.1)
    frame[z0, x0] = np.nan                    # one box pixel dead
    frame[z0 - 4, x0] = np.nan                # one strip pixel dead
    val = integrate_roi(frame, q_xy, q_z, 0.5, 0.5, 0.1, 0.1, 2, 4)
    assert val == pytest.approx(0.0)          # flat field still cancels


def test_integrate_roi_box_clipped_at_edge():
    """A box reaching over the image edge integrates its clipped part;
    a box entirely off the image yields NaN."""
    frame, q_xy, q_z = _frame_and_axes(value=1.0)
    val = integrate_roi(frame, q_xy, q_z, 0.0, 0.5, 0.05, 0.05, 2, 4)
    assert np.isfinite(val)
    assert np.isnan(
        integrate_roi(frame, q_xy, q_z, 5.0, 5.0, 0.05, 0.05, 2, 4)
    )


def test_integrate_roi_no_background_pixels_falls_back_to_sum():
    """When both strips clip away entirely (box at the very corner of
    a tiny image), the plain box sum is returned instead of NaN."""
    frame = np.full((4, 4), 3.0)
    q_xy = np.linspace(0.0, 1.0, 4)
    q_z = np.linspace(0.0, 1.0, 4)
    val = integrate_roi(frame, q_xy, q_z, 0.5, 0.5, 1.0, 1.0, 2, 4)
    assert val == pytest.approx(3.0 * 16)
