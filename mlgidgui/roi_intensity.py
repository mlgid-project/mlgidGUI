"""Background-subtracted integrated ROI intensity for tracked peaks.

The fitted-amplitude metric of the amplitude-evolution view has two
failure modes on real scans: frames where the fit fails produce gaps
or spurious zeros, and on very intense peaks the optimizer trades
amplitude against width frame to frame. The ROI metric integrates the
IMAGE data itself in a fixed q-space box around each tracked peak, so
it is gap-free and proportional to the diffracting volume.

Per tracked peak and frame:

* the box is centered on the fitted peak position of that frame (the
  frame's members averaged when several), or — when the frame has no
  fitted member — on the position of the NEAREST frame that has one,
  so a failed fit never leaves a gap in the trace;
* integrated intensity = sum of the box's counts minus the local
  background: the MEDIAN per-pixel intensity of TWO flanking strips
  just outside the box, times the box's (finite-)pixel count. Both
  flanks are pooled on purpose — a single background strip can land on
  the tail of a neighboring intense peak and over-subtract the whole
  trace — and the median (not the mean) is the central estimate so a
  bright feature under part of the pooled strips (a neighbor's tail,
  the peak's own reflection-split twin near the critical angle) is
  ignored instead of inflating the background: with the mean, exactly
  that made the 00285 (002) reference trace dip although the true
  intensity rises monotonically. The strips flank along the
  (axis-aligned approximation of the) radial direction: along q_z for
  near-axis peaks (|q_z| >= |q_xy|), along q_xy otherwise, so they
  sample off the powder ring the peak sits on.

Pure numpy — the per-frame file reads live in
``workers.RoiTraceWorker``, which calls ``integrate_roi`` once per
(frame, track) on a frame it read once.
"""
from __future__ import annotations

import numpy as np

# ROI half-widths in 1/Angstrom (the box spans center +- half-width on
# each q axis) and the background-strip geometry in pixels.
DEFAULT_HALF_Q_XY = 0.03
DEFAULT_HALF_Q_Z = 0.03
DEFAULT_BG_GAP_PX = 2
DEFAULT_BG_STRIP_PX = 4


def track_frame_positions(
    payload, k: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ROI centers ``(q_xy, q_z)`` for track ``k`` over
    frames ``0..n_frames-1``.

    Frames with fitted members use the mean fitted position of that
    frame's members; every other frame borrows the position of the
    nearest member frame (ties resolve to the earlier one). A track
    always has at least one fitted member, so the fill never fails —
    the trace extends over the whole scan, including frames before the
    peak first appeared.
    """
    members = payload.track_members(k)
    frames = np.asarray(payload.frame_num[members], dtype=int)
    uf, inv = np.unique(frames, return_inverse=True)
    sums_x = np.zeros(uf.size)
    sums_z = np.zeros(uf.size)
    counts = np.zeros(uf.size)
    np.add.at(sums_x, inv, np.asarray(payload.q_xy[members], dtype=float))
    np.add.at(sums_z, inv, np.asarray(payload.q_z[members], dtype=float))
    np.add.at(counts, inv, 1.0)
    cx = sums_x / counts
    cz = sums_z / counts

    all_frames = np.arange(int(n_frames))
    idx = np.clip(np.searchsorted(uf, all_frames), 0, uf.size - 1)
    prev = np.clip(idx - 1, 0, uf.size - 1)
    take_prev = (
        np.abs(all_frames - uf[prev]) <= np.abs(uf[idx] - all_frames)
    )
    choice = np.where(take_prev, prev, idx)
    return cx[choice], cz[choice]


def axis_slice(axis: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Half-open pixel index range ``[i0, i1)`` of the axis values in
    ``[lo, hi]``. The axis must be monotonically increasing (pygid's
    q axes are)."""
    i0 = int(np.searchsorted(axis, lo, side="left"))
    i1 = int(np.searchsorted(axis, hi, side="right"))
    return i0, i1


def integrate_roi(
    frame: np.ndarray,
    q_xy_axis: np.ndarray,
    q_z_axis: np.ndarray,
    q_xy: float,
    q_z: float,
    half_q_xy: float = DEFAULT_HALF_Q_XY,
    half_q_z: float = DEFAULT_HALF_Q_Z,
    gap_px: int = DEFAULT_BG_GAP_PX,
    strip_px: int = DEFAULT_BG_STRIP_PX,
) -> float:
    """Background-subtracted integrated intensity of one ROI on one
    frame (see the module docstring for the method).

    ``frame`` is ``(n_q_z, n_q_xy)`` with both axes increasing. Boxes
    reaching over the image edge are clipped; NaN pixels count neither
    into the sum nor into the background. Returns NaN when the box
    misses the image entirely (or is all-NaN); with no usable
    background pixels the plain box sum is returned.
    """
    frame = np.asarray(frame)
    z0, z1 = axis_slice(q_z_axis, q_z - half_q_z, q_z + half_q_z)
    x0, x1 = axis_slice(q_xy_axis, q_xy - half_q_xy, q_xy + half_q_xy)
    nz, nx = frame.shape
    z0, z1 = max(z0, 0), min(z1, nz)
    x0, x1 = max(x0, 0), min(x1, nx)
    if z0 >= z1 or x0 >= x1:
        return float("nan")
    box = frame[z0:z1, x0:x1]
    finite = np.isfinite(box)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return float("nan")

    gap_px, strip_px = int(gap_px), int(strip_px)
    if abs(q_z) >= abs(q_xy):
        # Near-axis peak: radial ~ q_z, flank below and above the box.
        strips = (
            frame[max(z0 - gap_px - strip_px, 0):max(z0 - gap_px, 0),
                  x0:x1],
            frame[min(z1 + gap_px, nz):min(z1 + gap_px + strip_px, nz),
                  x0:x1],
        )
    else:
        # Radial ~ q_xy, flank left and right of the box.
        strips = (
            frame[z0:z1,
                  max(x0 - gap_px - strip_px, 0):max(x0 - gap_px, 0)],
            frame[z0:z1,
                  min(x1 + gap_px, nx):min(x1 + gap_px + strip_px, nx)],
        )
    bg_vals = np.concatenate(
        [s[np.isfinite(s)].ravel() for s in strips]
    )
    background = float(np.median(bg_vals)) if bg_vals.size else 0.0
    return float(box[finite].sum() - background * n_finite)
