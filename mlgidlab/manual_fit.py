"""Single-peak 2D fit wrapper around ``pygidfit.fit_data``.

The GUI's "Add to fitted" action used to commit a 1D-Gaussian-derived
row to ``fitted_peaks`` with the 2D shape coefficients (A, B, C,
theta) zero-filled — physics-audit finding F-06's concrete failure
mode (the persisted manual fit was already partly fictitious).

This module replaces that path. ``fit_one_peak`` accepts a single
user-drawn box plus the active **cartesian** frame + q axes +
experimental geometry, replicates pygidfit's pipeline polar
conversion internally (``_get_polar_grid`` + ``polar_conversion``
+ ``img_preprocessing``), and runs ``pygidfit.fit_data`` on a
length-1 input. Using pygidfit's own conversion is what makes the
manual fit byte-identical to what the pipeline ``run_fitting``
writes for the same detected box — an earlier revision passed
mlgidlab's polar image (different resolution + scipy interpolation +
no ``img_preprocessing``) and produced visible drift between the
two paths.

The call also carries the frame's *neighbouring* peak boxes
(``select_neighbour_boxes``). pygidfit masks or joint-fits only the
boxes handed to one ``fit_data`` call, so a single-box call left a
neighbouring peak's intensity unmasked inside the target's ROI —
which, combined with pygidfit bounding the centre a quarter box
outside each edge, is how a manual fit could land outside the box
the user drew.

On any pygidfit failure (raises, returns empty, NaN amplitude) we
raise ``ManualFitError`` so callers can fall back to the legacy
1D + zero-fill behaviour or surface a clear error to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class ManualFitError(RuntimeError):
    """Raised when ``pygidfit.fit_data`` cannot produce a fit for a
    user-drawn box (clustering returned no boxes, the fit didn't
    converge, etc.). Caller is expected to fall back to the legacy
    1D + zero-fill path so the user is never blocked from committing
    a peak."""


@dataclass(frozen=True)
class ManualFitResult:
    """Output of a single-peak 2D fit, in the same field convention as
    the persisted ``fitted_peaks`` row (polar geometry + 2D shape
    coefficients + amplitude). All scalars."""

    radius: float
    radius_width: float
    angle: float
    angle_width: float
    amplitude: float
    A: float
    B: float
    C: float
    theta: float


# Hardcoded polar grid shape pygidfit's ``ProcessDataFromFile.process_single_frame``
# uses (see ``mlgidbase/pygidfit_functions.py::_run_pygidfit_from_file``
# at line ~51: ``polar_shape=np.array([512, 1024])``). Matching this
# resolution + interpolation method makes the manual fit byte-identical
# to the pipeline ``run_fitting`` result. Changing it without changing
# pygidfit would re-introduce the drift the wrapper exists to remove.
_PYGIDFIT_POLAR_SHAPE = (512, 1024)

#: The fit settings a bare ``mlgidBASE.run_fitting()`` uses since mlgidbase
#: 0.1.11 (it was 10 / 10 before). The Pipeline panel starts from these,
#: and every place that needs a fallback for the same settings takes them
#: from here instead of repeating the numbers.
DEFAULT_CLUSTERING_DISTANCE_PEAKS = 2.0
DEFAULT_CLUSTERING_DISTANCE_RINGS = 2.0
DEFAULT_CLUSTERING_EXTEND = 2

#: pygidfit's per-ROI hot-pixel filter: pixels above this percentile of
#: their own ROI are masked before a spot cluster is fitted
#: (``fit_data(hot_pixel_percentile=...)``, new in pygidfit 0.1.8 with 99.9
#: as its default). mlgidbase's ``run_fitting`` never passes it, so the
#: pipeline runs with the default; the manual 2D fit passes the SAME value
#: explicitly so that parity cannot drift if the default ever moves.
HOT_PIXEL_PERCENTILE = 99.9


@dataclass(frozen=True)
class PeakBox:
    """One peak box in the polar q convention the peak tables use:
    ``radius`` / ``radius_width`` in Å⁻¹, ``angle`` / ``angle_width``
    in degrees. Used to hand pygidfit the *neighbouring* peaks of the
    one being fitted so it can mask or joint-fit them."""

    radius: float
    radius_width: float
    angle: float
    angle_width: float


def _px_box(
    box: PeakBox, q_abs_max: float, polar_shape: tuple[int, int],
) -> tuple[float, float, int, int]:
    """Box limits in pygidfit's polar pixel grid, mirroring
    ``pygidfit.box_utils.boxes_preprocessing`` (pygidfit >= 0.1.8:
    including its radial ``radius ± radius_width`` — note *not* ``/2`` —
    the ``N - 1`` pixel scale of the ``np.linspace(.., N)`` grid, and the
    ±1 px rounding pad, then clipping into the image).

    Returns ``(r1, r2, t1, t2)``; the angular pair is integral because
    ``boxes_preprocessing`` casts it with ``.astype(int)`` and compares
    the two for equality.
    """
    n_ang, n_rad = int(polar_shape[0]), int(polar_shape[1])
    r1 = round((box.radius - box.radius_width) / q_abs_max * (n_rad - 1)) - 1
    r2 = round((box.radius + box.radius_width) / q_abs_max * (n_rad - 1)) + 1
    half_a = abs(box.angle_width) / 2.0
    t1 = int(round((box.angle - half_a) / 90.0 * (n_ang - 1))) - 1
    t2 = int(round((box.angle + half_a) / 90.0 * (n_ang - 1))) + 1
    return (
        float(min(max(r1, 0), n_rad)), float(min(max(r2, 0), n_rad)),
        int(min(max(t1, 0), n_ang)), int(min(max(t2, 0), n_ang)),
    )


def _overlaps_drawn(a: PeakBox, b: PeakBox) -> bool:
    """True when either box's centre lies inside the other's *drawn*
    extent (``centre ± width/2``) — i.e. the two rows describe the
    same physical peak under different names (a detected box and the
    fitted row made from it, or a manual box drawn over a detected
    one). Such a pair must not be handed to pygidfit twice: it would
    put two Gaussians on one peak."""
    def inside(inner: PeakBox, outer: PeakBox) -> bool:
        return (
            abs(inner.radius - outer.radius) <= abs(outer.radius_width) / 2.0
            and abs(inner.angle - outer.angle) <= abs(outer.angle_width) / 2.0
        )
    return inside(a, b) or inside(b, a)


def select_neighbour_boxes(
    target: PeakBox,
    candidates,
    *,
    q_abs_max: float,
    polar_shape: tuple[int, int] = _PYGIDFIT_POLAR_SHAPE,
    clustering_distance_peaks: float = DEFAULT_CLUSTERING_DISTANCE_PEAKS,
    clustering_extend: int = DEFAULT_CLUSTERING_EXTEND,
) -> list[PeakBox]:
    """Pick the peaks that must travel with ``target`` into ``fit_data``.

    pygidfit only knows about the boxes passed in one call. Within a
    call it does the right thing — boxes closer than
    ``clustering_distance_peaks`` are fitted jointly as a sum of
    Gaussians, boxes outside the current cluster are NaN-ed out of its
    ROI (``pygidfit.fitting_models.fit_peak_cluster``) — but a
    single-box call disables both, so the target's ROI still contains
    the neighbour's intensity and the fit can drift onto it. Passing
    the neighbours switches that machinery back on.

    Only *nearby* candidates are worth passing: a box that neither
    clusters with the target nor intersects its ROI changes nothing
    for the target and costs a whole extra cluster fit. The filter is
    an intersection test in pygidfit's polar pixel grid against the
    target box inflated by ``clustering_distance_peaks +
    2 × clustering_extend`` — one clustering hop plus the bbox
    padding. (A chain of three boxes where only the middle one is
    near the target is not followed; that second-order case would
    need the full frame and the full-frame cost.)

    Dropped, in order:

    * anything non-finite, zero-width or negative-width;
    * rings — mlgidlab stores them as ``angle_width = inf``, which
      ``boxes_preprocessing`` cannot cast to int, and a ring in the
      call would flip the target's cluster to pygidfit's peak-on-ring
      model. Ring neighbours stay out of scope for now;
    * boxes that clip to a degenerate pixel box (``boxes_preprocessing``
      raises a bare ``raise`` when a box's two angular limits land on
      the same pixel row, which would fail the user's fit because of a
      *neighbour*);
    * the target itself, and any candidate describing the same peak
      (see ``_overlaps_drawn``), including duplicates among the
      candidates themselves.

    Pure function: no Qt, no pygidfit import, no I/O.
    """
    if not (math.isfinite(q_abs_max) and q_abs_max > 0):
        return []
    pad = abs(float(clustering_distance_peaks)) + 2.0 * abs(float(clustering_extend))
    t_r1, t_r2, t_t1, t_t2 = _px_box(target, q_abs_max, polar_shape)
    lo_r, hi_r = t_r1 - pad, t_r2 + pad
    lo_t, hi_t = t_t1 - pad, t_t2 + pad

    kept: list[PeakBox] = []
    for cand in candidates:
        values = (cand.radius, cand.radius_width, cand.angle, cand.angle_width)
        if not all(math.isfinite(float(v)) for v in values):
            continue
        if abs(cand.radius_width) <= 0.0 or abs(cand.angle_width) <= 0.0:
            continue
        if _overlaps_drawn(target, cand):
            continue
        if any(_overlaps_drawn(other, cand) for other in kept):
            continue
        c_r1, c_r2, c_t1, c_t2 = _px_box(cand, q_abs_max, polar_shape)
        if c_r2 <= c_r1 or c_t2 <= c_t1:
            continue
        if c_r2 < lo_r or c_r1 > hi_r or c_t2 < lo_t or c_t1 > hi_t:
            continue
        kept.append(cand)
    return kept


def neighbour_boxes_from_tables(tables, *, exclude_kind=None, exclude_id=None):
    """Flatten a frame's ``{kind: PeakTable}`` mapping into ``PeakBox``es.

    Reads the ``detected`` table first and ``fitted`` second, so when
    the same peak appears in both it is the detected geometry that
    survives ``select_neighbour_boxes``'s duplicate rule — the same
    input the pipeline ``run_fitting`` would have used. Rows flagged
    ``is_ring`` are skipped here as well as in the selector, since
    they never make it into the fit.

    ``exclude_kind`` / ``exclude_id`` drop the row the user is fitting
    (the geometric duplicate rule would catch it too, but only when
    the committed row still sits where it was drawn).
    """
    out: list[PeakBox] = []
    for kind in ("detected", "fitted"):
        table = (tables or {}).get(kind)
        if table is None:
            continue
        ids = np.asarray(table.ids).ravel()
        is_ring = np.asarray(table.is_ring).ravel()
        for i in range(ids.size):
            if kind == exclude_kind and exclude_id is not None and int(ids[i]) == int(exclude_id):
                continue
            if bool(is_ring[i]):
                continue
            out.append(PeakBox(
                radius=float(np.asarray(table.radius).ravel()[i]),
                radius_width=float(np.asarray(table.radius_width).ravel()[i]),
                angle=float(np.asarray(table.angle).ravel()[i]),
                angle_width=float(np.asarray(table.angle_width).ravel()[i]),
            ))
    return out


def fit_one_peak(
    cartesian_image: np.ndarray,
    q_xy: np.ndarray,
    q_z: np.ndarray,
    *,
    radius: float,
    radius_width: float,
    angle: float,
    angle_width: float,
    wavelength_angstrom: float,
    q_xy_max: float,
    q_z_max: float,
    ai_deg: float = 0.0,
    crit_angle: float = 0.0,
    theta_fixed: bool = True,
    clustering_distance_peaks: float = DEFAULT_CLUSTERING_DISTANCE_PEAKS,
    clustering_distance_rings: float = DEFAULT_CLUSTERING_DISTANCE_RINGS,
    clustering_extend: int = DEFAULT_CLUSTERING_EXTEND,
    neighbours: "list[PeakBox] | None" = None,
    hot_pixel_percentile: "float | None" = HOT_PIXEL_PERCENTILE,
) -> ManualFitResult:
    """Run ``pygidfit.fit_data`` on a single user-drawn box.

    Inputs match pygidfit's pipeline convention:

    * ``cartesian_image``: the entry's ``data/img_gid_q`` frame in
      reciprocal space. Shape ``(n_qz, n_qxy)`` per pygid's array
      layout — this is what ``FrameSource.get_cartesian(frame)``
      returns.
    * ``q_xy`` / ``q_z``: 1-D coordinate axes in Å⁻¹.
    * ``radius`` / ``radius_width`` / ``angle`` / ``angle_width``:
      the user's box in polar q-space (Å⁻¹ / degrees).
    * geometry kwargs (``wavelength_angstrom``, ``q_*_max``,
      ``ai_deg``, ``crit_angle``): read from the entry's instrument
      metadata.
    * fit kwargs (``theta_fixed``, ``clustering_distance_*``,
      ``clustering_extend``): caller should pass the same values
      the pipeline panel uses so manual Add-to-fitted runs with
      the SAME config as the next ``run_fitting`` would.
    * ``neighbours``: the frame's other peak boxes. pygidfit only
      masks / joint-fits boxes handed to the *same* call, so without
      these a neighbouring peak's intensity sits unmasked in the
      target's ROI and can pull the fit out of the drawn box.
      ``select_neighbour_boxes`` filters them down to the ones that
      can actually reach the target's ROI; the rest are dropped so a
      click does not pay for a whole-frame fit. The target is always
      box 0, and only box 0's result is returned.

    Internally the wrapper replicates
    ``pygidfit.process_scans.ProcessDataFromFile.process_single_frame``:
    apply ``img_preprocessing`` (masks rows below the sample
    horizon to NaN using ``crit_angle``), build a polar grid via
    ``_get_polar_grid`` at the hardcoded ``(512, 1024)`` resolution,
    resample with ``polar_conversion`` (cv2 linear interp), then
    call ``fit_data``. Doing the conversion here — rather than
    accepting mlgidlab's polar resample — is what makes the result
    byte-identical to the pipeline output.

    Returns a ``ManualFitResult`` with the 2D-fit values pygidfit
    produces for the box. Raises ``ManualFitError`` on any pygidfit
    failure so the caller can fall back to the legacy 1D path or
    surface a clear error.
    """
    # Lazy import — pygidfit pulls torch / scipy / scikit-learn and
    # warming those at module import would slow GUI startup. The
    # first ``Add to fitted`` click pays the cost; subsequent clicks
    # are fast.
    try:
        import cv2  # noqa: F401
        from pygidfit.process_scans import (
            _get_polar_grid,
            fit_data,
            img_preprocessing,
            polar_conversion,
        )
    except Exception as exc:  # pragma: no cover — env-dependent
        raise ManualFitError(
            f"pygidfit / cv2 not available ({exc!r}); cannot run 2D fit"
        ) from exc

    # 1) Preprocess: mask rows below the sample horizon (q_z <
    # q_z_critical from ``calc_smpl_hor(ai, crit_angle, wavelength)``)
    # and any non-positive pixels to NaN. This matches what
    # ProcessDataFromFile.process_single_frame does and is the
    # difference that gave the most drift in the manual-vs-pipeline
    # A/B without it.
    try:
        img_pre = img_preprocessing(
            np.asarray(cartesian_image),
            float(ai_deg),
            float(crit_angle),
            float(wavelength_angstrom),
            np.asarray(q_z),
        )
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit.img_preprocessing raised: {exc!r}"
        ) from exc

    # 2) Build polar grid at pygidfit's hardcoded shape. The
    # ``beam_center=[0, 0]`` is what mlgidbase passes too — pygidfit
    # treats the image as a quadrant whose ``(0,0)`` index lies at
    # the q-space origin. (See pygidfit's _get_polar_grid for the
    # exact construction; the wrapper just mirrors the pipeline.)
    try:
        yy, zz, ang_deg_max = _get_polar_grid(
            img_pre.shape, _PYGIDFIT_POLAR_SHAPE, [0, 0],
        )
        polar_img = polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit polar conversion raised: {exc!r}"
        ) from exc

    # 3) Compute ``q_abs_max`` the same way ProcessDataFromFile does:
    # ``np.sqrt(np.nanmax(q_z)**2 + np.nanmax(q_xy)**2)``. NOT
    # ``max(polar_radius_axis)`` (which is what an earlier revision
    # of this wrapper used and was a second source of drift).
    q_abs_max = float(np.sqrt(
        np.nanmax(np.asarray(q_z)) ** 2 + np.nanmax(np.asarray(q_xy)) ** 2
    ))

    # 4) Run pygidfit's fit. The user's box is index 0; the
    # neighbours that can reach its ROI follow, so pygidfit's own
    # masking (other clusters NaN-ed out of the ROI) and joint
    # fitting (boxes within ``clustering_distance_peaks`` fitted as a
    # sum of Gaussians) apply, exactly as they do in the pipeline
    # ``run_fitting`` call that passes a whole frame at once.
    target = PeakBox(
        radius=float(radius), radius_width=float(radius_width),
        angle=float(angle), angle_width=float(angle_width),
    )
    extra = select_neighbour_boxes(
        target, list(neighbours or []),
        q_abs_max=q_abs_max,
        polar_shape=_PYGIDFIT_POLAR_SHAPE,
        clustering_distance_peaks=clustering_distance_peaks,
        clustering_extend=clustering_extend,
    )
    if extra:
        logger.debug(
            "manual 2D fit: %d neighbouring box(es) passed to pygidfit "
            "for masking / joint fitting", len(extra),
        )
    boxes = [target, *extra]
    radius_arr = np.array([b.radius for b in boxes], dtype=float)
    radius_width_arr = np.array([b.radius_width for b in boxes], dtype=float)
    angle_arr = np.array([b.angle for b in boxes], dtype=float)
    angle_width_arr = np.array([b.angle_width for b in boxes], dtype=float)
    try:
        container, _peaks_pool = fit_data(
            polar_img,
            radius=radius_arr,
            radius_width=radius_width_arr,
            angle=angle_arr,
            angle_width=angle_width_arr,
            wavelength=float(wavelength_angstrom),
            q_xy_max=float(q_xy_max),
            q_z_max=float(q_z_max),
            q_abs_max=q_abs_max,
            ang_deg_max=ang_deg_max,
            clustering_distance_peaks=float(clustering_distance_peaks),
            clustering_distance_rings=float(clustering_distance_rings),
            clustering_extend=int(clustering_extend),
            theta_fixed=bool(theta_fixed),
            debug=False,
            multiprocessing=False,
            peaks_pool=None,
            ai=float(ai_deg),
            hot_pixel_percentile=hot_pixel_percentile,
        )
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit.fit_data raised: {exc!r}"
        ) from exc

    # pygidfit may cluster the box away or produce a degenerate
    # output — both surface as a container with empty arrays.
    # ``_data2container`` builds every array by iterating the box list
    # in input order and writes ``container.id = box.index``, so the
    # user's box is row 0; the id lookup is belt and braces.
    amp = np.asarray(container.amplitude).ravel()
    if amp.size == 0:
        raise ManualFitError(
            "pygidfit produced no output for the user-drawn box "
            "(clustering may have dropped the input or the fit did "
            "not converge)"
        )
    idx = _target_row(container, amp.size)
    if not np.isfinite(amp[idx]):
        raise ManualFitError(
            "pygidfit returned NaN amplitude — the 2D fit did not "
            "converge on the user-drawn box"
        )

    # Return pygidfit's container values verbatim, no width
    # conversion. mlgidbase's pipeline ``run_fitting`` path stores
    # the same container directly, so manually-added peaks and
    # pipeline-fitted peaks share the same width convention
    # (pygidfit's ``2σ`` in both ``radius_width`` and ``angle_width``,
    # per ``_data2container``'s ``*2`` scaling).
    def _row(name: str) -> float:
        return float(np.asarray(getattr(container, name)).ravel()[idx])

    return ManualFitResult(
        radius=_row("radius"),
        radius_width=_row("radius_width"),
        angle=_row("angle"),
        angle_width=_row("angle_width"),
        amplitude=float(amp[idx]),
        A=_row("A"),
        B=_row("B"),
        C=_row("C"),
        theta=_row("theta"),
    )


def _target_row(container, n_rows: int) -> int:
    """Index of the user's box in a pygidfit container.

    ``_data2container`` preserves input order, so it is 0. It is
    looked up through ``container.id`` (which carries ``box.index``,
    the position in the input arrays) so a future pygidfit that
    reorders its output cannot silently return a neighbour's fit as
    the user's.
    """
    ids = getattr(container, "id", None)
    if ids is not None:
        ids = np.asarray(ids).ravel()
        hit = np.flatnonzero(ids == 0)
        if hit.size and int(hit[0]) < n_rows:
            return int(hit[0])
    return 0
