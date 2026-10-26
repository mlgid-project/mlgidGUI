"""Forward-simulated CIF patterns for the "Expected pattern" overlay.

Pure-numpy helpers that turn the pipeline panel's cached ``CifPattern``
(mlgidmatch ``preprocess.cif_preprocess.CifPattern``, built by
``pipeline.parse_cif_input`` with ``create_all=True``) into displayable
reflection lists, classify them against real peak boxes, and shape the
selected ones into detected-box injection specs.

The input object is duck-typed — only these attributes are read:

* ``cifs``                 — list of CIF basenames.
* ``pattern_3d.orientations`` — per-CIF ``(orient_num, 3)`` float arrays
  of symmetry-distinct hkl texture directions.
* ``all_patterns_q2d`` / ``all_patterns_int2d`` — per (CIF, orientation)
  2D peak positions ``(N, 2)`` = columns ``[q_xy, q_z]`` + intensities.
* ``all_patterns_q1d`` / ``all_patterns_int1d`` — per-CIF powder ring
  ``|q|`` positions + intensities.

Everything here reads PRECOMPUTED arrays only. Never simulate on
demand via ``pygidsim.GIWAXS.giwaxs_2d`` — its signature diverges
across installed pygidsim versions (``q_range=`` vs ``q_xy_range=`` /
``q_z_range=``), so an on-demand call would work in one environment
and TypeError in the next.

Coordinate conventions match the rest of the codebase (``polar.py``):
radius = |q| in Å⁻¹, angle in degrees from +q_xy toward +q_z. The
simulated 2D patterns are mirrored into negative q_xy upstream
(``move_fromMW=True``); ``extract_pattern`` folds them back to
q_xy >= 0 and dedupes, because every persisted peak row lives in the
first quadrant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from mlgidlab.phase_tracking import RING_BOX_ANGLE, RING_BOX_ANGLE_WIDTH

# Sentinel hkl for the powder (rings) pattern — matches the H5 schema,
# where matched_rings solutions store orientation (0, 0, 0) ("rand").
POWDER_HKL: tuple[int, int, int] = (0, 0, 0)

# Fallback box size for injected detected peaks when the frame has no
# fitted peaks to take a median from.
DEFAULT_RADIUS_WIDTH = 0.1  # Å⁻¹
DEFAULT_ANGLE_WIDTH = 8.0   # degrees

# Rounding used to merge mirror duplicates (±q_xy fold) — 1e-4 Å⁻¹ is
# far below any physical peak separation.
_DEDUPE_DECIMALS = 4


@dataclass(frozen=True)
class SimulatedReflection:
    """One predicted reflection, folded to the first quadrant."""

    index: int            # stable position within SimulatedPattern.reflections
    q_xy: float           # >= 0 (folded)
    q_z: float
    radius: float         # |q| in Å⁻¹
    angle: float          # degrees, +q_xy toward +q_z (polar.py convention)
    rel_intensity: float  # (0, 1], max-normalised within the pattern
    is_ring: bool


@dataclass
class SimulatedPattern:
    """A CIF's expected pattern for one orientation (or powder rings)."""

    cif: str                          # extensionless basename
    hkl: tuple[int, int, int]         # POWDER_HKL for rings
    reflections: list[SimulatedReflection] = field(default_factory=list)

    @property
    def is_powder(self) -> bool:
        return tuple(self.hkl) == POWDER_HKL


def cif_stem(name: str) -> str:
    """Extensionless lowercase basename — the canonical CIF identity
    used for cache lookups and matched-structure comparisons."""
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def cif_index(cif_pattern, cif_name: str) -> int | None:
    """Index of ``cif_name`` in ``cif_pattern.cifs`` (extensionless,
    case-insensitive), or None."""
    want = cif_stem(cif_name)
    for i, name in enumerate(getattr(cif_pattern, "cifs", []) or []):
        if cif_stem(name) == want:
            return i
    return None


def list_orientations(cif_pattern, cif_idx: int) -> list[tuple[int, int, int]]:
    """Distinct hkl orientations for CIF ``cif_idx``, original order.

    Order matters: it is the index alignment with
    ``all_patterns_q2d[cif_idx]``. Rows are float32 upstream; round to
    the integer Miller indices they encode.
    """
    orients = cif_pattern.pattern_3d.orientations[cif_idx]
    out: list[tuple[int, int, int]] = []
    for row in np.asarray(orients, dtype=float):
        h, k, l = (int(v) for v in np.rint(row))
        out.append((h, k, l))
    return out


def parse_hkl(text: str) -> tuple[int, int, int]:
    """Parse a user-typed orientation like ``"0 0 1"`` / ``"-1,1,0"``.

    Accepts three integers separated by spaces, commas or semicolons
    (signs allowed), or three single-digit indices written together
    with optional signs (``010``, ``-110``, ``1-10``), the way they are
    usually printed. Raises ``ValueError`` with a user-readable message
    on anything else — including the direction-less ``0 0 0``, which is
    what the random/powder mode is for.
    """
    import re

    stripped = str(text).strip()
    parts = [p for p in re.split(r"[,;\s]+", stripped) if p]
    if len(parts) == 1 and re.fullmatch(r"(?:[+-]?\d){3}", stripped):
        parts = re.findall(r"[+-]?\d", stripped)
    if len(parts) != 3:
        raise ValueError(
            f"{stripped!r} — type three integers, e.g. 0 0 1, or three "
            "digits written together, e.g. 001"
        )
    try:
        h, k, l = (int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"{str(text).strip()!r} — Miller indices must be integers"
        ) from None
    if (h, k, l) == POWDER_HKL:
        raise ValueError(
            "(0 0 0) has no direction — use the random (powder) mode "
            "for the ring pattern"
        )
    return h, k, l


def resolve_orientation(cif_pattern, cif_idx: int, hkl) -> tuple | None:
    """Match a typed ``hkl`` against the CIF's precomputed orientations.

    Two spellings describe the same texture direction when they differ
    only by a common integer factor or an overall sign, so both sides
    compare gcd-reduced (with the sign flip tried too): ``0 0 1``,
    ``0 0 -1`` and ``0 0 2`` all resolve to whichever spelling the
    precomputed set stores. Returns the stored orientation tuple (so
    ``extract_pattern`` will find it), or None when the direction is
    not simulated.
    """
    from math import gcd

    def reduced(triple) -> tuple[int, int, int]:
        h, k, l = (int(v) for v in triple)
        g = gcd(gcd(abs(h), abs(k)), abs(l))
        return (h // g, k // g, l // g) if g > 1 else (h, k, l)

    want = reduced(hkl)
    want_neg = tuple(-v for v in want)
    try:
        orients = list_orientations(cif_pattern, cif_idx)
    except (AttributeError, IndexError, TypeError):
        return None
    for stored in orients:
        if reduced(stored) in (want, want_neg):
            return stored
    return None


def orientation_index(cif_pattern, cif_idx: int, hkl) -> int | None:
    """Row index of ``hkl`` in the CIF's orientation list, or None.

    Float32-safe: upstream stores orientations as float32, so compare
    with a +-0.5 window rather than exact equality.
    """
    orients = np.asarray(
        cif_pattern.pattern_3d.orientations[cif_idx], dtype=float
    )
    if orients.ndim != 2 or orients.shape[1] != 3:
        return None
    target = np.asarray(hkl, dtype=float).reshape(1, 3)
    hits = np.all(np.abs(orients - target) < 0.5, axis=1)
    idx = np.flatnonzero(hits)
    return int(idx[0]) if idx.size else None


def extract_pattern(cif_pattern, cif_idx: int, hkl) -> SimulatedPattern:
    """Build the displayable pattern for (CIF, orientation).

    ``hkl`` None or ``POWDER_HKL`` selects the powder ring pattern.
    Raises ValueError with a human-readable message when the requested
    pattern was not precomputed (host catches and shows it in the
    status bar — no on-demand simulation fallback).
    """
    cif = cif_stem(cif_pattern.cifs[cif_idx])
    if hkl is None or tuple(int(v) for v in hkl) == POWDER_HKL:
        return SimulatedPattern(
            cif=cif, hkl=POWDER_HKL,
            reflections=_powder_reflections(cif_pattern, cif_idx),
        )
    hkl = tuple(int(v) for v in hkl)
    j = orientation_index(cif_pattern, cif_idx, hkl)
    q2d_all = getattr(cif_pattern, "all_patterns_q2d", None)
    if j is None or q2d_all is None:
        raise ValueError(
            f"No precomputed pattern for {cif} {hkl} — re-parse the CIFs."
        )
    q2d = np.asarray(q2d_all[cif_idx][j], dtype=float)
    ints = np.asarray(
        cif_pattern.all_patterns_int2d[cif_idx][j], dtype=float
    ).ravel()
    return SimulatedPattern(
        cif=cif, hkl=hkl, reflections=_spot_reflections(q2d, ints),
    )


def _powder_reflections(cif_pattern, cif_idx: int) -> list[SimulatedReflection]:
    q1d_all = getattr(cif_pattern, "all_patterns_q1d", None)
    int1d_all = getattr(cif_pattern, "all_patterns_int1d", None)
    if q1d_all is None or int1d_all is None:
        raise ValueError(
            "Powder patterns were not precomputed — re-parse the CIFs."
        )
    radii = np.asarray(q1d_all[cif_idx], dtype=float).ravel()
    ints = np.asarray(int1d_all[cif_idx], dtype=float).ravel()
    n = min(radii.size, ints.size)
    radii, ints = radii[:n], ints[:n]
    keep = np.isfinite(radii) & np.isfinite(ints) & (radii > 0) & (ints > 0)
    radii, ints = radii[keep], ints[keep]
    if radii.size == 0:
        return []
    rel = ints / ints.max()
    order = np.argsort(radii, kind="stable")
    out: list[SimulatedReflection] = []
    for idx, src in enumerate(order):
        r = float(radii[src])
        q_xy, q_z = _ring_nominal_xy(r)
        out.append(SimulatedReflection(
            index=idx, q_xy=q_xy, q_z=q_z, radius=r, angle=45.0,
            rel_intensity=float(rel[src]), is_ring=True,
        ))
    return out


def _ring_nominal_xy(radius: float) -> tuple[float, float]:
    """Nominal Cartesian point for a ring (its 45° bisector)."""
    a = np.deg2rad(45.0)
    return float(radius * np.cos(a)), float(radius * np.sin(a))


def _spot_reflections(q2d: np.ndarray, ints: np.ndarray) -> list[SimulatedReflection]:
    if q2d.ndim != 2:
        return []
    # Defensive: accept a (2, N) layout should upstream ever store the
    # untransposed array.
    if q2d.shape[1] != 2 and q2d.shape[0] == 2:
        q2d = q2d.T
    if q2d.shape[1] != 2:
        return []
    n = min(q2d.shape[0], ints.size)
    q_xy = q2d[:n, 0].astype(float)
    q_z = q2d[:n, 1].astype(float)
    ints = ints[:n].astype(float)
    keep = (
        np.isfinite(q_xy) & np.isfinite(q_z) & np.isfinite(ints)
        & (ints > 0)
        # Zero-padded rows (and the unphysical q=0 reflection).
        & ((q_xy != 0.0) | (q_z != 0.0))
    )
    q_xy, q_z, ints = q_xy[keep], q_z[keep], ints[keep]
    if q_xy.size == 0:
        return []
    # Fold the upstream ±q_xy mirror back into the first quadrant and
    # merge duplicates, keeping the strongest intensity per position.
    q_xy = np.abs(q_xy)
    merged: dict[tuple[float, float], list[float]] = {}
    for x, z, i in zip(q_xy, q_z, ints):
        key = (round(x, _DEDUPE_DECIMALS), round(z, _DEDUPE_DECIMALS))
        slot = merged.get(key)
        if slot is None:
            merged[key] = [x, z, i]
        elif i > slot[2]:
            merged[key] = [x, z, i]
    xs = np.array([v[0] for v in merged.values()])
    zs = np.array([v[1] for v in merged.values()])
    vals = np.array([v[2] for v in merged.values()])
    rel = vals / vals.max()
    radius = np.hypot(xs, zs)
    angle = np.degrees(np.arctan2(zs, xs))
    order = np.lexsort((angle, radius))
    out: list[SimulatedReflection] = []
    for idx, src in enumerate(order):
        out.append(SimulatedReflection(
            index=idx, q_xy=float(xs[src]), q_z=float(zs[src]),
            radius=float(radius[src]), angle=float(angle[src]),
            rel_intensity=float(rel[src]), is_ring=False,
        ))
    return out


def merge_boxes(tables):
    """Merge PeakTable-like box tables into one ``classify_explained``
    input (None when nothing usable). Only the five box-geometry
    columns are carried."""
    tables = [t for t in tables if t is not None and len(t) > 0]
    if not tables:
        return None
    return SimpleNamespace(
        radius=np.concatenate(
            [np.asarray(t.radius, dtype=float) for t in tables]
        ),
        radius_width=np.concatenate(
            [np.asarray(t.radius_width, dtype=float) for t in tables]
        ),
        angle=np.concatenate(
            [np.asarray(t.angle, dtype=float) for t in tables]
        ),
        angle_width=np.concatenate(
            [np.asarray(t.angle_width, dtype=float) for t in tables]
        ),
        is_ring=np.concatenate(
            [np.asarray(t.is_ring, dtype=bool) for t in tables]
        ),
    )


def specs_to_boxes(specs):
    """Duck-typed box table from injection-spec dicts (the
    ``build_injection_specs`` output), usable as a
    ``classify_explained`` / ``merge_boxes`` input — lets a planner
    treat boxes it has already decided to inject as coverage, so
    overlapping predictions from different patterns don't produce
    duplicate boxes. None for an empty spec list."""
    if not specs:
        return None
    return SimpleNamespace(
        radius=np.array([float(s["radius"]) for s in specs]),
        radius_width=np.array([float(s["radius_width"]) for s in specs]),
        angle=np.array([float(s["angle"]) for s in specs]),
        angle_width=np.array([float(s["angle_width"]) for s in specs]),
        is_ring=np.array([bool(s.get("is_ring", False)) for s in specs]),
    )


def classify_explained(
    reflections, boxes, *,
    seed_radius_width: float = DEFAULT_RADIUS_WIDTH,
    seed_angle_width: float = DEFAULT_ANGLE_WIDTH,
    min_overlap: float = 0.5,
) -> np.ndarray:
    """Boolean mask: which reflections are explained by ``boxes``.

    ``boxes`` is PeakTable-like (``radius`` / ``radius_width`` /
    ``angle`` / ``angle_width`` / ``is_ring`` parallel arrays), or
    None/empty for an all-False result.

    IoU-style overlap criterion: each reflection is treated as the
    detected box an injection would create for it (the ``seed_*``
    widths — the same numbers ``build_injection_specs`` uses), and it
    is explained when, against any box, the shared span on EVERY
    relevant axis is at least ``min_overlap`` of the SMALLER of the
    two extents (per-axis intersection-over-minimum).

    Why intersection-over-minimum instead of plain IoU: fitted rows
    carry pygidfit's tight 2σ widths around the peak's REAL position
    (systematically offset from the CIF prediction), so a freshly
    injected and matched peak is a tiny box inside the seed-sized
    injection box — near-zero on area IoU, but 1.0 on IoM. And a fit
    confined to its injection box sits at most seed/2 off the
    prediction, which bounds the per-axis IoM at exactly 0.5 — the
    default threshold — so peaks this feature itself created always
    classify as covering their reflection, while genuinely distinct
    neighbouring reflections stay uncovered.

    Spot reflections are explained by spot boxes (both axes) or ring
    boxes (radial axis only); ring reflections only by ring boxes
    (radial axis — a single spot doesn't account for a full ring).
    Rows with non-finite angle_width count as ring-style regardless
    of their is_ring flag (the stored ring convention is
    angle_width=inf).
    """
    n_refl = len(reflections)
    mask = np.zeros(n_refl, dtype=bool)
    if boxes is None or n_refl == 0:
        return mask
    radius = np.asarray(getattr(boxes, "radius", ()), dtype=float)
    if radius.size == 0:
        return mask
    r_width = np.abs(np.asarray(boxes.radius_width, dtype=float))
    angle = np.asarray(boxes.angle, dtype=float)
    a_width = np.abs(np.asarray(boxes.angle_width, dtype=float))
    ring_like = (
        np.asarray(boxes.is_ring, dtype=bool)
        | ~np.isfinite(a_width)
    )
    seed_rw = abs(float(seed_radius_width))
    seed_aw = abs(float(seed_angle_width))
    box_r_lo, box_r_hi = radius - r_width / 2, radius + r_width / 2
    box_a_lo, box_a_hi = angle - a_width / 2, angle + a_width / 2

    for i, refl in enumerate(reflections):
        overlap_r = (
            np.minimum(refl.radius + seed_rw / 2, box_r_hi)
            - np.maximum(refl.radius - seed_rw / 2, box_r_lo)
        )
        need_r = min_overlap * np.minimum(seed_rw, r_width)
        ok_r = (overlap_r > 0) & (overlap_r >= need_r)
        if refl.is_ring:
            mask[i] = bool(np.any(ok_r & ring_like))
            continue
        overlap_a = (
            np.minimum(refl.angle + seed_aw / 2, box_a_hi)
            - np.maximum(refl.angle - seed_aw / 2, box_a_lo)
        )
        need_a = min_overlap * np.minimum(seed_aw, a_width)
        ok_a = (overlap_a > 0) & (overlap_a >= need_a)
        mask[i] = bool(np.any(ok_r & ((~ring_like & ok_a) | ring_like)))
    return mask


def default_box_size(fitted_table) -> tuple[float, float]:
    """(radius_width, angle_width) seed for injected boxes.

    Medians of the frame's finite, positive fitted widths; each falls
    back to the module default when the table is missing/empty or the
    column has no usable values (ring rows' inf angle_width drops out
    via the finite filter).
    """
    def _median(values, fallback: float) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.median(arr)) if arr.size else fallback

    if fitted_table is None:
        return DEFAULT_RADIUS_WIDTH, DEFAULT_ANGLE_WIDTH
    return (
        _median(getattr(fitted_table, "radius_width", ()), DEFAULT_RADIUS_WIDTH),
        _median(getattr(fitted_table, "angle_width", ()), DEFAULT_ANGLE_WIDTH),
    )


def build_injection_specs(
    pattern: SimulatedPattern, selected_indices,
    radius_width: float, angle_width: float,
) -> list[dict]:
    """Detected-box injection specs for the selected reflections.

    Spot boxes are centred on the reflection with the given seed size.
    Ring boxes use the finite quadrant-spanning geometry pygidfit needs
    (``RING_BOX_ANGLE`` / ``RING_BOX_ANGLE_WIDTH`` — the persisted
    fitted row is normalised back to angle_width=inf by the executor).
    """
    specs: list[dict] = []
    n = len(pattern.reflections)
    for idx in selected_indices:
        if not 0 <= int(idx) < n:
            continue
        refl = pattern.reflections[int(idx)]
        if refl.is_ring:
            specs.append({
                "radius": float(refl.radius),
                "radius_width": float(radius_width),
                "angle": float(RING_BOX_ANGLE),
                "angle_width": float(RING_BOX_ANGLE_WIDTH),
                "is_ring": True,
            })
        else:
            specs.append({
                "radius": float(refl.radius),
                "radius_width": float(radius_width),
                "angle": float(refl.angle),
                "angle_width": float(angle_width),
                "is_ring": False,
            })
    return specs
