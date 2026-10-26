"""Pure math of ``simulation_pattern`` (the "Expected pattern" overlay).

The CifPattern inputs are ``SimpleNamespace`` duck-types: the module
must never import mlgidmatch (its arrays are consumed pre-computed), so
these tests run identically with and without the matching backend.

Source: simulation_pattern.py (extraction/fold/dedupe/classify/specs);
ring box constants from phase_tracking.py (RING_BOX_ANGLE/-_WIDTH).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlgidlab import simulation_pattern as sp
from mlgidlab.phase_tracking import RING_BOX_ANGLE, RING_BOX_ANGLE_WIDTH


def _fake_cifpattern():
    """One CIF, two precomputed orientations + a powder pattern.

    Orientation (1,1,1) deliberately contains a mirror duplicate
    (upstream ``move_fromMW=True`` mirrors into negative q_xy), a
    zero-padding row, and a non-finite row — the extraction must fold,
    dedupe (keeping max intensity), and drop the junk.
    """
    orientations = [np.array(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32
    )]
    q2d_0 = np.array([
        [0.5, 0.5],       # merges with the mirrored row below
        [-0.5, 0.5],      # mirror duplicate, stronger intensity -> wins
        [1.0, 0.2],
        [0.0, 1.2],
        [0.0, 0.0],       # zero padding
        [np.nan, 0.3],    # non-finite
    ])
    int_0 = np.array([10.0, 40.0, 100.0, 5.0, 50.0, 3.0])
    q2d_1 = np.array([[0.3, 0.1]])
    int_1 = np.array([7.0])
    return SimpleNamespace(
        cifs=["In2O3.cif"],
        pattern_3d=SimpleNamespace(orientations=orientations),
        all_patterns_q2d=[[q2d_0, q2d_1]],
        all_patterns_int2d=[[int_0, int_1]],
        all_patterns_q1d=[np.array([0.8, 0.4, 1.6])],
        all_patterns_int1d=[np.array([2.0, 8.0, 4.0])],
    )


def test_cif_index_case_insensitive_extensionless():
    fp = _fake_cifpattern()
    assert sp.cif_index(fp, "in2o3") == 0
    assert sp.cif_index(fp, "IN2O3.CIF") == 0
    assert sp.cif_index(fp, "/some/dir/In2O3.cif") == 0
    assert sp.cif_index(fp, "PbI2") is None


def test_list_orientations_rounds_float32():
    fp = _fake_cifpattern()
    assert sp.list_orientations(fp, 0) == [(1, 1, 1), (0, 0, 2)]


def test_orientation_index_tolerance_and_miss():
    fp = _fake_cifpattern()
    assert sp.orientation_index(fp, 0, (1, 1, 1)) == 0
    assert sp.orientation_index(fp, 0, (0, 0, 2)) == 1
    assert sp.orientation_index(fp, 0, (5, 5, 5)) is None
    # Float32 storage noise stays within the +-0.5 window.
    fp.pattern_3d.orientations[0][0] = np.array(
        [0.9999999, 1.0000001, 1.0], dtype=np.float32
    )
    assert sp.orientation_index(fp, 0, (1, 1, 1)) == 0


def test_parse_hkl_forms_and_errors():
    """User-typed orientations: three integers with space/comma/semi
    separators (signs fine); anything else raises with a readable
    message, and the direction-less 0 0 0 points at the powder mode."""
    assert sp.parse_hkl("0 0 1") == (0, 0, 1)
    assert sp.parse_hkl("-1,1, 0") == (-1, 1, 0)
    assert sp.parse_hkl(" 2;2;0 ") == (2, 2, 0)
    for bad in ("", "1 1", "1 1 1 1", "a b c", "1.5 0 0"):
        with pytest.raises(ValueError):
            sp.parse_hkl(bad)
    with pytest.raises(ValueError, match="random \\(powder\\) mode"):
        sp.parse_hkl("0 0 0")


def test_parse_hkl_accepts_digits_written_together():
    """Indices are usually printed without spaces (010, -110). Three
    single digits with optional signs parse the same; every form with
    separators parses exactly as before."""
    assert sp.parse_hkl("010") == (0, 1, 0)
    assert sp.parse_hkl("-110") == (-1, 1, 0)
    assert sp.parse_hkl("1-10") == (1, -1, 0)
    assert sp.parse_hkl(" +1-1-1 ") == (1, -1, -1)
    # Unchanged: separators, and multi-digit indices.
    assert sp.parse_hkl("0 1 0") == (0, 1, 0)
    assert sp.parse_hkl("-1,1,0") == (-1, 1, 0)
    assert sp.parse_hkl("2;2;0") == (2, 2, 0)
    assert sp.parse_hkl("10 0 0") == (10, 0, 0)
    with pytest.raises(ValueError, match="random \\(powder\\) mode"):
        sp.parse_hkl("000")
    for bad in ("0001", "10", "1-1", "01a", "--110", "1 10"):
        with pytest.raises(ValueError):
            sp.parse_hkl(bad)
    with pytest.raises(ValueError, match="e.g. 001"):
        sp.parse_hkl("1111")


def test_resolve_orientation_direction_equivalents():
    """Spellings differing by a common factor or an overall sign are
    the same texture direction: both sides compare gcd-reduced, so
    0 0 1 / 0 0 -1 / 0 0 4 all resolve to the stored (0, 0, 2), and
    unknown directions return None."""
    fp = _fake_cifpattern()          # orientations (1,1,1), (0,0,2)
    assert sp.resolve_orientation(fp, 0, (1, 1, 1)) == (1, 1, 1)
    assert sp.resolve_orientation(fp, 0, (2, 2, 2)) == (1, 1, 1)
    assert sp.resolve_orientation(fp, 0, (-1, -1, -1)) == (1, 1, 1)
    assert sp.resolve_orientation(fp, 0, (0, 0, 1)) == (0, 0, 2)
    assert sp.resolve_orientation(fp, 0, (0, 0, -4)) == (0, 0, 2)
    assert sp.resolve_orientation(fp, 0, (0, 1, 0)) is None
    assert sp.resolve_orientation(fp, 1, (1, 1, 1)) is None


def test_extract_oriented_folds_dedupes_and_normalises():
    fp = _fake_cifpattern()
    pattern = sp.extract_pattern(fp, 0, (1, 1, 1))
    assert pattern.cif == "in2o3"
    assert pattern.hkl == (1, 1, 1)
    assert not pattern.is_powder
    refl = pattern.reflections
    # 6 raw rows -> padding + nan dropped, mirror pair merged -> 3.
    assert len(refl) == 3
    # Sorted by radius; indices are their list positions.
    assert [r.index for r in refl] == [0, 1, 2]
    radii = [r.radius for r in refl]
    assert radii == sorted(radii)
    assert radii[0] == pytest.approx(np.hypot(0.5, 0.5))
    # The folded duplicate kept the STRONGER intensity (40, not 10).
    assert refl[0].q_xy == pytest.approx(0.5)
    assert refl[0].rel_intensity == pytest.approx(0.4)
    assert refl[0].angle == pytest.approx(45.0)
    # Normalisation against the pattern max (100).
    by_rel = {round(r.rel_intensity, 3) for r in refl}
    assert by_rel == {0.4, 1.0, 0.05}
    # Folded coordinates never leave the first quadrant.
    assert all(r.q_xy >= 0 for r in refl)
    assert all(not r.is_ring for r in refl)
    # The (0, 1.2) reflection sits on the q_z axis: angle 90.
    assert refl[-1].angle == pytest.approx(90.0)


def test_extract_powder_rings_sorted():
    fp = _fake_cifpattern()
    for hkl in (None, (0, 0, 0)):
        pattern = sp.extract_pattern(fp, 0, hkl)
        assert pattern.is_powder
        assert pattern.hkl == sp.POWDER_HKL
        refl = pattern.reflections
        assert [r.radius for r in refl] == pytest.approx([0.4, 0.8, 1.6])
        assert [r.rel_intensity for r in refl] == pytest.approx(
            [1.0, 0.25, 0.5]
        )
        assert all(r.is_ring for r in refl)
        assert all(r.angle == pytest.approx(45.0) for r in refl)


def test_extract_missing_orientation_raises():
    fp = _fake_cifpattern()
    with pytest.raises(ValueError, match="in2o3"):
        sp.extract_pattern(fp, 0, (3, 2, 1))


def test_extract_powder_not_precomputed_raises():
    fp = _fake_cifpattern()
    fp.all_patterns_q1d = None
    with pytest.raises(ValueError, match="re-parse"):
        sp.extract_pattern(fp, 0, None)


def test_extract_all_nonpositive_intensity_gives_empty():
    fp = _fake_cifpattern()
    fp.all_patterns_int2d[0][1] = np.array([0.0])
    assert sp.extract_pattern(fp, 0, (0, 0, 2)).reflections == []


def _boxes(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "radius": np.zeros(n), "radius_width": np.full(n, 0.2),
        "angle": np.zeros(n), "angle_width": np.full(n, 10.0),
        "is_ring": np.zeros(n, dtype=bool),
    }
    base.update({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    base["is_ring"] = np.asarray(base["is_ring"], dtype=bool)
    return SimpleNamespace(**base)


def _spot(radius, angle, index=0):
    q_xy, q_z = radius * np.cos(np.deg2rad(angle)), radius * np.sin(
        np.deg2rad(angle)
    )
    return sp.SimulatedReflection(
        index=index, q_xy=float(q_xy), q_z=float(q_z),
        radius=float(radius), angle=float(angle),
        rel_intensity=1.0, is_ring=False,
    )


def _ring(radius, index=0):
    return sp.SimulatedReflection(
        index=index, q_xy=float(radius), q_z=0.0, radius=float(radius),
        angle=45.0, rel_intensity=1.0, is_ring=True,
    )


def test_classify_spot_against_spot_boxes():
    # Widths chosen exact in binary floating point so the boundary
    # case (|delta| == width/2) is genuinely on the boundary.
    boxes = _boxes(
        radius=[1.0], radius_width=[0.5], angle=[30.0], angle_width=[10.0],
    )
    inside = _spot(1.05, 33.0)
    boundary = _spot(1.25, 35.0)  # exactly width/2 on both axes
    off_radius = _spot(1.3, 30.0)
    off_angle = _spot(1.0, 40.0)
    mask = sp.classify_explained(
        [inside, boundary, off_radius, off_angle], boxes
    )
    assert mask.tolist() == [True, True, False, False]


def test_classify_spot_explained_by_ring_box_radially():
    boxes = _boxes(
        radius=[1.0], radius_width=[0.2], angle=[45.0],
        angle_width=[np.inf], is_ring=[True],
    )
    assert sp.classify_explained([_spot(1.05, 80.0)], boxes).tolist() == [True]
    assert sp.classify_explained([_spot(1.3, 45.0)], boxes).tolist() == [False]


def test_classify_ring_needs_ring_box():
    spot_boxes = _boxes(
        radius=[1.0], radius_width=[0.2], angle=[45.0], angle_width=[10.0],
    )
    ring_boxes = _boxes(
        radius=[1.0], radius_width=[0.2], angle=[45.0],
        angle_width=[np.inf], is_ring=[True],
    )
    # A spot box at the same radius does NOT explain a full ring.
    assert sp.classify_explained([_ring(1.0)], spot_boxes).tolist() == [False]
    assert sp.classify_explained([_ring(1.0)], ring_boxes).tolist() == [True]


def test_classify_inf_angle_width_counts_as_ring_style():
    # is_ring False but angle_width inf — the stored ring convention.
    boxes = _boxes(
        radius=[1.0], radius_width=[0.2], angle=[45.0],
        angle_width=[np.inf], is_ring=[False],
    )
    assert sp.classify_explained([_ring(1.0)], boxes).tolist() == [True]
    # And it must not act as a spot box for angle containment.
    assert sp.classify_explained([_spot(1.0, 10.0)], boxes).tolist() == [True]


def test_classify_iom_recognizes_tight_fit_near_prediction():
    """Post-add scenario: a freshly fitted peak carries a tight 2σ box
    whose centre sits slightly off the CIF prediction (real lattice vs
    nominal). It lies inside the would-be injection box, so the
    per-axis intersection-over-minimum criterion covers the
    reflection — the old centre-in-box rule missed it."""
    boxes = _boxes(
        radius=[0.72], radius_width=[0.04], angle=[46.0], angle_width=[1.0],
    )
    refl = _spot(np.hypot(0.5, 0.5), 45.0)
    assert sp.classify_explained(
        [refl], boxes, seed_radius_width=0.3, seed_angle_width=8.0,
    ).tolist() == [True]
    # Sanity: with a near-point seed the angular spans no longer
    # overlap and the reflection stays uncovered.
    assert sp.classify_explained(
        [refl], boxes, seed_radius_width=0.001, seed_angle_width=0.1,
    ).tolist() == [False]


def test_classify_iom_worst_case_own_injection():
    """A fit confined to its injection box sits at most seed/2 off the
    prediction — per-axis IoM exactly 0.5, the threshold — so peaks
    this feature created always classify as covering (binary-exact
    numbers so the boundary is genuinely on the boundary)."""
    boxes = _boxes(
        radius=[1.25], radius_width=[0.5], angle=[49.0], angle_width=[8.0],
    )
    refl = _spot(1.0, 45.0)
    assert sp.classify_explained(
        [refl], boxes, seed_radius_width=0.5, seed_angle_width=8.0,
    ).tolist() == [True]


def test_classify_iom_adjacent_reflection_not_covered():
    boxes = _boxes(
        radius=[1.0], radius_width=[0.3], angle=[45.0], angle_width=[8.0],
    )
    far = _spot(1.4, 45.0)
    assert sp.classify_explained(
        [far], boxes, seed_radius_width=0.3, seed_angle_width=8.0,
    ).tolist() == [False]


def test_classify_empty_inputs():
    assert sp.classify_explained([], None).tolist() == []
    assert sp.classify_explained([_spot(1.0, 45.0)], None).tolist() == [False]
    empty = _boxes(radius=[])
    assert sp.classify_explained([_spot(1.0, 45.0)], empty).tolist() == [False]


def test_default_box_size_median_and_fallback():
    table = SimpleNamespace(
        radius_width=np.array([0.1, 0.2, 0.3]),
        angle_width=np.array([4.0, 6.0, np.inf]),
    )
    rw, aw = sp.default_box_size(table)
    assert rw == pytest.approx(0.2)
    assert aw == pytest.approx(5.0)  # inf filtered, median of [4, 6]
    assert sp.default_box_size(None) == (
        sp.DEFAULT_RADIUS_WIDTH, sp.DEFAULT_ANGLE_WIDTH
    )
    empty = SimpleNamespace(
        radius_width=np.array([]), angle_width=np.array([np.inf])
    )
    assert sp.default_box_size(empty) == (
        sp.DEFAULT_RADIUS_WIDTH, sp.DEFAULT_ANGLE_WIDTH
    )


def test_build_injection_specs_spot_and_ring():
    pattern = sp.SimulatedPattern(
        cif="x", hkl=(1, 0, 0),
        reflections=[_spot(1.0, 30.0, index=0), _ring(1.5, index=1)],
    )
    specs = sp.build_injection_specs(pattern, [0, 1, 7], 0.15, 6.0)
    assert len(specs) == 2  # out-of-range index 7 skipped
    spot_spec, ring_spec = specs
    assert spot_spec == {
        "radius": 1.0, "radius_width": 0.15, "angle": 30.0,
        "angle_width": 6.0, "is_ring": False,
    }
    assert ring_spec["is_ring"] is True
    assert ring_spec["radius"] == pytest.approx(1.5)
    assert ring_spec["radius_width"] == pytest.approx(0.15)
    # Rings use the finite quadrant-spanning fit-box geometry.
    assert ring_spec["angle"] == pytest.approx(RING_BOX_ANGLE)
    assert ring_spec["angle_width"] == pytest.approx(RING_BOX_ANGLE_WIDTH)


def test_specs_to_boxes_roundtrip_classifies():
    """Spec dicts become a classify_explained-compatible box table —
    the sweep planner counts boxes it has already queued as coverage,
    so overlapping predictions from different patterns dedupe."""
    pattern = sp.SimulatedPattern(
        cif="x", hkl=(1, 0, 0),
        reflections=[_spot(1.0, 30.0, index=0), _ring(1.5, index=1)],
    )
    specs = sp.build_injection_specs(pattern, [0, 1], 0.15, 6.0)
    boxes = sp.specs_to_boxes(specs)
    assert boxes.radius.tolist() == [1.0, 1.5]
    assert boxes.is_ring.tolist() == [False, True]
    # The queued spot box covers its own reflection; the queued ring
    # box covers a ring reflection at its radius (is_ring flag makes
    # it ring-style despite the finite fit-box angle_width).
    mask = sp.classify_explained(
        [_spot(1.0, 30.0), _ring(1.5, index=1), _spot(2.5, 30.0, index=2)],
        boxes, seed_radius_width=0.15, seed_angle_width=6.0,
    )
    assert mask.tolist() == [True, True, False]
    assert sp.specs_to_boxes([]) is None
