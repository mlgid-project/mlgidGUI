"""Conversion panel: orientation flips, value-driven overrides, PONI
autofill, and the append-frames output mode.

The flips moved OUT of the Manual-overrides subsection (routine
per-beamline settings, always honoured when checked); the remaining
override fields are value-driven — "(unset)" fields are skipped, set
fields are forwarded, no enable-checkbox gate. Loading a PONI pre-fills
the override fields with the values pygid would derive from it
(``parse_poni_overrides`` mirrors ``pygid.ExpParams`` math). Append
mode extends an existing entry instead of creating ``entry_NNNN``.
Source: conversion_panel.py ``_build_exp_params_section`` /
``_collect_config`` / ``parse_poni_overrides`` /
``_autofill_overrides_from_poni`` / ``_on_append_frames_toggled`` /
``_refresh_append_entries``; conversion.py ``_validate_append_target``.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import pytest

from mlgidlab import conversion
from mlgidlab.conversion_panel import (
    OUTPUT_SEPARATE_DATASETS,
    ConversionConfig,
    parse_poni_overrides,
)

pytestmark = pytest.mark.gui


PONI_V2 = """\
# Calibration converted at ...
poni_version: 2
Detector: Detector
Detector_config: {{"pixel1": 7.5e-05, "pixel2": 7.5e-05, "max_shape": [1679, 1475]}}
Distance: {dist}
Poni1: {poni1}
Poni2: {poni2}
Rot1: {rot1}
Rot2: {rot2}
Rot3: 0.0
Wavelength: {wl}
"""


def _write_poni(tmp_path, **kw) -> Path:
    defaults = dict(dist=0.2871, poni1=0.084, poni2=0.0405, rot1=0.0, rot2=0.0, wl=9.6e-11)
    defaults.update(kw)
    path = tmp_path / "test.poni"
    path.write_text(PONI_V2.format(**defaults))
    return path


# -- parse_poni_overrides ----------------------------------------------


def test_parse_poni_no_rotation(tmp_path):
    vals = parse_poni_overrides(_write_poni(tmp_path))
    assert vals["SDD"] == pytest.approx(0.2871)
    assert vals["wavelength"] == pytest.approx(0.96)  # m → Å
    # rot = 0 → centers are just poni/pixel.
    assert vals["centerX"] == pytest.approx(0.0405 / 7.5e-05)
    assert vals["centerY"] == pytest.approx(0.084 / 7.5e-05)


def test_parse_poni_rotation_matches_pygid_formula(tmp_path):
    rot1, rot2 = 0.01, -0.02
    vals = parse_poni_overrides(_write_poni(tmp_path, rot1=rot1, rot2=rot2))
    sdd, px = 0.2871, 7.5e-05
    assert vals["centerX"] == pytest.approx((-sdd * math.tan(rot1) + 0.0405) / px)
    assert vals["centerY"] == pytest.approx(
        (sdd * math.tan(rot2) / math.cos(rot1) + 0.084) / px
    )


def test_parse_poni_without_pixel_size_yields_sdd_and_wavelength(tmp_path):
    path = tmp_path / "nopx.poni"
    path.write_text("Distance: 0.5\nWavelength: 1e-10\nPoni1: 0.1\nPoni2: 0.1\n")
    vals = parse_poni_overrides(path)
    assert set(vals) == {"SDD", "wavelength"}


# -- panel: flips + value-driven overrides + autofill -------------------


def test_flips_collected_without_override_fields(main_window):
    panel = main_window.conversion_panel
    panel.flip_lr.setChecked(True)
    panel.flip_ud.setChecked(True)

    cfg = panel._collect_config()

    assert cfg.expmeta_overrides == {"fliplr": True, "flipud": True}
    # The old enable-gate is gone — no checkable box to forget.
    assert not hasattr(panel, "_override_box")


def test_overrides_are_value_driven(main_window):
    panel = main_window.conversion_panel
    panel.over_centerX.setValue(737.5)
    # Transpose lives in the Orientation row now, not under Manual
    # overrides, but it still travels in the same dict.
    panel.transp.setChecked(True)

    cfg = panel._collect_config()

    assert cfg.expmeta_overrides == {"centerX": 737.5, "transp": True}
    # Unset numeric fields stay absent.
    assert "SDD" not in cfg.expmeta_overrides


def test_poni_autofill_fills_override_fields(main_window, tmp_path):
    panel = main_window.conversion_panel
    poni = _write_poni(tmp_path)
    panel.poni_path.setText(str(poni))

    panel._autofill_overrides_from_poni()

    assert panel.over_SDD.value() == pytest.approx(0.2871)
    assert panel.over_wavelength.value() == pytest.approx(0.96)
    assert panel.over_centerX.value() == pytest.approx(540.0)
    assert panel.over_centerY.value() == pytest.approx(1120.0)


# -- panel: append-frames UI -------------------------------------------


def _existing_output(tmp_path, n_entries=2) -> Path:
    out = tmp_path / "converted.h5"
    with h5py.File(out, "w", track_order=True) as f:
        for i in range(n_entries):
            f.create_group(f"entry_{i:04d}/data")
    return out


def test_append_toggle_locks_overwrite_and_lists_entries(main_window, tmp_path):
    panel = main_window.conversion_panel
    _existing_output(tmp_path)
    panel.output_dir.setText(str(tmp_path))
    panel.output_mode_combo.setCurrentText(OUTPUT_SEPARATE_DATASETS)

    panel.append_frames_chk.setChecked(True)

    assert not panel.overwrite_file_chk.isEnabled()
    assert not panel.overwrite_file_chk.isChecked()
    assert not panel.overwrite_dataset_chk.isEnabled()
    items = [panel.append_entry_combo.itemText(i)
             for i in range(panel.append_entry_combo.count())]
    assert items == ["entry_0000", "entry_0001"]
    assert panel.append_entry_combo.currentText() == "entry_0001"  # last

    panel.append_frames_chk.setChecked(False)
    assert panel.overwrite_file_chk.isEnabled()


def test_append_config_collection_and_missing_entry_error(main_window, tmp_path):
    panel = main_window.conversion_panel
    _existing_output(tmp_path)
    panel.output_dir.setText(str(tmp_path))
    panel.output_mode_combo.setCurrentText(OUTPUT_SEPARATE_DATASETS)
    panel.append_frames_chk.setChecked(True)

    cfg = panel._collect_config()
    assert cfg.append_frames is True
    assert cfg.append_entry == "entry_0001"
    assert cfg.overwrite_file is False and cfg.overwrite_dataset is False

    # No target entry (empty combo) → collection refuses loudly.
    panel.append_entry_combo.clear()
    with pytest.raises(ValueError):
        panel._collect_config()


# -- engine: append-target validation (pure h5py, no pygid) -------------


def test_validate_append_target(tmp_path):
    out = _existing_output(tmp_path)
    cfg = ConversionConfig(append_frames=True, append_entry="entry_0001")

    # Happy path — single target, file + entry exist.
    conversion._validate_append_target(cfg, {Path("raw.h5"): out})

    # Multiple output files are ambiguous.
    with pytest.raises(ValueError, match="single output file"):
        conversion._validate_append_target(
            cfg, {Path("a.h5"): out, Path("b.h5"): tmp_path / "other.h5"}
        )

    # Output file missing.
    with pytest.raises(ValueError, match="does not exist"):
        conversion._validate_append_target(
            cfg, {Path("raw.h5"): tmp_path / "missing.h5"}
        )

    # Entry missing.
    cfg.append_entry = "entry_9999"
    with pytest.raises(ValueError, match="not found"):
        conversion._validate_append_target(cfg, {Path("raw.h5"): out})


# -- engine: single-entry output mode ------------------------------------


class _RecordingPygid:
    """Minimal pygid stand-in recording each conversion call's routing
    kwargs — lets the single-entry group/overwrite logic run in CI
    without the pipeline extra installed."""

    calls: list[dict]

    class ExpParams:
        def __init__(self, **kw):
            pass

    class CoordMaps:
        def __init__(self, params, **kw):
            pass

    class SampleMetadata:
        def __init__(self, **kw):
            pass

    class ExpMetadata:
        def __init__(self, **kw):
            pass

    def __init__(self):
        _RecordingPygid.calls = []
        outer = self

        class Conversion:
            def __init__(self, matrix=None, path=None, dataset=None,
                         frame_num=None, global_frame_offset=0):
                self._path = path
                self._offset = global_frame_offset

            def det2q_gid(self, **kwargs):
                outer.calls.append({
                    "path": self._path,
                    "global_frame_offset": self._offset,
                    **kwargs,
                })

        self.Conversion = Conversion


def test_single_entry_mode_routes_all_scans_into_one_group(
    tmp_path, monkeypatch
):
    """Every scan targets the SAME fresh entry; only the first call may
    overwrite (a later overwrite would truncate the growing stack)."""
    import sys

    from mlgidlab.conversion_panel import OUTPUT_SINGLE_ENTRY, RawScan

    fake = _RecordingPygid()
    monkeypatch.setitem(sys.modules, "pygid", fake)
    poni = _write_poni(tmp_path)
    scans = [
        RawScan(file_path=tmp_path / f"i{i}.tif", entry="", frame_num=None)
        for i in range(3)
    ]
    cfg = ConversionConfig(
        poni_path=poni,
        ai=0.1,
        output_mode=OUTPUT_SINGLE_ENTRY,
        output_dir=tmp_path,
        overwrite_file=True,
        overwrite_dataset=True,
    )
    written = conversion.execute(scans, cfg)

    assert written == [tmp_path / "converted.h5"]
    calls = _RecordingPygid.calls
    assert len(calls) == 3
    assert {c["h5_group"] for c in calls} == {"entry_0000"}
    assert [c["overwrite_file"] for c in calls] == [True, False, False]
    assert [c["overwrite_group"] for c in calls] == [True, False, False]

    # Append + single-entry are mutually exclusive.
    cfg.append_frames = True
    cfg.append_entry = "entry_0000"
    with pytest.raises(ValueError, match="mutually exclusive"):
        conversion.execute(scans, cfg)


def _fake_entries(panel, *entries):
    """Pin the panel's ticked entries without building a selection tree.

    The tree plumbing itself is covered in ``test_fabio_gui``; what is
    under test here is the frame axis those entries add up to.
    """
    panel._checked_entries = lambda: list(entries)


def _raw_entry(tmp_path, name, n_frames, fabio=False):
    from mlgidlab.file_model import RawEntry

    path = tmp_path / name
    return RawEntry(
        file_path=path,
        dataset_path="" if fabio else "raw/data",
        shape=(n_frames, 8, 8),
        dtype="uint32",
        frame_map=(
            [(tmp_path / f"{name}_{i}.tif", 0) for i in range(n_frames)]
            if fabio
            else None
        ),
    )


def test_the_angle_field_takes_a_list_and_a_ramp(main_window, tmp_path):
    """The frame axis decides how three numbers are read, so it has to
    be the ticked entries' frame count, not the tree's row count."""
    panel = main_window.conversion_panel
    _fake_entries(panel, _raw_entry(tmp_path, "stack.h5", 14))
    assert panel._selection_frame_axis() == 14

    panel.ai_input.setText("0.2")
    assert panel._collect_config().ai == pytest.approx(0.2)

    panel.ai_input.setText("(0.1, 1.5, 13)")
    ai = panel._collect_config().ai
    assert isinstance(ai, list) and len(ai) == 14

    panel.ai_input.setText("")
    assert panel._collect_config().ai is None


def test_the_angle_hint_names_the_resolved_count(main_window, tmp_path):
    """The ramp gives one MORE angle than typed; the hint is the only
    place that shows it before the conversion has already run."""
    panel = main_window.conversion_panel
    _fake_entries(panel, _raw_entry(tmp_path, "stack.h5", 14))

    panel.ai_input.setText("(0.1, 1.5, 13)")
    panel._refresh_ai_hint()
    assert panel.ai_hint.text().startswith("14 angles")
    assert panel.ai_hint.property("status") == "muted"

    # The off-by-one is caught while typing, not at Convert.
    panel.ai_input.setText("(0.1, 1.5, 14)")
    panel._refresh_ai_hint()
    assert "15 angles" in panel.ai_hint.text()
    assert "14 frames are selected" in panel.ai_hint.text()
    assert panel.ai_hint.property("status") == "error"

    panel.ai_input.setText("0.1, nonsense")
    panel._refresh_ai_hint()
    assert "Not a number" in panel.ai_hint.text()
    assert panel.ai_hint.property("status") == "error"

    panel.ai_input.setText("")
    panel._refresh_ai_hint()
    assert panel.ai_hint.text() == ""


def test_a_wrong_length_angle_list_refuses_to_start(main_window, tmp_path):
    panel = main_window.conversion_panel
    entry = _raw_entry(tmp_path, "stack.h5", 4, fabio=True)
    _fake_entries(panel, entry)
    panel.frame_mode.setCurrentText("All")
    panel.ai_input.setText("0.1, 0.2, 0.3, 0.4, 0.5")

    with pytest.raises(ValueError, match="5 angles.*4 frames"):
        panel._collect_run_inputs()


def test_too_few_angles_explains_the_ramp_reading(main_window, tmp_path):
    """Three angles for four frames is read as a ramp -- say so.

    Otherwise the user gets "step count must be a whole number" for the
    perfectly reasonable input ``0.1, 0.2, 0.3``.
    """
    panel = main_window.conversion_panel
    _fake_entries(panel, _raw_entry(tmp_path, "stack.h5", 4, fabio=True))
    panel.ai_input.setText("0.1, 0.2, 0.3")
    panel._refresh_ai_hint()
    hint = panel.ai_hint.text()
    assert "read as a ramp" in hint
    assert "4 are selected" in hint


def test_each_scan_says_which_frame_of_the_selection_it_is(
    main_window, tmp_path
):
    """``frame_offset`` is what makes a batch of single-frame scans
    per-frame correct: pygid reads ``ai[frame + offset]``."""
    panel = main_window.conversion_panel
    _fake_entries(
        panel,
        _raw_entry(tmp_path, "a.h5", 3, fabio=True),
        _raw_entry(tmp_path, "b.h5", 2, fabio=True),
    )
    panel.frame_mode.setCurrentText("All")
    assert panel._selection_frame_axis() == 5
    assert [s.frame_offset for s in panel._collect_scans()] == [0, 1, 2, 3, 4]

    # An HDF5 entry is ONE scan covering its whole span, so the next
    # entry starts past it.
    _fake_entries(
        panel,
        _raw_entry(tmp_path, "big.h5", 6),
        _raw_entry(tmp_path, "c.h5", 2, fabio=True),
    )
    assert [s.frame_offset for s in panel._collect_scans()] == [0, 6, 7]


def test_a_frame_subset_keeps_its_original_angle_indices(tmp_path):
    """Converting frames 3 and 7 must read angles 3 and 7, not 0 and 1.

    That is why the array is one value per SOURCE frame: pygid indexes
    by the frame's place in the data, not in the selection.
    """
    from mlgidlab.conversion_config import _expand_fabio_scans

    entry = _raw_entry(tmp_path, "stack", 14, fabio=True)
    scans = _expand_fabio_scans(entry, [3, 7])
    assert [s.frame_offset for s in scans] == [3, 7]
    # ...and a second entry's frames continue past the first's.
    assert [s.frame_offset for s in _expand_fabio_scans(entry, [0, 1], 14)] == [
        14, 15,
    ]


def test_per_frame_ai_end_to_end_with_pygid(tmp_path):
    """Real pygid run: 3 TIFFs, 3 angles, one per frame, IN ORDER.

    This is the load-bearing assertion of the whole feature. pygid looks
    an angle up as ``ai[frame + global_frame_offset]``, and every one of
    these scans is its own single-frame Conversion, so without
    ``RawScan.frame_offset`` reaching pygid all three frames would be
    recorded (and remapped) at the first angle. Reading
    ``angle_of_incidence`` back out of the written file is the only way
    to see the difference.
    """
    pytest.importorskip("pygid")
    import fabio.tifimage
    import numpy as np

    from mlgidlab.conversion_panel import (
        CONV_DET2Q_GID,
        OUTPUT_SINGLE_ENTRY,
        RawScan,
    )

    # THE SAME image three times, so any difference between converted
    # frames can only come from the angle each was converted at.
    rng = np.random.default_rng(0)
    img = (rng.random((220, 180)) * 1000).astype(np.uint32)
    tifs = []
    for i in range(3):
        p = tmp_path / f"img_{i}.tif"
        fabio.tifimage.TifImage(data=img).write(str(p))
        tifs.append(p)
    poni = tmp_path / "small.poni"
    poni.write_text(PONI_V2.format(
        dist=0.2871, poni1=0.0084, poni2=0.00675, rot1=0.0, rot2=0.0,
        wl=9.6e-11,
    ).replace("[1679, 1475]", "[220, 180]"))

    def convert(ai, name):
        cfg = ConversionConfig(
            conv_type=CONV_DET2Q_GID, poni_path=poni, ai=ai,
            output_mode=OUTPUT_SINGLE_ENTRY, output_dir=tmp_path,
            output_filename=name, overwrite_file=True,
            expmeta_overrides={"px_size": 7.5e-05},
        )
        scans = [
            RawScan(file_path=p, entry="", frame_num=None, frame_offset=i)
            for i, p in enumerate(tifs)
        ]
        written = conversion.execute(scans, cfg)
        with h5py.File(written[0], "r") as f:
            return (
                np.asarray(
                    f["entry_0000/instrument/angle_of_incidence"][()]
                ),
                np.asarray(f["entry_0000/data/img_gid_q"][()]),
            )

    angles, stack = convert([0.1, 0.2, 0.3], "ramp.h5")
    assert angles == pytest.approx([0.1, 0.2, 0.3])
    # The recorded angle and the coordinate map are looked up through
    # the same index, but only this shows the angle reached the REMAP:
    # identical input, different incidence, so the missing wedge (where
    # the detector maps nothing) sits differently.
    assert not np.array_equal(np.isnan(stack[0]), np.isnan(stack[2]))

    # A single angle still repeats, and then every frame is identical.
    flat_angles, flat_stack = convert(0.12, "flat.h5")
    assert flat_angles == pytest.approx([0.12, 0.12, 0.12])
    assert np.array_equal(np.isnan(flat_stack[0]), np.isnan(flat_stack[2]))
    np.testing.assert_allclose(
        np.nan_to_num(flat_stack[0]), np.nan_to_num(flat_stack[2]),
    )


def test_per_frame_ai_refuses_a_scan_pygid_would_batch(tmp_path):
    """Above 32 frames pygid re-indexes the angle list and walks off it.

    Its batch loop resets ``global_frame_offset`` and recomputes it per
    batch while ``_build_ai_list`` still adds the frame index on top, so
    ``ai[frame + offset]`` double-counts. Refused up front with a
    message that names the limit, rather than surfacing as an IndexError
    from inside pygid.
    """
    from mlgidlab.conversion_panel import CONV_DET2Q_GID, RawScan

    poni = tmp_path / "p.poni"
    poni.write_text("x")
    cfg = ConversionConfig(
        conv_type=CONV_DET2Q_GID, poni_path=poni,
        ai=[0.1] * 40, output_dir=tmp_path,
    )
    scans = [RawScan(file_path=tmp_path / "big.h5", entry="e", frame_num=None)]
    with pytest.raises(ValueError, match="only works up to 32 frames"):
        conversion.execute(scans, cfg)

    # Split into two scans of 20, the same 40 angles are fine (the
    # refusal is per scan, which is where pygid batches).
    scans = [
        RawScan(file_path=tmp_path / "a.h5", entry="e", frame_offset=0),
        RawScan(file_path=tmp_path / "b.h5", entry="e", frame_offset=20),
    ]
    conversion._check_per_frame_ai(cfg.ai, scans)  # must not raise


def test_per_frame_ai_refuses_a_scan_that_reaches_past_the_list(tmp_path):
    from mlgidlab.conversion_panel import RawScan

    scans = [
        RawScan(file_path=tmp_path / "a.h5", entry="e", frame_offset=0),
        RawScan(file_path=tmp_path / "b.h5", entry="e", frame_offset=3),
    ]
    with pytest.raises(ValueError, match="only 4 angles were given"):
        conversion._check_per_frame_ai([0.1, 0.2, 0.3, 0.4], scans + [
            RawScan(file_path=tmp_path / "c.h5", entry="e", frame_num=[9],
                    frame_offset=4),
        ])


def test_single_entry_mode_end_to_end_with_pygid(tmp_path):
    """Real pygid run: 3 TIFFs land as one 3-frame entry with per-frame
    frame_ind / angle_of_incidence / analysis groups, and an overwrite
    re-run replaces the stack instead of growing it."""
    pytest.importorskip("pygid")
    import fabio.tifimage
    import numpy as np

    from mlgidlab.conversion_panel import (
        CONV_DET2Q_GID,
        OUTPUT_SINGLE_ENTRY,
        RawScan,
    )

    rng = np.random.default_rng(0)
    tifs = []
    for i in range(3):
        p = tmp_path / f"img_{i}.tif"
        fabio.tifimage.TifImage(
            data=(rng.random((220, 180)) * 1000).astype(np.uint32)
        ).write(str(p))
        tifs.append(p)
    poni = tmp_path / "small.poni"
    poni.write_text(PONI_V2.format(
        dist=0.2871, poni1=0.0084, poni2=0.00675, rot1=0.0, rot2=0.0,
        wl=9.6e-11,
    ).replace("[1679, 1475]", "[220, 180]"))

    cfg = ConversionConfig(
        conv_type=CONV_DET2Q_GID,
        poni_path=poni,
        ai=0.12,
        output_mode=OUTPUT_SINGLE_ENTRY,
        output_dir=tmp_path,
        output_filename="single.h5",
        overwrite_file=True,
        # The fixture poni carries the generic "Detector" name, which
        # pygid cannot map to a pixel size — forward it explicitly via
        # the same override channel the panel uses.
        expmeta_overrides={"px_size": 7.5e-05},
    )
    scans = [RawScan(file_path=p, entry="", frame_num=None) for p in tifs]
    written = conversion.execute(scans, cfg)

    assert written == [tmp_path / "single.h5"]
    with h5py.File(written[0], "r") as f:
        entries = [k for k in f.keys() if k.startswith("entry")]
        assert entries == ["entry_0000"]
        g = f["entry_0000"]
        assert g["data/img_gid_q"].shape[0] == 3
        assert list(g["data/frame_ind"][()]) == [0.0, 1.0, 2.0]
        assert list(g["instrument/angle_of_incidence"][()]) == [0.12] * 3
        assert sorted(g["data/analysis"].keys()) == [
            "frame00000", "frame00001", "frame00002"
        ]

    # Overwrite re-run replaces (no growth to 6 frames).
    conversion.execute(scans, cfg)
    with h5py.File(written[0], "r") as f:
        assert f["entry_0000/data/img_gid_q"].shape[0] == 3
        assert [k for k in f.keys() if k.startswith("entry")] == [
            "entry_0000"
        ]


# -- the beam center must follow the image through a flip ---------------
#
# pygid applies fliplr / flipud / transp to the beam center inside
# ``ExpParams._exp_params_update_``, which takes ONE of two branches:
# poni → center, or center → poni. With poni1/poni2 *and*
# centerX/centerY both set it takes neither, so the image flipped and
# the center did not, and the missing wedge came out mirrored relative
# to the data. The PONI autofill was handing the PONI's own center back
# as an "override", which is exactly what put both of them on.


def test_unedited_autofill_values_are_not_sent_as_overrides(
    main_window, tmp_path,
):
    panel = main_window.conversion_panel
    panel.poni_path.setText(str(_write_poni(tmp_path)))
    panel._autofill_overrides_from_poni()
    panel.flip_ud.setChecked(True)

    cfg = panel._collect_config()

    # The boxes still show the readout...
    assert panel.over_centerX.value() > 0
    # ...but nothing the user did not touch travels as an override, so
    # pygid keeps the poni → center branch and flips the center.
    assert cfg.expmeta_overrides == {"flipud": True}


def test_an_edited_center_is_still_sent(main_window, tmp_path):
    panel = main_window.conversion_panel
    panel.poni_path.setText(str(_write_poni(tmp_path)))
    panel._autofill_overrides_from_poni()
    panel.over_centerX.setValue(panel.over_centerX.value() + 12.0)

    cfg = panel._collect_config()

    assert cfg.expmeta_overrides["centerX"] == pytest.approx(540.0 + 12.0)
    assert "centerY" not in cfg.expmeta_overrides  # untouched readout


def test_a_new_poni_replaces_the_autofill_snapshot(main_window, tmp_path):
    """A stale snapshot must not silence a real override on the next
    file: the same number can be an autofill for one PONI and a
    deliberate value for another."""
    panel = main_window.conversion_panel
    panel.poni_path.setText(str(_write_poni(tmp_path)))
    panel._autofill_overrides_from_poni()
    first = panel.over_centerX.value()

    other = tmp_path / "other.poni"
    other.write_text(PONI_V2.format(
        dist=0.2871, poni1=0.084, poni2=0.06, rot1=0.0, rot2=0.0, wl=9.6e-11,
    ))
    panel.poni_path.setText(str(other))
    panel._autofill_overrides_from_poni()
    panel.over_centerX.setValue(first)  # now a deliberate choice

    cfg = panel._collect_config()
    assert cfg.expmeta_overrides["centerX"] == pytest.approx(first)


def test_an_unparsable_poni_clears_the_snapshot(main_window, tmp_path):
    panel = main_window.conversion_panel
    panel.poni_path.setText(str(_write_poni(tmp_path)))
    panel._autofill_overrides_from_poni()
    kept = panel.over_centerX.value()

    junk = tmp_path / "junk.poni"
    junk.write_text("not a poni at all\n")
    panel.poni_path.setText(str(junk))
    panel._autofill_overrides_from_poni()

    cfg = panel._collect_config()
    assert cfg.expmeta_overrides["centerX"] == pytest.approx(kept)


def test_half_a_center_override_is_completed_from_the_poni(tmp_path):
    poni = _write_poni(tmp_path)
    kwargs = {"centerX": 600.0}

    conversion._complete_center_override(kwargs, poni)

    assert kwargs["centerX"] == 600.0
    assert kwargs["centerY"] == pytest.approx(0.084 / 7.5e-05)


def test_a_center_override_takes_precedence_over_the_poni():
    """``CoordMaps`` reads poni1/poni2 only, so a center override that
    leaves them in place is silently ignored — and is what stops the
    flip from reaching the center. Clearing them puts pygid on the
    center → poni branch, which does apply the flips."""
    class _Params:
        poni1 = 0.084
        poni2 = 0.0405

    p = _Params()
    conversion._prefer_center_over_poni(p, {"centerX": 1.0, "centerY": 2.0})
    assert (p.poni1, p.poni2) == (None, None)

    q = _Params()
    conversion._prefer_center_over_poni(q, {"flipud": True})
    assert (q.poni1, q.poni2) == (0.084, 0.0405)


def test_flipud_moves_the_beam_center_with_the_image(tmp_path):
    """The invariant the bug broke, checked against real pygid.

    Converting an image with ``flipud=True`` must equal converting the
    already-flipped image with ``flipud=False``: the flag says "the
    stored rows run the other way", so honouring it has to move the beam
    center too. Before the fix the second run's wedge sat on the
    opposite side of the frame.
    """
    pytest.importorskip("pygid")
    import fabio.tifimage
    import numpy as np

    from mlgidlab.conversion_panel import CONV_DET2Q_GID, OUTPUT_SINGLE_ENTRY
    from mlgidlab.conversion_config import RawScan

    rng = np.random.default_rng(3)
    img = (rng.random((220, 180)) * 1000).astype(np.uint32)
    plain = tmp_path / "plain.tif"
    flipped = tmp_path / "flipped.tif"
    fabio.tifimage.TifImage(data=img).write(str(plain))
    fabio.tifimage.TifImage(data=np.flipud(img).copy()).write(str(flipped))

    px, height = 7.5e-05, 220

    def write_poni(name: str, poni1: float) -> Path:
        path = tmp_path / name
        path.write_text(PONI_V2.format(
            dist=0.2871, poni1=poni1, poni2=0.00675, rot1=0.0, rot2=0.0,
            wl=9.6e-11,
        ).replace("[1679, 1475]", "[220, 180]"))
        return path

    # The same physical geometry described two ways: once for the stored
    # (unflipped) rows, once for the already-flipped ones. This mirror is
    # exactly what pygid must apply internally when flipud is set, and
    # what the bug skipped.
    poni_a = write_poni("a.poni", 0.0084)
    poni_b = write_poni("b.poni", (height - 1) * px - 0.0084)

    def convert(src: Path, name: str, poni: Path, flipud: bool):
        overrides = {"px_size": 7.5e-05}
        if flipud:
            overrides["flipud"] = True
        cfg = ConversionConfig(
            conv_type=CONV_DET2Q_GID, poni_path=poni, ai=0.12,
            output_mode=OUTPUT_SINGLE_ENTRY, output_dir=tmp_path,
            output_filename=name, overwrite_file=True,
            expmeta_overrides=overrides,
        )
        out = conversion.execute(
            [RawScan(file_path=src, entry="", frame_num=None)], cfg,
        )[0]
        with h5py.File(out, "r") as f:
            g = f["entry_0000/data"]
            return (
                np.asarray(g["img_gid_q"][0]),
                np.asarray(g["q_xy"][()]),
                np.asarray(g["q_z"][()]),
            )

    a_img, a_qxy, a_qz = convert(plain, "a.h5", poni_a, flipud=True)
    b_img, b_qxy, b_qz = convert(flipped, "b.h5", poni_b, flipud=False)

    # The two routes reach the same geometry through different float
    # arithmetic (flip the poni vs flip the centre and re-derive it), so
    # the axes agree to ~1e-5 relative rather than bit-for-bit. A wedge
    # on the wrong side is a mirror, nowhere near this tolerance.
    assert a_qxy == pytest.approx(b_qxy, rel=1e-4)
    assert a_qz == pytest.approx(b_qz, rel=1e-4)
    # The missing wedge is where the detector maps nothing, so comparing
    # the NaN masks compares wedge placement directly.
    assert np.array_equal(np.isnan(a_img), np.isnan(b_img))
    np.testing.assert_allclose(
        np.nan_to_num(a_img), np.nan_to_num(b_img), rtol=1e-4, atol=1e-4,
    )


# -- panel: Output section and the optional directory --------------------


def _ready_to_convert(panel, tmp_path, *entries):
    """Pin the ticked entries and fill the only other required field."""
    poni = _write_poni(tmp_path)
    panel.poni_path.setText(str(poni))
    panel.frame_mode.setCurrentText("All")
    _fake_entries(panel, *entries)


def test_output_sits_under_selection_and_starts_open(main_window):
    panel = main_window.conversion_panel
    assert [s.title() for s in panel._sections] == [
        "Selection", "Output", "Experimental parameters", "Metadata",
        "Conversion config",
    ]
    assert panel._sections[1].is_expanded()


def test_an_empty_directory_with_a_checked_entry_is_runnable(
        main_window, tmp_path, synthetic_raw):
    from PySide6.QtCore import Qt

    from mlgidlab.file_model import list_raw_entries

    panel = main_window.conversion_panel
    panel.poni_path.setText(str(_write_poni(tmp_path)))
    panel.set_raw_inputs([(synthetic_raw, list_raw_entries(synthetic_raw))])
    assert not panel.btn_convert.isEnabled()
    panel.selection_tree.topLevelItem(0).child(0).setCheckState(
        0, Qt.CheckState.Checked)
    assert panel.output_dir.text() == ""
    assert panel.btn_convert.isEnabled()
    # The placeholder is the live default, not "(required)".
    assert str(synthetic_raw.parent) in panel.output_dir.placeholderText()


def test_an_empty_directory_resolves_to_the_raw_file_s_folder(
        main_window, tmp_path):
    panel = main_window.conversion_panel
    raw_dir = tmp_path / "beamtime"
    raw_dir.mkdir()
    _ready_to_convert(panel, tmp_path, _raw_entry(raw_dir, "scan.h5", 3))
    logged: list[str] = []
    panel.logMessage.connect(logged.append)

    scans, cfg = panel._collect_run_inputs()

    assert cfg.output_dir == raw_dir
    assert any(str(raw_dir) in line for line in logged)


def test_an_explicit_directory_wins(main_window, tmp_path):
    panel = main_window.conversion_panel
    raw_dir = tmp_path / "beamtime"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    _ready_to_convert(panel, tmp_path, _raw_entry(raw_dir, "scan.h5", 3))
    panel.output_dir.setText(str(out_dir))

    _scans, cfg = panel._collect_run_inputs()

    assert cfg.output_dir == out_dir


def test_a_read_only_raw_folder_is_refused(main_window, tmp_path, monkeypatch):
    """Monkeypatched rather than chmod-ed: CI may run as root, which
    writes anywhere."""
    from mlgidlab import conversion_panel as cp

    panel = main_window.conversion_panel
    _ready_to_convert(panel, tmp_path, _raw_entry(tmp_path, "scan.h5", 3))
    monkeypatch.setattr(cp.os, "access", lambda *a, **k: False)

    with pytest.raises(ValueError, match="Choose an output directory"):
        panel._collect_run_inputs()


def test_the_append_target_follows_the_default_directory(
        main_window, tmp_path):
    panel = main_window.conversion_panel
    raw_dir = tmp_path / "beamtime"
    raw_dir.mkdir()
    _existing_output(raw_dir)            # raw_dir/converted.h5, two entries
    _ready_to_convert(panel, tmp_path, _raw_entry(raw_dir, "scan.h5", 3))
    panel.output_mode_combo.setCurrentText(OUTPUT_SEPARATE_DATASETS)

    assert panel._resolve_append_target() == raw_dir / "converted.h5"
    panel.append_frames_chk.setChecked(True)
    items = [panel.append_entry_combo.itemText(i)
             for i in range(panel.append_entry_combo.count())]
    assert items == ["entry_0000", "entry_0001"]

    _scans, cfg = panel._collect_run_inputs()
    assert cfg.append_entry == "entry_0001"
    assert cfg.output_dir == raw_dir
