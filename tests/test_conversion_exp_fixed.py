"""Conversion: the fixed experimental metadata fields.

The Experimental metadata dropdown shows six fixed fields (start_time,
end_time, source_type, source_name, instrument_name, temperature, pygid's
``ExpMetadata`` names) above the custom key/value table. They are written
after the conversion, because pygid passes its metadata object to the
first batch only and later batches overwrite the required fields with
'nan'. Left empty, a field is not touched, so pygid's own handling stays:
the required ones read 'nan', source_name and temperature are absent.
Source: conversion_config.py ``EXP_FIXED_FIELDS``; conversion.py
``_merge_fixed_exp_fields`` / ``_write_exp_fields`` /
``_build_exp_metadata``; conversion_panel.py ``exp_fixed_fields``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from mlgidgui import conversion
from mlgidgui.conversion_config import EXP_FIXED_FIELDS, ConversionConfig

pytestmark = pytest.mark.gui


def _text(ds) -> str:
    v = ds[()]
    return v.decode() if isinstance(v, bytes) else str(v)


# --- Cleaning (backend-free) ------------------------------------------------

def test_fixed_field_names():
    assert EXP_FIXED_FIELDS == (
        "start_time", "end_time", "source_type", "source_name",
        "instrument_name", "temperature",
    )


def test_clean_keeps_only_filled_fixed_fields():
    cfg = ConversionConfig(expmeta_fixed={
        "start_time": " 2026-11-22T14:30:00 ", "end_time": "",
        "instrument_name": "P08", "bogus": "x",
    })
    meta = conversion.clean_post_metadata(cfg)
    assert meta.experiment == {
        "start_time": "2026-11-22T14:30:00", "instrument_name": "P08",
    }
    assert not meta.is_empty()


def test_clean_all_empty_fixed_fields_is_empty():
    cfg = ConversionConfig(expmeta_fixed={k: "" for k in EXP_FIXED_FIELDS})
    assert conversion.clean_post_metadata(cfg).is_empty()


def test_custom_row_with_a_fixed_name_fills_the_field():
    cfg = ConversionConfig(
        expmeta_kv={"temperature": "300", "my_key": "v"},
        expmeta_fixed={"temperature": ""},
    )
    assert conversion.clean_post_metadata(cfg).experiment == {
        "temperature": "300",
    }


def test_same_value_twice_is_fine_different_values_raise():
    same = ConversionConfig(
        expmeta_kv={"source_name": "PETRA III"},
        expmeta_fixed={"source_name": "PETRA III"},
    )
    assert conversion.clean_post_metadata(same).experiment == {
        "source_name": "PETRA III",
    }
    clash = ConversionConfig(
        expmeta_kv={"source_name": "A"}, expmeta_fixed={"source_name": "B"},
    )
    with pytest.raises(ValueError, match="set twice"):
        conversion.clean_post_metadata(clash)


def test_pygid_object_gets_only_custom_keys():
    """Fixed names never reach pygid's ExpMetadata (its later batches
    would overwrite them with 'nan'); custom keys still do."""
    seen = {}

    class _Fake:
        class ExpMetadata:
            def __init__(self, **kw):
                seen.update(kw)

    obj = conversion._build_exp_metadata(
        _Fake, {"temperature": "300", "my_key": "v", "": "x"},
    )
    assert seen == {"my_key": "v"}
    assert obj.extend_fields == ["my_key"]
    assert conversion._build_exp_metadata(_Fake, {"start_time": "t"}) is None


# --- Panel ------------------------------------------------------------------

def test_panel_shows_fixed_fields_above_the_custom_table(qtbot):
    from mlgidgui.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)
    assert list(panel.exp_fixed_fields) == list(EXP_FIXED_FIELDS)
    assert all(e.text() == "" for e in panel.exp_fixed_fields.values())
    # The custom table and its buttons are still there and still empty.
    assert panel.exp_meta_table.rowCount() == 0
    panel._add_manual_meta_row()
    assert panel.exp_meta_table.rowCount() == 1

    panel.exp_fixed_fields["start_time"].setText("2026-11-22T14:30:00")
    panel.exp_fixed_fields["temperature"].setText(" 300 ")
    panel.exp_meta_table.item(0, 0).setText("my_key")
    panel.exp_meta_table.item(0, 1).setText("v")
    cfg = panel._collect_config()
    assert cfg.expmeta_fixed["start_time"] == "2026-11-22T14:30:00"
    assert cfg.expmeta_fixed["temperature"] == "300"
    assert cfg.expmeta_fixed["end_time"] == ""
    assert cfg.expmeta_kv == {"my_key": "v"}


# --- Writing ----------------------------------------------------------------

def _pygid_like_entry(path: Path) -> None:
    """An entry as pygid leaves it when no experimental field was given."""
    with h5py.File(path, "w") as f:
        e = f.create_group("entry_0000")
        e["start_time"] = "nan"
        e["end_time"] = "nan"
        e["instrument/name"] = "nan"
        e["instrument/source/type"] = "nan"
        e["instrument/source/probe"] = "x-ray"


def test_write_layout_and_untouched_empties(tmp_path):
    pytest.importorskip("pygid")
    out = tmp_path / "conv.h5"
    _pygid_like_entry(out)
    meta = conversion.clean_post_metadata(ConversionConfig(expmeta_fixed={
        "start_time": "2026-11-22T14:30:00", "source_name": "PETRA III",
        "instrument_name": "P08", "temperature": "300",
    }))
    conversion.write_post_metadata([(out, "entry_0000")], meta)
    # Twice, as an append run would: replaced, never listed or repeated.
    conversion.write_post_metadata([(out, "entry_0000")], meta)

    with h5py.File(out, "r") as f:
        e = f["entry_0000"]
        assert _text(e["start_time"]) == "2026-11-22T14:30:00"
        assert _text(e["instrument/name"]) == "P08"
        assert _text(e["instrument/source/name"]) == "PETRA III"
        assert e["instrument/temperature"][()] == pytest.approx(300.0)
        assert e["instrument/temperature"].dtype.kind == "f"
        # Not filled: what pygid wrote stays, nothing new appears.
        assert _text(e["end_time"]) == "nan"
        assert _text(e["instrument/source/type"]) == "nan"
        assert _text(e["instrument/source/probe"]) == "x-ray"


def test_temperature_that_is_not_a_number_is_kept_as_text(tmp_path):
    pytest.importorskip("pygid")
    out = tmp_path / "conv.h5"
    _pygid_like_entry(out)
    meta = conversion.clean_post_metadata(
        ConversionConfig(expmeta_fixed={"temperature": "300 K"}),
    )
    conversion.write_post_metadata([(out, "entry_0000")], meta)
    with h5py.File(out, "r") as f:
        assert _text(f["entry_0000/instrument/temperature"]) == "300 K"


def _convert(tmp_path, **cfg_kw):
    import fabio.tifimage
    import numpy as np

    from mlgidgui.conversion_config import CONV_DET2Q_GID, RawScan

    img = (np.random.default_rng(0).random((220, 180)) * 1000).astype(
        np.uint32
    )
    tif = tmp_path / "img.tif"
    fabio.tifimage.TifImage(data=img).write(str(tif))
    poni = tmp_path / "small.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Detector\n"
        'Detector_config: {"pixel1": 7.5e-05, "pixel2": 7.5e-05, '
        '"max_shape": [220, 180]}\n'
        "Distance: 0.2871\nPoni1: 0.0084\nPoni2: 0.00675\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 9.6e-11\n"
    )
    cfg = ConversionConfig(
        conv_type=CONV_DET2Q_GID, poni_path=poni, ai=0.1,
        output_dir=tmp_path, output_filename="out.h5",
        expmeta_overrides={"px_size": 7.5e-05}, **cfg_kw,
    )
    return conversion.execute([RawScan(file_path=tif, entry="")], cfg)


def test_empty_fields_behave_like_pygid_end_to_end(tmp_path):
    pytest.importorskip("pygid")
    written = _convert(tmp_path)
    with h5py.File(written[0], "r") as f:
        e = f["entry_0000"]
        for path in ("start_time", "end_time", "instrument/name",
                     "instrument/source/type"):
            assert _text(e[path]) == "nan"
        assert "name" not in e["instrument/source"]
        assert "temperature" not in e["instrument"]


def test_filled_fields_end_to_end_with_a_custom_field(tmp_path):
    pytest.importorskip("pygid")
    written = _convert(
        tmp_path,
        expmeta_kv={"humidity": "40", "temperature": "295"},
        expmeta_fixed={
            "start_time": "2026-11-22T14:30:00",
            "end_time": "2026-11-22T14:45:00",
            "source_type": "synchrotron", "source_name": "PETRA III",
            "instrument_name": "P08",
        },
    )
    with h5py.File(written[0], "r") as f:
        e = f["entry_0000"]
        assert _text(e["start_time"]) == "2026-11-22T14:30:00"
        assert _text(e["end_time"]) == "2026-11-22T14:45:00"
        assert _text(e["instrument/source/type"]) == "synchrotron"
        assert _text(e["instrument/source/name"]) == "PETRA III"
        assert _text(e["instrument/name"]) == "P08"
        # The custom row named temperature filled the fixed field.
        assert e["instrument/temperature"][()] == pytest.approx(295.0)
        # Custom fields still go through pygid unchanged.
        assert "humidity" in e["instrument"]
