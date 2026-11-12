"""Conversion: user, technique and beamtime NeXus metadata.

pygid 0.2.20 cannot take these during conversion, so the Conversion
panel collects them (Metadata section: "User information", "Technique
information", "Beamtime information", pygid's field names verbatim)
and the engine writes them into every converted entry afterwards with
pygid's own writers. Nothing is mandatory. Techniques are appended to
the ones pygid records itself; beamtime keys left empty become 'nan'
once any one is filled (pygid's behaviour).
Source: conversion_config.py ``USER_FIELDS`` / ``TECHNIQUE_FIELDS`` /
``BEAMTIME_FIELDS``; conversion.py ``clean_post_metadata`` /
``write_post_metadata``; conversion_panel.py ``_build_nexus_info_blocks``
/ ``_collect_nexus_info``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from mlgidgui import conversion
from mlgidgui.conversion_config import (
    BEAMTIME_FIELDS,
    USER_FIELDS,
    ConversionConfig,
)

pytestmark = pytest.mark.gui


def _text(ds) -> str:
    v = ds[()]
    return v.decode() if isinstance(v, bytes) else str(v)


# --- clean_post_metadata (backend-free) -------------------------------------

def test_clean_drops_empty_fields_and_blocks():
    cfg = ConversionConfig(
        user_info=[
            {"field_name": "user_1", "name": " Alex ", "email": ""},
            {"field_name": "user_2"},  # nothing filled: skipped
        ],
        technique_info=[{"name": "", "pid": ""}, {"name": "XRR", "pid": ""}],
        beamtime_info={"title": "", "proposal_id": "MA-1234"},
    )
    meta = conversion.clean_post_metadata(cfg)
    assert meta.users == [{"field_name": "user_1", "name": "Alex"}]
    assert meta.techniques == [{"name": "XRR"}]
    assert meta.beamtime == {"proposal_id": "MA-1234"}


def test_clean_all_empty_is_empty():
    cfg = ConversionConfig(
        user_info=[{"field_name": "user_1", "name": ""}],
        technique_info=[{"name": "", "pid": ""}],
        beamtime_info={k: "" for k in BEAMTIME_FIELDS},
    )
    assert conversion.clean_post_metadata(cfg).is_empty()


def test_clean_defaults_missing_field_name():
    cfg = ConversionConfig(user_info=[
        {"field_name": "", "name": "A"}, {"field_name": " ", "name": "B"},
    ])
    names = [u["field_name"] for u in conversion.clean_post_metadata(cfg).users]
    assert names == ["user_1", "user_2"]


@pytest.mark.parametrize("users, match", [
    ([{"field_name": "a/b", "name": "A"}], "must not contain"),
    ([{"field_name": "x", "name": "A"}, {"field_name": "x", "name": "B"}],
     "share the field_name"),
])
def test_clean_rejects_bad_field_names(users, match):
    with pytest.raises(ValueError, match=match):
        conversion.clean_post_metadata(ConversionConfig(user_info=users))


def test_execute_rejects_bad_field_name_before_backend(tmp_path):
    """The check runs before pygid is imported (CI has no backend)."""
    from mlgidgui.conversion_config import RawScan

    cfg = ConversionConfig(
        poni_path=tmp_path / "x.poni", ai=0.1, output_dir=tmp_path,
        user_info=[{"field_name": "a/b", "name": "A"}],
    )
    with pytest.raises(ValueError, match="must not contain"):
        conversion.execute([RawScan(file_path=tmp_path / "r.tif", entry="")],
                           cfg)


# --- Panel ------------------------------------------------------------------

def test_panel_lists_pygid_fields_and_collects(qtbot):
    from PySide6.QtWidgets import QAbstractButton

    from mlgidgui.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)

    # All five Metadata blocks are collapsed dropdowns.
    for sec, title in (
        (panel.smpl_meta_section, "Sample metadata (YAML)"),
        (panel.exp_meta_section, "Experimental metadata"),
        (panel.user_info_section, "User information"),
        (panel.technique_info_section, "Technique information"),
        (panel.beamtime_info_section, "Beamtime information"),
    ):
        assert not sec.is_expanded()
        assert title in [w.text() for w in sec.findChildren(QAbstractButton)
                         if hasattr(w, "text")]

    # One user block, one technique row and every beamtime field shown
    # from the start, labelled with pygid's names.
    assert len(panel._user_blocks) == 1
    _frame, inputs = panel._user_blocks[0]
    assert list(inputs) == ["field_name", *USER_FIELDS]
    assert inputs["field_name"].text() == "user_1"
    roles = [inputs["role"].itemText(i) for i in range(inputs["role"].count())]
    assert roles == ["", "local_contact", "principal_investigator", "proposer"]
    assert inputs["role"].isEditable()
    assert panel.technique_table.rowCount() == 1
    assert list(panel.beamtime_fields) == list(BEAMTIME_FIELDS)

    inputs["name"].setText("Alex Morgan")
    inputs["role"].setCurrentText("principal_investigator")
    panel._add_user_block()
    assert panel._user_blocks[1][1]["field_name"].text() == "user_2"
    panel._user_blocks[1][1]["ORCID"].setText("0000-0002-1234-5678")
    panel.technique_table.item(0, 0).setText("x-ray reflectivity")
    panel.beamtime_fields["DOI"].setText("10.1/abc")

    cfg = ConversionConfig()
    panel._collect_nexus_info(cfg)
    meta = conversion.clean_post_metadata(cfg)
    assert meta.users == [
        {"field_name": "user_1", "name": "Alex Morgan",
         "role": "principal_investigator"},
        {"field_name": "user_2", "ORCID": "0000-0002-1234-5678"},
    ]
    assert meta.techniques == [{"name": "x-ray reflectivity"}]
    assert meta.beamtime == {"DOI": "10.1/abc"}

    panel._remove_user_block(panel._user_blocks[0][0])
    assert [i["field_name"].text() for _f, i in panel._user_blocks] == ["user_2"]


# --- Writing (needs pygid's writers) ----------------------------------------

def _pygid_like_entry(path: Path) -> None:
    """An entry with pygid's default techniques, as conversion leaves it."""
    with h5py.File(path, "w") as f:
        tech = f.create_group("entry_0000/instrument/techniques")
        f.create_group("entry_0000/instrument/source")
        t0 = tech.create_group("technique_0")
        t0["name"] = "x-ray diffraction"
        t0["pid"] = "http://purl.org/pan-science/PaNET/PaNET01216"
        # A group pygid.NexusFile would refuse (no /data): the writer must
        # not go through NexusFile's whole-file parse.
        f.create_group("not_an_entry")


def test_write_post_metadata_layout(tmp_path):
    pytest.importorskip("pygid")
    out = tmp_path / "conv.h5"
    _pygid_like_entry(out)
    meta = conversion.clean_post_metadata(ConversionConfig(
        user_info=[{"field_name": "pi", "name": "Alex", "role": "proposer"}],
        technique_info=[
            {"name": "x-ray diffraction",  # already there: not repeated
             "pid": "http://purl.org/pan-science/PaNET/PaNET01216"},
            {"name": "x-ray reflectivity", "pid": "PaNET-x"},
        ],
        beamtime_info={"proposal_id": "MA-1234"},
    ))
    conversion.write_post_metadata([(out, "entry_0000")], meta)
    # Twice, as an append run would: still no repeats.
    conversion.write_post_metadata([(out, "entry_0000")], meta)

    with h5py.File(out, "r") as f:
        e = f["entry_0000"]
        user = e["users/pi"]
        assert sorted(user.keys()) == ["name", "role"]
        assert _text(user["name"]) == "Alex"
        assert user.attrs["NX_class"] == "NXuser"

        tech = e["instrument/techniques"]
        assert sorted(tech.keys()) == ["technique_0", "technique_1"]
        assert _text(tech["technique_0/name"]) == "x-ray diffraction"
        assert _text(tech["technique_1/name"]) == "x-ray reflectivity"
        assert _text(tech["technique_1/pid"]) == "PaNET-x"

        notes = e["instrument/source/notes"]
        assert _text(notes["proposal_id"]) == "MA-1234"
        # pygid pads its other required keys with 'nan'.
        for key in set(BEAMTIME_FIELDS) - {"proposal_id"}:
            assert _text(notes[key]) == "nan"


def test_write_post_metadata_empty_touches_nothing(tmp_path):
    out = tmp_path / "conv.h5"
    _pygid_like_entry(out)
    before = out.stat().st_mtime_ns
    conversion.write_post_metadata(
        [(out, "entry_0000")], conversion.PostMetadata(),
    )
    assert out.stat().st_mtime_ns == before


def test_write_post_metadata_failure_is_a_warning(tmp_path, caplog):
    pytest.importorskip("pygid")
    missing = tmp_path / "gone.h5"
    meta = conversion.PostMetadata(beamtime={"title": "T"})
    with caplog.at_level("WARNING"):
        conversion.write_post_metadata([(missing, "entry_0000")], meta)
    assert "could not be written" in caplog.text


def test_execute_writes_metadata_into_converted_entry(tmp_path):
    """End to end through a real pygid conversion (single TIFF)."""
    pytest.importorskip("pygid")
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
        expmeta_overrides={"px_size": 7.5e-05},
        user_info=[{"field_name": "user_1", "email": "a@b.c"}],
        technique_info=[{"name": "x-ray reflectivity", "pid": "P"}],
        beamtime_info={"title": "Beamtime 1"},
    )
    written = conversion.execute([RawScan(file_path=tif, entry="")], cfg)
    with h5py.File(written[0], "r") as f:
        e = f["entry_0000"]
        assert _text(e["users/user_1/email"]) == "a@b.c"
        names = [
            _text(e[f"instrument/techniques/{k}/name"])
            for k in sorted(e["instrument/techniques"].keys())
        ]
        # pygid's own two (GID conversion), then the user's.
        assert names == [
            "x-ray diffraction",
            "grazing incidence wide angle x-ray scattering",
            "x-ray reflectivity",
        ]
        assert _text(e["instrument/source/notes/title"]) == "Beamtime 1"
