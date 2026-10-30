"""The NeXus vocabulary the structure editor's forms are built from."""
from __future__ import annotations

import pytest

from mlgidgui import nexus_schema as nx


def test_every_listed_class_has_a_help_line():
    """The combo shows the help as a tooltip; a missing one reads as a bug."""
    missing = [c for c in nx.NX_CLASSES if not nx.class_help(c)]
    assert not missing, f"no help text for {missing!r}"


def test_the_two_classes_mlgidgui_depends_on_are_listed():
    """NXparameters is what pygid stamps on analysis groups; NXdata is how
    the viewer resolves an entry."""
    assert "NXparameters" in nx.NX_CLASSES
    assert "NXdata" in nx.NX_CLASSES


def test_class_list_has_no_duplicates():
    assert len(set(nx.NX_CLASSES)) == len(nx.NX_CLASSES)


def test_unknown_class_is_reported_without_being_an_error():
    assert not nx.is_known_class("NXsomethingelse")
    assert nx.class_help("NXsomethingelse") == ""


@pytest.mark.parametrize(
    "field_name,expected",
    [
        ("wavelength", "Angstrom"),
        ("q_xy", "1/Angstrom"),
        ("angle_of_incidence", "degrees"),
        ("sample_temperature", "K"),
        ("detector_distance", "m"),
    ],
)
def test_unit_suggestions_lead_with_the_expected_unit(field_name, expected):
    assert nx.suggest_units(field_name)[0] == expected


def test_a_longer_hint_beats_a_shorter_one():
    """``wavelength`` contains no other key, but ``q`` must not win over it."""
    assert nx.suggest_units("wavelength") == nx.UNIT_HINTS["wavelength"]


def test_unknown_field_falls_back_to_the_generic_list():
    assert nx.suggest_units("zzz") == nx.GENERIC_UNITS
    assert nx.suggest_units("") == nx.GENERIC_UNITS


def test_templates_are_self_consistent():
    for key, template in nx.TEMPLATES.items():
        assert template.nx_class == key
        assert template.label and template.help
        names = [f.name for f in template.fields]
        assert len(set(names)) == len(names), f"{key} repeats a field name"
        for spec in template.fields:
            assert spec.kind in ("str", "float", "int")


def test_nxdata_template_creates_no_signal_attribute():
    """There is no dataset to point at yet — the validation strip reports
    the gap instead, at the moment the user can fix it."""
    assert nx.TEMPLATES["NXdata"].fields == ()


def test_template_names_are_all_real_templates():
    assert set(nx.template_names()) == set(nx.TEMPLATES)
