"""NeXus vocabulary for the structure editor: base classes, unit hints
and the group templates behind "New group".

Data only, plus a handful of lookups — no Qt, no h5py, no backend
import, so the editor's forms can consult it from anywhere.

Why a curated list rather than the full NXDL set: the class list feeds
an *editable* combo box, so a name that is missing here can still be
typed. Shipping the ~30 classes a GIWAXS file actually uses keeps the
dropdown scannable, where all ~90 base classes would make it a wall.
The two mlgidGUI cares about most are pinned by a test:
``NXparameters`` is what pygid stamps on the analysis groups
(``file_model.ANALYSIS_GROUP_ATTRS``) and ``NXdata`` is what carries the
``signal`` attribute the viewer resolves an entry through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: The attribute that names a group's NeXus base class.
NX_CLASS_ATTR = "NX_class"

#: Curated base classes, grouped by what they are for. Order is the
#: order the combo shows them in: the containers a user reaches for
#: first, then instrument parts, then the odds and ends.
NX_CLASSES: tuple[str, ...] = (
    # containers
    "NXentry",
    "NXsubentry",
    "NXdata",
    "NXcollection",
    "NXparameters",
    "NXprocess",
    "NXnote",
    # sample and environment
    "NXsample",
    "NXsample_component",
    "NXenvironment",
    "NXsensor",
    "NXlog",
    # instrument and beamline
    "NXinstrument",
    "NXsource",
    "NXbeam",
    "NXdetector",
    "NXdetector_group",
    "NXdetector_module",
    "NXmonochromator",
    "NXmonitor",
    "NXattenuator",
    "NXaperture",
    "NXslit",
    "NXpinhole",
    "NXmirror",
    "NXcrystal",
    "NXgrating",
    "NXpositioner",
    "NXtransformations",
    # people and provenance
    "NXuser",
    "NXroot",
)

#: One line per class, shown as the combo's tooltip. Deliberately plain
#: language: the point is that someone who has never read the NeXus
#: standard can still pick the right container.
NX_CLASS_HELP: dict[str, str] = {
    "NXentry": "One measurement. The top-level container of a NeXus file.",
    "NXsubentry": "A measurement within a measurement, for multi-technique data.",
    "NXdata": "The plottable data of an entry: a signal plus its axes.",
    "NXcollection": "A free bag of related fields with no required layout.",
    "NXparameters": "Parameters of a processing step. What pygid stamps on analysis groups.",
    "NXprocess": "A record of a processing step applied to the data.",
    "NXnote": "Free text, or an arbitrary attached file.",
    "NXsample": "What was measured: name, formula, temperature, geometry.",
    "NXsample_component": "One component of a multi-part sample.",
    "NXenvironment": "Sample environment: cryostat, furnace, cell.",
    "NXsensor": "A single sensor reading of an environment.",
    "NXlog": "A time series: values with timestamps.",
    "NXinstrument": "The beamline: source, optics, detector.",
    "NXsource": "The x-ray source itself.",
    "NXbeam": "The beam at a point: wavelength, flux, size.",
    "NXdetector": "The detector: pixel size, distance, geometry.",
    "NXdetector_group": "A group of detectors treated together.",
    "NXdetector_module": "One module of a segmented detector.",
    "NXmonochromator": "Wavelength selection.",
    "NXmonitor": "A beam monitor's counts.",
    "NXattenuator": "An absorber in the beam.",
    "NXaperture": "A beam-defining aperture.",
    "NXslit": "A slit pair.",
    "NXpinhole": "A pinhole.",
    "NXmirror": "A reflecting optic.",
    "NXcrystal": "A crystal optic.",
    "NXgrating": "A diffraction grating optic.",
    "NXpositioner": "A motor or stage axis.",
    "NXtransformations": "The chain of rotations and translations placing a component.",
    "NXuser": "Who ran the measurement.",
    "NXroot": "The file root itself.",
}

#: Unit strings offered as completions, keyed by a lowercase substring of
#: the field name. Longest match wins, so ``q_xy`` beats the generic
#: fallback. NeXus fixes the unit *category* per field but leaves the
#: string free, so these are suggestions, never validation.
UNIT_HINTS: dict[str, tuple[str, ...]] = {
    "wavelength": ("Angstrom", "nm", "m"),
    "energy": ("keV", "eV", "J"),
    "distance": ("m", "mm", "cm"),
    "q": ("1/Angstrom", "1/nm", "1/m"),
    "angle": ("degrees", "rad"),
    "theta": ("degrees", "rad"),
    "incidence": ("degrees", "rad"),
    "time": ("s", "ms", "us"),
    "temperature": ("K", "degC"),
    "pixel": ("m", "mm", "um"),
    "size": ("m", "mm", "um"),
    "position": ("m", "mm", "um"),
    "pressure": ("Pa", "mbar", "bar"),
    "current": ("A", "mA", "uA"),
    "thickness": ("nm", "um", "mm"),
}

#: Offered when nothing in ``UNIT_HINTS`` matches.
GENERIC_UNITS: tuple[str, ...] = (
    "", "m", "mm", "um", "nm", "Angstrom", "1/Angstrom", "1/nm",
    "degrees", "rad", "s", "ms", "K", "degC", "eV", "keV", "counts",
)

FieldKind = Literal["str", "float", "int"]


@dataclass(frozen=True)
class FieldSpec:
    """One dataset a template creates inside its group."""

    name: str
    kind: FieldKind = "str"
    value: object = ""
    units: str | None = None
    help: str = ""


@dataclass(frozen=True)
class Template:
    """A "New group" preset: the class plus the fields worth pre-creating."""

    nx_class: str
    label: str
    help: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)


#: The presets offered by "New group". Kept deliberately thin: a
#: template pre-creates only fields that are near-universal and that a
#: user would otherwise have to look up. Anything optional is better
#: added deliberately than deleted from a bloated skeleton.
#:
#: NXdata gets no ``signal`` attribute even though the class requires
#: one, because at creation time there is no dataset to point it at. The
#: validation strip reports the gap instead, which is the moment the
#: user can actually fix it.
TEMPLATES: dict[str, Template] = {
    "NXentry": Template(
        nx_class="NXentry",
        label="Entry (NXentry)",
        help="A measurement. Usually the top level of the file.",
        fields=(
            FieldSpec("title", "str", "", help="Human-readable name of the measurement."),
            FieldSpec("start_time", "str", "", help="ISO 8601 timestamp, e.g. 2026-08-24T14:03:00+02:00."),
        ),
    ),
    "NXdata": Template(
        nx_class="NXdata",
        label="Plottable data (NXdata)",
        help=(
            "The signal and its axes. Set the group's 'signal' attribute "
            "to the name of the dataset that holds the data."
        ),
    ),
    "NXsample": Template(
        nx_class="NXsample",
        label="Sample (NXsample)",
        help="What was measured.",
        fields=(
            FieldSpec("name", "str", "", help="Sample name."),
            FieldSpec("chemical_formula", "str", "", help="Hill-notation formula, e.g. C62H68N2O2S5."),
            FieldSpec("temperature", "float", 0.0, units="K", help="Sample temperature."),
        ),
    ),
    "NXinstrument": Template(
        nx_class="NXinstrument",
        label="Instrument (NXinstrument)",
        help="The beamline. Add NXsource / NXdetector groups inside it.",
        fields=(
            FieldSpec("name", "str", "", help="Beamline or instrument name."),
        ),
    ),
    "NXmonitor": Template(
        nx_class="NXmonitor",
        label="Monitor (NXmonitor)",
        help="A beam monitor's readings.",
        fields=(
            FieldSpec("mode", "str", "monitor", help="'monitor' or 'timer'."),
            FieldSpec("preset", "float", 0.0, help="Preset value the count ran to."),
        ),
    ),
    "NXcollection": Template(
        nx_class="NXcollection",
        label="Collection (NXcollection)",
        help="A free bag of fields. Use when nothing more specific fits.",
    ),
}


def template_names() -> list[str]:
    """Template keys, in the order the New-group menu should show them."""
    return list(TEMPLATES)


def suggest_units(field_name: str) -> tuple[str, ...]:
    """Unit strings worth offering for a dataset called ``field_name``.

    Longest matching hint wins so a specific key (``wavelength``) beats
    a generic one (``length``) when both appear in the name. Falls back
    to ``GENERIC_UNITS`` when nothing matches.
    """
    name = (field_name or "").lower()
    best: tuple[str, ...] | None = None
    best_len = -1
    for key, units in UNIT_HINTS.items():
        if key in name and len(key) > best_len:
            best, best_len = units, len(key)
    return best if best is not None else GENERIC_UNITS


def class_help(nx_class: str) -> str:
    """One-line description of ``nx_class``; empty for unknown names."""
    return NX_CLASS_HELP.get(nx_class, "")


def is_known_class(nx_class: str) -> bool:
    """Whether ``nx_class`` is in the curated list.

    False is not an error — the class combo is editable on purpose, and
    a file may legitimately use a base class this module does not list.
    Callers use it only to decide whether to show the "not a base class
    we know" hint.
    """
    return nx_class in NX_CLASS_HELP
