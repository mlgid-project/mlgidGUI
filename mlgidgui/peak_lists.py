"""Extra per-frame peak tables, registered by the user as display layers.

A file's ``analysis/frameNNNNN/`` group can hold peak tables mlgidGUI
knows nothing about -- ``Br_peaks``, a second detector's picks, a
reference set from another tool. This module is the registry that turns
one of those into a layer: a dataset name, the label to show for it, and
whether it should look and read like a detected or a fitted layer.

**The ``list:`` prefix is the whole no-spill mechanism.** A registered
layer's kind is ``"list:Br_peaks"``, which can never equal
``"detected"``, ``"fitted"``, ``"manual"`` or ``"matched"`` -- and every
consumer in the app names its kinds explicitly (``tables.get("fitted")``,
``kind in ("manual", "detected")``). So the Peaks table, CSV export,
fitting, matching, tracking and the simulation overlay all ignore a
registered layer without a line of code, and the ways it CAN be reached
are the handful that were widened on purpose.

Pure and Qt-light on purpose (QSettings only, no h5py, no viewer): the
same shape as ``ai_values`` and ``frame_range``, so it stays testable
without a window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from PySide6.QtCore import QSettings

import logging

logger = logging.getLogger(__name__)

#: Where the registry lives. App-wide, like the pen overrides: a list is
#: a way of looking at files, not a property of one file.
SETTINGS_KEY = "extraPeakLists"

#: What a registered layer's kind string starts with. Chosen so it can
#: never collide with a built-in kind -- see the module docstring.
KIND_PREFIX = "list:"

TREAT_DETECTED = "detected"
TREAT_FITTED = "fitted"
TREAT_AS = (TREAT_DETECTED, TREAT_FITTED)

#: Tables mlgidGUI owns. Never offered as an extra list, because the app
#: already has a layer for each and registering one would draw it twice.
RESERVED_DATASETS = frozenset({
    "detected_peaks", "fitted_peaks", "fitted_peaks_errors",
})

#: Prefix of the per-solution matched tables, which are a derived view of
#: ``fitted_peaks`` rather than a peak list of their own.
RESERVED_PREFIX = "matched"

#: Where a standard table is parked while a primary list stands in for
#: it during a pipeline run. Deliberately unmistakable: a name starting
#: with this in a frame group means a run was interrupted mid-swap, and
#: the recovery sweep can put things back on that evidence alone.
STASH_PREFIX = "__mlgidgui_stash__"

#: What each redirectable pipeline op reads and writes. Only these two
#: ops can take a primary table -- see ``swap_plan``.
OP_WRITES = {
    "run_detection": ("detected_peaks",),
    "run_fitting": ("fitted_peaks", "fitted_peaks_errors"),
}
OP_READS = {
    "run_fitting": ("detected_peaks",),
}

#: Which flavour of primary stands in for which standard table.
_STANDARD_FOR = {
    TREAT_DETECTED: ("detected_peaks",),
    TREAT_FITTED: ("fitted_peaks", "fitted_peaks_errors"),
}


@dataclass(frozen=True)
class PeakListSpec:
    """One registered extra peak table."""

    dataset: str
    label: str
    treat_as: str = TREAT_DETECTED
    #: Whether the pipeline reads and writes THIS table in place of the
    #: standard one for its flavour. Pipeline runs only -- see
    #: ``swap_plan``; everything else in the app still uses the standard
    #: tables, so a primary list is otherwise the display-only layer it
    #: has always been.
    primary: bool = False

    @property
    def kind(self) -> str:
        """The viewer/selection kind string for this layer."""
        return f"{KIND_PREFIX}{self.dataset}"

    @property
    def display_label(self) -> str:
        """What the Display dock calls it; the dataset name if unnamed."""
        return self.label.strip() or self.dataset

    @property
    def standard_datasets(self) -> tuple[str, ...]:
        """The standard table(s) this list stands in for when primary.

        A fitted list stands in for two: pygidFIT writes
        ``fitted_peaks`` and ``fitted_peaks_errors`` in the same call,
        so a redirect that captured only the first would leave the
        errors of run N sitting beside the peaks of run N+1.
        """
        return _STANDARD_FOR.get(self.treat_as, ())

    @property
    def datasets(self) -> tuple[str, ...]:
        """This list's own name(s), positionally matching the above."""
        if self.treat_as == TREAT_FITTED:
            return (self.dataset, errors_name(self.dataset))
        return (self.dataset,)


def errors_name(dataset: str) -> str:
    """The twin a fitted primary needs for ``fitted_peaks_errors``."""
    return f"{dataset}_errors"


def is_list_kind(kind: str) -> bool:
    """Whether ``kind`` names a registered extra list rather than a built-in."""
    return str(kind).startswith(KIND_PREFIX)


def dataset_for_kind(kind: str) -> str:
    """The dataset name inside a list kind ("" if it is not one)."""
    text = str(kind)
    return text[len(KIND_PREFIX):] if is_list_kind(text) else ""


def spec_for_kind(specs, kind: str) -> PeakListSpec | None:
    """The spec a kind refers to, or None when it is not registered."""
    for spec in specs:
        if spec.kind == kind:
            return spec
    return None


def is_reserved(dataset: str) -> bool:
    """Whether ``dataset`` is a table mlgidGUI already draws itself."""
    name = str(dataset)
    return name in RESERVED_DATASETS or name.startswith(RESERVED_PREFIX)


def load_specs() -> list[PeakListSpec]:
    """The registered lists, in the order the user put them in.

    Anything unreadable is dropped rather than raised: this is a
    settings file, and a hand-edited or future-version value must not
    stop the app from starting.
    """
    raw = QSettings().value(SETTINGS_KEY, "")
    if not raw:
        return []
    try:
        stored = json.loads(str(raw))
    except (ValueError, TypeError):
        logger.debug("unreadable %s setting; ignoring", SETTINGS_KEY)
        return []
    if not isinstance(stored, list):
        return []
    out: list[PeakListSpec] = []
    seen: set[str] = set()
    claimed: set[str] = set()
    for item in stored:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset") or "").strip()
        if not dataset or dataset in seen or is_reserved(dataset):
            continue
        treat_as = str(item.get("treat_as") or TREAT_DETECTED)
        treat_as = treat_as if treat_as in TREAT_AS else TREAT_DETECTED
        # At most ONE primary per flavour, enforced on read as well as
        # in the dialog: two would make the swap plan ambiguous, and
        # this value is a settings string a user can hand-edit. First
        # one wins, the rest degrade to ordinary display layers.
        primary = bool(item.get("primary")) and treat_as not in claimed
        if primary:
            claimed.add(treat_as)
        out.append(PeakListSpec(
            dataset=dataset,
            label=str(item.get("label") or "").strip(),
            treat_as=treat_as,
            primary=primary,
        ))
        seen.add(dataset)
    return out


def primary_for(specs, treat_as: str) -> PeakListSpec | None:
    """The list standing in for ``treat_as``'s standard table, if any."""
    for spec in specs:
        if spec.primary and spec.treat_as == treat_as:
            return spec
    return None


def swap_plan(op_name: str, specs) -> dict | None:
    """What to rename around ``op_name`` so it uses the primary tables.

    Returns ``None`` when nothing is primary for anything this op
    touches, which is what keeps the ordinary pipeline path provably
    untouched: no plan, no file writes, not even a marker.

    The plan is plain data (lists of string pairs) so it can ride on a
    ``PipelineCommand`` across the worker-thread boundary without
    dragging Qt or this module into ``pipeline.py``:

    * ``stash``   -- standard names to park under ``STASH_PREFIX``
    * ``swap_in`` -- ``(primary, standard)`` moves, done after stashing
    * ``capture`` -- ``(standard, primary)`` moves, done after the run

    ``capture`` is not simply ``swap_in`` reversed in spirit: what sits
    at the standard name afterwards is whatever the RUN left there,
    which for an output table is new data and for an input table is the
    same rows that were swapped in.
    """
    touched: list[str] = list(OP_WRITES.get(op_name, ()))
    for name in OP_READS.get(op_name, ()):
        if name not in touched:
            touched.append(name)
    if not touched:
        return None

    stash: list[str] = []
    swap_in: list[tuple[str, str]] = []
    capture: list[tuple[str, str]] = []
    for spec in specs:
        if not spec.primary:
            continue
        for own, standard in zip(spec.datasets, spec.standard_datasets):
            if standard not in touched:
                continue
            stash.append(standard)
            swap_in.append((own, standard))
            capture.append((standard, own))
    if not stash:
        return None
    return {"stash": stash, "swap_in": swap_in, "capture": capture}


def swapped_name(plan: dict | None, dataset: str) -> str:
    """Which table a run under ``plan`` actually used for ``dataset``.

    ``"fitted_peaks"`` unless a primary stood in for it, in which case
    the primary's own name. Lets a follow-on pass work on the tables
    the run really touched instead of the standard pair.
    """
    for standard, own in (plan or {}).get("capture", ()):
        if standard == dataset:
            return own
    return dataset


def redirects_dataset(plan: dict | None, dataset: str) -> bool:
    """Whether ``plan`` moves ``dataset`` out of the way for a run.

    The question two follow-on passes ask about ``fitted_peaks``: when
    it is redirected the run never rewrote it, so invalidating matched
    solutions and phase tracks would be destroying good data.
    """
    return bool(plan) and dataset in plan.get("stash", ())


def save_specs(specs) -> None:
    QSettings().setValue(SETTINGS_KEY, json.dumps([
        {"dataset": s.dataset, "label": s.label, "treat_as": s.treat_as,
         "primary": bool(s.primary)}
        for s in specs
    ]))
