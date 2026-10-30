"""Parser for the angle-of-incidence field: one value, or one per frame.

A GIWAXS scan is often measured at a varying incidence angle, so the
Conversion panel (and the converted-image import dialog) take either a
single number for every frame or an array with one value per frame.
Pure-function, no Qt deps, same shape as ``frame_range.parse_frame_range``
so both fields are read the same way.

Grammar (whitespace-tolerant, surrounding ``(...)`` or ``[...]`` optional):

- ``0.2`` -- one angle, used for every frame.
- ``0.1, 0.3, 0.5, 0.7`` -- an explicit list, one angle per frame.
- ``0.1, 1.5, 13`` -- a linear ramp: start, end, **steps**.

Two things about the ramp are deliberate and easy to get wrong:

**The count is one MORE than the third number.** ``(0.1, 1.5, 13)``
gives 14 angles, because the third number is the number of *intervals*.
That is pygid's own convention -- ``ExpParams.__post_init__`` parses its
``scan="start end steps"`` string as
``np.round(np.linspace(start, end, steps + 1), 4)`` -- and this parser
mirrors it exactly, rounding included, so the same three numbers typed
here and into pygid produce an identical list. It reads backwards at
first glance, which is why every caller shows ``describe()`` next to the
field.

**Three numbers are ambiguous**, and the frame count breaks the tie: with
exactly three frames selected, three numbers are three angles; otherwise
they are a ramp. So a three-frame scan cannot be given a ramp -- write
the three angles out, which is the same length either way.
"""
from __future__ import annotations

import numpy as np

#: The range the field accepts, matching the spinbox it replaced.
AI_MIN = 0.0
AI_MAX = 90.0


def parse_ai(text: str, *, n_frames: int) -> float | list[float]:
    """Parse ``text`` into one angle or a per-frame list.

    ``n_frames`` is the number of frames on the selection's frame axis;
    it only decides how three numbers are read (see the module
    docstring). Raises ``ValueError`` naming the offending token if the
    input is empty or malformed. The caller maps empty input to "not
    set" -- an unset angle is not this function's business, because
    ``0.0`` is a legal angle and must not double as a sentinel.
    """
    stripped = text.strip()
    for opener, closer in (("(", ")"), ("[", "]")):
        if stripped.startswith(opener) and stripped.endswith(closer):
            stripped = stripped[1:-1].strip()
            break
    if not stripped:
        raise ValueError("Empty input")

    tokens = [tok.strip() for tok in stripped.split(",")]
    if tokens and not tokens[-1]:
        # A trailing comma is how a list is written out; drop it rather
        # than reporting an empty token the user cannot see.
        tokens.pop()
    values: list[float] = []
    for tok in tokens:
        if not tok:
            raise ValueError(f"Empty value in {text!r}")
        try:
            values.append(float(tok))
        except ValueError:
            raise ValueError(f"Not a number: {tok!r}") from None

    if not values:
        raise ValueError("Empty input")
    if len(values) == 1:
        return _checked(values[0])
    if len(values) == 3 and n_frames != 3:
        return _ramp(values, n_frames)
    return [_checked(v) for v in values]


def _ramp(values: list[float], n_frames: int) -> list[float]:
    """``start, end, steps`` -> ``steps + 1`` angles, exactly as pygid does.

    A failure here is most often someone who meant three *angles* and
    has more than three frames, so every message says which reading was
    taken and why -- otherwise "step count must be a whole number" for
    the input ``0.1, 0.2, 0.3`` is baffling.
    """
    start, end, steps = values
    if steps != int(steps):
        raise ValueError(
            f"Step count must be a whole number, got {steps:g}. "
            + _ramp_because(n_frames)
        )
    steps = int(steps)
    if steps < 1:
        raise ValueError(
            f"Step count must be at least 1, got {steps}. "
            + _ramp_because(n_frames)
        )
    _checked(start)
    _checked(end)
    # Mirrors pygid ExpParams.__post_init__ (expparams.py), rounding and
    # the +1 included, so a ramp typed here equals pygid's scan= string.
    return [
        float(v) for v in np.round(np.linspace(start, end, steps + 1), 4)
    ]


def _ramp_because(n_frames: int) -> str:
    """Why three numbers were read as a ramp rather than three angles."""
    tail = (
        f"{n_frames} are selected" if n_frames else "none are selected yet"
    )
    return (
        "Three numbers are read as a ramp (start, end, steps) unless "
        f"exactly 3 frames are selected -- {tail}."
    )


def _checked(value: float) -> float:
    if not AI_MIN <= value <= AI_MAX:
        raise ValueError(
            f"Angle {value:g}° is outside {AI_MIN:g}-{AI_MAX:g}°"
        )
    return float(value)


def describe(value: float | list[float] | None) -> str:
    """One line naming what a parsed value actually is.

    Shared by the panel's live hint and the run's log line so the two
    cannot drift -- and so the ramp's ``+1`` is visible *before* the
    conversion runs rather than after it produced the wrong count.
    """
    if value is None:
        return "no angle set"
    if not isinstance(value, list):
        return f"one angle ({value:g}°) for every frame"
    if not value:
        return "no angles"
    if len(value) == 1:
        return f"one angle ({value[0]:g}°) for every frame"
    return (
        f"{len(value)} angles, {value[0]:g}° to {value[-1]:g}°"
    )
