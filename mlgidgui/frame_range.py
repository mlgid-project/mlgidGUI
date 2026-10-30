"""Parser for human-typed frame-range expressions like ``0-34,37``.

Backs the Ctrl+Shift+V "Paste to frames..." dialog. Pure-function,
no Qt deps so it's trivially testable.

Grammar (whitespace-tolerant):
- Comma-separated tokens.
- Each token is either a non-negative integer ``N`` or a closed range
  ``A-B`` with ``A <= B``.

The parser returns ``(valid, dropped)`` after expanding every token
and deduplicating: ``valid`` are frames inside ``[0, n_frames)``,
``dropped`` are frames outside it. Out-of-range values aren't a
syntax error — the GUI prompts the user to confirm before pasting.
"""
from __future__ import annotations


def parse_frame_range(
    text: str, *, n_frames: int,
) -> tuple[list[int], list[int]]:
    """Parse ``text`` and split the expanded frames against ``n_frames``.

    Returns ``(valid_sorted_unique, dropped_sorted_unique)``. Raises
    ``ValueError`` with a message naming the offending token if the
    input is empty or contains a malformed token.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty input")
    raw_tokens = [tok.strip() for tok in stripped.split(",")]
    collected: set[int] = set()
    for tok in raw_tokens:
        if not tok:
            raise ValueError(f"Empty token in {text!r}")
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Malformed range {tok!r}")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise ValueError(f"Malformed range {tok!r}") from None
            if start < 0 or end < 0:
                raise ValueError(f"Negative frame in {tok!r}")
            if start > end:
                raise ValueError(
                    f"Range {tok!r} has start > end ({start} > {end})"
                )
            collected.update(range(start, end + 1))
        else:
            try:
                value = int(tok)
            except ValueError:
                raise ValueError(f"Malformed token {tok!r}") from None
            if value < 0:
                raise ValueError(f"Negative frame in {tok!r}")
            collected.add(value)
    valid = sorted(i for i in collected if 0 <= i < n_frames)
    dropped = sorted(i for i in collected if i < 0 or i >= n_frames)
    return valid, dropped


def compact_repr(frames: list[int]) -> str:
    """Format a sorted list of ints back into the input's range syntax.

    ``[31, 32, 33, 40]`` -> ``"31-33,40"``. ``[]`` -> ``""``. Used by
    the host to echo a list of dropped frames in the out-of-range
    confirmation dialog without dumping a 1000-element list.
    """
    if not frames:
        return ""
    sorted_frames = sorted(set(frames))
    runs: list[tuple[int, int]] = []
    run_start = sorted_frames[0]
    prev = sorted_frames[0]
    for value in sorted_frames[1:]:
        if value == prev + 1:
            prev = value
        else:
            runs.append((run_start, prev))
            run_start = value
            prev = value
    runs.append((run_start, prev))
    parts: list[str] = []
    for start, end in runs:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return ",".join(parts)
