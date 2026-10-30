"""Unit tests for ``mlgidgui.frame_range.parse_frame_range``.

Pure unit, no Qt. Backs the Ctrl+Shift+V "Paste to frames..." dialog
input. Out-of-range frames are not a syntax error — the parser splits
into ``(valid, dropped)`` and the GUI prompts to confirm before
pasting. Negative tokens and malformed tokens are errors.
"""
from __future__ import annotations

import pytest

from mlgidgui.frame_range import compact_repr, parse_frame_range


def test_single_integer():
    valid, dropped = parse_frame_range("7", n_frames=10)
    assert valid == [7]
    assert dropped == []


def test_simple_range_inclusive():
    valid, dropped = parse_frame_range("2-5", n_frames=10)
    assert valid == [2, 3, 4, 5]
    assert dropped == []


def test_mixed_comma_range():
    valid, dropped = parse_frame_range("0-3,7,9", n_frames=10)
    assert valid == [0, 1, 2, 3, 7, 9]
    assert dropped == []


def test_whitespace_tolerated():
    valid, dropped = parse_frame_range("  0 - 2 , 4 ,  6 - 7  ", n_frames=10)
    assert valid == [0, 1, 2, 4, 6, 7]
    assert dropped == []


def test_dedup_and_sort():
    valid, dropped = parse_frame_range("5,5-7,3", n_frames=10)
    assert valid == [3, 5, 6, 7]
    assert dropped == []


def test_single_value_range():
    valid, dropped = parse_frame_range("4-4", n_frames=10)
    assert valid == [4]
    assert dropped == []


def test_dropped_out_of_range():
    valid, dropped = parse_frame_range("0-100", n_frames=30)
    assert valid == list(range(0, 30))
    assert dropped == list(range(30, 101))


def test_dropped_mixed_with_valid():
    valid, dropped = parse_frame_range("0,5,30,45", n_frames=10)
    assert valid == [0, 5]
    assert dropped == [30, 45]


def test_empty_raises():
    with pytest.raises(ValueError, match="Empty"):
        parse_frame_range("", n_frames=10)
    with pytest.raises(ValueError, match="Empty"):
        parse_frame_range("   ", n_frames=10)


def test_empty_token_between_commas_raises():
    with pytest.raises(ValueError, match="Empty token"):
        parse_frame_range("3,,5", n_frames=10)


def test_malformed_alpha_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_frame_range("a-b", n_frames=10)
    with pytest.raises(ValueError, match="Malformed"):
        parse_frame_range("foo", n_frames=10)


def test_open_range_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_frame_range("3-", n_frames=10)
    with pytest.raises(ValueError, match="Malformed"):
        parse_frame_range("-3", n_frames=10)


def test_start_greater_than_end_raises():
    with pytest.raises(ValueError, match="start > end"):
        parse_frame_range("3-1", n_frames=10)


def test_multi_dash_token_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_frame_range("1-2-3", n_frames=10)


def test_negative_n_frames_raises():
    with pytest.raises(ValueError, match="n_frames"):
        parse_frame_range("0", n_frames=-1)


def test_n_frames_zero_drops_everything():
    valid, dropped = parse_frame_range("0,5", n_frames=0)
    assert valid == []
    assert dropped == [0, 5]


def test_compact_repr_empty():
    assert compact_repr([]) == ""


def test_compact_repr_single():
    assert compact_repr([7]) == "7"


def test_compact_repr_contiguous_run():
    assert compact_repr([3, 4, 5, 6]) == "3-6"


def test_compact_repr_mixed():
    assert compact_repr([31, 32, 33, 40]) == "31-33,40"


def test_compact_repr_unsorted_input_normalised():
    assert compact_repr([7, 1, 2, 3, 7, 9]) == "1-3,7,9"
