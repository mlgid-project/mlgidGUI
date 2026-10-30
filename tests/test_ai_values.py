"""The angle-of-incidence grammar: one value, a list, or a ramp.

Pure logic, no Qt. The two rules worth pinning are the ones that read
backwards: a ramp yields one MORE angle than its third number (pygid's
own ``scan=`` convention), and three numbers mean a ramp only when the
frame count disagrees with three.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.ai_values import describe, parse_ai


def test_a_single_number_is_one_angle_for_every_frame():
    assert parse_ai("0.2", n_frames=14) == pytest.approx(0.2)
    assert parse_ai("  0.2  ", n_frames=1) == pytest.approx(0.2)
    # 0.0 is a legal angle, not a sentinel for "unset" -- that is the
    # caller's job, and it is why empty input raises instead.
    assert parse_ai("0", n_frames=5) == pytest.approx(0.0)


def test_an_explicit_list_keeps_every_value():
    assert parse_ai("0.1, 0.3, 0.5, 0.7", n_frames=4) == pytest.approx(
        [0.1, 0.3, 0.5, 0.7]
    )
    # ...and does not care how many frames there are; the length check
    # belongs to the caller, which can say which two numbers disagree.
    assert len(parse_ai("0.1,0.3,0.5,0.7", n_frames=99)) == 4


def test_brackets_and_a_trailing_comma_are_tolerated():
    expected = pytest.approx([0.1, 0.3, 0.5, 0.7])
    assert parse_ai("(0.1, 0.3, 0.5, 0.7)", n_frames=4) == expected
    assert parse_ai("[0.1, 0.3, 0.5, 0.7]", n_frames=4) == expected
    assert parse_ai("0.1, 0.3, 0.5, 0.7,", n_frames=4) == expected


def test_a_ramp_yields_one_more_angle_than_its_step_count():
    """The third number is intervals, not values -- pygid's convention.

    Asserted against the exact expression pygid uses
    (``ExpParams.__post_init__``), rounding included, so a ramp typed in
    the GUI and a pygid ``scan=`` string of the same three numbers
    cannot diverge.
    """
    result = parse_ai("(0.1, 1.5, 13)", n_frames=14)
    assert len(result) == 14
    expected = np.round(np.linspace(0.1, 1.5, 14), 4).tolist()
    assert result == pytest.approx(expected)
    assert result[0] == pytest.approx(0.1)
    assert result[-1] == pytest.approx(1.5)


def test_a_ramp_of_one_step_is_two_angles():
    assert parse_ai("0.1, 0.5, 1", n_frames=2) == pytest.approx([0.1, 0.5])


def test_three_numbers_are_a_ramp_unless_three_frames_are_selected():
    """The one genuinely ambiguous input, decided by the frame count."""
    text = "0.1, 1.5, 13"
    assert len(parse_ai(text, n_frames=14)) == 14  # ramp
    assert parse_ai(text, n_frames=3) == pytest.approx([0.1, 1.5, 13.0])
    # A three-frame scan therefore cannot take a ramp -- writing the
    # three angles out is the same length either way.
    assert parse_ai("0.1, 0.8, 1.5", n_frames=3) == pytest.approx(
        [0.1, 0.8, 1.5]
    )


def test_empty_input_is_rejected_rather_than_read_as_zero():
    for text in ("", "   ", "()", "[ ]"):
        with pytest.raises(ValueError, match="Empty input"):
            parse_ai(text, n_frames=4)


def test_a_malformed_token_names_itself():
    with pytest.raises(ValueError, match="Not a number: 'abc'"):
        parse_ai("0.1, abc, 0.5, 0.9", n_frames=4)
    with pytest.raises(ValueError, match="Empty value"):
        parse_ai("0.1,, 0.5, 0.9", n_frames=4)


def test_a_ramp_needs_a_whole_positive_step_count():
    with pytest.raises(ValueError, match="whole number"):
        parse_ai("0.1, 1.5, 13.5", n_frames=14)
    with pytest.raises(ValueError, match="at least 1"):
        parse_ai("0.1, 1.5, 0", n_frames=14)
    with pytest.raises(ValueError, match="at least 1"):
        parse_ai("0.1, 1.5, -3", n_frames=14)


def test_angles_outside_the_field_range_are_rejected():
    """Same 0-90° bound the spinbox this replaced enforced."""
    with pytest.raises(ValueError, match="outside"):
        parse_ai("120", n_frames=1)
    with pytest.raises(ValueError, match="outside"):
        parse_ai("0.1, -0.2, 0.5, 0.9", n_frames=4)
    # A ramp's ends are checked too, so it cannot smuggle one past.
    with pytest.raises(ValueError, match="outside"):
        parse_ai("0.1, 95, 13", n_frames=14)


def test_describe_makes_the_resolved_count_visible():
    """The +1 has to be readable before the conversion runs, not after."""
    assert describe(None) == "no angle set"
    assert "0.2" in describe(0.2)
    assert "every frame" in describe(0.2)
    text = describe(parse_ai("(0.1, 1.5, 13)", n_frames=14))
    assert text.startswith("14 angles")
    assert "0.1" in text and "1.5" in text
