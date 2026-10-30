"""An empty simulated CIF pattern must read as a data problem.

pygidsim <= 0.1.4 ended ``GIWAXS.giwaxs_2d`` with a named
``ValueError("There are no peaks in the pattern.")``. Since 0.1.7 it
returns empty arrays, so the failure lands one frame later inside
mlgidmatch (``intensity / intensity.max()`` on a zero-size array) and
numpy's message mentions neither CIFs nor q ranges.
``pipeline._build_cif_pattern_from_raw`` re-raises it as a
``RuntimeError`` that names the input and the usual cause.

The wrap is narrow on purpose: only the two empty-pattern wordings are
relabelled (pygidsim's own and the numpy one it became). Any other
``ValueError`` - a malformed CIF out of xrayutilities, an unknown
element - already names its own cause and goes up untouched.

Driven with a stub ``CifPattern`` injected into ``sys.modules``, so
these run on a backend-free CI box as well as against the real stack.
"""

from __future__ import annotations

import sys
import types

import pytest

from mlgidgui import pipeline

# The numpy wording mlgidmatch lets through for an empty pattern.
_NUMPY_EMPTY = (
    "zero-size array to reduction operation maximum which has no identity"
)


@pytest.fixture
def cif_folder(tmp_path):
    """A folder with two .cif files. Content is irrelevant: the stubbed
    CifPattern never reads them, only the names reach the message."""
    folder = tmp_path / "cifs"
    folder.mkdir()
    for name in ("alpha.cif", "beta.cif"):
        (folder / name).write_text("data_stub\n")
    return folder


@pytest.fixture
def stub_cif_pattern(monkeypatch):
    """Install a fake ``mlgidmatch.preprocess.cif_preprocess.CifPattern``.

    ``_build_cif_pattern_from_raw`` imports it lazily inside the
    function, so replacing the module entries is enough and no backend
    has to be installed. Returns a setter that takes the callable the
    stub should delegate to.
    """
    # Params derivation is a separate concern (and imports pygidsim), so
    # it is short-circuited rather than stubbed field by field.
    monkeypatch.setattr(
        pipeline, "_exp_params_from_nexus",
        lambda nexus_file, entry=None: object(),
    )

    captured: dict = {}

    def install(behaviour):
        def cif_pattern(**kwargs):
            captured.update(kwargs)
            return behaviour(**kwargs)

        cif_preprocess = types.ModuleType(
            "mlgidmatch.preprocess.cif_preprocess"
        )
        cif_preprocess.CifPattern = cif_pattern
        preprocess = types.ModuleType("mlgidmatch.preprocess")
        preprocess.cif_preprocess = cif_preprocess
        root = types.ModuleType("mlgidmatch")
        root.preprocess = preprocess
        for name, module in (
            ("mlgidmatch", root),
            ("mlgidmatch.preprocess", preprocess),
            ("mlgidmatch.preprocess.cif_preprocess", cif_preprocess),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return captured

    return install


def test_empty_pattern_becomes_an_actionable_runtime_error(
    tmp_path, cif_folder, stub_cif_pattern
):
    def raise_numpy_empty(**_kwargs):
        raise ValueError(_NUMPY_EMPTY)

    stub_cif_pattern(raise_numpy_empty)

    with pytest.raises(RuntimeError) as excinfo:
        pipeline._build_cif_pattern_from_raw(
            [str(cif_folder)], tmp_path / "scan.h5", "entry_0000"
        )

    message = str(excinfo.value)
    # The input, the diagnosis and the two things to check.
    assert "alpha.cif" in message and "beta.cif" in message
    assert "q range" in message
    assert "q_xy / q_z extents" in message
    # The numpy wording is kept, so the real cause is still greppable.
    assert _NUMPY_EMPTY in message
    # Chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_the_pre_0_1_7_wording_is_recognised_too(
    tmp_path, cif_folder, stub_cif_pattern
):
    """pygidsim <= 0.1.4 raised its own named error for the same cause,
    and an environment pinned to it should get the same guidance."""
    def raise_pygidsim_empty(**_kwargs):
        raise ValueError("There are no peaks in the pattern.")

    stub_cif_pattern(raise_pygidsim_empty)

    with pytest.raises(RuntimeError) as excinfo:
        pipeline._build_cif_pattern_from_raw(
            [str(cif_folder)], tmp_path / "scan.h5", "entry_0000"
        )
    assert "q range" in str(excinfo.value)


def test_an_unrelated_value_error_is_not_relabelled(
    tmp_path, cif_folder, stub_cif_pattern
):
    """A malformed CIF must keep its own message. Calling that "the
    pattern came out empty" would send the user to check a q range that
    has nothing to do with it."""
    def raise_bad_cif(**_kwargs):
        raise ValueError("cannot parse symmetry operation 'x,y,z,'")

    stub_cif_pattern(raise_bad_cif)

    with pytest.raises(ValueError) as excinfo:
        pipeline._build_cif_pattern_from_raw(
            [str(cif_folder)], tmp_path / "scan.h5", "entry_0000"
        )
    assert not isinstance(excinfo.value, RuntimeError)
    assert "symmetry operation" in str(excinfo.value)
    assert "q range" not in str(excinfo.value)


def test_a_successful_parse_is_untouched(
    tmp_path, cif_folder, stub_cif_pattern
):
    """The wrap must be transparent: the CifPattern itself is returned,
    built with the same kwargs as before."""
    sentinel = object()
    captured = stub_cif_pattern(lambda **_kwargs: sentinel)

    result = pipeline._build_cif_pattern_from_raw(
        [str(cif_folder)], tmp_path / "scan.h5", "entry_0000"
    )

    assert result is sentinel
    assert captured["cifs"] == ["alpha.cif", "beta.cif"]
    assert captured["create_all"] is True


def test_many_cifs_are_summarised_not_listed():
    """A folder of CIFs must not turn a one-line error into a wall."""
    few = pipeline._describe_cif_input("/data/cifs", ["a.cif", "b.cif"])
    assert few == "a.cif, b.cif (in cifs)"

    many = pipeline._describe_cif_input(
        "/data/cifs", [f"s{i}.cif" for i in range(10)]
    )
    assert "s0.cif, s1.cif, s2.cif" in many
    assert "+7 more" in many
    assert "s9.cif" not in many
