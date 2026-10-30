"""The Display dock reports a SAVED fitted peak, not a refit of it.

The Fitted block of ``ParameterPanel`` used to be fed exclusively by
``set_fits``, i.e. by the profile viewer's live 1D scipy Gaussian on
whatever data is on screen. For a peak the pipeline had already fitted
and written to ``fitted_peaks`` that produced three separate
disagreements with the Peaks dock, which prints the stored row:

1. the widths were FWHM against the dock's stored ``2 sigma`` (a fixed
   1.177x),
2. centre and width came from a refit over ``+/- 3 sigma`` of live
   data rather than from the file,
3. the amplitude was a 1D fit height on an axis-averaged profile,
   which is not the 2D Gaussian height the dock prints at all.

A saved row now reads straight off the selection. A *manual* peak keeps
the preview, where forecasting the next commit is the entire point.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from mlgidgui.parameter_panel import EMPTY, ParameterPanel
from mlgidgui.viewer_items import SelectedPeak


def _fitted(**over) -> SelectedPeak:
    kw = dict(
        kind="fitted", frame=0, peak_id=7,
        radius=1.234567, angle=42.4242,
        radius_width=0.098765, angle_width=13.1313,
        is_ring=False, score=0.75, amplitude=1234.5678,
    )
    kw.update(over)
    return SelectedPeak(**kw)


def _live_fit(center, fwhm, amplitude):
    """Stand-in for a ``GaussianFit`` from the profile viewer, with
    values deliberately nowhere near the stored ones."""
    return SimpleNamespace(center=center, fwhm=fwhm, amplitude=amplitude)


def _panel(qtbot) -> ParameterPanel:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    return panel


def _fit_texts(panel) -> list[str]:
    return [
        panel._fit_radius_label.text(), panel._fit_fwhm_r_label.text(),
        panel._fit_angle_label.text(), panel._fit_fwhm_a_label.text(),
        panel._fit_amp_label.text(),
    ]


def _captions(panel) -> list[str]:
    return [w.text() for w in panel._fit_captions]


# -- the stored row is what shows -------------------------------------


def test_fitted_selection_shows_stored_values(qtbot):
    panel = _panel(qtbot)
    peak = _fitted()
    panel.set_peak(peak)
    assert _fit_texts(panel) == [
        "1.235 Å⁻¹", "0.099 Å⁻¹",
        "42.42°", "13.13°", "1.23e+03",
    ]


def test_stored_numbers_match_the_peaks_dock_formatting(qtbot):
    """The dock builds its Fitted row with ``.3f`` radius and width,
    ``.2f`` angles and ``.3g`` amplitude. Same peak, same digits, or
    the two readouts disagree on screen for no reason."""
    from mlgidgui.peaks_table_panel import _num_item

    panel = _panel(qtbot)
    peak = _fitted()
    panel.set_peak(peak)
    dock = [
        _num_item(peak.radius).text(),
        _num_item(peak.radius_width).text(),
        _num_item(peak.angle, fmt="{:.2f}").text(),
        _num_item(peak.angle_width, fmt="{:.2f}").text(),
        _num_item(peak.amplitude, fmt="{:.3g}").text(),
    ]
    shown = [t.rstrip(" Å⁻¹°") for t in _fit_texts(panel)]
    assert shown == dock


def test_matched_selection_shows_stored_values(qtbot):
    """A matched selection is a fitted row seen through a structure, so
    it reads from the file for the same reason."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted(kind="matched", structure_uid="s1"))
    assert panel._fit_radius_label.text() == "1.235 Å⁻¹"
    assert panel._fit_amp_label.text() == "1.23e+03"


def test_widths_are_stored_two_sigma_not_fwhm(qtbot):
    """The specific 1.177x that made one peak read 18% wider in the
    Display dock than in the Peaks dock."""
    panel = _panel(qtbot)
    peak = _fitted(radius_width=0.1, angle_width=10.0)
    panel.set_peak(peak)
    as_fwhm_r = 0.1 * math.sqrt(2.0 * math.log(2.0))
    assert panel._fit_fwhm_r_label.text() == "0.100 Å⁻¹"
    assert panel._fit_fwhm_r_label.text() != f"{as_fwhm_r:.3f} Å⁻¹"


# -- the live fit cannot overwrite it ---------------------------------


def test_live_fit_ignored_for_saved_peak(qtbot):
    """``set_fits`` fires on every profile recompute. It must not land
    on a saved row -- this is the actual regression guard."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    before = _fit_texts(panel)
    panel.set_fits(
        _live_fit(9.9, 9.9, 9.9), _live_fit(99.9, 99.9, 9.9),
    )
    assert _fit_texts(panel) == before


def test_live_fit_ignored_after_reselecting_the_same_peak(qtbot):
    """Selection churn (frame redraw, geometry sync) must not let a
    stale live fit through on the second pass either."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    panel.set_fits(_live_fit(9.9, 9.9, 9.9), _live_fit(99.9, 99.9, 9.9))
    panel.set_peak(_fitted())
    panel.set_fits(_live_fit(8.8, 8.8, 8.8), _live_fit(88.8, 88.8, 8.8))
    assert panel._fit_radius_label.text() == "1.235 Å⁻¹"


# -- the manual preview is untouched ----------------------------------


def test_manual_selection_still_takes_the_live_preview(qtbot):
    """A manual box has no stored row; the block forecasts what
    Add-to-fitted would write, which is what the user asked to keep."""
    panel = _panel(qtbot)
    panel.set_peak(SelectedPeak(
        kind="manual", frame=0, peak_id=-1,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    ))
    panel.set_fits(_live_fit(1.5, 0.3, 500.0), _live_fit(12.0, 6.0, 500.0))
    assert panel._fit_radius_label.text() == "1.500 Å⁻¹"
    assert panel._fit_fwhm_r_label.text() == "0.300 Å⁻¹"
    assert panel._fit_angle_label.text() == "12.00°"
    assert panel._fit_amp_label.text() == "500"


def test_switching_manual_to_fitted_repins_to_the_file(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(SelectedPeak(
        kind="manual", frame=0, peak_id=-1,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    ))
    panel.set_fits(_live_fit(1.5, 0.3, 500.0), _live_fit(12.0, 6.0, 500.0))
    panel.set_peak(_fitted())
    assert panel._fit_radius_label.text() == "1.235 Å⁻¹"


def test_switching_fitted_to_manual_releases_the_pin(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    panel.set_peak(SelectedPeak(
        kind="manual", frame=0, peak_id=-1,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    ))
    panel.set_fits(_live_fit(1.5, 0.3, 500.0), _live_fit(12.0, 6.0, 500.0))
    assert panel._fit_radius_label.text() == "1.500 Å⁻¹"


def test_clearing_the_selection_releases_the_pin(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    panel.set_peak(None)
    assert panel._showing_stored_fit is False


# -- captions name what is actually shown -----------------------------


def test_captions_switch_with_the_source(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    assert _captions(panel) == ["r:", "Δr:", "a:", "Δa:", "Amplitude:"]
    panel.set_peak(SelectedPeak(
        kind="manual", frame=0, peak_id=-1,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    ))
    assert _captions(panel) == [
        "Center r:", "FWHM r:", "Center a:", "FWHM a:", "Amplitude:",
    ]


def test_stored_captions_match_the_peaks_dock_headers(qtbot):
    """One peak, two docks, one set of names."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted())
    assert [c.rstrip(":") for c in _captions(panel)[:4]] == [
        "r", "Δr", "a", "Δa",
    ]


def test_manual_section_header_says_preview(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(SelectedPeak(
        kind="manual", frame=0, peak_id=-1,
        radius=1.0, angle=10.0, radius_width=0.2, angle_width=5.0,
    ))
    assert panel._fitted_section_label.text() == "Fitted peak (preview)"
    panel.set_peak(_fitted())
    assert panel._fitted_section_label.text() == "Fitted peak"


# -- edge cases -------------------------------------------------------


def test_ring_prints_its_stored_infinite_width(qtbot):
    """The Peaks dock prints ``inf`` for a ring, and so does the
    Detected block right above. Matching means matching."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted(is_ring=True, angle_width=math.inf))
    assert panel._fit_fwhm_a_label.text() == "inf°"


def test_missing_amplitude_blanks_rather_than_lying(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(_fitted(amplitude=None))
    assert panel._fit_amp_label.text() == EMPTY
    assert panel._fit_radius_label.text() == "1.235 Å⁻¹"


def test_nan_amplitude_blanks(qtbot):
    panel = _panel(qtbot)
    panel.set_peak(_fitted(amplitude=float("nan")))
    assert panel._fit_amp_label.text() == EMPTY


def test_detected_selection_still_blanks_the_block(qtbot):
    """Unchanged: a detected peak has no fitted readout at all."""
    panel = _panel(qtbot)
    panel.set_peak(_fitted(kind="detected", amplitude=0.0))
    panel.set_fits(_live_fit(9.9, 9.9, 9.9), _live_fit(99.9, 99.9, 9.9))
    assert _fit_texts(panel) == [EMPTY] * 5
