"""Per-action 1D/2D fit-mode selector for Add-to-fitted.

Two new pieces of plumbing:

* ``ParameterPanel`` exposes a radio pair and ``fit_mode()`` →
  ``"scipy_1d"`` | ``"pygidfit_2d"``; collapses to 1D when the ring
  checkbox is on (pygidfit can't fit rings).
* ``MainWindow._update_fitted_preview`` paints the dashed cyan
  preview only in 1D mode (where ``_build_fitted_row_1d`` saves the
  scipy fits directly, so the preview can predict the saved box
  exactly). In 2D mode the preview is hidden because pygidfit's
  refined centre / widths can't be predicted from the integrated
  scipy 1D fits — the drawn ROI / detected box already shows the
  fit input region, and the saved blue box appears at pygidfit's
  actual result after the commit.

These tests exercise the ParameterPanel widget directly (no host
needed) and the preview-painter contract via a stub viewer.
"""
from __future__ import annotations

import math

import pytest

from mlgidgui.parameter_panel import ParameterPanel


# -- ParameterPanel.fit_mode() ---------------------------------------


def test_fit_mode_default_is_2d(qtbot):
    """Default selection is 2D pygidfit — matches the post-F-06
    behaviour and what the pipeline run_fitting writes."""
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_2D
    assert panel.rb_fit_2d.isChecked() is True
    assert panel.rb_fit_1d.isChecked() is False


def test_fit_mode_round_trip_through_radios(qtbot):
    """Toggling the 1D radio flips ``fit_mode()`` and back."""
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.rb_fit_1d.setChecked(True)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_1D
    panel.rb_fit_2d.setChecked(True)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_2D


def test_fit_mode_changed_signal_fires(qtbot):
    """The fit-mode signal emits the new token (and only once per
    user click — even though both radios toggle in an exclusive
    group, the panel's lambda fans them in)."""
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    received: list[str] = []
    panel.fitModeChanged.connect(received.append)

    # Programmatic flip fires two ``buttonToggled`` events (one per
    # radio in the exclusive pair). The lambda emits one
    # fitModeChanged per buttonToggled, so we expect 2 emissions for
    # one flip. The important contract is that the LAST emitted
    # value reflects the new mode.
    panel.rb_fit_1d.setChecked(True)
    assert received[-1] == ParameterPanel.FIT_MODE_1D
    panel.rb_fit_2d.setChecked(True)
    assert received[-1] == ParameterPanel.FIT_MODE_2D


def test_ring_overrides_fit_mode_to_1d(qtbot):
    """When ring storage is on, ``fit_mode()`` reports 1D regardless
    of which radio is checked. pygidfit's segment model can't fit
    rings, so the ring code path must always use the legacy 1D
    machinery."""
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.rb_fit_2d.setChecked(True)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_2D
    panel.chk_save_as_ring.setChecked(True)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_1D
    panel.chk_save_as_ring.setChecked(False)
    assert panel.fit_mode() == ParameterPanel.FIT_MODE_2D


def test_ring_greys_out_radios(qtbot):
    """Visual gate: the 1D/2D radios are disabled while ring is on.
    Without this the UI would suggest the user can pick a different
    mode when in fact ring forces 1D."""
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    assert panel.rb_fit_1d.isEnabled() is True
    assert panel.rb_fit_2d.isEnabled() is True
    panel.chk_save_as_ring.setChecked(True)
    assert panel.rb_fit_1d.isEnabled() is False
    assert panel.rb_fit_2d.isEnabled() is False
    panel.chk_save_as_ring.setChecked(False)
    assert panel.rb_fit_1d.isEnabled() is True
    assert panel.rb_fit_2d.isEnabled() is True


# -- _update_fitted_preview width conventions ------------------------


class _DummyGaussianFit:
    """Minimal stand-in for ``mlgidgui.fit.GaussianFit`` so we can
    feed the preview slot without driving scipy."""
    def __init__(self, center: float, fwhm: float, amplitude: float = 1.0):
        self.center = float(center)
        self.fwhm = float(fwhm)
        self.amplitude = float(amplitude)


def _preview_args(main_window, monkeypatch, *, mode, rfit, afit, save_as_ring=False):
    """Drive ``_update_fitted_preview`` with the given mode + fits and
    return whatever ``viewer.set_fitted_preview`` was called with.

    Patches the panel's ``fit_mode()`` and ``save_as_ring()`` rather
    than driving the radios, so we can hold mode constant while we
    swap rfit/afit/sel.
    """
    captured: dict = {}

    def _capture(center_r, width_r, center_a, width_a, *, is_ring=False):
        captured["args"] = (center_r, width_r, center_a, width_a, is_ring)

    monkeypatch.setattr(main_window.viewer, "set_fitted_preview", _capture)
    monkeypatch.setattr(main_window.parameter_panel, "fit_mode", lambda: mode)
    monkeypatch.setattr(main_window.parameter_panel, "save_as_ring", lambda: save_as_ring)
    main_window._update_fitted_preview(rfit, afit)
    return captured.get("args")


def _select_synthetic_peak(main_window, *, kind="manual"):
    """Inject a SelectedPeak into the viewer so ``_update_fitted_preview``
    has a target to render the preview around. Synthesised directly —
    no need to drive the click handler."""
    from mlgidgui.image_viewer import SelectedPeak
    peak = SelectedPeak(
        kind=kind, frame=0, peak_id=0,
        radius=1.5, angle=45.0,
        radius_width=0.20, angle_width=8.0,
    )
    main_window.viewer._selected = peak
    return peak


def test_preview_2d_mode_paints_from_fit_params(main_window, monkeypatch):
    """``_update_fitted_preview`` paints at ``(centre, 2σ)`` for
    whatever fit params the host supplies — regardless of mode.

    In real use, 2D mode paints from the pygidfit results cached by
    ``_refresh_2d_preview``, so the box on screen equals what
    Add-to-fitted (2D) will save. The painter itself is
    mode-agnostic; the orchestration that picks which fits to feed
    in lives in ``_refresh_2d_preview`` and isn't exercised here."""
    _select_synthetic_peak(main_window)
    rfit = _DummyGaussianFit(center=1.5, fwhm=0.10)
    afit = _DummyGaussianFit(center=45.0, fwhm=6.0)
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_2D, rfit=rfit, afit=afit,
    )
    assert args is not None
    cr, wr, ca, wa, is_ring = args
    fwhm_to_2sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))
    assert cr == pytest.approx(1.5)
    assert wr == pytest.approx(0.10 * fwhm_to_2sigma)
    assert ca == pytest.approx(45.0)
    assert wa == pytest.approx(6.0 * fwhm_to_2sigma)
    assert is_ring is False


def test_preview_1d_mode_paints_scipy_fit_widths(main_window, monkeypatch):
    """1D mode paints ``(centre, 2σ)`` per axis from scipy's 1D
    fit, exactly matching what ``_build_fitted_row_1d`` will save
    for the same selection."""
    _select_synthetic_peak(main_window)
    rfit = _DummyGaussianFit(center=1.5, fwhm=0.10)
    afit = _DummyGaussianFit(center=45.0, fwhm=6.0)
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_1D, rfit=rfit, afit=afit,
    )
    assert args is not None
    cr, wr, ca, wa, is_ring = args
    fwhm_to_2sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))
    assert cr == pytest.approx(1.5)
    assert wr == pytest.approx(0.10 * fwhm_to_2sigma)
    assert ca == pytest.approx(45.0)
    assert wa == pytest.approx(6.0 * fwhm_to_2sigma)
    assert is_ring is False


def test_preview_falls_back_to_box_in_1d_mode_when_fit_missing(main_window, monkeypatch):
    """In 1D mode, when scipy's 1D fit hasn't converged on an axis
    (narrow detected box, weird shape), that axis falls back to
    the selected peak's literal drawn geometry so the preview is
    still visible."""
    peak = _select_synthetic_peak(main_window, kind="detected")
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_1D, rfit=None, afit=None,
    )
    assert args is not None
    cr, wr, ca, wa, is_ring = args
    assert cr == pytest.approx(peak.radius)
    assert wr == pytest.approx(peak.radius_width)
    assert ca == pytest.approx(peak.angle)
    assert wa == pytest.approx(peak.angle_width)
    assert is_ring is False


def test_preview_per_axis_fallback_in_1d_mode(main_window, monkeypatch):
    """Per-axis fallback in 1D mode: the radial axis uses the scipy
    fit when present, the angular axis falls back to the drawn box
    when scipy didn't converge — preview is always visible."""
    peak = _select_synthetic_peak(main_window)
    rfit = _DummyGaussianFit(center=1.45, fwhm=0.08)
    fwhm_to_2sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_1D, rfit=rfit, afit=None,
    )
    assert args is not None
    cr, wr, ca, wa, is_ring = args
    # Radial: fit present → 2σ.
    assert cr == pytest.approx(1.45)
    assert wr == pytest.approx(0.08 * fwhm_to_2sigma)
    # Angular: fit missing → drawn-box fallback.
    assert ca == pytest.approx(peak.angle)
    assert wa == pytest.approx(peak.angle_width)
    assert is_ring is False


def test_preview_visible_in_1d_ring_mode_without_angular(main_window, monkeypatch):
    """Ring mode (which forces 1D): the angular fit isn't required
    (the saved row uses ``angle=45°, angle_width=∞`` regardless),
    so the preview appears when the radial fit is available even
    if the angular fit isn't. Radial width uses scipy's ``2σ``."""
    _select_synthetic_peak(main_window)
    rfit = _DummyGaussianFit(center=1.5, fwhm=0.10)
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_1D, rfit=rfit, afit=None,
        save_as_ring=True,
    )
    cr, wr, ca, wa, is_ring = args
    fwhm_to_2sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))
    assert cr == pytest.approx(1.5)
    assert wr == pytest.approx(0.10 * fwhm_to_2sigma)
    assert ca is None
    assert wa is None
    assert is_ring is True


def test_preview_cleared_when_selection_is_not_candidate(main_window, monkeypatch):
    """fitted / matched / None selections must clear the preview —
    they already have a stored box and the cyan refit overlay would
    be visual noise."""
    main_window.viewer._selected = None
    args = _preview_args(
        main_window, monkeypatch,
        mode=ParameterPanel.FIT_MODE_2D, rfit=None, afit=None,
    )
    cr, wr, ca, wa, is_ring = args
    assert cr is None and wr is None and ca is None and wa is None




# -- The live 2D preview runs off the GUI thread ------------------------
#
# A pygidfit preview fit can take tens of seconds for a peak in a badly
# conditioned peak-on-ring cluster. It runs on ``PreviewFitRunner`` so
# selecting such a peak (by click, by T, by Down / Up) never blocks.


def _runner_wait(qtbot, runner, timeout=5000):
    qtbot.waitUntil(lambda: runner.pending_count() == 0, timeout=timeout)


def test_runner_delivers_a_result_and_skips_unwanted_jobs(qtbot):
    import threading
    from mlgidgui.preview_fit import PreviewFitRunner

    runner = PreviewFitRunner()
    got = []
    runner.done.connect(lambda k, r, e, ran: got.append((k, r, e, ran)))
    gate = threading.Event()
    runner.want({"a", "c"})                      # "b" is not wanted
    runner.submit("a", lambda: (gate.wait(5) and "ra", None))
    runner.submit("b", lambda: ("rb", None))
    runner.submit("c", lambda: ("rc", None))
    assert runner.pending_count() == 3
    gate.set()
    qtbot.waitUntil(lambda: len(got) == 3, timeout=5000)
    assert got == [
        ("a", "ra", None, True),
        ("b", None, None, False),                # skipped, not run
        ("c", "rc", None, True),                 # jobs run in order
    ]
    for k, *_ in got:
        runner.forget(k)
    assert runner.pending_count() == 0
    runner.shutdown()


def test_runner_reports_a_job_that_raises(qtbot):
    from mlgidgui.preview_fit import PreviewFitRunner

    runner = PreviewFitRunner()
    got = []
    runner.done.connect(lambda k, r, e, ran: got.append((k, r, e, ran)))
    runner.want({"x"})

    def boom():
        raise ValueError("no")

    runner.submit("x", boom)
    qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
    assert got[0][1] is None and "ValueError" in got[0][2] and got[0][3]
    runner.shutdown()


@pytest.mark.gui
def test_selecting_a_slow_peak_does_not_block_and_fills_in_later(
    main_window, synthetic_nexus_with_peaks, qtbot, monkeypatch,
):
    import threading
    import time
    from types import SimpleNamespace
    from mlgidgui.session import NexusSession

    w = main_window
    w._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    qtbot.wait(300)
    gate = threading.Event()
    fit = SimpleNamespace(
        radius=2.0, radius_width=0.2, angle=45.0, angle_width=5.0,
        amplitude=1.0, theta=0.0, A=1.0, B=0.0, C=1.0,
    )

    def prepare(sel, entry, frame):
        return (lambda: (gate.wait(10) and fit, None)), None

    monkeypatch.setattr(w, "_prepare_pygidfit_fit", prepare)
    det = (w.viewer._frame_peaks.get(0) or {})["detected"]
    sel = w.viewer._row_selection("detected", det, 1, 0)

    t0 = time.time()
    w.viewer._set_selected(sel)                  # the fit is gated shut
    assert time.time() - t0 < 1.0                # selecting returned at once
    assert w._preview_runner.pending_count() == 1
    assert "2D preview" in w.statusBar().currentMessage()
    assert w._current_pygidfit_box_for_selection(w.viewer.selected_peak) is None

    gate.set()                                    # the fit finishes
    _runner_wait(qtbot, w._preview_runner)
    box = w._current_pygidfit_box_for_selection(w.viewer.selected_peak)
    assert box == pytest.approx((2.0, 0.2, 45.0, 5.0))
    assert "2D preview" not in w.statusBar().currentMessage()


@pytest.mark.gui
def test_a_result_for_a_peak_no_longer_selected_is_cached_not_painted(
    main_window, synthetic_nexus_with_peaks, qtbot, monkeypatch,
):
    import threading
    from types import SimpleNamespace
    from mlgidgui.session import NexusSession

    w = main_window
    w._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    qtbot.wait(300)
    gate = threading.Event()
    fit = SimpleNamespace(
        radius=9.0, radius_width=0.9, angle=9.0, angle_width=9.0,
        amplitude=1.0, theta=0.0, A=1.0, B=0.0, C=1.0,
    )
    monkeypatch.setattr(
        w, "_prepare_pygidfit_fit",
        lambda sel, entry, frame: ((lambda: (gate.wait(10) and fit, None)), None),
    )
    painted = []
    monkeypatch.setattr(w, "_repaint_fitted_preview", lambda: painted.append(1))
    det = (w.viewer._frame_peaks.get(0) or {})["detected"]
    w.viewer._set_selected(w.viewer._row_selection("detected", det, 0, 0))
    first_key = next(iter(w._preview_runner._pending))
    w.viewer._set_selected(None)                  # selection moved on
    painted.clear()
    gate.set()
    _runner_wait(qtbot, w._preview_runner)
    assert w._2d_preview_cache[first_key] is fit  # kept for a re-selection
    assert painted == []                          # but not painted


@pytest.mark.gui
def test_closing_the_window_does_not_wait_for_a_running_fit(
    main_window, synthetic_nexus_with_peaks, qtbot, monkeypatch,
):
    import threading
    import time
    from mlgidgui.session import NexusSession

    w = main_window
    w._set_active_session(NexusSession.open(synthetic_nexus_with_peaks))
    qtbot.wait(300)
    never = threading.Event()
    monkeypatch.setattr(
        w, "_prepare_pygidfit_fit",
        lambda sel, entry, frame: ((lambda: (never.wait(30), None)), None),
    )
    det = (w.viewer._frame_peaks.get(0) or {})["detected"]
    w.viewer._set_selected(w.viewer._row_selection("detected", det, 0, 0))
    t0 = time.time()
    w._preview_runner.shutdown()
    assert time.time() - t0 < 1.0
    never.set()
