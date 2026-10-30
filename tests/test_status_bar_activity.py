"""The status bar as an activity centre.

Three things it now carries beyond plain text: an unsaved-changes dot, a
pipeline cell that shows a run in flight (colour plus a real bar), and a
cursor readout that includes the two derived quantities a GIWAXS user
would otherwise compute by hand.

The physics is worth pinning: d and 2theta are one-line formulas, and a
sign or factor slip in either would be believable on screen and wrong in
a notebook.
"""

from __future__ import annotations

import math

import pytest

from mlgidgui import skin, theme_tokens

pytestmark = pytest.mark.gui


def test_the_dirty_dot_replaces_the_asterisk(main_window, synthetic_nexus):
    """The marker used to be glued to the end of the file name, where it
    reads as part of the name and moves with the text."""
    from mlgidgui.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    name = synthetic_nexus.name

    assert main_window._sb_file.text() == name
    assert "*" not in main_window._sb_file.text()
    # isHidden, not isVisible: the fixture never shows the window, and
    # isVisible is False for every widget whose parent chain is unshown.
    assert main_window._sb_dirty.isHidden()

    main_window.session.dirty = True
    main_window._update_status_file()
    assert not main_window._sb_dirty.isHidden()
    assert main_window._sb_file.text() == name, "the name itself never changes"

    main_window.session.dirty = False
    main_window._update_status_file()
    assert main_window._sb_dirty.isHidden()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_dot_is_a_real_disc_in_the_live_accent(main_window, theme):
    """It is a painted pixmap, not a styled widget, so a theme flip has
    to repaint it — a stylesheet swap cannot reach inside a QPixmap."""
    main_window._set_theme(theme)
    pixmap = main_window._sb_dirty.pixmap()
    assert not pixmap.isNull()
    image = pixmap.toImage()
    centre = image.pixelColor(image.width() // 2, image.height() // 2)
    assert centre.name().lower() == theme_tokens.color("accent", theme)
    corner = image.pixelColor(0, 0)
    assert corner.alpha() == 0, "a disc, not a square"


def test_the_pipeline_cell_shows_a_run_in_flight(main_window):
    from mlgidgui.pipeline import PipelineCommand

    main_window._update_status_pipeline(running=False)
    assert main_window._sb_pipeline.text() == "idle"
    assert main_window._sb_pipe_bar.isHidden()
    assert main_window._sb_pipeline.property("status") in (None, "")

    command = PipelineCommand(op_name="run_detection", kwargs={"entry": "e1"})
    main_window._update_status_pipeline(command, running=True)
    assert "run_detection" in main_window._sb_pipeline.text()
    assert main_window._sb_pipeline.property("status") == "run"

    main_window._update_status_pipeline(running=False)
    assert main_window._sb_pipeline.text() == "idle"
    assert main_window._sb_pipeline.property("status") == ""


def test_frame_progress_promotes_the_bar_from_busy_to_determinate(main_window):
    """Several ops are one opaque backend call, so the bar starts as a
    marquee and only becomes a real 0..N bar once a count arrives."""
    from mlgidgui.pipeline import PipelineCommand

    main_window._pipe_thread = object()          # pretend a run is live
    try:
        command = PipelineCommand(op_name="run_fitting", kwargs={"entry": "e1"})
        main_window._pipe_command = command
        main_window._update_status_pipeline(command, running=True)
        assert main_window._sb_pipe_bar.maximum() == 0, "busy marquee"

        main_window._on_pipeline_frame_progress(3, 12, "run_fitting", "e1")
        assert main_window._sb_pipe_bar.maximum() == 12
        assert main_window._sb_pipe_bar.value() == 3
        assert "3/12 frames" in main_window._sb_pipeline.text()
    finally:
        main_window._pipe_thread = None
    main_window._update_status_pipeline(running=False)
    assert main_window._sb_pipe_bar.maximum() == 0, "reset for the next run"


def test_clicking_the_pipeline_cell_opens_the_logs(main_window, qtbot):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    main_window._logs_dock.hide()
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPoint(2, 2),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    handled = main_window.eventFilter(main_window._sb_pipeline, event)
    assert handled
    assert not main_window._logs_dock.isHidden()


def test_the_cursor_readout_carries_d_and_two_theta(main_window, monkeypatch):
    """d = 2*pi/|q| and 2theta = 2*asin(lambda*|q|/(4*pi)). Checked against
    a hand-computed value, because both would look plausible if wrong."""
    monkeypatch.setattr(type(main_window), "_entry_wavelength",
                        lambda self: 1.0)          # 1 Å, easy arithmetic
    main_window._status_cursor_visible = True

    main_window._on_status_cursor_moved(
        {"mode": "cartesian", "q_xy": 0.3, "q_z": 0.4, "intensity": 12.0})
    text = main_window._sb_cursor.text()
    # |q| = 0.5 exactly
    assert "d=12.566 Å" in text                    # 2*pi/0.5
    expected = 2 * math.degrees(math.asin(1.0 * 0.5 / (4 * math.pi)))
    assert f"2θ={expected:.2f}°" in text
    assert "q_xy=0.300" in text and "q_z=0.400" in text


def test_the_polar_azimuth_is_chi_not_theta(main_window, monkeypatch):
    """The polar map's azimuth used to be written θ, which cannot stand
    next to the scattering angle 2θ without being misread."""
    monkeypatch.setattr(type(main_window), "_entry_wavelength",
                        lambda self: 1.0)
    main_window._status_cursor_visible = True
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 0.5, "theta": 45.0, "intensity": 1.0})
    text = main_window._sb_cursor.text()
    assert "χ=45.0°" in text
    assert "2θ=" in text
    assert "θ=45.0°" not in text.replace("2θ=", "")


def test_two_theta_is_omitted_without_a_wavelength(main_window, monkeypatch):
    """d needs nothing but the cursor; 2theta needs the entry's
    wavelength, so a file without one shows d alone rather than a made-up
    angle."""
    monkeypatch.setattr(type(main_window), "_entry_wavelength",
                        lambda self: None)
    main_window._status_cursor_visible = True
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 0.5, "theta": 10.0, "intensity": 1.0})
    text = main_window._sb_cursor.text()
    assert "d=" in text
    assert "2θ=" not in text


def test_a_raw_pixel_readout_has_no_q_derived_values(main_window):
    main_window._status_cursor_visible = True
    main_window._on_status_cursor_moved(
        {"mode": "pixel", "row": 10, "col": 20, "intensity": 5.0})
    text = main_window._sb_cursor.text()
    assert "row=10" in text
    assert "d=" not in text and "2θ=" not in text


def test_the_skin_defines_the_new_status_roles():
    for theme in ("dark", "light"):
        qss = skin.build_qss(theme)
        assert 'QLabel[role="sb-cell-mono"]' in qss
        assert 'QLabel[role="sb-cell"][status="run"]' in qss
