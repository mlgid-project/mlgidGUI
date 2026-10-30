"""The user's contrast (histogram slider) must survive operation
re-renders. Adjusting the slider then adding a peak / running the
pipeline used to snap the contrast back to the robust default, because
every render re-applied the per-build robust levels.

The viewer now remembers the dialled-in levels (`_user_levels`,
captured from the histogram's `sigLevelChangeFinished`) and reuses them
on `preserve_view` re-renders and frame scrubs, while genuinely
re-auto-contrasting only when the data changes (new entry/stack,
log/linear toggle).
"""

from __future__ import annotations

import pytest

from mlgidgui.image_viewer import MODE_CARTESIAN
from mlgidgui.session import NexusSession


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_histogram_drag_is_captured(main_window, synthetic_nexus):
    """Finishing a histogram drag records the levels as sticky."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    assert viewer._user_levels is None  # fresh load auto-contrasts

    # setLevels emits sigLevelChangeFinished, same as a user drag release.
    viewer._view.setLevels(0.15, 0.35)
    assert viewer._user_levels == pytest.approx((0.15, 0.35))


def test_contrast_survives_preserve_render(main_window, synthetic_nexus):
    """An operation re-render (add-peak / pipeline path) keeps the user's
    contrast instead of snapping back to robust."""
    viewer = main_window.viewer
    entry = _open(main_window, synthetic_nexus) and main_window.entry_combo.currentText()
    viewer._view.setLevels(0.15, 0.35)
    user = viewer._user_levels
    assert user is not None

    # This is the exact call add-peak / edits / pipeline-reattach make.
    main_window._load_entry_into_viewer(entry, preserve_view=True)

    assert viewer._user_levels == pytest.approx(user)
    assert viewer._current_levels() == pytest.approx(user)


def test_contrast_survives_frame_scrub(main_window, synthetic_nexus):
    """Scrubbing frames after setting contrast keeps it too."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    viewer._view.setLevels(0.15, 0.35)
    user = viewer._user_levels
    viewer.set_frame(1)
    assert viewer.current_frame == 1
    assert viewer._current_levels() == pytest.approx(user)


def test_fresh_entry_load_resets_contrast(main_window, synthetic_nexus):
    """A non-preserving load (entry change, file open) re-auto-contrasts."""
    viewer = main_window.viewer
    entry = _open(main_window, synthetic_nexus) and main_window.entry_combo.currentText()
    viewer._view.setLevels(0.15, 0.35)
    assert viewer._user_levels is not None

    main_window._load_entry_into_viewer(entry)  # preserve_view defaults False
    assert viewer._user_levels is None


def test_log_toggle_resets_contrast(main_window, synthetic_nexus):
    """Switching linear/log drops the sticky contrast (different domain)."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    viewer._view.setLevels(0.15, 0.35)
    assert viewer._user_levels is not None

    viewer._log_check.setChecked(True)  # fires _on_log_toggled
    assert viewer._user_levels is None


def test_mode_toggle_keeps_contrast(main_window, synthetic_nexus):
    """Cartesian/polar is the same data resampled, so the user's contrast
    persists across a mode toggle."""
    viewer = main_window.viewer
    _open(main_window, synthetic_nexus)
    viewer._view.setLevels(0.15, 0.35)
    user = viewer._user_levels
    viewer.set_mode(MODE_CARTESIAN)
    assert viewer._user_levels == pytest.approx(user)
    assert viewer._current_levels() == pytest.approx(user)
