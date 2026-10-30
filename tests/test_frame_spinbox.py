"""The editable frame-index spinbox next to the frame slider.

Bidirectional sync: typing into the spinbox seeks the viewer (same
path as the slider), any frame change mirrors back into spinbox +
slider + "/ max" label, and the range follows the active stack.

Source: main_window.py ``frame_spin`` construction,
``_on_frame_spin_changed``, ``_on_viewer_frame_changed``,
``_refresh_frame_slider``, ``_frame_label_text``.
"""

from __future__ import annotations

import pytest

from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def test_frame_spinbox_sync(main_window, synthetic_nexus):
    mw = main_window
    mw._set_active_session(NexusSession.open(synthetic_nexus))

    # Range follows the 3-frame stack; cluster visible; label is the
    # "/ max" suffix completing the editable index.
    assert mw.frame_spin.minimum() == 0
    assert mw.frame_spin.maximum() == 2
    assert mw.frame_spin.isVisibleTo(mw)
    assert mw.frame_label.text() == "/ 2"

    # Typing an index seeks the viewer and mirrors into the slider.
    mw.frame_spin.setValue(2)
    assert mw.viewer.current_frame == 2
    assert mw.frame_slider.value() == 2

    # Any other seek path mirrors back into the spinbox.
    mw.viewer.set_frame(1)
    assert mw.frame_spin.value() == 1
    assert mw.frame_slider.value() == 1
    assert mw.frame_label.text() == "/ 2"
