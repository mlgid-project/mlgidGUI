"""Interactive raw-preview orientation: the Conversion panel's
Orientation row (fliplr, flipud, transpose) reorients the live raw
preview to match what pygid's conversion produces.

Covers:
  - the pure orientation helper ``_apply_raw_flips`` (mirrors pygid's
    ``process_image`` ORDER: transpose, then flipud, then fliplr, on the
    file-order frame);
  - ``_raw_pixel_index``, which inverts that for the cursor readout;
  - ``GIWAXSImageViewer.set_raw_flips`` re-rendering the displayed frame;
  - ``ConversionPanel.rawFlipsChanged`` emission on toggle;
  - the MainWindow panel -> viewer wiring.

Source: image_viewer.py ``_apply_raw_flips`` / ``set_raw_flips`` /
``_render_frame`` raw branch; conversion_panel.py ``_emit_raw_flips``;
main_window.py ``_on_raw_flips_changed``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.image_viewer import _apply_raw_flips
from mlgidgui.viewer_items import _raw_pixel_index

pytestmark = pytest.mark.gui


def test_apply_raw_flips_matches_numpy_for_all_combinations():
    arr = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # (H, W)
    assert np.array_equal(_apply_raw_flips(arr, False, False), arr)
    assert np.array_equal(_apply_raw_flips(arr, True, False), np.fliplr(arr))
    assert np.array_equal(_apply_raw_flips(arr, False, True), np.flipud(arr))
    # both -> flipud then fliplr == 180 deg rotation (order-independent)
    assert np.array_equal(
        _apply_raw_flips(arr, True, True), np.fliplr(np.flipud(arr))
    )
    # input is not mutated
    assert np.array_equal(arr, np.arange(3 * 4, dtype=np.float32).reshape(3, 4))


def test_transpose_is_applied_before_the_flips():
    """pygid's ``process_image`` transposes FIRST; order is not cosmetic.

    Transposing a flipped frame is a different image from flipping a
    transposed one, so getting this backwards would show a preview that
    disagrees with the file the conversion writes.
    """
    arr = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # (H, W)
    assert np.array_equal(_apply_raw_flips(arr, False, False, True), arr.T)
    assert _apply_raw_flips(arr, False, False, True).shape == (4, 3)
    assert np.array_equal(
        _apply_raw_flips(arr, True, True, True),
        np.fliplr(np.flipud(arr.T)),
    )
    # ...and that is NOT the same as transposing last.
    assert not np.array_equal(
        _apply_raw_flips(arr, True, False, True),
        np.fliplr(arr).T,
    )


def test_raw_pixel_index_inverts_every_orientation():
    """The cursor is over the oriented preview; the stack is in file order.

    Checked against the helper itself rather than by hand: whatever
    ``_apply_raw_flips`` puts at (row, col) must be the file-order pixel
    the index maps back to.
    """
    arr = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # (H, W)
    for transp in (False, True):
        for flip_lr in (False, True):
            for flip_ud in (False, True):
                shown = _apply_raw_flips(arr, flip_lr, flip_ud, transp)
                for r in range(shown.shape[0]):
                    for c in range(shown.shape[1]):
                        row, col = _raw_pixel_index(
                            r, c, arr.shape, flip_lr, flip_ud, transp,
                        )
                        assert arr[row, col] == shown[r, c]


def test_raw_pixel_index_rejects_a_point_off_the_frame():
    """Out of bounds is (-1, -1), which the readout turns into NaN.

    The bound that matters is the *oriented* one: after a transpose a
    3x4 frame is 4 rows of 3, and row 3 is valid while column 3 is not.
    """
    assert _raw_pixel_index(3, 0, (3, 4), False, False, True) == (0, 3)
    assert _raw_pixel_index(0, 3, (3, 4), False, False, True) == (-1, -1)
    assert _raw_pixel_index(3, 0, (3, 4), False, False, False) == (-1, -1)
    assert _raw_pixel_index(-1, 0, (3, 4), False, False, False) == (-1, -1)


def test_viewer_set_raw_flips_reorients_preview(qtbot):
    from mlgidgui.image_viewer import GIWAXSImageViewer

    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    # distinctive (N, H, W) so every orientation is distinguishable
    stack = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    viewer.show_raw_stack(stack)

    def displayed():
        return np.asarray(viewer._view.getImageItem().image)

    frame = stack[0]
    # default: shown as stored (transpose only), no flip
    assert not viewer._raw_flip_lr and not viewer._raw_flip_ud
    assert np.array_equal(displayed(), frame.T)
    # fliplr only
    viewer.set_raw_flips(True, False)
    assert np.array_equal(displayed(), np.fliplr(frame).T)
    # flipud only
    viewer.set_raw_flips(False, True)
    assert np.array_equal(displayed(), np.flipud(frame).T)
    # both
    viewer.set_raw_flips(True, True)
    assert viewer._raw_flip_lr and viewer._raw_flip_ud
    assert np.array_equal(displayed(), np.fliplr(np.flipud(frame)).T)
    # revert
    viewer.set_raw_flips(False, False)
    assert np.array_equal(displayed(), frame.T)


def test_viewer_transposes_the_preview_like_the_flips(qtbot):
    from mlgidgui.image_viewer import GIWAXSImageViewer

    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    stack = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    viewer.show_raw_stack(stack)

    def displayed():
        return np.asarray(viewer._view.getImageItem().image)

    frame = stack[0]
    assert not viewer._raw_transp
    viewer.set_raw_flips(False, False, True)
    assert viewer._raw_transp is True
    assert np.array_equal(displayed(), frame.T.T)
    # combined with a flip, in pygid's order
    viewer.set_raw_flips(True, False, True)
    assert np.array_equal(displayed(), np.fliplr(frame.T).T)
    viewer.set_raw_flips(False, False, False)
    assert np.array_equal(displayed(), frame.T)


def test_the_cursor_readout_follows_the_orientation(qtbot):
    """Reading the wrong pixel is the failure a flipped preview hides."""
    from PySide6.QtCore import QPointF
    from mlgidgui.image_viewer import GIWAXSImageViewer

    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    stack = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    viewer.show_raw_stack(stack)

    # Top-left of the displayed frame, unflipped: file-order (0, 0).
    info = viewer._compute_cursor_info(QPointF(0.0, 0.0))
    assert (info["row"], info["col"]) == (0, 0)
    assert info["intensity"] == pytest.approx(stack[0, 0, 0])

    # Flipped left-right, the same screen point is the far column.
    viewer.set_raw_flips(True, False)
    info = viewer._compute_cursor_info(QPointF(0.0, 0.0))
    assert (info["row"], info["col"]) == (0, 3)
    assert info["intensity"] == pytest.approx(stack[0, 0, 3])

    # Transposed, the displayed frame is 3 wide and 4 tall, so y=3 is
    # on it and names file-order pixel (0, 3)...
    viewer.set_raw_flips(False, False, True)
    info = viewer._compute_cursor_info(QPointF(0.0, 3.0))
    assert (info["row"], info["col"]) == (0, 3)
    assert info["intensity"] == pytest.approx(stack[0, 0, 3])
    # ...while x=3 is past the transposed width, which used to read a
    # pixel that happened to be in bounds of the untransposed frame.
    assert np.isnan(viewer._compute_cursor_info(QPointF(3.0, 0.0))["intensity"])


def test_conversion_panel_emits_raw_flips(qtbot):
    from mlgidgui.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)
    seen: list[tuple[bool, bool, bool]] = []
    panel.rawFlipsChanged.connect(
        lambda lr, ud, tr: seen.append((lr, ud, tr))
    )
    panel.flip_lr.setChecked(True)
    panel.flip_ud.setChecked(True)
    panel.transp.setChecked(True)
    panel.flip_lr.setChecked(False)
    assert seen == [
        (True, False, False),
        (True, True, False),
        (True, True, True),
        (False, True, True),
    ]


def test_transpose_sits_with_the_flips_not_under_manual_overrides(qtbot):
    """It was buried under Manual overrides, where it did nothing live.

    The three share one parent so they read as one decision; the old
    attribute is gone rather than aliased, so nothing can quietly go on
    using it.
    """
    from mlgidgui.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)
    assert panel.transp.parent() is panel.flip_lr.parent()
    assert panel.transp.parent() is panel.flip_ud.parent()
    assert not hasattr(panel, "over_transp")


def test_main_window_wires_flips_to_viewer(main_window):
    panel = main_window.conversion_panel
    viewer = main_window.viewer
    panel.flip_lr.setChecked(True)
    assert viewer._raw_flip_lr is True
    panel.flip_ud.setChecked(True)
    assert viewer._raw_flip_ud is True
    panel.transp.setChecked(True)
    assert viewer._raw_transp is True
    panel.flip_lr.setChecked(False)
    panel.flip_ud.setChecked(False)
    panel.transp.setChecked(False)
    assert viewer._raw_flip_lr is False and viewer._raw_flip_ud is False
    assert viewer._raw_transp is False
