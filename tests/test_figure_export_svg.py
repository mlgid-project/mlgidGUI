"""Figure export supports SVG (vector) alongside PNG (raster).

The format follows the chosen file extension: mlgidbase preserves it
(`os.path.splitext` → re-appends the same suffix to the `_fr_`/`_sol_`
variants) and matplotlib's `savefig` keys the writer off it. So the GUI
only has to (a) offer `.svg` in the Save dialog and stop forcing `.png`,
and (b) glob the chosen suffix when collecting the written files.

`_modified_save_paths` is pure (no Qt/mlgidbase). The `_browse_save_path`
test constructs the window with `_render` stubbed out so the
construction-time `QTimer.singleShot(0, self._render)` can't fire a real
(mlgidbase-dependent) render.
"""

from __future__ import annotations

import pytest

from mlgidgui import file_dialogs
from mlgidgui.figure_export_window import FigureExportWindow, _modified_save_paths

pytestmark = pytest.mark.gui


def test_modified_save_paths_preserves_svg_suffix(tmp_path):
    """The written-file collector globs the user's extension, so an SVG
    base finds the SVG variant and ignores a same-stem PNG."""
    (tmp_path / "figure_entry_0000_fr_0000.svg").write_text("x")
    (tmp_path / "figure_entry_0000_fr_0000.png").write_text("x")  # decoy

    res = _modified_save_paths(str(tmp_path / "figure.svg"), "entry_0000", 0)
    assert [p.name for p in res] == ["figure_entry_0000_fr_0000.svg"]

    res_png = _modified_save_paths(str(tmp_path / "figure.png"), "entry_0000", 0)
    assert [p.name for p in res_png] == ["figure_entry_0000_fr_0000.png"]


def test_browse_save_path_handles_png_and_svg(main_window, qtbot, monkeypatch):
    """The Save dialog appends the extension implied by the chosen filter
    and preserves an explicit .png/.svg the user typed."""
    monkeypatch.setattr(FigureExportWindow, "_render", lambda self: None)
    win = FigureExportWindow(main_window)
    qtbot.addWidget(win)

    def _dialog(ret):
        # Every picker is built by ``file_dialogs`` now, so that is the
        # seam to fake — patching ``QFileDialog`` statics would leave a
        # real modal dialog to block on.
        monkeypatch.setattr(
            file_dialogs, "save_file", lambda *a, **k: ret
        )

    # No extension + SVG filter -> .svg
    _dialog(("/tmp/fig", "SVG image (*.svg)"))
    win._browse_save_path()
    assert win.path_edit.text() == "/tmp/fig.svg"

    # No extension + PNG filter -> .png
    _dialog(("/tmp/fig2", "PNG image (*.png)"))
    win._browse_save_path()
    assert win.path_edit.text() == "/tmp/fig2.png"

    # Explicit .svg is kept even if the PNG filter was active
    _dialog(("/tmp/fig3.svg", "PNG image (*.png)"))
    win._browse_save_path()
    assert win.path_edit.text() == "/tmp/fig3.svg"
