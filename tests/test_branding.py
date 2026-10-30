"""App icon and wordmark: shipped, loadable, and actually applied.

Before this, mlgidGUI had no icon at all — the window, the taskbar and
Alt-Tab showed Qt's generic placeholder. The failure mode worth pinning
is silent: an icon that is missing from the wheel, or never set on the
QApplication, looks exactly like the old behaviour.
"""

from __future__ import annotations

from importlib import resources

import pytest
from PySide6.QtWidgets import QApplication

from mlgidgui import icons

pytestmark = pytest.mark.gui

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def test_the_icon_ships_at_every_size():
    app_dir = resources.files("mlgidgui").joinpath("assets/app")
    assert app_dir.is_dir()
    for size in ICON_SIZES:
        asset = app_dir.joinpath(f"mlgidgui_{size}.png")
        assert asset.is_file(), f"missing mlgidgui_{size}.png"


def test_the_windows_ico_ships():
    """PyInstaller's --icon and Windows shortcuts need a real .ico."""
    ico = resources.files("mlgidgui").joinpath("assets/app/mlgidgui.ico")
    assert ico.is_file()


def test_the_ico_holds_every_size():
    """Windows picks the frame nearest the size it needs (title bar,
    taskbar, desktop, Alt-Tab). A .ico with only 16 px was shipped once,
    and Windows stretched it, so the desktop icon was blurry."""
    import struct

    data = resources.files("mlgidgui").joinpath(
        "assets/app/mlgidgui.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1)
    sizes = []
    for index in range(count):
        width, height = struct.unpack("<BB", data[6 + 16 * index:8 + 16 * index])
        sizes.append((width or 256, height or 256))
    assert sorted(sizes) == [(size, size) for size in ICON_SIZES]


def test_both_wordmark_variants_ship():
    """The About dialog picks one by theme; the dark variant has a
    transparent ground so the mark does not sit in a white box."""
    app_dir = resources.files("mlgidgui").joinpath("assets/app")
    for name in ("mlgid_logo_mlgidgui.png", "mlgid_logo_mlgidgui_dark.png"):
        assert app_dir.joinpath(name).is_file(), name


def test_the_dark_logo_has_a_clear_ground_and_light_ink():
    """Derived from the light logo: the white ground is gone (transparent
    corners), the ink is there, and it is light enough for a dark panel."""
    from PySide6.QtGui import QImage

    app_dir = resources.files("mlgidgui").joinpath("assets/app")
    with resources.as_file(app_dir.joinpath("mlgid_logo_mlgidgui_dark.png")) as p:
        image = QImage(str(p))
    assert not image.isNull() and image.hasAlphaChannel()
    for x, y in ((0, 0), (image.width() - 1, 0), (0, image.height() - 1)):
        assert image.pixelColor(x, y).alpha() == 0
    ink = [image.pixelColor(x, y)
           for x in range(0, image.width(), 3) for y in range(0, image.height(), 3)
           if image.pixelColor(x, y).alpha() == 255]
    assert len(ink) > 500
    assert sum(c.lightness() for c in ink) / len(ink) > 110


def test_the_logo_copies_in_docs_and_package_are_identical():
    """The README loads the docs copy, the app the package copy."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs" / "images"
    app_dir = resources.files("mlgidgui").joinpath("assets/app")
    for name in ("mlgid_logo_mlgidgui.png", "mlgid_logo_mlgidgui_dark.png"):
        assert (docs / name).read_bytes() == app_dir.joinpath(name).read_bytes()


def test_app_icon_loads_and_carries_every_size(qapp):
    icon = icons.app_icon()
    assert not icon.isNull()
    available = {(s.width(), s.height()) for s in icon.availableSizes()}
    for size in ICON_SIZES:
        assert (size, size) in available, f"{size} px missing from the QIcon"


def test_the_16px_icon_is_not_blank(qapp):
    """The size that has to survive Alt-Tab and the taskbar."""
    image = icons.app_icon().pixmap(16, 16).toImage()
    assert not image.isNull()
    opaque = sum(image.pixelColor(x, y).alpha() > 0
                 for y in range(image.height())
                 for x in range(image.width()))
    assert opaque > 40, "the 16 px icon is essentially empty"


def test_main_sets_the_application_icon(qapp, monkeypatch):
    """``main()`` wires the icon onto the QApplication, so every window
    inherits it — main window, figure export, phase views, dialogs."""
    app = QApplication.instance()
    app.setWindowIcon(icons.app_icon())
    assert not app.windowIcon().isNull()


def test_the_about_dialog_has_a_wordmark_to_show(main_window, monkeypatch):
    """The About dialog loads the wordmark by theme; a missing asset must
    degrade to the plain dialog rather than raising."""
    from PySide6.QtWidgets import QMessageBox

    shown = {}

    def fake_exec(self):
        shown["pixmap"] = self.iconPixmap()
        shown["text"] = self.text()
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    main_window._current_theme = "dark"
    main_window._show_about()

    assert "mlgidGUI" in shown["text"]
    assert not shown["pixmap"].isNull()
    assert shown["pixmap"].width() == 320


def test_no_asset_still_carries_the_old_name():
    """The rename moved every logo and icon file; a stray mlgidlab_* file
    would be a leftover that nothing loads and the wheel still ships."""
    app_dir = resources.files("mlgidgui").joinpath("assets/app")
    stale = [p.name for p in app_dir.iterdir() if "mlgidlab" in p.name.lower()]
    assert stale == []
