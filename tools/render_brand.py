#!/usr/bin/env python
"""Regenerate the mlgidGUI brand assets: the app icons and the dark logo.

    python tools/render_brand.py [--out DIR]

Writes:

    mlgidgui/assets/app/mlgidgui_<size>.png          window / taskbar icon
    mlgidgui/assets/app/mlgidgui.ico                 Windows shortcuts, PyInstaller
    docs/images/mlgid_logo_mlgidgui_dark.png         dark-theme logo (README, About)
    mlgidgui/assets/app/mlgid_logo_mlgidgui_dark.png the same, loaded at runtime

With ``--out`` the files are written under DIR (same layout as the repo)
instead of into the repo, to look at a change before it goes in.

**The logo itself is not drawn here.** ``docs/images/mlgid_logo_mlgidgui.png``
is the mlgid-project's mlgidGUI logo (the one of the earlier annotation tool,
cropped to its content, with the last letters set in capitals by
``tools/retypeset_logo.py``; the group chose to keep it for 1.0). It sits on
a white ground. The dark variant is derived from it: every pixel is read as one
of the logo's ink colours over white, which gives back a clean transparent
logo, and the inks are recoloured for a dark ground (``DARK_MAP``).

**The app icon is drawn here** from the family node graph. The geometry is
measured, not eyeballed: the mlgid family logo
(``mlgidBASE/docs/images/mlgid_logo.png``, 1920x1440) was analysed by fitting
a circle to its main arc, centre (887.9, 503.5), r = 397.2 px, and every
constant below is expressed in units of that radius: six nodes with their
interior radii and stroke weight and nine edges with their widths and
colours. The icon is that graph recoloured to the detector colormap.

Dev-only: needs PySide6 (already a runtime dependency) and numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QApplication

REPO = Path(__file__).resolve().parents[1]
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

# --- mlgid family palette (sampled from the shipped family logos) -------
NAVY = "#183c52"     # the arcs
DEEP = "#012037"     # tagline, darkest nodes
CHAR = "#323c3b"     # the "mlgid" stem of the wordmark
STEEL = "#5b7585"    # the package suffix
LSTEEL = "#8d9faa"   # lighter nodes / edges
PALE = "#e5e9eb"     # the faintest edge
#: The ink colours of the logo file, as they are in the PNG (the tagline
#: and the darkest nodes are ``#012037``, the arcs ``#183c52``, ...).
LOGO_INKS = (DEEP, NAVY, CHAR, STEEL, LSTEEL)

# --- magma accents: the app's default detector colormap -----------------
MAGENTA = "#b73779"
ORANGE = "#fc8961"
CREAM = "#fcfdbf"
GROUND = "#19232d"   # matches the dark theme's panel colour

# (dx, dy, interior radius, colour)
FAMILY_NODES = [
    (-0.1194, -0.5565, 0.1448, STEEL),
    (-0.5386, -0.2028, 0.1410, NAVY),
    (+0.5364, -0.2330, 0.1410, DEEP),
    (+0.0228, -0.1398, 0.1422, LSTEEL),
    (-0.3523, +0.3637, 0.1410, NAVY),
    (+0.2129, +0.4228, 0.1448, LSTEEL),
]
NODE_STROKE = 0.0378

# (a, b, colour, width)
FAMILY_EDGES = [
    (0, 1, STEEL, 0.0403), (0, 2, STEEL, 0.0378), (0, 4, NAVY, 0.0378),
    (1, 3, LSTEEL, 0.0378), (1, 5, LSTEEL, 0.0378), (2, 3, LSTEEL, 0.0403),
    (2, 5, LSTEEL, 0.0378), (3, 5, NAVY, 0.0403), (4, 5, PALE, 0.0352),
]

# Family colour -> magma / on-dark equivalents. Both preserve the value
# order: what reads as most prominent on white (darkest navy) must read
# as most prominent on a dark tile (brightest cream).
MAGMA_MAP = {DEEP: CREAM, NAVY: ORANGE, STEEL: "#e05780",
             LSTEEL: MAGENTA, PALE: "#8c2981"}
DARK_MAP = {DEEP: "#e8f0f5", NAVY: "#a9c9dc", CHAR: "#e6e9ea",
            STEEL: "#8fb0c4", LSTEEL: "#5d7d95", PALE: "#41586b"}

def _c(colour: str, palette: dict[str, str] | None) -> QColor:
    return QColor(palette.get(colour, colour) if palette else colour)


def draw_graph(p: QPainter, cx: float, cy: float, r: float, *,
               palette: dict[str, str] | None = None,
               fill: str = "#ffffff") -> None:
    """The family node graph. ``r`` is the main-arc radius."""
    pts = [(cx + dx * r, cy + dy * r) for dx, dy, _, _ in FAMILY_NODES]
    for a, b, colour, width in FAMILY_EDGES:
        p.setPen(QPen(_c(colour, palette), max(1.0, width * r),
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(*pts[a], *pts[b])
    for (x, y), (_, _, rad, colour) in zip(pts, FAMILY_NODES):
        stroke = max(1.0, NODE_STROKE * r)
        rr = rad * r + stroke / 2.0          # traced radius is the interior
        p.setPen(QPen(_c(colour, palette), stroke))
        p.setBrush(QBrush(QColor(fill)))
        p.drawEllipse(QRectF(x - rr, y - rr, 2 * rr, 2 * rr))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _tight_bbox(image: QImage):
    import numpy as np

    w, h = image.width(), image.height()
    arr = np.frombuffer(image.constBits(), dtype=np.uint8)
    arr = arr.reshape(h, image.bytesPerLine() // 4, 4)[:, :w, :]
    alpha = arr[..., 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    if not rows.size or not cols.size:
        return None
    return (int(cols[0]), int(rows[0]),
            int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))


def render_app_icon(path: Path, size: int) -> Path:
    """The window/taskbar icon: the family graph alone, in magma.

    The arcs are dropped because a thin broken ring is the first thing to
    turn to mush at 16 px; the graph is measured, then centred by its own
    ink rather than by hand, so every size sits square in the tile.
    """
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    margin = size * 0.078
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(margin, margin, size - 2 * margin,
                               size - 2 * margin), size * 0.18, size * 0.18)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(GROUND)))
    p.drawPath(tile)
    p.setBrush(Qt.BrushStyle.NoBrush)

    hi = 512
    scratch = QImage(hi, hi, QImage.Format.Format_ARGB32_Premultiplied)
    scratch.fill(Qt.GlobalColor.transparent)
    q = QPainter(scratch)
    q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw_graph(q, hi * 0.5, hi * 0.5, hi * 0.42, palette=MAGMA_MAP, fill=GROUND)
    q.end()

    box = _tight_bbox(scratch)
    if box is not None:
        x, y, w, h = box
        scale = (size * 0.72) / max(w, h)
        dw, dh = w * scale, h * scale
        p.drawImage(QRectF(size / 2 - dw / 2, size / 2 - dh / 2, dw, dh),
                    scratch, QRectF(x, y, w, h))
    p.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PNG")
    return path


def render_ico(path: Path, pngs: list[Path]) -> Path:
    """Multi-size .ico built from the per-size PNGs, each stored as it was
    rendered.

    Pillow's ``save(..., sizes=...)`` resamples ONE image to every size and
    drops the sizes larger than that image, so saving from the 16 px frame
    wrote a 16 px-only icon that Windows then stretched to 32, 48 and 256 px
    (blurry, wrong shapes). Writing the ICONDIR by hand keeps the sharp
    hand-sized frames; PNG entries are valid in .ico since Windows Vista.
    """
    import struct

    entries = []
    for png in sorted(pngs, key=lambda p: QImage(str(p)).width()):
        image = QImage(str(png))
        entries.append((image.width(), image.height(), png.read_bytes()))
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    directory = b""
    for width, height, data in entries:
        # 0 stands for 256 in an ICONDIRENTRY's width and height bytes.
        directory += struct.pack(
            "<BBBBHHII", width % 256, height % 256, 0, 0, 1, 32,
            len(data), offset)
        offset += len(data)
    path.write_bytes(header + directory + b"".join(d for _, _, d in entries))
    return path


def render_dark_logo(src: Path, dst: Path) -> Path:
    """The dark-ground variant of the logo at ``src`` (a logo on white).

    Each pixel of the logo is a mix of one ink colour (``LOGO_INKS``) and
    white, so it can be split back into "which ink" and "how much of it"
    (the alpha). A pixel that equals an ink, within a small tolerance, is
    that ink at full strength. An anti-aliased edge pixel takes the ink of
    its nearest solid neighbours, because a light ink and a part-covered
    dark one can look the same. White becomes transparent. The inks are then
    swapped for their ``DARK_MAP`` colours, keeping the alpha.
    """
    import numpy as np

    image = QImage(str(src)).convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8)
    arr = raw.reshape(h, image.bytesPerLine() // 4, 4)[:, :w, :].astype(float)
    cover = arr[..., 3:4] / 255.0
    rgb = arr[..., :3] * cover + 255.0 * (1.0 - cover)      # on white

    def hex_rgb(colour: str) -> list[int]:
        return [int(colour[i:i + 2], 16) for i in (1, 3, 5)]

    inks = np.array([hex_rgb(c) for c in LOGO_INKS], dtype=float)
    white = np.full(3, 255.0)
    toward_ink = white - inks                                # (k, 3)
    t = np.einsum("hwc,kc->hwk", white - rgb, toward_ink)
    t = np.clip(t / np.einsum("kc,kc->k", toward_ink, toward_ink), 0.0, 1.0)
    mixed = white - t[..., None] * toward_ink                # (h, w, k, 3)
    mix_error = np.linalg.norm(mixed - rgb[..., None, :], axis=-1)
    solid_error = np.linalg.norm(rgb[..., None, :] - inks, axis=-1)

    blank = (white - rgb).max(axis=-1) <= 6.0
    solid = (solid_error.min(axis=-1) <= 8.0) & ~blank
    ink = np.where(solid, solid_error.argmin(axis=-1), -1)
    alpha = np.where(solid, 1.0, 0.0)
    for y, x in zip(*np.nonzero(~solid & ~blank)):
        near = ink[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        candidates = np.unique(near[near >= 0])
        if candidates.size == 0:
            candidates = np.arange(len(LOGO_INKS))
        best = candidates[np.argmin(mix_error[y, x, candidates])]
        ink[y, x] = best
        alpha[y, x] = t[y, x, best]

    out = np.zeros((h, w, 4), dtype=np.uint8)
    for index, colour in enumerate(LOGO_INKS):
        target = hex_rgb(DARK_MAP.get(colour, colour))
        out[ink == index, :3] = target
    out[..., 3] = np.round(alpha * 255).astype(np.uint8)
    result = QImage(out.tobytes(), w, h, w * 4,
                    QImage.Format.Format_RGBA8888).copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(dst), "PNG")
    return dst


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="write under this directory instead of the repo")
    args = parser.parse_args(argv)
    root = args.out or REPO
    app_dir = root / "mlgidgui" / "assets" / "app"
    docs_dir = root / "docs" / "images"

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app

    pngs = [render_app_icon(app_dir / f"mlgidgui_{s}.png", s)
            for s in ICON_SIZES]
    written = [*pngs, render_ico(app_dir / "mlgidgui.ico", pngs)]
    # The logo is a committed file (see the module docstring); the package
    # carries the same bytes, and the dark variant is derived from it.
    light = REPO / "docs" / "images" / "mlgid_logo_mlgidgui.png"
    if not light.is_file():
        raise SystemExit(f"missing {light.relative_to(REPO)}")
    app_dir.mkdir(parents=True, exist_ok=True)
    package_light = app_dir / light.name
    if package_light.resolve() != light.resolve():
        package_light.write_bytes(light.read_bytes())
        written.append(package_light)
    for directory in (docs_dir, app_dir):
        written.append(render_dark_logo(
            light, directory / "mlgid_logo_mlgidgui_dark.png"))
    for path in written:
        print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
