#!/usr/bin/env python
"""Make the committed light logo from the earlier mlgidGUI logo, with "GUI".

    python tools/retypeset_logo.py PATH/TO/old/mlgid_logo.png docs/images/mlgid_logo_mlgidgui.png

The earlier mlgidGUI logo is a raster file reading "mlgidgui". The group
wants the last three letters in capitals, and only the PNG exists, so this
keeps every pixel of the mark, of "mlgid" and of the tagline, erases the
lowercase "gui", and sets "GUI" in its place in the same ink colour.

The numbers below were measured on that PNG (it sits as a 480 x 360 white
box in the middle of a 1200 x 360 transparent canvas): the box, the ink
columns of the old "gui" and of the tagline, the baseline, and the stem
width of the letters (11 px at an em of about 86 px). The lettering is Noto
Sans, which is close to the original face, drawn heavier to match the
original stem width; only the three capitals are new, the rest is untouched.
The tagline stays flush with the right edge of the wordmark, as it was, so
it moves right by as much as the capitals are wider than the lowercase
letters; the canvas grows to keep the same margin on both sides.

Dev-only, run once: the result is committed. Needs PySide6.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import QApplication

CONTENT_BOX = (360, 0, 480, 360)       # the white box inside the old canvas
OLD_GUI_BOX = (283, 196, 133, 96)      # x, y, w, h around the old "gui"
TAGLINE_BOX = (50, 290, 362, 54)       # both tagline lines
OLD_TEXT_RIGHT = 409                   # right ink edge of "gui" and tagline
G_INK_LEFT = 289                       # where the old "g" starts
BASELINE = 266
MARGIN = 44                            # ink margin left of "mlgid"
STEEL = QColor(91, 117, 133)           # the ink of "gui"
EM = 84.0                              # Noto Sans pixel size
STROKE = 2.9                           # added weight, to match the stems


def retypeset(src: Path, dst: Path) -> Path:
    old = QImage(str(src)).copy(*CONTENT_BOX).convertToFormat(
        QImage.Format.Format_ARGB32)
    font = QFont("Noto Sans")
    font.setPixelSize(int(EM))
    path = QPainterPath()
    path.addText(0, BASELINE, font, "GUI")
    path.translate(G_INK_LEFT + STROKE / 2 - path.boundingRect().left(), 0)
    new_right = int(round(path.boundingRect().right() + STROKE / 2))
    shift = new_right - OLD_TEXT_RIGHT
    width = max(old.width(), new_right + MARGIN + 1)

    out = QImage(width, old.height(), QImage.Format.Format_ARGB32)
    out.fill(QColor("white"))
    p = QPainter(out)
    p.drawImage(0, 0, old)
    p.fillRect(*OLD_GUI_BOX, QColor("white"))
    tx, ty, tw, th = TAGLINE_BOX
    p.fillRect(tx, ty, tw, th, QColor("white"))
    p.drawImage(tx + shift, ty, old, tx, ty, tw, th)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(STEEL))
    p.setPen(QPen(STEEL, STROKE, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(path)
    p.end()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst), "PNG")
    return dst


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    _ = QApplication.instance() or QApplication(sys.argv[:1])
    print(retypeset(Path(argv[0]), Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
