"""Theme-aware SVG icons.

The shipped glyphs (``mlgidgui/assets/icons/*.svg``) are monochrome and
authored with ``stroke="currentColor"``. Qt's SVG renderer does not
resolve ``currentColor`` against the widget palette, so this module does
the substitution itself: it reads the SVG text, swaps in a hex from
``theme_tokens``, and rasterises the result at several sizes into one
``QIcon``. One source asset therefore stays crisp at every size and
adapts to both themes.

Two things a naive loader gets wrong, and how this one handles them:

* **Stale icons after a theme flip.** ``QIcon`` is a value type — clearing
  a cache does nothing for icons already handed to widgets. ``bind()``
  records the widget (weakly) against its glyph name and ``retheme()``
  re-applies them, so a theme switch walks ~50 registered targets rather
  than every widget in the app.
* **A broken install taking the window with it.** Every lookup failure
  returns a null ``QIcon``; a missing asset costs an icon, never a
  traceback. ``QToolButton`` renders text-only with a null icon, which is
  exactly what these buttons did before they had icons at all.
"""
from __future__ import annotations

import logging
import weakref
from functools import lru_cache
from importlib import resources

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from mlgidgui import theme_tokens

logger = logging.getLogger(__name__)

#: Rasterised sizes. Qt picks the closest and scales down, so the largest
#: entry is what keeps a HiDPI toolbar crisp.
SIZES = (16, 20, 24, 32, 48, 64)

_ASSET_DIR = "assets/icons"

# name -> QIcon, keyed by the resolved colour so both themes can be warm
# at once. Cleared by retheme().
_icon_cache: dict[tuple[str, str], QIcon] = {}

# Weak references to widgets/actions that should be repainted on a theme
# change, each with the glyph name it was bound to.
_bindings: "weakref.WeakKeyDictionary[object, str]" = weakref.WeakKeyDictionary()


@lru_cache(maxsize=None)
def _svg_text(name: str) -> str | None:
    """The raw SVG for ``name``, or None if it is not shipped."""
    try:
        path = resources.files("mlgidgui").joinpath(f"{_ASSET_DIR}/{name}.svg")
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError, AttributeError):
        logger.debug("icon asset missing: %s", name, exc_info=True)
        return None


def available() -> list[str]:
    """Every glyph name that ships with the package."""
    try:
        directory = resources.files("mlgidgui").joinpath(_ASSET_DIR)
        return sorted(p.name[:-4] for p in directory.iterdir()
                      if p.name.endswith(".svg"))
    except (FileNotFoundError, ModuleNotFoundError, OSError, AttributeError):
        logger.debug("icon asset directory unreadable", exc_info=True)
        return []


def _pixmap(svg: str, color: str, size: int) -> QPixmap:
    data = QByteArray(svg.replace("currentColor", color).encode("utf-8"))
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    QSvgRenderer(data).render(painter, QRectF(0, 0, size, size))
    painter.end()
    return QPixmap.fromImage(image)


def icon(name: str, *, color: str | None = None, theme: str | None = None) -> QIcon:
    """The glyph ``name`` painted for the live theme (or ``theme``).

    ``color`` overrides the token lookup — used where an icon sits on a
    coloured ground rather than a panel (the update banner).
    """
    resolved = color or theme_tokens.color("text", theme)
    key = (name, resolved)
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached

    svg = _svg_text(name)
    if svg is None:
        result = QIcon()
    else:
        result = QIcon()
        for size in SIZES:
            result.addPixmap(_pixmap(svg, resolved, size))
    _icon_cache[key] = result
    return result


def app_icon() -> QIcon:
    """The window / taskbar icon, assembled from the shipped PNGs.

    Pre-rendered rather than rasterised at startup: the mark is a
    coloured drawing, and pre-rendering keeps it identical across Qt
    versions while costing nothing on the cold path.
    """
    result = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        try:
            path = resources.files("mlgidgui").joinpath(
                f"assets/app/mlgidgui_{size}.png")
            with resources.as_file(path) as real:
                result.addFile(str(real))
        except (FileNotFoundError, ModuleNotFoundError, OSError, AttributeError):
            logger.debug("app icon missing at %d px", size, exc_info=True)
    return result


def bind(target, name: str) -> None:
    """Set ``target``'s icon and re-apply it on every later theme change.

    Accepts anything with ``setIcon`` (``QAction``, ``QAbstractButton``).
    """
    setter = getattr(target, "setIcon", None)
    if setter is None:
        return
    setter(icon(name))
    _bindings[target] = name


def unbind(target) -> None:
    _bindings.pop(target, None)


def retheme(theme: str | None = None) -> int:
    """Repaint every bound icon for ``theme``; returns how many.

    Called from the theme switch. The cache is left in place — it is
    keyed by colour, so the other theme's icons stay warm for a flip
    back.
    """
    count = 0
    for target, name in list(_bindings.items()):
        setter = getattr(target, "setIcon", None)
        if setter is None:
            continue
        try:
            setter(icon(name, theme=theme))
            count += 1
        except RuntimeError:
            # The underlying C++ object went away between the weakref
            # check and the call; nothing to repaint.
            _bindings.pop(target, None)
    return count


def clear_cache() -> None:
    """Drop every cached icon (and the SVG text cache). Tests use this;
    the app has no reason to."""
    _icon_cache.clear()
    _svg_text.cache_clear()
