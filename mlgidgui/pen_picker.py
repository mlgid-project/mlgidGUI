"""Compact pen editor popup: colour, line style and line width.

A small frameless ``Qt.Popup`` shown under a legend swatch: a 4 x 10
grid of preset colours (the matched palette plus lighter/darker shades),
a row of line-style buttons, a width spinbox, an "Automatic" button that
returns the layer to its preset, and "More..." for the full
``QColorDialog``.

The widget knows nothing about structures or overlays -- it emits the
partial pen the user has set and the caller applies it.

**It stays open while you edit.** The colour-only version it replaces
closed on the first click, which was right when a click WAS the whole
choice; with three controls, closing after the colour would mean
reopening for the width. Every change emits ``penChanged`` immediately
so the image follows live, and clicking anywhere outside dismisses
(``Qt.Popup``).

The pen it emits is PARTIAL: only the properties the user actually
touched are in it, so a structure whose width was raised keeps cycling
through the automatic palette for its colour.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidgui import theme_tokens
from mlgidgui.image_viewer import MATCHED_PALETTE
from mlgidgui.viewer_styles import (
    PEN_STYLES,
    PEN_WIDTH_MAX,
    PEN_WIDTH_MIN,
)
from mlgidgui.widgets import make_pen_swatch

_GRID_COLUMNS = len(MATCHED_PALETTE)
_SWATCH_PX = 18


def grid_colors() -> list[str]:
    """The popup's preset hexes: the matched palette row, then a lighter
    and two darker shade rows of the same hues (40 in total)."""
    shades = (
        lambda c: c,
        lambda c: c.lighter(140),
        lambda c: c.darker(130),
        lambda c: c.darker(180),
    )
    return [
        shade(QColor(base)).name()
        for shade in shades
        for base in MATCHED_PALETTE
    ]


class PenPopup(QWidget):
    """Pen chooser popup.

    ``penChanged(dict)`` carries the partial pen after every edit;
    ``resetPicked()`` means "back to automatic" and closes. Clicking
    outside dismisses without a further signal (``Qt.Popup``).

    ``current`` is the pen being shown (the effective one, so the
    controls start where the eye is); ``override`` is the part of it the
    user had already set, which is what gets extended and re-emitted.
    """

    penChanged = Signal(dict)
    resetPicked = Signal()
    #: Colour-only convenience, kept so callers that only care about the
    #: hue can connect one signal.
    colorPicked = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        current: dict | None = None,
        override: dict | None = None,
        title: str = "Colour",
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._current = dict(current or {})
        self._override = dict(override or {})
        self._title = title

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(2)
        self._swatch_buttons: list[QToolButton] = []
        for i, hex_color in enumerate(grid_colors()):
            btn = QToolButton(self)
            btn.setFixedSize(_SWATCH_PX, _SWATCH_PX)
            btn.setToolTip(hex_color)
            btn.setStyleSheet(
                f"background-color: {hex_color};"
                f" border: 1px solid {theme_tokens.color('border')};"
            )
            btn.clicked.connect(lambda _=False, c=hex_color: self._pick(c))
            grid.addWidget(btn, i // _GRID_COLUMNS, i % _GRID_COLUMNS)
            self._swatch_buttons.append(btn)
        box.addLayout(grid)

        # Line style: one button per style, each drawing the style it
        # sets. A combobox of the words "Dash-dot-dot" would make the
        # user read what they can simply be shown.
        styles = QHBoxLayout()
        styles.setSpacing(4)
        styles.addWidget(QLabel("Line"))
        self._style_buttons: dict[str, QToolButton] = {}
        for name, label, value in PEN_STYLES:
            btn = QToolButton(self)
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setToolTip(label)
            pix = make_pen_swatch(
                {"color": self._pen_color(), "style": value,
                 "width": self._pen_width()}
            )
            btn.setIcon(QIcon(pix))
            btn.setIconSize(pix.size())
            btn.clicked.connect(lambda _=False, n=name: self._pick_style(n))
            styles.addWidget(btn)
            self._style_buttons[name] = btn
        styles.addStretch(1)
        box.addLayout(styles)

        width_row = QHBoxLayout()
        width_row.setSpacing(4)
        width_row.addWidget(QLabel("Width"))
        self._width_spin = QDoubleSpinBox(self)
        self._width_spin.setRange(PEN_WIDTH_MIN, PEN_WIDTH_MAX)
        self._width_spin.setSingleStep(0.2)
        self._width_spin.setDecimals(1)
        self._width_spin.setValue(self._pen_width())
        self._width_spin.valueChanged.connect(self._pick_width)
        width_row.addWidget(self._width_spin)
        width_row.addStretch(1)
        box.addLayout(width_row)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._auto_button = QPushButton("Automatic")
        self._auto_button.setToolTip(
            "Back to the automatically assigned colour, line style and width"
        )
        self._auto_button.clicked.connect(self._reset)
        actions.addWidget(self._auto_button)
        self._more_button = QPushButton("More...")
        self._more_button.setToolTip("Pick any colour")
        self._more_button.clicked.connect(self._more)
        actions.addWidget(self._more_button)
        box.addLayout(actions)

        self._sync_controls()

    # --- reading the pen being edited ---

    def _pen_color(self) -> str:
        return str(self._current.get("color") or "#ffffff")

    def _pen_width(self) -> float:
        try:
            return float(self._current.get("width", 1.6))
        except (TypeError, ValueError):
            return 1.6

    def _pen_style_name(self) -> str:
        from mlgidgui.viewer_styles import pen_style_name

        style = self._current.get("style")
        return pen_style_name(style) if style is not None else PEN_STYLES[0][0]

    def _sync_controls(self) -> None:
        """Point the style buttons at the pen currently in effect.

        Each button previews its own style in the pen's CURRENT colour
        and width, so the row doubles as a preview of what picking it
        would give.
        """
        active = self._pen_style_name()
        for name, _label, value in PEN_STYLES:
            btn = self._style_buttons[name]
            pix = make_pen_swatch(
                {"color": self._pen_color(), "style": value,
                 "width": self._pen_width()}
            )
            btn.setIcon(QIcon(pix))
            btn.setIconSize(pix.size())
            btn.setChecked(name == active)

    # --- edits ---

    def _emit(self) -> None:
        self.penChanged.emit(dict(self._override))

    def _pick(self, hex_color: str) -> None:
        self._current["color"] = hex_color
        self._override["color"] = hex_color
        self._sync_controls()
        self.colorPicked.emit(hex_color)
        self._emit()

    def _pick_style(self, name: str) -> None:
        from mlgidgui.viewer_styles import pen_style_from_name

        style = pen_style_from_name(name)
        self._current["style"] = style
        self._override["style"] = style
        self._sync_controls()
        self._emit()

    def _pick_width(self, value: float) -> None:
        self._current["width"] = float(value)
        self._override["width"] = float(value)
        self._sync_controls()
        self._emit()

    def show_under(self, widget: QWidget) -> None:
        """Open the popup just below ``widget`` (the clicked swatch)."""
        self.adjustSize()
        self.move(widget.mapToGlobal(QPoint(0, widget.height())))
        self.show()

    def _reset(self) -> None:
        self.resetPicked.emit()
        self.close()

    def _more(self) -> None:
        # Hide first: a Qt.Popup would dismiss itself the moment the
        # modal dialog grabs focus, taking this slot's widget with it
        # (WA_DeleteOnClose). Hidden, it survives until we show() again.
        self.hide()
        color = QColorDialog.getColor(
            QColor(self._pen_color()), self.parentWidget(), self._title,
        )
        if color.isValid():
            self._pick(color.name())
        # Back to the editor rather than closing: the user may still
        # want to set the width, which is the whole reason this popup
        # no longer closes on a pick.
        self.show()
