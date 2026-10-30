"""A toolbar row that wraps instead of forcing a minimum width.

The image viewer's control strip used to be a single ``QHBoxLayout``.
That layout's minimum width is the sum of every control in it, and a
``QHBoxLayout`` has no way to give any of it back — so the strip set a
hard floor on how narrow the central column could get, and dragging a
side dock wider simply stopped at that floor.

``FlowLayout`` lays its items out left to right and starts a new row
when the next one would not fit, so the same controls occupy two rows
when the column is narrow and one row when it is wide. Its minimum
width is therefore the *widest single item*, not the sum.

Two additions over the well-known Qt flow-layout example, both driven
by what the strip actually holds:

* **Empty items are skipped.** The host hides whole clusters (the
  Cartesian/Polar pair in a raw session, the frame transport for a
  single-frame stack). A skipped item leaves no gap and no stray row.
* **Leftover width in a row goes to the items that want it.** The frame
  slider is meant to eat the space its row has left over, which a plain
  flow layout would never hand out.
"""
from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLayout,
    QSizePolicy,
    QWidget,
)

#: Gap between two clusters on the same row, and between two rows.
GROUP_GAP = 16
ROW_GAP = 4


class ToolGroup(QWidget):
    """A cluster of toolbar controls that wraps as one unit.

    Wrapping between, say, the "Colormap:" label and its combo box
    would read as a layout bug, so the strip is built from clusters and
    the flow layout only ever breaks between them.

    A cluster whose controls are all hidden reports an empty size hint,
    which is how :class:`FlowLayout` knows to leave it out entirely
    rather than reserve a sliver of row for it.
    """

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(spacing)

    @property
    def row(self) -> QHBoxLayout:
        return self._row

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._row.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._row.addLayout(layout)

    def has_visible_members(self) -> bool:
        return any(
            isinstance(child, QWidget) and not child.isHidden()
            for child in self.children()
        )

    def sizeHint(self) -> QSize:  # noqa: N802  (Qt casing)
        if not self.has_visible_members():
            return QSize(0, 0)
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802  (Qt casing)
        if not self.has_visible_members():
            return QSize(0, 0)
        return super().minimumSizeHint()


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto further rows as needed."""

    def __init__(
        self,
        parent: QWidget | None = None,
        margins: tuple[int, int, int, int] = (0, 0, 0, 0),
        hspacing: int = GROUP_GAP,
        vspacing: int = ROW_GAP,
    ) -> None:
        super().__init__(parent)
        self._items: list = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(QMargins(*margins))

    # -- QLayout plumbing -------------------------------------------------
    def addItem(self, item) -> None:  # noqa: N802  (Qt casing)
        self._items.append(item)

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802
        self.addWidget(widget)                 # parents + creates the item
        item = self._items.pop()
        self._items.insert(max(0, min(index, len(self._items))), item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802  (Qt casing)
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802  (Qt casing)
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        # Nothing to report upwards: the layout hands leftover width to
        # its own expanding children row by row (see ``_place_row``).
        return Qt.Orientations(0)

    # -- geometry ---------------------------------------------------------
    def invalidate(self) -> None:
        """Tell the host widget its height-for-width may have changed.

        Qt decides whether to re-run the parent layout by comparing size
        *hints*, and this layout's hint is always one row tall — only
        ``heightForWidth`` changes when a cluster appears or disappears.
        Without this nudge, hiding the frame transport left the strip
        two rows tall until the next resize.
        """
        super().invalidate()
        holder = self.parentWidget()
        if holder is not None:
            holder.updateGeometry()

    def hasHeightForWidth(self) -> bool:  # noqa: N802  (Qt casing)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802  (Qt casing)
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802  (Qt casing)
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802  (Qt casing)
        """One row of everything — what the strip wants when it can."""
        margins = self.contentsMargins()
        width = height = 0
        visible = [item for item in self._items if not self._is_empty(item)]
        for item in visible:
            hint = item.sizeHint()
            width += hint.width()
            height = max(height, hint.height())
        width += self._hspacing * max(0, len(visible) - 1)
        return QSize(
            width + margins.left() + margins.right(),
            height + margins.top() + margins.bottom(),
        )

    def minimumSize(self) -> QSize:  # noqa: N802  (Qt casing)
        """The widest single cluster — the whole point of this layout."""
        margins = self.contentsMargins()
        size = QSize(0, 0)
        for item in self._items:
            if self._is_empty(item):
                continue
            size = size.expandedTo(item.minimumSize())
        return QSize(
            size.width() + margins.left() + margins.right(),
            size.height() + margins.top() + margins.bottom(),
        )

    # -- internals --------------------------------------------------------
    @staticmethod
    def _is_empty(item) -> bool:
        widget = item.widget()
        if widget is not None and widget.isHidden():
            return True
        return item.sizeHint().isEmpty()

    @staticmethod
    def _collapse(item, area: QRect) -> None:
        """Shrink a skipped item to nothing instead of leaving it be.

        A layout never calls ``setGeometry`` on an item it skips, so the
        widget keeps whatever geometry it had — for one that has never
        been laid out, Qt's default 640x480 at the parent's origin. It is
        still visible and still hit-tested, so an empty cluster sat on
        top of the strip and swallowed every click on the controls
        underneath it: on a single-frame stack the whole frame transport
        is hidden, and its cluster then covered Cartesian, Polar and the
        colormap picker. Opening a second, multi-frame file appeared to
        fix it, because that gave the cluster real content and a real
        place to be.

        Collapsing rather than hiding: this touches geometry only, so a
        cluster that fills up again is laid out normally on the next
        pass, with no visibility state to get stuck in.
        """
        widget = item.widget()
        if widget is not None and not widget.geometry().isEmpty():
            widget.setGeometry(QRect(area.x(), area.y(), 0, 0))

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = area.x()
        y = area.y()
        row: list[tuple[object, int, QSize]] = []
        row_height = 0

        for item in self._items:
            if self._is_empty(item):
                if not test_only:
                    self._collapse(item, area)
                continue
            hint = item.sizeHint()
            if row and x + hint.width() > area.right() + 1:
                self._place_row(row, y, row_height, area, test_only)
                x = area.x()
                y += row_height + self._vspacing
                row = []
                row_height = 0
            row.append((item, x, hint))
            x += hint.width() + self._hspacing
            row_height = max(row_height, hint.height())

        if row:
            self._place_row(row, y, row_height, area, test_only)
        return y + row_height + margins.bottom() - rect.y()

    def _place_row(self, row, y: int, height: int, area: QRect,
                   test_only: bool) -> None:
        if test_only:
            return
        last_item, last_x, last_hint = row[-1]
        spare = area.right() + 1 - (last_x + last_hint.width())
        growers = [
            index for index, (item, _, _) in enumerate(row)
            if item.expandingDirections() & Qt.Orientation.Horizontal
        ]
        share = spare // len(growers) if growers and spare > 0 else 0

        offset = 0
        for index, (item, x, hint) in enumerate(row):
            width = hint.width()
            if index in growers:
                width += share
            item.setGeometry(
                QRect(
                    QPoint(x + offset, y + (height - hint.height()) // 2),
                    QSize(width, hint.height()),
                )
            )
            if index in growers:
                offset += share


def install_flow(widget: QWidget,
                 margins: tuple[int, int, int, int] = (0, 0, 0, 0),
                 hspacing: int = GROUP_GAP,
                 vspacing: int = ROW_GAP) -> FlowLayout:
    """Give ``widget`` a :class:`FlowLayout` and the policy it needs.

    The height-for-width flag is what lets the parent column give the
    strip a second row: without it Qt asks the widget for one height and
    clips whatever wraps below it.
    """
    layout = FlowLayout(widget, margins=margins, hspacing=hspacing,
                        vspacing=vspacing)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred,
                         QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)
    return layout


def wrapping_bar(parent: QWidget | None = None,
                 margins: tuple[int, int, int, int] = (8, 4, 8, 4)) -> QWidget:
    """A widget hosting a :class:`FlowLayout`, sized by its own content."""
    holder = QWidget(parent)
    install_flow(holder, margins=margins)
    return holder
