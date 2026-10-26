"""The stage strip above the image: where this scan stands.

mlgidLAB drives a linear pipeline — convert, detect, fit, match, track —
but those live in three different dock tabs (Pipeline, Conversion, Scan
tracking), so nothing on screen says what order they go in or which of
them has already run. This is that missing spine.

It deliberately holds no pipeline logic. Clicking a stage raises the dock
that owns it; clicking its run glyph clicks that dock's own Run button,
so every ``PipelineCommand`` is still built in exactly one place, with
the kwargs the panel collected. The rail cannot run something the panel
would refuse: the glyph mirrors that button's enabled state.

Counts are for the **current frame**, and the chips say so. Entry-wide
totals would mean reading peak-table shapes across every frame, which on
a 602-frame scan belongs on a worker thread; the frame the user is
looking at is already in memory.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import icons
from mlgidlab.flow_layout import install_flow
from mlgidlab.widgets import GAP, PAD

import logging

logger = logging.getLogger(__name__)

#: (key, label, glyph). Order is the order of the workflow, which is the
#: whole point of the strip.
STAGES = (
    ("convert", "Convert", "dock-conversion"),
    ("detect", "Detect", "dock-pipeline"),
    ("fit", "Fit", "dock-peaks"),
    ("match", "Match", "dock-expected"),
    ("track", "Track", "dock-tracking"),
)


class _StageChip(QWidget):
    """One stage: name, state line, and a run glyph."""

    activated = Signal(str)
    runRequested = Signal(str)

    def __init__(self, key: str, label: str, glyph: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        self.button = QToolButton(self)
        self.button.setProperty("stage", "chip")
        self.button.setText(label)
        self.button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.clicked.connect(lambda: self.activated.emit(self.key))
        icons.bind(self.button, glyph)
        row.addWidget(self.button)

        self.run_button = QToolButton(self)
        self.run_button.setProperty("stage", "run")
        self.run_button.setToolTip(f"Run {label.lower()}")
        self.run_button.clicked.connect(
            lambda: self.runRequested.emit(self.key))
        icons.bind(self.run_button, "play")
        row.addWidget(self.run_button)

        # The state line sits under the pair, so a chip is two rows tall
        # at most and the strip stays a strip.
        self.state = QLabel("", self)
        self.state.setProperty("status", "muted")

    def set_state(self, text: str, level: str = "muted") -> None:
        self.state.setText(text)
        self.state.setProperty("status", level)
        style = self.state.style()
        style.unpolish(self.state)
        style.polish(self.state)
        self.button.setProperty("state", level)
        style = self.button.style()
        style.unpolish(self.button)
        style.polish(self.button)

    def set_available(self, available: bool, reason: str = "") -> None:
        self.button.setEnabled(available)
        if reason:
            self.button.setToolTip(reason)

    def set_runnable(self, runnable: bool) -> None:
        self.run_button.setEnabled(runnable)


class _StageBlock(QWidget):
    """One stage as the rail wraps it: chip over its state line, and the
    arrow to the next stage trailing it.

    Keeping the arrow *inside* the block means a wrapped row ends with an
    arrow pointing on to the next row, instead of a stray arrow opening a
    row like a bullet.
    """

    def __init__(self, chip: _StageChip, with_arrow: bool,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(GAP)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(chip)
        column.addWidget(chip.state)
        row.addLayout(column)
        #: The trailing arrow, or ``None`` on the last stage. Hidden when
        #: no visible stage follows (``WorkflowRail.set_stage_visible``).
        self.arrow: QLabel | None = None
        if with_arrow:
            self.arrow = QLabel("\u2192", self)
            self.arrow.setProperty("status", "muted")
            row.addWidget(self.arrow)


class WorkflowRail(QWidget):
    """The five stages, left to right, above the central view.

    The row wraps: five stages plus their arrows are wider than a
    narrowed central column, and a strip that cannot wrap would set a
    floor under how far the side docks can be dragged out. Each stage
    wraps as one block — see ``mlgidlab.flow_layout``.
    """

    #: A stage was clicked: bring the dock that owns it forward.
    stageActivated = Signal(str)
    #: A stage's run glyph was clicked.
    stageRunRequested = Signal(str)
    #: The strip was collapsed or expanded, so the choice can be persisted.
    expandedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("mlgid", "rail")
        outer = install_flow(self, margins=(PAD, 2, PAD, 2),
                             hspacing=GAP, vspacing=GAP)

        # The collapse control leads the row, so the strip reads as one
        # thing that can be folded away rather than as chips with a
        # stray button after them. The flow layout skips hidden widgets,
        # so collapsing genuinely reclaims the height.
        self._toggle = QToolButton(self)
        self._toggle.setAutoRaise(True)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setIconSize(QSize(14, 14))
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self.chips: dict[str, _StageChip] = {}
        self._blocks: list[_StageBlock] = []
        #: Stages taken off the strip because they do not apply to the
        #: data (``set_stage_visible``), as opposed to the whole strip
        #: being folded.
        self._hidden_stages: set[str] = set()
        for index, (key, label, glyph) in enumerate(STAGES):
            chip = _StageChip(key, label, glyph, self)
            chip.activated.connect(self.stageActivated)
            chip.runRequested.connect(self.stageRunRequested)
            block = _StageBlock(
                chip, with_arrow=index < len(STAGES) - 1, parent=self)
            outer.addWidget(block)
            self._blocks.append(block)
            self.chips[key] = chip

        # Stands in for the chips when they are folded away, so the
        # remaining row still says what it is.
        self._folded_label = QLabel("Workflow", self)
        self._folded_label.setProperty("status", "muted")
        self._folded_label.hide()
        outer.addWidget(self._folded_label)
        self._apply_expanded()


    # -- collapsing ----------------------------------------------------

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Show or fold the stage chips. Emits ``expandedChanged``."""
        if bool(expanded) != self._toggle.isChecked():
            self._toggle.setChecked(bool(expanded))

    def _on_toggled(self, expanded: bool) -> None:
        self._apply_expanded()
        self.expandedChanged.emit(bool(expanded))

    def _apply_expanded(self) -> None:
        expanded = self._toggle.isChecked()
        shown = [key not in self._hidden_stages for key, _l, _g in STAGES]
        for index, block in enumerate(self._blocks):
            block.setVisible(expanded and shown[index])
            # An arrow only where a visible stage follows, so the strip
            # never ends on one pointing at nothing.
            if block.arrow is not None:
                block.arrow.setVisible(any(shown[index + 1:]))
        self._folded_label.setVisible(not expanded)
        self._toggle.setToolTip(
            "Fold the workflow strip away" if expanded
            else "Show the workflow strip"
        )
        try:
            from mlgidlab import icons
            icons.bind(
                self._toggle,
                "chevron-down" if expanded else "chevron-right",
            )
        except Exception:  # pragma: no cover - a missing asset costs an icon
            logger.debug("no chevron for the workflow rail", exc_info=True)
        self.updateGeometry()

    def set_stage_visible(self, key: str, visible: bool) -> None:
        """Take a stage off the strip, or put it back.

        For a stage that does not apply to the data at all (tracking a
        single frame), unlike ``set_available``, which greys a stage that
        applies later. The previous stage's arrow goes with it.
        """
        if key not in self.chips or visible == self.is_stage_visible(key):
            return
        if visible:
            self._hidden_stages.discard(key)
        else:
            self._hidden_stages.add(key)
        self._apply_expanded()

    def is_stage_visible(self, key: str) -> bool:
        return key in self.chips and key not in self._hidden_stages

    # -- state ---------------------------------------------------------

    def set_state(self, key: str, text: str, level: str = "muted") -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_state(text, level)

    def set_available(self, key: str, available: bool, reason: str = "") -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_available(available, reason)

    def set_runnable(self, key: str, runnable: bool) -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_runnable(runnable)

    def set_mode(self, kind: str | None) -> None:
        """Adapt to the session kind.

        Raw sessions can only be converted; converted (NeXus) sessions
        are past that stage. Unavailable stages stay visible and greyed
        rather than disappearing — the point of the strip is to show the
        whole workflow, including the parts that do not apply yet.
        """
        is_raw = kind == "raw"
        self.set_available("convert", is_raw,
                           "Raw sessions only — this file is already converted"
                           if not is_raw else "")
        for key in ("detect", "fit", "match", "track"):
            self.set_available(
                key, kind == "nexus",
                "Convert this raw scan first" if is_raw else "")
