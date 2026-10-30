"""Controls & shortcuts reference (Help menu, F1).

Three pieces: a data-driven inventory of every user control
(``CONTROL_SECTIONS`` — the single source for rendering AND the
completeness tests), a pure theme-aware HTML renderer
(``build_controls_html``, restricted to the CSS subset Qt rich text
actually supports: td background-color + cellpadding/cellspacing do the
"key badge" look, since border-radius and inline padding are ignored),
and the modeless, filterable ``ControlsDialog`` the Help menu opens.
"""

from __future__ import annotations

import html

from mlgidgui import theme_tokens

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# (badges, kind, description-HTML). kind: "key" = monospace chip,
# "mouse" = chip in the default font, "ui" = bold label without a chip.
Row = tuple[tuple[str, ...], str, str]
Section = tuple[str, tuple[Row, ...]]

CONTROL_SECTIONS: tuple[Section, ...] = (
    ("Application & files", (
        (("Ctrl+O",), "key", "Open a data file."),
        (("Ctrl+S", "Ctrl+Shift+S"), "key",
         "Save the active file / save a copy under a new name."),
        (("Ctrl+W",), "key",
         "Close (remove) the active file from the browser."),
        (("Ctrl+Q",), "key", "Quit mlgidGUI."),
        (("F5",), "key",
         "Refresh the file browser (same as its toolbar button): close "
         "files deleted on disk (unsaved changes stay open), reload files "
         "changed on disk (unless they have unsaved changes)."),
        (("Drag & drop",), "mouse",
         "Drop data files anywhere on the window to open them."),
        (("Click", "Double-click"), "mouse",
         "File browser: click activates the file/entry in the viewer; "
         "double-click additionally switches to the Data tab and renders "
         "the node there."),
        (("Delete",), "key",
         "File browser (focused): remove the selected file, same as "
         "Ctrl+W."),
        (("F1",), "key",
         "This reference — type in the filter box to narrow it."),
    )),
    ("Frame navigation", (
        (("← / →", "J / K"), "key",
         "Previous / next frame (J/K is the Vim-style fallback)."),
        (("Home / End",), "key", "First / last frame."),
        (("Ctrl+F",), "key",
         "Find peak by ID — jumps to the peak's frame and selects it."),
        (("F11",), "key",
         "Fullscreen image viewer (hides every dock; F11 again to "
         "leave)."),
    )),
    ("Image display & zoom", (
        (("Wheel",), "mouse", "Zoom in / out."),
        (("Wheel over an axis",), "mouse",
         "Switch Aspect to <b>Custom</b> and adjust the ratio live — "
         "x axis wider, y axis taller."),
        (("Double-click",), "mouse",
         "Reset aspect to Default (per-mode shape) and refit the zoom."),
        (("Right-click",), "mouse",
         "Context menu: <b>Reset zoom</b> plus the standard view "
         "options."),
        (("Drag the contrast bar",), "mouse",
         "Adjust the displayed levels on the histogram; your contrast "
         "choice persists across frame changes."),
        (("Aspect (toolbar)",), "ui",
         "<b>Default</b> uses a per-mode shape (Cartesian 1:1, polar "
         "2:1); <b>Fit</b> fills the panel; <b>Custom</b> locks a "
         "width:height ratio (e.g. <code>2</code> = twice as wide as "
         "tall)."),
    )),
    ("Peaks: select & edit", (
        (("Click a peak",), "mouse",
         "Select it (any kind: manual / detected / fitted / matched)."),
        (("Ctrl+click",), "mouse",
         "Add / remove a <b>detected or fitted</b> peak to the "
         "multi-selection (extras are drawn dashed)."),
        (("Ctrl+A",), "key",
         "Select every peak of the current kind on the frame — fitted "
         "if a fitted peak is primary, otherwise detected."),
        (("↓ / ↑",), "key",
         "Select the next / previous peak of the same kind (detected or "
         "fitted, by id) on the frame, zooming like a click. Spin boxes, "
         "lists and tables keep Up / Down for themselves."),
        (("T",), "key",
         "Switch the selection from a detected peak to its fitted peak "
         "(same id), or back. Needs the peak link in Settings."),
        (("Drag ROI edges",), "mouse",
         "Resize the selected <b>manual or detected</b> peak. Fitted "
         "and matched selections show a box but are not editable."),
        (("Drag profile regions",), "mouse",
         "With a manual or detected peak selected, drag the region "
         "edges in the radial / angular profiles to resize it from the "
         "1D view."),
        (("Ctrl+C", "Ctrl+V"), "key",
         "Copy the selected detected peak(s) / paste them onto the "
         "current frame (same entry only)."),
        (("Ctrl+Shift+V",), "key",
         "Paste copied peaks onto a frame range "
         "(e.g. <code>0-34,40</code>)."),
        (("Delete",), "key",
         "Delete the selected peak(s); a one-kind multi-selection "
         "deletes in one undoable step, file-resident peaks ask for "
         "confirmation."),
        (("Ctrl+Z", "Ctrl+Shift+Z / Ctrl+Y"), "key",
         "Undo / redo manual + geometry edits."),
    )),
    ("Manual peak workflow", (
        (("Ctrl+Alt-drag",), "mouse",
         "Draw a manual peak rectangle (polar mode only)."),
        (("Ctrl+Return",), "key",
         "Add to fitted for the selected manual or detected peak (one "
         "peak only; the status bar says why when it cannot run)."),
        (("Add to detected / Add to fitted",), "ui",
         "Commit the box as a detected peak, or 1D-Gaussian-fit it into "
         "fitted. Tick <b>Save fitted as ring</b> first to widen the "
         "angular extent to the full sweep (ring peaks only)."),
        (("Quick select",), "ui",
         "Label without leaving the image: drawing the next box commits "
         "the previous one as the kind chosen beside the checkbox. A box "
         "drawn on top of the pending one replaces it instead — that is "
         "how you correct an attempt. The last box commits when you "
         "click away, press <b>Enter</b>, change frame or turn the mode "
         "off."),
        (("Esc",), "key",
         "Delete the selected <b>manual</b> peak (in-memory, no "
         "confirmation; Ctrl+Z restores)."),
        (("Click off the box",), "mouse",
         "Abandon the candidate (Ctrl+Z restores it)."),
    )),
    ("Docks & tables", (
        (("Click a Peaks-table row",), "mouse",
         "Select that peak on the image (selecting on the image "
         "highlights the row in return)."),
        (("Display-dock filter",), "ui",
         "Type a CIF substring above the matched-structures list to "
         "hide non-matching rows and their image overlays."),
        (("Click a colour swatch",), "mouse",
         "Matched-peaks legend: pick a colour for that structure from a "
         "grid (plus <b>Automatic</b> and <b>More…</b>); the colour "
         "follows the structure everywhere and is remembered across "
         "sessions."),
        (("Click a scan-tracking row",), "mouse",
         "Jump the viewer to that track's frame and select its peak."),
        (("Delete (tracking table)",), "key",
         "Remove the selected track from the tracking results and from "
         "the tracking table saved in the file. No peaks are deleted."),
    )),
    ("Tracking views", (
        (("Click a trajectory point",), "mouse",
         "Jump the image viewer to that frame."),
        (("Frames: from..to",), "ui",
         "Narrow the trajectories and amplitude-evolution plots to a "
         "frame interval; the q-map and waterfall stay frame-complete."),
    )),
    ("Tools", (
        (("Tools → Export figure…",), "ui",
         "Non-modal window driving "
         "<code>mlgidbase.plot_analysis_results</code> with a live "
         "preview; <b>Render preview</b> updates the image, <b>Save "
         "figure</b> writes the PNG."),
        (("Tools → Clear peaks",), "ui",
         "Reset detected + fitted + matched at three scopes (active "
         "entry / all entries / active frame); manual peaks are dropped "
         "from memory."),
    )),
)

# Hand-picked to sit with the qdarkstyle chrome (#19232d/#dfe1e2 dark,
# #fafafa/#000000 light). Qt rich text cannot read the app palette, so
# the renderer bakes these in and the dialog re-renders on theme flips.
def _palette(theme: str) -> dict[str, str]:
    """The rich-text colours for ``theme``, read from the design tokens.

    Qt rich text cannot resolve the app palette, so the renderer bakes
    hexes into the HTML and re-renders on a theme flip. Those hexes used
    to be a literal table here; they now come from ``theme_tokens`` so
    this dialog cannot drift away from the rest of the chrome.
    """
    t = theme_tokens.tokens(theme)
    return {
        "band_bg": t["band_bg"],
        "band_fg": t["band_fg"],
        "badge_bg": t["badge_bg"],
        "badge_fg": t["badge_fg"],
        "text_fg": t["text"],
        "muted_fg": t["text_muted"],
    }


_COLORS: dict[str, dict[str, str]] = {
    theme: _palette(theme) for theme in ("dark", "light")
}


def _row_matches(row: Row, needle: str) -> bool:
    badges, _kind, description = row
    haystack = " ".join(badges) + " " + description
    return needle in haystack.lower()


def build_controls_html(theme: str, query: str = "") -> str:
    """Render the inventory as Qt-rich-text HTML for ``theme`` ("dark" /
    "light"; unknown falls back to dark), keeping only rows whose badges
    or description contain ``query`` (case-insensitive). Sections left
    empty by the filter are omitted entirely."""
    c = _COLORS.get(theme, _COLORS["dark"])
    needle = query.strip().lower()
    parts: list[str] = []
    for title, rows in CONTROL_SECTIONS:
        kept = [r for r in rows if not needle or _row_matches(r, needle)]
        if not kept:
            continue
        parts.append(
            f"<table width='100%' cellspacing='0' cellpadding='5'>"
            f"<tr><td style='background-color:{c['band_bg']}'>"
            f"<b><span style='color:{c['band_fg']}'>"
            f"{html.escape(title.upper())}</span></b>"
            f"</td></tr></table>"
        )
        row_html: list[str] = []
        for badges, kind, description in kept:
            joiner = (
                f"<span style='color:{c['muted_fg']}'>&nbsp;/&nbsp;</span>"
            )
            badge_text = joiner.join(
                html.escape(b).replace(" ", "&nbsp;") for b in badges
            )
            if kind == "key":
                cell = (
                    f"<td style='background-color:{c['badge_bg']};"
                    f" white-space:nowrap'>"
                    f"<span style='font-family:monospace;"
                    f" color:{c['badge_fg']}'><b>{badge_text}</b></span>"
                    f"</td>"
                )
            elif kind == "mouse":
                cell = (
                    f"<td style='background-color:{c['badge_bg']};"
                    f" white-space:nowrap'>"
                    f"<span style='color:{c['badge_fg']}'>"
                    f"{badge_text}</span></td>"
                )
            else:  # "ui"
                cell = (
                    f"<td style='white-space:nowrap'>"
                    f"<b><span style='color:{c['text_fg']}'>"
                    f"{badge_text}</span></b></td>"
                )
            row_html.append(
                f"<tr>{cell}"
                f"<td style='color:{c['text_fg']}'>{description}</td></tr>"
            )
        parts.append(
            "<table cellspacing='3' cellpadding='4'>"
            + "".join(row_html) + "</table><br>"
        )
    if not parts:
        return (
            f"<p style='color:{c['muted_fg']}'>No controls match "
            f"\"{html.escape(query.strip())}\".</p>"
        )
    return "".join(parts)


class ControlsDialog(QDialog):
    """Modeless Controls & shortcuts reference with a live filter.

    One instance per MainWindow (cached by the host — Close hides it,
    so the filter text, size and position survive re-open). Re-renders
    itself on theme switches via the duck-typed ``apply_theme_colors``
    hook ``MainWindow._set_theme`` already calls on its child views.
    """

    def __init__(
        self, parent: QWidget | None = None, theme: str = "dark"
    ) -> None:
        super().__init__(parent)
        self._theme = theme if theme in _COLORS else "dark"
        self.setWindowTitle("Controls & shortcuts")
        self.resize(700, 560)

        layout = QVBoxLayout(self)
        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText(
            "Type to filter — e.g. paste, F11, track…"
        )
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(lambda _t: self._refresh())
        layout.addWidget(self._filter)
        self._browser = QTextBrowser(self)
        self._browser.setOpenLinks(False)
        self._browser.setOpenExternalLinks(False)
        layout.addWidget(self._browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self._refresh()

    def set_theme(self, theme: str) -> None:
        """Switch the rendered colours (unknown theme -> dark),
        preserving the reader's scroll position."""
        self._theme = theme if theme in _COLORS else "dark"
        bar = self._browser.verticalScrollBar()
        pos = bar.value()
        self._refresh()
        bar.setValue(pos)

    def apply_theme_colors(self, background: str, foreground: str) -> None:
        """Duck-typed hook for ``MainWindow._set_theme``: map the
        pyqtgraph background hex back to the theme name."""
        from mlgidgui.theme import PG_COLORS

        for name, (bg, _fg) in PG_COLORS.items():
            if str(background).lower() == bg.lower():
                self.set_theme(name)
                return
        self.set_theme("dark")

    def focus_filter(self) -> None:
        self._filter.selectAll()
        self._filter.setFocus()

    def _refresh(self) -> None:
        self._browser.setHtml(
            build_controls_html(self._theme, self._filter.text())
        )
