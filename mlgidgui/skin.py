"""The mlgidGUI skin: a QSS layer built from ``theme_tokens``, appended
after the qdarkstyle sheet in ``theme._apply``.

Scoping rule — the thing that keeps silx, pyFAI and pyqtgraph untouched
-----------------------------------------------------------------------
Qt matches QSS descendant selectors by walking ``QWidget::parentWidget()``,
and that walk does **not** stop at window boundaries. Verified:

    app.setStyleSheet('[mlgidPanel="true"] QPushButton { color: #ff0000; }')
    dlg = QDialog(panel)     # a separate window, merely parented
    -> a QPushButton inside dlg renders #ff0000

``CalibrationDialog(self, ...)`` (conversion_panel.py) is exactly that
shape, and it hosts five pyFAI task widgets; the file browser dock hosts
silx's ``Hdf5TreeView``. So panel-root markers, class-name scoping and
dock-objectName scoping are all unsafe, however tidy they look.

Every rule below is therefore scoped to the **leaf widget itself**, via an
objectName or a dynamic property that this application sets:

    QPushButton[variant="primary"]     QToolButton#SectionHeader
    QLabel[status="error"]             QLabel[role="sb-cell"]

The single exception is a descendant rooted at a widget we tagged that
never parents a dialog (a table and its own header). ``tests/
test_skin_qss.py`` enforces this, so the rule cannot rot.

Property values are always strings — ``setProperty("mlgid", "table")``,
never a bool: Qt stringifies attribute comparisons and the bool spelling
has differed between Qt versions.

Nothing opts in yet; adopting call sites is the following stage.
"""
from __future__ import annotations

from mlgidgui import theme_tokens

#: Tagged as ``widget.setProperty("variant", …)``.
PRIMARY = "primary"
DANGER = "danger"

#: objectName for the CollapsibleSection header button.
SECTION_HEADER = "SectionHeader"

#: ``role`` value on a non-collapsible Card's title label.
CARD_TITLE = "card-title"

#: The one descendant-scoped rule, kept in a constant so the guard test
#: and the sheet cannot disagree about what is allowed.
TABLE_SELECTOR = 'QTableView[mlgid="table"]'

#: Progress bars this application owns (``widgets.skin_progress``).
PROGRESS_SELECTOR = 'QProgressBar[mlgid="progress"]'

#: objectName for a bordered box around one region of a split workspace
#: (``widgets.boxed``). A ``Card``'s only chrome is a hairline under its
#: title, which reads correctly stacked inside a dock and reads as a
#: line stopping in mid-air when the section is a pane of its own.
#:
#: An objectName rather than the usual dynamic property, because
#: qdarkstyle ships ``.QFrame[frameShape="0"] { border: 1px transparent }``
#: — a class selector plus an attribute, which outranks a type selector
#: plus an attribute, so ``QFrame[mlgid="region"]`` loses and the border
#: silently paints transparent. An id selector outranks both.
REGION_NAME = "MlgidRegion"
REGION_SELECTOR = f'QFrame#{REGION_NAME}'

#: objectName for the Structure header's inline-rename field
#: (``structure_panel._NameEdit``). A ``QLineEdit`` that has to read as
#: part of the title until it is being edited, so it borrows the
#: card-title weight and loses every bit of chrome a line edit normally
#: has. An objectName again, for the reason above: qdarkstyle styles
#: ``QLineEdit`` heavily, and a plain property selector on the same type
#: would be a coin toss.
NODE_NAME = "MlgidNodeName"
NODE_NAME_SELECTOR = f'QLineEdit#{NODE_NAME}'

#: ``role`` on the muted half of that header — where the node lives, as
#: opposed to what it is called. Muted and upright: it is a path, and
#: the italics the other muted labels use read as prose.
NODE_PATH = "node-path"


def marker(theme: str) -> str:
    """The comment stamped at the top of the sheet, so a test (or a bug
    report) can tell which theme's skin is live."""
    return f"/* mlgid-skin {theme} */"


def build_qss(theme: str) -> str:
    """The skin layer for ``theme``. Appended to the qdarkstyle sheet."""
    t = theme_tokens.tokens(theme)
    return f"""{marker(theme)}

/* --- primary action: accent border and label, transparent ground ----
   Outlined rather than filled so a dock full of controls keeps one
   obvious action without a block of colour fighting the detector
   image next to it. */
QPushButton[variant="{PRIMARY}"] {{
    background-color: transparent;
    color: {t['accent']};
    border: 1px solid {t['accent']};
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
}}
QPushButton[variant="{PRIMARY}"]:hover {{
    background-color: {t['accent_soft']};
    color: {t['accent_hover']};
    border-color: {t['accent_hover']};
}}
QPushButton[variant="{PRIMARY}"]:pressed {{
    background-color: {t['accent_soft']};
    color: {t['accent_pressed']};
    border-color: {t['accent_pressed']};
}}
QPushButton[variant="{PRIMARY}"]:disabled {{
    background-color: transparent;
    color: {t['text_disabled']};
    border-color: {t['border']};
}}

/* --- destructive action: a red edge, never a red fill ---------------
   Reserved for operations that lose data. A filled red button reads as
   "danger" even while idle, which is wrong for a control the user is
   expected to press deliberately. */
QPushButton[variant="{DANGER}"] {{
    color: {t['danger']};
    border: 1px solid {t['border']};
    border-radius: 4px;
    padding: 4px 12px;
}}
QPushButton[variant="{DANGER}"]:hover {{
    color: {t['danger_hover']};
    border-color: {t['danger']};
}}
QPushButton[variant="{DANGER}"]:disabled {{
    color: {t['text_disabled']};
    border-color: {t['border']};
}}

/* --- section header (CollapsibleSection's own button) ----------------
   qdarkstyle ships 19 QToolButton rules, so the background and border
   must be restated explicitly or the 13 headers grow button chrome. */
QToolButton#{SECTION_HEADER} {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {t['divider']};
    border-radius: 0px;
    padding: 6px 2px 5px 2px;
    font-weight: bold;
    color: {t['text']};
    text-align: left;
}}
QToolButton#{SECTION_HEADER}:hover {{
    color: {t['accent']};
    border-bottom: 1px solid {t['accent']};
}}
QToolButton#{SECTION_HEADER}:checked {{
    color: {t['text']};
}}

/* --- card title (a section header without a chevron) ------------------
   Same weight, padding and hairline as the collapsible header above, so
   a fixed section and a collapsible one read as the same furniture. */
QLabel[role="{CARD_TITLE}"] {{
    border-bottom: 1px solid {t['divider']};
    padding: 6px 2px 5px 2px;
    font-weight: bold;
    color: {t['text']};
}}

/* --- segmented pair (viewer: Cartesian | Polar) ----------------------
   One exclusive choice shown as two joined halves. The inner border is
   dropped on the left half so the two meet on a single shared edge
   rather than showing a double rule. */
QToolButton[segment="left"], QToolButton[segment="right"] {{
    background: transparent;
    border: 1px solid {t['border']};
    padding: 3px 10px;
    color: {t['text_muted']};
}}
QToolButton[segment="left"] {{
    border-right: none;
    border-top-left-radius: 4px;
    border-bottom-left-radius: 4px;
}}
QToolButton[segment="right"] {{
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}
QToolButton[segment="left"]:checked, QToolButton[segment="right"]:checked {{
    background-color: {t['accent_soft']};
    color: {t['accent']};
    font-weight: bold;
}}
QToolButton[segment="left"]:hover, QToolButton[segment="right"]:hover {{
    color: {t['accent_hover']};
}}
QToolButton[segment="left"]:disabled, QToolButton[segment="right"]:disabled {{
    color: {t['text_disabled']};
    border-color: {t['divider']};
}}

/* --- on/off toggle (viewer: Log scale) -------------------------------
   A mode the image is in, not a form field, so it reads as a pressed
   button rather than a tick. */
QToolButton[variant="toggle"] {{
    background: transparent;
    border: 1px solid {t['border']};
    border-radius: 4px;
    padding: 3px 8px;
    color: {t['text_muted']};
}}
QToolButton[variant="toggle"]:checked {{
    background-color: {t['accent_soft']};
    border-color: {t['accent']};
    color: {t['accent']};
    font-weight: bold;
}}
QToolButton[variant="toggle"]:hover {{
    color: {t['accent_hover']};
}}
QToolButton[variant="toggle"]:disabled {{
    color: {t['text_disabled']};
    border-color: {t['divider']};
}}

/* --- workflow rail ---------------------------------------------------
   Each stage is a flat chip; the one that has produced something for
   this frame carries the accent, so the strip reads as a progress line
   rather than five identical buttons. */
QWidget[mlgid="rail"] {{
    background-color: {t['surface_raised']};
    border-bottom: 1px solid {t['divider']};
}}
QToolButton[stage="chip"] {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 3px 8px;
    color: {t['text']};
}}
QToolButton[stage="chip"]:hover {{
    background-color: {t['accent_soft']};
    color: {t['accent_hover']};
}}
QToolButton[stage="chip"][state="ok"] {{
    color: {t['accent']};
    font-weight: bold;
}}
QToolButton[stage="chip"][state="run"] {{
    color: {t['accent']};
    font-weight: bold;
}}
QToolButton[stage="chip"]:disabled {{
    color: {t['text_disabled']};
}}
QToolButton[stage="run"] {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 2px;
}}
QToolButton[stage="run"]:hover {{
    background-color: {t['accent_soft']};
}}

/* --- semantic status text -------------------------------------------
   Replaces the hardcoded per-state hexes scattered through the panels,
   which never followed a theme flip. */
QLabel[status="muted"] {{ color: {t['text_muted']}; font-style: italic; }}
QLabel[status="ok"] {{ color: {t['status_ok']}; font-style: italic; }}
QLabel[status="warn"] {{ color: {t['status_warn']}; font-style: italic; }}
QLabel[status="error"] {{ color: {t['status_error']}; font-style: italic; }}
QLabel[status="info"] {{ color: {t['status_info']}; font-style: italic; }}
QLabel[role="section"] {{ color: {t['text']}; font-weight: bold; }}
QLabel[role="hint"] {{ color: {t['text_muted']}; font-style: italic; }}

/* --- a region of a split workspace ----------------------------------
   The Structure tab divides into five panes, and a pane needs an edge:
   without one the cards' title hairlines run out into open space and
   the splitter handles are the only thing suggesting where one section
   ends. Ground stays `surface` so the tab's overall tone is unchanged —
   this adds an edge, not a second background. */
{REGION_SELECTOR} {{
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}

/* --- where the node lives, beside what it is called -------------------
   Muted, so the eye lands on the name, which is the half that can be
   typed over. The padding matches the field beside it so the two sit on
   one baseline. */
QLabel[role="{NODE_PATH}"] {{
    color: {t['text_muted']};
    padding: 6px 0px 5px 2px;
}}

/* --- the node's own name, editable in place --------------------------
   Read-only it is the title: same weight, same padding, no chrome, so
   the eye reads one line of "where am I". Editing it gives it a real
   box, because at that point it is a field and has to look like one. */
{NODE_NAME_SELECTOR} {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 6px 2px 5px 2px;
    font-weight: bold;
    color: {t['text']};
}}
{NODE_NAME_SELECTOR}:hover:read-only {{
    border: 1px solid {t['border']};
}}
{NODE_NAME_SELECTOR}:!read-only {{
    background: {t['surface']};
    border: 1px solid {t['accent']};
    padding: 6px 4px 5px 4px;
}}

/* --- item views this application owns -------------------------------
   qdarkstyle ships no QTableView rules at all, so alternating rows fall
   back to the OS palette and are near-invisible in light theme. */
{TABLE_SELECTOR} {{
    gridline-color: {t['table_grid']};
    alternate-background-color: {t['table_alt_row']};
}}
{TABLE_SELECTOR}::item {{
    padding: 1px 3px;
}}
{TABLE_SELECTOR} QHeaderView::section {{
    background-color: {t['table_header']};
    color: {t['text_muted']};
    border: none;
    border-right: 1px solid {t['divider']};
    border-bottom: 1px solid {t['divider']};
    padding: 3px 4px;
    font-weight: bold;
}}

/* --- progress bars ---------------------------------------------------
   The accent, so a run in flight is the one moving, coloured thing in a
   panel of grey controls. Only the chunk and the groove are restated:
   qdarkstyle's own QProgressBar rule stays in force for everything else,
   including the label colour of the bars that show a percentage.
   ``::chunk`` is a sub-control of the tagged widget, not a descendant, so
   it cannot reach into anything third-party. */
{PROGRESS_SELECTOR} {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 4px;
    text-align: center;
}}
{PROGRESS_SELECTOR}::chunk {{
    background-color: {t['accent']};
    border-radius: 3px;
}}
{PROGRESS_SELECTOR}:disabled {{
    border-color: {t['divider']};
    color: {t['text_disabled']};
}}
{PROGRESS_SELECTOR}::chunk:disabled {{
    background-color: {t['border']};
}}

/* --- status bar cells ------------------------------------------------ */
QLabel[role="sb-cell"] {{
    padding: 0 8px;
    border-left: 1px solid {t['divider']};
    color: {t['text_muted']};
}}
QLabel[role="sb-cell-active"] {{
    padding: 0 8px;
    border-left: 1px solid {t['divider']};
    color: {t['text']};
}}
/* The cursor readout: monospace so the digits stop jittering sideways
   as the cursor moves. */
QLabel[role="sb-cell-mono"] {{
    padding: 0 8px;
    border-left: 1px solid {t['divider']};
    color: {t['text_muted']};
    font-family: monospace;
}}
/* A run in flight. Chained attribute selectors, so this beats the plain
   sb-cell rule above without either one having to know about the other. */
QLabel[role="sb-cell"][status="run"] {{
    color: {t['accent']};
    font-weight: bold;
}}

/* --- figure-export preview pane (a plot ground, not a panel) -------- */
QLabel[role="preview-canvas"] {{
    background-color: {t['plot_bg']};
    color: {t['text_muted']};
}}

/* --- update banner (objectName already set in update_ui.py) --------- */
#UpdateBanner {{
    background-color: {t['banner_bg']};
    border: none;
}}
QLabel#UpdateBannerText {{
    color: {t['banner_fg']};
}}
"""
