"""Tools → Export figure window.

Non-modal QMainWindow that drives
``mlgidbase.mlgidBASE.plot_analysis_results`` for a live, debounced
preview and a final figure export (PNG raster or SVG vector — the
format follows the chosen file extension, which mlgidbase preserves and
matplotlib's ``savefig`` writer keys off). Replaces the previous
pyqtgraph ``ImageExporter``-based Tools entry; the menu wiring lives in
``main_window._action_export_figure``.

The preview pipeline always writes a temp PNG and displays it via
``QLabel`` pixmap (a raster preview regardless of the save format), but
through the same ``plot_analysis_results`` code path the export uses, so
the preview matches what ``Save figure`` writes.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)


# Module-level: mlgidbase import status (populated lazily on first
# render so a missing pipeline dep doesn't break window construction).
_mlgidbase_import_error: str | None = None


# Tutorial-canonical defaults for the three layer-styling dicts.
# Used to seed both the advanced-section widgets and the fallback
# values when widgets aren't yet built.
_DETECTED_DEFAULTS: dict[str, object] = {
    "line_width": 0.5,
    "line_style": "--",
    "line_color": "black",
    "plot_id": True,
    "text_color": "white",
    "text_size": 8,
    "plot": False,
}
_FITTED_DEFAULTS: dict[str, object] = {
    "plot_segments": True,
    "marker": "o",
    "marker_size": 50,
    "marker_facecolor": "none",
    "marker_edgecolor": "bone",
    "plot_rings": True,
    "line_width": 1,
    "line_style": "--",
    "line_color": "bone",
    "plot_id": False,
    "text_color": "white",
    "text_size": 8,
    "plot": False,
}
# Note: matched values get list-wrapped at gather time. mlgidbase
# wraps each style key in ``itertools.cycle()`` and a bare string
# like "none" would cycle through individual characters. List-wrap
# is the safe way to feed scalar choices.
_MATCHED_DEFAULTS: dict[str, object] = {
    "solution": None,
    "plot_segments": True,
    "marker": "o",
    "marker_size": 50,
    "marker_facecolor": "none",
    "marker_edgecolor": "bone",
    "plot_rings": True,
    "line_width": 1,
    "line_style": "--",
    "line_color": "bone",
    "plot_id": False,
    "text_color": "white",
    "text_size": 8,
    "legend": True,
    "plot": False,
}

# Keys whose values mlgidbase pipes through itertools.cycle for
# matched_params — must be wrapped as 1-element lists when we feed
# scalar UI values back.
_MATCHED_LIST_KEYS = {
    "marker",
    "marker_size",
    "marker_facecolor",
    "marker_edgecolor",
    "line_width",
    "line_style",
    "line_color",
    "text_color",
}

_CMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "gray", "bone", "hot", "jet", "turbo",
]
_LINE_STYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "v", "D", "x", "+", "*"]


# ---------------- shared widgets ----------------
# Canonical homes moved to widgets.py in the 2026 source split; the
# underscore aliases keep this file's internals working unchanged.

from mlgidgui import file_dialogs
from mlgidgui.widgets import CollapsibleSection as _CollapsibleSection
from mlgidgui.widgets import row_wrap as _row_wrap
from mlgidgui.widgets import spin_double as _spin_double
from mlgidgui.widgets import spin_int as _spin_int
from mlgidgui.widgets import PRIMARY as _PRIMARY, set_variant as _set_variant


def _modified_save_paths(base: str, entry: str, frame_num: int) -> list[Path]:
    """All paths mlgidbase might have written for one call: the
    no-matched/no-solutions single file plus the per-solution
    ``_sol_{IIII}`` variants. Sorted by mtime descending so callers
    can pick the most-recently-written for preview display.
    """
    p = Path(base)
    stem = f"{p.stem}_{entry}_fr_{frame_num:04d}"
    parent = p.parent if p.parent != Path("") else Path(".")
    candidates = sorted(
        parent.glob(f"{stem}*{p.suffix}"),
        key=lambda q: q.stat().st_mtime if q.exists() else 0,
        reverse=True,
    )
    return candidates


# ---------------- the window ----------------

class FigureExportWindow(QMainWindow):
    """Non-modal export window with on-demand preview rendering.

    Previously auto-rendered ~200 ms after each settings change. That
    proved annoying — every spin / dropdown click triggered a 3–6 s
    mlgidbase render including a silx-tree detach/reattach which
    interrupts the main window. The current design instead marks the
    preview "stale" on any change and renders only when the user
    clicks **Render preview** (or on initial window open).
    """

    def __init__(self, main_window) -> None:
        # QMainWindow with the main window as logical parent gives us
        # a separate top-level window with its own taskbar entry.
        super().__init__(main_window)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle("Export figure")

        self._main = main_window
        self._temp_png_base: Path | None = None
        self._last_rendered_path: Path | None = None
        self._render_in_flight = False
        # ``True`` once any setting has changed since the last
        # successful render. Drives the Render button's emphasis and
        # the status label hint.
        self._settings_dirty = False
        # Suspended while the constructor seeds defaults from the
        # host so each setter's signal doesn't flip the dirty flag
        # before the user has touched anything.
        self._suspend_dirty = True

        # Per-section widget dicts populated in _build_layer_section.
        # Lets _gather_layer_params iterate generically.
        self._layer_widgets: dict[str, dict[str, QWidget]] = {}

        self._build_ui()
        self._populate_from_main()

        self.resize(1100, 760)
        self._suspend_dirty = False
        # Kick off the first render after the event loop is running.
        # Subsequent renders require an explicit Render-button click.
        QTimer.singleShot(0, self._render)

    # ---------- UI build ----------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_settings_pane())
        splitter.addWidget(self._build_preview_pane())
        splitter.setSizes([400, 700])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._status_label = QLabel("Idle")
        sb = QStatusBar()
        sb.addPermanentWidget(self._status_label)
        self.setStatusBar(sb)

    def _build_settings_pane(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_basics_section())
        # Per-layer advanced sections — collapsed by default.
        layout.addWidget(self._build_layer_section(
            "Detected styling (advanced)", "detected", _DETECTED_DEFAULTS,
            include_matched_extras=False,
        ))
        layout.addWidget(self._build_layer_section(
            "Fitted styling (advanced)", "fitted", _FITTED_DEFAULTS,
            include_matched_extras=False,
        ))
        layout.addWidget(self._build_layer_section(
            "Matched styling (advanced)", "matched", _MATCHED_DEFAULTS,
            include_matched_extras=True,
        ))
        layout.addWidget(self._build_defaults_section())
        layout.addWidget(self._build_save_section())
        layout.addStretch(1)
        return scroll

    def _build_basics_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Basic settings", expanded=True)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Layer toggles. Each maps to <kind>_params['plot'].
        self.cb_detected = QCheckBox("Detected")
        self.cb_fitted = QCheckBox("Fitted")
        self.cb_matched = QCheckBox("Matched")
        for cb in (self.cb_detected, self.cb_fitted, self.cb_matched):
            cb.toggled.connect(self._mark_stale)
        layer_row = QHBoxLayout()
        layer_row.addWidget(self.cb_detected)
        layer_row.addWidget(self.cb_fitted)
        layer_row.addWidget(self.cb_matched)
        layer_row.addStretch(1)
        form.addRow("Layers:", _row_wrap(layer_row))

        # Frame / entry — populated from host on open.
        self.cmb_entry = QComboBox()
        self.cmb_entry.currentTextChanged.connect(lambda _t: self._mark_stale())
        form.addRow("Entry:", self.cmb_entry)
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 9999)
        self.spin_frame.valueChanged.connect(lambda _v: self._mark_stale())
        form.addRow("Frame:", self.spin_frame)

        # Colormap.
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(_CMAPS)
        self.cmb_cmap.setCurrentText("inferno")
        self.cmb_cmap.currentTextChanged.connect(lambda _t: self._mark_stale())
        form.addRow("Colormap:", self.cmb_cmap)

        # Intensity range (clims). LogNorm requires vmin > 0, so we
        # clamp the lower bound to a tiny positive value rather than
        # 0.0.
        self.spin_clim_min = _spin_double(1e-6, 1e12, 50.0)
        self.spin_clim_max = _spin_double(1e-6, 1e12, 1e4)
        for s in (self.spin_clim_min, self.spin_clim_max):
            s.valueChanged.connect(lambda _v: self._mark_stale())
        clim_row = QHBoxLayout()
        clim_row.addWidget(self.spin_clim_min)
        clim_row.addWidget(QLabel("→"))
        clim_row.addWidget(self.spin_clim_max)
        form.addRow("Intensity:", _row_wrap(clim_row))

        # q-range. Each spin paired with an "auto" checkbox that
        # collapses to None when checked.
        self.spin_xmin = _spin_double(-10.0, 10.0, -1.0)
        self.spin_xmax = _spin_double(-10.0, 10.0, 2.5)
        self.spin_ymin = _spin_double(-10.0, 10.0, -1.0)
        self.spin_ymax = _spin_double(-10.0, 10.0, 2.5)
        self.cb_xmin_auto = QCheckBox("auto")
        self.cb_xmax_auto = QCheckBox("auto")
        self.cb_ymin_auto = QCheckBox("auto")
        self.cb_ymax_auto = QCheckBox("auto")
        for cb in (self.cb_xmin_auto, self.cb_xmax_auto, self.cb_ymin_auto, self.cb_ymax_auto):
            cb.setChecked(True)
            cb.toggled.connect(self._on_auto_toggled)
        for s in (self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax):
            s.setEnabled(False)
            s.valueChanged.connect(lambda _v: self._mark_stale())
        form.addRow("q_xy min:", self._auto_row(self.spin_xmin, self.cb_xmin_auto))
        form.addRow("q_xy max:", self._auto_row(self.spin_xmax, self.cb_xmax_auto))
        form.addRow("q_z min:", self._auto_row(self.spin_ymin, self.cb_ymin_auto))
        form.addRow("q_z max:", self._auto_row(self.spin_ymax, self.cb_ymax_auto))

        # DPI + figure size. DPI default 150 for previewing; the user
        # can crank to 600 (tutorial) when ready to save.
        self.spin_dpi = _spin_int(150, lo=50, hi=1200)
        self.spin_dpi.valueChanged.connect(lambda _v: self._mark_stale())
        form.addRow("DPI:", self.spin_dpi)
        self.spin_fig_w = _spin_double(1.0, 30.0, 6.4, decimals=2)
        self.spin_fig_h = _spin_double(1.0, 30.0, 4.8, decimals=2)
        for s in (self.spin_fig_w, self.spin_fig_h):
            s.valueChanged.connect(lambda _v: self._mark_stale())
        fig_row = QHBoxLayout()
        fig_row.addWidget(self.spin_fig_w)
        fig_row.addWidget(QLabel("×"))
        fig_row.addWidget(self.spin_fig_h)
        fig_row.addWidget(QLabel("in"))
        form.addRow("Figure size:", _row_wrap(fig_row))

        section.body_layout.addLayout(form)
        return section

    def _auto_row(self, spin: QDoubleSpinBox, cb: QCheckBox) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(spin, 1)
        row.addWidget(cb)
        return _row_wrap(row)

    def _build_layer_section(
        self,
        title: str,
        layer_key: str,
        defaults: dict[str, object],
        *,
        include_matched_extras: bool,
    ) -> _CollapsibleSection:
        section = _CollapsibleSection(title, expanded=False)
        form = QFormLayout()
        widgets: dict[str, QWidget] = {}

        def hook(w: QWidget) -> None:
            # Generic change-notify wire.
            if isinstance(w, QCheckBox):
                w.toggled.connect(self._mark_stale)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda _t: self._mark_stale())
            elif isinstance(w, QLineEdit):
                w.editingFinished.connect(self._mark_stale)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(lambda _v: self._mark_stale())

        # Order chosen to match the tutorial's docs reading order.
        if "plot_segments" in defaults:
            w = QCheckBox(); w.setChecked(bool(defaults["plot_segments"]))
            widgets["plot_segments"] = w; form.addRow("plot_segments:", w); hook(w)
        if "plot_rings" in defaults:
            w = QCheckBox(); w.setChecked(bool(defaults["plot_rings"]))
            widgets["plot_rings"] = w; form.addRow("plot_rings:", w); hook(w)
        if "marker" in defaults:
            w = QComboBox(); w.addItems(_MARKERS); w.setCurrentText(str(defaults["marker"]))
            widgets["marker"] = w; form.addRow("marker:", w); hook(w)
        if "marker_size" in defaults:
            w = _spin_int(int(defaults["marker_size"]), lo=1, hi=2000)
            widgets["marker_size"] = w; form.addRow("marker_size:", w); hook(w)
        if "marker_facecolor" in defaults:
            w = QLineEdit(str(defaults["marker_facecolor"]))
            w.setToolTip("matplotlib face colour. Use 'none' for hollow.")
            widgets["marker_facecolor"] = w; form.addRow("marker_facecolor:", w); hook(w)
        if "marker_edgecolor" in defaults:
            w = QLineEdit(str(defaults["marker_edgecolor"]))
            w.setToolTip(
                "matplotlib edge colour. mlgidbase also accepts a "
                "colormap name (e.g. 'bone') to colour markers by "
                "intensity."
            )
            widgets["marker_edgecolor"] = w; form.addRow("marker_edgecolor:", w); hook(w)
        if "line_width" in defaults:
            w = _spin_double(0.0, 20.0, float(defaults["line_width"]))
            widgets["line_width"] = w; form.addRow("line_width:", w); hook(w)
        if "line_style" in defaults:
            w = QComboBox(); w.addItems(_LINE_STYLES); w.setCurrentText(str(defaults["line_style"]))
            widgets["line_style"] = w; form.addRow("line_style:", w); hook(w)
        if "line_color" in defaults:
            w = QLineEdit(str(defaults["line_color"]))
            widgets["line_color"] = w; form.addRow("line_color:", w); hook(w)
        if "plot_id" in defaults:
            w = QCheckBox(); w.setChecked(bool(defaults["plot_id"]))
            widgets["plot_id"] = w; form.addRow("plot_id:", w); hook(w)
        if "text_color" in defaults:
            w = QLineEdit(str(defaults["text_color"]))
            widgets["text_color"] = w; form.addRow("text_color:", w); hook(w)
        if "text_size" in defaults:
            w = _spin_int(int(defaults["text_size"]), lo=1, hi=72)
            widgets["text_size"] = w; form.addRow("text_size:", w); hook(w)
        if include_matched_extras:
            # Solution: stored as currentData() — None for "All",
            # otherwise the integer solution index.
            w = QComboBox()
            w.addItem("All", None)
            for i in range(10):
                w.addItem(str(i), i)
            widgets["solution"] = w; form.addRow("solution:", w); hook(w)
            w = QCheckBox(); w.setChecked(bool(defaults.get("legend", True)))
            widgets["legend"] = w; form.addRow("legend:", w); hook(w)

        section.body_layout.addLayout(form)
        self._layer_widgets[layer_key] = widgets
        return section

    def _build_defaults_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Other plot defaults", expanded=False)
        form = QFormLayout()
        self._defaults_widgets: dict[str, QWidget] = {}
        d = self._defaults_widgets

        d["font_size"] = _spin_int(14); form.addRow("font_size:", d["font_size"])
        d["axes_titlesize"] = _spin_int(14); form.addRow("axes_titlesize:", d["axes_titlesize"])
        d["axes_labelsize"] = _spin_int(18); form.addRow("axes_labelsize:", d["axes_labelsize"])
        d["xtick_labelsize"] = _spin_int(14); form.addRow("xtick_labelsize:", d["xtick_labelsize"])
        d["ytick_labelsize"] = _spin_int(14); form.addRow("ytick_labelsize:", d["ytick_labelsize"])
        d["legend_fontsize"] = _spin_int(12); form.addRow("legend_fontsize:", d["legend_fontsize"])
        d["legend_loc"] = QLineEdit("best"); form.addRow("legend_loc:", d["legend_loc"])
        d["legend_frameon"] = QCheckBox(); d["legend_frameon"].setChecked(True)
        form.addRow("legend_frameon:", d["legend_frameon"])
        d["grid"] = QCheckBox(); form.addRow("grid:", d["grid"])
        d["grid_color"] = QLineEdit("gray"); form.addRow("grid_color:", d["grid_color"])
        d["grid_linestyle"] = QComboBox()
        d["grid_linestyle"].addItems(_LINE_STYLES)
        d["grid_linestyle"].setCurrentText("--")
        form.addRow("grid_linestyle:", d["grid_linestyle"])
        d["grid_linewidth"] = _spin_double(0.0, 10.0, 0.5)
        form.addRow("grid_linewidth:", d["grid_linewidth"])
        d["savefig_transparent"] = QCheckBox()
        form.addRow("savefig_transparent:", d["savefig_transparent"])

        for w in d.values():
            if isinstance(w, QCheckBox):
                w.toggled.connect(self._mark_stale)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda _t: self._mark_stale())
            elif isinstance(w, QLineEdit):
                w.editingFinished.connect(self._mark_stale)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(lambda _v: self._mark_stale())

        section.body_layout.addLayout(form)
        return section

    def _build_save_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("Save", expanded=True)
        form = QFormLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("pick a destination .png or .svg")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_save_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow("File:", _row_wrap(path_row))

        self.btn_save = _set_variant(QPushButton("Save figure"), _PRIMARY)
        self.btn_save.setToolTip(
            "Write the figure using the current settings. The format "
            "follows the file extension (.png raster or .svg vector). "
            "Saves to the exact path you pick; with the Matched layer "
            "on, mlgidbase writes one file per solution with '_sol_NNNN' "
            "appended to the filename."
        )
        self.btn_save.clicked.connect(self._on_save)
        form.addRow("", self.btn_save)

        section.body_layout.addLayout(form)
        return section

    def _build_preview_pane(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        self._preview = QLabel("Preview will render here.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # A plot ground, not a panel: it must match the exported
        # figure's background in either theme (it used to be a hardcoded
        # dark rectangle sitting inside a light window).
        self._preview.setProperty("role", "preview-canvas")
        self._preview.setMinimumSize(400, 300)
        v.addWidget(self._preview, 1)

        # Render preview button lives below the image — its action
        # is "redraw the thing above", so the visual association is
        # tighter here than buried in the settings column. Earlier
        # placement inside the Save form row caused visual overlap
        # with the section header on tight layouts.
        self.btn_render = QPushButton("Render preview")
        self.btn_render.setToolTip(
            "Re-draw the image above using the current settings. "
            "Rendering is on-demand to avoid stuttering on every "
            "spin / dropdown click (one render ≈ 3–6 s)."
        )
        self.btn_render.setMinimumHeight(34)
        self.btn_render.clicked.connect(self._render)
        # Centered button row — stretchers on both sides keep it
        # at its natural width even when the splitter is wide.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_render)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        return host

    # ---------- state seeding ----------

    def _populate_from_main(self) -> None:
        """Read host state once at open time.

        Subsequent frame / entry changes on the main window do not
        auto-propagate into this window — the user controls frame
        and entry from inside the export window from this point on.
        """
        session = getattr(self._main, "session", None)
        if session is None or not hasattr(session, "temp_path"):
            return
        try:
            self.setWindowTitle(f"Export figure — {Path(session.display_path).name}")
        except Exception:
            logger.debug("suppressed exception in FigureExportWindow._populate_from_main", exc_info=True)
            self.setWindowTitle("Export figure")

        main_combo = getattr(self._main, "entry_combo", None)
        if main_combo is not None:
            self.cmb_entry.clear()
            for i in range(main_combo.count()):
                self.cmb_entry.addItem(main_combo.itemText(i))
            cur = main_combo.currentText()
            if cur:
                self.cmb_entry.setCurrentText(cur)

        viewer = getattr(self._main, "viewer", None)
        if viewer is not None:
            try:
                n = max(0, int(viewer.n_frames) - 1)
            except Exception:
                logger.debug("suppressed exception in FigureExportWindow._populate_from_main", exc_info=True)
                n = 9999
            self.spin_frame.setMaximum(max(n, 0))
            try:
                self.spin_frame.setValue(int(viewer.current_frame))
            except Exception:
                logger.debug("suppressed exception in FigureExportWindow._populate_from_main", exc_info=True)
                pass
            dp = getattr(viewer, "_display_params", None)
            if dp is not None and getattr(dp, "levels", None):
                try:
                    lo, hi = float(dp.levels[0]), float(dp.levels[1])
                    # LogNorm requires vmin > 0; if the viewer's
                    # level range starts at or below zero, lift the
                    # floor so the first render doesn't blow up.
                    lo = max(lo, 1.0)
                    if 0 < lo < hi:
                        self.spin_clim_min.setValue(lo)
                        self.spin_clim_max.setValue(hi)
                except Exception:
                    logger.debug("suppressed exception in FigureExportWindow._populate_from_main", exc_info=True)
                    pass

    # ---------- param gathering ----------

    def _gather_layer_params(self, layer_key: str, defaults: dict, plot_on: bool) -> dict:
        """Build a layer params dict using current widget values,
        falling back to ``defaults`` for unbuilt keys.

        ``list_wrap`` is enabled automatically for the matched layer
        — mlgidbase wraps each style key in ``itertools.cycle()``
        and bare strings would cycle character-by-character.
        """
        params = dict(defaults)
        widgets = self._layer_widgets.get(layer_key, {})
        for key, w in widgets.items():
            if isinstance(w, QCheckBox):
                params[key] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                if key == "solution":
                    params[key] = w.currentData()
                else:
                    params[key] = w.currentText()
            elif isinstance(w, QLineEdit):
                txt = w.text().strip()
                params[key] = txt if txt else defaults.get(key)
            elif isinstance(w, QSpinBox):
                params[key] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                params[key] = float(w.value())
        params["plot"] = bool(plot_on)
        if layer_key == "matched":
            for k in _MATCHED_LIST_KEYS:
                if k in params and not isinstance(params[k], (list, tuple)):
                    params[k] = [params[k]]
        return params

    def _gather_defaults(self) -> dict:
        d = {}
        d["cmap"] = self.cmb_cmap.currentText()
        d["savefig_dpi"] = int(self.spin_dpi.value())
        d["figsize"] = (float(self.spin_fig_w.value()), float(self.spin_fig_h.value()))
        w = self._defaults_widgets
        d["font_size"] = int(w["font_size"].value())
        d["axes_titlesize"] = int(w["axes_titlesize"].value())
        d["axes_labelsize"] = int(w["axes_labelsize"].value())
        d["xtick_labelsize"] = int(w["xtick_labelsize"].value())
        d["ytick_labelsize"] = int(w["ytick_labelsize"].value())
        d["legend_fontsize"] = int(w["legend_fontsize"].value())
        d["legend_loc"] = w["legend_loc"].text() or "best"
        d["legend_frameon"] = bool(w["legend_frameon"].isChecked())
        d["grid"] = bool(w["grid"].isChecked())
        d["grid_color"] = w["grid_color"].text() or "gray"
        d["grid_linestyle"] = w["grid_linestyle"].currentText()
        d["grid_linewidth"] = float(w["grid_linewidth"].value())
        d["savefig_transparent"] = bool(w["savefig_transparent"].isChecked())
        return d

    def _gather_call_kwargs(self) -> dict:
        clims = (float(self.spin_clim_min.value()), float(self.spin_clim_max.value()))
        xlim = (
            None if self.cb_xmin_auto.isChecked() else float(self.spin_xmin.value()),
            None if self.cb_xmax_auto.isChecked() else float(self.spin_xmax.value()),
        )
        ylim = (
            None if self.cb_ymin_auto.isChecked() else float(self.spin_ymin.value()),
            None if self.cb_ymax_auto.isChecked() else float(self.spin_ymax.value()),
        )
        return {
            "entry": self.cmb_entry.currentText() or None,
            "frame_num": int(self.spin_frame.value()),
            "clims": clims,
            "xlim": xlim,
            "ylim": ylim,
            "detected_params": self._gather_layer_params(
                "detected", _DETECTED_DEFAULTS, self.cb_detected.isChecked(),
            ),
            "fitted_params": self._gather_layer_params(
                "fitted", _FITTED_DEFAULTS, self.cb_fitted.isChecked(),
            ),
            "matched_params": self._gather_layer_params(
                "matched", _MATCHED_DEFAULTS, self.cb_matched.isChecked(),
            ),
        }

    # ---------- render pipeline ----------

    def _mark_stale(self, *_args) -> None:
        """Flag the preview as out-of-date with the current
        settings. Called by every widget signal in the settings
        column. Triggers no render — the user clicks Render
        preview when they're ready.

        ``*_args`` swallows the various payloads Qt signals carry
        (toggled→bool, valueChanged→int/float, currentTextChanged→
        str) so we can connect the same slot to all of them.
        """
        if self._suspend_dirty or self._render_in_flight:
            return
        if self._settings_dirty:
            return
        self._settings_dirty = True
        self._status_label.setText(
            "Settings changed — click Render preview to update."
        )
        if hasattr(self, "btn_render"):
            # Visual cue: bolden the Render button when there's a
            # pending change. Plain styling once a render lands.
            self.btn_render.setStyleSheet(
                "QPushButton { font-weight: bold; }"
            )

    def _render(self) -> None:
        if self._render_in_flight:
            return
        self._render_in_flight = True
        self.btn_save.setEnabled(False)
        if hasattr(self, "btn_render"):
            self.btn_render.setEnabled(False)
        self._status_label.setText("Rendering…")
        QApplication.processEvents()
        # Build kwargs first so a malformed config doesn't trigger a
        # silx detach we'd then have to undo.
        try:
            kwargs = self._gather_call_kwargs()
            defaults = self._gather_defaults()
        except Exception as exc:
            logger.debug("suppressed exception in FigureExportWindow._render", exc_info=True)
            self._status_label.setText(f"Settings error: {exc}")
            self._render_in_flight = False
            self.btn_save.setEnabled(True)
            return
        # mlgidGUI's silx tree holds an ``r`` handle on the temp
        # file; pygid's NexusFile opens ``r+`` for reads (see
        # ``pygid/nexus_reader.py::get_dataset``) so HDF5 refuses
        # the second open with "file is already open for read-only".
        # The pipeline path solves this by detaching silx before
        # ``mlgidBASE`` opens the file and reattaching after; we
        # mirror that dance here. The detach is cheap (silx clear
        # + viewer FrameSource release); the reattach rebuilds the
        # tree from the live session list.
        try:
            self._run_with_detached_tree(self._do_render, kwargs, defaults)
        finally:
            self._render_in_flight = False
            self.btn_save.setEnabled(True)
            if hasattr(self, "btn_render"):
                self.btn_render.setEnabled(True)
                self.btn_render.setStyleSheet("")
            self._settings_dirty = False

    def _do_render(self, kwargs: dict, defaults: dict) -> None:
        try:
            from mlgidbase import mlgidBASE  # type: ignore
        except Exception as exc:
            logger.debug("suppressed exception in FigureExportWindow._do_render", exc_info=True)
            self._status_label.setText(f"mlgidbase unavailable: {exc}")
            return
        session = getattr(self._main, "session", None)
        if session is None or not hasattr(session, "temp_path"):
            self._status_label.setText("No NeXus session open.")
            return
        try:
            analysis = mlgidBASE(filename=str(Path(session.temp_path)))
        except Exception as exc:
            logger.debug("suppressed exception in FigureExportWindow._do_render", exc_info=True)
            self._status_label.setText(f"Couldn't open file: {exc}")
            return
        try:
            analysis.set_plot_defaults(**defaults)
            if self._temp_png_base is None:
                fd, name = tempfile.mkstemp(suffix=".png", prefix="mlgidgui-figexp-")
                os.close(fd)
                try:
                    Path(name).unlink()
                except OSError:
                    pass
                self._temp_png_base = Path(name)
            entry = kwargs["entry"] or ""
            frame_num = kwargs["frame_num"]
            # Clean any files from previous renders that share the
            # target stem so glob results reflect only the current
            # render. Without this, the status "N solutions"
            # readout counts stale files left over from prior runs.
            for stale in _modified_save_paths(str(self._temp_png_base), entry, frame_num):
                try:
                    stale.unlink()
                except OSError:
                    pass
            analysis.plot_analysis_results(
                save_fig=True,
                path_to_save_fig=str(self._temp_png_base),
                plot_result=False,
                return_fig=False,
                **kwargs,
            )
            # mlgidbase writes one file when matched is off / no
            # solutions, or N files (one per ``_sol_NNNN``) when
            # matched is on. Pick the most-recently-written for the
            # preview pane.
            candidates = _modified_save_paths(str(self._temp_png_base), entry, frame_num)
            if candidates:
                self._last_rendered_path = candidates[0]
                self._show_preview(candidates[0])
                if len(candidates) > 1:
                    self._status_label.setText(
                        f"Last rendered {time.strftime('%H:%M:%S')} "
                        f"({len(candidates)} solutions; showing first)"
                    )
                else:
                    self._status_label.setText(
                        f"Last rendered {time.strftime('%H:%M:%S')}"
                    )
            else:
                self._status_label.setText("Render produced no file.")
        except Exception as exc:
            logger.debug("suppressed exception in FigureExportWindow._do_render", exc_info=True)
            self._status_label.setText(f"Render failed: {exc}")

    def _run_with_detached_tree(self, fn, *args, **kwargs) -> None:
        """Run ``fn`` with mlgidGUI's silx tree + viewer FrameSource
        temporarily released, then reattach.

        Mirrors ``MainWindow._detach_silx_tree`` / ``_reattach_silx_tree``
        which already exists for the pipeline run. We're not running
        on a worker thread, but mlgidbase needs the file exclusively
        for ``r+`` so the detach is still required.

        Caller is responsible for resetting in-flight / button state;
        this wrapper only owns the detach/reattach pair.
        """
        detach = getattr(self._main, "_detach_silx_tree", None)
        reattach = getattr(self._main, "_reattach_silx_tree", None)
        if callable(detach):
            try:
                detach()
            except Exception:
                logger.debug("suppressed exception in FigureExportWindow._run_with_detached_tree", exc_info=True)
                pass
        try:
            fn(*args, **kwargs)
        finally:
            if callable(reattach):
                try:
                    reattach()
                except Exception:
                    logger.debug("suppressed exception in FigureExportWindow._run_with_detached_tree", exc_info=True)
                    pass

    def _show_preview(self, path: Path) -> None:
        pix = QPixmap(str(path))
        if pix.isNull():
            self._preview.setText("Preview unavailable.")
            return
        self._preview.setPixmap(pix.scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_rendered_path is not None and self._last_rendered_path.exists():
            self._show_preview(self._last_rendered_path)

    # ---------- actions ----------

    def _on_auto_toggled(self) -> None:
        for spin, cb in (
            (self.spin_xmin, self.cb_xmin_auto),
            (self.spin_xmax, self.cb_xmax_auto),
            (self.spin_ymin, self.cb_ymin_auto),
            (self.spin_ymax, self.cb_ymax_auto),
        ):
            spin.setEnabled(not cb.isChecked())
        self._mark_stale()

    def _browse_save_path(self) -> None:
        # Only the *name* of the current text carries over; the
        # directory is the app-wide browsing one (``file_dialogs``).
        start = self.path_edit.text() or "figure.png"
        path, selected = file_dialogs.save_file(
            self, "Save figure",
            "PNG image (*.png);;SVG image (*.svg)",
            suggested_name=start,
        )
        if path:
            # Honour an explicit .png/.svg the user typed; otherwise
            # append the extension implied by the chosen filter.
            # matplotlib (via mlgidbase) picks the writer from the
            # extension, so this is all that selects raster vs vector.
            if not path.lower().endswith((".png", ".svg")):
                path += ".svg" if "svg" in selected.lower() else ".png"
            self.path_edit.setText(path)

    def _on_save(self) -> None:
        target = self.path_edit.text().strip()
        if not target:
            QMessageBox.information(
                self, "Pick a path",
                "Pick a destination path (.png or .svg) before saving.",
            )
            return
        # Save goes through the same detach/reattach dance as the
        # preview — pygid still wants ``r+`` on the temp file even
        # when only reading.
        self._save_error: str | None = None
        try:
            kwargs = self._gather_call_kwargs()
            defaults = self._gather_defaults()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Settings error: {exc}")
            return
        self.btn_save.setEnabled(False)
        self.btn_render.setEnabled(False)
        try:
            self._run_with_detached_tree(
                self._do_save, target, kwargs, defaults,
            )
        finally:
            self.btn_save.setEnabled(True)
            self.btn_render.setEnabled(True)
        if self._save_error is not None:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write the figure:\n{self._save_error}",
            )
            return
        # mlgidbase appends ``_<entry>_fr_<N>`` to the path; if the
        # matched layer is on it additionally writes one file per
        # solution with ``_sol_NNNN``. Single-file case: rename
        # back onto the user's chosen path so the on-disk filename
        # matches what they asked for. Multi-file case: leave the
        # suffixed files in place and tell the user how many landed.
        written = self._save_written_paths or []
        if len(written) == 1 and written[0].exists():
            actual = written[0]
            if actual.resolve() != Path(target).resolve():
                try:
                    shutil.move(str(actual), target)
                except Exception:
                    logger.debug("suppressed exception in FigureExportWindow._on_save", exc_info=True)
                    self.statusBar().showMessage(f"Wrote {actual}", 6000)
                    return
            self.statusBar().showMessage(f"Wrote {target}", 5000)
        elif len(written) > 1:
            # Report directory + count rather than dumping every
            # path into the status bar.
            self.statusBar().showMessage(
                f"Wrote {len(written)} solution figures to {written[0].parent}",
                8000,
            )
        else:
            self.statusBar().showMessage("Save produced no file.", 5000)

    def _do_save(self, target: str, kwargs: dict, defaults: dict) -> None:
        """Run inside the silx-detached scope. Sets ``_save_error``
        and ``_save_written_paths`` for the caller to read after
        reattach."""
        self._save_written_paths: list[Path] = []
        try:
            from mlgidbase import mlgidBASE  # type: ignore
            session = getattr(self._main, "session", None)
            if session is None or not hasattr(session, "temp_path"):
                self._save_error = "No NeXus session open."
                return
            analysis = mlgidBASE(filename=str(Path(session.temp_path)))
            analysis.set_plot_defaults(**defaults)
            entry = kwargs["entry"] or ""
            frame_num = kwargs["frame_num"]
            # Sweep stale files at this stem so the post-call glob
            # reports only what this save actually wrote.
            for stale in _modified_save_paths(target, entry, frame_num):
                try:
                    stale.unlink()
                except OSError:
                    pass
            analysis.plot_analysis_results(
                save_fig=True,
                path_to_save_fig=target,
                plot_result=False,
                return_fig=False,
                **kwargs,
            )
            self._save_written_paths = _modified_save_paths(
                target, entry, frame_num,
            )
        except Exception as exc:
            logger.debug("suppressed exception in FigureExportWindow._do_save", exc_info=True)
            self._save_error = str(exc)

    # ---------- lifecycle ----------

    def refresh_for_session(self) -> None:
        """Called by the host when the active session changes so the
        basics pane is re-seeded from the new file. The ``mlgidBASE``
        object is never cached across renders, so there is nothing
        else to invalidate here."""
        # Clear the previous render so the user doesn't stare at
        # stale pixels while the next render kicks off.
        if self._last_rendered_path is not None and self._last_rendered_path.exists():
            try:
                self._last_rendered_path.unlink()
            except OSError:
                pass
        self._last_rendered_path = None
        self._suspend_dirty = True
        try:
            self._populate_from_main()
        finally:
            self._suspend_dirty = False
        self._mark_stale()

    def closeEvent(self, event) -> None:
        # Clean up the temp PNG we've been overwriting.
        for p in (self._temp_png_base, self._last_rendered_path):
            if p is None:
                continue
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        super().closeEvent(event)
