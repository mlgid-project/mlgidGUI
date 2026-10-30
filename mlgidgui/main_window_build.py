"""Construction of the central widget, the eight dock panels with all cross-object signal wiring, the ghost-tab-bar cleanup, the window title and the status bar.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from mlgidgui.browser_widgets import _MlgidHdf5TreeView
from mlgidgui.conversion_panel import ConversionPanel
from mlgidgui.image_viewer import (
    GIWAXSImageViewer,
    OVERLAY_STYLE,
    SIM_MISSED_COLOR,
)
from mlgidgui.main_window_constants import (
    APP_NAME,
    DEFAULT_PLAYBACK_FRAME_MS,
)
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui.peak_link import read_bool
from mlgidgui.peaks_table_panel import PeaksTablePanel
from mlgidgui.pipeline import is_mlgidbase_available
from mlgidgui.pipeline_panel import PipelinePanel
from mlgidgui.profile_viewer import ProfileViewer
from mlgidgui.scan_tracking_panel import ScanTrackingPanel
from mlgidgui.structure_panel import StructurePanel
from mlgidgui.update_ui import _UpdateBanner
from mlgidgui.pen_picker import PenPopup
from mlgidgui.welcome_view import WelcomeView
from mlgidgui.workflow_rail import WorkflowRail
from mlgidgui import file_model, peak_lists, theme_tokens
from mlgidgui.widgets import (
    GAP,
    PRIMARY,
    Card,
    make_debounced_timer,
    make_pen_swatch as _make_pen_swatch,
    section_label,
    set_variant,
    skin_progress,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from mlgidgui import icons

import logging

logger = logging.getLogger(__name__)


#: QSettings key for whether the workflow strip is open or folded.
RAIL_EXPANDED_KEY = "view/workflow_rail_expanded"
#: QSettings key for whether a stage's run glyph also brings its dock
#: forward. Default on.
RAIL_RUN_OPENS_DOCK_KEY = "view/rail_run_opens_dock"


def _dirty_dot(diameter: int = 8) -> QPixmap:
    """A filled accent disc marking unsaved changes."""
    pix = QPixmap(diameter + 4, diameter + 4)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(theme_tokens.color("accent")))
    painter.drawEllipse(2, 2, diameter, diameter)
    painter.end()
    return pix


class BuildMixin:
    def _build_central(self) -> None:
        self.viewer = GIWAXSImageViewer(self)
        self.data_viewer = DataViewerFrame(self)
        self.structure_panel = StructurePanel(self)
        self.structure_panel.recheckRequested.connect(self._on_structure_recheck)
        self.structure_panel.attributeEdited.connect(
            self._on_structure_attribute_edited)
        self.structure_panel.attributeRenamed.connect(
            self._on_structure_attribute_renamed)
        self.structure_panel.attributeAdded.connect(
            self._on_structure_attribute_added)
        self.structure_panel.attributeRemoved.connect(
            self._on_structure_attribute_removed)
        self.structure_panel.scalarValueEdited.connect(
            self._on_structure_value_edited)
        self.structure_panel.copyChangesRequested.connect(
            self._on_structure_copy_changes)
        self.structure_panel.retargetLinkRequested.connect(
            self._on_structure_retarget_link)
        self.structure_panel.followLinkRequested.connect(
            self._on_structure_follow_link)
        self.structure_panel.editValuesRequested.connect(
            self._on_structure_edit_values)
        self.structure_panel.searchRequested.connect(
            self._on_structure_search)
        self.structure_panel.searchResultActivated.connect(
            self._on_structure_search_result)
        # The tab's own tree. It reads nothing itself — the lister goes
        # through MainWindow's handle-aware accessor — so the panel keeps
        # its no-I/O rule and the tree never holds a file open.
        self.structure_panel.node_tree.set_lister(self._structure_list_children)
        self.structure_panel.node_tree.nodeSelected.connect(
            self._on_structure_tree_selected)
        self.structure_panel.node_tree.contextRequested.connect(
            self._on_structure_tree_context)
        self.structure_panel.nodeActionRequested.connect(
            self._on_structure_action)
        self.structure_panel.renameRequested.connect(
            self._on_structure_header_rename)

        self.tabs = QTabWidget(self)
        # documentMode flattens the tab-pane border so the image fills
        # the full tab area without the small inset that lets pyqtgraph
        # show a few pixels of scrollable margin.
        self.tabs.setDocumentMode(True)
        # documentMode is partial under qdarkstyle — the pane keeps a
        # small border + padding from the dark stylesheet which traps
        # ~2 px of overflow from the central widget. Override via an
        # explicit zero-pad stylesheet so the viewer fills flush.
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 0px; padding: 0px; margin: 0px; }"
        )
        self.tabs.addTab(self.viewer, "Image")
        self.tabs.addTab(self.data_viewer, "Data")
        # The structure editor. Deliberately last: Image is what the app
        # opens on, and inserting a tab before Data would move a tab the
        # user's muscle memory already knows.
        self.tabs.addTab(self.structure_panel, "Structure")
        # Render a deferred (clicked-while-hidden) tree node when the user
        # switches to the Data or Structure tab — see
        # ``_set_or_defer_data_node`` / ``_set_or_defer_structure_node``.
        self.tabs.currentChanged.connect(self._on_main_tab_changed)

        # Central column: a hidden "update available" banner above the tabs
        # (shown by the startup update check — see _on_update_check_finished).
        central = QWidget(self)
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._update_banner = _UpdateBanner(central)
        self._update_banner.hide()
        self._update_banner.installRequested.connect(
            self._on_install_update_requested
        )
        col.addWidget(self._update_banner)
        # The workflow spine, between the banner and the view. Hidden
        # with no session — the welcome page owns that state.
        self.workflow_rail = WorkflowRail(central)
        self.workflow_rail.stageActivated.connect(self._on_rail_stage_activated)
        self.workflow_rail.stageRunRequested.connect(self._on_rail_stage_run)
        # Folded or open is a per-user preference, like the playback
        # settings: someone who works from the docks wants the height
        # back and should not have to reclaim it every launch.
        self.workflow_rail.set_expanded(
            str(QSettings().value(RAIL_EXPANDED_KEY, "true")).lower() != "false"
        )
        self.workflow_rail.expandedChanged.connect(
            lambda expanded: QSettings().setValue(
                RAIL_EXPANDED_KEY, bool(expanded))
        )
        self.workflow_rail.hide()
        col.addWidget(self.workflow_rail)
        # The tabs share the column with a welcome page, shown whenever
        # no session is open (``_apply_session_mode``). A stack rather
        # than show/hide on the tabs so the two can never both be up, and
        # so the docks' own visibility logic is untouched — nothing in
        # the dock layout lives in the central widget.
        self.welcome_view = WelcomeView(central)
        self.welcome_view.openRequested.connect(self._action_open)
        self.welcome_view.importRequested.connect(self._action_import_converted)
        self.welcome_view.recentRequested.connect(self._open_recent)
        self._central_stack = QStackedWidget(central)
        self._central_stack.addWidget(self.welcome_view)   # index 0
        self._central_stack.addWidget(self.tabs)           # index 1
        col.addWidget(self._central_stack)
        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        # Make the side docks own the bottom corners so the bottom Profile
        # dock stays aligned with the central image/data tabs.
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Left: HDF5 tree (silx) — subclass swaps the root icon for raw
        # sessions so NeXus and raw files are distinguishable at a glance.
        self.tree = _MlgidHdf5TreeView(self)
        # Sorting is DISABLED on purpose. silx's NexusSortFilterProxyModel
        # sorts the NAME column by resolving each node — ``lessThan`` calls
        # ``__isNXentry`` (reads ``node.obj.attrs['NX_class']``) and
        # ``childDatasetLessThan(..., 'start_time')``, both of which
        # DEREFERENCE the external link. Expanding a master that links 226
        # external scans therefore opened all 226 scans on the GUI thread
        # (h5py holds the GIL across those network opens) — the freeze that
        # made the file browser unusable and entry clicks unresponsive.
        # With sorting off, ``lessThan`` is never called, so only the rows
        # actually scrolled into view resolve (for their icon), lazily.
        # Entries show in the file's native order (acquisition order for the
        # track_order pygid masters this targets).
        self.tree.setSortingEnabled(False)
        # Shift/Ctrl-click and Ctrl+A pick several rows so Delete can close
        # them together.
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # Single-click silently updates Data tab; double-click jumps to it.
        self.tree.selectionModel().selectionChanged.connect(
            self._on_tree_selection_changed
        )
        self.tree.activated.connect(self._on_tree_activated)
        # Delete (browser-focused) removes the selected file from the tree,
        # mirroring File → Close (Ctrl+W). See _remove_selected_file_from_browser.
        self.tree.deleteFileRequested.connect(self._remove_selected_file_from_browser)
        # Small header row above the tree with the manual Refresh action:
        # re-sync every open session with the filesystem (close deleted
        # originals, reload changed ones). Also on F5.
        self._tree_refresh_btn = QToolButton()
        self._tree_refresh_btn.setText("Refresh")
        self._tree_refresh_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons.bind(self._tree_refresh_btn, "refresh")
        self._tree_refresh_btn.setAutoRaise(True)
        self._tree_refresh_btn.setToolTip(
            "Re-check every open file on disk (F5): close files whose "
            "original was deleted (unsaved changes are kept open), "
            "reload files that changed on disk (unless they have "
            "unsaved changes)."
        )
        self._tree_refresh_btn.clicked.connect(self._refresh_file_tree)
        tree_header = QHBoxLayout()
        tree_header.setContentsMargins(2, 2, 2, 0)
        tree_header.addWidget(self._tree_refresh_btn)
        tree_header.addStretch(1)
        tree_box = QWidget(self)
        tree_box_layout = QVBoxLayout(tree_box)
        tree_box_layout.setContentsMargins(0, 0, 0, 0)
        tree_box_layout.setSpacing(0)
        tree_box_layout.addLayout(tree_header)
        tree_box_layout.addWidget(self.tree)
        self._tree_dock = QDockWidget("File browser", self)
        self._tree_dock.setWidget(tree_box)
        self._tree_dock.setObjectName("FileBrowserDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._tree_dock)
        # Edits made while the browser is folded away defer its rebuild
        # (see ``_ensure_browser_current``). Re-opening it by hand is the
        # one route back that does not pass through a tab switch.
        self._tree_dock.visibilityChanged.connect(
            self._on_browser_visibility_changed)
        # Edit actions on the browser's own context menu. silx's
        # addContextMenuCallback is the supported hook, so the tree's
        # policy, model and click handling stay exactly as they were.
        self._install_structure_context_menu()
        refresh_action = QAction("Refresh file browser", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self._refresh_file_tree)
        self.addAction(refresh_action)

        # Right: entry selector + overlay toggles
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self.entry_combo = QComboBox()
        self.entry_combo.currentTextChanged.connect(self._on_entry_changed)
        form.addRow("Entry:", self.entry_combo)

        # Frame-navigation controls live on the *image viewer's*
        # toolbar (alongside the Log-scale checkbox) so they're
        # reachable from any right-dock tab — not just Display. We
        # still build them here because MainWindow owns the slot
        # wiring (slider valueChanged → viewer, play timer, …);
        # ``viewer.insert_frame_controls`` re-parents them onto the
        # toolbar in ``_build_docks`` once the toolbar is ready.
        # Hidden for single-frame stacks where they have no function.
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        # Editable frame index next to the slider: type an exact frame
        # and confirm with Enter (keyboard tracking is off so partial
        # input like the "2" of "25" doesn't seek mid-typing).
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.setToolTip(
            "Current frame — type an index and press Enter to jump."
        )
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        # Compact "/ max" readout completing the spinbox ("idx / max").
        # The "Frame" word was dropped to save toolbar space — its
        # meaning is obvious from context (next to play / prev / next
        # icons and a slider).
        self.frame_label = QLabel("—")
        self.frame_label.setMinimumWidth(36)
        self.frame_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # Play / pause toggle. Drives a QTimer that calls
        # viewer.set_frame(current + 1) on every tick; stops when the
        # last frame is reached. Standard-icon based so qdarkstyle
        # picks up the right colour automatically.
        self.play_button = QToolButton()
        self.play_button.setCheckable(True)
        self.play_button.setIcon(icons.icon("play"))
        self.play_button.setToolTip(
            "Play frames from the current position to the end.\n"
            "Stops at the last frame; click again to pause."
        )
        self.play_button.toggled.connect(self._on_play_toggled)
        # Previous / next single-step buttons. Step by one frame and
        # clamp at boundaries (the buttons disable themselves at
        # frame 0 / last via ``_refresh_frame_nav_enabled``).
        self.prev_frame_button = QToolButton()
        icons.bind(self.prev_frame_button, "prev")
        self.prev_frame_button.setToolTip("Previous frame")
        self.prev_frame_button.setAutoRepeat(True)
        self.prev_frame_button.setAutoRepeatDelay(300)
        self.prev_frame_button.setAutoRepeatInterval(80)
        self.prev_frame_button.clicked.connect(self._on_prev_frame_clicked)
        self.next_frame_button = QToolButton()
        icons.bind(self.next_frame_button, "next")
        self.next_frame_button.setToolTip("Next frame")
        self.next_frame_button.setAutoRepeat(True)
        self.next_frame_button.setAutoRepeatDelay(300)
        self.next_frame_button.setAutoRepeatInterval(80)
        self.next_frame_button.clicked.connect(self._on_next_frame_clicked)
        # Driver for playback. Interval + step are resolved from
        # QSettings (see ``_compute_play_schedule``) on every Play
        # start, so a setting change picks up on the next press
        # without restarting the timer. Default mode is "time per
        # frame" at 50 ms = 20 fps. Requested rates below 50 ms /
        # frame don't speed up the timer — instead, the play-head
        # advances by ``self._play_step`` frames per tick so the
        # target total time is honoured while the timer stays at the
        # 20 fps practical ceiling.
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(DEFAULT_PLAYBACK_FRAME_MS)
        self._play_timer.timeout.connect(self._on_play_tick)
        layout.addLayout(form)
        # Hand the controls to the image viewer's toolbar. Order reads
        # left-to-right: prev / play / next / slider / index / "max".
        self.viewer.insert_frame_controls([
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.frame_slider,
            self.frame_spin,
            self.frame_label,
        ])
        # Start hidden — only useful once a multi-frame stack is loaded.
        self._set_frame_slider_visible(False)

        layout.addWidget(section_label("Overlays"))
        # Manual peaks intentionally omitted: the GUI now keeps at most
        # one manual box per frame (drawn → replaced → committed via
        # Add-to-fitted/detected, removed via Esc / Delete), so a
        # visibility toggle for "all manual peaks" no longer has work
        # to do. The viewer's internal _visibility["manual"] stays True
        # by default — see GIWAXSImageViewer.__init__.
        self._overlay_checks: dict[str, QCheckBox] = {}
        # The swatch is a button, like the matched-structure rows': it
        # opens the same pen editor, so the colour, line style and width
        # of every overlay are set where that overlay is named.
        self._overlay_swatches: dict[str, QToolButton] = {}
        for kind, label in (
            ("detected", "Detected peaks"),
            ("fitted", "Fitted peaks"),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QToolButton()
            swatch.setAutoRaise(True)
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(
                f"Click to set the {label.lower()} colour, line style "
                f"and width"
            )
            swatch.clicked.connect(
                lambda _=False, k=kind: self._open_overlay_pen_popup(k)
            )
            self._overlay_swatches[kind] = swatch
            self._refresh_overlay_swatch(kind)
            row.addWidget(swatch)
            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.toggled.connect(
                lambda v, k=kind: self.viewer.set_overlay_visible(k, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            layout.addWidget(row_widget)
            self._overlay_checks[kind] = chk

            # Per-detected min-score slider. Sits indented under the
            # Detected checkbox so the affordance is right next to
            # the layer it controls. Range 0–100 → 0.00–1.00 cutoff
            # forwarded to ``viewer.set_detected_score_cutoff``.
            # Initial value is seeded to the minimum score on the
            # current frame in ``_seed_detected_score_slider`` so
            # the default shows every detection; the user drags up
            # to hide weak ones.
            if kind == "detected":
                score_row = QHBoxLayout()
                score_row.setContentsMargins(20, 0, 0, 0)
                score_row.setSpacing(6)
                score_row.addWidget(QLabel("Min score:"))
                self._detected_score_slider = QSlider(Qt.Orientation.Horizontal)
                self._detected_score_slider.setRange(0, 100)
                self._detected_score_slider.setValue(0)
                self._detected_score_slider.setToolTip(
                    "Hide detected peaks whose model score is below "
                    "the cutoff. The slider starts at the lowest "
                    "score on the current frame so nothing is hidden "
                    "by default."
                )
                self._detected_score_slider.valueChanged.connect(
                    self._on_detected_score_changed
                )
                score_row.addWidget(self._detected_score_slider, 1)
                self._detected_score_value_label = QLabel("0.00")
                self._detected_score_value_label.setMinimumWidth(36)
                score_row.addWidget(self._detected_score_value_label)
                score_row_widget = QWidget()
                score_row_widget.setLayout(score_row)
                layout.addWidget(score_row_widget)

        # Rows for the user's registered extra peak lists, below the two
        # built-in layers. Its own container so a Settings visit rebuilds
        # only these rows -- the Detected row owns the min-score slider,
        # and tearing that down to add an unrelated layer would be a good
        # way to lose the user's cutoff.
        self._peak_list_rows_widget = QWidget()
        self._peak_list_rows_layout = QVBoxLayout(self._peak_list_rows_widget)
        self._peak_list_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._peak_list_rows_layout.setSpacing(GAP)
        layout.addWidget(self._peak_list_rows_widget)
        self._refresh_peak_list_rows()

        # Matched peaks: master toggle + per-structure rows. The per-structure
        # rows are rebuilt on every frame change because different frames can
        # have different matching solutions.
        matched_master_row = QHBoxLayout()
        matched_master_row.setContentsMargins(0, 0, 0, 0)
        matched_master_row.setSpacing(6)
        # Match the Detected/Fitted layout exactly: those rows put a
        # 26×12 pixmap-bearing QLabel before the checkbox. QLabel
        # renders with different content margins depending on whether
        # it carries a pixmap or not, so a setFixedSize-only label
        # ends up a couple pixels off vertically. Give the matched
        # spacer a transparent pixmap of the same dimensions so its
        # sizing semantics line up byte-for-byte with the real
        # swatches above.
        _ref_swatch = _make_pen_swatch(self.viewer.overlay_pen("detected"))
        _spacer_pixmap = QPixmap(_ref_swatch.size())
        _spacer_pixmap.fill(Qt.GlobalColor.transparent)
        _matched_swatch_spacer = QLabel()
        _matched_swatch_spacer.setPixmap(_spacer_pixmap)
        matched_master_row.addWidget(_matched_swatch_spacer)
        self._matched_master_check = QCheckBox("Matched peaks")
        self._matched_master_check.setChecked(True)
        self._matched_master_check.toggled.connect(self._on_matched_master_toggled)
        # Per-structure checkboxes are rebuilt in _refresh_matched_panel
        # but kept indexed here so the master-toggle cascade and the
        # "single structure on while master off" promotion path can
        # reach them by uid.
        self._matched_struct_checkboxes: dict[str, QCheckBox] = {}
        matched_master_row.addWidget(self._matched_master_check)
        matched_master_row.addStretch(1)
        # Matched display style: boxes (default) vs q-map-style markers
        # (circles for peaks, dashed arcs for rings) — nicer for scan
        # playback where varying box sizes are distracting.
        matched_master_row.addWidget(QLabel("style:"))
        self._matched_style_combo = QComboBox()
        self._matched_style_combo.addItem("Boxes", userData="boxes")
        self._matched_style_combo.addItem("Markers", userData="markers")
        self._matched_style_combo.setToolTip(
            "How matched structures are drawn on the image: 'Boxes' "
            "(per-peak boxes) or 'Markers' (hollow circles for peaks, "
            "dashed arcs for rings — the q-map look, steadier during "
            "playback since it ignores per-peak box size)."
        )
        self._matched_style_combo.currentIndexChanged.connect(
            self._on_matched_style_changed
        )
        matched_master_row.addWidget(self._matched_style_combo)
        matched_master_widget = QWidget()
        matched_master_widget.setLayout(matched_master_row)
        layout.addWidget(matched_master_widget)

        # Substring filter for the per-structure rows below. Useful
        # when matching has been run against a folder of many CIFs
        # (``cif_organic`` has 32 entries — 32 rows is hard to scan
        # by eye). Filter is case-insensitive, applied live as the
        # user types. Indented to align with the structure rows.
        matched_filter_row = QHBoxLayout()
        matched_filter_row.setContentsMargins(20, 0, 0, 0)
        matched_filter_row.setSpacing(6)
        matched_filter_row.addWidget(QLabel("Filter:"))
        self._matched_filter_edit = QLineEdit()
        self._matched_filter_edit.setPlaceholderText("CIF name substring…")
        self._matched_filter_edit.setClearButtonEnabled(True)
        self._matched_filter_edit.textChanged.connect(self._apply_matched_filter)
        matched_filter_row.addWidget(self._matched_filter_edit, 1)
        matched_filter_widget = QWidget()
        matched_filter_widget.setLayout(matched_filter_row)
        layout.addWidget(matched_filter_widget)

        # Min-probability slider — hides matched rows whose structure
        # probability falls below the cutoff. Composes with the
        # CIF-substring filter above and the per-structure visibility
        # checkboxes below. Integer slider 0–100 represents a 0.00–1.00
        # threshold; rendered live next to the slider for readability.
        prob_row = QHBoxLayout()
        prob_row.setContentsMargins(20, 0, 0, 0)
        prob_row.setSpacing(6)
        prob_row.addWidget(QLabel("Min p:"))
        self._matched_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self._matched_prob_slider.setRange(0, 100)
        self._matched_prob_slider.setValue(0)
        self._matched_prob_slider.setToolTip(
            "Hide matched structures whose probability is below the "
            "cutoff. Composes with the CIF-name filter above."
        )
        self._matched_prob_slider.valueChanged.connect(
            self._on_matched_prob_changed
        )
        prob_row.addWidget(self._matched_prob_slider, 1)
        self._matched_prob_value_label = QLabel("0.00")
        self._matched_prob_value_label.setMinimumWidth(36)
        prob_row.addWidget(self._matched_prob_value_label)
        prob_widget = QWidget()
        prob_widget.setLayout(prob_row)
        layout.addWidget(prob_widget)

        # Container for the dynamic per-structure rows. Indented so it reads
        # as a sub-list of the master toggle.
        self._matched_struct_container = QWidget()
        self._matched_struct_layout = QVBoxLayout(self._matched_struct_container)
        self._matched_struct_layout.setContentsMargins(20, 0, 0, 0)
        self._matched_struct_layout.setSpacing(2)
        layout.addWidget(self._matched_struct_container)
        # Per-uid row widgets — used by _apply_matched_filter to
        # show / hide individual rows without rebuilding from data.
        self._matched_struct_rows: dict[str, QWidget] = {}
        # Per-uid structure probability snapshot. Populated in
        # ``_refresh_matched_panel`` and consumed by the min-p
        # slider filter in ``_apply_matched_filter``.
        self._matched_struct_probs: dict[str, float] = {}
        # Shown when a non-empty filter hides every row (distinct
        # from the "no matched solutions" empty-list label, which is
        # torn down with the rest of the rows on every rebuild).
        self._matched_filter_empty_label: QLabel | None = None

        # Twin legend for the Expected-pattern dock: the same master
        # toggle and per-structure rows, rebuilt alongside the Display
        # ones in _refresh_matched_panel with checkbox state mirrored
        # both ways, so structures can be shown/hidden without
        # switching back to the Display tab. Built here so the initial
        # _refresh_matched_panel call below can populate it; the
        # container joins the Expected-pattern dock's layout when that
        # dock is assembled. Deliberately NOT part of
        # _sim_section_container: matched structures exist (and stay
        # toggleable) even while no CIF cache is parsed and the rest
        # of that section is disabled.
        self._sim_legend_container = QWidget()
        sim_legend_box = QVBoxLayout(self._sim_legend_container)
        sim_legend_box.setContentsMargins(0, 0, 0, 0)
        sim_legend_box.setSpacing(2)
        sim_legend_master_row = QHBoxLayout()
        sim_legend_master_row.setContentsMargins(0, 0, 0, 0)
        sim_legend_master_row.setSpacing(6)
        self._sim_legend_master_check = QCheckBox("Matched peaks")
        self._sim_legend_master_check.setChecked(
            self._matched_master_check.isChecked()
        )
        self._sim_legend_master_check.setToolTip(
            "Mirror of the Display tab's Matched-peaks master toggle "
            "(the two stay in sync): show or hide every "
            "matched-structure overlay. The CIF-name filter and the "
            "min-probability slider live in the Display tab and also "
            "apply to this list."
        )
        self._sim_legend_master_check.toggled.connect(
            self._on_matched_master_toggled
        )
        sim_legend_master_row.addWidget(self._sim_legend_master_check)
        sim_legend_master_row.addStretch(1)
        sim_legend_master_widget = QWidget()
        sim_legend_master_widget.setLayout(sim_legend_master_row)
        sim_legend_box.addWidget(sim_legend_master_widget)
        self._sim_legend_struct_container = QWidget()
        self._sim_legend_struct_layout = QVBoxLayout(
            self._sim_legend_struct_container
        )
        self._sim_legend_struct_layout.setContentsMargins(20, 0, 0, 0)
        self._sim_legend_struct_layout.setSpacing(2)
        sim_legend_box.addWidget(self._sim_legend_struct_container)
        self._sim_legend_struct_checkboxes: dict[str, QCheckBox] = {}
        self._sim_legend_struct_rows: dict[str, QWidget] = {}
        self._sim_legend_filter_empty_label: QLabel | None = None
        # Every "Unmatched fitted peaks" checkbox currently built (one
        # per legend) — kept for cross-legend sync in
        # _on_unmatched_row_toggled.
        self._unmatched_row_checks: list[QCheckBox] = []

        self._refresh_matched_panel(0, [])
        self.viewer.matchedStructuresChanged.connect(self._refresh_matched_panel)
        # The legend swatch has to follow the image, including
        # after "Automatic" puts the preset back.
        self.viewer.overlayPenChanged.connect(self._refresh_overlay_swatch)

        # "Expected pattern": forward-simulated reflections of a parsed
        # CIF, rendered as a display-only overlay. Fed by the Pipeline
        # panel's cached CifPattern (Parse CIFs) — the whole section is
        # disabled until a cache exists. Wrapped in one container so
        # enable/disable + tooltip reach every row at once. The
        # cifCacheChanged connection is made after the Pipeline panel
        # is constructed (it doesn't exist yet at this point).
        sim_box = QVBoxLayout()
        sim_box.setContentsMargins(0, 0, 0, 0)
        sim_box.setSpacing(2)
        sim_master_row = QHBoxLayout()
        sim_master_row.setContentsMargins(0, 0, 0, 0)
        sim_master_row.setSpacing(6)
        sim_swatch = QLabel()
        sim_swatch.setPixmap(_make_pen_swatch({
            "color": SIM_MISSED_COLOR,
            "style": Qt.PenStyle.DashLine,
            "width": 1.5,
        }))
        sim_master_row.addWidget(sim_swatch)
        self._sim_master_check = QCheckBox("Show expected pattern")
        self._sim_master_check.setChecked(False)
        self._sim_master_check.setToolTip(
            "Overlay the forward-simulated reflections of a parsed CIF "
            "(diamonds for spots, dashed arcs for rings; marker size "
            "encodes simulated intensity). Display-only — nothing is "
            "written to the file."
        )
        self._sim_master_check.toggled.connect(self._on_sim_master_toggled)
        sim_master_row.addWidget(self._sim_master_check)
        sim_master_row.addStretch(1)
        sim_master_widget = QWidget()
        sim_master_widget.setLayout(sim_master_row)
        sim_box.addWidget(sim_master_widget)

        sim_cif_row = QHBoxLayout()
        sim_cif_row.setContentsMargins(20, 0, 0, 0)
        sim_cif_row.setSpacing(6)
        sim_cif_row.addWidget(QLabel("CIF:"))
        self._sim_cif_combo = QComboBox()
        self._sim_cif_combo.setToolTip(
            "Structure to simulate, from the parsed CIF cache. Entries "
            "matched on the current frame are marked."
        )
        self._sim_cif_combo.currentIndexChanged.connect(
            self._on_sim_cif_changed
        )
        sim_cif_row.addWidget(self._sim_cif_combo, 1)
        sim_cif_widget = QWidget()
        sim_cif_widget.setLayout(sim_cif_row)
        sim_box.addWidget(sim_cif_widget)

        sim_orient_row = QHBoxLayout()
        sim_orient_row.setContentsMargins(20, 0, 0, 0)
        sim_orient_row.setSpacing(6)
        sim_orient_row.addWidget(QLabel("Orientation:"))
        self._sim_orient_mode = QComboBox()
        self._sim_orient_mode.addItem("matched", userData="matched")
        self._sim_orient_mode.addItem(
            "random (powder rings)", userData="random"
        )
        self._sim_orient_mode.addItem(
            "user-specified hkl", userData="user"
        )
        self._sim_orient_mode.setToolTip(
            "How to pick the texture orientation to render: 'matched' "
            "follows the structure's matched orientation(s) on the "
            "current frame; 'random' is the orientation-free powder "
            "ring pattern (randomly oriented crystallites); "
            "'user-specified' renders the (h k l) you type."
        )
        self._sim_orient_mode.currentIndexChanged.connect(
            self._on_sim_orient_mode_changed
        )
        sim_orient_row.addWidget(self._sim_orient_mode)
        self._sim_matched_combo = QComboBox()
        self._sim_matched_combo.setToolTip(
            "Matched orientation to render, labelled with its matching "
            "probability on the current frame."
        )
        self._sim_matched_combo.currentIndexChanged.connect(
            self._on_sim_matched_changed
        )
        sim_orient_row.addWidget(self._sim_matched_combo, 1)
        self._sim_hkl_edit = QLineEdit()
        self._sim_hkl_edit.setPlaceholderText("h k l   e.g. 0 0 1 or 001")
        self._sim_hkl_edit.setToolTip(
            "Miller indices of the orientation to render: three "
            "integers separated by spaces or commas, or three single "
            "digits written together (001, -110, 1-10). Validated "
            "against the structure's precomputed orientations; "
            "equivalent spellings (2 2 0, 0 0 -1) resolve to the "
            "simulated one (1 1 0, 0 0 1)."
        )
        self._sim_hkl_edit.editingFinished.connect(
            self._on_sim_hkl_edited
        )
        self._sim_hkl_edit.setVisible(False)
        sim_orient_row.addWidget(self._sim_hkl_edit, 1)
        sim_orient_widget = QWidget()
        sim_orient_widget.setLayout(sim_orient_row)
        sim_box.addWidget(sim_orient_widget)
        # One-line state/validation hint under the orientation row:
        # invalid typed hkl, equivalent-spelling note, or "no matched
        # orientation on this frame". Hidden when there is nothing to
        # say.
        self._sim_orient_hint = QLabel()
        self._sim_orient_hint.setContentsMargins(20, 0, 0, 0)
        self._sim_orient_hint.setWordWrap(True)
        self._sim_orient_hint.setVisible(False)
        sim_box.addWidget(self._sim_orient_hint)

        sim_int_row = QHBoxLayout()
        sim_int_row.setContentsMargins(20, 0, 0, 0)
        sim_int_row.setSpacing(6)
        sim_int_row.addWidget(QLabel("Min int:"))
        # Structure-factor intensities span decades, so a linear slider
        # is useless here (everything interesting sits between 0 and
        # ~1% of the strongest reflection). A lone spinbox with
        # adaptive-decimal stepping instead: the arrow/wheel step is
        # always one decade below the current value's magnitude, so it
        # moves in useful increments at 0.01% just as at 50%.
        self._sim_int_spin = QDoubleSpinBox()
        self._sim_int_spin.setRange(0.0, 100.0)
        self._sim_int_spin.setDecimals(4)
        self._sim_int_spin.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )
        self._sim_int_spin.setSuffix(" %")
        self._sim_int_spin.setValue(1.0)
        self._sim_int_spin.setToolTip(
            "Hide simulated reflections whose intensity is below this "
            "percentage of the pattern's strongest reflection. "
            "Arrows/wheel step relative to the current value, so tiny "
            "cutoffs stay reachable."
        )
        self._sim_int_spin.valueChanged.connect(
            self._on_sim_int_spin_changed
        )
        sim_int_row.addWidget(self._sim_int_spin, 1)
        sim_int_widget = QWidget()
        sim_int_widget.setLayout(sim_int_row)
        sim_box.addWidget(sim_int_widget)

        # Selection row: bulk-select the reflections the data does not
        # account for, plus a live count of what's queued. Clicking
        # individual markers on the image toggles them one by one.
        sim_sel_row = QHBoxLayout()
        sim_sel_row.setContentsMargins(20, 0, 0, 0)
        sim_sel_row.setSpacing(6)
        self._sim_select_missed_btn = QPushButton("Select missed")
        self._sim_select_missed_btn.setToolTip(
            "Select every visible simulated reflection that is neither "
            "explained by a matched peak nor covered by any fitted "
            "peak on this frame. Click markers on the image to toggle "
            "reflections individually."
        )
        self._sim_select_missed_btn.clicked.connect(
            lambda: self.viewer.select_missed_simulated(
                self.viewer.current_frame
            )
        )
        sim_sel_row.addWidget(self._sim_select_missed_btn)
        self._sim_sel_count_label = QLabel("0 selected")
        sim_sel_row.addWidget(self._sim_sel_count_label)
        sim_sel_row.addStretch(1)
        sim_sel_widget = QWidget()
        sim_sel_widget.setLayout(sim_sel_row)
        sim_box.addWidget(sim_sel_widget)
        self.viewer.simulationSelectionChanged.connect(
            self._on_sim_selection_changed
        )

        # The action button: inject the selected reflections as real
        # detected boxes, 2D-fit each, re-match the frame. Enabled only
        # with a non-empty selection on a frame that already has
        # detected/fitted datasets, while the pipeline is idle.
        sim_add_row = QHBoxLayout()
        sim_add_row.setContentsMargins(20, 0, 0, 0)
        sim_add_row.setSpacing(6)
        self._sim_add_btn = set_variant(
            QPushButton("Add selected peaks (fit + match)"), PRIMARY)
        self._sim_add_btn.setEnabled(False)
        self._sim_add_btn.clicked.connect(self._on_add_predicted_peaks)
        sim_add_row.addWidget(self._sim_add_btn)
        sim_add_row.addStretch(1)
        sim_add_widget = QWidget()
        sim_add_widget.setLayout(sim_add_row)
        sim_box.addWidget(sim_add_widget)

        # The all-frames sweep: every matched (CIF, orientation) combo
        # found anywhere in the entry becomes a template applied to
        # EVERY frame — "here should be a peak, can we find it?".
        # Independent of the overlay checkbox and of the reflection
        # selection.
        sim_sweep_row = QHBoxLayout()
        sim_sweep_row.setContentsMargins(20, 0, 0, 0)
        sim_sweep_row.setSpacing(6)
        self._sim_sweep_btn = QPushButton(
            "Find all matched structures (all frames)"
        )
        self._sim_sweep_btn.clicked.connect(self._on_fill_all_matched)
        sim_sweep_row.addWidget(self._sim_sweep_btn)
        sim_sweep_row.addStretch(1)
        sim_sweep_widget = QWidget()
        sim_sweep_widget.setLayout(sim_sweep_row)
        sim_box.addWidget(sim_sweep_widget)
        # The per-structure reflection caps live in the sweep's
        # confirmation dialog (one spinbox per found structure) — a
        # permanent cap row here would crowd the dock for a setting
        # that only matters at launch time.
        self._update_sim_add_button()

        self._sim_section_container = QWidget()
        self._sim_section_container.setLayout(sim_box)
        self._sim_section_container.setEnabled(False)
        # Not added to the Display panel: the whole Expected-pattern
        # workflow lives in its own tabbed dock (built right after the
        # Pipeline dock below).
        # (stem, hkl) of the pattern currently installed in the viewer
        # (None = overlay clear). Lets frame changes rebuild the combo
        # labels without re-applying an unchanged pattern — re-applying
        # would drop the reflection selection.
        self._sim_applied: tuple | None = None
        # User-typed orientation, kept only when it validated against
        # the selected CIF's precomputed set (None = empty/invalid).
        self._sim_user_hkl: tuple | None = None
        # Until the user touches the mode combo, the mode follows the
        # data: "matched" when the frame has matched orientations for
        # the CIF, "random" (powder) otherwise — the pre-mode default.
        self._sim_mode_auto = True
        self.viewer.matchedStructuresChanged.connect(
            self._refresh_sim_matched_entries
        )

        layout.addSpacing(6)

        self.parameter_panel = ParameterPanel(self)
        # So a list registered in a previous session reads through its
        # treat-as flavour before Settings is ever opened.
        self.parameter_panel.set_peak_list_specs(
            self.viewer.peak_list_specs()
        )
        layout.addWidget(self.parameter_panel)

        # Note: the long polar-mode hint that used to live here has
        # moved to **Help → Controls & shortcuts…** to free up
        # vertical space in the Display dock.

        layout.addStretch(1)

        # Wrap the dock content in a QScrollArea so files with many
        # matched structures (one row per (CIF, hkl) match) don't push
        # the parameter panel and shortcut hint off the bottom of the
        # screen. Vertical scrolling kicks in on demand; horizontal is
        # locked off so narrow docks wrap their form rows instead of
        # introducing an x-axis scrollbar.
        display_scroll = QScrollArea(self)
        display_scroll.setWidgetResizable(True)
        display_scroll.setFrameShape(QFrame.Shape.NoFrame)
        display_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        display_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        display_scroll.setWidget(panel)
        self._display_dock = QDockWidget("Display", self)
        self._display_dock.setWidget(display_scroll)
        self._display_dock.setObjectName("DisplayDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._display_dock)

        # Pipeline dock — tabified with Display on the right.
        self.pipeline_panel = PipelinePanel(self)
        # Let the panel resolve "Active entry" / "Active frame" at click time
        # without pulling MainWindow into its imports. Returning None for
        # either falls through to mlgidBASE's all-entries / all-frames default.
        self.pipeline_panel.set_active_entry_resolver(
            lambda: self.entry_combo.currentText() or None
        )
        self.pipeline_panel.set_active_frame_resolver(
            lambda: self.viewer.current_frame if self.session is not None else None
        )
        self.pipeline_panel.set_frame_count_resolver(
            lambda: self.viewer.n_frames if self.session is not None else None
        )
        self.pipeline_panel.runRequested.connect(self._on_run_requested)
        self.pipeline_panel.parseCifsRequested.connect(self._on_parse_cifs_requested)
        self.pipeline_panel.cifCacheChanged.connect(self._on_cif_cache_changed)
        self._update_sim_section_state()
        self._pipeline_dock = QDockWidget("Pipeline", self)
        self._pipeline_dock.setWidget(self.pipeline_panel)
        self._pipeline_dock.setObjectName("PipelineDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._pipeline_dock)
        self.tabifyDockWidget(self._display_dock, self._pipeline_dock)

        # Expected pattern dock — the forward-simulation overlay plus
        # the add/sweep workflow, in its own right-side tab so the
        # Display dock stays a pure view-settings panel. Same scroll
        # treatment as Display.
        sim_panel = QWidget()
        sim_panel_layout = QVBoxLayout(sim_panel)
        sim_panel_layout.setContentsMargins(8, 8, 8, 8)
        sim_panel_layout.addWidget(self._sim_section_container)
        # The matched-structures twin legend (built next to the Display
        # legend above) — outside the section container so it stays
        # enabled even when the simulation workflow is greyed out.
        sim_panel_layout.addSpacing(8)
        sim_panel_layout.addWidget(self._sim_legend_container)
        sim_panel_layout.addStretch(1)
        sim_scroll = QScrollArea(self)
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sim_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        sim_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sim_scroll.setWidget(sim_panel)
        self._sim_dock = QDockWidget("Expected pattern", self)
        self._sim_dock.setWidget(sim_scroll)
        self._sim_dock.setObjectName("ExpectedPatternDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sim_dock)
        self.tabifyDockWidget(self._pipeline_dock, self._sim_dock)

        # Peaks panel — sortable per-frame view of detected / fitted /
        # matched peaks with bidirectional click-sync to the image
        # viewer's selection. The dock itself is built further down
        # so it can be tabified with the Profile dock at the bottom
        # of the window (the two are read together — peak row +
        # cross-section profile — so sharing a tab area is more
        # practical than burying Peaks among the right-side
        # control docks).
        self.peaks_table_panel = PeaksTablePanel(self)

        # Conversion dock — mode-exclusive sibling of the Pipeline dock.
        # Visible only when the active session is a RawSession; switching
        # between Nexus and Raw sessions hides one and shows the other.
        # Both share the same dock slot (tabified with Display) so the
        # right side never grows beyond two visible tabs.
        self.conversion_panel = ConversionPanel(self)
        self.conversion_panel.conversionRunRequested.connect(
            self._on_conversion_run
        )
        # Let the conversion panel's in-GUI calibration dialog ask
        # for the currently displayed raw frame so it can pre-load
        # the user's image without an extra browse step. Returns
        # None when no raw session is active or the viewer hasn't
        # been populated yet.
        self.conversion_panel.set_active_raw_frame_resolver(
            self._active_raw_frame_for_calibration
        )
        # Flip the live raw preview when the panel's fliplr/flipud checkboxes
        # toggle, so the user sees the orientation the conversion will produce.
        self.conversion_panel.rawFlipsChanged.connect(self._on_raw_flips_changed)
        self._conversion_dock = QDockWidget("Conversion", self)
        self._conversion_dock.setWidget(self.conversion_panel)
        self._conversion_dock.setObjectName("ConversionDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._conversion_dock)
        # Peaks is no longer on the right side, so the chain is just
        # Display | Pipeline | Conversion | Logs. Conversion is
        # tabified with Pipeline (they're mode-exclusive siblings)
        # so the visible tab triplet is Display | <Pipeline or
        # Conversion> | Logs.
        self.tabifyDockWidget(self._sim_dock, self._conversion_dock)
        # Default state matches the default session (none): pipeline dock
        # shown so the user can see what would be available once they
        # open a converted file. ``_apply_session_mode`` handles toggles
        # from then on.
        self._conversion_dock.setVisible(False)

        # Shared Logs dock — tabified next to Display / Pipeline / Conversion.
        # Both panels emit ``logMessage`` / ``logCleared``; we route them
        # through this single widget so the log history is visible in
        # either mode (and a switch from Conversion to NeXus doesn't hide
        # the running log).
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("monospace"))
        self._log_view.setMaximumBlockCount(4000)
        self._log_view.setPlaceholderText(
            "Pipeline and conversion logs land here."
        )
        self._logs_dock = QDockWidget("Logs", self)
        self._logs_dock.setWidget(self._log_view)
        self._logs_dock.setObjectName("LogsDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._logs_dock)
        self.tabifyDockWidget(self._conversion_dock, self._logs_dock)

        # Route both panels' log messages into the shared widget. Both
        # panels' ``append_log`` / ``clear_log`` already emit these
        # signals — every existing call site keeps working.
        self.pipeline_panel.logMessage.connect(self._log_view.appendPlainText)
        self.pipeline_panel.logCleared.connect(self._log_view.clear)
        self.conversion_panel.logMessage.connect(self._log_view.appendPlainText)
        self.conversion_panel.logCleared.connect(self._log_view.clear)

        # Remember which right-hand tab the user has in front, so a
        # session switch that re-tabifies the column can put it back
        # (``_apply_session_mode``). A tabbed dock emits
        # ``visibilityChanged(True)`` when its tab becomes current.
        self._right_front_dock = None
        for dock in (self._display_dock, self._pipeline_dock, self._sim_dock,
                     self._conversion_dock, self._logs_dock):
            dock.visibilityChanged.connect(
                lambda visible, d=dock: self._note_right_front_dock(d, visible))
        self._pipeline_dock.raise_()

        # Bottom: profile viewer + peaks table, tabified together.
        # Default to ~30% of window height so the central image
        # stays the main focus. Profile is the first tab (raised)
        # because the live cross-section is more frequently read
        # than the peak table; the peak table sits behind it,
        # one click away.
        self.profile_viewer = ProfileViewer(self)
        self.profile_viewer.set_neighbour_provider(self._profile_neighbour_boxes)
        # The neighbour mask leaves out peaks pygidfit would fit together
        # with the selected one; that distance is the Pipeline panel's.
        # (A backend-less install builds a stub panel without the fitting
        # form: no provider then, and the viewer uses the default.)
        _dist = getattr(self.pipeline_panel, "fit_dist_peaks", None)
        if _dist is not None:
            self.profile_viewer.set_cluster_distance_provider(
                lambda: float(_dist.value())
            )
            _dist.valueChanged.connect(
                lambda _v: self.profile_viewer.refresh()
            )
        self._profile_dock = QDockWidget("Profiles", self)
        self._profile_dock.setWidget(self.profile_viewer)
        self._profile_dock.setObjectName("ProfileDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._profile_dock)
        self._peaks_dock = QDockWidget("Peaks", self)
        self._peaks_dock.setWidget(self.peaks_table_panel)
        self._peaks_dock.setObjectName("PeaksDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._peaks_dock)
        self.tabifyDockWidget(self._profile_dock, self._peaks_dock)
        # Scan tracking joins the bottom tab group behind Profiles and
        # Peaks — it shares their per-peak viewing context (click a
        # track, see the peak) and, like them, only means something for
        # converted multi-frame data.
        self.scan_tracking_panel = ScanTrackingPanel(self)
        self._scan_tracking_dock = QDockWidget("Scan tracking", self)
        self._scan_tracking_dock.setWidget(self.scan_tracking_panel)
        self._scan_tracking_dock.setObjectName("ScanTrackingDock")
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self._scan_tracking_dock
        )
        self.tabifyDockWidget(self._peaks_dock, self._scan_tracking_dock)
        self.scan_tracking_panel.loadSavedRequested.connect(
            lambda: self._request_saved_tracking(force=True)
        )
        self.scan_tracking_panel.trackRequested.connect(
            self._on_track_scan_requested
        )
        self.scan_tracking_panel.showViewsRequested.connect(
            self._on_show_phase_views
        )
        self.scan_tracking_panel.trackRowSelected.connect(
            self._on_track_row_selected
        )
        self.scan_tracking_panel.onlyTrackedToggled.connect(
            self._on_only_tracked_toggled
        )
        self.scan_tracking_panel.interpolateRequested.connect(
            self._on_interpolate_tracks_requested
        )
        self.scan_tracking_panel.trackDeleteRequested.connect(
            self._on_scan_track_deleted
        )
        # Image -> table sync: selecting a fitted peak anywhere (image
        # click or Peaks-dock row) highlights its cross-frame track row
        # and surfaces the frame span in the status bar.
        self.viewer.selectionChanged.connect(
            self._forward_selection_to_scan_tracking
        )
        self._profile_dock.raise_()
        self.resizeDocks(
            [self._profile_dock], [max(self.height() // 3, 280)], Qt.Orientation.Vertical
        )
        # Default column widths. The file browser stays pinned at 260,
        # which leaves room for typical HDF5 paths (it was previously
        # squeezed to ~100 px and truncated them). The right-hand column
        # is measured instead of pinned — see
        # ``_preferred_right_dock_width``.
        self._apply_default_dock_widths()
        self.viewer.frameChanged.connect(self.profile_viewer.set_frame)
        # Bidirectional Display-dock slider sync: viewer pushes frame
        # changes into the slider (e.g. user scrubs the pyqtgraph
        # timeline below the image), and the slider's valueChanged
        # already pushes back into the viewer via _on_frame_slider_changed.
        self.viewer.frameChanged.connect(self._on_viewer_frame_changed)
        # Bidirectional sync between 2D ROI and profile-edge regions. The
        # profile viewer only handles ManualPeak, so we filter the
        # SelectedPeak-typed signals down to the manual case before forwarding.
        self.viewer.selectionChanged.connect(self._forward_selection_to_profile)
        self.viewer.peakGeometryChanged.connect(self._forward_geom_to_profile)
        self.profile_viewer.peakGeometryChanged.connect(self.viewer.update_peak_geometry_external)
        # Detected-peak profile region drag — live updates flow into
        # the viewer's in-memory PeakTable so the colored overlay
        # tracks the drag; the disk write fires once on drag-end via
        # _on_detected_border_commit (mirrors how the image-side ROI
        # drag commits via peakRowWriteRequested).
        self.profile_viewer.detectedPeakGeometryChanged.connect(
            self.viewer.update_detected_geometry_external
        )
        self.profile_viewer.detectedPeakBorderCommit.connect(
            self._on_detected_border_commit
        )
        # The faint fitted-preview box for the selected manual peak follows
        # the profile viewer's 1D Gaussian fits. It also has to drop when
        # the selection changes away from a manual peak.
        self.profile_viewer.fitParamsChanged.connect(self._update_fitted_preview)
        self.viewer.selectionChanged.connect(self._on_selection_for_preview)
        # Live readout in the parameter panel — routed through a
        # host dispatcher (instead of straight to ``set_fits``) so
        # 2D mode can substitute pygidfit's refined values for
        # scipy's 1D fits. The dispatcher matches what
        # Add-to-fitted (2D) will actually save, so the readout no
        # longer claims to commit scipy-derived numbers in 2D mode.
        self.profile_viewer.fitParamsChanged.connect(self._dispatch_fit_params)

        # Parameter readout — both selection and geometry changes feed the same slot.
        self.viewer.selectionChanged.connect(self.parameter_panel.set_peak)
        self.viewer.peakGeometryChanged.connect(self.parameter_panel.set_peak)

        # Peaks table sync. Image → table: mirror the selection onto
        # the relevant row (auto-switches tab). Table → image: route
        # row clicks back through the viewer's selection setter.
        self.viewer.selectionChanged.connect(self.peaks_table_panel.set_external_selection)
        self.viewer.frameChanged.connect(self._refresh_peaks_table_on_frame)
        self.peaks_table_panel.peakSelectedFromTable.connect(
            self._on_peak_selected_from_table
        )
        # Ctrl / Shift multi-select in the table drives the same
        # multi-selection Ctrl+click on the image does, and Delete in
        # the table routes into the same handlers the image-side
        # Delete key uses — one confirmation, one write path, one undo
        # entry, whichever half of the window the user is looking at.
        self.peaks_table_panel.peaksSelectedFromTable.connect(
            self._on_peaks_selected_from_table
        )
        self.peaks_table_panel.deletePeaksRequested.connect(
            self._on_delete_peaks_from_table
        )
        self.viewer.selectionsChanged.connect(
            self.peaks_table_panel.set_external_selections
        )

        # Commit / delete actions on the parameter panel. Add-to-detected and
        # delete reuse the existing PipelineWorker path.
        # Quick select: the panel owns the flag, the viewer owns the
        # gesture, and the host owns every file write — so the mode is
        # pushed one way and the commit request comes back the other.
        self.parameter_panel.quickSelectChanged.connect(
            self._on_quick_select_toggled
        )
        self.parameter_panel.quickTargetChanged.connect(
            lambda _t: self._update_status_quick_select()
        )
        self.viewer.manualPeakCommitRequested.connect(
            self._on_quick_commit_requested
        )
        # The pending marker in the status cell has to follow the box
        # appearing and disappearing, whoever caused it.
        self.viewer.manualPeakAdded.connect(
            lambda *_a: self._update_status_quick_select()
        )
        self.viewer.manualPeakRemoved.connect(
            lambda *_a: self._update_status_quick_select()
        )
        self.parameter_panel.addToDetectedRequested.connect(self._on_add_to_detected)
        self.parameter_panel.addToFittedRequested.connect(self._on_add_to_fitted)
        # Batch 2D fit of the multi-selection. The button on the
        # parameter panel emits ``batchFit2DRequested``; the host runs
        # pygidfit inside a modal QProgressDialog (see
        # ``_on_batch_fit_2d``). Enable state is re-evaluated on every
        # selectionsChanged + saveAsRingChanged tick by
        # ``_refresh_fit_buttons``.
        self.parameter_panel.batchFit2DRequested.connect(self._on_batch_fit_2d)
        self.viewer.selectionsChanged.connect(
            lambda _sels: self._refresh_fit_buttons()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._refresh_fit_buttons()
        )
        # Apply once so the cold-start panel reflects the initial
        # empty-selection state (both fit buttons hidden).
        self._refresh_fit_buttons()
        # Refresh the cyan preview overlay immediately when the user
        # toggles ring/segment — otherwise the preview would lag until
        # the next fit recompute.
        self.parameter_panel.saveAsRingChanged.connect(self._on_save_as_ring_changed)
        # Flipping the 1D / 2D fit-mode radios changes how the dashed
        # cyan preview's box widths are computed — re-invoke the
        # preview slot with the cached 1D fits so the box redraws at
        # the new mode's convention immediately, instead of lagging
        # until the next fit recompute (frame change, ROI drag, etc.).
        self.parameter_panel.fitModeChanged.connect(self._on_fit_mode_changed)
        # Live 2D preview: run pygidfit on selection / frame / mode
        # changes so the profile fits + cyan box mirror what
        # Add-to-fitted (2D) will save. Selection already calls
        # ``_refresh_2d_preview`` via ``_on_selection_for_preview``;
        # also wire frame + mode + ring so all four user-triggers
        # refresh the override.
        self.viewer.frameChanged.connect(
            lambda _f: self._refresh_2d_preview()
        )
        self.parameter_panel.fitModeChanged.connect(
            lambda _m: self._refresh_2d_preview()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._refresh_2d_preview()
        )
        # ROI drag-end: the user repositions a manual / detected box
        # to a new peak. ``peakGeometryChanged`` fires per drag tick
        # (too slow for pygidfit), ``peakGeometryDragFinished`` fires
        # once after the handle settles. Cache fingerprint includes
        # the geometry, so the next refresh is a real recompute.
        self.viewer.peakGeometryDragFinished.connect(
            lambda _sel: self._refresh_2d_preview()
        )
        # Live 2D preview during drag: debounce pygidfit runs so the
        # cyan preview + parameter readout track the user's drag.
        # Per-tick refits would freeze the GUI (pygidfit is ~100-
        # 500 ms); a single-shot timer that restarts on every drag
        # tick collapses bursts into one fit every ~150 ms while
        # idle, then a final fit on drag-end via the existing
        # ``peakGeometryDragFinished`` wiring. While the timer is
        # pending, ``_current_pygidfit_box_for_selection`` returns
        # the *cached* refined box (ignoring the geometry fingerprint
        # mismatch) so the cyan box stays at pygidfit's last known
        # result instead of falling through to scipy 1D widths.
        self._drag_pygidfit_timer = make_debounced_timer(
            self, 150, self._refresh_2d_preview
        )
        self.viewer.peakGeometryChanged.connect(
            self._schedule_drag_2d_preview
        )
        # Pink fit curves on the profile plots are hidden only when
        # a manual / detected peak is selected in 2D mode (the
        # source-of-fit case where the 2D-projected pink preview
        # was misleading). Selection changes also re-evaluate so
        # switching to a fitted peak immediately restores its
        # overlay — see ``_apply_fit_curve_visibility``.
        self.parameter_panel.fitModeChanged.connect(
            lambda _m: self._apply_fit_curve_visibility()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._apply_fit_curve_visibility()
        )
        self.viewer.selectionChanged.connect(
            lambda _s: self._apply_fit_curve_visibility()
        )
        # Apply the initial state so a cold-start GUI reflects the
        # parameter panel's default mode + no selection.
        self._apply_fit_curve_visibility()
        self.parameter_panel.deletePeakRequested.connect(
            lambda: self._on_delete_peak_requested(self.viewer.selected_peak)
        )
        # Relabelling: the Score row's editor writes the whole selection.
        self.parameter_panel.scoreEditRequested.connect(
            self._on_score_edit_requested
        )
        self.viewer.selectionsChanged.connect(
            lambda sels: self.parameter_panel.set_selection_count(len(sels))
        )

        # Direct-h5py geometry writes for detected/fitted ROI edits.
        self.viewer.peakRowWriteRequested.connect(self._on_peak_row_write_requested)
        # Delete keypress on file-resident peaks.
        self.viewer.deletePeakRequested.connect(self._on_delete_peak_requested)
        # Delete keypress with >= 2 detected peaks selected → bulk delete.
        self.viewer.deletePeaksRequested.connect(self._on_delete_peaks_requested)

        # Keep _ring_pre_geom in sync with the manual peak it points at.
        # When the user replaces the box (single-box policy) while ring
        # is active, the new box also needs ring expansion; when the
        # box is removed (Esc / Delete / Add-to-detected), the stash
        # goes stale and must be invalidated.
        self.viewer.manualPeakRemoved.connect(self._on_manual_peak_removed)
        self.viewer.manualPeakAdded.connect(self._on_manual_peak_added)

    # --- Registered extra peak lists (Display dock) ---

    def _refresh_peak_list_rows(self) -> None:
        """Rebuild the Display dock's rows for the registered peak lists.

        One row per list: the same pen-editor swatch and visibility
        checkbox the built-in layers get, labelled with the name the
        user gave it. Called at construction and after Settings changes.
        """
        layout = getattr(self, "_peak_list_rows_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for spec in self.viewer.peak_list_specs():
            kind = spec.kind
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QToolButton()
            swatch.setAutoRaise(True)
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(
                f"Click to set the {spec.display_label} colour, line "
                f"style and width"
            )
            swatch.clicked.connect(
                lambda _=False, k=kind: self._open_overlay_pen_popup(k)
            )
            self._overlay_swatches[kind] = swatch
            self._refresh_overlay_swatch(kind)
            row.addWidget(swatch)
            chk = QCheckBox(spec.display_label)
            chk.setToolTip(
                f"Show the {spec.dataset} table from this file's analysis "
                f"group, drawn like {spec.treat_as} peaks. A display "
                f"layer: nothing else in the app reads it."
            )
            chk.setChecked(self.viewer._visibility.get(kind, True))
            chk.toggled.connect(
                lambda v, k=kind: self.viewer.set_overlay_visible(k, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            layout.addWidget(row_widget)
            self._overlay_checks[kind] = chk

    def _apply_peak_list_specs(self) -> None:
        """Re-read the registry after Settings and put it into effect.

        Three things have to move together: the viewer's overlay items,
        the dock's rows, and the frame's peak tables -- a newly
        registered list has never been read from the file.
        """
        specs = peak_lists.load_specs()
        stale = set(self._overlay_swatches) - {"detected", "fitted"}
        for kind in stale:
            self._overlay_swatches.pop(kind, None)
            self._overlay_checks.pop(kind, None)
        self.viewer.set_peak_list_specs(specs)
        self.parameter_panel.set_peak_list_specs(specs)
        self._refresh_peak_list_rows()
        # Force the frame's peaks to be re-read: the cached tables were
        # loaded without the newly registered dataset in them.
        self._loaded_peak_frames = set()
        entry = self.entry_combo.currentText()
        if entry:
            self._load_frame_peaks(entry, int(self.viewer.current_frame))

    # --- Overlay pens (Display dock) ---

    def _refresh_overlay_swatch(self, kind: str) -> None:
        """Redraw one overlay row's swatch from the pen in effect.

        The legend has to follow the image, including after "Automatic"
        puts the preset back, so this is driven by the viewer's
        ``overlayPenChanged`` rather than by the popup.
        """
        swatch = self._overlay_swatches.get(kind)
        if swatch is None:
            return
        pix = _make_pen_swatch(self.viewer.overlay_pen(kind))
        swatch.setIcon(QIcon(pix))
        swatch.setIconSize(pix.size())
        swatch.setFixedSize(pix.width() + 4, pix.height() + 4)

    def _open_overlay_pen_popup(self, kind: str) -> None:
        """Pen editor under a Detected / Fitted swatch."""
        swatch = self._overlay_swatches.get(kind)
        if swatch is None:
            return
        popup = PenPopup(
            self,
            current=self.viewer.overlay_pen(kind),
            override=self.viewer.overlay_pen_override(kind),
            title=f"{kind.capitalize()} peak colour",
        )
        popup.penChanged.connect(
            lambda pen, k=kind: self.viewer.set_overlay_pen(k, pen)
        )
        popup.resetPicked.connect(
            lambda k=kind: self.viewer.set_overlay_pen(k, None)
        )
        popup.show_under(swatch)

    def _hide_stale_dock_tab_bars(self) -> None:
        """Hide ghost dock tab bars.

        Qt quirk: tabifying the Scan-tracking dock into the existing
        Profiles/Peaks bottom group can leave the group's OLD two-tab
        bar behind as a second, stale ``QTabBar`` — painted into the
        file-browser corner of the window. The ghost's signature is a
        MainWindow-child dock tab bar whose tab set is a STRICT subset
        of another dock tab bar's (the real bar of the same group);
        distinct dock groups have disjoint tab sets and never match.
        Ran on show and after every dock re-tabify; a no-op when Qt
        behaves.
        """
        from PySide6.QtWidgets import QTabBar

        infos = [
            (tb, {tb.tabText(i) for i in range(tb.count())})
            for tb in self.findChildren(QTabBar)
            if tb.parent() is self
        ]
        for tb, labs in infos:
            if labs and any(
                labs < other for bar, other in infos if bar is not tb
            ):
                tb.hide()

    def _update_title(self) -> None:
        if self.session is None:
            self.setWindowTitle(APP_NAME)
            self._update_status_file()
            return
        marker = "*" if self.session.dirty else ""
        self.setWindowTitle(
            f"{self.session.original_path.name}{marker} — {APP_NAME}"
        )
        self._update_status_file()

    #: Never open the right-hand column wider than this share of the
    #: window, nor than this many pixels: the image is the point of the
    #: screen, and a dock the user has to drag back is worse than one
    #: they drag out once.
    _RIGHT_DOCK_WINDOW_SHARE = 3
    _RIGHT_DOCK_MAX_PX = 500
    _RIGHT_DOCK_MIN_PX = 380

    def _note_right_front_dock(self, dock, visible: bool) -> None:
        if not visible or dock.isFloating():
            return
        # Folding the docks for the Structure tab brings each next tab
        # forward for a moment on the way out; that is not a choice.
        tabs = getattr(self, "tabs", None)
        structure = getattr(self, "structure_panel", None)
        if tabs is not None and structure is not None \
                and tabs.currentWidget() is structure:
            return
        self._right_front_dock = dock

    def _apply_default_dock_widths(self) -> None:
        """Open the two side columns at their default widths.

        Called once during construction and again on the first show,
        because a request made before the window has geometry is scaled
        down to whatever QMainWindow thinks it has (260 + 466 came out
        as 197 + 403). The second call is the one that lands.
        """
        self.resizeDocks(
            [self._tree_dock, self._display_dock],
            [self._TREE_DOCK_PX, self._preferred_right_dock_width()],
            Qt.Orientation.Horizontal,
        )

    #: File-browser column. Pinned: it holds HDF5 paths, which were
    #: truncated when the area collapsed to ~100 px.
    _TREE_DOCK_PX = 260

    def _preferred_right_dock_width(self) -> int:
        """How wide to open the right-hand dock column.

        Every dock on the right shares one column, so the width that
        matters is the one the fullest panel needs, not the one that
        happens to be in front. It is measured rather than pinned, so a
        panel that grows a longer label moves this with it:

        * each panel's own ``sizeHint``, plus
        * the width of any **closed** ``Card``, via ``open_width_hint``
          — Pipeline's Fitting section wants ~465 px and starts closed,
          so measuring only what is open reports far too little,
        * plus the scroll area's frame and scrollbar.

        The panels sit in resizable scroll areas, so a column that is
        too narrow does not scroll: it compresses and elides. That is
        what the pinned 350 looked like — the Pipeline form's path
        fields (the Matching section's CIF row today; the Detection
        section's "Config (yaml)" field when this was measured)
        squeezed down to a stub.
        """
        docks = (
            self._display_dock, self._pipeline_dock, self._sim_dock,
            self._conversion_dock, self._logs_dock,
        )
        wanted = 0
        for dock in docks:
            content = dock.widget()
            if content is None:
                continue
            scroll = content.findChild(QScrollArea)
            if scroll is not None and scroll.widget() is not None:
                content = scroll.widget()
            wanted = max(wanted, content.sizeHint().width())
            for card in content.findChildren(Card):
                if not card.is_expanded():
                    wanted = max(wanted, card.open_width_hint())
        wanted += self.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent) + 8   # frame + margins
        share = max(self.width() // self._RIGHT_DOCK_WINDOW_SHARE,
                    self._RIGHT_DOCK_MIN_PX)
        return max(
            self._RIGHT_DOCK_MIN_PX,
            min(wanted, share, self._RIGHT_DOCK_MAX_PX),
        )

    def _build_status_bar(self) -> None:
        """Permanent status-bar widgets: file / entry / frame / pipeline + cursor.

        Each label lives in the status bar's permanent-widget slot so
        Qt's transient ``showMessage`` calls (PNG / CSV export confirmations)
        still render correctly alongside them — Qt clears the transient
        message after its timeout but leaves the permanent labels alone.
        """
        sb = self.statusBar()
        # Unsaved-changes dot, in front of the file name. It replaces the
        # "*" the name used to carry: a marker glued to the end of a
        # filename is easy to read as part of the name, and it moved with
        # the text. (The window title keeps its "*" — that is the
        # platform convention for a modified document.)
        self._sb_dirty = QLabel()
        self._sb_dirty.setPixmap(_dirty_dot())
        self._sb_dirty.setToolTip("Unsaved changes")
        self._sb_dirty.hide()
        self._sb_file = QLabel("no file")
        self._sb_entry = QLabel("")
        self._sb_frame = QLabel("")
        self._sb_pipeline = QLabel("idle")
        # Quick-select labelling changes what a drag does, and the
        # Display dock that owns its checkbox is tabbed and scrollable —
        # so the mode needs somewhere it cannot hide. Empty and hidden
        # while the mode is off, which is most of the time.
        self._sb_quick = QLabel("")
        self._sb_quick.hide()
        self._sb_cursor = QLabel("")
        # A run in flight gets its own bar in the row, right after the
        # pipeline cell. Indeterminate until a frame count arrives, which
        # is honest: several ops are one opaque backend call.
        self._sb_pipe_bar = skin_progress(QProgressBar())
        self._sb_pipe_bar.setRange(0, 0)
        self._sb_pipe_bar.setFixedWidth(70)
        self._sb_pipe_bar.setFixedHeight(12)
        self._sb_pipe_bar.setTextVisible(False)
        self._sb_pipe_bar.hide()
        sb.addPermanentWidget(self._sb_dirty)
        for w in (self._sb_file, self._sb_entry, self._sb_frame,
                  self._sb_pipeline, self._sb_pipe_bar, self._sb_quick,
                  self._sb_cursor):
            # Light separation so the eye can scan the row. The divider
            # colour and padding come from the skin, which is why this
            # is a role tag rather than a stylesheet: the old hardcoded
            # #444 stayed dark-grey in the light theme. The file name is
            # the one field that says *what you are looking at*, so it
            # keeps the full text colour while the rest read as context.
            if isinstance(w, QLabel):
                w.setProperty(
                    "role",
                    "sb-cell-active"
                    if w in (self._sb_file, self._sb_quick) else "sb-cell")
            sb.addPermanentWidget(w)
        # The pipeline cell is the one place a run reports from, so make
        # it the way into the log of that run.
        self._sb_pipeline.setToolTip("Pipeline activity — click to open the Logs dock")
        self._sb_pipeline.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sb_pipeline.installEventFilter(self)
        # The cursor readout is the chattiest widget; let it stretch
        # so values don't truncate, others stay tight. Monospace so the
        # numbers stop jittering sideways as the cursor moves.
        self._sb_cursor.setProperty("role", "sb-cell-mono")
        self._sb_cursor.setMinimumWidth(430)
        self.viewer.cursorMoved.connect(self._on_status_cursor_moved)
        self._status_cursor_visible = True
        # Bottom-left, non-modal open-progress: a small busy (indeterminate)
        # bar + a stage label, shown while a file loads in the background
        # (copy + first frame warmed off the GUI thread by CopyWorker). The
        # bar is indeterminate on purpose: the open no longer resolves the
        # external links, so there is no granular per-entry progress to
        # show — a continuously marching bar reads as "smoothly loading"
        # during the (GUI-responsive) worker phase, whereas a determinate
        # bar would just jump between a couple of coarse stages. The
        # worker's ``progress`` ticks drive the LABEL ("Copying file",
        # "Loading first entry"). Non-modal so the window stays usable; the
        # file only appears in the browser once the load finishes
        # (_on_open_finished). addWidget puts these on the left, before the
        # transient showMessage text.
        self._sb_open_label = QLabel("")
        self._sb_open_bar = skin_progress(QProgressBar())
        self._sb_open_bar.setRange(0, 0)          # indeterminate "busy" march
        self._sb_open_bar.setMaximumWidth(110)
        self._sb_open_bar.setTextVisible(False)
        self._sb_open_label.hide()
        self._sb_open_bar.hide()
        sb.addWidget(self._sb_open_label)
        sb.addWidget(self._sb_open_bar)
        # Separate determinate bar for the chunked file-browser fill
        # ("Browser: n/N files"). Its OWN widget pair on purpose: the
        # open bar above is driven concurrently by the async first-image
        # load ("Loading <entry>…" → dismissed on arrival), so sharing
        # it would let that dismiss hide an still-running browser fill.
        self._sb_tree_label = QLabel("")
        self._sb_tree_bar = skin_progress(QProgressBar())
        self._sb_tree_bar.setRange(0, 100)
        self._sb_tree_bar.setMaximumWidth(110)
        self._sb_tree_bar.setTextVisible(False)
        self._sb_tree_label.hide()
        self._sb_tree_bar.hide()
        sb.addWidget(self._sb_tree_label)
        sb.addWidget(self._sb_tree_bar)

    def _update_status_file(self) -> None:
        if self.session is None:
            self._sb_file.setText("no file")
            self._sb_dirty.hide()
            return
        self._sb_file.setText(self.session.original_path.name)
        self._sb_dirty.setVisible(bool(self.session.dirty))

    def _active_raw_frame_for_calibration(self):
        """Return a 2D ndarray to seed the pyFAI calibration dialog.

        For multi-frame raw scans (the typical calibrant case —
        LaB6 / Si / CeO2 measured as a short scan to boost ring
        statistics) this returns the per-pixel mean across all
        frames so faint outer rings come out of the noise. For a
        single-frame stack the lone frame is returned unchanged.

        Returns None when no raw session is active, when the
        viewer is in NeXus mode, or when the raw stack hasn't been
        populated yet — the dialog then opens with an empty image
        slot and the user can browse to a file from inside it.

        Lazy stacks (``LazyRawStack``) are averaged over at most the
        first 20 frames, read one at a time — a calibrant scan is
        short, and materializing an arbitrary beamtime stack here
        would re-introduce the multi-GB GUI-thread read the lazy
        path exists to avoid.
        """
        try:
            stack = getattr(self.viewer, "_raw_image_stack", None)
        except Exception:
            logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
            return None
        if stack is None:
            return None
        if not isinstance(stack, np.ndarray):
            try:
                n = int(stack.shape[0])
                if n == 0:
                    return None
                if n == 1:
                    return np.asarray(stack[0])
                take = min(n, 20)
                acc = np.zeros(stack.shape[1:], dtype=np.float64)
                for i in range(take):
                    acc += stack[i]
                return acc / take
            except Exception:
                logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
                return None
        try:
            arr = np.asarray(stack)
            if arr.ndim != 3 or arr.shape[0] == 0:
                # Defensive: viewer should always hand back a 3D
                # stack in raw mode, but if something upstream
                # changes that contract fall back to whatever the
                # current-frame index points at.
                idx = int(self.viewer.current_frame)
                if 0 <= idx < arr.shape[0]:
                    return arr[idx]
                return arr[0] if arr.shape[0] else None
            if arr.shape[0] == 1:
                return arr[0]
            # Mean in float64 to keep the average stable for high-
            # dynamic-range detector data; pyFAI's image model
            # accepts arbitrary numeric dtypes.
            return arr.mean(axis=0, dtype=np.float64)
        except Exception:
            logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
            return None

    # -- Workflow rail -------------------------------------------------

    #: stage key -> (dock attribute, attribute path of the panel's own
    #: Run button). The rail never builds a command: it clicks the
    #: button the panel already owns, so the kwargs, the enablement and
    #: the queueing all stay in one place.
    _RAIL_TARGETS = {
        "convert": ("_conversion_dock", ("conversion_panel", "btn_convert")),
        "detect": ("_pipeline_dock", ("pipeline_panel", "btn_detect")),
        "fit": ("_pipeline_dock", ("pipeline_panel", "btn_fit")),
        "match": ("_pipeline_dock", ("pipeline_panel", "btn_match")),
        "track": ("_scan_tracking_dock", ("scan_tracking_panel", "btn_track")),
    }

    def _rail_run_button(self, key: str):
        target = self._RAIL_TARGETS.get(key)
        if target is None:
            return None
        panel_attr, button_attr = target[1]
        panel = getattr(self, panel_attr, None)
        return getattr(panel, button_attr, None) if panel is not None else None

    def _on_rail_stage_activated(self, key: str) -> None:
        """Bring the dock that owns ``key`` forward."""
        target = self._RAIL_TARGETS.get(key)
        if target is None:
            return
        dock = getattr(self, target[0], None)
        if dock is not None:
            dock.show()
            dock.raise_()

    def _on_rail_stage_run(self, key: str) -> None:
        button = self._rail_run_button(key)
        if button is not None and button.isEnabled():
            # Read per click, so the Settings dialog applies at once.
            if read_bool(RAIL_RUN_OPENS_DOCK_KEY, True):
                self._on_rail_stage_activated(key)
            button.click()

    def _refresh_workflow_rail(self) -> None:
        """Re-read the current frame's tables and retag the stages.

        Counts are per frame, from what the overlays already hold — no
        extra I/O, and no claim about the rest of the scan.
        """
        rail = getattr(self, "workflow_rail", None)
        if rail is None:
            return
        self._sync_scan_tracking_visibility()
        session = self.session
        rail.setVisible(session is not None)
        if session is None:
            return
        rail.set_mode(session.kind)

        running = self._pipe_thread is not None
        for key in self._RAIL_TARGETS:
            button = self._rail_run_button(key)
            rail.set_runnable(
                key, bool(button is not None and button.isEnabled()))

        if session.kind == "raw":
            rail.set_state("convert", "ready" if not running else "running…",
                           "run" if running else "muted")
            for key in ("detect", "fit", "match", "track"):
                rail.set_state(key, "after conversion", "muted")
            return

        rail.set_state("convert", "done", "ok")
        frame = int(getattr(self.viewer, "current_frame", 0))
        peaks = getattr(self.viewer, "_frame_peaks", {}).get(frame) or {}
        counts = {
            kind: (0 if table is None else int(len(table.ids)))
            for kind, table in peaks.items()
        }
        matched = len(self.viewer.matched_structures(frame))
        payload = getattr(self, "_scan_payload", None)
        tracks = len(getattr(payload, "components", None) or []) if payload else 0

        for key, count, empty in (
            ("detect", counts.get("detected", 0), "not run"),
            ("fit", counts.get("fitted", 0), "not run"),
            ("match", matched, "not run"),
        ):
            if count:
                rail.set_state(key, f"{count} this frame", "ok")
            else:
                rail.set_state(key, empty, "muted")
        rail.set_state("track", f"{tracks} tracks" if tracks else "not run",
                       "ok" if tracks else "muted")

    def _refresh_welcome_view(self) -> None:
        """Re-seed the welcome page's theme-dependent and live content."""
        view = getattr(self, "welcome_view", None)
        if view is None:
            return
        view.set_theme(getattr(self, "_current_theme", "dark"))
        view.set_recent(self._load_recent_files())
        view.set_backend_available(is_mlgidbase_available())

    def _update_status_entry(self) -> None:
        entry = self.entry_combo.currentText() if hasattr(self, "entry_combo") else ""
        self._sb_entry.setText(entry or "")

    def _update_status_frame(self) -> None:
        n = getattr(self.viewer, "n_frames", 0)
        if n <= 0:
            self._sb_frame.setText("")
            return
        cur = int(getattr(self.viewer, "current_frame", 0))
        # Frames are 0-indexed everywhere else in the GUI (peak rows,
        # NeXus group keys), so the denominator is the max index
        # ``n - 1``, not the count. 17-frame stack → "frame 16 / 16"
        # at the end. Single-frame entries elide the "/ total" since
        # there's no navigation possible.
        if n == 1:
            self._sb_frame.setText(f"frame {cur}")
        else:
            self._sb_frame.setText(f"frame {cur} / {n - 1}")
        # The rail's counts are per frame, so they move with this.
        self._refresh_workflow_rail()

    def _set_pipeline_running(self, running: bool) -> None:
        """Colour the pipeline cell and show/hide its bar.

        The ``status`` tag rides alongside the cell's ``role`` tag; the
        skin pairs the two so a running cell is accented while an idle
        one keeps the muted context colour.
        """
        self._sb_pipeline.setProperty("status", "run" if running else "")
        style = self._sb_pipeline.style()
        style.unpolish(self._sb_pipeline)
        style.polish(self._sb_pipeline)
        self._sb_pipe_bar.setVisible(running)
        if not running:
            # Back to the busy marquee for the next run, whose frame
            # count is not known until its first progress tick.
            self._sb_pipe_bar.setRange(0, 0)

    def _update_status_quick_select(self) -> None:
        """Show the quick-select mode, and whether a box is pending.

        The pending marker is the point: with the mode on, a box that
        has not been committed yet is the one piece of state the image
        alone does not make obvious (a manual box looks like a
        selection), and it is what a frame change or a click is about
        to write.
        """
        panel = getattr(self, "parameter_panel", None)
        cell = getattr(self, "_sb_quick", None)
        if cell is None or panel is None:
            return
        if not panel.quick_select_enabled():
            cell.hide()
            cell.setText("")
            return
        target = panel.quick_select_target()
        pending = self.viewer.pending_manual_peak() is not None
        cell.setText(f"quick: {target}" + (" • 1 pending" if pending else ""))
        cell.setToolTip(
            "Quick select is on: drawing the next box commits the "
            "previous one as a "
            f"{target} peak."
            + (
                "\nOne box is waiting — it commits when you draw the "
                "next one, click away, press Enter or change frame."
                if pending else ""
            )
        )
        cell.show()

    def _update_status_pipeline(self, command=None, *, running: bool) -> None:
        self._set_pipeline_running(running)
        # A finished run changes what the stages can report, and a
        # started one disables their run glyphs.
        self._refresh_workflow_rail()
        if not running:
            self._sb_pipeline.setText("idle")
            # Drop any progress tail from the previous run so a stale
            # "3/12 frames" counter doesn't haunt the status bar after
            # an op finishes.
            self._pipe_progress_tail = ""
            return
        if command is None:
            self._sb_pipeline.setText("running…")
            return
        op = command.op_name if hasattr(command, "op_name") else str(command)
        entry = command.kwargs.get("entry") if hasattr(command, "kwargs") else None
        # Fold in the entry-queue position and the most recent
        # ``frameProgress`` tail when each is known. Multi-entry runs
        # get "· entry K/N"; multi-frame runs get "· K/N frames";
        # single-of-both contributes nothing.
        head = f"running: {op} on {entry}" if entry else f"running: {op}"
        entry_tail = ""
        if getattr(self, "_entry_queue_total", 0) > 1:
            entry_tail = (
                f" · entry {self._entry_queue_pos}/{self._entry_queue_total}"
            )
        frame_tail = getattr(self, "_pipe_progress_tail", "")
        self._sb_pipeline.setText(head + entry_tail + frame_tail)

    def _on_pipeline_frame_progress(
        self, done: int, total: int, op_name: str, entry: str
    ) -> None:
        """Mirror ``PipelineWorker.frameProgress`` into the status bar.

        Single-frame and indeterminate runs (``total <= 1``) clear the
        tail so the existing "running: op on entry" remains unadorned.
        Skips the status-bar repaint when the tail string is unchanged
        from the last emit — a fast pipeline can fire many
        ``frameProgress`` signals per second and an unchanged
        ``setText`` still schedules a paint event.
        """
        # Interpolate-track chain: drive the loading dialog with the
        # SAME per-frame ticks the Pipeline panel's bar shows, summed
        # across the chain's commands (fill + re-match passes), so the
        # two never disagree and the dialog no longer idles at ~95%.
        chain = self._interp_chain
        cmd = getattr(self, "_pipe_command", None)
        dlg = self._track_progress_dialog
        if (
            chain is not None
            and dlg is not None
            and cmd is not None
            and id(cmd) in chain["ticks"]
        ):
            ticks = chain["ticks"][id(cmd)]
            overall = chain["base"] + min(int(done), ticks)
            label = chain.get("label", "Interpolate track")
            if op_name in ("interpolate_tracks", "inject_fitted_peaks"):
                dlg.setLabelText(
                    f"{label}: fitting injected boxes… "
                    f"({min(int(done), ticks)}/{ticks} frames)"
                )
            elif op_name == "run_matching":
                dlg.setLabelText(
                    f"{label}: matching "
                    f"{cmd.kwargs.get('peaks_type', 'segments')}… "
                    f"({min(int(done), ticks)}/{ticks} frames)"
                )
            # setValue MUST come last: on a window-modal QProgressDialog
            # it runs processEvents(), which can deliver this command's
            # queued opFinished re-entrantly — completing the chain and
            # tearing the dialog down (``_track_progress_dialog`` is
            # None again on return). Nothing may touch the dialog after
            # this call.
            dlg.setValue(int(round(100 * overall / chain["total"])))

        # If that re-entrant opFinished drained the queue, the status
        # bar already reads "idle" — do not resurrect a stale
        # "running: …" line for a command that is finished.
        if self._pipe_thread is None and not self._pipeline_queue:
            return

        if op_name == "track_peaks":
            # Busy (indeterminate) in the panel — no frame count to show.
            new_tail = " · tracking…"
        elif total <= 1:
            new_tail = ""
        else:
            new_tail = f" · {done}/{total} frames"
        # Mirror the same ticks onto the status-bar bar. Setting the
        # range on every tick is what promotes it from the busy marquee
        # to a real 0..total bar the first time a count arrives.
        if total > 1 and op_name != "track_peaks":
            if self._sb_pipe_bar.maximum() != total:
                self._sb_pipe_bar.setRange(0, int(total))
            self._sb_pipe_bar.setValue(int(done))
        if getattr(self, "_pipe_progress_tail", "") == new_tail:
            return
        self._pipe_progress_tail = new_tail
        # Re-render the status line so the new tail is visible
        # immediately. Reuse the existing op + entry from the in-flight
        # command rather than rebuilding here.
        cmd = getattr(self, "_pipe_command", None)
        self._update_status_pipeline(cmd, running=True)

    def _entry_wavelength(self) -> float | None:
        """λ in Å for the active entry, cached; None when unknown.

        Resolved lazily on the first cursor move after an entry change
        rather than during the switch itself: the switch path is the one
        the big-scan work made I/O-free, and a readout is not worth
        putting a file open back into it.
        """
        session = self.session
        entry = self.entry_combo.currentText() if hasattr(self, "entry_combo") else ""
        if session is None or not entry:
            return None
        key = (str(getattr(session, "temp_path", "")), entry)
        if getattr(self, "_wavelength_key", None) == key:
            return self._wavelength_value
        value = None
        try:
            geom = file_model.read_geometry_for_entry(session.temp_path, entry)
            if geom:
                wl = float(geom.get("wavelength_angstrom") or 0.0)
                value = wl if wl > 0 else None
        except Exception:
            logger.debug("suppressed exception reading wavelength", exc_info=True)
        self._wavelength_key = key
        self._wavelength_value = value
        return value

    def _q_derived_tail(self, q: float) -> str:
        """`d` and `2θ` for a |q|, as far as the entry's metadata allows.

        d = 2π/|q| needs nothing but the cursor; 2θ = 2·asin(λ|q|/4π)
        needs the entry's wavelength, so it appears only when the file
        carries one.
        """
        if not q or q <= 0:
            return ""
        tail = f"  d={2 * math.pi / q:.3f} Å"
        wl = self._entry_wavelength()
        if wl:
            arg = wl * q / (4 * math.pi)
            if arg <= 1.0:
                tail += f"  2θ={2 * math.degrees(math.asin(arg)):.2f}°"
        return tail

    def _on_status_cursor_moved(self, info) -> None:
        if not self._status_cursor_visible:
            self._sb_cursor.setText("")
            return
        if not info:
            self._sb_cursor.setText("")
            return
        mode = info.get("mode")
        inten = info.get("intensity", float("nan"))
        inten_str = "—" if inten != inten else f"{inten:.3g}"  # NaN check
        # Overlapping boxes under the cursor: says that clicking again
        # steps to the next one, which is otherwise invisible.
        depth = int(info.get("overlapping", 0) or 0)
        stack = (f"  |  {depth} boxes here, click again for the next"
                 if depth > 1 else "")
        if mode == "pixel":
            # Raw detector frames have no q-axes, so no d / 2θ either.
            self._sb_cursor.setText(
                f"row={info['row']}, col={info['col']}, I={inten_str}"
            )
        elif mode == "cartesian":
            q = math.hypot(info["q_xy"], info["q_z"])
            self._sb_cursor.setText(
                f"q_xy={info['q_xy']:.3f}, q_z={info['q_z']:.3f}, "
                f"I={inten_str}{self._q_derived_tail(q)}{stack}"
            )
        elif mode == "polar":
            # ``theta`` here is the azimuth of the polar map, written χ
            # so it cannot be misread as the scattering angle 2θ now
            # standing next to it.
            self._sb_cursor.setText(
                f"r={info['r']:.3f}, χ={info['theta']:.1f}°, "
                f"I={inten_str}{self._q_derived_tail(float(info['r']))}{stack}"
            )
        else:
            self._sb_cursor.setText("")

    def _set_cursor_readout_visible(self, visible: bool) -> None:
        self._status_cursor_visible = bool(visible)
        self._sb_cursor.setVisible(self._status_cursor_visible)

    def _update_actions(self) -> None:
        has_session = self.session is not None
        # Save/Save As only apply to NeXus sessions — raw sessions have no
        # writable temp copy. Close still works either way.
        is_nexus = has_session and self.session.kind == "nexus"
        self.action_save.setEnabled(is_nexus)
        self.action_save_as.setEnabled(is_nexus)
        self.action_close_file.setEnabled(has_session)
