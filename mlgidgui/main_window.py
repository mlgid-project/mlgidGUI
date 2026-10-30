from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QFileInfo,
    QMetaObject,
    QObject,
    QProcess,
    QSettings,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFileIconProvider,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QFrame,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.hdf5 import Hdf5TreeModel, Hdf5TreeView
from silx.gui.hdf5.Hdf5Node import Hdf5Node
from silx.gui.hdf5.NexusSortFilterProxyModel import NexusSortFilterProxyModel

from mlgidgui import file_dialogs
from mlgidgui import file_model
from mlgidgui import frame_range
from mlgidgui import peak_clipboard
from mlgidgui import pipeline
from mlgidgui import update_check
from mlgidgui.pen_picker import PenPopup
from mlgidgui.preview_fit import PreviewFitRunner
from mlgidgui.controls_help import ControlsDialog
from mlgidgui.image_viewer import (
    GIWAXSImageViewer,
    MATCHED_STYLE,
    SIM_MISSED_COLOR,
    UNMATCHED_COLOR,
    ManualPeak,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    SelectedPeak,
)
from mlgidgui.parameter_panel import ParameterPanel
from mlgidgui import phase_tracking
from mlgidgui import simulation_pattern
from mlgidgui.peaks_table_panel import PeaksTablePanel
from mlgidgui.phase_views_window import PhaseViewsWindow
from mlgidgui.pipeline import (
    PipelineCommand,
)
from mlgidgui.pipeline_panel import PipelinePanel
from mlgidgui.profile_viewer import ProfileViewer
from mlgidgui.conversion_panel import ConversionPanel
from mlgidgui.scan_tracking_panel import ScanTrackingPanel
from mlgidgui.session import BaseSession, NexusSession, RawSession, Session
from mlgidgui.widgets import make_debounced_timer
from mlgidgui.main_window_sim import SimOverlayMixin
from mlgidgui.main_window_matched import MatchedLegendMixin
from mlgidgui.main_window_tracking import TrackingMixin
from mlgidgui.main_window_preview import PreviewMixin, _pygidfit_to_fit_like
from mlgidgui.main_window_pipeline import PipelineMixin
from mlgidgui.main_window_peaks import PeaksMixin, _CallbackAction
from mlgidgui.main_window_frames import FramesMixin
from mlgidgui.main_window_files import FilesMixin, _natural_key
from mlgidgui.main_window_build import BuildMixin
from mlgidgui.main_window_structure import StructureMixin
from mlgidgui.main_window_update import UpdateMixin
from mlgidgui.main_window_menus import MenusMixin
from mlgidgui.workers import (
    CifParseWorker,
    ConversionWorker,
    CopyWorker,
    EntryLoadWorker,
    ImportWorker,
    PipelineWorker,
    PrefetchWorker,
    TrackingLoadWorker,
)

import logging
logger = logging.getLogger(__name__)

# Re-exports: canonical homes moved in the 2026 source split; kept here
# so tests and older references resolve unchanged.
from mlgidgui.browser_widgets import (
    _FastFileIconProvider,
    _ImageFileNode,
    _MlgidHdf5TreeModel,
    _MlgidHdf5TreeView,
)
from mlgidgui.main_window_constants import (
    APP_NAME,
    DEFAULT_PLAYBACK_FRAME_MS,
    DEFAULT_PLAYBACK_TOTAL_S,
    NEXUS_FILTER,
    OPEN_FILTER,
    PLAYBACK_FRAME_MS_MAX,
    PLAYBACK_FRAME_MS_MIN,
    PLAYBACK_MODE_FRAME,
    PLAYBACK_MODE_TOTAL,
    PLAYBACK_TICK_FLOOR_MS,
    PLAYBACK_FRAME_MS_KEY,
    PLAYBACK_MODE_KEY,
    PLAYBACK_TOTAL_S_KEY,
    PLAYBACK_TOTAL_S_MAX,
    PLAYBACK_TOTAL_S_MIN,
)
from mlgidgui.main_window_dialogs import _ExportPeaksDialog, _SettingsDialog
from mlgidgui.update_ui import (
    _UpdateBanner,
    _UpdateCheckWorker,
    _UpdateInstallWorker,
)
from mlgidgui.widgets import ValueWheelBlocker as _ValueWheelBlocker
from mlgidgui.widgets import make_pen_swatch as _make_pen_swatch




# Widgets that use Up / Down themselves. The peak-step shortcut leaves
# them the key (see ``MainWindow.eventFilter``): a spin box steps its
# value, a combo box and a list or table move their current item, a
# slider moves its handle, a text field moves its caret.
_ARROW_KEY_WIDGETS = (
    QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QComboBox,
    QLineEdit, QPlainTextEdit, QTextEdit,
)


class MainWindow(
    SimOverlayMixin,
    MatchedLegendMixin,
    TrackingMixin,
    PreviewMixin,
    PipelineMixin,
    PeaksMixin,
    FramesMixin,
    FilesMixin,
    StructureMixin,
    BuildMixin,
    UpdateMixin,
    MenusMixin,
    QMainWindow,
):
    # Cross-thread invocation signals for the prefetch worker (queued
    # auto-connection to slots on the worker's own QThread). Emitting
    # is the safe cross-thread equivalent of calling the worker's
    # methods directly; the queued delivery serialises with the
    # worker's other queued slots and its internal QTimer ticks.
    _prefetchConfigure = Signal(str, str, int, int)
    _prefetchUpdate = Signal(int, bool, int)
    _prefetchRelease = Signal()
    # Queued request to the persistent EntryLoadWorker: (file_path, entry,
    # request_id, extra_peak_datasets). The worker opens + warms that
    # entry's first frame off the GUI thread and emits ``loaded`` back;
    # stale request_ids (rapid switching) are dropped on arrival. The
    # trailing tuple carries the user's registered extra peak lists so the
    # worker's overlay read covers them too. See ``_load_entry_async``.
    _entryLoadRequest = Signal(str, str, int, object)
    # Queued request to the TrackingLoadWorker: (file_path, entry,
    # request_id, force). Builds the Scan-tracking state from the table
    # the entry carries in the file; see ``_request_saved_tracking``.
    _trackLoadRequest = Signal(str, str, int, bool)
    # Raw-mode counterpart: (RawEntry, request_id) → ``raw_loaded`` with a
    # ready LazyRawStack. Shares the same request-id counter so raw and
    # NeXus switches supersede each other. See ``_load_raw_entry_into_viewer``.
    _rawLoadRequest = Signal(object, int)
    _THEME_KEY = "theme"

    _RECENT_FILES_KEY = "recentFiles"
    _MAX_RECENT_FILES = 10

    # Playback settings (persisted via QSettings). See the module-level
    # PLAYBACK_* constants for defaults and bounds.
    _PLAYBACK_MODE_KEY = PLAYBACK_MODE_KEY
    _PLAYBACK_FRAME_MS_KEY = PLAYBACK_FRAME_MS_KEY
    _PLAYBACK_TOTAL_S_KEY = PLAYBACK_TOTAL_S_KEY
    # Last mlgidGUI version this profile ran, for the post-update changelog.
    _LAST_SEEN_VERSION_KEY = "lastSeenVersion"
    # Opt-in: install a newer release automatically on launch (Help menu).
    _AUTO_UPDATE_KEY = "autoUpdate"


    def __init__(self) -> None:
        super().__init__()
        # Kill wheel-scrolling through closed comboboxes and spin boxes
        # everywhere — the wheel scrolls the page instead of moving the
        # value (kept as an attribute: installEventFilter does not own
        # the filter, so it must not be garbage-collected).
        self._value_wheel_blocker = _ValueWheelBlocker(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._value_wheel_blocker)
        # Multiple files can be open at once — each as its own Session in the
        # file browser. The "active" one drives entry_combo, the image viewer,
        # and per-file actions (save, save-as, close, pipeline). Switching is
        # automatic when the user clicks a node from a different file.
        self._sessions: list[BaseSession] = []
        self._active_session: BaseSession | None = None
        # Opens run serially through the existing single-thread CopyWorker
        # plumbing; extra paths from a multi-select dialog wait here.
        self._open_queue: list[Path] = []
        # Classification is now done in the worker (off the GUI thread), so
        # raw files and unclassifiable files are reported back as each
        # worker finishes and collected here; the batch is finalized when
        # the queue drains (raw files bundle into one RawSession).
        self._pending_raw_paths: list[Path] = []
        self._pending_rejected: list[Path] = []
        # RawEntry lists CopyWorker found while classifying, keyed by
        # str(path). Handed to the RawSession in ``_finalize_open_batch``
        # so ``_populate_raw_entries`` never re-walks the file's metadata
        # on the GUI thread (the walk takes seconds on a big beamtime
        # file and was the raw-open freeze).
        self._pending_raw_entry_cache: dict[str, list] = {}
        # Raw-file rows enter the silx tree in chunks driven by a 0 ms
        # timer instead of one synchronous loop: ``insertFile`` decodes a
        # standalone image fully on the GUI thread, and a 1000-image batch
        # froze the window for seconds (both on open and on every tree
        # reattach). The queue is dropped on ``_detach_silx_tree`` — the
        # rows belong to the model being cleared; the following reattach
        # re-queues every session's files from scratch.
        self._tree_insert_queue: list[str] = []
        # Progress accounting for the status-bar "Browser: n/N files"
        # indicator (reset whenever a fresh fill starts from an empty
        # queue; see _queue_tree_inserts / _drain_tree_insert_queue).
        self._tree_insert_total = 0
        self._tree_insert_done = 0
        self._tree_insert_timer = QTimer(self)
        # A real (nonzero) interval: a 0 ms timer can fire several times
        # within one event-processing pass, stacking insert chunks into
        # a single user-visible stall. 15 ms guarantees paints and input
        # get a slot between chunks.
        self._tree_insert_timer.setInterval(15)
        self._tree_insert_timer.timeout.connect(self._drain_tree_insert_queue)
        # str(file path) → entry-combo label for the active raw session's
        # image entries; lets a click on an ``_ImageFileNode`` browser row
        # select its image without any scan. Rebuilt by
        # ``_populate_raw_entries``.
        self._raw_entry_label_by_path: dict[str, str] = {}
        # Shared by every file dialog in the app; ``setIconProvider``
        # does not take ownership, so the provider must outlive the
        # dialog. ``file_dialogs`` owns the single instance.
        self._file_dialog_icons = file_dialogs.icon_provider()
        self._thread: QThread | None = None
        self._worker: CopyWorker | None = None
        self._pipe_thread: QThread | None = None
        self._pipe_worker: PipelineWorker | None = None
        # Phase-tracking (mlgidBASE track_peaks) results. Per-entry and
        # referencing FITTED rows, so they invalidate on entry switch
        # and on any fitted mutation — see _invalidate_scan_tracks.
        self._scan_payload: phase_tracking.TrackingPayload | None = None
        self._scan_member_ids: list | None = None
        self._scan_matched_tables: dict = {}
        self._scan_track_entry: str | None = None
        # Fitted tables read during member-id reconstruction, kept for
        # the only-tracked display options (gap interpolation anchors,
        # GUI-side ring tracking). {frame: PeakTable | None}.
        self._scan_fitted_tables: dict[int, object] = {}
        # {track_index: [cif, ...]} — the matched crystal phases each
        # track's tracked peaks belong to (dominant first), for the
        # phase-views q-map "Color by matched phase" overlay. Empty
        # until Matching has run; cleared with the rest of the results.
        self._scan_track_phases: dict = {}
        # {member index: [cif, ...]} — which members were actually
        # claimed by a matched structure ON THEIR FRAME. The phase views
        # use this to split a track's rendering at the frame where
        # matching starts (claimed = phase colour, rest = unmatched)
        # instead of painting the whole span with the dominant phase.
        self._scan_member_phases: dict = {}
        # Track indices that are rings — the q-map draws them as dashed
        # quarter-circle arcs rather than trajectory points.
        self._scan_ring_tracks: set = set()
        # Interpolate-track (gap fill) run in flight: the fill records
        # returned by the interpolate_tracks op (each {"track",
        # "frame", "detected_id", "fitted_id", "is_ring"}), applied as
        # new track members once the LAST chained re-match finishes;
        # and the chain bookkeeping that drives the loading dialog with
        # the workers' REAL per-frame progress:
        #   {"commands": [cmd, ...],          # keeps ids alive
        #    "ticks": {id(cmd): n_frames},    # per-command bar weight
        #    "total": N, "base": 0}           # overall progress
        self._interp_fill_result: list | None = None
        self._interp_chain: dict | None = None
        # Expected-pattern validation stash: set when an
        # inject_fitted_peaks chain is enqueued, filled with the fill
        # records when the inject command lands, consumed by
        # _finalize_predicted_fill at chain completion.
        self._predicted_fill: dict | None = None
        # Phase-tracking views window: lazily built on first "Show
        # views…", kept alive so its settings and computed caches
        # persist across re-opens (same pattern as _figure_export_window).
        self._phase_views_window: PhaseViewsWindow | None = None
        # Modal progress dialog shown during a track_peaks run. The
        # tracking button lives in the Scan-tracking dock but the
        # pipeline panel's bar is in a (usually tabbed-away) dock, so a
        # centred dialog is the only reliably-visible feedback. Its bar
        # is advanced by a GUI-thread timer (the op has no clean
        # per-frame progress — see the busy-bar note in pipeline_panel).
        self._track_progress_dialog: QProgressDialog | None = None
        # pygid normalization is lazy (deferred off the open path). This
        # records which (temp-file, entry) pairs have already been
        # normalized so ``_ensure_entry_normalized`` runs the scoped
        # ``normalize_for_pygid`` at most once per entry. Keyed by the
        # temp_path string (matches the file_path snapshotted into the
        # pipeline queue); pruned in ``_close_session``.
        self._normalized_entries: dict[str, set[str]] = {}
        # Lazy per-frame peak loading (see ``_load_frame_peaks``). Peaks
        # for a frame are pulled into the viewer the first time that frame
        # is shown, not all up front. Reset on each entry (re)load.
        self._loaded_peaks_entry: str | None = None
        self._loaded_peak_frames: set[int] = set()
        # A tree node clicked while the Data tab was hidden, held until the
        # user switches to that tab (see ``_set_or_defer_data_node``). This
        # keeps a single Image-tab click from eagerly resolving a huge
        # external-linked dataset into a widget the user can't see.
        self._pending_data_node = None
        # The same bargain for the Structure tab, and the file whose
        # layout check is currently displayed (None re-runs it).
        self._pending_structure_node = None
        # The node the Structure panel is currently showing, whoever
        # selected it — its own tree, the browser, or a search hit.
        self._structure_node = None
        self._structure_checked_file: str | None = None
        # (file, h5 path) the Structure tab is showing and may write to;
        # None whenever it is showing something read-only.
        self._structure_target: tuple | None = None
        # The editor's r+ handle on the active working copy, and one undo
        # history per file. The handle is acquired lazily on the first
        # edit and released by ``_detach_silx_tree``.
        self._h5_edit_handle = None
        self._structure_histories: dict = {}
        # Docks folded away because the Structure tab is in front, to be
        # restored on the way back to Image or Data.
        self._structure_folded_docks: set[str] = set()
        # Scan tracking was hidden because the data has a single frame
        # (``_sync_scan_tracking_visibility``), not closed by the user.
        self._tracking_hidden_by_rule = False
        # Set when a structural edit landed while the File browser was
        # folded away: rebuilding a tree nobody can see is a teardown for
        # nothing, so it is paid once, on the way out.
        self._browser_needs_rebuild = False
        # The last node clicked in the Structure tab's own tree. Its
        # entry is applied when the tab is left, so browsing a file to
        # edit it never reloads the image behind the tab.
        self._structure_pending_activation = None
        # The entry the combo was moved to while the tab was in front.
        # The read is async but the install is not, and installing a
        # frame into a viewer nobody can see is the whole cost for none
        # of the benefit. Reconciled from the combo on the way out.
        self._structure_pending_entry = None
        # Raised while the File browser's selection is being put back
        # after a rebuild, so the selection handler can tell that apart
        # from a click. See ``_restore_tree_state``.
        self._restoring_tree_selection = False
        # What the Structure tab holds between a Copy and a Paste. Kept
        # apart from the peak clipboard: Ctrl+C means a different thing
        # on each tab.
        # A list, because the tab's tree is multi-select: one gesture can
        # put six nodes on it. Empty when nothing has been copied.
        self._structure_clip: list = []
        # Tools → Export figure window. Built lazily on first open
        # (see ``_action_export_figure``); kept alive across re-opens
        # so settings persist. None until the user invokes Tools →
        # Export figure… for the first time.
        self._figure_export_window = None  # type: ignore[var-annotated]
        # Queue of (file_path, PipelineCommand) tuples waiting to run
        # sequentially. The file_path is **snapshotted at enqueue time**
        # so a mid-queue active-session switch (user clicks the other
        # loaded file in the tree, etc.) can't cause later commands to
        # dispatch against the wrong file. The "All entries" option in
        # the pipeline panel expands one runRequested into one command
        # per entry — all sharing the path captured at expansion time.
        self._pipeline_queue: list[tuple[Path, PipelineCommand]] = []
        # Entry-level progress tracking — the depth of the current
        # "All entries" expansion plus the 1-indexed position we are
        # at. Reset to (0, 0) when the queue drains; an "Active entry"
        # run sets total=1 (no entry bar shown). Surfaced to the panel
        # via ``on_queue_progress`` and folded into the status-bar tail.
        self._entry_queue_total: int = 0
        self._entry_queue_pos: int = 0
        # CIF-parse worker thread. CifPattern construction is slow for
        # raw CIFs so we run it off the GUI thread; only one parse runs
        # at a time (the panel's button stays disabled while it's in
        # flight).
        self._cif_parse_thread: QThread | None = None
        self._cif_parse_worker: CifParseWorker | None = None
        # Conversion worker thread (raw → NeXus). Kept separate from
        # the pipeline worker because conversion runs on raw inputs,
        # while pipeline runs on converted NeXus files; the two never
        # need to share a worker.
        self._conv_thread: QThread | None = None
        self._conv_worker: ConversionWorker | None = None
        self._conv_progress: QProgressDialog | None = None
        # Converted-image import (File menu / float-batch offer): same
        # thread + progress-dialog shape as the conversion run above.
        self._import_thread: QThread | None = None
        self._import_worker = None
        self._import_progress: QProgressDialog | None = None

        # Background prefetch worker. Lives on its own QThread so it
        # can read frames + compute polar resamples without ever
        # blocking the GUI. Spawned lazily on first entry load (no
        # cost on cold startup) and survives across entry switches —
        # the worker is reconfigured per-entry rather than rebuilt.
        # See ``_ensure_prefetch_worker`` and the ``_prefetch*``
        # signals on the class for the cross-thread wiring.
        self._prefetch_thread: QThread | None = None
        self._prefetch_worker: PrefetchWorker | None = None

        # Persistent EntryLoadWorker: opens + warms an entry's first frame
        # off the GUI thread so switching entries (combo or file browser)
        # never blocks on a slow network read. Lazily spawned on the first
        # interactive switch. ``_entry_req_id`` is bumped per switch so a
        # late result from a superseded switch is dropped, not rendered.
        self._entry_load_thread: QThread | None = None
        self._entry_load_worker: EntryLoadWorker | None = None
        # Persistent TrackingLoadWorker (saved tracking table -> panel),
        # spawned on first use; ``_track_load_req_id`` drops superseded
        # results, ``_track_autoload_key`` remembers which (session,
        # entry) already had its automatic look so overlay reloads of
        # the same entry do not repeat it.
        self._track_load_thread: QThread | None = None
        self._track_load_worker: TrackingLoadWorker | None = None
        self._track_load_req_id: int = 0
        self._track_autoload_key: tuple | None = None
        self._entry_req_id: int = 0

        # Frame step per play-tick. Stays at 1 unless the requested
        # per-frame interval drops below ``PLAYBACK_TICK_FLOOR_MS``,
        # in which case ``_compute_play_schedule`` bumps it so the
        # play-head jumps multiple frames per tick to honour the
        # total-time target without overrunning the 20 fps practical
        # ceiling. Refreshed on every Play press + every settings
        # change while playing.
        self._play_step: int = 1

        # Stash of the manual peak's geometry captured the moment the
        # "Save fitted as ring" toggle goes ON. Set to a tuple of
        # (peak_ref, radius, angle, radius_width, angle_width, is_ring)
        # while ring is active; cleared on the toggle's OFF transition
        # after the box has been restored. Allows the auto-uncheck
        # that follows a successful Add-to-fitted to revert the box
        # to its pre-ring shape without the host needing to track a
        # commit/cancel distinction.
        self._ring_pre_geom: tuple[
            ManualPeak, float, float, float, float, bool
        ] | None = None

        # 2D-preview cache: fingerprint -> ManualFitResult | None.
        # Fingerprint is the user-controlled inputs to the live
        # pygidfit call; identical fingerprint → reuse the cached
        # result instead of rerunning the (slow) 2D fit. A dict
        # (not just one slot) so the multi-selection cyan-preview
        # path can show N boxes without recomputing N peaks every
        # selection-tick. ``None`` value means pygidfit ran but
        # didn't converge — cached so we don't retry the same fail.
        self._2d_preview_cache: dict[tuple, object] = {}
        # The fits themselves run on a background thread (see
        # ``preview_fit``); results come back through ``done``.
        self._preview_runner = PreviewFitRunner(self)
        self._preview_runner.done.connect(self._on_preview_fit_done)
        self._preview_status_shown = False
        # The Add to fitted (2D) click waiting on a background fit.
        self._add_fitted_wait: dict | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(*self._default_window_size())

        self._build_menu()
        self._build_central()
        self._build_docks()
        # View menu is built after docks because it pulls
        # toggleViewAction()s from them. Settings is built next.
        # Help comes last so it sits at the right end of the menu
        # bar — the conventional rightmost-menu placement.
        self._build_view_menu()
        self._build_settings_menu()
        self._build_help_menu()
        # Menu glyphs in one pass, now that every action exists.
        self._apply_menu_icons()
        # Frame-navigation shortcuts. Installed last so the viewer +
        # entry combo exist. Window-context QActions; text inputs
        # (QLineEdit, QSpinBox) consume Left/Right/Home/End for
        # caret nav before the shortcut fires, so the bindings only
        # trigger when focus is on a non-text widget (viewer, dock
        # frame, menu bar). J/K give a Vim-style fallback that
        # works even when the viewer has unconventional focus
        # handling.
        self._install_frame_shortcuts()
        # Down / Up select the next / previous peak (see the method).
        self._install_peak_step_shortcuts()
        # Ctrl+Return runs Add to fitted (see the method).
        self._install_add_to_fitted_shortcut()
        # Make undo/redo fire even when a third-party widget registers the
        # same standard shortcuts (see ``eventFilter``). Installed on the
        # application so it sees key events regardless of focus.
        QApplication.instance().installEventFilter(self)
        # Status bar depends on the viewer + entry combo existing; build
        # after central + docks.
        self._build_status_bar()
        self._update_title()
        self._update_actions()
        # Seed the welcome page. Its wordmark, recent list and backend
        # notice are all filled by _refresh_welcome_view, which is
        # otherwise only reached through _apply_session_mode — and that
        # first runs when a file is opened. Without this call the page
        # the user sees at cold start is an empty label under a "Recent"
        # heading with nothing beneath it.
        self._refresh_welcome_view()
        # Accept dropped files anywhere on the main window so the user
        # can drag NeXus / raw paths in from a file manager. The drop
        # handler classifies each file by content and dispatches.
        self.setAcceptDrops(True)
        # Snapshot the default dock arrangement now that everything
        # has been built and the initial ``_apply_session_mode(None)``
        # has tabified the right-side stack. View → Reset layout
        # restores from this snapshot — see ``_reset_layout``.
        self._default_layout_state = self.saveState()
        # One-shot guard for the first-show dock sizing (see showEvent).
        self._dock_widths_applied = False

    @staticmethod
    def _default_window_size() -> tuple[int, int]:
        """Open at 1600x950 where the screen allows it.

        The window holds two dock columns with the image between them,
        and QMainWindow shares the width out between them: at 1400 the
        right-hand column could only be opened to ~350 px, which
        compressed its forms (the panels sit in resizable scroll areas,
        so a narrow column elides rather than scrolls). Falls back to
        90% of the available geometry, so a laptop still gets a window
        that fits its screen.
        """
        screen = QApplication.primaryScreen()
        if screen is None:                      # headless / no display
            return (1400, 900)
        available = screen.availableGeometry()
        # Never smaller than the long-standing 1400x900: a screen that
        # cannot fit it could not fit it before either, and shrinking
        # the default there would squeeze the docks instead of helping.
        return (max(1400, min(1600, int(available.width() * 0.9))),
                max(900, min(950, int(available.height() * 0.9))))

    @property
    def session(self) -> Session | None:
        """The currently-active session — drives viewer + entry_combo + save.

        Most callers were written before multi-file support and reach for
        ``self.session``; making it a property of the active session keeps
        those call sites working without per-call refactors.
        """
        return self._active_session

    def _is_busy(self) -> bool:
        # No session loaded, or a pipeline/tracking worker owns the file.
        return self.session is None or self._pipe_thread is not None

    def eventFilter(self, obj, ev):  # type: ignore[override]
        """Drive undo / redo from the keyboard even when the chord is an
        "ambiguous shortcut".

        The embedded pyFAI calibration dialog pulls in silx's
        ``MaskToolsWidget`` and pyFAI's peak-picking task, each of which
        binds Undo/Redo to the standard sequences (``Ctrl+Z`` / ``Ctrl+Y``,
        and on some silx/pyFAI versions the full standard Redo set, which
        includes ``Ctrl+Shift+Z``). Those widgets outlive the dialog
        (pyFAI's ``CalibrationContext`` is a singleton), so afterwards Qt
        finds two actions claiming the redo chord, logs "Ambiguous
        shortcut overload", and fires neither — the Edit-menu item still
        works, but the keyboard doesn't.

        We intercept the chord at the ``ShortcutOverride`` stage, before
        Qt's shortcut map runs, and run our own handler. Guarded so the
        calibration dialog's own mask undo/redo and text-field undo are
        left alone: only act while this window holds focus and the focus
        is not a text-editing widget.
        """
        # Every "not ours" exit returns False, which is all QObject's own
        # eventFilter does. Calling super() would hand ``obj`` back to C++,
        # and this filter sits on the whole application, so ``obj`` can be a
        # widget in the middle of its destruction (a QMenu's hide event fires
        # from ~QWidget, after PySide has invalidated the wrapper): that
        # raised "Internal C++ object already deleted" from here.
        # Once closed (or mid-teardown), do nothing — the filter may still
        # be installed on the app and our child widgets may be gone. Pytest
        # creates many windows per process; a stale filter touching a
        # destroyed ``self`` would otherwise raise during shutdown.
        if getattr(self, "_closed", False):
            return False
        et = ev.type()
        # Status-bar pipeline cell: clicking it opens the log of the run
        # it is reporting on.
        if (et == QEvent.Type.MouseButtonPress
                and obj is getattr(self, "_sb_pipeline", None)):
            dock = getattr(self, "_logs_dock", None)
            if dock is not None:
                dock.show()
                dock.raise_()
            return True
        if et not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        if (
            et == QEvent.Type.ShortcutOverride
            and ev.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down)
            and ev.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            # A widget that uses Up / Down keeps them: accepting the
            # override stops the peak-step shortcut and delivers the key.
            focus = QApplication.focusWidget()
            if isinstance(focus, _ARROW_KEY_WIDGETS) and focus.window() is self:
                ev.accept()
                return True
        fw = QApplication.focusWidget()
        if fw is not None:
            if fw.window() is not self:
                return False  # a child dialog owns the key
            if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit)):
                return False  # native text undo/redo
        elif not self.isActiveWindow():
            return False
        mods = ev.modifiers()
        if not (mods & Qt.KeyboardModifier.ControlModifier) or (
            mods & Qt.KeyboardModifier.AltModifier
        ):
            return False
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        key = ev.key()
        is_redo = (key == Qt.Key.Key_Z and shift) or (
            key == Qt.Key.Key_Y and not shift
        )
        is_undo = key == Qt.Key.Key_Z and not shift
        if not (is_redo or is_undo):
            return False
        if et == QEvent.Type.ShortcutOverride:
            # Claim the key so Qt skips (ambiguous) shortcut resolution and
            # re-delivers it as a plain KeyPress, handled on the next pass.
            ev.accept()
            return True
        (self._action_redo if is_redo else self._action_undo)()
        return True
    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # Dock tab bars only materialise with the first layout pass —
        # sweep for ghosts right after it (see the method docstring).
        QTimer.singleShot(0, self._hide_stale_dock_tab_bars)
        # Dock widths asked for during construction are only advisory:
        # QMainWindow has no real geometry yet and scales the request
        # down to whatever it thinks it has. Ask again on the first
        # show, when the numbers actually land. Once only — after that
        # the column belongs to the user.
        if not self._dock_widths_applied:
            self._dock_widths_applied = True
            QTimer.singleShot(0, self._apply_default_dock_widths)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drops carrying local file URLs.

        Acceptance is loose at enter time — content classification
        happens in ``dropEvent`` so the cursor reflects "yes you can
        drop" while the user drags over the window. Non-file payloads
        (text, internal Qt drags) are ignored.
        """
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        # Same gate as dragEnter — Qt fires move events repeatedly while
        # the drag is in flight and the proposed-action state has to
        # stay accepted across them or the drop won't fire.
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open every dropped file via the unified _open_paths classifier."""
        urls = event.mimeData().urls()
        paths: list[Path] = []
        for u in urls:
            if not u.isLocalFile():
                continue
            local = u.toLocalFile()
            if not local:
                continue
            paths.append(Path(local))
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._open_paths(paths)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Idempotent: close() can be delivered more than once (Qt can
        # re-emit a close event, and pytest-qt's teardown calls close()
        # again on every registered widget after our own close). Running
        # the teardown twice would re-clear already-destroyed pyqtgraph
        # widgets — on some PySide6 builds that raises
        # "Internal C++ object already deleted" during shutdown. Skip a
        # second pass.
        if getattr(self, "_closed", False):
            event.accept()
            return
        # Each loaded file may have unsaved changes — prompt per dirty
        # session in load order so the user gets the same per-file save
        # dialog they would on _action_close_file.
        for s in list(self._sessions):
            if not self._confirm_discard_changes(s):
                event.ignore()
                return
        # Past the point of no return — mark closed so a re-entrant or
        # pytest-qt second close() short-circuits above.
        self._closed = True
        # A preview fit still running must not hold the window open.
        self._preview_runner.shutdown()
        # Drop the app-wide undo/redo event filter so a closed window
        # (which may linger under the test session's gc.disable) stops
        # receiving key events.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Stop frame playback so the timer doesn't fire one last tick
        # against a torn-down viewer during shutdown.
        self._pause_playback()
        # All clear — tear everything down. silx must release its handles
        # before we delete the temp files. One-way detach: the app is
        # closing, so there is no reattach (hence not _detached_silx_tree).
        self._detach_silx_tree()
        self.viewer.clear()
        self.profile_viewer.clear()
        # Drop the shared open-progress dialog if it's still up (rare —
        # the modal would normally have blocked the close). Otherwise its
        # leftover modal overlay would dim the next opened window too.
        self._dismiss_open_progress()
        # Stop a CIF parse if one is running — closing the window while
        # CifPattern construction is in flight otherwise drops the worker
        # thread on the floor and Qt complains at exit.
        if self._cif_parse_thread is not None:
            self._cif_parse_thread.quit()
            self._cif_parse_thread.wait()
        # Stop a conversion run if one is in flight — pygid + h5py do
        # their own cleanup on a clean thread exit.
        if self._conv_thread is not None:
            self._conv_thread.quit()
            self._conv_thread.wait()
        # Close the phase-tracking views window (its closeEvent waits
        # for any in-flight view-computation worker).
        if self._phase_views_window is not None:
            self._phase_views_window.close()
        self._close_tracking_progress()
        # Shut the background prefetch worker down cleanly. Release
        # its h5py handle first (so the worker stops trying to read
        # frames), then quit + wait the thread so its event loop
        # exits before we delete its Q objects.
        if self._prefetch_worker is not None:
            self._prefetchRelease.emit()
            # Process the queued release so it lands on the worker's
            # thread before we quit it; otherwise the worker would
            # try to read on a destroyed h5py.File during shutdown.
            QCoreApplication.processEvents()
            self._prefetch_thread.quit()
            self._prefetch_thread.wait()
            self._prefetch_worker.deleteLater()
            self._prefetch_worker = None
            self._prefetch_thread.deleteLater()
            self._prefetch_thread = None
        # Shut the async entry-load worker down. Bump the request id first
        # so any result still in the queue is treated as stale (its source
        # released) rather than installed into a tearing-down viewer.
        if self._entry_load_worker is not None:
            self._entry_req_id += 1
            self._entry_load_thread.quit()
            self._entry_load_thread.wait()
            self._entry_load_worker.deleteLater()
            self._entry_load_worker = None
            self._entry_load_thread.deleteLater()
            self._entry_load_thread = None
        if self._track_load_worker is not None:
            self._track_load_req_id += 1
            self._track_load_thread.quit()
            self._track_load_thread.wait()
            self._track_load_worker.deleteLater()
            self._track_load_worker = None
            self._track_load_thread.deleteLater()
            self._track_load_thread = None
        for s in list(self._sessions):
            # Inactive sessions may hold a stashed (parked) FrameSource —
            # release the handle before the temp file is deleted.
            stash = getattr(s, "_prewarm", None)
            if stash is not None:
                self._release_prewarm(stash)
            s.close()
        self._sessions.clear()
        self._active_session = None
        event.accept()
