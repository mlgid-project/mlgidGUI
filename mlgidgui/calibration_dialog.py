"""In-GUI PONI calibration and mask creation.

Wraps pyFAI's standard five-task calibration widget set
(``ExperimentTask`` → ``MaskTask`` → ``PeakPickingTask`` →
``GeometryTask`` → ``IntegrationTask``) inside a modal QDialog so
mlgidGUI users can produce a PONI and / or a mask file without
leaving the app. The dialog returns:

- ``saved_poni_path`` — path the user saved a PONI to via the
  Integration task's built-in "Save as PONI" button. Read by the
  host after ``exec()`` returns ``Accepted``.
- ``saved_mask_path`` — path the user saved the mask to via this
  dialog's bottom-row "Save mask…" button.

Lazy imports of ``pyFAI.gui.*`` happen at the *module level* of
this file, which is itself only imported when one of the
Conversion dock's "Create…" buttons is clicked — see
``conversion_panel._create_poni`` / ``_create_mask``. So pyFAI's
Qt-heavy import chain stays out of cold startup.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mlgidgui import file_dialogs
from mlgidgui.widgets import PRIMARY as _PRIMARY, set_variant as _set_variant
from mlgidgui.settings_migration import calibration_settings

# These imports trigger a large Qt + matplotlib + silx chain. Done
# at module level (not inside the class) because the module itself
# is import-deferred by the host — so the chain only fires when the
# user actually opens the dialog.
import pyFAI  # noqa: F401  (sanity check the install before we dig deeper)
from pyFAI.app import calib2
from pyFAI.gui.CalibrationContext import CalibrationContext
from pyFAI.gui.tasks.ExperimentTask import ExperimentTask
from pyFAI.gui.tasks.GeometryTask import GeometryTask
from pyFAI.gui.tasks.IntegrationTask import IntegrationTask
from pyFAI.gui.tasks.MaskTask import MaskTask
from pyFAI.gui.tasks.PeakPickingTask import PeakPickingTask

import logging
logger = logging.getLogger(__name__)


def _adopting(create):
    """Wrap a pyFAI dialog factory so the result follows the shared dir."""

    @functools.wraps(create)
    def wrapper(*args, **kwargs):
        dialog = create(*args, **kwargs)
        try:
            file_dialogs.adopt(dialog)
        except Exception:
            logger.debug(
                "could not adopt a pyFAI file dialog", exc_info=True
            )
        return dialog

    return wrapper


# Task identifiers keyed by stable string IDs so callers don't have
# to import pyFAI to choose a starting tab. Ordered the same way
# the standalone pyFAI-calib2 CLI presents them.
_TASK_ORDER = ("experiment", "mask", "peaks", "geometry", "integration")
_TASK_LABELS = {
    "experiment": "1. Experiment setup",
    "mask": "2. Mask",
    "peaks": "3. Peak picking",
    "geometry": "4. Geometry",
    "integration": "5. Integration / Save PONI",
}


class CalibrationDialog(QDialog):
    """Modal calibration window backed by pyFAI's CalibrationContext.

    The dialog drives the canonical pyFAI workflow without
    sub-classing or rewriting any of pyFAI's task widgets — we just
    instantiate them, share a single ``CalibrationModel`` between
    them, and place them in a left-list / right-stack layout.

    Parameters
    ----------
    parent:
        The Qt parent.
    initial_image:
        Optional 2D numpy array to seed the experiment task with.
        When the host has a raw stack open, the active frame is a
        natural default; the user can still browse to a different
        calibration image inside the dialog.
    start_task:
        One of ``_TASK_ORDER``. Selects which task tab is visible
        on open.
    """

    # Emitted when the user clicks the bottom-row "Add PONI / Mask
    # to conversion" buttons. The host (ConversionPanel) connects
    # these to the QLineEdits in the Conversion dock so the freshly-
    # written paths land in the right field without the user having
    # to type or browse.
    applyPoniRequested = Signal(str)
    applyMaskRequested = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_image: np.ndarray | None = None,
        initial_poni: str | Path | None = None,
        initial_mask: str | Path | None = None,
        start_task: str = "experiment",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("pyFAI calibration")
        # qdarkstyle gives QGroupBox a very tight ``margin-top: 6px``
        # paired with ``padding-top: -4px`` on the title, which is
        # fine for label-only contents but overlaps the first widget
        # when that widget is a toolbar — see the Peak picking task's
        # "Picked rings" group, where the Load / Save icons end up
        # half-hidden under the title text. Widen the top margin /
        # padding locally so pyFAI's task widgets render with enough
        # headroom; scoped to this dialog so we don't perturb the
        # rest of the app.
        self.setStyleSheet(
            "QGroupBox { margin-top: 16px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin;"
            " subcontrol-position: top left; left: 8px;"
            " padding: 0 4px; }"
        )
        # Calibration UIs need real estate — ring picking on a 2k
        # detector is impossible at the default 600 px. Use 80% of
        # the parent's size as a starting point and let the user
        # resize from there.
        if parent is not None:
            parent_w = parent.window()
            self.resize(
                max(900, int(parent_w.width() * 0.85)),
                max(700, int(parent_w.height() * 0.85)),
            )
        else:
            self.resize(1200, 800)

        # Captured here when the user saves; the host reads these
        # back after exec() returns Accepted to populate the
        # Conversion dock's path fields.
        self.saved_poni_path: Path | None = None
        self.saved_mask_path: Path | None = None

        self._setup_pyfai_context()
        self._build_tasks()
        self._build_ui()
        self._wire_signals()

        # Pre-populate the image if the host handed one in. Done
        # last so the ExperimentTask widget has already been
        # mounted into the stack and its plot is ready to render.
        if initial_image is not None:
            self._seed_initial_image(initial_image)
        # Continuity with the Conversion dock: if PONI / mask paths
        # are already filled in there, treat them as the starting
        # point so the user doesn't have to re-pick them when they
        # came in here to produce the *other* file. Both seeders
        # also flag the corresponding "Add … to conversion" CTA as
        # enabled so the user can apply the existing path again
        # (idempotent) without going through Save first.
        if initial_poni is not None:
            self._seed_initial_poni(initial_poni)
        if initial_mask is not None:
            self._seed_initial_mask(initial_mask)

        # Default tab.
        if start_task not in _TASK_ORDER:
            start_task = "experiment"
        self._task_list.setCurrentRow(_TASK_ORDER.index(start_task))

    # --- pyFAI context setup -------------------------------------------

    def _setup_pyfai_context(self) -> None:
        """Build the singleton CalibrationContext + CalibrationModel.

        Uses our own QSettings namespace (``mlgidGUI / pyFAI-calib``)
        so pyFAI's own preferences (recent calibrants, last-used
        directories, etc.) persist between launches without
        colliding with mlgidGUI's main settings.
        """
        # pyFAI.resources.silx_integration registers icons + style;
        # safe to call repeatedly.
        try:
            pyFAI.resources.silx_integration()  # type: ignore[attr-defined]
        except Exception:
            # Older pyFAI versions don't expose this; harmless if
            # it's not there.
            logger.debug("suppressed exception in CalibrationDialog._setup_pyfai_context", exc_info=True)
            pass

        settings = calibration_settings()
        # CalibrationContext is a singleton; release any previous
        # instance so a second dialog launch starts from clean
        # state instead of inheriting the last session's QSettings
        # cursor inside the same process.
        CalibrationContext._releaseSingleton()
        ctx = CalibrationContext(settings)
        ctx.restoreSettings()

        # Apply the same defaults pyFAI-calib2 sets up. Passing
        # default-parsed argv keeps the model in a known state.
        parser = argparse.ArgumentParser()
        calib2.configure_parser_arguments(parser)
        opts, _ = parser.parse_known_args([])
        calib2.setup_model(ctx.getCalibrationModel(), opts)

        self._patch_pyfai_dialogs(ctx)

        self._calib_context = ctx
        self._calib_context.setParent(self)
        self._calib_model = ctx.getCalibrationModel()

    def _patch_pyfai_dialogs(self, ctx) -> None:
        """Make pyFAI's own file dialogs share mlgidGUI's browsing directory.

        Every calibration picker -- load image, load mask, load/save
        control points, and the "Save as PONI" button the whole
        dialog exists for -- is built inside pyFAI by
        ``CalibrationContext.createFileDialog``, which starts it at
        the process working directory (``/`` under a desktop
        launcher). Wrapping the factory is the only place we can
        reach them, and it is why saving a PONI now lands in the same
        directory the scan and the mask came from.

        Wrapped per instance, not on the class: the context is a
        singleton that ``_setup_pyfai_context`` releases and rebuilds
        on each launch, so patching the class would stack wrappers.
        """
        for name in ("createFileDialog", "createImageFileDialog"):
            original = getattr(ctx, name, None)
            if original is None:
                continue
            setattr(ctx, name, _adopting(original))

    def _build_tasks(self) -> None:
        """Instantiate the five canonical pyFAI task widgets."""
        self._task_widgets: dict[str, Any] = {
            "experiment": ExperimentTask(),
            "mask": MaskTask(),
            "peaks": PeakPickingTask(),
            "geometry": GeometryTask(),
            "integration": IntegrationTask(),
        }
        # Each task takes the shared CalibrationModel — that's how
        # state flows between tabs (e.g. an image picked in
        # Experiment becomes visible in Mask).
        for task in self._task_widgets.values():
            task.setModel(self._calib_model)

    # --- UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Main split: task list (left) + stacked task widget (right).
        body = QHBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body, 1)

        self._task_list = QListWidget()
        self._task_list.setFixedWidth(180)
        for key in _TASK_ORDER:
            item = QListWidgetItem(_TASK_LABELS[key])
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._task_list.addItem(item)
        body.addWidget(self._task_list)

        self._stack = QStackedWidget()
        body.addWidget(self._stack, 1)
        for key in _TASK_ORDER:
            self._stack.addWidget(self._task_widgets[key])

        # Visible separator between the task stack and the apply /
        # close bar so the action region reads as a distinct zone.
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(separator)

        # Bottom action row. Two prominent "Add … to conversion"
        # buttons let the user push the calibration results into
        # mlgidGUI's Conversion dock at any point during the
        # workflow — they stay enabled as long as the corresponding
        # file has been saved, and emit signals the host listens to
        # so the user can keep iterating in the dialog. ``Save
        # mask…`` writes the mask array to disk first (pyFAI's
        # IntegrationTask provides its own "Save as PONI" button so
        # we don't need one for PONI).
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        hint = QLabel(
            "Save the PONI (Integration tab) and mask, then push them to the "
            "Conversion dock with the buttons on the right →"
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        btn_row.addWidget(hint, 1)

        # The two "Add to conversion" buttons are the final step of the
        # calibration flow, so they carry the primary variant. They used
        # to hardcode an 18-line blue button skin here, which stayed blue
        # on a light theme and matched nothing else in the app.
        self._btn_save_mask = QPushButton("Save mask…")
        self._btn_save_mask.setToolTip(
            "Write the mask drawn in the Mask task to a .npy / "
            ".tif file. The 'Add mask to conversion' button "
            "lights up once a mask has been saved."
        )
        self._btn_save_mask.clicked.connect(self._on_save_mask)
        btn_row.addWidget(self._btn_save_mask)

        self._btn_apply_poni = _set_variant(
            QPushButton("Add PONI to conversion"), _PRIMARY)
        self._btn_apply_poni.setToolTip(
            "Push the saved PONI path into the Conversion dock's "
            "PONI field. Enabled once you've used the Integration "
            "task's 'Save as PONI' button."
        )
        self._btn_apply_poni.setEnabled(False)
        self._btn_apply_poni.clicked.connect(self._on_apply_poni)
        btn_row.addWidget(self._btn_apply_poni)

        self._btn_apply_mask = _set_variant(
            QPushButton("Add mask to conversion"), _PRIMARY)
        self._btn_apply_mask.setToolTip(
            "Push the saved mask path into the Conversion dock's "
            "Mask field. Enabled once you've saved a mask via "
            "'Save mask…'."
        )
        self._btn_apply_mask.setEnabled(False)
        self._btn_apply_mask.clicked.connect(self._on_apply_mask)
        btn_row.addWidget(self._btn_apply_mask)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_close)
        outer.addLayout(btn_row)

    def _wire_signals(self) -> None:
        # Task list selection drives the stack.
        self._task_list.currentRowChanged.connect(self._stack.setCurrentIndex)
        # Each pyFAI task widget has a "Next" button at the bottom
        # that emits ``nextTaskRequested``. pyFAI's own
        # ``CalibrationWindow`` is what normally bridges that signal
        # to the side-list cursor; without this hookup the Next
        # button looks broken because nothing moves. Mirror the
        # behaviour from
        # ``pyFAI/gui/CalibrationWindow.CalibrationWindow.__init__``.
        for task in self._task_widgets.values():
            try:
                task.nextTaskRequested.connect(self._advance_task)
            except Exception:
                # If a future pyFAI rev drops the signal, fail open —
                # the side list is still navigable.
                logger.debug("suppressed exception in CalibrationDialog._wire_signals", exc_info=True)
                pass
        # The final task has nothing to advance to; hide its Next
        # button so the user isn't left clicking a dead control.
        last_task = self._task_widgets[_TASK_ORDER[-1]]
        try:
            last_task.setNextStepVisible(False)
        except Exception:
            logger.debug("suppressed exception in CalibrationDialog._wire_signals", exc_info=True)
            pass
        # Observe poniFile changes so we can capture the path the
        # user just wrote, without disconnecting pyFAI's own save
        # dialog. The model's ``poniFile`` is updated *inside*
        # IntegrationTask.__saveAsPoni after the file is written.
        try:
            poni_model = self._calib_model.experimentSettingsModel().poniFile()
            poni_model.changed.connect(self._on_poni_model_changed)
        except Exception:
            # Defensive: if the API moves between pyFAI versions,
            # the worst that happens is the host doesn't auto-fill
            # the PONI path field (user can still type it). Don't
            # break the rest of the dialog.
            logger.debug("suppressed exception in CalibrationDialog._wire_signals", exc_info=True)
            pass
        # Mask model: pyFAI uses ``ImageFromFilenameModel`` for the
        # mask, which fires ``filenameChanged`` when a file path is
        # set by either the Experiment task's mask picker or the
        # Mask task's load action. Hook it so the apply-mask CTA
        # button activates when a mask is loaded from disk (not just
        # when our own "Save mask…" button writes one out).
        try:
            mask_model = self._calib_model.experimentSettingsModel().mask()
            mask_model.filenameChanged.connect(self._on_mask_filename_changed)
            # ``changed`` covers the "user drew a mask in MaskTask"
            # case — there's no file behind the array yet, but we
            # still want the apply-mask button live so the user can
            # click it and be funnelled into Save.
            mask_model.changed.connect(self._refresh_apply_mask_state)
            # If the host pre-loaded an image that included a
            # detector-derived mask, neither signal fires (no file,
            # no edit). Re-poll once after construction so the
            # button reflects whatever mask state the model already
            # holds.
            self._refresh_apply_mask_state()
        except Exception:
            logger.debug("suppressed exception in CalibrationDialog._wire_signals", exc_info=True)
            pass

    def _advance_task(self) -> None:
        """Move the task-list cursor (and so the stack) one step
        forward when a task emits ``nextTaskRequested``."""
        next_row = self._task_list.currentRow() + 1
        if 0 <= next_row < self._task_list.count():
            self._task_list.setCurrentRow(next_row)

    # --- Image / mask plumbing -----------------------------------------

    def _seed_initial_image(self, image: np.ndarray) -> None:
        """Push a 2D numpy array into the ExperimentTask as if the
        user had loaded it via the task's image-loader button.

        pyFAI's image model expects an ``ImageFilenameModel``-like
        object, but its experimentSettingsModel exposes a direct
        ``.image()`` accessor whose ``.setValue(ndarray)`` does the
        right thing for in-memory data.
        """
        try:
            arr = np.asarray(image)
            if arr.ndim != 2:
                # If a 3D stack snuck in, take the first frame.
                arr = arr[0] if arr.ndim == 3 else None
            if arr is None:
                return
            self._calib_model.experimentSettingsModel().image().setValue(arr)
        except Exception as exc:
            # Pre-fill is a convenience; if it fails the user can
            # still load an image manually inside the dialog. Log
            # to stderr so the failure isn't completely silent.
            import sys
            print(
                f"[CalibrationDialog] couldn't pre-load image: {exc}",
                file=sys.stderr,
            )

    def _seed_initial_poni(self, poni_path: str | Path) -> None:
        """Apply an existing PONI file to the calibration model.

        Mirrors the load path that ``pyFAI.app.calib2.setup_model``
        takes for its ``--poni`` argument: parse the PONI, push the
        detector onto the detector model, copy the seven geometry
        scalars onto a ``GeometryModel``, then transfer that onto
        ``fittedGeometry``. The path string is also stashed on
        ``poniFile()`` so any pyFAI status display picks it up.

        ``saved_poni_path`` is set so the "Add PONI to conversion"
        CTA button is enabled immediately — the user opened this
        dialog to produce a *mask*, the PONI is already valid, so
        re-applying it should be one click away.
        """
        try:
            p = Path(poni_path)
            if not p.exists():
                return
            from pyFAI.io.ponifile import PoniFile
            from pyFAI.gui.model.GeometryModel import GeometryModel

            poni_file = PoniFile()
            poni_file.read_from_file(str(p))

            settings = self._calib_model.experimentSettingsModel()
            with settings.poniFile().lockContext():
                settings.poniFile().setValue(str(p))
                settings.poniFile().setSynchronized(True)

            if poni_file.detector is not None:
                settings.detectorModel().setDetector(poni_file.detector)

            geom = GeometryModel()
            geom.distance().setValue(poni_file.dist)
            geom.poni1().setValue(poni_file.poni1)
            geom.poni2().setValue(poni_file.poni2)
            geom.rotation1().setValue(poni_file.rot1)
            geom.rotation2().setValue(poni_file.rot2)
            geom.rotation3().setValue(poni_file.rot3)
            geom.wavelength().setValue(poni_file.wavelength)
            self._calib_model.fittedGeometry().setFrom(geom)

            self.saved_poni_path = p
            self._btn_apply_poni.setEnabled(True)
        except Exception as exc:
            import sys
            print(
                f"[CalibrationDialog] couldn't pre-load PONI '{poni_path}': {exc}",
                file=sys.stderr,
            )

    def _seed_initial_mask(self, mask_path: str | Path) -> None:
        """Apply an existing mask file to the calibration model.

        Reads the file with ``pyFAI.io.image.read_image_data`` for
        TIFF/EDF/CBF/etc.; falls back to ``np.load`` for ``.npy``
        which pyFAI doesn't carry a reader for. Sets both the
        backing filename and the in-memory array on the model so
        downstream tasks see the mask immediately.

        ``saved_mask_path`` is set so the "Add mask to conversion"
        CTA button is enabled — useful when the user opened this
        dialog to produce a *PONI* but already has a working mask.
        """
        try:
            p = Path(mask_path)
            if not p.exists():
                return
            data = None
            if p.suffix.lower() == ".npy":
                data = np.load(str(p))
            else:
                from pyFAI.io import image as image_io
                data = image_io.read_image_data(str(p))
            if data is None:
                return

            settings = self._calib_model.experimentSettingsModel()
            with settings.mask().lockContext() as image_model:
                image_model.setFilename(str(p))
                image_model.setValue(np.asarray(data))
                image_model.setSynchronized(True)

            self.saved_mask_path = p
            self._btn_apply_mask.setEnabled(True)
        except Exception as exc:
            import sys
            print(
                f"[CalibrationDialog] couldn't pre-load mask '{mask_path}': {exc}",
                file=sys.stderr,
            )

    def _on_poni_model_changed(self) -> None:
        """Capture whatever PONI path pyFAI's IntegrationTask just
        wrote so the host can pull it on accept, and light up the
        "Add PONI to conversion" call-to-action button."""
        try:
            poni_model = self._calib_model.experimentSettingsModel().poniFile()
            value = poni_model.value()
            if value:
                self.saved_poni_path = Path(value)
                self._btn_apply_poni.setEnabled(True)
        except Exception:
            logger.debug("suppressed exception in CalibrationDialog._on_poni_model_changed", exc_info=True)
            return

    def _on_apply_poni(self) -> None:
        """Emit the apply-PONI signal and flash a brief
        confirmation so the user knows the click landed."""
        if self.saved_poni_path is None:
            return
        self.applyPoniRequested.emit(str(self.saved_poni_path))
        original = "Add PONI to conversion"
        self._btn_apply_poni.setText("✓ Added")
        QTimer.singleShot(
            1500, lambda: self._btn_apply_poni.setText(original)
        )

    def _on_apply_mask(self) -> None:
        """Emit the apply-mask signal and flash a brief
        confirmation.

        The button is enabled in two distinct cases: (a) a mask
        file was loaded via pyFAI's mask picker, so we already have
        a path on disk to forward, and (b) the user drew a mask in
        the Mask task but never saved it. In case (b) we prompt the
        Save-As dialog first so the conversion field receives a
        real, openable path."""
        if self.saved_mask_path is None:
            self._on_save_mask()
            if self.saved_mask_path is None:
                # User cancelled the save dialog; don't emit.
                return
        self.applyMaskRequested.emit(str(self.saved_mask_path))
        original = "Add mask to conversion"
        self._btn_apply_mask.setText("✓ Added")
        QTimer.singleShot(
            1500, lambda: self._btn_apply_mask.setText(original)
        )

    def _on_mask_filename_changed(self) -> None:
        """Fired by pyFAI when the user loads a mask file. Capture
        the path so apply-mask emits the right value, and light up
        the CTA button."""
        try:
            mask_model = self._calib_model.experimentSettingsModel().mask()
            filename = mask_model.filename()
            if filename:
                self.saved_mask_path = Path(filename)
                self._btn_apply_mask.setEnabled(True)
        except Exception:
            logger.debug("suppressed exception in CalibrationDialog._on_mask_filename_changed", exc_info=True)
            return

    def _refresh_apply_mask_state(self) -> None:
        """Sync the apply-mask button with the current mask model
        state. Called once after wiring so any pre-existing mask
        (loaded by the host or carried over from the previous
        dialog session via QSettings) lights up the button without
        the user having to re-trigger an event.

        The button enables when *either* a backing file path is
        known *or* a mask array is present in memory — for the
        in-memory-only case ``_on_apply_mask`` will prompt the user
        to save before emitting."""
        try:
            mask_model = self._calib_model.experimentSettingsModel().mask()
            filename = mask_model.filename()
            if filename:
                self.saved_mask_path = Path(filename)
                self._btn_apply_mask.setEnabled(True)
                return
            mask = mask_model.value()
            if mask is not None:
                # Array exists but no path on disk yet. Enable so
                # the user can click and be funnelled into Save.
                self._btn_apply_mask.setEnabled(True)
        except Exception:
            logger.debug("suppressed exception in CalibrationDialog._refresh_apply_mask_state", exc_info=True)
            pass

    def _on_save_mask(self) -> None:
        """Pop a Save-As dialog and write the current mask array.

        Reads ``model.experimentSettingsModel().mask().value()`` —
        the same numpy mask the MaskTask edits in place. Writes
        ``.npy`` natively or asks ``fabio`` / ``silx.io`` for
        ``.tif`` / ``.edf`` if the user picks one of those.
        """
        try:
            mask = self._calib_model.experimentSettingsModel().mask().value()
        except Exception as exc:
            QMessageBox.warning(
                self, "Mask unavailable",
                f"Couldn't read the current mask: {exc}",
            )
            return
        if mask is None:
            QMessageBox.information(
                self, "No mask",
                "Draw a mask in the Mask task before saving.",
            )
            return

        path_str, name_filter = file_dialogs.save_file(
            self,
            "Save mask",
            "NumPy mask (*.npy);;TIFF mask (*.tif *.tiff);;"
            "EDF mask (*.edf);;All files (*)",
            suggested_name="mask.npy",
        )
        if not path_str:
            return
        path = Path(path_str)
        # Add default extension if the user didn't type one.
        if path.suffix == "":
            if "*.npy" in name_filter:
                path = path.with_suffix(".npy")
            elif "*.tif" in name_filter:
                path = path.with_suffix(".tif")
            elif "*.edf" in name_filter:
                path = path.with_suffix(".edf")
            else:
                path = path.with_suffix(".npy")

        try:
            if path.suffix.lower() == ".npy":
                np.save(path, np.asarray(mask, dtype=np.int8))
            elif path.suffix.lower() in (".tif", ".tiff"):
                import fabio
                fabio.tifimage.tifimage(
                    data=np.asarray(mask, dtype=np.int8)
                ).save(str(path))
            elif path.suffix.lower() == ".edf":
                import fabio
                fabio.edfimage.edfimage(
                    data=np.asarray(mask, dtype=np.int8)
                ).save(str(path))
            else:
                np.save(path, np.asarray(mask, dtype=np.int8))
                path = path.with_suffix(".npy")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Couldn't write mask to {path}:\n{exc}",
            )
            return
        self.saved_mask_path = path
        # Light up the "Add mask to conversion" call-to-action
        # button now that we have a path to hand back to the host.
        self._btn_apply_mask.setEnabled(True)
