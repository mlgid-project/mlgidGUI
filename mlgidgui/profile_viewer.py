from __future__ import annotations

import os
import warnings

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg

from mlgidgui import theme_tokens
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from mlgidgui.fit import (
    GaussianFit,
    fit_gaussian_on_axis,
    gaussian_at_stored_shape,
)
from mlgidgui.manual_fit import PeakBox
from mlgidgui.profile_masking import neighbour_mask
from mlgidgui.image_viewer import (
    OVERLAY_STYLE,
    ManualPeak,
    SelectedPeak,
    _disable_viewport_scroll,
)

# Multiple of box width on each side over which the rendered Gaussian curve
# extends past the box bounds. Big enough that the tails fade into the
# baseline; small enough that very tight peaks don't render as a spike on a
# nearly empty axis.
FIT_RENDER_PAD_FACTOR = 3.0

# Interval for fitted / matched peaks, expressed as a multiple of the
# stored width around the peak center. Stored widths are pygidfit's ``2σ``
# on both axes (shared by Add-to-fitted 1D + 2D + pipeline; see
# ``manual_fit.fit_one_peak``). Nothing is fitted over this interval any
# more (a saved peak is drawn as stored, see ``_update_fit_curves``); it
# only sizes the span the curve is rendered over.
FITTED_FIT_REGION_FACTOR = 3.0
# Half-extent, in stored widths, of the band a saved (fitted / matched)
# peak's profile averages over on the other axis. The stored width is
# pygidfit's 2σ, so 1.0 means ±2σ.
STORED_BAND_HALF_FACTOR = 1.0

PROFILE_PEN_COLOR = "#e8e8e8"
PROFILE_PEN_WIDTH = 1.2

REGION_COLOR = OVERLAY_STYLE["manual"]["color"]
REGION_BRUSH_ALPHA = 30  # 0-255

# Distinct from the white data curve and the yellow region markers.
FIT_PEN_COLOR = "#ff7eb6"
FIT_PEN_WIDTH = 1.6
# Centre / border marks drawn for a saved (fitted / matched) selection.
# Same colour as the fit curve so the two read as one "this peak's fit"
# layer; the centre is solid, the box edges dashed.
STORED_MARK_COLOR = FIT_PEN_COLOR
STORED_MARK_CENTRE_WIDTH = 1.8
STORED_MARK_EDGE_WIDTH = 1.2
# Linear background under a saved peak's curve, drawn on its own so the
# line the height is measured above is visible.
BACKGROUND_PEN_WIDTH = 1.2

# Multiple of the box width added on each side when auto-zooming to a peak.
# View window = (1 + 2 * ZOOM_PAD_FACTOR) * box_width. Matched to
# FIT_RENDER_PAD_FACTOR below so the rendered Gaussian's tails are visible
# without manual panning.
ZOOM_PAD_FACTOR = 3.0


class ProfileViewer(QWidget):
    """Two stacked plots showing the radial and angular intensity profiles
    of the polar image.

    Radial profile is mean intensity across all angles, indexed by radius.
    Angular profile is mean intensity across all radii, indexed by angle.
    When a manual peak is selected, draggable ``LinearRegionItem``s appear in
    each profile representing the peak's bounds. Dragging a region mutates the
    peak in place and emits ``peakGeometryChanged`` so the 2D ROI can sync.
    """

    peakGeometryChanged = Signal(object)  # ManualPeak whose geometry changed
    # Live signal fired during a detected-peak region drag — carries the
    # updated ``SelectedPeak`` snapshot so the image viewer can mutate
    # its in-memory PeakTable and re-render the colored overlay in
    # real time. Mirrors how the image-side ROI drag updates the
    # profile via ``peakGeometryChanged`` (image_viewer.py) — same
    # mechanism, opposite direction.
    detectedPeakGeometryChanged = Signal(object)  # SelectedPeak (kind="detected")
    # Fires once at the end of a detected-peak region drag with the
    # final SelectedPeak; the host writes the new geometry through to
    # ``detected_peaks`` on disk via the existing
    # ``peakRowWriteRequested`` flow.
    detectedPeakBorderCommit = Signal(object)  # SelectedPeak (kind="detected")
    # Emitted whenever the cached fit pair changes (computed, cleared, or both
    # axes failed). Carries (radial_fit, angular_fit) — either may be None.
    fitParamsChanged = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Outer column: small toolbar row on top, two side-by-side
        # plots below. The toolbar carries the Log-y toggle that
        # applies to both plots simultaneously.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        self._log_y_check = QCheckBox("Log y")
        self._log_y_check.setToolTip(
            "Switch both profile y-axes to log10 scale. Useful for "
            "GIWAXS data where peak amplitudes span multiple orders "
            "of magnitude."
        )
        self._log_y_check.toggled.connect(self._on_log_y_toggled)
        toolbar.addWidget(self._log_y_check)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        # Plot row stays a QHBoxLayout so the radial + angular plots
        # sit side-by-side as before.
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        outer.addLayout(layout, 1)

        pen = pg.mkPen(theme_tokens.color("profile_curve"),
                       width=PROFILE_PEN_WIDTH)
        region_pen = pg.mkPen(REGION_COLOR, width=1.5)
        region_pen.setCosmetic(True)
        region_brush = pg.mkBrush(QColor(*_hex_to_rgb(REGION_COLOR), REGION_BRUSH_ALPHA))
        fit_pen = pg.mkPen(FIT_PEN_COLOR, width=FIT_PEN_WIDTH)
        fit_pen.setCosmetic(True)
        bg_pen = pg.mkPen(FIT_PEN_COLOR, width=BACKGROUND_PEN_WIDTH)
        bg_pen.setStyle(Qt.PenStyle.DotLine)
        bg_pen.setCosmetic(True)

        self._radial_plot = pg.PlotWidget()
        self._radial_plot.setLabel("bottom", "radius", units="Å⁻¹")
        self._radial_plot.setLabel("left", "intensity")
        self._radial_plot.setTitle("Radial profile")
        self._radial_plot.showGrid(x=True, y=True, alpha=0.2)
        self._radial_plot.getViewBox().setMouseEnabled(x=True, y=False)
        # Bottom margin keeps the "radius" axis label off the viewport's
        # lower edge — pyqtgraph's auto-sized bottom-axis cell can leave
        # too little room and the clipped label opens a small scrollable
        # region in the dock.
        self._radial_plot.plotItem.layout.setContentsMargins(0, 0, 0, 12)
        # PlotWidget is a QGraphicsView under the hood — pin the
        # scrollbars off and drop the frame border so the plot
        # contents (axes + labels) sit flush against the dock and
        # the user can't accidentally pan/scroll the scene.
        self._radial_plot.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._radial_plot.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._radial_plot.setFrameStyle(QFrame.Shape.NoFrame)
        _disable_viewport_scroll(self._radial_plot)
        self._radial_curve = self._radial_plot.plot([], [], pen=pen)
        self._radial_fit_curve = self._radial_plot.plot([], [], pen=fit_pen)
        self._radial_bg_curve = self._radial_plot.plot([], [], pen=bg_pen)
        self._radial_region = pg.LinearRegionItem(
            values=(0.0, 0.0), brush=region_brush, pen=region_pen
        )
        self._radial_region.setZValue(50)
        self._radial_region.setVisible(False)
        self._radial_region.sigRegionChanged.connect(self._on_radial_changed)
        self._radial_region.sigRegionChangeFinished.connect(self._on_radial_finished)
        self._radial_plot.addItem(self._radial_region)
        self._radial_marks = self._build_stored_marks(self._radial_plot)
        layout.addWidget(self._radial_plot)

        self._angular_plot = pg.PlotWidget()
        self._angular_plot.setLabel("bottom", "angle", units="deg")
        self._angular_plot.setLabel("left", "intensity")
        self._angular_plot.setTitle("Angular profile")
        self._angular_plot.showGrid(x=True, y=True, alpha=0.2)
        self._angular_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self._angular_plot.plotItem.layout.setContentsMargins(0, 0, 0, 12)
        self._angular_plot.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._angular_plot.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._angular_plot.setFrameStyle(QFrame.Shape.NoFrame)
        _disable_viewport_scroll(self._angular_plot)
        self._angular_curve = self._angular_plot.plot([], [], pen=pen)
        self._angular_fit_curve = self._angular_plot.plot([], [], pen=fit_pen)
        self._angular_bg_curve = self._angular_plot.plot([], [], pen=bg_pen)
        self._angular_region = pg.LinearRegionItem(
            values=(0.0, 0.0), brush=region_brush, pen=region_pen
        )
        self._angular_region.setZValue(50)
        self._angular_region.setVisible(False)
        self._angular_region.sigRegionChanged.connect(self._on_angular_changed)
        self._angular_region.sigRegionChangeFinished.connect(self._on_angular_finished)
        self._angular_plot.addItem(self._angular_region)
        self._angular_marks = self._build_stored_marks(self._angular_plot)
        layout.addWidget(self._angular_plot)

        self._polar_stack: np.ndarray | None = None
        self._radius: np.ndarray | None = None
        self._angle: np.ndarray | None = None
        # SelectedPeak (any kind) so profiles/fits can render for detected /
        # fitted / matched / manual alike. Region drags only mutate when the
        # kind is "manual" — file-resident peaks edit through the 2D ROI.
        self._selected: SelectedPeak | None = None
        self._current_frame = 0
        # ``provider(frame, SelectedPeak) -> list[PeakBox]``: the other
        # peaks to mask out of the selected peak's profiles. None (or an
        # empty list) leaves the profiles untouched. The host owns the
        # on/off setting and answers [] while it is off.
        self._neighbour_provider = None
        self._cluster_distance_provider = None
        # When True, the angular Gaussian fit is skipped + cleared. Driven
        # by the host (MainWindow) from the parameter panel's ring toggle:
        # if Add-to-fitted will commit a ring there's no angular dimension
        # to save, so showing the angular fit would be misleading.
        self._skip_angular_fit: bool = False
        # Last successfully-computed fits — exposed via last_fit_params() so
        # the parameter panel's "Add to fitted" button can write them to the
        # fitted_peaks dataset. Cleared whenever the curves are cleared.
        self._last_radial_fit: GaussianFit | None = None
        self._last_angular_fit: GaussianFit | None = None

    # -- Public API --

    def set_polar_stack(
        self, polar_stack: np.ndarray, radius: np.ndarray, angle: np.ndarray
    ) -> None:
        """Load the polar-stack data; renders frame 0 immediately."""
        self._polar_stack = polar_stack
        self._radius = radius
        self._angle = angle
        self._current_frame = 0
        self._recompute_curves()

    def set_neighbour_provider(self, provider) -> None:
        """Install the callable that lists the peaks to mask (see above)."""
        self._neighbour_provider = provider

    def set_cluster_distance_provider(self, provider) -> None:
        """Install the callable returning pygidfit's clustering distance
        for peaks (pixels of its polar grid; None = the default). Read
        each time the profiles are drawn, so a changed Pipeline setting
        applies on the next redraw."""
        self._cluster_distance_provider = provider

    def refresh(self) -> None:
        """Redraw the profiles, e.g. after the masking setting changed."""
        self._recompute_curves()

    def _masked_for_neighbours(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """``img`` with the neighbours of the selected peak set to NaN.

        Returns ``(image, masked)``. The cached frame is never edited in
        place; a masked frame is a copy. Any provider failure means no
        masking rather than a broken plot.
        """
        peak = self._selected
        if peak is None or self._neighbour_provider is None:
            return img, False
        try:
            boxes = self._neighbour_provider(self._current_frame, peak)
        except Exception:
            return img, False
        if not boxes:
            return img, False
        target = PeakBox(
            radius=float(peak.radius), radius_width=float(peak.radius_width),
            angle=float(peak.angle), angle_width=float(peak.angle_width),
        )
        distance = None
        if self._cluster_distance_provider is not None:
            try:
                distance = self._cluster_distance_provider()
            except Exception:
                distance = None
        mask = neighbour_mask(
            img.shape, self._radius, self._angle, target, boxes,
            cluster_distance=distance,
        )
        if mask is None:
            return img, False
        out = np.array(img, dtype=np.float32 if img.dtype.kind != "f" else img.dtype)
        out[mask] = np.nan
        return out, True

    def set_frame(self, frame: int) -> None:
        if self._polar_stack is None:
            return
        if not 0 <= frame < self._polar_stack.shape[0]:
            return
        self._current_frame = frame
        self._recompute_curves()

    def clear(self) -> None:
        self._radial_curve.setData([], [])
        self._angular_curve.setData([], [])
        self._radial_fit_curve.setData([], [])
        self._angular_fit_curve.setData([], [])
        self._clear_backgrounds()
        self._hide_stored_marks()
        self._polar_stack = None
        self._radius = None
        self._angle = None
        self._set_fit_cache(None, None)
        self.set_selected_peak(None)

    def apply_theme_colors(self, background, foreground) -> None:
        """Recolour both profile plots' background + axes live for a theme
        switch (pyqtgraph bakes colours in at creation, so an explicit
        push is needed for the already-built plots)."""
        pen = pg.mkPen(foreground)
        # The data curves are the reason this matters: a #e8e8e8 trace on
        # the light theme's #fafafa ground is invisible.
        curve_pen = pg.mkPen(theme_tokens.color("profile_curve"),
                             width=PROFILE_PEN_WIDTH)
        for curve in (getattr(self, "_radial_curve", None),
                      getattr(self, "_angular_curve", None)):
            if curve is not None:
                try:
                    curve.setPen(curve_pen)
                except Exception:
                    pass
        for plot in (self._radial_plot, self._angular_plot):
            try:
                plot.setBackground(background)
            except Exception:
                pass
            pi = plot.getPlotItem()
            for name in ("left", "bottom", "right", "top"):
                ax = pi.getAxis(name)
                if ax is None:
                    continue
                try:
                    ax.setPen(pen)
                    ax.setTextPen(pen)
                except Exception:
                    pass

    def set_skip_angular_fit(self, skip: bool) -> None:
        """Tell the viewer not to compute or render the angular Gaussian.

        Used while the parameter panel's ring toggle is active — saving
        as ring drops the angular dimension entirely, so showing a fit
        the user can't actually save would be misleading.
        """
        if self._skip_angular_fit == skip:
            return
        self._skip_angular_fit = skip
        # Re-render so the curve appears / disappears immediately.
        self._recompute_curves()

    def set_fit_curves_visible(self, visible: bool) -> None:
        """Show or hide the pink Gaussian-fit overlay on both plots.

        Driven by the host from the fit-mode radio: in 2D mode the
        projected 1D Gaussian from pygidfit's 2D fit does not
        perfectly match the integrated 1D profile (the 2D centroid +
        integration-window choice differ enough to be visible). The
        mismatch was misleading, so 2D mode hides the pink overlay
        entirely and lets the cyan image-side preview box be the
        single source of truth for "what the next Add-to-fitted will
        save". 1D mode shows the pink curves again because those are
        the scipy fits Add-to-fitted (1D) actually stores.
        """
        self._radial_fit_curve.setVisible(visible)
        self._angular_fit_curve.setVisible(visible)
        self._radial_bg_curve.setVisible(visible)
        self._angular_bg_curve.setVisible(visible)

    def _on_log_y_toggled(self, log: bool) -> None:
        """Apply the Log-y toggle to both profile plots.

        pyqtgraph's ``setLogMode`` transforms data internally during
        rendering — no need to remap the cached curve data ourselves
        and no impact on the linear-space fit math (the fit curve is
        also auto-log-scaled by pyqtgraph). Non-positive intensities
        (rare on mean-of-row / mean-of-column profiles) render as
        -inf and effectively drop from the visible trace.
        """
        for plot in (self._radial_plot, self._angular_plot):
            plot.setLogMode(x=False, y=log)

    def last_fit_params(self) -> dict[str, GaussianFit | None]:
        """Most recent radial / angular Gaussian fits, or None if not fitted.

        Used by MainWindow's "Add to fitted" handler to read amplitude /
        center / sigma without re-running the fit. Stays valid until the
        next ``_recompute_curves`` call (frame change, deselect, etc.).
        """
        return {
            "radial": self._last_radial_fit,
            "angular": self._last_angular_fit,
        }

    def radius_axis(self) -> np.ndarray | None:
        """Polar radius axis the viewer is currently plotting on, or
        None if no polar stack has been loaded yet."""
        return self._radius

    def angle_axis(self) -> np.ndarray | None:
        """Polar angle axis the viewer is currently plotting on."""
        return self._angle

    # -- Selected-peak edge handles --

    def set_selected_peak(self, peak: SelectedPeak | None) -> None:
        """Show / hide / sync the edge regions for the given selection.

        Profiles re-integrate over the *complementary* axis of the box (radial
        profile uses the box's angular range, angular profile uses the radial
        range). Deselecting restores full integration over the entire image.
        Also auto-zooms each profile to a window slightly wider than the box.

        Region markers (yellow LinearRegionItems) are shown only for kinds
        the user actively shapes — ``manual`` (draggable) and ``detected``
        (read-only but tracks the box). For ``fitted`` / ``matched`` the
        box is fixed by the storage convention and the region markers
        would just clutter the view; the live Gaussian curve still renders.
        """
        self._selected = peak
        # Only manual + detected are user-shaped — show the region marker
        # only for those. Fitted / matched still get the Gaussian curve
        # (handled in _update_fit_curves) but no region overlay.
        show_regions = peak is not None and peak.kind in ("manual", "detected")
        # For ring peaks (or any peak whose angle_width is non-finite) the
        # angular region would span the entire axis or fail to render — hide
        # it so the profile stays usable. The radial region still tracks.
        is_ring_box = (
            peak is not None
            and (peak.is_ring or not np.isfinite(peak.angle_width))
        )

        # Region drag is now allowed for both manual AND detected
        # peaks. Manual drags update in-memory state only (the peak
        # lives in the viewer's ``_manual_peaks`` list); detected
        # drags also commit to ``detected_peaks`` on disk via the
        # existing ``peakRowWriteRequested`` flow on drag-end. See
        # ``_on_radial_finished`` / ``_on_angular_finished``.
        is_draggable = peak is not None and peak.kind in ("manual", "detected")
        self._radial_region.setVisible(show_regions)
        self._radial_region.setMovable(is_draggable)
        self._angular_region.setVisible(show_regions and not is_ring_box)
        self._angular_region.setMovable(is_draggable)

        if peak is not None:
            self.sync_regions_from_peak(peak)
            self._zoom_to_peak(peak)
        else:
            self._radial_plot.enableAutoRange()
            self._angular_plot.enableAutoRange()
        self._recompute_curves()

    def sync_regions_from_peak(self, peak: SelectedPeak | ManualPeak) -> None:
        """Programmatically update region bounds without re-emitting changes.

        Accepts ``SelectedPeak`` (the new wire) or ``ManualPeak`` (legacy
        callers). Returns silently when the argument doesn't match the
        currently-shown selection.
        """
        if isinstance(peak, ManualPeak):
            if (
                self._selected is None
                or self._selected.kind != "manual"
                or self._selected.manual_ref is not peak
            ):
                return
            sel = self._selected  # SelectedPeak with kind == "manual"
        else:
            if peak is not self._selected:
                return
            sel = peak
        r_range = _radial_fit_range(sel)
        a_range = _angular_fit_range(sel)
        # Geometry changes can flip ``is_ring`` (via the Save-fitted-as-
        # ring toggle) or the angular_width (via the 2D ROI), so the
        # angular region's visibility has to be re-evaluated here.
        # ``set_selected_peak`` only fires on selection *change*, which
        # would otherwise leave the angular region painting the pre-
        # ring borders even after the box is expanded to a full sweep.
        # Radial visibility / movability depend only on selection kind,
        # which doesn't change here.
        is_ring_box = sel.is_ring or not np.isfinite(sel.angle_width)
        show_regions = sel.kind in ("manual", "detected")
        self._radial_region.setVisible(show_regions)
        self._angular_region.setVisible(show_regions and not is_ring_box)

        self._radial_region.blockSignals(True)
        try:
            self._radial_region.setRegion(
                (float(r_range[0]), float(r_range[1]))
            )
        finally:
            self._radial_region.blockSignals(False)
        # Skip the angular region for ring boxes — its bounds would be
        # ±inf / span the entire axis. setVisible already hid it.
        if a_range is not None:
            self._angular_region.blockSignals(True)
            try:
                self._angular_region.setRegion(
                    (float(a_range[0]), float(a_range[1]))
                )
            finally:
                self._angular_region.blockSignals(False)
        # The integration window changed → recompute the slice profiles.
        self._recompute_curves()
        # When the box moved via the 2D ROI (manual/detected drag), the
        # region may now sit outside the profile's current X range.
        # Pan/expand each plot so the borders stay visible — without
        # this the user dragging the ROI past the visible profile
        # window loses sight of the interval.
        if self._radial_region.isVisible():
            self._ensure_region_in_view(self._radial_plot, self._radial_region)
        if a_range is not None and self._angular_region.isVisible():
            self._ensure_region_in_view(self._angular_plot, self._angular_region)

    # -- Internals --

    def _recompute_curves(self) -> None:
        """Re-render both profile curves and any fit overlays.

        Selected: radial profile averages columns within the angular
        slice of the integration box (the user-drawn ROI,
        ``self._selected.{radius, angle, *_width}``); angular profile
        averages rows within the radial slice. Unselected: full-image
        averages, fit curves cleared.
        """
        if (
            self._polar_stack is None
            or self._radius is None
            or self._angle is None
        ):
            return
        if not 0 <= self._current_frame < self._polar_stack.shape[0]:
            return
        # Selection-change signals can race the silx detach/reattach
        # dance (pipeline runs, file close): the FrameSource is briefly
        # released and the lazy polar stack raises until the host
        # re-acquires. Bail silently — the next `frameChanged` or
        # `selectionChanged` after reattach will redraw correctly.
        try:
            img = self._polar_stack[self._current_frame]
        except (RuntimeError, ValueError, OSError, KeyError):
            return

        img, masked = self._masked_for_neighbours(img)
        suffix = " (neighbours masked)" if masked else ""
        self._radial_plot.setTitle("Radial profile" + suffix)
        self._angular_plot.setTitle("Angular profile" + suffix)

        # Polar maps legitimately hold all-NaN rows/columns — bins the
        # detector never covers (missing wedge, gaps). ``nanmean`` of
        # such a slice is NaN by design and pyqtgraph gaps the curve
        # there, but numpy emits a "Mean of empty slice" RuntimeWarning
        # for every one, spamming the console on each frame or
        # selection change. Suppress exactly that warning, exactly here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice",
                category=RuntimeWarning,
            )
            if self._selected is not None:
                peak = self._selected
                box_r, box_dr = peak.radius, peak.radius_width
                box_a, box_da = peak.angle, peak.angle_width
                # Half-extent of the band each profile averages over,
                # as a multiple of the stored/drawn width. A manual /
                # detected box is a drawn extent (width / 2 each side).
                # A saved peak stores 2σ, so width / 2 would average
                # over only ±σ and the profile height would depend on
                # how wide the box it came from was; the full stored
                # width (±2σ) takes in ~95% of the other axis' Gaussian
                # whatever the original box was.
                half = (
                    STORED_BAND_HALF_FACTOR
                    if peak.kind in ("fitted", "matched") else 0.5
                )
                a_slice = _bounds_to_slice(
                    self._angle,
                    box_a - half * box_da,
                    box_a + half * box_da,
                )
                r_slice = _bounds_to_slice(
                    self._radius,
                    box_r - half * box_dr,
                    box_r + half * box_dr,
                )
                radial_src = (
                    img[:, a_slice] if a_slice.stop > a_slice.start else img
                )
                angular_src = (
                    img[r_slice, :] if r_slice.stop > r_slice.start else img
                )
                radial = np.nanmean(radial_src, axis=1)
                angular = np.nanmean(angular_src, axis=0)
            else:
                radial = np.nanmean(img, axis=1)
                angular = np.nanmean(img, axis=0)

        self._radial_curve.setData(self._radius, radial)
        self._angular_curve.setData(self._angle, angular)

        if self._selected is not None:
            self._update_fit_curves(self._selected, radial, angular)
        else:
            self._radial_fit_curve.setData([], [])
            self._angular_fit_curve.setData([], [])
            self._clear_backgrounds()
            self._hide_stored_marks()
            self._set_fit_cache(None, None)

    @staticmethod
    def _build_stored_marks(plot: pg.PlotWidget) -> tuple:
        """Centre line + two edge lines for a saved peak, hidden until used."""
        centre_pen = pg.mkPen(STORED_MARK_COLOR, width=STORED_MARK_CENTRE_WIDTH)
        centre_pen.setCosmetic(True)
        edge_pen = pg.mkPen(STORED_MARK_COLOR, width=STORED_MARK_EDGE_WIDTH)
        edge_pen.setStyle(Qt.PenStyle.DashLine)
        edge_pen.setCosmetic(True)
        lines = (
            pg.InfiniteLine(angle=90, movable=False, pen=edge_pen),
            pg.InfiniteLine(angle=90, movable=False, pen=centre_pen),
            pg.InfiniteLine(angle=90, movable=False, pen=edge_pen),
        )
        for line in lines:
            line.setZValue(40)
            line.setVisible(False)
            plot.addItem(line, ignoreBounds=True)
        return lines

    @staticmethod
    def _set_marks(marks: tuple, centre: float, width: float) -> None:
        """Place (lo edge, centre, hi edge); edges at ``centre ± width/2``."""
        lo, mid, hi = marks
        lo.setPos(centre - width / 2.0)
        mid.setPos(centre)
        hi.setPos(centre + width / 2.0)
        for line in marks:
            line.setVisible(True)

    def _hide_stored_marks(self) -> None:
        for marks in (self._radial_marks, self._angular_marks):
            for line in marks:
                line.setVisible(False)

    def _show_stored_marks(self, peak: SelectedPeak) -> None:
        """Mark a saved peak's centre and box edges in both profiles.

        The edges sit where the drawn box ends (``centre ± width / 2``,
        the stored width being the full box extent), so the profile
        shows the stored numbers and nothing the viewer computed. A
        ring has no angular extent (``angle_width`` is inf), so only the
        radial marks appear for it.
        """
        r, dr = peak.radius, peak.radius_width
        if np.isfinite(r) and np.isfinite(dr) and dr > 0:
            self._set_marks(self._radial_marks, float(r), float(dr))
        else:
            for line in self._radial_marks:
                line.setVisible(False)
        a, da = peak.angle, peak.angle_width
        if not peak.is_ring and np.isfinite(a) and np.isfinite(da) and da > 0:
            self._set_marks(self._angular_marks, float(a), float(da))
        else:
            for line in self._angular_marks:
                line.setVisible(False)

    def _clear_backgrounds(self) -> None:
        self._radial_bg_curve.setData([], [])
        self._angular_bg_curve.setData([], [])

    @staticmethod
    def _draw_background(curve, fit: GaussianFit | None, enabled: bool) -> None:
        """Draw ``fit``'s linear background (``slope * x + intercept``) over
        its own span, or clear it. Only a saved peak's curve gets one; a
        live fit keeps just its curve."""
        if enabled and fit is not None:
            curve.setData(fit.x, fit.slope * fit.x + fit.intercept)
        else:
            curve.setData([], [])

    def _update_fit_curves(
        self, peak: SelectedPeak, radial: np.ndarray, angular: np.ndarray
    ) -> None:
        # Two displays, by what the selection is:
        #   manual / detected  → a live 1D scipy fit over the box (a
        #                        forecast of what Add-to-fitted would
        #                        save; nothing is stored yet).
        #   fitted / matched   → the STORED row: a Gaussian centred on
        #                        the stored centre with the stored width,
        #                        its maximum read off the profile at that
        #                        centre above the flank background
        #                        (``gaussian_at_stored_shape``), plus
        #                        marks at the centre and the two edges
        #                        where the drawn box ends. Never a
        #                        refit: a stored value that disagrees
        #                        with the data shows as a curve off the
        #                        profile peak.
        stored = peak.kind in ("fitted", "matched")
        if stored:
            self._show_stored_marks(peak)
        else:
            self._hide_stored_marks()
        rfit: GaussianFit | None = None
        afit: GaussianFit | None = None
        r_range = _radial_fit_range(peak)
        if self._radius is not None and r_range is not None:
            r_lo, r_hi = r_range
            r_pad = FIT_RENDER_PAD_FACTOR * (r_hi - r_lo)
            render_r = (r_lo - r_pad, r_hi + r_pad)
            if stored:
                rfit = gaussian_at_stored_shape(
                    self._radius, radial, peak.radius, peak.radius_width,
                    render_range=render_r,
                )
            else:
                rfit = fit_gaussian_on_axis(
                    self._radius, radial, peak.radius, peak.radius_width,
                    fit_range=(r_lo, r_hi),
                    render_range=render_r,
                )
            if rfit is not None:
                self._radial_fit_curve.setData(rfit.x, rfit.y)
            else:
                self._radial_fit_curve.setData([], [])
            self._draw_background(self._radial_bg_curve, rfit, stored)
        # Skip angular fit when the host has flagged ring-save mode — the
        # angular dimension won't be saved, so showing a fit would lie
        # about what Add-to-fitted will write.
        a_range = None if self._skip_angular_fit else _angular_fit_range(peak)
        if self._angle is not None and a_range is not None:
            a_lo, a_hi = a_range
            a_pad = FIT_RENDER_PAD_FACTOR * (a_hi - a_lo)
            render_a = (a_lo - a_pad, a_hi + a_pad)
            if stored:
                afit = gaussian_at_stored_shape(
                    self._angle, angular, peak.angle, peak.angle_width,
                    render_range=render_a,
                )
            else:
                afit = fit_gaussian_on_axis(
                    self._angle, angular, peak.angle, peak.angle_width,
                    fit_range=(a_lo, a_hi),
                    render_range=render_a,
                )
            if afit is not None:
                self._angular_fit_curve.setData(afit.x, afit.y)
            else:
                self._angular_fit_curve.setData([], [])
            self._draw_background(self._angular_bg_curve, afit, stored)
        elif self._angle is not None:
            self._angular_fit_curve.setData([], [])
            self._angular_bg_curve.setData([], [])
        self._set_fit_cache(rfit, afit)

    def _set_fit_cache(
        self, rfit: GaussianFit | None, afit: GaussianFit | None,
    ) -> None:
        if rfit is self._last_radial_fit and afit is self._last_angular_fit:
            return
        self._last_radial_fit = rfit
        self._last_angular_fit = afit
        self.fitParamsChanged.emit(rfit, afit)

    def _zoom_to_peak(self, peak: SelectedPeak) -> None:
        """Set each profile's X range to a window slightly wider than the box,
        and let Y auto-scale to whatever data is currently visible in X.
        """
        for plot, center, width in (
            (self._radial_plot, peak.radius, peak.radius_width),
            (self._angular_plot, peak.angle, peak.angle_width),
        ):
            if not (np.isfinite(center) and np.isfinite(width)) or width <= 0:
                plot.enableAutoRange()
                continue
            half = width / 2.0 + ZOOM_PAD_FACTOR * width
            plot.setXRange(center - half, center + half, padding=0)
            vb = plot.getViewBox()
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _on_radial_changed(self) -> None:
        # Live region drag on the radial profile. Manual peaks update
        # the in-memory ManualPeak via the existing peakGeometryChanged
        # path. Detected peaks update the SelectedPeak snapshot and
        # fire ``detectedPeakGeometryChanged`` so the host can refresh
        # the image overlay; the disk write fires once on drag-end
        # via ``_on_radial_finished``. Region is set non-movable for
        # other kinds so this slot doesn't fire for them.
        if self._selected is None:
            return
        lo, hi = self._radial_region.getRegion()
        new_w = abs(float(hi) - float(lo))
        new_r = (float(hi) + float(lo)) / 2.0
        self._selected.radius_width = new_w
        self._selected.radius = new_r
        kind = self._selected.kind
        if kind == "manual" and self._selected.manual_ref is not None:
            self._selected.manual_ref.radius_width = new_w
            self._selected.manual_ref.radius = new_r
        # Angular profile slices over the radial range — needs refresh
        # for both manual and detected.
        self._recompute_curves()
        # Keep the dragged region inside the visible plot area so the
        # user can always see what they're editing.
        self._ensure_region_in_view(self._radial_plot, self._radial_region)
        if kind == "manual" and self._selected.manual_ref is not None:
            self.peakGeometryChanged.emit(self._selected.manual_ref)
        elif kind == "detected":
            self.detectedPeakGeometryChanged.emit(self._selected)

    def _on_angular_changed(self) -> None:
        if self._selected is None:
            return
        lo, hi = self._angular_region.getRegion()
        new_h = abs(float(hi) - float(lo))
        new_a = (float(hi) + float(lo)) / 2.0
        self._selected.angle_width = new_h
        self._selected.angle = new_a
        kind = self._selected.kind
        if kind == "manual" and self._selected.manual_ref is not None:
            self._selected.manual_ref.angle_width = new_h
            self._selected.manual_ref.angle = new_a
        # Radial profile slices over the angular range — needs refresh.
        self._recompute_curves()
        self._ensure_region_in_view(self._angular_plot, self._angular_region)
        if kind == "manual" and self._selected.manual_ref is not None:
            self.peakGeometryChanged.emit(self._selected.manual_ref)
        elif kind == "detected":
            self.detectedPeakGeometryChanged.emit(self._selected)

    def _on_radial_finished(self) -> None:
        # Commit boundary for detected peaks — fires once when the
        # user releases the radial region drag. Manual peaks need no
        # commit (they live in memory only). For detected we emit
        # ``detectedPeakBorderCommit`` so the host writes the new
        # geometry through to ``detected_peaks`` on disk via the
        # existing ``peakRowWriteRequested`` flow (same flow the
        # image-side ROI drag-end uses).
        if self._selected is None or self._selected.kind != "detected":
            return
        self.detectedPeakBorderCommit.emit(self._selected)

    def _on_angular_finished(self) -> None:
        if self._selected is None or self._selected.kind != "detected":
            return
        self.detectedPeakBorderCommit.emit(self._selected)

    def _ensure_region_in_view(
        self, plot: pg.PlotWidget, region_item: pg.LinearRegionItem,
    ) -> None:
        """Keep the selection region's borders inside the plot's X
        range. Pans the view to follow the region when it has moved
        outside the current window, expands when the region is wider
        than the current view. Padding leaves a small gap so the
        edges aren't flush with the plot boundary.

        Called both on direct profile-region drags (manual peak edge
        edits) and on programmatic syncs from the 2D ROI (manual /
        detected peak moves) — the latter is what was previously
        leaving the box stranded off-screen.
        """
        lo, hi = region_item.getRegion()
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        span = max(abs(hi - lo), 1e-9)
        pad = 0.5 * span
        target_lo, target_hi = lo - pad, hi + pad
        vb = plot.getViewBox()
        cur_lo, cur_hi = vb.viewRange()[0]
        cur_lo, cur_hi = float(cur_lo), float(cur_hi)
        cur_width = cur_hi - cur_lo
        target_width = target_hi - target_lo

        if target_width > cur_width:
            # Region is wider than the visible window — expand to fit
            # without losing the existing centre point.
            new_lo = min(cur_lo, target_lo)
            new_hi = max(cur_hi, target_hi)
        elif target_lo < cur_lo:
            # Region's left edge fell off-screen → slide the view left
            # by exactly the overshoot, keeping the original width.
            shift = cur_lo - target_lo
            new_lo, new_hi = cur_lo - shift, cur_hi - shift
        elif target_hi > cur_hi:
            # Region's right edge fell off-screen → slide the view right.
            shift = target_hi - cur_hi
            new_lo, new_hi = cur_lo + shift, cur_hi + shift
        else:
            return  # already fully visible
        plot.setXRange(new_lo, new_hi, padding=0)


def _radial_fit_range(peak: SelectedPeak) -> tuple[float, float] | None:
    """Radial Gaussian fit interval for ``peak``.

    For manual / detected peaks the interval is the box itself (the user
    controls the box bounds, so we fit over what they drew). For
    fitted / matched peaks nothing is fitted: the stored
    ``radius_width`` is pygidfit's ``2σ`` (mlgidbase pass-through,
    mirrored by ``manual_fit.fit_one_peak``) and the interval,
    ``FITTED_FIT_REGION_FACTOR × radius_width`` around the centre, only
    sizes the span the stored curve is rendered over. Returns ``None``
    only when widths are non-finite or zero (caller hides the curve in
    that case).
    """
    r = peak.radius
    dr = peak.radius_width
    if not (np.isfinite(r) and np.isfinite(dr) and dr > 0):
        return None
    if peak.kind in ("manual", "detected"):
        half = dr / 2.0
    else:  # fitted / matched
        # Symmetric with _angular_fit_range — both axes use 2σ as
        # the stored width.
        half = FITTED_FIT_REGION_FACTOR * dr / 2.0
    return (r - half, r + half)


def _angular_fit_range(peak: SelectedPeak) -> tuple[float, float] | None:
    """Angular Gaussian fit interval for ``peak``. None for ring peaks
    (angle_width inf or non-finite — angular slice is the whole axis).
    """
    a = peak.angle
    da = peak.angle_width
    if peak.is_ring or not (np.isfinite(a) and np.isfinite(da) and da > 0):
        return None
    if peak.kind in ("manual", "detected"):
        half = da / 2.0
    else:
        # Both pipeline-fitted and Add-to-fitted-saved rows store
        # ``angle_width`` as pygidfit's ``2σ`` (per the wrapper in
        # ``manual_fit.fit_one_peak`` and mlgidbase's pass-through
        # of the same container — they share an identical write
        # convention). Same render-span interval as the radial helper;
        # the stored curve is drawn, not fitted, over it.
        half = FITTED_FIT_REGION_FACTOR * da / 2.0
    return (a - half, a + half)


def _hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _bounds_to_slice(axis: np.ndarray, lo: float, hi: float) -> slice:
    """Convert a (lo, hi) range in physical units to an index slice into a
    monotonic 1D axis. Tolerates non-finite bounds (treated as full range).
    """
    if not np.isfinite(lo):
        lo_idx = 0
    else:
        lo_idx = int(np.searchsorted(axis, lo, side="left"))
    if not np.isfinite(hi):
        hi_idx = len(axis)
    else:
        hi_idx = int(np.searchsorted(axis, hi, side="right"))
    lo_idx = max(0, min(lo_idx, len(axis)))
    hi_idx = max(lo_idx, min(hi_idx, len(axis)))
    return slice(lo_idx, hi_idx)
