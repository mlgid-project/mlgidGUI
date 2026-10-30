"""App-level constants shared by the main window and its dialogs."""
from __future__ import annotations



APP_NAME = "mlgidGUI"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"
# Open dialog auto-classifies NeXus vs raw HDF5 vs fabio image (TIFF/CBF/EDF);
# one combined filter does for all, with per-family filters below it.
OPEN_FILTER = (
    "All supported (*.h5 *.hdf5 *.nxs *.tif *.tiff *.cbf *.edf);;"
    "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;"
    "Detector images (*.tif *.tiff *.cbf *.edf);;"
    "All files (*)"
)

PLAYBACK_MODE_FRAME = "frame_interval_ms"
PLAYBACK_MODE_TOTAL = "total_time_s"
DEFAULT_PLAYBACK_FRAME_MS = 50          # 20 fps — was 100 ms / 10 fps
DEFAULT_PLAYBACK_TOTAL_S = 3.0
PLAYBACK_FRAME_MS_MIN = 10              # 100 fps requested ceiling
PLAYBACK_FRAME_MS_MAX = 2000            # 0.5 fps floor
PLAYBACK_TOTAL_S_MIN = 0.5
PLAYBACK_TOTAL_S_MAX = 600.0            # 10 minutes max

# Real tick cap. The eye stops perceiving extra frames much above
# ~20 fps, and large frames cannot be painted faster than ~50 ms
# regardless. When the user requests a faster per-frame rate (e.g.
# 3 s total over 300 frames = 10 ms / frame) we keep the timer at
# 50 ms and skip frames instead — see ``_compute_play_schedule``.
PLAYBACK_TICK_FLOOR_MS = 50

# QSettings keys for the playback preferences. Shared by MainWindow
# (class-attr aliases preserved there) and the Settings dialog.
PLAYBACK_MODE_KEY = "playbackMode"
PLAYBACK_FRAME_MS_KEY = "playbackFrameIntervalMs"
PLAYBACK_TOTAL_S_KEY = "playbackTotalTimeS"

# QSettings keys for the fitted/detected peak link. See
# ``mlgidgui.peak_link`` for what each one changes; the defaults are the
# shipped behaviour (linked, and a fitted delete leaves the detection).
# What a quick-select commit writes ("detected" | "fitted" | "both").
# The target is remembered; the mode itself is not (see
# ``ParameterPanel._restore_quick_target``).
QUICK_TARGET_KEY = "quickSelectTarget"

PEAK_LINK_KEY = "peakLinkFittedToDetected"
PEAK_LINK_REVERSE_KEY = "peakLinkFittedDeleteRemovesDetected"
DEFAULT_PEAK_LINK = True
DEFAULT_PEAK_LINK_REVERSE = False

# Mask the other peaks of the frame out of the selected peak's profiles
# (``mlgidgui.profile_masking``). Read when the profiles are redrawn.
PROFILE_MASK_NEIGHBOURS_KEY = "profile/mask_neighbours"
DEFAULT_PROFILE_MASK_NEIGHBOURS = False

# Zoom the image onto a detected / fitted peak when it is selected
# (``GIWAXSImageViewer.zoom_to_peak``). Rings are never zoomed.
ZOOM_TO_SELECTED_PEAK_KEY = "view/zoom_to_selected_peak"
DEFAULT_ZOOM_TO_SELECTED_PEAK = True

# Outline, in dashed red, the area pygidfit really fits around a selected
# detected peak (``manual_fit.fit_area``).
SHOW_FIT_AREA_KEY = "view/show_fit_area"
DEFAULT_SHOW_FIT_AREA = True
