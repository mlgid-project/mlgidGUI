"""Shared fixtures for the mlgidLAB smoke harness.

The environment is locked down *before* any Qt or h5py import so the
suite is headless and hermetic:

* ``QT_QPA_PLATFORM=offscreen`` — render with no display, no xvfb.
* ``XDG_CONFIG_HOME`` redirected to a temp dir — QSettings (recent
  files, theme, playback) cannot read or clobber the real user config,
  so construction starts from a clean slate every run.
* ``MLGIDLAB_LOG_DIR`` redirected into the same temp dir, so the update
  log is never written to the real per-user log directory.
* ``HDF5_USE_FILE_LOCKING=FALSE`` — mirrors ``mlgidlab.__init__.main``;
  matters once the file-open increments land, set here for parity.

These run at conftest import, which pytest executes before collecting
tests and before the pytest-qt ``qapp`` fixture constructs the
QApplication, so the offscreen platform is in place in time.
"""

from __future__ import annotations

import os
import tempfile

# --- environment lockdown (must precede Qt / h5py import) -------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# A per-process temp config root so test runs never share QSettings state.
# Removed in pytest_unconfigure: that hook ends in os._exit (skipping atexit),
# so without an explicit rmtree these would pile up in /tmp run after run.
_CONFIG_ROOT = tempfile.mkdtemp(prefix="mlgidlab-test-config-")
os.environ["XDG_CONFIG_HOME"] = _CONFIG_ROOT
# Same for the crash / update logs (mlgidlab.diagnostics): never the real
# per-user log directory.
os.environ["MLGIDLAB_LOG_DIR"] = os.path.join(_CONFIG_ROOT, "logs")

import gc  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

# --- disable the cyclic garbage collector for the whole session ------
# A cyclic-GC pass that fires *while* PySide6/shiboken is mid-way
# through constructing a Qt object reenters that half-built C++ wrapper
# and SIGSEGVs. We observed it on CI (PySide6 6.11.1, Python 3.13)
# crashing inside MainWindow() -> pyqtgraph ImageView -> ROI
# construction, with the faulting thread reported as "Garbage-
# collecting"; it does not reproduce on the dev box (PySide6 6.11.0,
# Python 3.12), so it is a teardown-order / GC-timing crash, not a
# logic bug. The ``_PIN`` shim below keeps *test-owned* widgets alive,
# but the crash is during construction — before the widget ever reaches
# qtbot — so reference-pinning can't reach it. Disabling automatic
# collection removes the only thing that can run mid-construction;
# refcounting still frees objects promptly, and the os._exit in
# pytest_unconfigure means leaked reference cycles never matter for this
# short-lived process. No test depends on gc (verified by grep).
gc.disable()

# --- QSettings hermeticity, all platforms ----------------------------
# The XDG_CONFIG_HOME redirect above only reaches QSettings' INI backend,
# i.e. Linux. On Windows the default backend is the REGISTRY: tests would
# touch real machine state, and with no organization name set (only
# mlgidlab.main() sets one, tests never call it) registry writes don't
# round-trip at all — setValue() followed by value() returned None on the
# windows-latest runner. Forcing the INI format into the per-run config
# root gives every platform the exact hermetic behaviour Linux already
# had. Importing QtCore here is fine: the environment lockdown above has
# already run, and pytest-qt (imported just below) pulls in Qt anyway.
from PySide6.QtCore import QSettings  # noqa: E402

QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(
    QSettings.Format.IniFormat, QSettings.Scope.UserScope, _CONFIG_ROOT
)

import pytestqt.qtbot as _qtbot_mod  # noqa: E402
_PIN=[]
_oaw=_qtbot_mod._add_widget
def _paw(item,widget,**kw):
    _PIN.append(widget); _oaw(item,widget,**kw)
_qtbot_mod._add_widget=_paw

# Real pytest exit status, captured at session end and used by
# pytest_unconfigure to hard-exit before the crashy native teardown.
_PYTEST_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _PYTEST_EXIT_STATUS
    _PYTEST_EXIT_STATUS = int(exitstatus)


def pytest_unconfigure(config):
    """Skip the crashy native interpreter teardown.

    On headless CI the PySide6 + silx(OpenGL) + h5py C++ stack tears
    down its static singletons in an order that SIGSEGVs at interpreter
    exit *after* a clean run: pytest prints ``N passed`` and returns 0,
    then the process dies with exit code 139. It reproduces on every
    Python (3.11-3.14) and never locally (a real GL/display masks it),
    so it is not a test failure and faulthandler cannot catch it (it is
    gone by then).

    ``pytest_unconfigure`` is the final hook, run *after* the terminal
    reporter has printed the summary, so no output is lost. We flush and
    ``os._exit`` with pytest's real status (captured in
    ``pytest_sessionfinish``): a genuine failure still exits non-zero
    (CI stays honest), only the post-success native teardown is
    bypassed. ``os._exit`` skips Python atexit/cleanup, which is safe
    for a short-lived test process (no coverage plugin in the dev
    deps).
    """
    # os._exit below skips atexit, so reclaim temp artifacts explicitly here
    # (the final hook): the per-run config root and any session working-copy
    # dirs a test left open. Otherwise they accumulate in /tmp across runs.
    import shutil
    shutil.rmtree(_CONFIG_ROOT, ignore_errors=True)
    try:
        from mlgidlab import session
        session.cleanup_registered_temp_dirs()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        # os._exit is NOT immediate on Windows: it reaches ExitProcess,
        # which still runs every loaded DLL's process-detach handler --
        # and the Qt/silx native teardown dies there with an access
        # violation that clobbers the exit status (observed on
        # windows-latest: "50 passed", then faulthandler's "Windows
        # fatal exception: access violation" pointing at this hook, then
        # exit code 1). TerminateProcess skips DLL detach entirely and
        # preserves the real pytest status. The argtypes declaration is
        # load-bearing: the current-process pseudo-handle is -1, and
        # ctypes' default int conversion truncates it to a 32-bit
        # 0xFFFFFFFF on x64, making the call fail silently and fall
        # through to the crashy os._exit below.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(
            ctypes.c_void_p(-1), _PYTEST_EXIT_STATUS
        )
    os._exit(_PYTEST_EXIT_STATUS)


# The peak structured dtype, kept as a plain list of (name, fmt) tuples
# so numpy stays out of module scope (the env lockdown above must apply
# before any numpy/h5py import). Field set + order is the ground truth
# from mlgidLAB itself, NOT pygid (absent from this checkout): it mirrors
# the ``fields`` dict written by ``add_fitted_peak_row``
# (file_model.py:840-858) and the names read by
# ``PeakTable.from_dataset`` (file_model.py:93-104).
PYGID_PEAK_DTYPE = [
    ("amplitude", "f4"),
    ("angle", "f4"),
    ("angle_width", "f4"),
    ("radius", "f4"),
    ("radius_width", "f4"),
    ("q_z", "f4"),
    ("q_xy", "f4"),
    ("theta", "f4"),
    ("score", "f4"),
    ("A", "f4"),
    ("B", "f4"),
    ("C", "f4"),
    ("is_ring", "bool"),
    ("is_cut_qz", "bool"),
    ("is_cut_qxy", "bool"),
    ("visibility", "i4"),
    ("id", "i4"),
]


@pytest.fixture
def main_window(qtbot):
    """Construct a fresh ``MainWindow`` and guarantee teardown.

    Imported lazily inside the fixture so the environment lockdown
    above is fully applied before ``mlgidlab`` (and its Qt/h5py
    imports) is touched. ``qtbot`` ensures a QApplication exists.

    Teardown runs even if a test fails: ``close()`` exercises the real
    ``closeEvent`` shutdown path (silx detach, worker quit). With no
    session loaded it must not raise or prompt.
    """
    from mlgidlab.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    try:
        yield window
    finally:
        try:
            window.close()
        except RuntimeError as exc:
            # PySide6 can raise "Internal C++ object (<widget>) already
            # deleted" while closeEvent tears down pyqtgraph widgets
            # (viewer / profile-viewer clear()) — the C++ half of a
            # child is freed before a Python-side close handler touches
            # it. With gc.disable() (see top of file) the dead wrapper
            # lingers instead of being collected, so the access raises
            # instead of being skipped. It is timing-dependent shutdown
            # noise in this many-windows-per-process test run (observed
            # on CI py3.12 + PySide6 6.11.1, never locally or on 3.14),
            # not a product fault: the real app closes once, with a
            # display, and the process exits immediately after. Swallow
            # only that specific message; re-raise anything else so a
            # genuine teardown regression still fails the suite.
            if "already deleted" not in str(exc):
                raise


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Destroy the widgets pytest-qt closed — really destroy them.

    pytest-qt's teardown closes every ``qtbot.addWidget`` widget,
    calls ``deleteLater()`` and then ``processEvents()`` — but Qt
    never delivers ``DeferredDelete`` events through
    ``processEvents``; they need a running event loop or an explicit
    ``sendPostedEvents(None, DeferredDelete)``. In an exec()-less
    pytest process every closed window's C++ widget tree therefore
    stayed alive until interpreter exit: ~1,400 widgets per
    ``MainWindow`` test, tens of thousands over one serial CI shard.
    That accumulation pushed CI shard 0 past its 30-minute timeout
    the moment ``test_theme_persistence`` landed late in the shard
    (adding a test FILE re-deals the round-robin shards):
    ``_set_theme`` swaps the app stylesheet and repolishes
    ``allWidgets()`` — three theme switches in that test — and both
    costs scale with every widget the process ever leaked.

    ``trylast``: fixture finalizers (e.g. ``main_window``'s second,
    defensive ``close()``) must run BEFORE the flush so they never
    touch an already-destroyed C++ object.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    app = QCoreApplication.instance()
    if app is not None:
        QCoreApplication.sendPostedEvents(
            None, QEvent.Type.DeferredDelete
        )


@pytest.fixture(autouse=True)
def _no_real_file_dialog():
    """Turn a file dialog that actually opens into a failure, not a hang.

    Every picker in the app is built by ``mlgidlab.file_dialogs``, so a
    test that exercises a Browse button fakes it there -- or patches
    ``QFileDialog.exec`` itself, which lands on top of this and wins.
    Anything else reaches a modal ``exec()`` with nobody to answer it
    and blocks the run forever; a loud error names the test instead.

    Patches by hand rather than through ``monkeypatch`` ON PURPOSE.
    An autouse fixture is set up before the ones a test asks for, so
    requesting ``monkeypatch`` here would make it the *first* fixture
    created and therefore the *last* undone -- after the ``main_window``
    finalizer instead of before it. Tests that patch a method
    ``closeEvent`` calls (``test_reopen_keeps_dirty_old_session_on_cancel``
    patches ``_confirm_discard_changes``) rely on the opposite order:
    with the patch still live, the close is refused, the window is torn
    down with its worker threads running, and the deferred-delete flush
    below hangs the run.
    """
    from PySide6.QtWidgets import QFileDialog

    def _refuse(self):
        raise RuntimeError(
            "a real QFileDialog.exec() was reached: fake "
            "mlgidlab.file_dialogs (or patch QFileDialog.exec) in this "
            "test -- see tests/test_file_dialogs.py"
        )

    # ``exec`` is inherited from QDialog, so restoring means removing
    # the override, not writing the parent's method back onto the class.
    QFileDialog.exec = _refuse
    try:
        yield
    finally:
        try:
            del QFileDialog.exec
        except AttributeError:
            pass


@pytest.fixture(autouse=True)
def _cleanup_session_temp_dirs():
    """Remove any NexusSession working-copy temp dir a test left open.

    Many tests build ``NexusSession.open(...)`` directly without ``close()``;
    each otherwise leaks an ``mlgidlab_*`` dir under the system temp dir
    (atexit can't help -- ``pytest_unconfigure`` exits via ``os._exit``)."""
    yield
    try:
        from mlgidlab import session
        session.cleanup_registered_temp_dirs()
    except Exception:
        pass


@pytest.fixture
def clean_matched_colors():
    """Drop the persisted custom pens around a test.

    ``GIWAXSImageViewer`` loads the ``matchedPens`` and ``overlayPens``
    QSettings keys at construction, so a test that picks a colour, line
    style or width would otherwise leak it into every later
    viewer/MainWindow built in the same run. ``matchedColors`` is the
    pre-pen key, still read as a fallback, so it has to go too.
    """
    from PySide6.QtCore import QSettings

    keys = ("matchedPens", "matchedColors", "overlayPens")

    def drop() -> None:
        settings = QSettings()
        for key in keys:
            settings.remove(key)

    drop()
    yield
    drop()


@pytest.fixture
def peak_link_off():
    """Turn the fitted/detected link off for one test.

    The link ships **on**, so with no fixture a fitted write takes its
    detected peak's id and replaces that peak's previous fit. Tests that
    predate the setting assert the unlinked behaviour — two independent
    tables, every fit appended under a fresh id — which is exactly what
    the setting promises when it is off. They take this fixture; the
    linked behaviour is covered in ``test_peak_linking.py``.
    """
    from PySide6.QtCore import QSettings
    from mlgidlab.main_window_constants import PEAK_LINK_KEY

    s = QSettings()
    s.setValue(PEAK_LINK_KEY, False)
    s.sync()
    yield
    s.remove(PEAK_LINK_KEY)
    s.sync()


@pytest.fixture
def synthetic_nexus(tmp_path):
    """A minimal valid NeXus file: 1 entry, 3 frames, no peaks.

    Matches exactly what the read path requires, verified against
    source:

    * ``file_model`` constants ``IMG_REL='data/img_gid_q'``,
      ``QXY_REL='data/q_xy'``, ``QZ_REL='data/q_z'``
      (``file_model.py:42-44``).
    * ``list_entries`` keeps an entry only if its ``data`` group has a
      ``signal`` attr equal to ``"img_gid_q"`` (``file_model.py:132``)
      and the group name passes ``is_entry_group_name``
      (``entry``/``entry_0000``/``entry_horiz`` all valid).

    The ``analysis`` group is intentionally omitted: it is written by
    ``normalize_for_pygid`` on the file-open *worker* path, which the
    tests bypass by calling ``_set_active_session`` directly. Peaks
    are optional and their absence is handled gracefully, so the
    viewer just shows empty overlays.

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    return _write_synthetic_nexus(tmp_path / "synthetic.h5", n_frames=3)


@pytest.fixture
def synthetic_nexus_single(tmp_path):
    """``synthetic_nexus`` with a single frame: nothing to track across."""
    return _write_synthetic_nexus(tmp_path / "synthetic_single.h5", n_frames=1)


def _write_synthetic_nexus(path, n_frames):
    import h5py
    import numpy as np

    n_qz, n_qxy = 16, 24
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data")
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=rng.random((n_frames, n_qz, n_qxy), dtype=np.float32),
        )
        data.create_dataset(
            "q_xy", data=np.linspace(-1.0, 3.0, n_qxy, dtype=np.float32)
        )
        data.create_dataset(
            "q_z", data=np.linspace(0.0, 4.0, n_qz, dtype=np.float32)
        )
    return path


@pytest.fixture
def synthetic_nexus_with_peaks(tmp_path):
    """A valid NeXus file (1 entry, 3 frames) plus a populated analysis
    tree on ``frame00000`` only.

    Self-contained — does not chain off ``synthetic_nexus`` — so the
    analysis groups can be written in the same ``h5py.File`` open. The
    layout matches what the read path expects, verified against source:

    * group path ``entry_0000/data/analysis/frame00000/<kind>_peaks``
      (``ANALYSIS_REL`` / ``FRAME_KEY_FMT``, file_model.py:21,44).
    * the structured dtype is ``PYGID_PEAK_DTYPE`` (see its comment).
    * frames 1 & 2 get *no* analysis group, so ``load_peaks`` returns
      ``{detected:None, fitted:None}`` (``file_model.read_peaks``).
      Adding a peak to them still works — ``ensure_peak_dataset``
      creates the group and an empty table on demand, which is what
      lets a frame be labelled before any pipeline stage has run
      (see ``tests/test_peak_dataset_bootstrap.py``).

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic_peaks.h5"
    n_frames, n_qz, n_qxy = 3, 16, 24
    dt = np.dtype(PYGID_PEAK_DTYPE)
    rng = np.random.default_rng(0)

    detected = np.zeros(3, dtype=dt)
    detected["id"] = [0, 1, 2]
    detected["score"] = [0.40, 0.75, 1.00]
    detected["amplitude"] = [10.0, 20.0, 30.0]
    detected["angle"] = [10.0, 45.0, 80.0]
    detected["radius"] = [1.0, 2.0, 3.0]
    detected["angle_width"] = [5.0, 5.0, 5.0]
    detected["radius_width"] = [0.2, 0.2, 0.2]
    detected["q_xy"] = detected["radius"] * np.cos(np.deg2rad(detected["angle"]))
    detected["q_z"] = detected["radius"] * np.sin(np.deg2rad(detected["angle"]))

    fitted = np.zeros(2, dtype=dt)
    fitted["id"] = [0, 1]
    fitted["score"] = [0.50, 0.90]
    fitted["amplitude"] = [12.0, 22.0]
    fitted["angle"] = [20.0, 60.0]
    fitted["radius"] = [1.5, 2.5]
    fitted["angle_width"] = [4.0, 4.0]
    fitted["radius_width"] = [0.3, 0.3]
    fitted["q_xy"] = fitted["radius"] * np.cos(np.deg2rad(fitted["angle"]))
    fitted["q_z"] = fitted["radius"] * np.sin(np.deg2rad(fitted["angle"]))

    fitted_errors = np.zeros(0, dtype=dt)

    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=rng.random((n_frames, n_qz, n_qxy), dtype=np.float32),
        )
        data.create_dataset(
            "q_xy", data=np.linspace(-1.0, 3.0, n_qxy, dtype=np.float32)
        )
        data.create_dataset(
            "q_z", data=np.linspace(0.0, 4.0, n_qz, dtype=np.float32)
        )
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("detected_peaks", data=detected)
        g.create_dataset("fitted_peaks", data=fitted)
        g.create_dataset("fitted_peaks_errors", data=fitted_errors)
    return path


@pytest.fixture
def synthetic_fitted_scan(tmp_path):
    """A pygid-valid NeXus scan with fitted peaks on every frame.

    6 frames; one persistent fitted peak drifting slowly in radius
    (frames 0-5, id 0 each) plus one single-frame blip on frame 2
    (id 1). Built for the phase-tracking tests: the ``axes`` attr on
    the data group is REQUIRED — ``pygid.NexusFile.get_entry_type``
    rejects entries without it (nexus_reader.py:526), and mlgidBASE's
    ``track_peaks`` opens the file through pygid. The blip forms a
    1-member IoU component, so upstream's ``length`` cut (strictly
    greater) drops it for any ``length >= 1``.

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic_fitted_scan.h5"
    n_frames, n_qz, n_qxy = 6, 16, 24
    dt = np.dtype(PYGID_PEAK_DTYPE)
    rng = np.random.default_rng(0)

    def _rows(specs):
        arr = np.zeros(len(specs), dtype=dt)
        for i, (radius, angle, amp) in enumerate(specs):
            arr["id"][i] = i
            arr["score"][i] = 0.9
            arr["amplitude"][i] = amp
            arr["angle"][i] = angle
            arr["radius"][i] = radius
            arr["angle_width"][i] = 5.0
            arr["radius_width"][i] = 0.2
            arr["q_xy"][i] = radius * np.cos(np.deg2rad(angle))
            arr["q_z"][i] = radius * np.sin(np.deg2rad(angle))
        return arr

    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.attrs["axes"] = ["frame_num", "q_z", "q_xy"]
        data.create_dataset(
            "img_gid_q",
            data=rng.random((n_frames, n_qz, n_qxy), dtype=np.float32),
        )
        data.create_dataset(
            "q_xy", data=np.linspace(-1.0, 3.0, n_qxy, dtype=np.float32)
        )
        data.create_dataset(
            "q_z", data=np.linspace(0.0, 4.0, n_qz, dtype=np.float32)
        )
        for frame in range(n_frames):
            specs = [(1.0 + 0.002 * frame, 45.0, 100.0 + frame)]
            if frame == 2:
                specs.append((2.5, 10.0, 50.0))
            g = data.create_group(
                f"analysis/frame{frame:05d}", track_order=True
            )
            g.create_dataset("fitted_peaks", data=_rows(specs))
            g.create_dataset(
                "fitted_peaks_errors", data=np.zeros(0, dtype=dt)
            )
    return path


@pytest.fixture
def synthetic_raw(tmp_path):
    """A raw HDF5 file with one qualifying 3-D detector dataset.

    ``list_raw_entries`` keeps a dataset only if it is 3-D with both
    spatial dims ≥ ``RAW_MIN_DETECTOR_HW`` (==32) and a numeric dtype
    (file_model.py:595,634-640). This file deliberately includes two
    datasets that must be *filtered out* so the size / ndim guards are
    exercised, and carries no ``entry_*`` groups so the open flow
    (``CopyWorker``'s inline classification) reads it as ``raw``
    (not ``nexus``).

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic_raw.h5"
    rng = np.random.default_rng(1)
    with h5py.File(path, "w", track_order=True) as f:
        f.create_dataset(
            "raw/data0/image",
            data=rng.integers(0, 1000, size=(4, 64, 64), dtype=np.uint32),
        )
        # Filtered: spatial dims 16 < RAW_MIN_DETECTOR_HW (32).
        f.create_dataset(
            "raw/small",
            data=rng.integers(0, 100, size=(2, 16, 16), dtype=np.uint16),
        )
        # Filtered: ndim != 3.
        f.create_dataset(
            "raw/flat",
            data=rng.integers(0, 100, size=(64, 64), dtype=np.uint16),
        )
    return path
