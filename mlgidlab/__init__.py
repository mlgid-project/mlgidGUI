from __future__ import annotations

import logging
import os
import sys

# Disable HDF5's SWMR file locking. silx's Hdf5TreeModel opens each
# loaded file as ``r`` and pygid (via mlgidbase) reopens the same
# path as ``r+`` for the Figure Export window's renderer; with
# default locking on, the second open fails with
# "file is already open for read-only". Setting this before any
# ``h5py`` import means neither side acquires the OS-level lock that
# would block the other. Must be set before the first ``h5py``
# import — keep it at the very top of the package init.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from PySide6.QtWidgets import QApplication

__version__ = "0.1.0a22"


def _enable_crash_log() -> None:
    """Dump the Python stack of every thread on a hard crash.

    A segfault out of Qt, h5py or a GPU driver takes the window with it
    and leaves nothing behind — which is the difference between a bug
    that can be fixed and one that can only be waited for. ``faulthandler``
    costs nothing until the process dies and then names the exact frame.

    The dump goes to a file rather than stderr because the app is
    launched from a desktop entry (and, on Windows, from ``pythonw``)
    more often than from a terminal, and in both cases stderr goes
    nowhere. The file is kept open for the life of the process on
    purpose: the handler runs in a broken process and cannot open one.

    It lives in the per-user log directory (``diagnostics.log_dir``) so a
    reboot does not wipe it, with the system temp dir as the fallback.
    """
    try:
        import datetime
        import faulthandler

        from mlgidlab import diagnostics

        opened = diagnostics.open_crash_log()
        if opened is None:
            return
        # Deliberately not closed, and deliberately not a context
        # manager: the file has to still be open when the process dies.
        stream, _path = opened
        started = datetime.datetime.now().isoformat(timespec="seconds")
        stream.write(
            f"--- mlgidLAB {__version__} started {started}, "
            f"pid {os.getpid()} ---\n")
        faulthandler.enable(file=stream, all_threads=True)
    except Exception:
        # A missing crash log must never be what stops the app starting.
        pass


def main() -> int:
    # Before anything imports and scans sys.path: a launcher exe on it would
    # be held open by importlib.metadata and could not be updated on Windows.
    from mlgidlab import update_check
    update_check.drop_launcher_from_sys_path()
    from PySide6.QtCore import QSettings, QTimer
    from pathlib import Path
    from mlgidlab import desktop_entry
    from mlgidlab.main_window import MainWindow
    from mlgidlab.theme import apply_dark_theme, apply_light_theme

    # Before the QApplication, so a crash inside Qt's own start-up is
    # caught too.
    _enable_crash_log()
    app = QApplication(sys.argv)
    # Reclaim working-copy temp dirs leaked by a previous run that was killed
    # before its graceful close ran (see mlgidlab.session). Only removes dirs
    # whose owning PID is dead, so a second running instance is left alone.
    try:
        import logging
        from mlgidlab import session as _session
        _n = _session.sweep_stale_temp_dirs()
        if _n:
            logging.getLogger("mlgidlab").info(
                "Reclaimed %d stale temp working-copy dir(s) from a prior run.", _n
            )
    except Exception:
        pass
    # Set both org + app names so QSettings has a stable key path on
    # every platform (used by the Recent files menu, may grow other
    # persisted preferences over time).
    app.setOrganizationName("mlgidLAB")
    app.setApplicationName("mlgidLAB")
    # The name the shell matches a window to an *application* by: Qt
    # advertises it as the Wayland ``app_id`` and the X11 ``WM_CLASS``
    # instance name. Both are stamped when a window is created and never
    # re-read, so this has to happen before ``MainWindow()`` below.
    # Without it there is nothing for GNOME to match a ``.desktop`` file
    # against, and the dash / switcher / top bar fall back to a generic
    # icon no matter what ``setWindowIcon`` says.
    app.setDesktopFileName(desktop_entry.DESKTOP_NAME)
    # Window, taskbar and Alt-Tab icon. Set on the application so every
    # window inherits it (main window, figure export, phase views, the
    # dialogs). Guarded: a packaging slip should cost an icon, not a
    # startup.
    try:
        from mlgidlab.icons import app_icon
        app.setWindowIcon(app_icon())
    except Exception:
        logging.getLogger("mlgidlab").debug(
            "could not set the application icon", exc_info=True)
    if desktop_entry.is_supported():
        # The Linux counterpart of the AppUserModelID below. GNOME takes
        # an app's icon from its ``.desktop`` file, and pip installs no
        # such file, so one is written to the user's XDG data dir on
        # first run (and refreshed if the environment moved). Guarded
        # for the same reason as the icon above: a read-only $HOME
        # should cost the dash icon, not the startup.
        try:
            desktop_entry.install()
        except Exception:
            logging.getLogger("mlgidlab").debug(
                "could not install the desktop entry", exc_info=True)
    if sys.platform == "win32":
        # Without an explicit AppUserModelID, Windows groups the app
        # under python.exe's icon in the taskbar no matter what
        # setWindowIcon says.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "mlgidLAB.mlgidLAB")
        except Exception:
            logging.getLogger("mlgidlab").debug(
                "could not set the Windows AppUserModelID", exc_info=True)
    # Honor the persisted theme choice (View → Theme). Defaults to
    # dark; the menu sync inside MainWindow reads the same key.
    theme = str(QSettings().value("theme", "dark")).lower()
    if theme == "light":
        apply_light_theme(app)
    else:
        apply_dark_theme(app)
    window = MainWindow()
    # Tell MainWindow which theme is now live so its View → Theme
    # menu shows the right checked entry on first open.
    window._current_theme = theme if theme in ("dark", "light") else "dark"
    # Re-sync menu checkmark (the menu was built before _current_theme
    # was set on this path; the default check goes to Dark, so update
    # if necessary).
    if window._current_theme == "light":
        window.action_theme_light.setChecked(True)
    else:
        window.action_theme_dark.setChecked(True)
    # The welcome page was seeded during __init__, before the theme above
    # was known; re-seed so a light-theme user gets the light wordmark.
    window._refresh_welcome_view()
    window.show()
    # Post-update changelog (offline) + async "newer release available"
    # check. Deferred into the event loop so it runs after the window is up
    # and never blocks show(). Not started from MainWindow.__init__ so the
    # headless test fixture never performs a network call.
    QTimer.singleShot(0, window._run_startup_update_checks)
    # argv[0] is the program; treat any extra args as files to open.
    paths = [Path(a) for a in sys.argv[1:] if Path(a).exists()]
    if paths:
        # Defer into the event loop: _open_paths spawns the async
        # CopyWorker and touches widgets, which is only safe once
        # exec() is running. A 0 ms single-shot timer runs the
        # callback on the first loop iteration, after exec() starts.
        QTimer.singleShot(0, lambda: window._open_paths(paths))
    return app.exec()
