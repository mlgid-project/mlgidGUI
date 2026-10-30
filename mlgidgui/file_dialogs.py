"""One browsing directory shared by every file dialog in the app.

Alpha testers hit this constantly: opening a scan, then picking a
mask, then saving a PONI walked through three unrelated starting
directories. Each call site passed its own ``dir`` argument, and the
most common one was a bare suggested filename -- which Qt resolves
against the *process working directory*, i.e. ``/`` under a desktop
launcher.

Every picker in mlgidGUI now goes through this module, and the rule
is one sentence: **a dialog opens in the directory the user last
browsed, and a caller may only suggest a file name.** Suggested
paths are reduced to their basename on purpose (see ``start_at``) --
honouring one caller's absolute path is exactly how the starting
directory used to jump around. The directory is written back to
``QSettings`` on every accept, so it survives a restart as well as
the session.

Two things live outside this module and are wired to it:

- pyFAI's calibration window builds its own dialogs (the "Save as
  PONI" button among them), so they are adopted through
  ``CalibrationContext.createFileDialog`` -- see
  ``calibration_dialog._patch_pyfai_dialogs``, which calls ``adopt``.
- The File-browser dock's own Open path is ``_pick_files``, which
  uses ``open_files`` here.

All dialogs are non-native, which used to be true of the Open dialog
only. The reason generalised with this change: every picker now
starts in whatever directory the user last worked in, which on a
beamtime machine is routinely a folder of thousands of detector
images, and the platform-native picker thumbnails every one of them
before the listing becomes scrollable. ``_FastFileIconProvider``
keeps it constant-time.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QFileDialog
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

#: Where the shared directory is stored.
SETTINGS_KEY = "files/last_dir"

#: Read-only fallback: before this module existed the Open dialog kept
#: its own key. Honoured so an upgrading user does not lose the
#: directory they had, never written to again.
LEGACY_KEY = "open/last_dir"

_icon_provider = None


def icon_provider():
    """The shared constant-time icon provider (built on first use).

    Lazy because it needs a live ``QApplication`` for ``style()``, and
    this module is imported at panel-import time.
    """
    global _icon_provider
    if _icon_provider is None:
        from mlgidgui.browser_widgets import _FastFileIconProvider

        _icon_provider = _FastFileIconProvider()
    return _icon_provider


def last_dir() -> str:
    """The directory the user browsed last; home if there isn't one.

    A stored directory that no longer exists (removed USB disk, a
    beamtime mount that is gone) is skipped rather than handed to Qt,
    which would silently fall back to the working directory.
    """
    settings = QSettings()
    for key in (SETTINGS_KEY, LEGACY_KEY):
        value = str(settings.value(key, "") or "")
        if value and Path(value).is_dir():
            return value
    return str(Path.home())


def remember(path) -> None:
    """Record where the user just browsed. Takes a file or a directory."""
    if not path:
        return
    candidate = Path(str(path))
    directory = candidate if candidate.is_dir() else candidate.parent
    if not directory.is_dir():
        return
    QSettings().setValue(SETTINGS_KEY, str(directory))


def start_at(suggested_name: str = "") -> str:
    """The ``dir`` argument for a dialog: the shared directory, plus a name.

    Only the *name* part of ``suggested_name`` is used. Callers that
    hold a full path (a session's original file, the current text of a
    path field) pass it as-is and get its basename placed in the
    last-browsed directory -- that is what keeps every button
    consistent with every other one.
    """
    base = last_dir()
    name = Path(str(suggested_name)).name if suggested_name else ""
    return str(Path(base) / name) if name else base


# --- The four pickers -------------------------------------------------


def _dialog(parent, title: str, name_filter: str, suggested_name: str):
    dlg = QFileDialog(parent, title)
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dlg.setIconProvider(icon_provider())
    if name_filter:
        dlg.setNameFilter(name_filter)
    dlg.setDirectory(last_dir())
    name = Path(str(suggested_name)).name if suggested_name else ""
    if name:
        dlg.selectFile(name)
    return dlg


def _run(dlg) -> list[str]:
    """Execute, remember where we ended up, return the picked paths."""
    if not dlg.exec():
        return []
    picked = [p for p in dlg.selectedFiles() if p]
    if picked:
        remember(picked[0])
    return picked


def open_file(
    parent, title: str, name_filter: str = "", suggested_name: str = "",
) -> str:
    """Pick one existing file. Returns "" if the user cancelled."""
    dlg = _dialog(parent, title, name_filter, suggested_name)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    picked = _run(dlg)
    return picked[0] if picked else ""


def open_files(parent, title: str, name_filter: str = "") -> list[str]:
    """Pick any number of existing files."""
    dlg = _dialog(parent, title, name_filter, "")
    dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
    return _run(dlg)


def save_file(
    parent,
    title: str,
    name_filter: str = "",
    suggested_name: str = "",
    default_suffix: str = "",
) -> tuple[str, str]:
    """Pick a save destination.

    Returns ``(path, selected_filter)`` to match
    ``QFileDialog.getSaveFileName``, because several callers pick the
    output format from the chosen filter.
    """
    dlg = _dialog(parent, title, name_filter, suggested_name)
    dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    if default_suffix:
        dlg.setDefaultSuffix(default_suffix)
    picked = _run(dlg)
    return (picked[0] if picked else "", dlg.selectedNameFilter())


def existing_directory(parent, title: str) -> str:
    """Pick a directory. Returns "" if the user cancelled."""
    dlg = _dialog(parent, title, "", "")
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    picked = _run(dlg)
    return picked[0] if picked else ""


# --- Dialogs we did not build ----------------------------------------


def adopt(dialog) -> None:
    """Make a foreign dialog follow (and feed) the shared directory.

    Used for pyFAI's calibration dialogs, which are constructed inside
    pyFAI and start at the process working directory. Duck-typed on
    purpose: ``createImageFileDialog`` hands back silx's
    ``ImageFileDialog``, which is not a ``QFileDialog`` but exposes
    the same ``setDirectory`` / ``directory`` pair. Every step is
    guarded -- a third-party dialog that does not cooperate must still
    open.
    """
    try:
        dialog.setDirectory(last_dir())
    except Exception:
        logger.debug("could not seed the dialog's directory", exc_info=True)
    if isinstance(dialog, QFileDialog):
        try:
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            dialog.setIconProvider(icon_provider())
        except Exception:
            logger.debug("could not de-nativise the dialog", exc_info=True)
    try:
        dialog.finished.connect(
            lambda code, d=dialog: _remember_from(d, code)
        )
    except Exception:
        logger.debug("could not track the dialog's directory", exc_info=True)


def _remember_from(dialog, code) -> None:
    if int(code) != int(QDialog.DialogCode.Accepted):
        return
    for attr in ("selectedFiles", "selectedFile"):
        getter = getattr(dialog, attr, None)
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:
            continue
        picked = value[0] if isinstance(value, (list, tuple)) and value else value
        if picked:
            remember(picked)
            return
    try:
        directory = dialog.directory()
    except Exception:
        logger.debug("could not read the dialog's directory", exc_info=True)
        return
    remember(
        directory.absolutePath()
        if hasattr(directory, "absolutePath")
        else directory
    )
