"""Linux desktop integration: the ``.desktop`` entry and themed icons.

``QApplication.setWindowIcon`` is enough for the window's own titlebar,
and on X11 it also fills ``_NET_WM_ICON``. It is *not* enough for the
shell's app surfaces -- the dash, the window switcher, the top bar. GNOME
resolves a window to an **application** first and takes the icon from
that application's ``.desktop`` file; the window's own icon is only the
fallback for a window it could not match, and on Wayland there is no
fallback at all, because ``app_id`` is the only thing the compositor
gets.

mlgidGUI (then still called mlgidLAB) shipped no ``.desktop`` file, so
there was nothing to match.
Windows already had its counterpart of this problem solved (the explicit
AppUserModelID in ``mlgidgui.main``, without which the taskbar groups the
app under ``python.exe``); this is the Linux half.

Two pieces have to agree for the match to happen:

* ``QGuiApplication.setDesktopFileName(DESKTOP_NAME)`` -- Qt advertises
  this as the Wayland ``app_id`` and as the X11 ``WM_CLASS`` instance
  name. Called from ``mlgidgui.main`` before the first window exists,
  because both are set at window-creation time and are not re-read.
* A ``DESKTOP_NAME.desktop`` file on the XDG search path, whose ``Icon=``
  names an icon installed in the hicolor theme.

``StartupWMClass`` is written too. It is redundant while the two names
above agree, and it is what keeps the match working if they ever stop
agreeing (a renamed entry point, say).

Everything here is best-effort: a read-only or unusual ``$HOME`` should
cost the dash icon, never the startup. The caller logs and moves on.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

#: Basename of the ``.desktop`` file, the ``Icon=`` name, and the value
#: handed to ``setDesktopFileName``. These three MUST stay equal: the
#: whole mechanism is GNOME matching a window's app id to a file of this
#: name. Lowercase because it is a filename, not a display name -- the
#: display name is ``Name=`` below.
DESKTOP_NAME = "mlgidgui"

#: What the mlgidLAB alphas called all three. Their entry and icons are
#: removed on the first start of the renamed app (``remove_legacy``), so
#: the dash does not show two entries, one of them pointing at a launcher
#: that the update uninstalled.
LEGACY_DESKTOP_NAME = "mlgidlab"

#: Sizes installed into the hicolor theme. Mirrors the shipped PNGs in
#: ``assets/app``; the shell picks whichever fits the surface it is
#: drawing, so shipping the small end matters as much as the large.
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def is_supported() -> bool:
    """Whether desktop-entry installation applies to this platform.

    Linux only. macOS gets its icon from the app bundle and Windows from
    the AppUserModelID, so on both this would write a file nothing reads.
    """
    return sys.platform.startswith("linux")


def _data_home() -> Path:
    """``$XDG_DATA_HOME``, honouring the spec's default and ignoring a
    relative value (the spec says a relative path is invalid)."""
    raw = os.environ.get("XDG_DATA_HOME", "")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
    return Path.home() / ".local" / "share"


def _exec_command() -> str:
    """The ``Exec=`` line, as an absolute path wherever possible.

    The launcher runs from the session's environment, not from the shell
    the user starts mlgidGUI in, and the console script usually lives in
    a conda env or venv that the session's ``PATH`` never sees. A bare
    ``mlgidlab`` would therefore resolve for the person who typed it and
    fail for the same person clicking the dash icon.

    Preference order:

    1. the installed ``gui-script`` next to the running interpreter,
       which is the normal pip/conda layout;
    2. one found on ``PATH``;
    3. ``<python> -m mlgidgui``, which always works even from a source
       checkout with no script installed.
    """
    script = Path(sys.executable).parent / DESKTOP_NAME
    if script.is_file() and os.access(script, os.X_OK):
        return str(script)
    found = shutil.which(DESKTOP_NAME)
    if found:
        return found
    return f"{sys.executable} -m mlgidgui"


def desktop_file_path() -> Path:
    return _data_home() / "applications" / f"{DESKTOP_NAME}.desktop"


def icon_path(size: int) -> Path:
    return (
        _data_home() / "icons" / "hicolor" / f"{size}x{size}" / "apps"
        / f"{DESKTOP_NAME}.png"
    )


def desktop_file_contents() -> str:
    """The entry's text.

    ``Categories`` is ``Science;Physics;`` -- exactly one main category
    plus its subcategory, because ``desktop-file-validate`` warns that a
    second main category (``Education``, say) lists the app twice in a
    menu that still has categories. ``MimeType`` is left
    off deliberately: claiming ``application/x-hdf`` would put mlgidGUI
    in the "Open with" list for every HDF5 file on the machine, which is
    a bigger claim than a viewer for one flavour of NeXus should make.

    ``Exec`` ends in ``%F``, not ``%f``: ``main`` opens *every* extra
    argv entry that exists as a file, so the launcher may hand it a
    whole selection. With ``%f`` the shell passes one path per process
    and opening three files would start three copies of the app.
    """
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        "Name=mlgidGUI\n"
        "GenericName=GIWAXS analysis\n"
        "Comment=Detect, fit and match GIWAXS peaks in NeXus files\n"
        f"Exec={_exec_command()} %F\n"
        f"Icon={DESKTOP_NAME}\n"
        "Terminal=false\n"
        "Categories=Science;Physics;\n"
        "Keywords=GIWAXS;NeXus;HDF5;diffraction;scattering;\n"
        f"StartupWMClass={DESKTOP_NAME}\n"
    )


def _write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` unless it is already exactly that.

    Rewriting on every launch would be harmless but noisy (file watchers,
    backup tools, and a menu rebuild each time). Comparing first also
    means a moved environment DOES get a corrected ``Exec`` line, which a
    plain "skip if it exists" check would leave stale forever.
    """
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def remove_legacy() -> int:
    """Delete the entry and icons the mlgidLAB alphas installed.

    Only a ``.desktop`` file that is recognisably ours goes (it names the
    old launcher and the old window class); the icons carry the old name
    and live in the hicolor theme only because we put them there. A user
    who still runs an alpha from another environment gets the entry back
    when that alpha next starts. Returns how many files were removed;
    never raises.
    """
    removed = 0
    legacy = _data_home() / "applications" / f"{LEGACY_DESKTOP_NAME}.desktop"
    try:
        text = legacy.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        text = ""
    if "Name=mlgidLAB" in text and f"StartupWMClass={LEGACY_DESKTOP_NAME}" in text:
        try:
            legacy.unlink()
            removed += 1
        except OSError:
            logger.debug("could not remove the old desktop entry", exc_info=True)
    for size in ICON_SIZES:
        old_icon = (_data_home() / "icons" / "hicolor" / f"{size}x{size}"
                    / "apps" / f"{LEGACY_DESKTOP_NAME}.png")
        try:
            if old_icon.is_file():
                old_icon.unlink()
                removed += 1
        except OSError:
            logger.debug("could not remove an old icon", exc_info=True)
    return removed


def _install_icons() -> int:
    """Copy the shipped PNGs into the hicolor theme. Returns how many
    were written. A size missing from the wheel is skipped, not fatal --
    the shell scales whichever sizes it does find."""
    written = 0
    for size in ICON_SIZES:
        target = icon_path(size)
        try:
            source = resources.files("mlgidgui").joinpath(
                f"assets/app/mlgidgui_{size}.png")
            with resources.as_file(source) as real:
                data = Path(real).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError, OSError,
                AttributeError):
            logger.debug("no app icon asset at %d px", size, exc_info=True)
            continue
        try:
            if target.is_file() and target.read_bytes() == data:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written += 1
        except OSError:
            logger.debug("could not install the %d px icon", size,
                         exc_info=True)
    return written


def install() -> bool:
    """Install the desktop entry and its icons for the current user.

    Idempotent, and cheap on the common path: with everything already in
    place it reads eight small files and writes none. Returns True when
    something was actually written, which is the signal the caller uses
    to decide whether the icon cache is worth refreshing.

    Never raises. A failure here is a missing dash icon, not a failed
    launch, and the app is perfectly usable without it.
    """
    if not is_supported():
        return False
    removed = remove_legacy()
    try:
        icons_written = _install_icons()
        entry_written = _write_if_changed(
            desktop_file_path(), desktop_file_contents())
    except OSError:
        logger.debug("could not install the desktop entry", exc_info=True)
        return False
    if removed:
        logger.info("Removed %d file(s) of the old mlgidLAB desktop entry.",
                    removed)
    if entry_written or icons_written or removed:
        logger.info(
            "Installed the mlgidGUI desktop entry (%s) and %d icon(s). "
            "The shell may need a moment, or a re-login, to pick it up.",
            desktop_file_path(), icons_written,
        )
        return True
    return False
