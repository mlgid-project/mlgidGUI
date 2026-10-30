"""Startup update check + post-update changelog (offline).

Two independent, best-effort features, both fail silently:

* ``latest_release`` asks the GitHub Releases API for the newest published
  release (pre-releases included — our alphas are pre-releases) and
  ``is_outdated`` compares it to the running version. Any network / parse
  failure returns ``None`` so the GUI simply shows nothing.
* ``whats_new`` reads the bundled ``CHANGELOG.md`` and returns the sections
  strictly newer than the last version the user ran (persisted by the host
  in ``QSettings``), so opening a freshly updated build can show what changed.

This module is deliberately Qt-free so the parsing / comparison logic is unit
testable without a QApplication; the host (``main_window``) runs
``latest_release`` off-thread and renders the results.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import sys
import sysconfig
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

logger = logging.getLogger("mlgidgui")

# Single source of truth for the GitHub repo. It is only used to learn that a
# newer release exists (the Releases API) and to link to the releases page;
# the install itself comes from PyPI and does not depend on the repo name.
# (The repo was called ``mlgidBASE_GUI`` and then ``mlgidLAB`` before the
# rename to mlgidGUI; GitHub redirects the old names.)
_OWNER_REPO = "mlgid-project/mlgidGUI"
RELEASES_API = f"https://api.github.com/repos/{_OWNER_REPO}/releases"
RELEASES_PAGE = f"https://github.com/{_OWNER_REPO}/releases"

# Our own installed distribution, as named on PyPI.
DIST_NAME = "mlgidgui"
# The distribution that anchors the heavy ``[pipeline]`` extra. If it is
# installed we keep the extra on self-update so the backend stays wired up.
_PIPELINE_ANCHOR_DIST = "mlgidbase"


def parse_version(tag: str | None) -> Version | None:
    """Parse a PEP 440 version from a tag like ``v0.1.0a9`` (``None`` if bad)."""
    if not tag:
        return None
    try:
        return Version(str(tag).strip().lstrip("vV"))
    except InvalidVersion:
        return None


def pick_latest_release(releases: object) -> tuple[str, str] | None:
    """From the GitHub ``/releases`` JSON list return ``(tag, html_url)`` of the
    highest PEP 440 version, skipping drafts.

    Pre-releases are kept (the project ships ``aN`` alphas). Returns ``None``
    when the list is empty or nothing parses as a version.
    """
    if not isinstance(releases, list):
        return None
    best: tuple[Version, str, str] | None = None
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        tag = rel.get("tag_name") or rel.get("name") or ""
        version = parse_version(tag)
        if version is None:
            continue
        url = rel.get("html_url") or RELEASES_PAGE
        if best is None or version > best[0]:
            best = (version, tag, url)
    if best is None:
        return None
    return best[1], best[2]


def latest_release(timeout: float = 4.0) -> tuple[str, str] | None:
    """Fetch the latest release ``(tag, html_url)`` from GitHub.

    Returns ``None`` on any failure (offline, rate-limited, no releases
    published) so callers stay silent. Blocking — run off the GUI thread.
    """
    req = urllib.request.Request(
        RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mlgidGUI-update-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        logger.debug("update check: release fetch failed", exc_info=True)
        return None
    return pick_latest_release(data)


def is_outdated(current: str, latest: str) -> bool:
    """True when ``latest`` is a strictly newer PEP 440 version than
    ``current`` (False if either is unparsable)."""
    c, l = parse_version(current), parse_version(latest)
    if c is None or l is None:
        return False
    return l > c


# --------------------------------------------------------------------- #
# In-app self-update (pip install --upgrade from PyPI)
# --------------------------------------------------------------------- #
def _read_direct_url(dist_name: str = DIST_NAME) -> dict | None:
    """Return the parsed ``direct_url.json`` of the installed ``dist_name``.

    ``direct_url.json`` records how a distribution was installed (PEP 610).
    Returns ``None`` when the distribution isn't found, has no such record
    (e.g. installed from a plain wheel), or the JSON can't be parsed — kept
    as its own helper so tests can monkeypatch it.
    """
    try:
        from importlib.metadata import distribution

        raw = distribution(dist_name).read_text("direct_url.json")
    except Exception:  # PackageNotFoundError, OSError, ...
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _dist_present(name: str) -> bool:
    """True when a distribution named ``name`` is importable metadata-wise."""
    try:
        from importlib.metadata import distribution

        distribution(name)
        return True
    except Exception:
        return False


def is_editable_install(dist_name: str = DIST_NAME) -> bool:
    """True when ``dist_name`` is an editable (``pip install -e``) install.

    Editable installs point at a live source checkout, so a
    ``pip install --upgrade`` would clobber the link rather than update it —
    the host uses this to refuse self-update for dev checkouts.
    """
    data = _read_direct_url(dist_name)
    if not data:
        return False
    dir_info = data.get("dir_info")
    return bool(isinstance(dir_info, dict) and dir_info.get("editable"))


def pipeline_extra_installed() -> bool:
    """True when the heavy ``[pipeline]`` extra looks installed.

    Detected via its anchor distribution (``mlgidbase``) so self-update can
    preserve the extra instead of silently dropping the backend.
    """
    return _dist_present(_PIPELINE_ANCHOR_DIST)


def is_frozen() -> bool:
    """True inside the Windows installer build (PyInstaller sets ``sys.frozen``)."""
    return bool(getattr(sys, "frozen", False))


def _installer_name(dist_name: str = DIST_NAME) -> str | None:
    """The tool that installed ``dist_name`` (the ``INSTALLER`` file of its
    metadata: ``pip``, ``conda``, a distro's package manager), or ``None``."""
    try:
        from importlib.metadata import distribution

        raw = distribution(dist_name).read_text("INSTALLER")
    except Exception:
        return None
    return raw.strip() if raw and raw.strip() else None


def self_update_supported(dist_name: str = DIST_NAME) -> tuple[bool, str]:
    """Whether an in-app ``pip install --upgrade`` is safe for this install.

    Returns ``(True, "")`` for a normal pip-managed install: one pip put
    there from PyPI (no ``direct_url.json``, ``INSTALLER`` says ``pip``) or
    from a URL or a wheel (``direct_url.json``). Returns ``(False, reason)``
    for the frozen installer build, for an editable/dev checkout, or when
    the install can't be identified as a pip one (installed by conda or a
    system package manager, or no metadata) — upgrading in place could break
    the environment, so the host offers only "View release" instead.
    """
    if is_frozen():
        # The installer build has no pip to run: an update is a new
        # installer, so the host offers the releases page instead.
        return (
            False,
            "This is the installed app. To update, download the new "
            "installer from the releases page and run it.",
        )
    if is_editable_install(dist_name):
        return (
            False,
            "This is a development (editable) install; update it with "
            "'git pull' or 'pip install -e .' instead.",
        )
    if _read_direct_url(dist_name) is None and _installer_name(dist_name) != "pip":
        return (
            False,
            "Could not determine how mlgidGUI was installed with pip; "
            "update it with the tool you installed it with.",
        )
    return True, ""


def build_upgrade_command(
    tag: str,
    *,
    executable: str | None = None,
    with_pipeline: bool | None = None,
) -> list[str]:
    """Build the ``pip install`` argv for a given release ``tag``.

    Installs exactly the version the banner announced from PyPI
    (``pip install "mlgidgui[pipeline]==<version>"``). A pin rather than
    ``--upgrade`` alone: a plain upgrade skips a pre-release, while an exact
    ``==`` pin installs it without ``--pre``, and the user gets the release
    they were told about. ``executable`` defaults to the running interpreter
    so the install lands in the active environment; ``with_pipeline`` defaults
    to whether the ``[pipeline]`` extra is currently installed (preserving it
    across the update).
    """
    version = parse_version(tag)
    if version is None:
        raise ValueError(f"not a release tag: {tag!r}")
    if executable is None:
        executable = sys.executable
    if with_pipeline is None:
        with_pipeline = pipeline_extra_installed()
    extra = "[pipeline]" if with_pipeline else ""
    return [executable, "-m", "pip", "install", f"{DIST_NAME}{extra}=={version}"]


# --------------------------------------------------------------------- #
# Windows: free the running launcher exe before pip touches it
# --------------------------------------------------------------------- #
LAUNCHER_BACKUP_SUFFIX = ".old"


def _launcher_entries() -> list[str]:
    """``sys.path`` entries that are launcher executables.

    Started as ``Scripts\\mlgidgui.exe``, the interpreter runs that exe as a
    zip archive, so its path becomes ``sys.path[0]``.
    """
    return [
        entry for entry in sys.path
        if isinstance(entry, str) and entry.lower().endswith(".exe")
    ]


def drop_launcher_from_sys_path() -> list[str]:
    """Take launcher executables off ``sys.path``; return the entries removed.

    ``importlib.metadata`` scans every ``sys.path`` entry, and for a zip
    (the launcher) it opens the archive and keeps it open for the life of
    the process. An open file cannot be renamed or deleted on Windows, so
    the exe could never be moved aside for an update and pip failed with
    ``[WinError 32]``. The zip only holds the ``__main__`` that already
    ran, so nothing needs it on the path afterwards.
    """
    removed = _launcher_entries()
    for entry in removed:
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path_importer_cache.pop(entry, None)
    return removed


def release_launcher_handles() -> None:
    """Close the handle on the launcher that this process may already hold.

    Companion to :func:`drop_launcher_from_sys_path` for what was opened
    before it ran: take the entry off the path and drop the
    ``importlib.metadata`` caches that keep the archive open, so the file
    is closed by the time the launcher is renamed aside.
    """
    drop_launcher_from_sys_path()
    with contextlib.suppress(Exception):
        from importlib.metadata import FastPath

        FastPath.__new__.cache_clear()
    gc.collect()


def _scripts_dir() -> Path:
    """The environment's console/GUI-scripts directory (``Scripts`` on
    Windows, ``bin`` elsewhere)."""
    return Path(sysconfig.get_path("scripts"))


def free_locked_launchers(
    *,
    platform: str | None = None,
    scripts_dir: Path | None = None,
) -> list[tuple[Path, Path]]:
    """On Windows, rename this dist's launcher exe out of pip's way.

    When the GUI runs via the ``gui-scripts`` launcher
    (``Scripts\\mlgidgui.exe``), that exe is the image of a live process
    and Windows refuses to delete or overwrite it — pip's uninstall of
    the old version then dies with ``[WinError 32]``. Renaming a running
    exe IS allowed, so the launcher is moved aside first; pip logs the
    RECORD entry as already-missing and installs a fresh exe. The backup
    stays locked until the app exits and is swept on a later call.

    Returns the ``(original, backup)`` renames performed, for
    :func:`restore_launchers` to undo if pip fails.
    """
    if platform is None:
        platform = os.name
    if platform != "nt":
        return []
    # Our own open handle on the launcher would block the rename below.
    release_launcher_handles()
    if scripts_dir is None:
        scripts_dir = _scripts_dir()
    exe = scripts_dir / f"{DIST_NAME}.exe"
    # Sweep backups from earlier updates whose process has since exited.
    for stale in scripts_dir.glob(f"{exe.name}{LAUNCHER_BACKUP_SUFFIX}*"):
        with contextlib.suppress(OSError):
            stale.unlink()
    if not exe.exists():
        return []
    renames: list[tuple[Path, Path]] = []
    for n in range(1, 10):
        suffix = LAUNCHER_BACKUP_SUFFIX + (str(n) if n > 1 else "")
        backup = exe.with_name(exe.name + suffix)
        if backup.exists():
            continue  # a still-locked backup survived the sweep
        try:
            exe.rename(backup)
        except OSError:
            # Leave things as they were; pip will fail exactly as it
            # does today and the failure dialog shows the manual path.
            break
        renames.append((exe, backup))
        break
    return renames


def restore_launchers(renames: list[tuple[Path, Path]]) -> None:
    """Undo :func:`free_locked_launchers` after a failed install.

    If pip never wrote a replacement, the backup is renamed back so the
    launcher is not lost. If the original exists again (pip rolled back
    the uninstall or got far enough to install the new exe), the backup
    is just dropped, best-effort.
    """
    for original, backup in renames:
        with contextlib.suppress(OSError):
            if original.exists():
                backup.unlink()
            else:
                backup.rename(original)


def sweep_legacy_launchers(
    scripts_dir: Path | None = None, *, platform: str | None = None,
) -> int:
    """Windows: delete the ``mlgidlab.exe.old*`` the move from mlgidLAB left.

    Updating an mlgidLAB alpha to mlgidGUI renames the running ``mlgidlab.exe``
    aside (a running exe cannot be deleted) and uninstalls the old package.
    The backup stays locked until that old process exits, and nothing else
    ever removes it, so the renamed app deletes it on a later start. Best
    effort and silent: a file that is still locked is tried again next time.
    Returns how many files were removed.
    """
    if (platform or os.name) != "nt" or is_frozen():
        return 0
    if scripts_dir is None:
        scripts_dir = _scripts_dir()
    removed = 0
    try:
        stale = list(scripts_dir.glob(f"mlgidlab.exe{LAUNCHER_BACKUP_SUFFIX}*"))
    except OSError:
        return 0
    for path in stale:
        try:
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("old launcher %s is still in use", path)
    return removed


# --------------------------------------------------------------------- #
# Changelog (offline, from the bundled CHANGELOG.md)
# --------------------------------------------------------------------- #
def _changelog_path() -> Path | None:
    """Locate the bundled ``CHANGELOG.md`` (``None`` if absent).

    An installed wheel carries it inside the package (``setup.py`` copies
    it there at build time); a source checkout or editable install has it
    at the repository root.
    """
    try:
        import mlgidgui

        pkg = Path(mlgidgui.__file__).resolve().parent
        for p in (pkg / "CHANGELOG.md", pkg.parent / "CHANGELOG.md"):
            if p.is_file():
                return p
        return None
    except Exception:  # pragma: no cover - defensive
        logger.debug("update check: changelog path lookup failed", exc_info=True)
        return None


def parse_changelog_sections(text: str) -> list[tuple[Version | None, str, str]]:
    """Split a changelog into ``(version, heading, body)`` per ``## `` heading.

    ``version`` is parsed from the first token of the heading (e.g.
    ``0.1.0a9`` in ``## 0.1.0a9 — ninth alpha (2026-06-30)``) or ``None``.
    """
    sections: list[tuple[str, list[str]]] = []
    head: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if head is not None:
                sections.append((head, lines))
            head = line[3:].strip()
            lines = []
        elif head is not None:
            lines.append(line)
    if head is not None:
        sections.append((head, lines))
    out: list[tuple[Version | None, str, str]] = []
    for h, body_lines in sections:
        token = h.split()[0] if h.split() else ""
        out.append((parse_version(token), h, "\n".join(body_lines).strip()))
    return out


def whats_new(current: str, last_seen: str | None) -> list[tuple[str, str]] | None:
    """Changelog sections to show after an update: versions ``v`` with
    ``last_seen < v <= current`` (newest first).

    When ``last_seen`` is falsy (fresh install / first run ever) returns
    ``None`` — nothing is shown, the host just records the current version.
    Returns ``None`` when the changelog can't be read or nothing qualifies.
    """
    if not last_seen:
        return None
    path = _changelog_path()
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("update check: changelog read failed", exc_info=True)
        return None
    cur, low = parse_version(current), parse_version(last_seen)
    if cur is None or low is None or not (low < cur):
        return None
    picked: list[tuple[Version, str, str]] = []
    for version, head, body in parse_changelog_sections(text):
        if version is not None and low < version <= cur:
            picked.append((version, head, body))
    picked.sort(key=lambda t: t[0], reverse=True)
    return [(h, b) for _, h, b in picked] or None
