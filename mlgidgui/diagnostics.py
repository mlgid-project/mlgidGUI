"""Persistent per-user log files: the crash log and the self-update log.

Qt-free so the crash log can be opened before the QApplication exists.
Every filesystem failure is swallowed: a log that cannot be written must
never be what stops the app or an update.

The per-user directory survives a reboot, unlike the system temp dir the
crash log used to live in, so a crash reported a day later can still be
read. ``MLGIDGUI_LOG_DIR`` overrides the location (the test suite points
it at a temp dir so runs never write to the real profile); the name the
alphas used, ``MLGIDLAB_LOG_DIR``, still works when the new one is unset.
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

LOG_DIR_ENV = "MLGIDGUI_LOG_DIR"
LEGACY_LOG_DIR_ENV = "MLGIDLAB_LOG_DIR"
CRASH_LOG_NAME = "crash.log"
UPDATE_LOG_NAME = "update_install.log"
CRASH_LOG_MAX_BYTES = 1024 * 1024

# Where the last update-log write actually landed (the per-user dir, or
# the temp fallback), so the failure dialog can point at the real file.
_last_update_log: Path | None = None


def _platform_log_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "mlgidgui" / "Logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "mlgidgui"
    # XDG says a relative XDG_STATE_HOME is invalid and must be ignored.
    state = os.environ.get("XDG_STATE_HOME", "")
    root = Path(state) if os.path.isabs(state) else (
        Path.home() / ".local" / "state")
    return root / "mlgidgui"


def log_dir() -> Path | None:
    """Return the per-user log directory, creating it; ``None`` on failure."""
    try:
        override = (os.environ.get(LOG_DIR_ENV)
                    or os.environ.get(LEGACY_LOG_DIR_ENV))
        path = Path(override) if override else _platform_log_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _candidates(name: str, temp_name: str) -> list[Path]:
    paths = []
    directory = log_dir()
    if directory is not None:
        paths.append(directory / name)
    try:
        paths.append(Path(tempfile.gettempdir()) / temp_name)
    except Exception:
        pass
    return paths


def _rotate(path: Path, max_bytes: int) -> None:
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except Exception:
        # Missing file, or (Windows) another instance holds it open:
        # keep appending to the current one.
        pass


def open_crash_log():
    """Open the crash log for appending; return ``(stream, path)`` or ``None``.

    Rotates to ``crash.log.1`` first when the file exceeds
    ``CRASH_LOG_MAX_BYTES``. Falls back to ``mlgidgui_crash.log`` in the
    system temp dir when the per-user directory is not writable. The
    caller keeps the stream open for the life of the process.
    """
    for path in _candidates(CRASH_LOG_NAME, "mlgidgui_crash.log"):
        _rotate(path, CRASH_LOG_MAX_BYTES)
        try:
            return open(path, "a", buffering=1, encoding="utf-8"), path
        except Exception:
            continue
    return None


def _timestamp() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def append_update_log(text: str) -> Path | None:
    """Append ``text`` to the update log; return the file written, or ``None``.

    Synced to disk before returning, so a block written just before pip
    runs survives a crash during the install. Never raises.
    """
    global _last_update_log
    try:
        for path in _candidates(UPDATE_LOG_NAME, "mlgidgui_update_install.log"):
            try:
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except OSError:
                        pass
            except Exception:
                continue
            _last_update_log = path
            return path
    except Exception:
        pass
    return None


def update_log_path() -> Path | None:
    """The update log last written in this process, or ``None``."""
    return _last_update_log


def log_update_started(command: list[str]) -> Path | None:
    """Record an install about to run: versions, platform, full command."""
    try:
        import platform

        from mlgidgui import __version__

        try:
            plat = platform.platform()
        except Exception:
            plat = sys.platform
        text = (
            f"\n=== update started {_timestamp()} (pid {os.getpid()}) ===\n"
            f"mlgidGUI: {__version__}\n"
            f"Python: {sys.version}\n"
            f"Platform: {plat}\n"
            f"Command: {' '.join(str(part) for part in command)}\n"
        )
    except Exception:
        return None
    return append_update_log(text)


def log_update_finished(returncode: int, output: str) -> Path | None:
    """Record the install's return code and pip's full combined output."""
    try:
        body = output if output.endswith("\n") or not output else output + "\n"
        text = (
            f"=== update finished {_timestamp()} (pid {os.getpid()}), "
            f"return code {returncode} ===\n{body}"
        )
    except Exception:
        return None
    return append_update_log(text)


def log_restart_requested(module: str = "mlgidgui") -> Path | None:
    """Record that the user asked to relaunch after a successful update.

    ``module`` is what ``python -m`` relaunches, so the log shows what was
    really started.
    """
    try:
        text = (
            f"=== restart requested {_timestamp()} (pid {os.getpid()}): "
            f"{sys.executable} -m {module} ===\n"
        )
    except Exception:
        return None
    return append_update_log(text)
