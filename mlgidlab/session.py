from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from mlgidlab import h5_repack


SessionKind = Literal["nexus", "raw"]

# NexusSession copies each opened file into its own per-session temp dir named
# ``mlgidlab_<pid>_*`` under the system temp dir; all edits target that working
# copy and Save copies it back. Three layers keep these from piling up in /tmp:
#   - close()/closeEvent removes the dir on a graceful exit (the common path);
#   - the registry + atexit handler below catch normal-exit / unhandled
#     exceptions (anything that runs Python's atexit);
#   - sweep_stale_temp_dirs(), called once at GUI startup, reclaims dirs left by
#     a previous run that was killed before either of the above could run.
# The dir name embeds the creating PID so the sweep only ever deletes dirs whose
# owning process is gone — a concurrently running instance's dirs are untouched.
_TEMP_DIR_PREFIX = "mlgidlab_"
_active_temp_dirs: set[str] = set()
_atexit_armed = False


def _arm_atexit() -> None:
    global _atexit_armed
    if not _atexit_armed:
        atexit.register(cleanup_registered_temp_dirs)
        _atexit_armed = True


def _register_temp_dir(path: Path) -> None:
    _active_temp_dirs.add(str(path))
    _arm_atexit()


def _unregister_temp_dir(path: Path) -> None:
    _active_temp_dirs.discard(str(path))


def cleanup_registered_temp_dirs() -> None:
    """Remove every still-open session temp dir.

    Registered with ``atexit`` and also called explicitly by the test suite,
    which exits via ``os._exit`` (skipping atexit)."""
    for p in list(_active_temp_dirs):
        shutil.rmtree(p, ignore_errors=True)
    _active_temp_dirs.clear()


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (signal 0 probes without sending)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # PermissionError etc. -> exists but not ours
    return True


def sweep_stale_temp_dirs() -> int:
    """Delete ``mlgidlab_<pid>_*`` temp dirs whose owning process is gone.

    Call once at GUI startup to reclaim working copies leaked by a previous run
    that exited without ``close()`` (crash / kill / power loss). Only PID-tagged
    dirs whose PID is no longer alive are removed, so a concurrently running
    instance is never disturbed. Returns the number of dirs removed."""
    root = Path(tempfile.gettempdir())
    me = os.getpid()
    pat = re.compile(r"^mlgidlab_(\d+)_")
    removed = 0
    try:
        candidates = list(root.glob(_TEMP_DIR_PREFIX + "*"))
    except OSError:
        return 0
    for d in candidates:
        m = pat.match(d.name)
        if m is None or not d.is_dir():
            continue
        pid = int(m.group(1))
        if pid == me or _pid_alive(pid):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    return removed


def _disk_signature(path: Path) -> tuple[int, int] | None:
    """Cheap change-detection signature: ``(mtime_ns, size)``.

    None when the file is unreadable/missing — callers treat that as
    "can't tell", not "changed".
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class BaseSession:
    """Common state for any opened-file session.

    Subclasses carry mode-specific paths. ``display_path`` is the path the
    UI title and tree label should show; ``temp_path`` is whatever HDF5
    file the rest of the app should treat as the active file (the writable
    NeXus working copy, or the read-only first raw input for a raw batch).
    """

    kind: SessionKind
    display_path: Path
    temp_path: Path

    def __init__(self) -> None:
        self.dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    @property
    def original_path(self) -> Path:
        """Back-compat shim — old call sites read ``session.original_path``."""
        return self.display_path

    def close(self) -> None:
        raise NotImplementedError


class NexusSession(BaseSession):
    """Working copy of a converted NeXus file.

    The original is copied into a fresh per-session temp directory on open,
    keeping the original basename so the silx tree shows the right filename.
    All edits target the temp copy; the original is only touched on Save.
    """

    kind: SessionKind = "nexus"

    def __init__(self, original_path: Path, temp_path: Path) -> None:
        super().__init__()
        self._original_path = original_path
        self.temp_path = temp_path
        # Snapshot of the original's on-disk state at open/save time, so
        # the file-browser Refresh can tell "changed underneath us" from
        # "still what we copied". Updated by save()/save_as()/reload.
        self._disk_stat = _disk_signature(original_path)

    def disk_changed(self) -> bool:
        """Whether the original changed on disk since open/save/reload.

        False when the original is unreadable (deletion is detected
        separately via ``exists()``) or when no baseline was recorded.
        """
        sig = _disk_signature(self._original_path)
        return (
            sig is not None
            and self._disk_stat is not None
            and sig != self._disk_stat
        )

    def reload_from_disk(self) -> None:
        """Re-copy the original over the temp working copy.

        Discards any edits in the temp copy — callers guard on
        ``dirty``. Refreshes the disk baseline so a follow-up Refresh
        sees the session as up to date.
        """
        shutil.copy2(self._original_path, self.temp_path)
        self.dirty = False
        self._disk_stat = _disk_signature(self._original_path)

    @property
    def display_path(self) -> Path:  # type: ignore[override]
        return self._original_path

    @display_path.setter
    def display_path(self, value: Path) -> None:
        self._original_path = value

    @classmethod
    def open(
        cls,
        original_path: Path | str,
        progress: "callable | None" = None,
    ) -> NexusSession:
        """Copy ``original_path`` into a fresh temp dir and wrap it.

        ``progress(bytes_done, bytes_total)`` is called as the copy
        advances — a converted NeXus file can be several GB, so the
        copy is the dominant open cost and the open bar shows it as
        real byte progress. Without a callback the copy is a plain
        ``shutil.copy2``.
        """
        original = Path(original_path).resolve()
        if not original.is_file():
            raise FileNotFoundError(original)

        temp_dir = Path(tempfile.mkdtemp(prefix=f"{_TEMP_DIR_PREFIX}{os.getpid()}_"))
        temp_path = temp_dir / original.name
        try:
            if progress is None:
                shutil.copy2(original, temp_path)
            else:
                cls._copy_with_progress(original, temp_path, progress)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        _register_temp_dir(temp_dir)
        return cls(original_path=original, temp_path=temp_path)

    @staticmethod
    def _copy_with_progress(
        src: Path, dst: Path, progress, chunk_size: int = 16 * 1024 * 1024
    ) -> None:
        """Chunked ``copy2`` equivalent (data + stat) with a byte tick
        per chunk."""
        total = src.stat().st_size
        done = 0
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                done += len(buf)
                progress(done, total)
        shutil.copystat(src, dst)

    def save(self) -> None:
        """Overwrite the original from the temp file.

        Repacks instead of copying when the working copy is carrying the
        holes a delete left behind -- HDF5 unlinks an object without
        shrinking the file, so a plain byte copy would hand the user back
        a file exactly as large as before they deleted a 2 GB stack. See
        ``h5_repack``: the decision is measured, and a file with nothing
        to reclaim takes the same byte copy it always did.
        """
        h5_repack.copy_or_repack(self.temp_path, self._original_path)
        self.dirty = False
        self._disk_stat = _disk_signature(self._original_path)

    def save_as(self, new_path: Path | str) -> None:
        """Write the temp to a new path; adopt it as the new original.

        Renames the temp file in place so its basename matches the new path —
        callers that re-open the silx tree afterward will see the new name.
        """
        new = Path(new_path).resolve()
        h5_repack.copy_or_repack(self.temp_path, new)

        new_temp = self.temp_path.parent / new.name
        if new_temp != self.temp_path:
            self.temp_path.rename(new_temp)
            self.temp_path = new_temp

        self._original_path = new
        self.dirty = False
        self._disk_stat = _disk_signature(new)

    def close(self) -> None:
        """Delete the temp file and its per-session directory. Idempotent."""
        temp_dir = self.temp_path.parent
        shutil.rmtree(temp_dir, ignore_errors=True)
        _unregister_temp_dir(temp_dir)


class RawSession(BaseSession):
    """A batch of raw HDF5 detector files awaiting conversion.

    Unlike a NeXus session, the raw inputs are read-only — pygid only reads
    them. ``raw_paths`` is ordered as the user selected them in the open
    dialog. ``temp_path`` exposes the first raw path so generic call sites
    that ask "which file is this?" still work; mode-aware code should
    iterate ``raw_paths`` directly.
    """

    kind: SessionKind = "raw"

    def __init__(self, raw_paths: list[Path]) -> None:
        super().__init__()
        if not raw_paths:
            raise ValueError("RawSession requires at least one raw file path")
        self._raw_paths = list(raw_paths)
        self._raw_path_strs: frozenset[str] | None = None
        # The "temp" of a raw session is just the first raw file: no
        # writable copy is made. Saving to a raw input is meaningless;
        # the user produces output via the Conversion panel instead.
        self.temp_path = self._raw_paths[0]
        self.output_paths: list[Path] = []

    @property
    def display_path(self) -> Path:  # type: ignore[override]
        # Show the first file's name as the session label; multi-file
        # batches read as "<first.h5>" with the rest visible in the tree.
        return self._raw_paths[0]

    @property
    def raw_paths(self) -> list[Path]:
        return list(self._raw_paths)

    @property
    def raw_path_strs(self) -> frozenset[str]:
        """The (already-resolved) raw paths as a cached string set.

        File-browser click resolution matches a clicked node's file
        against the session; re-resolving thousands of ``Path`` objects
        per click is measurable syscall load (and painful on network
        mounts), while this set is O(1) per lookup. ``open`` resolved
        every path once, so plain string equality is canonical.
        """
        if self._raw_path_strs is None:
            self._raw_path_strs = frozenset(str(p) for p in self._raw_paths)
        return self._raw_path_strs

    @classmethod
    def open(cls, raw_paths: list[Path | str]) -> RawSession:
        resolved: list[Path] = []
        for p in raw_paths:
            path = Path(p).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved.append(path)
        if not resolved:
            raise ValueError("RawSession.open requires at least one path")
        return cls(raw_paths=resolved)

    def add_paths(self, raw_paths: list[Path | str]) -> list[Path]:
        """Append files to the batch and return the ones actually added.

        Files already in the batch are skipped, so dropping the same file
        twice does not duplicate its row. Paths are resolved the way
        ``open`` resolves them, which keeps ``raw_path_strs`` canonical.
        """
        known = set(self.raw_path_strs)
        added: list[Path] = []
        for p in raw_paths:
            path = Path(p).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if str(path) in known:
                continue
            known.add(str(path))
            added.append(path)
        if added:
            self._raw_paths.extend(added)
            self._raw_path_strs = None
        return added

    def remove_paths(self, raw_paths: list[Path | str]) -> list[Path]:
        """Drop files from the batch and return the ones actually removed.

        Unknown paths are ignored. A batch cannot be emptied this way (the
        session would have no files to stand for); callers close the
        session instead. ``temp_path`` follows the new first file.
        """
        drop = {str(Path(p).resolve()) for p in raw_paths}
        removed = [p for p in self._raw_paths if str(p) in drop]
        if not removed:
            return []
        kept = [p for p in self._raw_paths if str(p) not in drop]
        if not kept:
            raise ValueError("remove_paths would leave the session empty")
        self._raw_paths = kept
        self._raw_path_strs = None
        self.temp_path = kept[0]
        return removed

    def close(self) -> None:
        """Raw inputs are not owned by the GUI; nothing to delete.

        Defined for parity with NexusSession.close so callers can treat
        every session uniformly.
        """
        return None


# Back-compat alias: existing call sites import ``Session`` from this
# module. Keep the name pointing at the converted-NeXus subclass since
# that's what every legacy call site expects (it carries a writable
# temp_path and a save() method).
Session = NexusSession
