"""Run the live 2D preview fits on a background thread.

Selecting a manual or detected peak in 2D mode previews what Add to
fitted (2D) would save, which needs a pygidfit fit. That fit is cheap
for most peaks (0.1 to 1 s) but a peak that sits in a badly conditioned
peak-on-ring cluster can take tens of seconds, and run on the GUI thread
it froze the window for that long. This runner moves it off the GUI
thread.

One daemon worker thread takes the jobs in order, so two fits never run
at once and pygidfit sees one caller at a time. A fit cannot be
interrupted once it has started, so what the runner does instead is skip
a queued job that is no longer wanted (the selection moved on before the
worker got to it) and report a finished job to the GUI thread by signal,
where the window decides whether the result is still current.

The thread is a daemon on purpose: closing the window must not wait for
a fit that may still have half a minute to run.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class PreviewFitRunner(QObject):
    """Serial background executor for preview fits.

    ``done(key, result, error, ran)`` is emitted once per submitted job.
    ``ran`` is False when the job was skipped because its key was
    neither wanted nor pinned; ``result`` and ``error`` are then None.
    """

    done = Signal(object, object, object, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: queue.Queue = queue.Queue()
        self._wanted: frozenset = frozenset()
        self._pinned: set = set()
        self._pending: set = set()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._loop, name="mlgidgui-preview-fit", daemon=True,
        )
        self._thread.start()

    # -- GUI-thread API --

    def want(self, keys) -> None:
        """Declare which keys are still of interest; queued jobs for any
        other key are skipped when the worker reaches them."""
        self._wanted = frozenset(keys)

    def pin(self, key) -> None:
        """Make ``key`` run whether or not it is still wanted. For a fit
        somebody is waiting on (Add to fitted), as opposed to a preview
        that is only worth having while its peak stays selected."""
        self._pinned.add(key)

    def is_pending(self, key) -> bool:
        """True while ``key`` is queued or running."""
        return key in self._pending

    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, key, job: Callable[[], tuple]) -> None:
        """Queue ``job`` (returns ``(result, error)``) under ``key``."""
        if self._stopped or key in self._pending:
            return
        self._pending.add(key)
        self._jobs.put((key, job))

    def forget(self, key) -> None:
        """Called by the owner once it has handled ``done`` for ``key``."""
        self._pending.discard(key)
        self._pinned.discard(key)

    def shutdown(self) -> None:
        """Stop taking work. A fit already running is left to finish on
        its daemon thread; its result is dropped."""
        self._stopped = True
        self._wanted = frozenset()
        self._jobs.put(None)

    # -- worker thread --

    def _loop(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            key, job = item
            if self._stopped:
                return
            ran = key in self._wanted or key in self._pinned
            result = error = None
            if ran:
                try:
                    result, error = job()
                except Exception as exc:  # the job reports its own errors
                    logger.debug("preview fit raised", exc_info=True)
                    result, error = None, f"Unexpected 2D fit error: {exc!r}"
            if self._stopped:
                return
            try:
                self.done.emit(key, result, error, ran)
            except RuntimeError:
                # The window was destroyed under a running fit.
                return
