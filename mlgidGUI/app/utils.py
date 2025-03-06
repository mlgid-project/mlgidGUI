import logging
import sys
import traceback

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QRunnable

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d

import numpy as np

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

    def clear(cls):
        cls._instance = None

class InternalError(Exception):
    pass


class CorruptedFileError(Exception):
    pass


def baseline_correction(y: np.ndarray,
                        smoothness_param: float,
                        asymmetry_param: float,
                        max_niter: int = 1000) -> np.ndarray:
    z = np.zeros_like(y)
    if smoothness_param <= 0 or asymmetry_param <= 0:
        return z
    y_size = y.size
    laplacian = sparse.diags([1, -2, 1], [0, -1, -2], shape=(y_size, y_size - 2))
    laplacian_matrix = laplacian.dot(laplacian.transpose())

    w = np.ones(y_size)
    for i in range(max_niter):
        W = sparse.spdiags(w, 0, y_size, y_size)
        Z = W + smoothness_param * laplacian_matrix
        z = spsolve(Z, w * y)
        w_new = asymmetry_param * (y > z) + (1 - asymmetry_param) * (y < z)
        if np.allclose(w, w_new):
            break
        w = w_new
    else:
        logger.info(f'Solution has not converged, max number of iterations reached.')
    return np.nan_to_num(z)


def smooth_curve(y: np.ndarray, sigma: float) -> np.ndarray or None:
    if y is not None:
        if sigma > 0:
            return gaussian_filter1d(y, sigma)
        else:
            return y


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    log = logging.getLogger(__name__)

    def __init__(self, fn, *args, signals: WorkerSignals = None, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = signals or WorkerSignals()

    def call_func(self):
        return self.fn(*self.args, **self.kwargs)

    @pyqtSlot()
    def run(self):

        try:
            result = self.call_func()
        except Exception as err:
            self.log.exception(err)
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            try:
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            except RuntimeError:
                return
        else:
            try:
                self.signals.result.emit(result)
            except RuntimeError:
                return
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                return


class UpdateWorkerSignals(WorkerSignals):
    sigSetMax = pyqtSignal(int)
    sigSetProgress = pyqtSignal(int)


class UpdateWorker(Worker):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(fn, *args, signals=UpdateWorkerSignals(), **kwargs)

    def call_func(self):
        return self.fn(*self.args, process_callback=self.signals.sigSetProgress.emit,
                       set_max_callback=self.signals.sigSetMax.emit, **self.kwargs)
