"""Detected-overlay score cutoff (incl. the inclusive-at-max epsilon).

There is no public "visible detected count"; the filter
(``score >= value - 0.005``) is inlined in ``_render_overlays``
(``ViewerOverlaysMixin`` / ``image_viewer``) and the result is drawn into
the ``_PeakShapeItem`` QPainterPath. Peaks are injected through the
public ``set_peaks``; ``set_detected_score_cutoff`` re-renders. All three rows share identical geometry so
angular clipping treats them the same and only ``score`` discriminates.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.file_model import PeakTable
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _detected_table():
    n = 3
    return PeakTable(
        q_xy=np.full(n, 1.4142, dtype=float),
        q_z=np.full(n, 1.4142, dtype=float),
        angle=np.full(n, 45.0, dtype=float),
        radius=np.full(n, 2.0, dtype=float),
        angle_width=np.full(n, 4.0, dtype=float),
        radius_width=np.full(n, 0.2, dtype=float),
        is_ring=np.zeros(n, dtype=bool),
        ids=np.array([0, 1, 2]),
        score=np.array([0.40, 0.75, 1.00]),
        amplitude=np.array([10.0, 20.0, 30.0]),
    )


def test_score_cutoff_filters_and_is_inclusive_at_max(
    main_window, synthetic_nexus
):
    _open(main_window, synthetic_nexus)
    v = main_window.viewer
    v.set_peaks(0, {"detected": _detected_table(), "fitted": None})

    # No filter (guard is ``> 0.0``): all rows drawn.
    v.set_detected_score_cutoff(0.0)
    assert not v._detected._path.isEmpty()

    # Cutoff at the max score: eff cutoff 0.995, the 1.00 row still
    # passes (inclusive-at-max) → something is drawn.
    v.set_detected_score_cutoff(1.00)
    assert not v._detected._path.isEmpty()

    # Just above the max: eff cutoff 1.005, nothing passes → empty.
    v.set_detected_score_cutoff(1.01)
    assert v._detected._path.isEmpty()
