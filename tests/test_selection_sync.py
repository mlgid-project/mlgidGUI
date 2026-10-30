"""Image-driven selection mirrored onto the peaks table tabs.

``PeaksTablePanel`` constructs standalone.
``set_external_selection`` clears all three tables first,
then for a non-manual/non-None selection switches to the matching tab
*before* row selection — so the tab switch is observable even with
empty models. ``_TAB_BY_KIND`` is {detected:0, fitted:1, matched:2}
(``peaks_table_panel``). Wiring is in ``main_window_build``.
"""

from __future__ import annotations

import pytest

from mlgidgui.image_viewer import SelectedPeak
from mlgidgui.peaks_table_panel import PeaksTablePanel
from mlgidgui.session import NexusSession

pytestmark = pytest.mark.gui


def _sel(kind):
    return SelectedPeak(
        kind=kind, frame=0, peak_id=0,
        radius=1.0, angle=45.0, radius_width=0.1, angle_width=10.0,
    )


def _no_selection(panel) -> bool:
    return not any(
        t.selectionModel().hasSelection()
        for t in (
            panel._detected_table, panel._fitted_table, panel._matched_table,
        )
    )


def test_set_external_selection_switches_tab_and_clears(qtbot):
    panel = PeaksTablePanel()
    qtbot.addWidget(panel)

    assert panel._tabs.currentIndex() == 0

    # None: clears every table, no tab switch.
    panel.set_external_selection(None)
    assert panel._tabs.currentIndex() == 0
    assert _no_selection(panel)

    # Fitted → tab index 1.
    panel.set_external_selection(_sel("fitted"))
    assert panel._tabs.currentIndex() == 1

    # None again: selection cleared, current tab left where it was.
    panel.set_external_selection(None)
    assert panel._tabs.currentIndex() == 1
    assert _no_selection(panel)

    # Manual peaks have no table row → all cleared, no tab switch.
    panel.set_external_selection(_sel("manual"))
    assert panel._tabs.currentIndex() == 1
    assert _no_selection(panel)

    # Detected → tab index 0.
    panel.set_external_selection(_sel("detected"))
    assert panel._tabs.currentIndex() == 0


def test_viewer_selection_signal_is_wired_to_panel(
    main_window, synthetic_nexus
):
    session = NexusSession.open(synthetic_nexus)
    main_window._set_active_session(session)

    main_window.viewer.selectionChanged.emit(_sel("fitted"))
    assert main_window.peaks_table_panel._tabs.currentIndex() == 1

    main_window.viewer.selectionChanged.emit(_sel("detected"))
    assert main_window.peaks_table_panel._tabs.currentIndex() == 0


def test_mirroring_a_large_multi_selection_is_one_pass(qtbot):
    """Ctrl+A on the image hands the table every peak on the frame. Each
    one used to rescan the whole model and select its row on its own,
    so the work grew with the square of the peak count; now the model is
    read once and the rows are selected in one go."""
    import numpy as np

    from mlgidgui.file_model import PeakTable

    n = 300
    zeros = np.zeros(n, dtype=float)
    table = PeakTable(
        q_xy=zeros, q_z=zeros, angle=np.full(n, 45.0),
        radius=1.0 + np.arange(n) * 1e-3,
        angle_width=np.ones(n), radius_width=np.full(n, 0.1),
        is_ring=np.zeros(n, dtype=bool), ids=np.arange(n),
        score=np.ones(n), amplitude=zeros,
    )
    panel = PeaksTablePanel()
    qtbot.addWidget(panel)
    panel.set_frame_peaks(0, table, None, [])

    model = panel._detected_model
    reads = []
    real_item = model.item
    model.item = lambda *args: reads.append(args) or real_item(*args)
    changes = []
    panel._detected_table.selectionModel().selectionChanged.connect(
        lambda *_: changes.append(1)
    )
    sels = [
        SelectedPeak(
            kind="detected", frame=0, peak_id=int(i),
            radius=1.0, angle=45.0, radius_width=0.1, angle_width=1.0,
        )
        for i in (7, *range(n))
    ]
    panel.set_external_selections(sels)
    n_reads, n_changes = len(reads), len(changes)

    selection_model = panel._detected_table.selectionModel()
    proxy = panel._detected_proxy
    picked = sorted(
        int(model.item(proxy.mapToSource(i).row(), 0).text())
        for i in selection_model.selectedRows()
    )
    assert picked == list(range(n))
    current = proxy.mapToSource(selection_model.currentIndex())
    assert model.item(current.row(), 0).text() == "7"
    assert n_reads <= n
    assert n_changes == 1
