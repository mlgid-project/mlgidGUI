"""Extra peak tables registered as display layers.

A file's analysis group can hold peak tables mlgidLAB knows nothing
about. Registering one draws it, lets a box be picked and nudged, and
must do **nothing else** -- the layer may not leak into the Peaks table,
export, fitting, matching or tracking.

The mechanism is the ``list:`` kind prefix: it can never equal a
built-in kind, so every consumer that names its kinds explicitly
excludes it for free. What these tests actually pin is the handful of
places that were widened on purpose, and the two gates that had to be
closed by hand.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QPointF, QSettings, Qt

from mlgidlab import file_model, peak_lists
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui

EXTRA = "Br_peaks"


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is app-wide, so it must not leak between tests."""
    QSettings().remove(peak_lists.SETTINGS_KEY)
    QSettings().remove("overlayPens")
    yield
    QSettings().remove(peak_lists.SETTINGS_KEY)
    QSettings().remove("overlayPens")


def _add_extra_table(path, name=EXTRA, n=2, entry="entry_0000", frame=0):
    """Write a peak-shaped table alongside the frame's own tables."""
    with h5py.File(path, "r+") as f:
        group = f[f"{entry}/{file_model.ANALYSIS_REL}/"
                  f"{file_model.FRAME_KEY_FMT.format(frame)}"]
        source = np.asarray(group["detected_peaks"][()])
        rows = np.zeros(n, dtype=source.dtype)
        for i in range(n):
            rows[i] = source[min(i, len(source) - 1)]
            rows["id"][i] = 900 + i
            rows["radius"][i] = 2.5 + 0.4 * i
            rows["angle"][i] = 20.0 + 10.0 * i
            rows["radius_width"][i] = 0.2
            rows["angle_width"][i] = 6.0
            rows["is_ring"][i] = False
        group.create_dataset(name, data=rows)
    return name


def _register(dataset=EXTRA, label="Bromine", treat_as="detected"):
    peak_lists.save_specs([
        peak_lists.PeakListSpec(dataset=dataset, label=label,
                                treat_as=treat_as)
    ])


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


# --- the registry -----------------------------------------------------


def test_a_list_kind_can_never_be_a_builtin_kind():
    """The whole no-spill mechanism in one assertion."""
    spec = peak_lists.PeakListSpec(dataset=EXTRA, label="Bromine")
    assert spec.kind == "list:Br_peaks"
    assert spec.kind not in ("detected", "fitted", "manual", "matched")
    assert peak_lists.is_list_kind(spec.kind)
    for builtin in ("detected", "fitted", "manual", "matched"):
        assert not peak_lists.is_list_kind(builtin)
    assert peak_lists.dataset_for_kind(spec.kind) == EXTRA
    assert peak_lists.dataset_for_kind("detected") == ""


def test_the_prefix_matches_the_one_the_reader_writes():
    """``file_model`` keeps its own copy to stay free of the Qt import."""
    assert file_model._LIST_PREFIX == peak_lists.KIND_PREFIX


def test_the_registry_round_trips():
    specs = [
        peak_lists.PeakListSpec(EXTRA, "Bromine", "detected"),
        peak_lists.PeakListSpec("ref_peaks", "Reference", "fitted"),
    ]
    peak_lists.save_specs(specs)
    assert peak_lists.load_specs() == specs


def test_an_unreadable_registry_does_not_stop_the_app():
    """It is a settings file: hand-edited and future-version values
    both have to degrade rather than raise."""
    for raw in ("not json", "[1, 2, 3]", '{"a": 1}', '[{"label": "no name"}]'):
        QSettings().setValue(peak_lists.SETTINGS_KEY, raw)
        assert peak_lists.load_specs() == []


def test_the_registry_refuses_tables_the_app_already_draws():
    """Registering ``detected_peaks`` would draw it twice."""
    peak_lists.save_specs([
        peak_lists.PeakListSpec("detected_peaks", "Nope"),
        peak_lists.PeakListSpec("matched_segments_0000", "Nope"),
        peak_lists.PeakListSpec(EXTRA, "Bromine"),
    ])
    assert [s.dataset for s in peak_lists.load_specs()] == [EXTRA]


def test_a_duplicate_dataset_is_kept_once():
    peak_lists.save_specs([
        peak_lists.PeakListSpec(EXTRA, "First"),
        peak_lists.PeakListSpec(EXTRA, "Second"),
    ])
    assert [s.label for s in peak_lists.load_specs()] == ["First"]


# --- reading ----------------------------------------------------------


def test_list_peak_tables_offers_only_what_is_registerable(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    _add_extra_table(path)
    with h5py.File(path, "r+") as f:
        group = f["entry_0000/data/analysis/frame00000"]
        # A non-peak-shaped dataset must not be offered.
        group.create_dataset("some_curve", data=np.arange(10.0))

    with h5py.File(path, "r") as f:
        found = file_model.list_peak_tables(f, "entry_0000", 0)

    assert found == [EXTRA]
    assert "detected_peaks" not in found
    assert "fitted_peaks" not in found
    assert "fitted_peaks_errors" not in found
    assert "some_curve" not in found


def test_an_extra_table_reads_under_its_list_key(synthetic_nexus_with_peaks):
    path = synthetic_nexus_with_peaks
    _add_extra_table(path, n=2)
    tables = file_model.load_peaks(path, "entry_0000", 0, (EXTRA,))
    assert len(tables[f"list:{EXTRA}"]) == 2
    # ...and a frame without it yields None rather than raising, so a
    # list registered for one project is inert in every other file.
    assert file_model.load_peaks(
        path, "entry_0000", 1, (EXTRA,)
    )[f"list:{EXTRA}"] is None


# --- the layer --------------------------------------------------------


def test_a_registered_list_renders_and_toggles(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    window._apply_peak_list_specs()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    assert kind in window.viewer._list_items
    assert len(window.viewer._frame_peaks[0][kind]) == 2
    assert kind in window._overlay_checks

    window._overlay_checks[kind].setChecked(False)
    assert window.viewer._list_items[kind].isVisible() is False
    window._overlay_checks[kind].setChecked(True)
    assert window.viewer._list_items[kind].isVisible() is True


def test_a_list_takes_a_pen_of_its_own(
    main_window, synthetic_nexus_with_peaks,
):
    """It starts from the preset of the flavour it is treated as, and
    the pen editor works on it exactly as on the built-in layers."""
    from mlgidlab.viewer_styles import OVERLAY_STYLE

    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register(treat_as="fitted")
    window._apply_peak_list_specs()
    kind = f"list:{EXTRA}"

    assert window.viewer.overlay_pen(kind) == OVERLAY_STYLE["fitted"]
    window.viewer.set_overlay_pen(kind, {"color": "#00ff00", "width": 3.0})
    assert window.viewer.overlay_pen(kind)["color"] == "#00ff00"
    assert window.viewer._list_items[kind]._pen.widthF() == pytest.approx(3.0)


def test_a_list_box_selects_and_reads_through_its_flavour(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register(treat_as="detected")
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    window.viewer._mode = "polar"
    window.viewer._on_select_at(
        QPointF(float(table.radius[0]), float(table.angle[0]))
    )

    sel = window.viewer.selected_peak
    assert sel is not None and sel.kind == kind
    assert int(sel.peak_id) == int(table.ids[0])
    # The panel reads it through its treat-as flavour and NAMES it, so
    # it is never unclear which table is on screen.
    panel = window.parameter_panel
    assert panel._detected_section_label.text() == "Bromine"
    assert panel._form.isRowVisible(panel._row_detected_header)


def test_a_drag_writes_to_that_dataset_and_no_other(
    main_window, synthetic_nexus_with_peaks,
):
    """The assertion that proves the dataset routing: read it back."""
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    session = _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    peak_id = int(table.ids[0])
    detected_before = [
        float(r) for r in
        file_model.load_peaks(session.temp_path, "entry_0000", 0)["detected"].radius
    ]

    window.viewer._apply_file_geom(0, kind, peak_id, (3.21, 44.0, 0.3, 8.0))

    after = file_model.load_peaks(
        session.temp_path, "entry_0000", 0, (EXTRA,)
    )
    moved = after[kind]
    idx = int(np.where(moved.ids == peak_id)[0][0])
    assert float(moved.radius[idx]) == pytest.approx(3.21)
    assert float(moved.angle[idx]) == pytest.approx(44.0)
    # ...and the table the app owns is untouched.
    assert [float(r) for r in after["detected"].radius] == pytest.approx(
        detected_before
    )


# --- no spill ---------------------------------------------------------


def test_the_peak_actions_are_all_refused_for_a_list_box(
    main_window, synthetic_nexus_with_peaks,
):
    """Add-to-detected / Add-to-fitted exclude it for free (they name
    their kinds); Delete does NOT, so it is the one closed by hand."""
    from mlgidlab.image_viewer import SelectedPeak

    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    window.viewer._set_selected(SelectedPeak(
        kind=kind, frame=0, peak_id=int(table.ids[0]),
        radius=float(table.radius[0]), angle=float(table.angle[0]),
        radius_width=float(table.radius_width[0]),
        angle_width=float(table.angle_width[0]),
        score=float(table.score[0]),
    ))

    panel = window.parameter_panel
    assert panel.btn_add_detected.isEnabled() is False
    assert panel.btn_add_fitted.isEnabled() is False
    assert panel.btn_delete_peak.isEnabled() is False


def test_the_delete_handler_is_a_no_op_for_a_list_box(
    main_window, synthetic_nexus_with_peaks,
):
    """The keyboard route bypasses the disabled button, and without the
    guard it would fall through to the matched branch and hand
    mlgidbase a kind it has never heard of."""
    from mlgidlab.image_viewer import SelectedPeak

    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    session = _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    before = session.temp_path.read_bytes()

    window._on_delete_peak_requested(SelectedPeak(
        kind=kind, frame=0, peak_id=int(table.ids[0]),
        radius=float(table.radius[0]), angle=float(table.angle[0]),
        radius_width=float(table.radius_width[0]),
        angle_width=float(table.angle_width[0]),
    ))

    assert session.temp_path.read_bytes() == before


def test_the_peaks_table_never_grows_a_tab_for_a_list(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    tabs = window.peaks_table_panel._tabs
    labels = {tabs.tabText(i) for i in range(tabs.count())}
    assert not any("Bromine" in t or EXTRA in t for t in labels)


def test_ctrl_a_and_ctrl_click_do_not_reach_a_list_layer(
    main_window, synthetic_nexus_with_peaks,
):
    """Both are positive lists of detected/fitted, so this needs no
    code -- which is exactly why it is worth a test."""
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    viewer = window.viewer
    viewer._mode = "polar"
    viewer._select_all_of_kind_on_frame("detected")
    assert all(s.kind == "detected" for s in viewer.selected_peaks())

    table = viewer._frame_peaks[0][kind]
    viewer._on_select_at(
        QPointF(float(table.radius[0]), float(table.angle[0])),
        Qt.KeyboardModifier.ControlModifier,
    )
    assert all(s.kind != kind for s in viewer.selected_peaks())


def test_unregistering_a_list_takes_its_layer_away(
    main_window, synthetic_nexus_with_peaks,
):
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()
    kind = f"list:{EXTRA}"
    assert kind in window.viewer._list_items

    peak_lists.save_specs([])
    window._apply_peak_list_specs()

    assert kind not in window.viewer._list_items
    assert kind not in window._overlay_checks
    assert window._peak_list_rows_layout.count() == 0


def test_unregistering_survives_a_view_box_that_refuses_the_item(
    qtbot, monkeypatch,
):
    """Taking the item off the view box is best-effort: when that raises
    (its C++ half already gone) the failure is logged and the layer is
    still dropped, rather than the handler dying on its own log call."""
    from mlgidlab.image_viewer import GIWAXSImageViewer

    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    spec = peak_lists.PeakListSpec(dataset=EXTRA, label="Bromine")
    viewer.set_peak_list_specs([spec])
    assert spec.kind in viewer._list_items

    def _refuse(item):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(viewer._plot.getViewBox(), "removeItem", _refuse)
    viewer.set_peak_list_specs([])
    assert spec.kind not in viewer._list_items


# --- the off-thread reads ---------------------------------------------
#
# Peaks for the frame an entry lands on are read by a WORKER, not by
# ``_load_frame_peaks``, and the GUI then marks that frame loaded. So a
# worker read that omits the registered lists does not merely arrive
# late -- it arrives final: the layer stays blank for the rest of the
# session, and re-registering it in Settings is the only way back. Both
# worker paths must therefore ask for the lists, and the install must
# refuse an overlay dict that does not carry them.


def test_the_entry_worker_reads_the_registered_lists(
    qtbot, synthetic_nexus_with_peaks,
):
    """The entry-switch path. This is the read that fills the landed
    frame's overlays on every switch after the first."""
    from mlgidlab.workers import EntryLoadWorker

    _add_extra_table(synthetic_nexus_with_peaks)
    worker = EntryLoadWorker()
    got: list = []
    worker.loaded.connect(lambda *a: got.append(a))
    worker.load(str(synthetic_nexus_with_peaks), "entry_0000", 1, (EXTRA,))

    _rid, _entry, source, overlays = got[0]
    try:
        assert overlays is not None
        frame, peaks, _matched = overlays
        assert frame == 0
        assert len(peaks[f"list:{EXTRA}"]) == 2
    finally:
        source.release()


def test_the_open_worker_reads_the_registered_lists(
    qtbot, synthetic_nexus_with_peaks,
):
    """The open path's prewarm, same failure mode."""
    from mlgidlab.workers import CopyWorker

    _add_extra_table(synthetic_nexus_with_peaks)
    worker = CopyWorker(synthetic_nexus_with_peaks, (EXTRA,))
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    try:
        _frame, peaks, _matched = got["prewarm_overlays"]
        assert len(peaks[f"list:{EXTRA}"]) == 2
    finally:
        got["prewarm"][1].release()
        got["session"].close()


def test_an_overlay_read_that_predates_the_list_is_not_trusted(
    main_window, synthetic_nexus_with_peaks,
):
    """The guard behind the two tests above.

    A read can always be in flight when the registry changes, so the
    install site checks the dict rather than assuming the worker was
    current -- a missing key means "never asked for", which is why the
    fallback re-reads instead of rendering an empty layer.
    """
    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register()
    _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()
    kind = f"list:{EXTRA}"

    stale = {"detected": None, "fitted": None}
    assert window._overlays_have_registered_lists(stale) is False
    window.viewer.set_peaks(0, dict(stale))
    window._loaded_peak_frames = set()

    window._install_stack_into_viewer(
        "entry_0000", window.viewer._stack, preserve_view=True,
        overlays=(0, stale, []),
    )

    assert len(window.viewer._frame_peaks[0][kind]) == 2


# --- scores on a detected-flavoured list ------------------------------
#
# A list registered as "detected" is a list of detections, so its score
# column is exactly the thing the editor exists for -- relabelling a
# validation set is why such a table gets registered at all. The write
# still lands in that list's own dataset, so this widens what a layer
# can do to ITSELF without widening what it touches.


def test_a_detected_flavoured_list_takes_a_score_edit(
    main_window, synthetic_nexus_with_peaks,
):
    from mlgidlab.image_viewer import SelectedPeak

    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register(treat_as="detected")
    session = _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    detected_before = file_model.load_peaks(
        session.temp_path, "entry_0000", 0
    )["detected"]
    window.viewer._set_selected(SelectedPeak(
        kind=kind, frame=0, peak_id=int(table.ids[0]),
        radius=float(table.radius[0]), angle=float(table.angle[0]),
        radius_width=float(table.radius_width[0]),
        angle_width=float(table.angle_width[0]),
        score=float(table.score[0]),
    ))
    panel = window.parameter_panel
    assert panel._score_stack.currentWidget() is panel._score_editor

    panel._score_preset_buttons[2].click()  # "Low" = 0.1

    on_disk = file_model.load_peaks(
        session.temp_path, "entry_0000", 0, (EXTRA,)
    )
    assert float(on_disk[kind].score[0]) == pytest.approx(0.1)
    # ...and nowhere else. The list is still only itself.
    after = file_model.load_peaks(session.temp_path, "entry_0000", 0)
    assert [float(s) for s in after["detected"].score] == pytest.approx(
        [float(s) for s in detected_before.score]
    )
    # Undo restores it, same as for the built-in layer.
    window.viewer.undo_last_action()
    reread = file_model.load_peaks(
        session.temp_path, "entry_0000", 0, (EXTRA,)
    )
    assert float(reread[kind].score[0]) == pytest.approx(
        float(table.score[0])
    )


def test_a_fitted_flavoured_list_does_not_take_a_score_edit(
    main_window, synthetic_nexus_with_peaks,
):
    """The flavour is the whole switch: a table the user calls fitted
    shows the fitted block, which has no score row to edit."""
    from mlgidlab.image_viewer import SelectedPeak

    window = main_window
    _add_extra_table(synthetic_nexus_with_peaks)
    _register(treat_as="fitted")
    session = _open(window, synthetic_nexus_with_peaks)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    sel = SelectedPeak(
        kind=kind, frame=0, peak_id=int(table.ids[0]),
        radius=float(table.radius[0]), angle=float(table.angle[0]),
        radius_width=float(table.radius_width[0]),
        angle_width=float(table.angle_width[0]),
        score=float(table.score[0]),
    )
    window.viewer._set_selected(sel)
    panel = window.parameter_panel
    assert panel._score_stack.currentWidget() is not panel._score_editor

    # The handler refuses it too -- the editor is not the only route in
    # (a stale selection, a keyboard preset).
    window._on_score_edit_requested(0.1)
    on_disk = file_model.load_peaks(
        session.temp_path, "entry_0000", 0, (EXTRA,)
    )
    assert float(on_disk[kind].score[0]) == pytest.approx(
        float(table.score[0])
    )


def test_a_list_without_a_score_column_is_not_offered_the_editor(
    main_window, synthetic_nexus_with_peaks,
):
    """``update_peak_row`` writes only fields the dtype has, so a table
    with no score column would take the edit in memory and drop it on
    disk. Such a table has no score to show either, so the row goes
    rather than reading a synthesised 0.000."""
    from mlgidlab.image_viewer import SelectedPeak

    window = main_window
    path = synthetic_nexus_with_peaks
    with h5py.File(path, "r+") as f:
        group = f[f"entry_0000/{file_model.ANALYSIS_REL}/"
                  f"{file_model.FRAME_KEY_FMT.format(0)}"]
        source = np.asarray(group["detected_peaks"][()])
        keep = [n for n in source.dtype.names if n != "score"]
        rows = np.zeros(2, dtype=[(n, source.dtype[n]) for n in keep])
        for n in keep:
            rows[n] = source[n][:2]
        group.create_dataset(EXTRA, data=rows)
    _register(treat_as="detected")
    _open(window, path)
    window._apply_peak_list_specs()

    kind = f"list:{EXTRA}"
    table = window.viewer._frame_peaks[0][kind]
    assert table.has_score is False

    window.viewer._set_selected(window.viewer._row_selection(kind, table, 0, 0))
    panel = window.parameter_panel
    assert window.viewer.selected_peak.score is None
    assert panel._score_stack.currentWidget() is not panel._score_editor
    assert panel._form.isRowVisible(panel._row_score) is False

    sel = SelectedPeak(
        kind=kind, frame=0, peak_id=int(table.ids[0]),
        radius=float(table.radius[0]), angle=float(table.angle[0]),
        radius_width=float(table.radius_width[0]),
        angle_width=float(table.angle_width[0]),
    )
    window.viewer._set_selected(sel)
    window._on_score_edit_requested(0.1)  # must not raise, must not write
    with h5py.File(window.session.temp_path, "r") as f:
        group = f[f"entry_0000/{file_model.ANALYSIS_REL}/"
                  f"{file_model.FRAME_KEY_FMT.format(0)}"]
        assert "score" not in group[EXTRA].dtype.names
