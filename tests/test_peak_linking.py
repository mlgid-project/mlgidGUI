"""Linking a fitted peak to the detected peak it came from, by id.

detected_peaks and fitted_peaks are independent tables: each assigns its
own ``max(id) + 1`` and nothing records which fit belongs to which
detection. With the link setting on (the default), the two are paired by
carrying the *same* id — one fitted peak per detected peak, refitting
replaces rather than appends, and deleting the detection takes its fit
with it.

This file starts at the storage layer, which is where the pairing has to
be expressible at all: ``file_model``'s add helpers had no way to write a
row under a chosen id, and the snapshot helper had no way to report the
id a row had. (A pipeline fitting run used to assign fitted ids
positionally and needed a re-key pass to keep the pairs; pygidfit 0.1.8
copies the detected ids itself, so the pass is gone.)

No Qt in this section — it runs in the backend-less CI environment too.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mlgidlab import file_model

ENTRY = "entry_0000"

_FIT_DTYPE = np.dtype(
    [
        ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
        ("radius", "f4"), ("radius_width", "f4"), ("q_z", "f4"),
        ("q_xy", "f4"), ("theta", "f4"), ("score", "f4"),
        ("A", "f4"), ("B", "f4"), ("C", "f4"),
        ("is_ring", "bool"), ("is_cut_qz", "bool"), ("is_cut_qxy", "bool"),
        ("visibility", "i4"), ("id", "i4"),
    ]
)
_VLEN_INT = h5py.vlen_dtype(np.int32)
_MATCHED_DTYPE = np.dtype(
    [
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"), ("peak_list", _VLEN_INT),
    ]
)


def _rows(ids: list[int]) -> np.ndarray:
    arr = np.zeros(len(ids), dtype=_FIT_DTYPE)
    arr["id"] = ids
    # radius doubles as a position marker so a row stays identifiable
    # after its id changes.
    arr["radius"] = [float(i + 1) for i in range(len(ids))]
    arr["angle"] = 45.0
    arr["angle_width"] = 5.0
    arr["radius_width"] = 0.2
    return arr


def _build(
    tmp_path: Path,
    detected_ids: list[int],
    fitted_ids: list[int],
    *,
    peak_list: list[int] | None = None,
    frames: int = 1,
) -> Path:
    """One-entry NeXus file carrying the given detected / fitted ids on
    every frame, optionally with one matched solution row."""
    path = tmp_path / "linked.h5"
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group(f"{ENTRY}/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset("img_gid_q", data=np.zeros((frames, 8, 8), np.float32))
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        for fr in range(frames):
            g = data.create_group(f"analysis/frame{fr:05d}", track_order=True)
            g.create_dataset("detected_peaks", data=_rows(detected_ids))
            g.create_dataset("fitted_peaks", data=_rows(fitted_ids))
            g.create_dataset("fitted_peaks_errors", data=_rows([]))
            if peak_list is not None:
                m = np.zeros(1, dtype=_MATCHED_DTYPE)
                m["CIF"] = b"phase.cif"
                m["probability"] = 0.9
                m["peak_list"][0] = np.asarray(peak_list, dtype=np.int32)
                g.create_dataset("matched_segments_0000", data=m)
    return path


def _ids(path: Path, kind: str, frame: int = 0) -> list[int]:
    ds = (
        f"{ENTRY}/data/analysis/frame{frame:05d}/{kind}_peaks"
    )
    with h5py.File(path, "r") as f:
        return [int(v) for v in f[ds][()]["id"]]


# -- Writing a row under a chosen id ------------------------------------

def test_a_fitted_row_can_be_written_under_a_given_id(tmp_path):
    path = _build(tmp_path, [0, 1, 2], [0, 1])
    new_id = file_model.add_fitted_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        amplitude=5.0, peak_id=7,
    )
    assert new_id == 7
    assert _ids(path, "fitted") == [0, 1, 7]


def test_a_detected_row_can_be_written_under_a_given_id(tmp_path):
    path = _build(tmp_path, [0, 1, 2], [0, 1])
    new_id = file_model.add_detected_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        peak_id=9,
    )
    assert new_id == 9
    assert _ids(path, "detected") == [0, 1, 2, 9]


def test_without_an_explicit_id_the_old_rule_still_applies(tmp_path):
    """max(id) + 1, unchanged — the link is opt-out, and everything that
    does not ask for an id must behave exactly as before."""
    path = _build(tmp_path, [0, 1, 2], [3, 8])
    new_id = file_model.add_fitted_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        amplitude=5.0,
    )
    assert new_id == 9


def test_the_snapshot_can_carry_the_id_it_had(tmp_path):
    """Undo of a linked delete has to put the row back under its old id,
    or the restored pair is no longer a pair."""
    path = _build(tmp_path, [0, 1, 2], [4, 5])
    plain = file_model.read_peak_rows(path, ENTRY, 0, "fitted", [5])
    assert plain and "id" not in plain[0]
    with_ids = file_model.read_peak_rows(
        path, ENTRY, 0, "fitted", [5], with_ids=True,
    )
    assert with_ids[0]["id"] == 5
    # The rest of the dict is unchanged, so it still splats into the add
    # helper once the id is popped off.
    stripped = dict(with_ids[0])
    stripped.pop("id")
    assert stripped == plain[0]


# -- The two settings ---------------------------------------------------

@pytest.fixture
def clean_link_settings():
    """Remove both keys before and after, so one test's choice cannot
    leak into another's default (the whole point of these tests is what
    happens when nothing is stored)."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.setOrganizationName("mlgidLAB")
        app.setApplicationName("mlgidLAB")
    keys = ("peakLinkFittedToDetected", "peakLinkFittedDeleteRemovesDetected")
    s = QSettings()
    for k in keys:
        s.remove(k)
    s.sync()
    yield s
    for k in keys:
        s.remove(k)
    s.sync()


@pytest.mark.gui
def test_the_link_is_on_and_the_reverse_cascade_off_by_default(
    clean_link_settings,
):
    from mlgidlab import peak_link

    assert peak_link.link_enabled() is True
    assert peak_link.reverse_delete_enabled() is False


@pytest.mark.gui
def test_the_settings_round_trip(clean_link_settings):
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", False)
    s.sync()
    assert peak_link.link_enabled() is False
    s.setValue("peakLinkFittedToDetected", True)
    s.setValue("peakLinkFittedDeleteRemovesDetected", True)
    s.sync()
    assert peak_link.reverse_delete_enabled() is True


@pytest.mark.gui
def test_the_reverse_cascade_needs_the_link(clean_link_settings):
    """Without paired ids there is no partner to delete, so the reverse
    setting cannot act on its own however it is stored."""
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", False)
    s.setValue("peakLinkFittedDeleteRemovesDetected", True)
    s.sync()
    assert peak_link.reverse_delete_enabled() is False


@pytest.mark.gui
@pytest.mark.parametrize(
    "stored,expected",
    [("true", True), ("false", False), (True, True), (False, False)],
)
def test_a_default_on_setting_survives_the_ini_backend(
    clean_link_settings, stored, expected,
):
    """QSettings hands back the string "false" on Linux and a real bool
    on macOS. The repo's older idiom only reads back a default-off
    setting correctly, which would have made this one unturnoffable."""
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", stored)
    s.sync()
    assert peak_link.link_enabled() is expected


@pytest.mark.gui
def test_the_settings_dialog_shows_and_saves_both(main_window, clean_link_settings):
    from mlgidlab.main_window_dialogs import _SettingsDialog

    dlg = _SettingsDialog(main_window)
    assert dlg._chk_peak_link.isChecked()
    assert not dlg._chk_peak_link_reverse.isChecked()
    # The dependent box greys out with the link rather than lying about
    # what it would do.
    assert dlg._chk_peak_link_reverse.isEnabled()
    dlg._chk_peak_link.setChecked(False)
    assert not dlg._chk_peak_link_reverse.isEnabled()

    dlg._chk_peak_link.setChecked(True)
    dlg._chk_peak_link_reverse.setChecked(True)
    dlg.save_to_qsettings()
    clean_link_settings.sync()

    from mlgidlab import peak_link
    assert peak_link.link_enabled() is True
    assert peak_link.reverse_delete_enabled() is True
    # A fresh dialog reflects what was stored.
    assert _SettingsDialog(main_window)._chk_peak_link_reverse.isChecked()


# -- Fitting: the fit takes its detected peak's id ----------------------
#
# The fixture file carries detected ids [0, 1, 2] and fitted ids [0, 1]
# on frame 0, so detected 2 is unfitted and detected 0/1 already have a
# fit — exactly the two cases the link has to tell apart.

from mlgidlab.image_viewer import ManualPeak, SelectedPeak  # noqa: E402
from mlgidlab.session import NexusSession  # noqa: E402

_FIT_ROW = {
    "radius": 2.75, "radius_width": 0.11,
    "angle": 33.0, "angle_width": 6.0,
    "amplitude": 42.0, "theta": 0.0, "A": 1.0, "B": 0.0, "C": 1.0,
}


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _confirm(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window.session.temp_path


def _table(path, kind, frame=0):
    return file_model.load_peaks(path, ENTRY, frame)[kind]


def _table_ids(path, kind, frame=0) -> list[int]:
    t = _table(path, kind, frame)
    return [] if t is None else [int(v) for v in t.ids]


def _select_detected(window, index: int, frame: int = 0) -> SelectedPeak:
    det = (window.viewer._frame_peaks.get(frame) or {})["detected"]
    sel = SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[index]),
        radius=float(det.radius[index]), angle=float(det.angle[index]),
        radius_width=float(det.radius_width[index]),
        angle_width=float(det.angle_width[index]),
        is_ring=bool(det.is_ring[index]),
        score=float(det.score[index]),
        amplitude=float(det.amplitude[index]),
    )
    window.viewer._set_selected(sel)
    return sel


def _stub_fit(window, monkeypatch, row=None):
    """Skip pygidfit; the link is about which id the row lands under.

    Stubs both ways a 2D fit is obtained: the blocking one the quick
    paths use (``_build_fitted_row_2d``) and the background one the
    Add to fitted button and the live preview use
    (``_prepare_pygidfit_fit``). The preview cache is cleared so a
    second stub with other numbers is not answered from the first.
    """
    from types import SimpleNamespace

    fit = SimpleNamespace(**dict(row or _FIT_ROW))
    monkeypatch.setattr(
        window, "_build_fitted_row_2d",
        lambda sel, entry, frame, **_kw: dict(row or _FIT_ROW),
    )
    monkeypatch.setattr(
        window, "_prepare_pygidfit_fit",
        lambda sel, entry, frame: ((lambda: (fit, None)), None),
    )
    window._2d_preview_cache.clear()


def _add_to_fitted(window, timeout: float = 15.0) -> None:
    """Press Add to fitted and wait for the background fit to commit."""
    import time
    from PySide6.QtWidgets import QApplication

    window._on_add_to_fitted()
    deadline = time.time() + timeout
    while window._add_fitted_wait is not None and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert window._add_fitted_wait is None, "Add to fitted never finished"


@pytest.mark.gui
def test_a_fit_is_stored_under_its_detected_peaks_id(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)  # detected id 2, not yet fitted
    _add_to_fitted(main_window)
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_refitting_replaces_the_previous_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """The whole point: one fitted peak per detected peak, however many
    times the user refits it with different borders."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)  # detected id 0, already fitted
    _add_to_fitted(main_window)
    assert _table_ids(path, "fitted") == [1, 0]  # 0 rewritten at the end
    fit = _table(path, "fitted")
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(_FIT_ROW["radius"])
    # And again, with different borders — still one row for peak 0.
    _stub_fit(main_window, monkeypatch, {**_FIT_ROW, "radius": 2.9})
    _select_detected(main_window, 0)
    _add_to_fitted(main_window)
    assert sorted(_table_ids(path, "fitted")) == [0, 1]
    fit = _table(path, "fitted")
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(2.9)


@pytest.mark.gui
def test_with_the_link_off_a_refit_appends_as_before(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)
    _add_to_fitted(main_window)
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_undoing_a_refit_brings_the_previous_fit_back(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    before = _table(path, "fitted")
    old_radius = float(before.radius[0])
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)
    _add_to_fitted(main_window)

    main_window.viewer.undo_last_action()
    fit = _table(path, "fitted")
    assert sorted(int(v) for v in fit.ids) == [0, 1]
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(old_radius)


@pytest.mark.gui
def test_a_hand_drawn_box_gets_a_detected_partner(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """A manual box has no detected peak behind it, so one is written
    first and the fit pairs to it — every fitted peak has a partner."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    manual = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    main_window.viewer.add_manual_peak(0, manual)
    main_window.viewer._set_selected(SelectedPeak.from_manual(manual, 0))
    _add_to_fitted(main_window)

    assert _table_ids(path, "detected") == [0, 1, 2, 3]
    assert _table_ids(path, "fitted") == [0, 1, 3]

    # One Ctrl+Z removes both halves.
    main_window.viewer.undo_last_action()
    assert _table_ids(path, "detected") == [0, 1, 2]
    assert _table_ids(path, "fitted") == [0, 1]


@pytest.mark.gui
def test_with_the_link_off_a_hand_drawn_box_adds_no_detected_row(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    manual = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    main_window.viewer.add_manual_peak(0, manual)
    main_window.viewer._set_selected(SelectedPeak.from_manual(manual, 0))
    _add_to_fitted(main_window)
    assert _table_ids(path, "detected") == [0, 1, 2]
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_a_batch_fit_pairs_every_fit_with_its_detection(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Three detected peaks, two of them already fitted → three fits,
    one per detection, not five rows."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Stub:
        radius: float = 2.75
        radius_width: float = 0.11
        angle: float = 33.0
        angle_width: float = 6.0
        amplitude: float = 42.0
        A: float = 1.0
        B: float = 0.0
        C: float = 1.0
        theta: float = 0.0

    path = _open(main_window, synthetic_nexus_with_peaks)
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_Stub(), None),
    )
    main_window.viewer._select_all_of_kind_on_frame("detected")
    main_window._on_batch_fit_2d()
    assert sorted(_table_ids(path, "fitted")) == [0, 1, 2]

    main_window.viewer.undo_last_action()
    assert sorted(_table_ids(path, "fitted")) == [0, 1]


# -- Deleting one side of the pair --------------------------------------

@pytest.mark.gui
def test_deleting_a_detected_peak_takes_its_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "detected") == [1, 2]
    assert _table_ids(path, "fitted") == [1]


@pytest.mark.gui
def test_deleting_a_fitted_peak_keeps_the_detection_by_default(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Discarding a fit says the prediction was wrong, not the
    detection — so the reverse cascade is opt-in."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    fit = _table(path, "fitted")
    sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        is_ring=False, score=0.0, amplitude=0.0,
    )
    main_window.viewer._set_selected(sel)
    main_window._on_delete_peak_requested(sel)
    assert _table_ids(path, "fitted") == [1]
    assert _table_ids(path, "detected") == [0, 1, 2]

    # With the second setting on, the detection goes too.
    clean_link_settings.setValue("peakLinkFittedDeleteRemovesDetected", True)
    clean_link_settings.sync()
    fit = _table(path, "fitted")
    sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        is_ring=False, score=0.0, amplitude=0.0,
    )
    main_window.viewer._set_selected(sel)
    main_window._on_delete_peak_requested(sel)
    assert _table_ids(path, "fitted") == []
    assert _table_ids(path, "detected") == [0, 2]


@pytest.mark.gui
def test_undo_of_a_linked_delete_restores_both_under_their_ids(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Restoring under fresh ids would hand back two rows that are no
    longer a pair, so the undo would not have undone."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    main_window.viewer.undo_last_action()
    assert sorted(_table_ids(path, "detected")) == [0, 1, 2]
    assert sorted(_table_ids(path, "fitted")) == [0, 1]

    main_window.viewer.redo_last_action()
    assert sorted(_table_ids(path, "detected")) == [1, 2]
    assert sorted(_table_ids(path, "fitted")) == [1]


@pytest.mark.gui
def test_with_the_link_off_a_detected_delete_leaves_the_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "detected") == [1, 2]
    assert _table_ids(path, "fitted") == [0, 1]


# -- Clearing ------------------------------------------------------------

@pytest.mark.gui
def test_clearing_detected_also_clears_fitted_when_linked(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._action_clear_file_peaks("detected", "entry")
    assert _table_ids(path, "detected") == []
    assert _table_ids(path, "fitted") == []


@pytest.mark.gui
def test_clearing_detected_leaves_fitted_when_unlinked(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._action_clear_file_peaks("detected", "entry")
    assert _table_ids(path, "detected") == []
    assert _table_ids(path, "fitted") == [0, 1]


# -- Matched, and the pipeline ------------------------------------------

@pytest.mark.gui
def test_a_linked_delete_remaps_matched_the_way_a_fitted_delete_does(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """The link gives matched no new semantics. Deleting a detected peak
    removes its fit, and that removal goes through the same
    ``prepare_fitted_delete`` / ``remap_matched_peak_lists`` pair a plain
    fitted delete uses: the structure loses exactly that peak and keeps
    the rest."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    ds_path = f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            m = np.zeros(1, dtype=_MATCHED_DTYPE)
            m["CIF"] = b"phase.cif"
            m["probability"] = 0.9
            m["peak_list"][0] = np.asarray([0, 1], dtype=np.int32)
            f.create_dataset(ds_path, data=m)

    _confirm(monkeypatch)
    # Detected 0 pairs with fitted 0, which sits at position 0.
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "fitted") == [1]
    with h5py.File(path, "r") as f:
        row = f[ds_path][()][0]
    # The legacy positional list [0, 1] was put in id form (fitted ids 0
    # and 1) and id 0 dropped. The structure itself survives with its
    # remaining peak, now stored as id 1.
    assert [int(v) for v in row["peak_list"]] == [1]
    with h5py.File(path, "r") as f:
        assert f[ds_path].attrs["peak_list_kind"] == "id"


# -- Switching the selection between a detected peak and its fit ---------

def _select_fitted(window, index: int, frame: int = 0) -> SelectedPeak:
    fit = (window.viewer._frame_peaks.get(frame) or {})["fitted"]
    sel = window.viewer._row_selection("fitted", fit, index, frame)
    window.viewer._set_selected(sel)
    return sel


@pytest.mark.gui
def test_t_moves_a_detected_selection_to_its_fitted_partner(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    _open(main_window, synthetic_nexus_with_peaks)
    _select_detected(main_window, 1)
    main_window.action_switch_detected_fitted.trigger()
    sel = main_window.viewer.selected_peak
    assert (sel.kind, sel.peak_id) == ("fitted", 1)
    assert sel.radius == pytest.approx(2.5)          # the fitted row's geometry

    main_window.action_switch_detected_fitted.trigger()
    sel = main_window.viewer.selected_peak
    assert (sel.kind, sel.peak_id) == ("detected", 1)
    assert sel.radius == pytest.approx(2.0)


@pytest.mark.gui
def test_t_has_the_shortcut_and_a_menu_entry(main_window):
    assert main_window.action_switch_detected_fitted.shortcut().toString() == "T"
    assert main_window.action_switch_detected_fitted in (
        main_window.menuBar().actions()[1].menu().actions()
    )


@pytest.mark.gui
def test_t_without_a_partner_keeps_the_selection_and_says_why(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    _open(main_window, synthetic_nexus_with_peaks)
    _select_detected(main_window, 2)                  # detected 2 has no fit
    main_window._on_switch_detected_fitted()
    sel = main_window.viewer.selected_peak
    assert (sel.kind, sel.peak_id) == ("detected", 2)
    assert "No fitted partner" in main_window.statusBar().currentMessage()


@pytest.mark.gui
def test_t_does_nothing_with_the_link_off(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    _open(main_window, synthetic_nexus_with_peaks)
    _select_detected(main_window, 0)
    main_window._on_switch_detected_fitted()
    assert main_window.viewer.selected_peak.kind == "detected"
    assert "peak link" in main_window.statusBar().currentMessage()


@pytest.mark.gui
def test_t_needs_a_detected_or_fitted_selection(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    _open(main_window, synthetic_nexus_with_peaks)
    main_window._on_switch_detected_fitted()          # nothing selected
    assert main_window.viewer.selected_peak is None
    assert "select a detected or fitted" in (
        main_window.statusBar().currentMessage()
    )


@pytest.mark.gui
def test_t_switches_every_selected_peak_that_has_a_partner(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    _open(main_window, synthetic_nexus_with_peaks)
    main_window.viewer._select_all_of_kind_on_frame("detected")  # ids 0, 1, 2
    main_window._on_switch_detected_fitted()
    sels = main_window.viewer.selected_peaks()
    assert [(s.kind, s.peak_id) for s in sels] == [("fitted", 0), ("fitted", 1)]
    assert "2 of 3" in main_window.statusBar().currentMessage()

    main_window._on_switch_detected_fitted()          # and back
    sels = main_window.viewer.selected_peaks()
    assert [(s.kind, s.peak_id) for s in sels] == [
        ("detected", 0), ("detected", 1),
    ]



# -- Ctrl+Return runs Add to fitted --------------------------------------

@pytest.mark.gui
def test_ctrl_return_is_bound_to_add_to_fitted(main_window):
    keys = {
        s.toString()
        for s in main_window.action_add_to_fitted_shortcut.shortcuts()
    }
    assert keys == {"Ctrl+Return", "Ctrl+Enter"}


@pytest.mark.gui
def test_the_shortcut_adds_the_selected_detected_peak_to_fitted(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    from PySide6.QtCore import Qt

    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    main_window.show()
    qtbot.waitExposed(main_window)
    main_window.activateWindow()
    qtbot.wait(50)
    _select_detected(main_window, 2)                  # not yet fitted
    main_window.viewer.setFocus()
    qtbot.keyClick(
        main_window, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier,
    )
    qtbot.waitUntil(lambda: _table_ids(path, "fitted") == [0, 1, 2], timeout=10000)


@pytest.mark.gui
@pytest.mark.parametrize("what", ["nothing", "fitted", "several"])
def test_the_shortcut_writes_nothing_and_says_why_when_it_cannot_run(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    what,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    if what == "fitted":
        _select_fitted(main_window, 0)
    elif what == "several":
        main_window.viewer._select_all_of_kind_on_frame("detected")
    main_window._on_add_to_fitted_shortcut()
    assert _table_ids(path, "fitted") == [0, 1]
    assert "Add to fitted" in main_window.statusBar().currentMessage()


@pytest.mark.gui
def test_the_shortcut_says_why_with_no_file_open(main_window):
    main_window._on_add_to_fitted_shortcut()
    assert "open a file" in main_window.statusBar().currentMessage()


# -- An infinite angular width is a ring --------------------------------
#
# pygidfit reports a ring as angle_width = inf (angle NaN). Every path
# that stores a 2D fit types such a row as a ring with the shared ring
# angle, whatever the Save-as-ring checkbox said.

_INF_ROW = {**_FIT_ROW, "angle": float("nan"), "angle_width": float("inf")}


def test_ring_if_infinite_width_is_the_whole_rule():
    inf, nan = float("inf"), float("nan")
    assert file_model.ring_if_infinite_width(nan, inf, False) == (45.0, True)
    assert file_model.ring_if_infinite_width(30.0, inf, False) == (45.0, True)
    assert file_model.ring_if_infinite_width(30.0, 6.0, True) == (45.0, True)
    assert file_model.ring_if_infinite_width(30.0, 6.0, False) == (30.0, False)
    # -inf is not the ring sentinel
    assert file_model.ring_if_infinite_width(30.0, -inf, False) == (30.0, False)


def _write_fitted(path, rows):
    """Replace frame 0's fitted table with ``rows``: (id, angle, aw, is_ring)."""
    with h5py.File(path, "r+") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        arr = np.zeros(len(rows), dtype=ds.dtype)
        for i, (pid, ang, aw, ring) in enumerate(rows):
            arr[i]["id"], arr[i]["angle"] = pid, ang
            arr[i]["angle_width"], arr[i]["is_ring"] = aw, ring
            arr[i]["radius"], arr[i]["radius_width"] = 1.0 + i, 0.2
        del f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        f[f"{ENTRY}/data/analysis/frame00000"].create_dataset(
            "fitted_peaks", data=arr,
        )


def test_the_file_pass_types_only_infinite_width_rows_as_rings(
    synthetic_nexus_with_peaks,
):
    path = synthetic_nexus_with_peaks
    inf = float("inf")
    _write_fitted(path, [
        (0, 20.0, 4.0, False),     # segment: untouched
        (1, float("nan"), inf, False),   # the case: becomes a ring
        (2, 45.0, inf, True),      # already a ring: untouched
        (3, 70.0, 80.0, True),     # ring with a finite width: untouched
    ])
    assert file_model.flag_infinite_width_fitted_as_rings(path, ENTRY) == 1
    fit = _table(path, "fitted")
    assert [bool(v) for v in fit.is_ring] == [False, True, True, True]
    assert float(fit.angle[0]) == pytest.approx(20.0)
    assert float(fit.angle[1]) == pytest.approx(45.0)
    assert float(fit.angle[3]) == pytest.approx(70.0)
    assert file_model.flag_infinite_width_fitted_as_rings(path, ENTRY) == 0


@pytest.mark.gui
def test_a_run_fitting_pass_types_its_infinite_width_rows_as_rings(
    main_window, synthetic_nexus_with_peaks,
):
    from mlgidlab.pipeline import PipelineCommand

    path = _open(main_window, synthetic_nexus_with_peaks)
    with main_window._detached_silx_tree():
        _write_fitted(path, [(0, float("nan"), float("inf"), False)])
        main_window._flag_rings_after_fit(
            PipelineCommand("run_fitting", {"entry": ENTRY, "frame_num": 0})
        )
    fit = _table(path, "fitted")
    assert bool(fit.is_ring[0]) and float(fit.angle[0]) == pytest.approx(45.0)


@pytest.mark.gui
def test_add_to_fitted_2d_stores_an_infinite_width_fit_as_a_ring(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch, _INF_ROW)
    _select_detected(main_window, 2)
    _add_to_fitted(main_window)
    fit = _table(path, "fitted")
    i = list(fit.ids).index(2)
    assert bool(fit.is_ring[i])
    assert float(fit.angle[i]) == pytest.approx(45.0)
    assert not np.isfinite(float(fit.angle_width[i]))
    sel = main_window.viewer.selected_peak          # auto-switched to the fit
    assert sel.kind == "fitted" and sel.is_ring


@pytest.mark.gui
def test_add_to_fitted_2d_leaves_a_finite_fit_a_segment(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)
    _add_to_fitted(main_window)
    fit = _table(path, "fitted")
    assert not bool(fit.is_ring[list(fit.ids).index(2)])


@pytest.mark.gui
def test_the_2d_batch_stores_an_infinite_width_fit_as_a_ring(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Stub:
        radius: float = 2.75
        radius_width: float = 0.11
        angle: float = float("nan")
        angle_width: float = float("inf")
        amplitude: float = 42.0
        A: float = 1.0
        B: float = 0.0
        C: float = 1.0
        theta: float = 0.0

    path = _open(main_window, synthetic_nexus_with_peaks)
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_Stub(), None),
    )
    main_window.viewer._select_all_of_kind_on_frame("detected")
    main_window._on_batch_fit_2d()
    fit = _table(path, "fitted")
    assert [bool(v) for v in fit.is_ring] == [True] * len(fit)
    assert all(float(a) == pytest.approx(45.0) for a in fit.angle)


# -- Add to fitted (2D) does not block the window -----------------------
#
# The click used to run the pygidfit fit inline, which for a peak in a
# badly conditioned peak-on-ring cluster is tens of seconds with the
# window frozen. It now runs in the background and commits on arrival.

def _gated_fit(window, monkeypatch, row=None):
    """A stubbed 2D fit that blocks until the returned event is set.
    Returns ``(gate, calls)``; ``calls`` counts how many fits were
    actually started."""
    import threading
    from types import SimpleNamespace

    gate = threading.Event()
    calls = []
    fit = SimpleNamespace(**dict(row or _FIT_ROW))

    def prepare(sel, entry, frame):
        calls.append(sel.peak_id)

        def job():
            gate.wait(10)
            return fit, None

        return job, None

    monkeypatch.setattr(window, "_prepare_pygidfit_fit", prepare)
    window._2d_preview_cache.clear()
    return gate, calls


@pytest.mark.gui
def test_add_to_fitted_2d_returns_at_once_and_commits_when_the_fit_lands(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    import time

    path = _open(main_window, synthetic_nexus_with_peaks)
    gate, calls = _gated_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)                  # not yet fitted
    t0 = time.time()
    main_window._on_add_to_fitted()
    assert time.time() - t0 < 1.0                     # the click returned
    assert _table_ids(path, "fitted") == [0, 1]       # nothing written yet
    assert not main_window.parameter_panel.btn_add_fitted.isEnabled()
    assert "Add to fitted" in main_window.statusBar().currentMessage()

    gate.set()                                        # the fit finishes
    qtbot.waitUntil(lambda: main_window._add_fitted_wait is None, timeout=10000)
    assert _table_ids(path, "fitted") == [0, 1, 2]
    assert len(calls) == 1                            # preview and click shared one fit
    _select_detected(main_window, 2)                  # the button is not left disabled
    if main_window.parameter_panel._mlgidbase_available:   # else the panel keeps it off
        assert main_window.parameter_panel.btn_add_fitted.isEnabled()


@pytest.mark.gui
def test_add_to_fitted_2d_reuses_the_finished_preview_without_refitting(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    gate, calls = _gated_fit(main_window, monkeypatch)
    gate.set()
    _select_detected(main_window, 2)
    qtbot.waitUntil(lambda: main_window._preview_runner.pending_count() == 0, timeout=10000)
    assert len(calls) == 1
    main_window._on_add_to_fitted()                   # cache hit: instant
    assert main_window._add_fitted_wait is None
    assert _table_ids(path, "fitted") == [0, 1, 2]
    assert len(calls) == 1                            # no second fit


@pytest.mark.gui
def test_a_changed_fit_setting_or_neighbour_is_not_answered_from_the_cache(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    _open(main_window, synthetic_nexus_with_peaks)
    gate, calls = _gated_fit(main_window, monkeypatch)
    gate.set()
    sel = _select_detected(main_window, 2)
    qtbot.waitUntil(lambda: main_window._preview_runner.pending_count() == 0, timeout=10000)
    entry = main_window.entry_combo.currentText()
    before = main_window._preview_fingerprint(entry, sel)
    panel = main_window.pipeline_panel
    if hasattr(panel, "fit_dist_peaks"):     # the fit form needs the backend extra
        panel.fit_dist_peaks.setValue(panel.fit_dist_peaks.value() + 3)
        assert main_window._preview_fingerprint(entry, sel) != before
        panel.fit_dist_peaks.setValue(panel.fit_dist_peaks.value() - 3)
        assert main_window._preview_fingerprint(entry, sel) == before
    main_window.viewer._frame_peaks[0]["detected"].radius[0] += 0.5   # an edited neighbour
    assert main_window._preview_fingerprint(entry, sel) != before


@pytest.mark.gui
def test_add_to_fitted_2d_is_dropped_if_the_file_closes_while_it_fits(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    gate, _calls = _gated_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)
    main_window._on_add_to_fitted()
    assert main_window._add_fitted_wait is not None
    main_window._add_fitted_wait["session"] = "somewhere/else.h5"   # as if the file changed
    gate.set()
    qtbot.waitUntil(lambda: main_window._add_fitted_wait is None, timeout=10000)
    assert _table_ids(path, "fitted") == [0, 1]       # nothing was written


@pytest.mark.gui
def test_a_second_click_while_one_is_fitting_does_nothing(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    gate, calls = _gated_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)
    main_window._on_add_to_fitted()
    main_window._on_add_to_fitted()
    assert "still fitting" in main_window.statusBar().currentMessage()
    gate.set()
    qtbot.waitUntil(lambda: main_window._add_fitted_wait is None, timeout=10000)
    assert _table_ids(path, "fitted") == [0, 1, 2]    # one row, not two
    assert len(calls) == 1


@pytest.mark.gui
def test_add_to_fitted_2d_does_not_yank_the_selection_if_the_user_moved_on(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
    qtbot,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    gate, _calls = _gated_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)
    main_window._on_add_to_fitted()
    _select_detected(main_window, 0)                  # moved to another peak
    gate.set()
    qtbot.waitUntil(lambda: main_window._add_fitted_wait is None, timeout=10000)
    assert _table_ids(path, "fitted") == [0, 1, 2]    # the click still committed
    sel = main_window.viewer.selected_peak
    assert (sel.kind, sel.peak_id) == ("detected", 0)  # and the selection stayed


# -- Detected boxes for fitted peaks that have none ------------------------

def _frames(path, frame=0):
    """Frame ``frame``'s detected table as a structured array."""
    with h5py.File(path, "r") as f:
        return f[f"{ENTRY}/data/analysis/frame{frame:05d}/detected_peaks"][()]


def test_the_score_follows_the_visibility_grade():
    score = file_model.detected_score_for_visibility
    assert [score(v) for v in (3, 2, 1)] == [1.0, 0.5, 0.1]
    assert score(0) == 1.0                    # no grade: nothing to carry over


def test_each_fitted_id_gets_a_detected_box_of_the_same_size(tmp_path):
    path = _build(tmp_path, [0, 1], [0, 1, 5, 6])
    added, unusable = file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    assert added == {0: [5, 6]} and unusable == 0
    assert _ids(path, "detected") == [0, 1, 5, 6]
    det, fit = _frames(path), None
    with h5py.File(path, "r") as f:
        fit = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"][()]
    for pid in (5, 6):
        d = det[det["id"] == pid][0]
        t = fit[fit["id"] == pid][0]
        for col in ("angle", "angle_width", "radius", "radius_width"):
            assert d[col] == pytest.approx(t[col])
        assert not d["is_ring"]
        assert d["q_xy"] == pytest.approx(d["radius"] * np.cos(np.deg2rad(d["angle"])))


def test_the_new_box_carries_only_geometry_id_ring_flag_and_score(tmp_path):
    path = _build(tmp_path, [0], [0, 3])
    with h5py.File(path, "r+") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        arr = ds[()]
        arr["amplitude"], arr["theta"] = 9.0, 0.3
        arr["A"], arr["is_cut_qz"] = 1.0, True
        arr["visibility"] = [0, 2]
        ds[...] = arr
    file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    d = _frames(path)
    d = d[d["id"] == 3][0]
    assert d["score"] == pytest.approx(0.5)          # visibility 2
    for col in ("amplitude", "theta", "A", "B", "C", "visibility"):
        assert d[col] == 0
    assert not d["is_cut_qz"] and not d["is_cut_qxy"]


def test_a_fitted_ring_gets_a_finite_quadrant_spanning_box(tmp_path):
    path = _build(tmp_path, [0], [0, 4])
    with h5py.File(path, "r+") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        arr = ds[()]
        arr["angle"], arr["angle_width"] = [45.0, float("nan")], [5.0, float("inf")]
        ds[...] = arr
    file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    d = _frames(path)
    d = d[d["id"] == 4][0]
    assert (d["angle"], d["angle_width"], d["is_ring"]) == (45.0, 90.0, True)


def test_a_fitted_ring_with_a_finite_width_is_copied_as_it_is(tmp_path):
    path = _build(tmp_path, [0], [0, 4])
    with h5py.File(path, "r+") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        arr = ds[()]
        arr["angle"], arr["angle_width"], arr["is_ring"] = 40.0, 70.0, True
        ds[...] = arr
    file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    d = _frames(path)
    d = d[d["id"] == 4][0]
    assert (d["angle"], d["angle_width"], d["is_ring"]) == (40.0, 70.0, True)


def test_every_frame_of_the_entry_is_covered_and_complete_ones_are_left_alone(
    tmp_path,
):
    path = _build(tmp_path, [0, 1], [0, 1], frames=3)
    with h5py.File(path, "r+") as f:           # frame 1 gets an unpaired fit
        g = f[f"{ENTRY}/data/analysis/frame00001"]
        del g["fitted_peaks"]
        g.create_dataset("fitted_peaks", data=_rows([0, 1, 8]))
        # frame 2 has no detected table at all
        del f[f"{ENTRY}/data/analysis/frame00002/detected_peaks"]
    added, _ = file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    assert added == {1: [8], 2: [0, 1]}
    assert _ids(path, "detected", 0) == [0, 1]
    assert _ids(path, "detected", 1) == [0, 1, 8]
    assert _ids(path, "detected", 2) == [0, 1]


def test_afterwards_every_fitted_id_has_a_detected_partner_and_a_rerun_adds_nothing(
    tmp_path,
):
    path = _build(tmp_path, [2], [0, 1, 2, 3, 3], frames=2)
    file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    for fr in (0, 1):
        assert set(_ids(path, "fitted", fr)) <= set(_ids(path, "detected", fr))
        assert len(_ids(path, "detected", fr)) == len(set(_ids(path, "detected", fr)))
    again, _ = file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    assert again == {}


def test_a_fitted_row_with_no_radius_is_skipped_and_counted(tmp_path):
    path = _build(tmp_path, [0], [0, 1, 2])
    with h5py.File(path, "r+") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"]
        arr = ds[()]
        arr["radius"][1] = float("nan")
        ds[...] = arr
    added, unusable = file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    assert added == {0: [2]} and unusable == 1


def test_delete_detected_rows_undoes_the_batch(tmp_path):
    path = _build(tmp_path, [0], [0, 1, 2], frames=2)
    added, _ = file_model.add_missing_detected_for_fitted(
        path, ENTRY, ring_angle=45.0, ring_width=90.0,
    )
    assert file_model.delete_detected_rows(path, ENTRY, added) == 4
    assert _ids(path, "detected", 0) == [0] and _ids(path, "detected", 1) == [0]


@pytest.mark.gui
def test_the_edit_menu_adds_detected_boxes_and_one_undo_removes_them(
    main_window, synthetic_nexus_with_peaks, clean_link_settings, qtbot,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    with main_window._detached_silx_tree():
        file_model.add_fitted_peak_row(
            path, ENTRY, 0, angle=30.0, angle_width=6.0, radius=1.2,
            radius_width=0.1, amplitude=3.0, peak_id=9,
        )
    main_window._load_entry_into_viewer(ENTRY, preserve_view=True)
    main_window.action_add_detected_for_fitted.trigger()
    assert _table_ids(path, "detected") == [0, 1, 2, 9]
    det = _table(path, "detected")
    i = list(det.ids).index(9)
    assert float(det.angle[i]) == pytest.approx(30.0)
    assert "Added 1 detected box" in main_window.statusBar().currentMessage()
    main_window._action_undo()
    assert _table_ids(path, "detected") == [0, 1, 2]
    main_window._action_redo()
    assert _table_ids(path, "detected") == [0, 1, 2, 9]


@pytest.mark.gui
def test_the_menu_item_says_so_when_nothing_is_missing(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    main_window.action_add_detected_for_fitted.trigger()
    assert "already has a detected partner" in main_window.statusBar().currentMessage()
    assert _table_ids(path, "detected") == [0, 1, 2]


@pytest.mark.gui
def test_the_menu_item_is_a_no_op_without_a_file(main_window):
    main_window.action_add_detected_for_fitted.trigger()
    assert "open a file" in main_window.statusBar().currentMessage()


@pytest.mark.gui
def test_a_fitted_delete_takes_its_fit_errors_and_undo_brings_them_back(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """fitted_peaks_errors is paired with fitted_peaks by id. Deleting the
    fit drops its errors row with it (no ghost row for a peak that is
    gone), and undo restores the errors with the fit."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    err_path = f"{ENTRY}/data/analysis/frame00000/fitted_peaks_errors"
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            fit = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"][()]
            err = np.zeros(len(fit), dtype=fit.dtype)
            err["id"] = fit["id"]
            err["radius"] = [0.01 * (int(i) + 1) for i in fit["id"]]
            if "fitted_peaks_errors" in f[f"{ENTRY}/data/analysis/frame00000"]:
                del f[err_path]
            f[f"{ENTRY}/data/analysis/frame00000"].create_dataset(
                "fitted_peaks_errors", data=err,
            )

    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_fitted(main_window, 1))
    assert _table_ids(path, "fitted") == [0]
    with h5py.File(path, "r") as f:
        assert f[err_path][()]["id"].tolist() == [0]

    main_window.viewer.undo_last_action()
    assert sorted(_table_ids(path, "fitted")) == [0, 1]
    with h5py.File(path, "r") as f:
        errs = f[err_path][()]
    assert sorted(errs["id"].tolist()) == [0, 1]
    assert errs["radius"][errs["id"] == 1][0] == pytest.approx(0.02)


@pytest.mark.gui
def test_deleting_a_matched_peak_leaves_detected_and_fitted_alone(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """mlgidbase.delete_peak removed the id from all three tables whatever
    the dialog said. The matched delete is matched-only now, and undoable."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    ds_path = f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            m = np.zeros(2, dtype=_MATCHED_DTYPE)
            m["CIF"] = [b"a.cif", b"b.cif"]
            m["probability"] = [0.9, 0.8]
            m["peak_list"][0] = np.asarray([0, 1], dtype=np.int32)
            m["peak_list"][1] = np.asarray([1], dtype=np.int32)
            f.create_dataset(ds_path, data=m)
    before_det = _table_ids(path, "detected")
    before_fit = _table_ids(path, "fitted")

    def _lists():
        with h5py.File(path, "r") as f:
            return [[int(v) for v in r] for r in f[ds_path][()]["peak_list"]]

    _confirm(monkeypatch)
    sel = SelectedPeak(
        kind="matched", frame=0, peak_id=1, radius=1.0, angle=45.0,
        radius_width=0.2, angle_width=5.0, is_ring=False,
        score=0.9, amplitude=1.0,
    )
    main_window._on_delete_peak_requested(sel)
    assert _lists() == [[0], []]
    assert _table_ids(path, "detected") == before_det
    assert _table_ids(path, "fitted") == before_fit

    main_window.viewer.undo_last_action()
    assert _lists() == [[0, 1], [1]]
    with h5py.File(path, "r") as f:
        assert "peak_list_kind" not in f[ds_path].attrs   # back to verbatim

    main_window.viewer.redo_last_action()
    assert _lists() == [[0], []]
    assert _table_ids(path, "fitted") == before_fit
