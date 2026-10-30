"""Import pre-converted images as one N-frame scan.

Writer (``conversion.import_converted_stack``): mlgid-minimum schema
(signal / axes / resizable stack / per-frame analysis groups), q axes
from typed ranges vs pixel fallback, vertical flip, shape-mismatch
error, provenance note, overwrite. GUI: the written file opens as a
regular NeXus session with the frame slider active; a float-pixel
batch triggers the import offer on open while integer batches open as
raw untouched; imported entries are refused by the pipeline pre-flight
(``_entry_missing_geometry``) with a clear log line.

Source: conversion.py ``import_converted_stack``; import_dialog.py;
workers.py ``ImportWorker``; main_window.py
``_offer_import_for_float_images`` / ``_run_import_dialog`` /
``_on_import_finished`` / ``_entry_missing_geometry``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mlgidgui.conversion import import_converted_stack

pytestmark = pytest.mark.gui


def _write_float_tiffs(dirpath, n=3, h=40, w=50, order=None) -> list[Path]:
    import fabio.tifimage

    dirpath.mkdir(parents=True, exist_ok=True)
    paths = []
    indices = order if order is not None else range(n)
    for i in indices:
        p = dirpath / f"map_{i:04d}.tif"
        data = np.full((h, w), float(i) + 0.5, dtype=np.float32)
        fabio.tifimage.TifImage(data=data).write(str(p))
        paths.append(p)
    return paths


# -- writer ---------------------------------------------------------------


def test_import_writes_mlgid_minimum_schema(tmp_path):
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(
        tifs,
        tmp_path / "imported.h5",
        qxy_range=(0.0, 2.5),
        qz_range=(-0.1, 3.0),
        ai=0.2,
    )
    with h5py.File(out, "r") as f:
        assert f.attrs["default"] == "entry_0000"
        g = f["entry_0000"]
        data = g["data"]
        assert data.attrs["signal"] == "img_gid_q"
        assert list(data.attrs["axes"]) == ["frame_ind", "q_z", "q_xy"]
        img = data["img_gid_q"]
        assert img.shape == (3, 40, 50) and img.dtype == np.float32
        assert img.maxshape == (None, 40, 50)
        # Frame pixels arrive verbatim, in order.
        assert float(img[1].mean()) == pytest.approx(1.5)
        q_xy, q_z = data["q_xy"][()], data["q_z"][()]
        assert q_xy[0] == 0.0 and q_xy[-1] == 2.5 and len(q_xy) == 50
        assert q_z[0] == -0.1 and q_z[-1] == 3.0 and len(q_z) == 40
        assert data["q_xy"].attrs["units"] == "1/Angstrom"
        assert list(data["frame_ind"][()]) == [0.0, 1.0, 2.0]
        assert sorted(data["analysis"].keys()) == [
            "frame00000", "frame00001", "frame00002"
        ]
        assert list(g["instrument/angle_of_incidence"][()]) == [0.2] * 3
        # q ranges but NO wavelength → still no geometry (the pipeline
        # needs both); provenance note explains the re-import path.
        assert "instrument/detector" not in g
        assert "no detector" in str(g["process/mlgidgui/NOTE"][()]).lower()

    # The GUI classifies it as a converted mlgid entry.
    from mlgidgui import file_model

    with h5py.File(out, "r") as f:
        assert file_model.entry_group_names(f) == ["entry_0000"]
    assert file_model.classify_entry_data(out, "entry_0000") == "mlgid"


def test_import_pixel_fallback_flip_and_mismatch(tmp_path):
    tifs = _write_float_tiffs(tmp_path / "maps", n=2)
    out = import_converted_stack(
        tifs, tmp_path / "px.h5", flip_vertical=True
    )
    with h5py.File(out, "r") as f:
        data = f["entry_0000/data"]
        # Pixel axes: plain indices, no q units claimed.
        assert list(data["q_xy"][()][:3]) == [0.0, 1.0, 2.0]
        assert "units" not in data["q_xy"].attrs
        assert data["img_gid_q"].shape == (2, 40, 50)

    # Vertical flip actually flips rows.
    import fabio.tifimage

    grad = np.outer(
        np.arange(40, dtype=np.float32), np.ones(50, dtype=np.float32)
    )
    gp = tmp_path / "grad.tif"
    fabio.tifimage.TifImage(data=grad).write(str(gp))
    out2 = import_converted_stack([gp], tmp_path / "flip.h5",
                                  flip_vertical=True)
    with h5py.File(out2, "r") as f:
        assert f["entry_0000/data/img_gid_q"][0, 0, 0] == 39.0

    # A mismatched frame names the offender.
    small = _write_float_tiffs(tmp_path / "small", n=1, h=8, w=9)
    with pytest.raises(ValueError, match="does not match"):
        import_converted_stack(
            [tifs[0], small[0]], tmp_path / "bad.h5"
        )


def test_failed_import_leaves_the_output_path_as_it_was(tmp_path):
    """An import that fails part-way writes nothing: no truncated file at
    a new path, and an earlier import at the same path survives intact
    (re-importing a folder reuses the default ``<folder>_imported.h5``)."""
    tifs = _write_float_tiffs(tmp_path / "maps", n=2)
    small = _write_float_tiffs(tmp_path / "small", n=1, h=8, w=9)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="does not match"):
        import_converted_stack([tifs[0], small[0]], out / "new.h5")
    assert list(out.iterdir()) == []

    earlier = import_converted_stack(tifs, out / "maps_imported.h5")
    before = earlier.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        import_converted_stack([tifs[0], small[0]], earlier)
    with pytest.raises(ValueError, match="2 angles.*1 image"):
        import_converted_stack([tifs[0]], earlier, ai=[0.1, 0.2])
    assert earlier.read_bytes() == before
    assert list(out.iterdir()) == [earlier]


def test_import_with_wavelength_writes_pipeline_geometry(
    main_window, tmp_path
):
    """q ranges + wavelength → full instrument block: real wavelength
    (Å → m) and incidence angle, zero placeholders for the detector
    fields nothing consumes — the standard pipeline then accepts the
    entry (guard passes). Wavelength without q ranges writes nothing."""
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(
        tifs,
        tmp_path / "geom.h5",
        qxy_range=(0.0, 2.0),
        qz_range=(0.0, 2.0),
        ai=0.12,
        wavelength_A=0.9601,
    )
    with h5py.File(out, "r") as f:
        g = f["entry_0000"]
        assert g["instrument/monochromator/wavelength"][()] == pytest.approx(
            0.9601e-10
        )
        det = g["instrument/detector"]
        for field in (
            "distance", "x_pixel_size", "y_pixel_size", "beam_center_x",
            "beam_center_y", "polar_angle", "aequatorial_angle",
            "rotation_angle",
        ):
            assert det[field][()] == 0.0
            assert "placeholder" in det[field].attrs
        assert "placeholder" in str(g["process/mlgidgui/NOTE"][()]).lower()
    # The pipeline pre-flight accepts it now.
    assert not main_window._entry_missing_geometry(out, "entry_0000")

    # Wavelength alone (no q ranges) does NOT unlock the pipeline —
    # pixel axes would feed the model meaningless q scales.
    out2 = import_converted_stack(
        tifs, tmp_path / "nogeom.h5", wavelength_A=0.9601
    )
    with h5py.File(out2, "r") as f:
        assert "instrument/detector" not in f["entry_0000"]
    assert main_window._entry_missing_geometry(out2, "entry_0000")


def test_imported_geometry_loads_through_pygid(tmp_path):
    """The former blocker: pygid's ``load_entry`` (→ ``load_params``)
    must succeed on an imported entry carrying the placeholder
    geometry, with the file's axes and the real wavelength intact.
    Fitting/matching consume only wavelength + ai from the params —
    verified against mlgidbase 0.1.5 (see docs/backend_compatibility.md);
    re-check on backend bumps."""
    pygid = pytest.importorskip("pygid")

    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(
        tifs,
        tmp_path / "geom.h5",
        qxy_range=(0.0, 2.0),
        qz_range=(0.0, 2.0),
        ai=0.12,
        wavelength_A=0.9601,
    )
    conv = pygid.NexusFile(str(out)).load_entry("entry_0000", 0)
    assert conv.params.wavelength == pytest.approx(0.9601)  # Å internally
    assert list(conv.params.ai) == [0.12]
    assert conv.matrix[0].q_xy[-1] == pytest.approx(2.0)
    assert conv.img_gid_q[0].shape == (40, 50)


# -- GUI round trip -------------------------------------------------------


def test_imported_file_opens_as_nexus_scan(main_window, qtbot, tmp_path):
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(
        tifs, tmp_path / "imported.h5", qxy_range=(0.0, 2.0),
        qz_range=(0.0, 2.0),
    )
    main_window._open_paths([out])
    qtbot.waitUntil(
        lambda: main_window.session is not None
        and main_window.session.kind == "nexus",
        timeout=30000,
    )
    assert main_window.entry_combo.currentText() == "entry_0000"
    qtbot.waitUntil(
        lambda: main_window.viewer.n_frames == 3, timeout=30000
    )


def test_float_batch_offers_import_integer_opens_raw(
    main_window, qtbot, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox as _RealBox

    from mlgidgui import main_window as mw_mod
    from mlgidgui import main_window_files as mwf_mod

    offered: dict = {}

    class _StubBox:
        ButtonRole = _RealBox.ButtonRole
        StandardButton = _RealBox.StandardButton

        def __init__(self, parent=None):
            self._buttons = []

        def setWindowTitle(self, *_):
            pass

        def setText(self, *_):
            pass

        def addButton(self, *args):
            token = object()
            self._buttons.append(token)
            return token

        def exec(self):
            pass

        def clickedButton(self):
            # First added button = "Import as one scan…".
            return self._buttons[0]

    # _offer_import_for_float_images resolves QMessageBox from the
    # files-mixin module since the 2026 source split.
    monkeypatch.setattr(mwf_mod, "QMessageBox", _StubBox)
    monkeypatch.setattr(
        mw_mod.MainWindow,
        "_run_import_dialog",
        lambda self, paths: offered.setdefault("paths", list(paths)),
    )

    # Float pixels → the offer fires and consumes the batch.
    tifs = _write_float_tiffs(tmp_path / "maps")
    main_window._open_paths(list(tifs))
    assert offered["paths"] == tifs
    assert main_window.session is None

    # Integer pixels → no offer, normal raw session.
    import fabio.tifimage

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "det_0000.tif"
    fabio.tifimage.TifImage(
        data=np.full((40, 50), 7, dtype=np.uint16)
    ).write(str(raw))
    offered.clear()
    main_window._open_paths([raw])
    assert "paths" not in offered
    assert main_window.session is not None
    assert main_window.session.kind == "raw"


def test_pipeline_guard_refuses_imported_entry(main_window, tmp_path):
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(tifs, tmp_path / "imported.h5")
    assert main_window._entry_missing_geometry(out, "entry_0000")
    assert main_window._entry_missing_geometry(out, None)

    # A real conversion output (detector group present) passes.
    real = tmp_path / "real.h5"
    with h5py.File(real, "w") as f:
        g = f.create_group("entry_0000")
        g.attrs["NX_class"] = "NXentry"
        g.create_dataset("instrument/detector/distance", data=0.3)
        g.create_group("data").attrs["signal"] = "img_gid_q"
    assert not main_window._entry_missing_geometry(real, "entry_0000")

    # A minimal/foreign mlgid file WITHOUT the import provenance also
    # passes even though it lacks the detector group — the guard keys
    # on ``process/mlgidgui`` so synthetic fixtures and foreign files
    # keep the pre-existing let-pygid-report-it behavior (a
    # geometry-only probe silently skipped every pipeline run on the
    # test fixtures).
    minimal = tmp_path / "minimal.h5"
    with h5py.File(minimal, "w") as f:
        g = f.create_group("entry_0000")
        g.attrs["NX_class"] = "NXentry"
        g.create_group("data").attrs["signal"] = "img_gid_q"
    assert not main_window._entry_missing_geometry(minimal, "entry_0000")


def test_import_dialog_takes_an_angle_per_image(
    main_window, qtbot, tmp_path
):
    """Same grammar as the Conversion panel, checked against the image count."""
    from mlgidgui.import_dialog import ImportConvertedDialog

    # Five images, so three numbers are unambiguously a ramp (with
    # exactly three they would read as three angles, by design).
    tifs = _write_float_tiffs(tmp_path / "maps", n=5)
    dlg = ImportConvertedDialog(tifs, parent=main_window)
    qtbot.addWidget(dlg)

    dlg.ai_input.setText("0.1, 0.2, 0.3, 0.4, 0.5")
    assert dlg.values()["ai"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert dlg.ai_hint.text().startswith("5 angles")

    # A ramp gives one MORE angle than its step count.
    dlg.ai_input.setText("(0.1, 0.5, 4)")
    assert dlg.values()["ai"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])

    # ...and getting that wrong is caught while typing, not at Import.
    dlg.ai_input.setText("(0.1, 0.5, 5)")
    assert dlg.ai_hint.property("status") == "error"
    assert "5 image(s)" in dlg.ai_hint.text()

    # An empty field is the old default, not an error.
    dlg.ai_input.setText("")
    assert dlg.values()["ai"] == pytest.approx(0.0)


def test_import_writes_one_angle_per_frame(tmp_path):
    """The array lands in instrument/angle_of_incidence, in order."""
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(
        tifs, tmp_path / "out.h5", ai=[0.1, 0.2, 0.3],
    )
    with h5py.File(out, "r") as f:
        ai = np.asarray(f["entry_0000/instrument/angle_of_incidence"][()])
    assert ai == pytest.approx([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="2 angles.*3 image"):
        import_converted_stack(
            tifs, tmp_path / "bad.h5", ai=[0.1, 0.2],
        )


def test_import_dialog_values_round_trip(
    main_window, qtbot, tmp_path, monkeypatch
):
    from mlgidgui.import_dialog import ImportConvertedDialog

    tifs = _write_float_tiffs(tmp_path / "maps")
    dlg = ImportConvertedDialog(tifs, parent=main_window)
    qtbot.addWidget(dlg)

    # Defaults: pixel axes (no q ranges), entry_0000, out path next to
    # the batch.
    values = dlg.values()
    assert values["entry_name"] == "entry_0000"
    assert values["qxy_range"] is None and values["qz_range"] is None
    assert values["out_path"].parent == tifs[0].parent

    assert values["wavelength_A"] is None

    dlg.use_q_ranges.setChecked(True)
    dlg.qxy_min.setValue(0.0)
    dlg.qxy_max.setValue(2.5)
    dlg.qz_min.setValue(-0.5)
    dlg.qz_max.setValue(3.0)
    dlg.ai_input.setText("0.15")
    dlg.flip_vertical.setChecked(True)
    dlg.wavelength_input.setValue(0.9601)
    values = dlg.values()
    assert values["qxy_range"] == (0.0, 2.5)
    assert values["qz_range"] == (-0.5, 3.0)
    assert values["ai"] == pytest.approx(0.15)
    assert values["flip_vertical"] is True
    assert values["wavelength_A"] == pytest.approx(0.9601)

    # Wavelength is gated with the q ranges: unchecking drops both.
    dlg.use_q_ranges.setChecked(False)
    values = dlg.values()
    assert values["qxy_range"] is None
    assert values["wavelength_A"] is None
    dlg.use_q_ranges.setChecked(True)

    # Invalid range (min >= max) blocks accept.
    dlg.qxy_max.setValue(-1.0)
    warned: list = []
    import mlgidgui.import_dialog as dlg_mod

    class _W:
        @staticmethod
        def warning(*a, **k):
            warned.append(a)

    monkeypatch.setattr(dlg_mod, "QMessageBox", _W)
    dlg._validate_and_accept()
    assert warned and dlg.result() != 1


def test_an_import_written_by_an_alpha_is_still_recognised(main_window, tmp_path):
    """The alphas stamped the provenance group ``process/mlgidlab``; the guard
    must keep refusing pipeline runs on those files."""
    tifs = _write_float_tiffs(tmp_path / "maps")
    out = import_converted_stack(tifs, tmp_path / "imported.h5")
    with h5py.File(out, "r+") as f:
        f["entry_0000/process"].move("mlgidgui", "mlgidlab")
    assert main_window._entry_missing_geometry(out, "entry_0000")
