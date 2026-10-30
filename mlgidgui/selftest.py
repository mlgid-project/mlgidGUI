"""``mlgidgui --self-test``: a start-up check of an installed or frozen build.

It opens a real window, loads a tiny NeXus file it writes itself, steps a
frame, reads the bundled changelog and icons, and (with ``--backends``) imports
every package the pipeline needs. One line per check goes to stdout, or to the
file named by ``MLGIDGUI_SELFTEST_LOG`` (the Windows build has no console).
The exit code is 0 only when every check passed.

CI runs this against the frozen Windows build on the offscreen platform, where
a data file or a hidden import that PyInstaller missed shows up as a failed
check here instead of as a crash on a user's machine.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

#: Packages the pipeline imports directly (see ``[project.optional-dependencies]``).
BACKENDS = (
    "pygid", "mlgidbase", "pygidfit", "mlgidmatch", "pygidsim", "mlgiddetect",
    "onnxruntime",
)


def _write_nexus(path: Path) -> None:
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.attrs["axes"] = ["frame_num", "q_z", "q_xy"]
        data.create_dataset(
            "img_gid_q", data=rng.random((3, 16, 24), dtype=np.float32))
        data.create_dataset("q_xy", data=np.linspace(-1.0, 3.0, 24, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0.0, 4.0, 16, dtype=np.float32))


_NACL_CIF = """data_NaCl
_symmetry_space_group_name_H-M 'F m -3 m'
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na Na 0 0 0
Cl Cl 0.5 0.5 0.5
"""


def _fit_a_peak() -> str:
    """The Add to fitted (2D) path on a synthetic Gaussian spot."""
    import numpy as np

    from mlgidgui.manual_fit import fit_one_peak

    n = 160
    q_xy = np.linspace(-1.0, 3.0, n)
    q_z = np.linspace(0.0, 4.0, n)
    xx, zz = np.meshgrid(q_xy, q_z)
    image = 1000.0 * np.exp(
        -((xx - 1.414) ** 2 + (zz - 1.414) ** 2) / (2 * 0.05 ** 2)) + 5.0
    result = fit_one_peak(
        image.astype(np.float32), q_xy, q_z,
        radius=2.0, radius_width=0.3, angle=45.0, angle_width=8.0,
        wavelength_angstrom=1.0, q_xy_max=3.0, q_z_max=4.0,
    )
    # The check is that the fitter runs end to end and returns numbers, not
    # how close a synthetic image on invented geometry lands.
    values = [result.radius, result.radius_width, result.angle,
              result.angle_width, result.amplitude]
    if not all(np.isfinite(values)) or result.amplitude <= 0:
        raise RuntimeError(f"unusable fit: {values}")
    return f"r={result.radius:.3f}, angle={result.angle:.1f}"


def _read_a_cif() -> str:
    from pymatgen.io.cif import CifParser

    import warnings

    with warnings.catch_warnings():
        # The minimal CIF has no symmetry-operation list; pymatgen says so.
        warnings.simplefilter("ignore")
        structure = CifParser.from_str(_NACL_CIF).parse_structures(primitive=False)[0]
    formula = structure.composition.reduced_formula
    if formula != "NaCl":
        raise RuntimeError(f"read {formula!r}, expected NaCl")
    return formula


def _run_nms() -> str:
    import torch
    from torchvision.ops import nms

    keep = nms(torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]),
               torch.tensor([0.9, 0.8]), 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError(f"nms kept {keep.tolist()}")
    return "kept [0]"


def run(argv: list[str]) -> int:
    """Run the checks and return the process exit code."""
    out_path = os.environ.get("MLGIDGUI_SELFTEST_LOG")
    out = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    failed: list[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failed.append(name)
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}",
              file=out, flush=True)

    def check(name: str, fn) -> None:
        try:
            report(name, True, str(fn() or ""))
        except Exception:
            report(name, False, traceback.format_exc().strip().splitlines()[-1])

    from mlgidgui import __version__, update_check

    print(f"mlgidGUI {__version__}, frozen={update_check.is_frozen()}, "
          f"python {sys.version.split()[0]}", file=out, flush=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def window_and_file() -> str:
        from PySide6.QtWidgets import QApplication

        from mlgidgui.main_window import MainWindow
        from mlgidgui.session import NexusSession

        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = MainWindow()
        try:
            with tempfile.TemporaryDirectory(prefix="mlgidgui-selftest-") as tmp:
                path = Path(tmp) / "selftest.h5"
                _write_nexus(path)
                window._set_active_session(NexusSession.open(path))
                app.processEvents()
                frames = window.viewer.n_frames
                if frames != 3:
                    raise RuntimeError(f"expected 3 frames, the viewer has {frames}")
                window.close()
        finally:
            window.deleteLater()
        return "3 frames"

    check("window opens a NeXus file", window_and_file)

    def changelog() -> str:
        path = update_check._changelog_path()
        if path is None:
            raise RuntimeError("CHANGELOG.md is not bundled")
        sections = update_check.parse_changelog_sections(
            path.read_text(encoding="utf-8"))
        if not sections:
            raise RuntimeError("no sections in the bundled changelog")
        return f"{len(sections)} sections"

    check("bundled changelog", changelog)

    def icons() -> str:
        from mlgidgui.icons import app_icon

        if app_icon().isNull():
            raise RuntimeError("the app icon is empty")
        return ""

    check("bundled icons", icons)

    if "--backends" in argv:
        import importlib

        for name in BACKENDS:
            check(f"import {name}", lambda n=name: importlib.import_module(n).__name__)

    if "--backends" in argv:
        # Importing is not enough: these run the code paths that read data
        # files and load compiled libraries by path, which is what a frozen
        # build tends to lose.
        check("2D peak fit (pygidfit)", _fit_a_peak)
        check("read a CIF (pymatgen)", _read_a_cif)
        check("torchvision nms", _run_nms)

    print("self-test " + ("FAILED: " + ", ".join(failed) if failed else "passed"),
          file=out, flush=True)
    if out is not sys.stdout:
        out.close()
    return 1 if failed else 0
