# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the one-folder build of the app.

Parameterised on the app name:

    APP_NAME      window / exe / folder name        (default mlgidGUI)
    APP_MODULE    the importable package            (default mlgidgui)
    WITH_PIPELINE 1 = full build with the backends, 0 = lite (default 1)

Run through ``build.py``, which also writes the version resource and runs
Inno Setup. The build must be done in an environment that has the app
installed NON-editable (``pip install ".[pipeline]"``): PyInstaller reads the
installed package and its metadata.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

HERE = Path(SPECPATH)                  # packaging/windows
ROOT = HERE.parents[1]                 # the repo root
APP_NAME = os.environ.get("APP_NAME", "mlgidGUI")
APP_MODULE = os.environ.get("APP_MODULE", "mlgidgui")
WITH_PIPELINE = os.environ.get("WITH_PIPELINE", "1") == "1"
VERSION_FILE = HERE / "version_info.txt"   # written by build.py, Windows only

# Packages whose data files, binaries or lazily imported submodules
# PyInstaller's static analysis does not see. ``collect_all`` takes all three.
BASE_PACKAGES = [
    APP_MODULE,      # the SVG icons and the app icons (package data)
    "silx",          # colormaps, icons, OpenCL kernels, lazy widgets
    "pyFAI",         # detector and calibrant tables, the calibration widgets
    "fabio",         # image format plugins
    "hdf5plugin",    # the HDF5 compression filters (binary plugins)
    "qdarkstyle",    # the stylesheet resources
    "pyqtgraph",     # lazily imported graphics items
]
PIPELINE_PACKAGES = [
    "pygid", "pygidfit", "mlgidbase", "mlgidmatch", "pygidsim", "mlgiddetect",
    "onnxruntime",
    # pygid imports it at start-up and it reads its own default .conf file
    # from the package folder, which static analysis does not copy.
    "xrayutilities",
    # mlgiddetect's models use torchvision's compiled operators (nms); the
    # extension module is not an import the analysis can follow.
    "torchvision",
    # Dependencies of the backends that read data files or load compiled
    # libraries by path (pymatgen's element tables, rtree's spatial index,
    # shapely's GEOS, numba's JIT runtime).
    "pymatgen", "monty", "spglib", "rtree", "shapely", "numba", "llvmlite",
    "lmfit", "tifffile",
]
# Distributions whose version the app reads at run time (importlib.metadata).
METADATA = [APP_MODULE, "silx", "pyFAI", "mlgidbase"]

datas, binaries, hiddenimports = [], [], []
packages = BASE_PACKAGES + (PIPELINE_PACKAGES if WITH_PIPELINE else [])
for package in packages:
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
for dist in METADATA:
    try:
        datas += copy_metadata(dist)
    except Exception as exc:  # a missing dist must not stop the build
        print(f"warning: no metadata for {dist}: {exc}")

# "What's new" reads CHANGELOG.md from inside the package; in the source tree
# it sits at the repo root (setup.py copies it into wheels).
datas.append((str(ROOT / "CHANGELOG.md"), APP_MODULE))

# Tests ship with several of these packages and are never imported.
excludes = ["tkinter", "pytest", "IPython.testing"]

a = Analysis(
    [str(HERE / "launch.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,                 # UPX trips antivirus scanners
    console=False,             # a windowed app, like pythonw
    icon=str(ROOT / APP_MODULE / "assets" / "app" / f"{APP_MODULE}.ico"),
    version=str(VERSION_FILE) if sys.platform == "win32" and VERSION_FILE.is_file() else None,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name=APP_NAME,
)
