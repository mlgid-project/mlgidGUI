# Getting started

Install + first-run guide for **mlgidLAB**, the desktop GUI for the
[mlgidBASE](https://github.com/mlgid-project/mlgidBASE) GIWAXS pipeline.
For the pipeline-backend version policy see
[backend_compatibility.md](backend_compatibility.md). Contributor-facing
code documentation (architecture, per-module reference) lives in the
`Documentation/` tree of the development workspace.

## Requirements

- **Python ≥ 3.11** (CPython). Linux, macOS, or Windows.
- A working Qt-capable display. (Headless use is only for the test
  suite, which runs offscreen.)
- For detection / fitting / matching / raw conversion: the in-house
  pipeline stack (the `[pipeline]` extra) — see below.

mlgidLAB runs **view-only** without the pipeline stack: open and browse
NeXus files, edit peaks manually, run the in-GUI pyFAI calibration. The
Run buttons stay disabled until `[pipeline]` is installed.

## Install

A clean virtual environment is strongly recommended (the `[pipeline]`
extra pulls a large ML stack including PyTorch).

### Recommended: conda env

```bash
conda create -n mlgidlab python=3.12 -y
conda activate mlgidlab

# GUI only
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a22"

# or, full pipeline (detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a22"
```

### From a local clone (for development)

```bash
git clone https://github.com/mlgid-project/mlgidLAB
cd mlgidLAB
pip install -e ".[pipeline]"      # or -e . for view-only, -e ".[dev]" for tests
```

### Per-OS notes

- **Linux**: PySide6's offscreen/X11 plugins need a few system Qt libs
  if they're missing (`libegl1 libgl1 libglib2.0-0 libxkbcommon0
  libxcb-cursor0 libdbus-1-3 libfontconfig1` on Debian/Ubuntu). pip
  wheels bundle the rest of Qt.
- **Windows**: install into a fresh venv/conda env; the pip wheels for
  PySide6, silx, pyFAI, h5py are self-contained. No extra system libs.
- **macOS** (Intel + Apple Silicon): the same `pip install` works; the
  first launch may take a moment while Qt initializes.

The `[pipeline]` extra pins the verified-good backend versions
(`mlgidbase==0.1.11`, `pygid==0.2.20`, `pygidfit==0.1.8`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.7`, `mlgiddetect==0.2.8`); bumping them is a deliberate,
test-rechecked step (see [backend_compatibility.md](backend_compatibility.md)).

## Launch

```bash
mlgidlab                 # console entry point
mlgidlab path/to/file.h5 # open a file on startup
python -m mlgidlab       # equivalent module form
```

## Updating

On launch mlgidLAB checks GitHub for a newer release. If one exists a
banner appears with an **Update now** button that installs the new version
into the current environment (equivalent to the `pip install --upgrade
"…@<tag>"` command above, run in the background) and then offers to
restart. `Help → Update now…` runs the same check-and-install on demand
(no need to wait for the banner), and `Help → Automatically install
updates` (off by default) does it automatically on launch, after a single
confirmation.

**mlgidLAB is becoming mlgidGUI.** With release 1.0.0 the app is renamed
mlgidGUI (package and command `mlgidgui`). Nothing has to be done in
advance: when 1.0.0 is out, the same banner offers **Update now**, which
installs mlgidGUI from PyPI and only then removes mlgidLAB, so a failed
install leaves mlgidLAB working. Scripts that `import mlgidlab` must be
changed to `import mlgidgui`; there is no compatibility layer. A one-time
notice in the app explains this (`Help → mlgidLAB is becoming mlgidGUI…`).

Self-update is only offered for normal pip installs. For a **development /
editable** checkout (`pip install -e`) the button is hidden — update it with
`git pull` (or re-run the `pip install -e` command). The `[pipeline]` extra
is kept across an update when it is installed.

## First-file walkthrough

1. **Open** a file: `File → Open…`, drag-and-drop, or `File → Open
   recent`. Supported inputs are HDF5/NeXus results (`.h5 .hdf5 .nxs`),
   raw HDF5 detector stacks, and standalone detector images (`.tif .tiff
   .cbf .edf`). The GUI classifies each automatically; the right dock
   shows **Pipeline** for NeXus, **Conversion** for raw (including
   images). Selecting several images — or dropping a folder of them —
   opens each as its own entry, switchable from the entry combo and the
   file browser.
2. **Look around**: with nothing open the window shows a welcome page
   (Open, Import images, recent files). Once a file is loaded: left =
   HDF5 tree; centre = q-image (toggle Cartesian / Polar) under a
   workflow rail showing what Convert / Detect / Fit / Match / Track
   have produced for the current frame; right-top = Pipeline /
   Conversion; right-bottom = Profiles / Peaks / Scan tracking tabs;
   status bar = file / entry / frame / pipeline activity / cursor
   (`q`, `d` and `2θ`).
3. **Raw → NeXus** (raw file): in the Conversion dock set PONI, mask,
   angle of incidence (the **Create…** buttons open the embedded pyFAI
   calibration), choose geometry + output, click **Convert**. The
   converted file opens automatically.
4. **Detect / fit / match** (NeXus file): Pipeline dock → **Run
   Detection** → **Run Fitting**; set a CIF source (or load the bundled
   `example/prepr_cifs.pickle`) → **Run Matching**. Overlays and the
   Peaks tables populate. Only load a pickle you made yourself or got
   from someone you trust: opening a pickle runs the code inside it.
5. **Edit peaks**: click a peak to select; drag ROI edges to resize;
   `Ctrl+Alt`-drag a new manual box (polar mode) and **Add to
   detected** / **Add to fitted**. Overlapping boxes follow one rule:
   the smallest box under the cursor wins, and clicking again steps to
   the next one underneath. Multi-select detected/fitted peaks with
   `Ctrl+click` / `Ctrl+A`, on the image **or** in the Peaks tables;
   copy/paste detected peaks with `Ctrl+C` / `Ctrl+V` (or
   `Ctrl+Shift+V` to paste to a frame range); **Fit selected (2D)**
   batch-fits; `Delete` removes (bulk for a multi-selection, from
   either the image or the tables). `Ctrl+Z` / `Ctrl+Shift+Z` undo /
   redo every edit.
6. **Label a frame quickly**: tick **Quick select** in the Display
   dock and pick what each box becomes (detected, fitted, or both).
   Drawing the next box commits the previous one, so labelling is
   draw-draw-draw; a box drawn on top of the pending one replaces it
   instead. The last box commits when you click away, press `Enter`,
   change frame or switch the mode off — `Esc` discards it.

   By default each fit is stored under the id of the detected peak it
   came from: one fit per detection, refitting replaces rather than
   appends, and deleting a detected peak deletes its fit. Both that
   link and the reverse (a fitted delete removing the detection) are
   switches under `Settings → Settings…`.
7. **Save**: edits live on a temp copy; `File → Save` / `Save As`
   writes back. The title bar shows a `*` while there are unsaved
   changes.
8. **Export**: `Tools → Export figure…` (matplotlib) or `Export peaks
   as CSV…`.

A working sample is bundled under `example/` (`organic_labeled.h5`,
`eiger4m_0000.h5`, `prepr_cifs.pickle`).

## Keyboard shortcuts

| Key | Action |
|-|-|
| `←` / `→`, `J` / `K` | previous / next frame |
| `Home` / `End` | first / last frame |
| `Ctrl+click`, `Ctrl+A` | multi-select detected/fitted peaks |
| `Ctrl+C` / `Ctrl+V` | copy / paste detected peaks |
| `Ctrl+Shift+V` | paste detected peaks to a frame range |
| `Delete` | delete selected peak(s) — on the image or in the Peaks tables |
| `Enter` | commit the pending Quick-select box |
| `Esc` | discard the selected manual box |
| `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) | undo / redo |
| `Ctrl+F` | find peak by id |
| `F1` | controls & shortcuts reference |
| `F11` | fullscreen image viewer |

## Troubleshooting

- **Run buttons are disabled / "view-only"**: the `[pipeline]` extra
  isn't installed. `pip install "mlgidlab[pipeline] @ git+…"`.
- **Qt / "could not load platform plugin"** on Linux: install the
  system Qt libs listed above.
- **An operation seems to do nothing**: check the **Logs** dock — some
  calibration/worker errors are logged there rather than shown as a
  dialog.
- **Matching returns no solutions**: confirm the CIF source loaded for
  the active entry (multi-energy files parse CIFs per entry).
