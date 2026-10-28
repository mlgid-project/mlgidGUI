<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mlgid-project/mlgidLAB/main/docs/images/mlgid_logo_mlgidlab_dark.png">
    <img src="https://raw.githubusercontent.com/mlgid-project/mlgidLAB/main/docs/images/mlgid_logo_mlgidlab.png" width="400" alt="mlgidLAB">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/mlgid-project/mlgidLAB/actions/workflows/ci.yml"><img src="https://github.com/mlgid-project/mlgidLAB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mlgid-project/mlgidLAB/releases/latest"><img src="https://img.shields.io/github/v/release/mlgid-project/mlgidLAB?include_prereleases&label=release" alt="Latest release"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python 3.11 or newer">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  A desktop GUI for the
  <a href="https://github.com/mlgid-project/mlgidBASE">mlgidBASE</a>
  GIWAXS analysis pipeline.<br>
  Convert, detect, fit, match, edit and track, with the same algorithms and visual control.
</p>

<p align="center">
  <a href="#see-it-in-action">Demo</a> &middot;
  <a href="#install--launch">Install</a> &middot;
  <a href="#what-you-can-do">Features</a> &middot;
  <a href="docs/getting_started.md">Getting started</a> &middot;
  <a href="CHANGELOG.md">Changelog</a> &middot;
  <a href="https://github.com/mlgid-project/mlgidLAB/issues">Issues</a>
</p>

> **Alpha (`v0.1.0a22`).** The detect, fit, match and edit loop works end to end. Expect
> rough edges and please report issues. See [`CHANGELOG.md`](CHANGELOG.md) for
> highlights.

mlgidLAB drives the full `pygid → mlgidDETECT → pygidFIT → mlgidMATCH`
workflow in one PySide6 window: convert raw detector images to NeXus, then
**detect / fit / match**, **review and edit** peaks, **track** them across a
scan, and **export**.

## See it in action

Three short clips, one per stage of the workflow.

### 1. Convert raw detector images

Open a raw 2D detector image, load the pyFAI PONI calibration and the detector
mask, set the angle of incidence, attach sample metadata from a YAML file and
convert to a q-space map.

https://github.com/user-attachments/assets/95fdcdc0-4c5b-4893-a6d9-02ece98f89ab

### 2. Detect, fit and match

Detect diffraction peaks with the ML model, 2D-fit a strong peak by hand, fit
every detected peak and inspect the radial and angular profiles, then match the
fitted peaks against a CIF library and review the identified phases.

https://github.com/user-attachments/assets/c05ea716-4c7e-4dbe-8584-f130b7394060

### 3. Track peaks across a scan

Open a completed scan, track each peak across its frames, interpolate the gaps
(every missing frame is fitted and re-matched), play the scan and follow the
matched phases as they grow and decay.

https://github.com/user-attachments/assets/ddc65852-fc9e-4cb4-a1ef-f6bb1911357b

<sub>Recorded against `v0.1.0a22`.</sub>

## Install & launch

Requires **Python ≥ 3.11** (Linux / macOS / Windows). A fresh conda
environment is recommended: the `[pipeline]` extra pulls a large ML
stack (PyTorch + the in-house mlgid packages), so keep it isolated.

```bash
# create + activate an isolated environment
conda create -n mlgidlab python=3.12 -y
conda activate mlgidlab

# full pipeline (detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a22"
mlgidlab
```

No conda? Any Python ≥ 3.11 virtual environment works
(`python -m venv mlgidlab && source mlgidlab/bin/activate`).

Drop `[pipeline]` for view-only mode (browse and edit existing NeXus
results, in-GUI pyFAI calibration; Run buttons disabled). Full per-OS
install, a first-file walkthrough, and shortcuts are in
**[docs/getting_started.md](docs/getting_started.md)**.

## What you can do

| | |
|---|---|
| **Open and view** | NeXus or raw HDF5 by auto-detection, several files at once. Cartesian (q_xy / q_z) or polar view, frame playback, colormap, log/linear contrast, live cursor readout. Multi-GB stacks load lazily. |
| **Convert** | Raw detector images to NeXus, in batch, with an embedded pyFAI calibration dialog for PONI and mask. |
| **Detect, fit, match** | Run each stage from the Pipeline dock with every mlgidBASE parameter exposed, at frame, entry or all-entries scope. A workflow rail above the image shows what each stage has produced for the current frame. |
| **Edit peaks** | Manual boxes, Add to fitted (2D pygidfit or 1D), multi-select of detected or fitted peaks (`Ctrl+click`, `Ctrl+A`) on the image or in the tables, copy and paste (`Ctrl+C` / `Ctrl+V`, `Ctrl+Shift+V` for a frame range), batch 2D fit, bulk delete, full undo and redo. |
| **Label fast** | *Quick select* commits each hand-drawn box as you draw the next one, as a detected peak, a fitted one, or both. Each fit is stored under its detected peak's id, so a peak has one fit and deleting the detection takes the fit with it (both switchable in Settings). |
| **Inspect** | Sortable Detected / Fitted / Matched tables with click-sync to the image, live radial and angular profile fits, overlay filters. |
| **Track** | Follow peaks across the frames of a scan, fill the gaps, and view the matched phases and their amplitudes over time. |
| **Edit the file** | The **Structure** tab edits every group, field, attribute and link of the open NeXus/HDF5 file in place, with undo, copy and paste between files, and a check for anything that would stop the viewer or the pipeline from reading it. Nothing touches the file on disk until you save. |
| **Export** | Figures (matplotlib) and peaks as CSV. |

## Documentation

- **[Getting started](docs/getting_started.md)**: install, first run,
  shortcuts, troubleshooting.
- **[Backend compatibility](docs/backend_compatibility.md)**: pipeline
  version policy and how to verify a backend bump.
- **[Manual test plan](docs/manual_test_plan.html)**: click-through
  checklist for alpha testers. Download it and open it in a browser. Grab
  the `example.zip` dataset, hosted once on the
  [v0.1.0a9 release](https://github.com/mlgid-project/mlgidLAB/releases/download/v0.1.0a9/example.zip)
  and reused for every version, so later releases carry no attachments.
- **[Changelog](CHANGELOG.md)**: release highlights.

For contributor-facing architecture and module reference docs please contact me.

## License

MIT, see [`LICENSE`](LICENSE). The optional `[pipeline]` extra pulls in
GPL-3.0 `mlgidMATCH` as a separate, optional dependency the GUI calls
(aggregation, not a derived work).

## Contact

Nico Lerch, <nico.lerch@uni-tuebingen.de>. Issues and feedback via
[GitHub issues](https://github.com/mlgid-project/mlgidLAB/issues).
