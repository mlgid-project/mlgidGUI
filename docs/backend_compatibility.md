# Backend compatibility — verifying mlgidbase & the pipeline stack

mlgidLAB runs standalone (view + edit existing NeXus results) on its
public-PyPI runtime dependencies alone. Detection, fitting, and
matching require the in-house pipeline stack — installed via the
`[pipeline]` extra. That stack moves faster than the GUI, so this
document is the procedure for confirming a given pipeline release still
drives mlgidLAB the same way, plus the exact API surface the GUI
depends on (the things that break if an upstream signature drifts).

Run this whenever you bump `mlgidbase` (or any backend), before
shipping a release, and as part of the alpha pre-flight.

## Verified-good baseline

The full suite passes with this set installed (the backend-dependent
tests un-skip and run, not just the pure-h5py ones):

| Package | Baseline version | Reached how |
|-|-|-|
| `mlgidbase` | 0.1.11 | declared in `[pipeline]` |
| `pygid` | 0.2.20 | declared in `[pipeline]` |
| `pygidfit` | 0.1.8 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgidmatch` | 0.1.3 | declared in `[pipeline]` (GUI imports it directly) |
| `pygidsim` | 0.1.7 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgiddetect` | 0.2.8 | transitive via `mlgidbase` (GUI never imports it) |
| `networkx` | any | declared in `[pipeline]` (pygidfit imports and declares it; mlgidbase 0.1.11 no longer imports it, so the line is a harmless leftover) |

**Pinning policy — exact `==`, not floors.** Each in-house backend
release can shift the detection/fitting/matching numerics or the
on-disk schema, so a bump must be a deliberate, test-rechecked change
rather than something pip picks up silently. The `[pipeline]` extra
therefore pins every directly-imported backend to an exact version
(the baseline above). `mlgidbase 0.1.11` pins `mlgiddetect==0.2.8` and
`mlgidmatch==0.1.3` with `==`, but takes `pygid` (`>=0.2.20`) and
`pygidfit` (`>=0.1.8`) on floors, which is why the GUI's own exact
`pygid==0.2.20` and `pygidfit==0.1.8` matter: without them pip would
follow those floors forward on its own. `pygidsim==0.1.7` is
exact here for the same reason: `mlgidmatch` and `pygid` both take it
on a floor. The GUI still declares `pygidfit`/`mlgidmatch`/`pygidsim`
explicitly — rather than leaning on `mlgidbase`'s transitive closure —
because it imports those three *directly*, so a future `mlgidbase`
dropping one must surface as a clear resolver error, not a runtime
`ImportError`. `mlgiddetect` is left transitive (the GUI never imports
it); its per-Python `onnxruntime-gpu` pins (CUDA 12 builds; 1.26.0 on
Python 3.11+) carry the GPU runtime, so the GUI no longer pins
onnxruntime itself. **To move the baseline up:** bump the pins here
and in `pyproject.toml`, re-run the full suite + the end-to-end demo
loop, then commit.

**Runtime (non-pipeline) deps stay on `>=` floors on purpose.** The GUI
stack — PySide6, pyqtgraph, silx, numpy, etc. — is *not* pinned exact,
so the CI matrix can keep testing forward-compat across Python
3.11-3.14 against whatever those projects release. A cyclic-GC-during-
construction segfault that the newest PySide6 (6.11.1) + Python 3.13/14
exposed is handled at the source in `tests/conftest.py` (`gc.disable()`
for the session) plus an idempotent `closeEvent`, not by pinning the
GUI stack down — verified green with PySide6 6.11.1 + pyqtgraph 0.14.0
on both Python 3.12 and 3.14.

**One exclusion: PySide6 6.12.0 on Python 3.11.** That release loses one
reference to `None` per void method call (`w.update()`, `w.setText(...)`)
on 3.11, where `None` is not immortal, so the interpreter dies with
`Fatal Python error: none_dealloc` once the count runs out; the suite
aborted partway through on every 3.11 shard (exit code 134), on `main`
as well. 3.12 and later are unaffected. `pyproject.toml` therefore
excludes exactly that version on 3.11 and leaves later releases allowed.

## Step 1 — Is mlgidbase installable from your index?

```bash
pip index versions mlgidbase        # lists versions visible on the active index
pip download mlgidbase --no-deps -d /tmp/mlgidbase_check   # confirms it resolves + fetches
```

If those fail against public PyPI, the package is coming from a
private index — inspect the configured indexes:

```bash
pip config list                     # look for index-url / extra-index-url
cat ~/.config/pip/pip.conf 2>/dev/null
```

Web cross-check: open `https://pypi.org/project/mlgidbase/`.

## Step 2 — Does the latest stack still drive the GUI?

In a throwaway environment, so the verified-good env stays intact:

```bash
conda create -n mlgidlab_compat python=3.12 -y
conda activate mlgidlab_compat
git clone https://github.com/mlgid-project/mlgidLAB && cd mlgidLAB
pip install mlgidbase pygid pygidfit mlgidmatch     # pull the latest from PyPI
pip install -e ".[dev]"
python -c "import importlib.metadata as m; \
  [print(p, m.version(p)) for p in \
  ('mlgidbase','pygid','pygidfit','mlgidmatch','pygidsim','mlgiddetect')]"
pytest -q          # backend tests un-skip when the stack imports
```

Watch for: the pipeline / manual-fit / energy-guard / matched-
invalidation tests must **run** (not `importorskip`-skip) and pass. A
skip there means a backend failed to import — read the collection log.

Then a behavioral end-to-end pass in the GUI (the demo loop):

```bash
python -m mlgidlab
```

1. Open `example/eiger4m_0000.h5` (raw) and convert it.
2. On the converted file: Pipeline dock → Run Detection → Run Fitting.
3. Parse CIFs (or load `example/prepr_cifs.pickle`) → Run Matching.
4. Overlays + Peaks-dock tables populate for all three kinds.

If any stage errors, the failing call is in the API-surface table
below — that tells you which upstream signature changed.

## Step 3 — API surface the GUI depends on

If a stage breaks after a bump, the cause is almost always one of
these. Verbatim call sites are in the code; this is the contract.

**`mlgidbase.mlgidBASE`** (`pipeline.py`, `figure_export_window.py`)
- `mlgidBASE(filename=str(path))` constructor.
- `analysis.run_detection(**kwargs)`, `run_fitting(**kwargs)`,
  `run_matching(**kwargs)` — resolved dynamically by name.
- `analysis.set_plot_defaults(**style)` and
  `analysis.plot_analysis_results(save_fig=True, path_to_save_fig=...,
  plot_result=False, return_result=False, **kwargs)` (figure export).
- `mlgiddetect_functions.load_config`'s **dict form**, used by the
  Pipeline dock's detection parameters (`mlgidlab.detection_config`):
  a `{SECTION: {KEY: value}}` dict passed as `config_detect` is applied
  onto a fresh `mlgiddetect` `Config` with
  `setattr(config, f"{SECTION}_{KEY}", value)`, so the panel's widgets
  reach mlgidDETECT without a config file on disk. Precedence matters:
  the dict is applied first and an explicit `model_type` kwarg is
  assigned *after* it, so the **Model** combo still wins over anything
  the dict says about `MODEL_TYPE` (which is why the panel never puts
  `TYPE` in the dict). Introduced in mlgidbase 0.1.8; a bump that
  narrowed `config` back to `str | Config` would break every detection
  run from the GUI, not just a configured one. Regression:
  `tests/test_detection_params.py` (including a drift guard that
  compares every exposed default against `Config.init_default`).
- **The `POSTPROCESSING_*` settings apply to the `dino` model only.**
  `mlgiddetect.postprocessing.standard_postprocessing` branches on
  `MODEL_TYPE`: the `dino` branch calls `filter_boxes(config, ...)`,
  which reads `POSTPROCESSING_SCORE`, `POSTPROCESSING_NMSIOU`,
  `POSTPROCESSING_CLASSAWARE_NMS`, `POSTPROCESSING_NMSIOU_RING` and
  `POSTPROCESSING_NMSIOU_SEG`, while the `faster_rcnn` branch reads none
  of them and hardcodes its own chain (`SmallQFilter(50) +
  MergeBoxesPostprocessing(min_score=0.01) +
  StandardPostprocessing(nms_level=0.01, score_level=0.01)`). So with
  `faster_rcnn` selected in the Pipeline dock's **Model** box, the panel's
  Score threshold, NMS IoU, Class-aware NMS and the two per-class IoU
  fields would have no effect. The same holds for the dino ensemble:
  `mlgiddetect.inference` only fuses models when `MODEL_TYPE == 'dino'`
  and `MODEL_ENSEMBLE_ENABLED`, so under `faster_rcnn` the Ensemble and
  ONNX ensemble fields are ignored too. The combo defaults to
  "(default)", which resolves to `dino`, so this only bites a deliberate
  `faster_rcnn` run. The panel therefore **hides** those seven rows
  while `faster_rcnn` is selected (`PipelinePanel._on_det_model_changed`)
  and shows them again for `dino` and `(default)`. The hidden widgets
  keep their values and still travel in `config_detect`, which is
  harmless because the backend never reads them under `faster_rcnn`.
  Regression: `tests/test_detection_params.py`. Re-check this branching
  on any mlgiddetect bump, and keep the hidden set in step with it.

**`pygid`** (`conversion.py`)
- `pygid.ExpParams(poni_path=..., ai=..., mask_path=...)`
- `pygid.CoordMaps(params, hor_positive=..., vert_positive=..., dq=...,
  dang=..., q_xy_range=..., q_z_range=..., ...)`
- `pygid.Conversion(matrix=..., path=..., dataset=..., frame_num=...)`
  then `det2q_gid` / `det2q` / `det2pol_gid` / `det2pol`
  `(frame_num=..., return_result=False, save_result=True,
  path_to_save=..., h5_group=..., overwrite_file=..., overwrite_group=...)`
- `pygid.SampleMetadata(data=...)`, `pygid.ExpMetadata(**fields)`, and
  the writable `ExpMetadata.extend_fields` attribute (set inside a
  `try/except` — older pygid lacking it is tolerated).

**Imported-scan geometry contract** (`conversion.import_converted_stack`)
- Imported image scans carry a REAL `monochromator/wavelength` +
  `angle_of_incidence` and ZERO placeholders for every other
  `instrument/detector/*` dataset (attr `placeholder`). This is sound
  because, as of mlgidbase 0.1.8 / pygidfit 0.1.4 / mlgidmatch 0.1.3:
  detection consumes only image + q axes
  (`mlgiddetect_functions._run_detection_single_frame`), fitting
  additionally only `params.wavelength` + `params.ai`
  (`pygidfit.process_scans.ProcessDataFromFile.run_frame` →
  missing-wedge / critical-angle math), and matching only fitted peaks
  + q maxima (`mlgidmatch_functions.run_mlgidmatch_from_file` — no
  `load_params` at all). `pygid.load_params` still hard-reads every
  detector dataset, which is why the placeholders must EXIST.
  **On any pygidfit / mlgidmatch / mlgidbase bump: re-check that the
  ops still ignore SDD / beam center / pixel size** — if one starts
  consuming them, imported-scan results would silently use the zeros.
  Regression tripwire: `test_imported_geometry_loads_through_pygid`
  plus the empirical fitting run recorded 2026-07-30.

**Peak tracking** (`pipeline.execute` -> `phase_tracking`)
- mlgidbase >= 0.1.11: `Analysis.track_peaks(entry, threshold, length,
  window_size=None)` computes the IoU graph in bounded-memory row blocks
  (scipy sparse connected components; this is the approach the GUI
  carried itself from the 0.1.5 stack until the 0.1.11 bump, as
  `track_peaks_blocked`), saves the table to
  `<entry>/data/analysis/general/tracked_peaks` (+ metadata under
  `<entry>/process/peak_tracking`) and returns it. `plot_tracked_peaks`
  draws the saved table. The GUI reads the table via
  `phase_tracking.payload_from_tracked_table` (fields `frame_list`,
  `peak_list` = fitted ids). Older tables (`frames` / `peak_ids`, or
  per-frame columns) are rejected with a named error.
- **A `general` group under `analysis` is not a frame**, but pygid's
  `fill_analysis_group` counts every group there when it numbers
  appended frames (checked in 0.2.20). The GUI therefore deletes the
  table and the then-empty group before an append-frames conversion
  (`conversion._drop_stale_tracking`), and on a refit.
- **On any mlgidbase bump:** re-run `tests/test_phase_tracking.py` and
  `tests/test_blocked_tracking.py`; they run the real backend and
  compare the payload with upstream's own return value.

**`pygidsim.experiment.ExpParameters`** (`pipeline.py`)
- `ExpParameters(q_xy_max=..., q_z_max=..., ai=..., en=...)` — `en` in
  eV; the energy guard expects `1e3 <= en <= 2e5`.

**`mlgidmatch.preprocess.cif_preprocess.CifPattern`** (`pipeline.py`)
- `CifPattern(params=..., folder_path=..., cifs=..., create_all=True)`;
  the GUI reads `cif_pattern.params.en` and `.params.ai` (via
  `getattr(..., default)`) to detect a parameter mismatch.

**`pygidfit.process_scans`** (`manual_fit.py`)
- `img_preprocessing(cartesian, ai_deg, crit_angle, wavelength_A, q_z)`
- `_get_polar_grid(img_shape, (512, 1024), [0, 0])` — the polar grid is
  hardcoded `(512, 1024)`; a change here breaks byte-identity between
  manual 2D fits and pipeline fits.
- `polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)`
- `fit_data(polar_img, radius=..., radius_width=..., angle=...,
  angle_width=..., wavelength=..., q_xy_max=..., q_z_max=..., ...)`

**Data contracts** (file_model + pipeline post-processing)
- NeXus paths: `{entry}/data/img_gid_q`, `{entry}/data/q_xy`,
  `{entry}/data/q_z`, `{entry}/data/analysis/frameNNNNN/<kind>_peaks`.
- `fitted_peaks` dtype fields: `q_xy, q_z, radius, angle,
  radius_width, angle_width, amplitude, id, score, is_ring`.
- `matched_peaks` per-solution fields: `CIF, h, k, l, peak_list,
  probability` (`peak_list` recast to int32; rows deduped by max
  `probability`).
- `matched_*` `peak_list` KIND. mlgidbase <= 0.1.10 stores row
  POSITIONS into the frame's `fitted_peaks`; 0.1.11 stores the
  `fitted_peaks['id']` VALUES. There is no marker in the file, and one
  frame can hold both (rings and segments are replaced independently).
  The GUI stamps each dataset it writes or migrates with the HDF5
  attribute `peak_list_kind` (`"id"` / `"position"`;
  `file_model.MATCHED_KIND_ATTR`). An unstamped dataset is classified
  from its values (`classify_peak_list_kind`): values outside
  `0..n-1` can only be ids, values that are no fitted id can only be
  positions; when every value is valid both ways (always so when the
  fitted ids are `0..n-1`, where the readings coincide) it takes the
  majority kind of the frame's other datasets, else "position". The
  reader (`read_matched_peaks`) translates to positions, so
  `MatchedStructure.peak_list` stays positional for the viewer; a
  fitted delete first puts the frame's datasets in id form
  (`prepare_fitted_delete`) and then only drops ids
  (`remap_matched_peak_lists`). `pipeline.execute` normalises the
  scope's existing datasets before a `run_matching` and stamps what the
  run wrote with the kind of the installed mlgidbase
  (`backend_peak_list_kind`).

## Step 4 — Record the result

If the latest stack passes, note the confirmed versions here (update
the baseline table) and, if a version moved, bump the `[pipeline]` pins
in `pyproject.toml`. If it fails, file the failing call + the version
that introduced the break as a known issue and keep the last-good
pin in place.

## Recorded bumps

### 2026-07-05 — mlgidbase main @ `561edfa` (unreleased, self-labelled 0.1.3)

Installed for the phase-tracking feature (`track_peaks`, "First
implementation of the peak tracking function"):

```bash
# reference clone fast-forwarded to origin/main first
pip install --force-reinstall --no-deps project_repos/mlgidBASE
```

- **Full suite green** against it (352 passed locally, backends
  installed) — the standard Step-2 verification.
- The `[pipeline]` pin in `pyproject.toml` is **unchanged** (still
  `mlgidbase` 0.1.3): the commit is unreleased and still calls itself
  0.1.3, so there is nothing meaningful to pin. Re-pin when upstream
  tags the release containing `561edfa`.
- **Caveats the GUI codes around** (checked at this sha; see
  `mlgidlab/phase_tracking.py`):
  - `track_peaks` exists only past the 0.1.3 release — the GUI
    feature-gates on availability and raises a named RuntimeError
    ("needs mlgidbase newer than 0.1.3") for older installs.
  - `networkx` is imported by `mlgidbase.peak_operations` but **not
    declared** in mlgidbase's dependencies (upstream packaging gap).
    It is present in the GUI_project env; a fresh env needs
    `pip install networkx` until upstream declares it.
  - (Superseded by the 2026-10-22 entry: mlgidbase 0.1.11 saves a
    table and the capture hook is gone.) Upstream `_track_peaks`
    **returns None** (docstring claims
    otherwise); the GUI recovers the data via a capture hook on
    `mlgidbase.peak_operations._plot_tracked_peaks` (positional
    8-arg contract, validated at call time —
    `phase_tracking.UpstreamContractError` fires loudly if a future
    mlgidbase changes the call shape).
  - **Rings are never tracked by upstream**: a ring's
    `angle_width = inf` makes every IoU against its box NaN,
    `_track_peaks` zeroes NaNs, so each ring is a 1-member component
    discarded by the `length` cut. The GUI compensates by tracking
    rings itself — `phase_tracking.track_rings` runs the same
    all-against-all / connected-components / `length`-cut algorithm on
    the 1-D IoU of the radius intervals (rings have no azimuthal
    position) and appends the ring tracks to the captured payload. A
    proper upstream fix (ring↔ring radial IoU inside `_track_peaks`,
    e.g. treating `angle_width = inf` as full-arc overlap) would let
    the GUI drop this pass — report alongside the return-None and
    networkx items.
- Rollback: `pip install mlgidbase==0.1.3` (phase tracking then
  reports the feature as unavailable; everything else unchanged).

### 2026-07-21 — mlgidbase main @ `1b6514f` (unreleased, self-labelled 0.1.4)

Moves the env past the 0.1.4 release (2026-07-07) onto current main,
still `--no-deps` so the rest of the verified stack (pygid 0.2.10,
mlgiddetect 0.2.3, onnxruntime-gpu 1.26.0) stays untouched — main
pins `pygid==0.2.12` / `mlgiddetect==0.2.5`, neither vetted here:

```bash
pip install --no-deps --force-reinstall \
    "git+https://github.com/mlgid-project/mlgidBASE@1b6514f"
```

- **Full suite green** against it after the GUI adaptations below.
- What changed upstream since `561edfa` (0.1.4 tracking rework + the
  fig/ax return commits):
  - `track_peaks` **removed the `'amplitude'` axis** (ValueError:
    invalid axis). The GUI now requests `axis='radius'` and recovers
    amplitudes from the **return value**, which is no longer None:
    pre-0.1.4 flat `(axis_arr, amplitude_all, G_comps_list)` and
    0.1.4+ per-component frame-sorted `(axis_list, amplitude_list,
    frame_num_list)` are both understood
    (`phase_tracking.amplitudes_from_track_result`). Members outside
    every component get NaN amplitude; `member_ids` therefore uses
    amplitude only as a duplicate-breaker when finite.
  - The `_plot_tracked_peaks` capture contract (positional 8-arg
    call) is **unchanged** — verified at this sha.
  - `plot_analysis_results` renamed `return_result` → `return_fig`;
    the figure-export window now omits the kwarg entirely (default is
    off under both names), so it works against every build.
  - Upstream now clamps ring `angle_width = inf` to 45° inside
    `_track_peaks` (in-memory only). Rings still form 1-member
    components in practice; the GUI's own `track_rings` pass stays.
  - The networkx packaging gap from the `561edfa` entry still holds.
- Rollback: `pip install mlgidbase==0.1.3` or reinstall `561edfa`
  (the GUI adaptations are backward-compatible with both).

### 2026-07-21 — `onnxruntime-gpu==1.26.0` pinned in `[pipeline]`

`mlgiddetect<=0.2.3` requires `onnxruntime-gpu` without a version.
onnxruntime-gpu **1.27.0 (2026-06-18) hard-links `libcudart.so.13`
at import time**; the CUDA 13 runtime arrives as pip `nvidia-*`
wheels (via torch >= 2.12), but nothing puts them on the loader path,
so in a fresh `[pipeline]` env `import mlgidbase` (→ mlgiddetect →
onnxruntime) fails with *"libcudart.so.13: cannot open shared object
file"* and the GUI misread it as "backend not installed". Two-part
fix:

- pin `onnxruntime-gpu==1.26.0` (the verified GPU-capable build that
  imports lazily), and
- `pipeline.is_mlgidbase_available` retries the import with `torch`
  imported first (torch dlopens its bundled CUDA libraries, which
  also satisfies onnxruntime 1.27+) and logs the real ImportError
  instead of silently reporting "not installed".

Trap for the next bump: **mlgiddetect >= 0.2.4 (2026-07-17) switched
its dependency to CPU-only `onnxruntime`**, which ships the same
Python module as `onnxruntime-gpu` — co-installing both clobbers
files nondeterministically, and mlgidbase 0.1.4 additionally declares
plain `onnxruntime` itself. When the extra moves to a stack with
mlgiddetect >= 0.2.4, drop this pin and decide the GPU story
explicitly (GPU detection needs `onnxruntime-gpu` present INSTEAD of
the CPU package, and `pipeline.detection_on_gpu` handles provider
selection + `preload_dlls`).

### 2026-07-23 — mlgidbase 0.1.5 from PyPI, full released stack

The first PyPI release containing the tracking rework: the env moves
off git installs entirely — `pip install mlgidbase==0.1.5` WITH
dependencies (pygid 0.2.13, mlgiddetect 0.2.8, pygidfit 0.1.3,
mlgidmatch 0.1.3). Code-wise 0.1.5 is the `1b6514f` build the env
already ran plus ensemble METADATA keys; the meaningful changes are
dependency-level. Full suite green before and after the GUI
adaptations below.

- **Workarounds dropped** (fixed upstream):
  - `is_mlgidbase_available`'s torch-first import retry — mlgiddetect
    0.2.8 pins per-Python `onnxruntime-gpu` (CUDA 12, lazily
    importing; 1.26.0 on py3.11+), so the libcudart.so.13 import trap
    is gone; the `[pipeline]` `onnxruntime-gpu==1.26.0` pin is dropped
    for the same reason.
  - `detection_on_gpu`'s `torch.cuda.is_available` monkeypatch —
    mlgiddetect >= 0.2.7 gates GPU on the ORT providers + a CUDA
    driver probe (never torch) and falls back to CPU;
    `MODEL_FORCE_CPU` forces CPU. The context manager now only
    preloads onnxruntime's CUDA DLLs.
  - `amplitudes_from_track_result`'s pre-0.1.4 flat-shape branch —
    the pin guarantees the per-component return; the flat shape now
    raises `UpstreamContractError`.
  - `figure_export_window`'s omitted figure-return kwarg — passes
    `return_fig=False` explicitly (the 0.1.4 rename is settled).
- **Ring tracking, the fine print.** 0.1.5 clamps ring
  `angle_width` inf -> 45 inside `_track_peaks`, but pygidfit
  persists rings with `angle = NaN` (`pygidfit/box_utils.py`,
  `make_box_attributes`), and NaN still poisons the (angle, radius)
  IoU box — backend-fitted rings STILL end as discarded 1-member
  components, so the GUI's `track_rings` pass stays. Rings stored
  with a FINITE angle (mlgidLAB's injected rings persist
  `angle = 45`) CAN now form native components; the result handler
  (`_on_phase_track_result`) drops ring-member components from the
  native payload before appending `track_rings` output, so every
  physical ring yields exactly one track (regression:
  `test_native_ring_components_not_double_counted`). Report upstream:
  ring tracking wants radial IoU for `is_ring` rows, not an angle
  clamp.
- **Still true in 0.1.5 / still GUI-side** (checked at this bump):
  `_plot_tracked_peaks` capture hook (the return value has per-track
  arrays but no member coordinates or peak ids), `member_ids`
  reconstruction, `_backfill_fitted_peaks_polar_to_cartesian`,
  `_dedupe_matched_groups`, matched-row invalidation before refit,
  `create_all=True` on CifPattern, the pygid pre-flight guards,
  the energy-range guard, `normalize_for_pygid` (pygid 0.2.13's
  `get_ai` still assumes 1-D `angle_of_incidence`), the manual
  injection stack (`add_peak` upstream appends an unfitted detected
  row only — no 2D fit, no plausibility gate, no rollback), GUI-side
  peak delete (different cascade semantics), and the networkx
  packaging gap (still undeclared by mlgidbase; now declared in the
  GUI's `[pipeline]` extra).
- **pygid 0.2.10 -> 0.2.13** (skimmed `v0.2.10..origin/main`):
  fig/ax returns on visualization, `Conversion.get_result()`,
  `NexusFile.set_beamtime_info()`, simulation range follows the image
  range, and the module-level root-handler stripping in
  `coordmaps`/`datasaver` is REMOVED (only a harmless plain
  `basicConfig` remains in `conversion`). `det2*`, the datasaver
  results dtype, and `get_ai` are unchanged — the GUI's row writers
  and conversion path are unaffected.
- **Detection results change on purpose:** mlgiddetect 0.2.6+
  defaults to the 2-class `DINO_classAwareBaseline` (vs the
  single-class dino that 0.2.3 shipped). Decision 2026-07-23: adopt
  the new default; re-vet detection quality manually (legacy model
  reachable via `ONNX_BASE: dino_old` + `CLASSAWARE_NMS: False`).
- Rollback: reinstall the previous set
  (`pip install --no-deps --force-reinstall "git+https://github.com/mlgid-project/mlgidBASE@1b6514f" pygid==0.2.10 mlgiddetect==0.2.3 onnxruntime-gpu==1.26.0`)
  and revert the GUI commits of this bump (the dropped torch-gate
  workaround matters on mlgiddetect <= 0.2.3).

### 2026-09-10 — mlgidbase 0.1.8 from PyPI (pygid 0.2.17, pygidfit 0.1.4)

The bump that finally lands the critical-angle fix. `mlgidbase 0.1.8`
(PyPI, 2026-09-10) drops the `pygidfit==0.1.3` pin that had blocked it
since 0.1.5, so the whole set moves: `mlgidbase` 0.1.5 -> 0.1.8,
`pygid` 0.2.13 -> 0.2.17, `pygidfit` 0.1.3 -> 0.1.4. `mlgiddetect`
(0.2.8), `mlgidmatch` (0.1.3) and `pygidsim` (0.1.4) are unchanged.
Full suite green before and after the GUI adaptations below, both with
the backends installed and with them blocked at import (the CI
profile).

- **mlgidbase source diff is three files.** `main.py`,
  `peak_operations.py`, `pygid_functions.py`, `nexus_operations.py`,
  `pygidfit_functions.py`, `mlgidmatch_functions.py` and `widgets.py`
  are byte-identical to 0.1.5, so every contract in Step 3 that names
  a `run_*` signature or `_track_peaks` still holds verbatim (checked
  by diffing the installed 0.1.8 tree against `git archive v0.1.5`).
  What changed: `__init__.py` (version), `mlgiddetect_functions.py`,
  `visualization.py`.
- **`load_config` rewritten** (`mlgiddetect_functions.py`), and one
  change reaches the GUI: `model_type` is now applied with a real
  assignment on EVERY branch, where 0.1.5's str-config branch dropped
  it (`config.MODEL_TYPE == model_type`, a comparison). An explicit
  combo choice therefore beats a config file's own `MODEL:  TYPE:`
  instead of losing to it. `detection_model.resolve_model_key`, which
  decides which weights the pre-flight downloads, is updated to match
  and now also understands the new `dict` config form (mlgidbase
  flattens `{SECTION: {KEY: value}}` onto `SECTION_KEY`). Regression:
  `test_resolve_model_key_lets_the_combo_override_a_config_file`,
  `test_resolve_model_key_reads_a_dict_config`.
- **`set_valid_config` / `set_postprocessing_config` are gone**, and
  that is a no-op against mlgiddetect 0.2.8: every value they forced
  (`PREPROCESSING_CUDA=False`, `FLIPHORIZONTAL=False`,
  `QUAZIPOLAR=False`, `POLAR_SHAPE=[512, 1024]`,
  `POLAR_CONVERSION=True`, and dino's `POSTPROCESSING_SCORE=0.4` /
  `NMSIOU=0.4`) is already the `Config()` default. The three keys with
  no `Config` default (`LINEAR_CONTRAST`, `NO_CONTRASTCORRECTION`,
  `LINEAR_PERC_977`) appear nowhere in mlgiddetect 0.2.8 or mlgidbase,
  so dropping them changes nothing either. Detection numerics are
  unchanged by this bump.
- **mlgiddetect imports are lazy now** (`Inference` and
  `ImageContainer` moved inside `load_inference` / `run_mlgiddetect`,
  the upstream `mlgiddetect_segfault` branch). Importing mlgidbase no
  longer drags onnxruntime in. The GUI's own guards
  (`is_mlgidbase_available`, `safe_console`, the model pre-flight) are
  unaffected and stay.
- **`visualization.py`**: new `get_clims`, so
  `plot_analysis_results` defaults to 5/95 percentile colour limits
  instead of min/max, and detected boxes are drawn as one `Wedge`
  rather than two `Arc`s. The only consumer is the official figure
  export (`figure_export_window`). Its `_gather_call_kwargs` always
  sends an explicit `clims` from the Intensity spin boxes, so the new
  default never fires; the `Wedge` does change how detected peaks look
  in an exported figure. Call contract unchanged.
- **pygidfit 0.1.3 -> 0.1.4, the actual fix.**
  `process_scans.calc_smpl_hor` loses its stray `/ 10` (the docstring
  said metres while every caller passes Angstrom), so the
  sample-horizon `q_z` mask the fitting step applies is now the right
  size instead of 10x too thin. Two further changes: ring/cluster
  association in `clustering_and_errors` now measures box-edge
  distance rather than centre distance, and
  `ProcessDataFromFile.run()` skips entries whose `img_type` is not
  `img_gid_q` when `entry is None`, which the GUI never hits:
  `MainWindow._run_pipeline` expands "all entries" into one command
  per concrete entry before mlgidbase is called.
  **Consequence for users:** with a non-zero Critical angle, fitting
  results change (the mask now really masks). The GUI's manual 2D fit
  imports the same `img_preprocessing`, so preview and pipeline stay
  byte-identical to each other. The wrong "Critical angle:" tooltip in
  `pipeline_panel.py` is fixed in the same commit.
- **pygid 0.2.13 -> 0.2.17** (diffed installed 0.2.13 against the
  0.2.17 sdist; `conversion.py`, `coordmaps.py`, `dataloader.py`,
  `datasaver.py`, `expparams.py`, `nexus_reader.py`, `__init__.py`):
  - `ExpMetadata` now validates on `__setattr__`: only `str`,
    `Number`, `bool`, and lists/tuples of those. The GUI's
    `_build_exp_metadata` stores the panel's key/value strings, so it
    passes; `None` would not, which is worth remembering if that panel
    ever grows a "clear this field" path. `SampleMetadata` sanitizes
    keys (spaces/hyphens to `_`, brackets stripped, collisions raise),
    so a YAML key with spaces now lands under a different name than it
    used to.
  - `Conversion.dataset` default changed from `'measurement/eiger4m'`
    to `None`. The GUI always passes `dataset=scan.entry`, so no
    effect.
  - Profile helpers (`horiz_profile_gid` / `vert_profile_gid` and the
    polar profiles) changed their `dang` / `q_*_range` defaults from
    fixed values to `None` (full range). The GUI calls none of them.
  - `coordmaps`: `_find_ranges_phi_` now computes phi in the LAB frame
    (`_q_smpl_to_q_lab_(self.q, ai=-self.ai)`), and `q_min` is 0
    whenever the beam centre falls inside the image rather than when
    the array is long. Polar conversion ranges can therefore differ
    from 0.2.13 for the same input; the GUI reads whatever axes the
    file carries, so nothing on our side assumes the old values.
  - `expparams`: `_calc_rot_` rebuilt on `scipy.spatial.transform`
    (new `calc_rots` / `apply_rotation` / `calc_center_after_rots`);
    the transpose branch is the fixed one. `_calc_center_` and
    `_calc_poni_` are unchanged apart from a moved log line, so
    `conversion_config`'s poni -> centre autofill still mirrors
    upstream exactly.
  - `dataloader`: `_find_3d_dataset` / `_validate_dataset` improve
    pygid's own raw-dataset search. mlgidLAB does its own walk
    (`list_raw_entries`), so this is upside only.
  - `nexus_reader`: new `set_techniques_info`; the GUI does not call
    it.
- **Still true at 0.1.8 / still GUI-side** (re-checked at this bump):
  everything in the 0.1.5 entry's "still true" list, including the
  ring-tracking clamp, `_backfill_fitted_peaks_polar_to_cartesian`,
  `_dedupe_matched_groups`, `normalize_for_pygid`, and the networkx
  packaging gap (0.1.8 still does not declare it).
- Rollback: `pip install --no-deps --force-reinstall mlgidbase==0.1.5
  pygid==0.2.13 pygidfit==0.1.3` and revert the GUI commit of this
  bump (the `resolve_model_key` change is wrong against <= 0.1.6).

### 2026-10-06 — pygidsim 0.1.7 (pygidfit 0.1.5 held back)

A full PyPI sweep of the six backends found exactly two releases ahead
of the pins: `pygidsim` 0.1.7 and `pygidfit` 0.1.5. `mlgidbase` (0.1.8),
`mlgiddetect` (0.2.8), `pygid` (0.2.17) and `mlgidmatch` (0.1.3) are
still the newest published versions, so only `pygidsim` moves:
0.1.4 -> 0.1.7. It resolves cleanly against the rest of the pinned set
(`pip install --dry-run`), because `mlgidmatch` and `pygid` both take
`pygidsim` on a floor rather than an exact pin. Full suite green before
and after, both with the backends installed and with them blocked at
import (the CI profile).

- **The reason for the bump: a real q_xy mask bug in
  `GIWAXS.giwaxs_2d`.** The simulated 2D pattern is mirrored into
  negative q_xy, then re-masked against the entry's q range. 0.1.4
  wrote that mask as
  `(q_2d_fin[0] >= q_xy_range[0]) & (q_2d_fin[1] <= q_xy_range[1])`:
  row 0 is q_xy, row 1 is q_z, so the second term compared **q_z**
  against the **q_xy** ceiling and the q_xy upper limit was never
  enforced at all. Reflections beyond q_xy_max survived into every
  simulated pattern (they passed whenever their q_z happened to sit
  under q_xy_max, which for a near-square q range is almost always).
  0.1.7 compares `q_2d_fin[0]`. Since `CifPattern` is built from
  `giwaxs_2d` output, this changes what mlgidLAB draws as the
  **Expected pattern** overlay and which candidates matching scores:
  fewer reflections, and the ones that remain are the ones inside the
  measured q range. Re-parse any cached `.pickle` built with 0.1.4;
  the GUI cannot tell which pygidsim a pickle came from.
- **Empty patterns no longer raise, they return empty arrays.**
  `giwaxs_2d` ended with `raise ValueError("There are no peaks in the
  pattern.")` when clustering left nothing; 0.1.7 returns
  `GIWAXS._return_empty(dim=2)` instead, matching what the 1D path
  already did. That turns a named error into an opaque one one frame
  later: `mlgidmatch.preprocess.cif_preprocess._create_all_possible_patterns`
  computes `intensity / intensity.max()` on the empty array and numpy
  raises `ValueError: zero-size array to reduction operation maximum
  which has no identity`. `pipeline._build_cif_pattern_from_raw` now
  catches that and re-raises a `RuntimeError` naming the CIF input and
  the likely cause (no reflections inside the entry's q range), so
  "Parse CIFs" on a CIF with nothing in range reads as a data problem
  rather than a GUI bug. Regression:
  `tests/test_cif_empty_pattern.py`.
- **`ExpParameters` is lazier, and compatible.** The `create_FF`
  constructor argument is gone, and `wavelength`, `q_max` and
  `database` became properties, with the form-factor `Database` built
  on first access to `.database` instead of inside `__init__`. The GUI
  never passed `create_FF` (zero hits in the tree) and reads only
  `.wavelength`, `.q_xy_range`, `.q_z_range` and `.ai` from an
  `ExpParameters`, so the construction call in
  `pipeline._exp_params_from_nexus` (`q_xy_max`, `q_z_max`, `ai`, `en`)
  is unchanged and the energy guard still sees the same `en` contract.
  The practical gain is that deriving parameters per entry no longer
  builds a form-factor database the GUI may never consult; the pattern
  simulation that does consult it pays for it on first touch.
- **`Crystal` moved to `pygidsim/crystal.py`**, re-exported from
  `giwaxs_sim` and now also from the package root beside
  `ExpParameters`, `GIWAXS` and `GIWAXSFromCif`. Internal imports went
  relative and `setup.py` was replaced by `pyproject.toml`. Every
  import path mlgidLAB and mlgidmatch use still resolves
  (`from pygidsim.experiment import ExpParameters` on our side), so
  this is packaging hygiene rather than an API change.
- **pygidfit 0.1.5 is blocked, and stays blocked.** `mlgidbase 0.1.8`
  pins `pygidfit==0.1.4` exactly; asking pip for 0.1.5 alongside it is
  a `ResolutionImpossible`. What is waiting there, for when mlgidbase
  moves: `Model.fit` from lmfit is replaced by a direct
  `scipy.optimize.least_squares` with a numba analytic Jacobian
  (switchable via the `PYGIDFIT_ANALYTIC_JAC` environment variable and
  auto-disabled above 6 Gaussian components), the neighbour-box ROI
  masking is vectorised, and three initial-guess changes move fit
  results: the amplitude seed becomes `np.nanpercentile(sub, 99)`
  instead of `np.nanmax(sub)`, both sigmas are divided by 2.355 (the
  FWHM-to-sigma factor), and the 1D ring `radius_width` upper bound
  becomes the box width `x1_box - x0_box` rather than the x-axis
  bound. The first two change every fitted amplitude and width seed,
  so that bump is a numerics change to re-verify against stored
  results, not a drop-in. Reported upstream so the mlgidbase pin can
  follow.
- **`tifffile`: pygid's upper bound is stale, and it does not matter.**
  Installing into an environment that already carries a recent
  `tifffile` makes pip print
  `pygid 0.2.17 requires tifffile<2024.0,>=2023.7.18, but you have
  tifffile 2026.9.15 which is incompatible`. It is a post-install
  consistency warning, not a failed install, and it can be ignored:
  **pygid never imports tifffile.** A recursive grep for `tifffile`,
  `TiffFile`, `imread` and `imwrite` across the installed `pygid`
  package returns zero hits, so the bound guards a dependency that is
  declared and unused. The only real consumer in the stack is
  mlgidDETECT (`dataloader/fromdisk.py`:
  `tifffile.imread(config.INPUT_IMGPATH)`), which is unpinned, and the
  GUI never takes that path anyway - mlgidbase hands mlgidDETECT the
  image array directly through `from_pygid`. Of the other declarers,
  `scikit-image` wants `>=2022.8.12` and `imageio` is unpinned.

  Do not downgrade to satisfy it. A **clean** environment never sees
  the warning at all: pip resolves `tifffile` to 2023.12.9, which
  satisfies pygid's ceiling and every other floor, so it only appears
  in an environment where something upgraded `tifffile` earlier.
  Reported upstream: pygid should drop the dependency or raise the
  `<2024.0` cap, which as written forces a years-old `tifffile` on
  every consumer that also uses scikit-image, imageio or silx, for a
  library pygid does not import.
- Rollback: `pip install --no-deps --force-reinstall pygidsim==0.1.4`
  and revert the pin here and in `pyproject.toml`. The GUI-side
  error-message wrap is harmless against 0.1.4 (the `ValueError` it
  catches simply never arrives).

### 2026-10-22 — mlgidbase 0.1.11, pygidfit 0.1.8, pygid 0.2.20

The whole in-house set moved on 2026-10-21/22. `pip install --dry-run`
of `mlgidbase==0.1.11 mlgiddetect==0.2.8 pygid==0.2.20 pygidfit==0.1.8
mlgidmatch==0.1.3 pygidsim==0.1.7` resolves to exactly three changes
(mlgidbase, pygid, pygidfit). `mlgidbase 0.1.11` requires
`pygid>=0.2.20`, `pygidfit>=0.1.8`, `mlgiddetect==0.2.8`,
`mlgidmatch==0.1.3`. Everything below was checked by diffing the
unpacked wheels (0.1.8 -> 0.1.11, 0.1.4 -> 0.1.8, 0.2.17 -> 0.2.20),
not taken from the upstream changelog alone. This entry is extended
stage by stage as the GUI adapts; the status of each item is at the
end.

- **Fitted ids come from the detected ids (pygidfit 0.1.8).**
  `ProcessDataFromFile.process_single_frame` now does
  `img_container_fit.id = detected_peaks['id'][img_container_fit.id]`.
  Before, `fitted_peaks['id']` was `0..N-1` by position. Fitted ids
  therefore carry gaps whenever the detected ids do. Only the
  pipeline path does this; `fit_data` (what `manual_fit.fit_one_peak`
  calls) still returns box indices and the GUI assigns the id itself.
  `file_model.rekey_fitted_ids_to_detected` becomes a no-op on files
  fitted by 0.1.8.
- **Polar-grid N-1 fix (pygidfit 0.1.8).** `boxes_preprocessing` and
  `_data2container` use `polar_shape - 1`, matching the
  `np.linspace(.., N)` grid. 2D fits are no longer biased low by
  r/1024 and a/512. Fits stored by earlier versions stay biased until
  refit.
- **Hot-pixel mask (pygidfit 0.1.8, new default).** `fit_data(...,
  hot_pixel_percentile=99.9)`; `fit_peak_cluster` and
  `fit_peak_on_ring_cluster` set every ROI pixel above the ROI's own
  99.9th percentile to NaN (ring-only clusters are not masked).
  `mlgidbase` never passes the argument, so the pipeline and the
  GUI's manual fit both get 99.9; `manual_fit` now passes it
  explicitly so the parity cannot drift silently. Narrow peaks in small
  ROIs lose their apex pixel to this.
- **`_data2container` / `compute_qzqxy_with_error` lost their q-max
  arguments and the clamp** (pygidfit 0.1.8). The GUI calls neither;
  `fit_data` keeps `q_xy_max`/`q_z_max`.
- **Matched `peak_list` holds fitted ids (mlgidbase 0.1.11).**
  `solution2container(..., fitted_ids=)` translates positions through
  `fitted_peaks['id']`. Files matched by 0.1.8 / 0.1.10 hold row
  positions. Nothing on disk distinguishes the two, and one frame can
  hold both, because 0.1.11 replaces `matched_rings_*` and
  `matched_segments_*` independently (`_merge_matched_containers`).
  The GUI therefore stamps a `peak_list_kind` attribute per dataset
  (`file_model.MATCHED_KIND_ATTR`, see the Step 3 `matched_peaks` contract).
- **`delete_peak` no longer renumbers; `add_peak` uses `max(id)+1`;
  matched and tracked entries are dropped by id** (mlgidbase 0.1.11).
  This is the convention the GUI already used.
- **`track_peaks` is a different function.** 0.1.8:
  `track_peaks(entry, threshold, length, axis, plot_params)` computed
  and plotted. 0.1.11: `track_peaks(entry, threshold, length,
  window_size=None)` computes with bounded memory (blocked IoU, scipy
  connected components), saves the table to
  `{entry}/data/analysis/general/tracked_peaks` (+ metadata under
  `{entry}/process/peak_tracking`) and returns it; `get_tracked_peaks`
  and `plot_tracked_peaks(entry, axis, ...)` are new. Table fields are
  `id, CIF, h, k, l, frame_list, peak_list` (vlen int32; `peak_list` =
  fitted ids; was `frames` / `peak_ids`, and per-frame columns in
  0.1.10). The capture hook on `_plot_tracked_peaks` the GUI used is no
  longer reached.
- **`run_fitting` clustering defaults 10 -> 2** (distance peaks,
  distance rings; `clustering_extend` stays 2). The GUI always sends
  the panel value, whose default now follows the new 2/2.
- **`mlgidmatch` results are merged per frame** (mlgidbase 0.1.11):
  running 'rings' then 'segments' keeps both. In-memory matching passes
  the q diagonal for both q maxima. `run_mlgidmatch_from_file` returns
  `(solutions, fitted_ids)`; the GUI does not call it.
- **pygid 0.2.20:** `plot_*` methods return the image object instead of
  `(fig, ax)` (the GUI never uses them); `px_size` is read as a float
  and falls back to the `detector_config` JSON; new
  `_save_tracked_peaks_data`. A tracked table adds a `general` group
  under `data/analysis`; pygid's `fill_analysis_group` counts every
  group there as a frame (checked in the tracking stage).

**Status of the items above, as adapted in this bump:**
- Fitted ids = detected ids: verified with a real `run_fitting` on a
  dataset with unordered, gapped detected ids (fitted and errors ids
  came out equal to the detected ids, in order); the GUI's re-key pass
  after a run was removed.
- Matched `peak_list` kinds: per-dataset `peak_list_kind` stamp,
  classified when absent, normalised to ids before a delete, a matching
  run and a tracking run (Step 3 `matched_*` contract).
- Tracking: `track_peaks(entry, threshold, length, window_size)` +
  `payload_from_tracked_table`; the memory routing, the capture hook and
  the amplitude recovery are gone; the table is dropped on refit and on
  appending frames, and a fitted delete removes the id from it.
  pygidfit 0.1.8 stores a ring with angle 45 (not NaN) and an infinite
  width, so upstream links rings natively; the GUI still drops those
  native ring components and tracks rings itself (radial IoU), so every
  ring is one track (`test_rings_stored_the_0_1_8_way_yield_one_ring_track_beside_the_spots`).
  Reported upstream earlier: ring tracking wants radial IoU for
  `is_ring` rows, not an angle clamp.
- Still open for the next stage: clustering defaults 2/2 in the panel,
  the explicit `hot_pixel_percentile` in the manual fit, and the
  `(n-1)` neighbour-box scale.
