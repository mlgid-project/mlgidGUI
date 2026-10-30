"""Conversion run configuration — the Qt-free vocabulary shared by the
Conversion panel (UI), the conversion engine and the workers.

Moved out of ``conversion_panel`` so the engine no longer has to import
the Qt panel module for its dataclasses (which used to form the
package's only import cycle: ``conversion`` <-> ``conversion_panel``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mlgidgui.file_model import RawEntry

# Conversion-type identifiers — kept as plain strings so ``ConversionConfig``
# stays pickleable and can pass through Qt signals without custom marshalling.
CONV_DET2Q_GID = "det2q_gid"
CONV_DET2Q = "det2q"
CONV_DET2POL_GID = "det2pol_gid"
CONV_DET2POL = "det2pol"

GEOM_GID = "GID"
GEOM_TRANSMISSION = "Transmission"

# Frame-selection modes for the Selection section.
FRAME_ALL = "All"
FRAME_SINGLE = "Single"
FRAME_LIST = "List"

OUTPUT_SEPARATE_FILES = "Separate files"
OUTPUT_SEPARATE_DATASETS = "Separate datasets in single file"
# All selected scans land as FRAMES of one entry in one (new) output
# file — the converted result browses with the frame slider and can be
# peak-tracked as a scan, unlike N separate entry_NNNN groups.
OUTPUT_SINGLE_ENTRY = "Single entry in single file"


@dataclass
class RawScan:
    """One (file, entry, frames) triple selected for conversion.

    ``frame_num`` follows the pygid convention:
    - ``None`` → all frames in the dataset
    - ``int`` → a single frame index
    - ``list[int]`` → an explicit subset

    ``frame_offset`` is where this scan's frame 0 sits on the whole
    selection's frame axis. It only matters for a per-frame angle of
    incidence: pygid looks the angle up as ``ai[frame + offset]``, so a
    batch of single-frame scans needs each one to say which frame of the
    scan it actually is. Defaults to 0, which is correct for every
    single-scan run and for a scalar angle.
    """

    file_path: Path
    entry: str
    frame_num: int | list[int] | None = None
    frame_offset: int = 0


def _expand_fabio_scans(
    entry: RawEntry, frame_num: int | list[int] | None, base: int = 0,
) -> list[RawScan]:
    """Turn a selected fabio stack entry into one ``RawScan`` per frame-file.

    pygid reads a fabio image via ``fabio.open(path).data`` (frame 0 only)
    and ignores both ``dataset`` and ``frame_num``, so each converted frame
    must be its OWN single-file scan (``entry=""``, ``frame_num=None``).
    Here ``frame_num`` selects WHICH stack frames to convert (and therefore
    which files), following the panel's frame mode: ``None`` → every file,
    ``int`` → one, ``list`` → a subset. Out-of-range indices are dropped.

    ``frame_offset`` is ``base`` plus the file's **original** index in the
    stack, not its position in the filtered result: converting frames 3
    and 7 of a fourteen-image stack must read angles 3 and 7, not 0 and
    1. ``base`` is where this entry starts on the selection's frame axis.
    """
    fmap = entry.frame_map or []
    n = len(fmap)
    if frame_num is None:
        idxs: list[int] = list(range(n))
    elif isinstance(frame_num, int):
        idxs = [frame_num] if 0 <= frame_num < n else []
    else:
        idxs = [i for i in frame_num if 0 <= i < n]
    return [
        RawScan(
            file_path=fmap[i][0], entry="", frame_num=None,
            frame_offset=base + i,
        )
        for i in idxs
    ]


@dataclass
class ConversionConfig:
    """Everything the conversion engine needs except the scan list."""

    geometry: str = GEOM_GID
    conv_type: str = CONV_DET2Q_GID
    # Orientation flags — passed to pygid.CoordMaps. Default True to
    # match the pygid example notebook's recommended workflow: with
    # both off (pygid's library default), the converted q ranges can
    # extend into negative quadrants depending on detector flips +
    # beam center, which is rarely what the user wants when reviewing
    # a single GIWAXS frame. Users who want the full quadrant range
    # can uncheck either box in the Conversion panel.
    vert_positive: bool = True
    hor_positive: bool = True
    # Reciprocal-space ranges. Empty (None) means "auto" (pygid's default).
    dq: float | None = None
    dang: float | None = None
    q_xy_range: tuple[float, float] | None = None
    q_z_range: tuple[float, float] | None = None
    q_x_range: tuple[float, float] | None = None
    q_y_range: tuple[float, float] | None = None
    radial_range: tuple[float, float] | None = None
    angular_range: tuple[float, float] | None = None
    # Experimental params.
    poni_path: Path | None = None
    mask_path: Path | None = None
    # One angle for every frame, or one per frame of the selection's
    # frame axis (see ``ai_values`` and ``RawScan.frame_offset``).
    ai: float | list[float] | None = None
    # Per-field manual overrides (centerX, centerY, SDD, wavelength,
    # fliplr, flipud, transp). Filled by the panel when the user changes
    # the corresponding field; otherwise pygid reads the value from the
    # PONI file.
    expmeta_overrides: dict = field(default_factory=dict)
    # Sample metadata YAML text (parsed by the engine via yaml.safe_load).
    smplmeta_yaml: str = ""
    # Experimental metadata key/value pairs from the metadata table.
    expmeta_kv: dict[str, str] = field(default_factory=dict)
    # Output config.
    output_mode: str = OUTPUT_SEPARATE_FILES
    output_dir: Path | None = None
    # Optional custom output filename. Behaviour depends on output_mode:
    #   - separate-datasets: this becomes the single output filename
    #     (defaults to "converted.h5").
    #   - separate-files: with one raw file, used verbatim; with multiple,
    #     used as a prefix (the raw stem is appended).
    # Empty string falls through to the per-mode defaults.
    output_filename: str = ""
    overwrite_file: bool = True
    overwrite_dataset: bool = False
    # Append mode: instead of creating a new entry_NNNN, append the
    # converted frames to ``append_entry`` (an existing entry group) of
    # the existing output file. Implies no file/group overwrite.
    append_frames: bool = False
    append_entry: str = ""


def parse_poni_overrides(path: Path) -> dict[str, float]:
    """Parse a pyFAI ``.poni`` file into override-field values.

    Returns a subset of ``{"SDD", "wavelength", "centerX", "centerY"}``
    in the panel's units (SDD in m, wavelength in Å, centers in px) —
    whatever the file allows to be derived. The math mirrors
    ``pygid.ExpParams`` exactly (``read_from_dict`` + ``_calc_center_``,
    rotation-aware, single square ``px_size`` from ``pixel1``), so the
    pre-filled values equal what pygid would compute internally from the
    same PONI — sending them back as overrides is a no-op until edited.

    Pure text parsing — no pyFAI/pygid import (their import chains take
    seconds and the pipeline extra may be absent). Pixel size comes from
    a ``pixel1`` / ``pixelsize1`` key or the ``Detector_config`` JSON;
    a named detector without explicit pixel size yields only SDD +
    wavelength. Raises ``OSError`` if the file can't be read; malformed
    values just drop the affected outputs.
    """
    import json
    import math

    data: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip().lower()] = value.strip()

    def _float(key: str) -> float | None:
        if key not in data:
            return None
        try:
            return float(data[key])
        except ValueError:
            return None

    out: dict[str, float] = {}
    sdd = _float("distance")
    if sdd is None:
        sdd = _float("dist")
    if sdd is not None:
        out["SDD"] = sdd
    wl = _float("wavelength")
    if wl is not None:
        out["wavelength"] = wl * 1e10  # m → Å (pygid's internal unit)

    px = _float("pixel1")
    if px is None:
        px = _float("pixelsize1")
    if px is None and "detector_config" in data:
        try:
            px = float(json.loads(data["detector_config"]).get("pixel1"))
        except (ValueError, TypeError):
            px = None
    poni1, poni2 = _float("poni1"), _float("poni2")
    rot1 = _float("rot1") or 0.0
    rot2 = _float("rot2") or 0.0
    if None not in (sdd, poni1, poni2, px) and px > 0:
        # pygid.ExpParams._calc_center_ (no flips at this stage — flips
        # are applied by pygid later, on top of these values).
        out["centerY"] = (sdd * math.tan(rot2) / math.cos(rot1) + poni1) / px
        out["centerX"] = (-sdd * math.tan(rot1) + poni2) / px
    return out
