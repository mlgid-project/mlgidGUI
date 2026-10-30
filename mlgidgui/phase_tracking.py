"""Adapter around mlgidBASE's official peak tracking (``track_peaks``).

Upstream (mlgidbase >= 0.1.11) tracks FITTED peaks across a scan's
frames: boxes ``(angle ± angle_width/2, radius ± radius_width/2)`` for
every fitted peak of every frame are IoU-compared (in bounded-memory row
blocks), thresholded into a graph, and the connected components become
the tracks; components with more than ``length`` members survive. An
optional ``window_size`` drops links between peaks more than that many
frames apart. The result is SAVED into the file as a ``tracked_peaks``
table (one row per track: ``id``, the majority-vote ``CIF`` / ``h`` /
``k`` / ``l``, and the parallel vlen arrays ``frame_list`` /
``peak_list`` holding each member's frame and fitted-peak id) and
returned.

``payload_from_tracked_table`` turns that table into the
``TrackingPayload`` the views consume: member arrays over ALL fitted
peaks of the entry (read straight from the file, so every value is
exact), the tracks as lists of member indices, and the members'
``(frame, fitted id)`` carried along in ``peak_id`` so nothing has to be
reconstructed. (Before mlgidbase 0.1.11 the member data was only
reachable through a hook on a plotting call and the ids were guessed
back from coordinates.)

Everything else here is pure numpy/scipy helpers for the views: derived
axes (convention pinned to ``mlgidgui.polar.polar_to_qxyz``: angle in
degrees from +q_xy toward +q_z), per-track spans/means, amplitude
smoothing (median filter + nanmax normalization, mirroring the
prototype notebook), the GUI-side ring tracking, gap-fill planning and
the track-to-phase attribution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.ndimage import median_filter

from mlgidgui.file_model import (
    ANALYSIS_REL,
    FRAME_KEY_FMT,
    IMG_REL,
    TRACKED_PEAKS_REL,
    TRACKING_PROCESS_REL,
)

logger = logging.getLogger(__name__)

# Axis names offered to the user. "radius" follows upstream's meaning:
# the fitted ``radius`` column IS the q magnitude, so radius == |q|.
AXIS_NAMES = ("radius", "angle", "amplitude", "q_xy", "q_z")

AXIS_LABELS = {
    "radius": "|q| (Å⁻¹)",
    "angle": "angle (°)",
    "amplitude": "amplitude (arb. u.)",
    "q_xy": "q_xy (Å⁻¹)",
    "q_z": "q_z (Å⁻¹)",
}

def read_fitted_members(file_path, entry: str) -> dict:
    """Every fitted peak of ``entry`` as parallel arrays, frames in
    numeric order, rows verbatim (frames without a ``fitted_peaks``
    dataset are skipped): ``angle``, ``angle_width``, ``radius``,
    ``radius_width``, ``amplitude``, ``q_xy``, ``q_z``, ``frame`` and
    ``peak_id``. These are the members a ``TrackingPayload`` indexes.

    Reads the file with h5py only. Raises ``ValueError`` when the entry
    has no fitted rows.
    """
    cols: dict[str, list] = {
        "angle": [], "angle_width": [], "radius": [], "radius_width": [],
        "amplitude": [], "q_xy": [], "q_z": [], "frame": [], "peak_id": [],
    }
    with h5py.File(file_path, "r") as f:
        sig = f.get(f"{entry}/{IMG_REL}")
        n_frames = int(sig.shape[0]) if isinstance(sig, h5py.Dataset) else 0
        ana = f.get(f"{entry}/{ANALYSIS_REL}")
        for frame in range(n_frames):
            grp = ana.get(FRAME_KEY_FMT.format(frame)) if ana is not None else None
            ds = grp.get("fitted_peaks") if grp is not None else None
            if not isinstance(ds, h5py.Dataset):
                continue
            rows = ds[()]
            for name in (
                "angle", "angle_width", "radius", "radius_width",
                "amplitude", "q_xy", "q_z",
            ):
                cols[name].append(np.asarray(rows[name], dtype=float))
            cols["frame"].append(np.full(len(rows), frame, dtype=int))
            cols["peak_id"].append(np.asarray(rows["id"], dtype=int))
    if not cols["frame"]:
        raise ValueError(f"entry {entry!r} has no fitted_peaks datasets")
    out = {k: np.concatenate(v) for k, v in cols.items()}
    if out["angle"].size == 0:
        raise ValueError(f"entry {entry!r} has no fitted rows")
    return out


@dataclass
class TrackingPayload:
    """Everything one upstream ``track_peaks`` run computed.

    The four member arrays are parallel over ALL fitted-peak instances
    of the scan (every fitted row of every frame, concatenated in
    upstream's frame order). ``components`` holds the surviving tracks
    as lists of indices into those arrays. Note a component may contain
    several members of the SAME frame (upstream clusters spatially
    overlapping same-frame boxes too), so "frames present" and "member
    count" are distinct quantities.
    """

    entry: str
    threshold: float
    length: int
    q_xy: np.ndarray
    q_z: np.ndarray
    frame_num: np.ndarray
    amplitude: np.ndarray
    components: list = field(default_factory=list)  # list[list[int]]
    #: Fitted-peak id of every member (parallel to the arrays above).
    #: Set by ``payload_from_tracked_table``; None only for hand-built
    #: payloads (tests), where ``member_ids`` falls back to matching
    #: coordinates.
    peak_id: "np.ndarray | None" = None
    #: Frame window the tracking was run with (None = unlimited).
    window_size: "int | None" = None
    #: Saved-table members that named no fitted row (see
    #: ``payload_from_tracked_table``); 0 for a table made from the
    #: current fits.
    n_unresolved: int = 0

    def __post_init__(self) -> None:
        self.q_xy = np.asarray(self.q_xy, dtype=float)
        self.q_z = np.asarray(self.q_z, dtype=float)
        self.frame_num = np.asarray(self.frame_num, dtype=int)
        self.amplitude = np.asarray(self.amplitude, dtype=float)
        n = self.q_xy.size
        if not (self.q_z.size == self.frame_num.size == self.amplitude.size == n):
            raise ValueError(
                "TrackingPayload member arrays disagree in length: "
                f"q_xy={n}, q_z={self.q_z.size}, "
                f"frame_num={self.frame_num.size}, "
                f"amplitude={self.amplitude.size}"
            )
        if self.peak_id is not None:
            self.peak_id = np.asarray(self.peak_id, dtype=int)
            if self.peak_id.size != n:
                raise ValueError(
                    "TrackingPayload peak_id disagrees in length: "
                    f"peak_id={self.peak_id.size}, q_xy={n}"
                )
        self.components = [
            sorted(int(i) for i in comp) for comp in self.components
        ]

    @property
    def n_members(self) -> int:
        return int(self.q_xy.size)

    @property
    def n_tracks(self) -> int:
        return len(self.components)

    @property
    def radius(self) -> np.ndarray:
        """|q| of every member. The fitted ``radius`` column upstream
        is exactly this magnitude, recomputed here from (q_xy, q_z)."""
        return np.hypot(self.q_xy, self.q_z)

    @property
    def angle(self) -> np.ndarray:
        """Polar angle in degrees, +q_xy toward +q_z — the same
        convention as ``mlgidgui.polar.polar_to_qxyz``."""
        return np.rad2deg(np.arctan2(self.q_z, self.q_xy))

    def axis_values(self, name: str) -> np.ndarray:
        if name not in AXIS_NAMES:
            raise ValueError(
                f"unknown axis {name!r}; valid: {AXIS_NAMES}"
            )
        return getattr(self, name)

    # --- Per-track derivations (k = index into components) ---

    def track_members(self, k: int) -> np.ndarray:
        """Member indices of track ``k``, sorted by frame number."""
        idx = np.asarray(self.components[k], dtype=int)
        return idx[np.argsort(self.frame_num[idx], kind="stable")]

    def track_span(self, k: int) -> tuple:
        """``(first_frame, last_frame, frames_present, n_members)``."""
        frames = self.frame_num[np.asarray(self.components[k], dtype=int)]
        return (
            int(frames.min()),
            int(frames.max()),
            int(np.unique(frames).size),
            int(frames.size),
        )

    def track_mean(self, name: str, k: int) -> float:
        vals = self.axis_values(name)[
            np.asarray(self.components[k], dtype=int)
        ]
        return float(np.nanmean(vals))


def payload_from_tracked_table(
    file_path, entry: str, tracked, threshold: float, length: int,
    window_size: "int | None" = None,
) -> TrackingPayload:
    """Build a ``TrackingPayload`` from upstream's ``tracked_peaks`` table.

    ``tracked`` is the structured array ``Analysis.track_peaks`` returned
    (or ``read_tracked_peaks`` read back): one row per track with the
    vlen int arrays ``frame_list`` and ``peak_list``. The members are
    every fitted peak of the entry (``read_fitted_members``); each
    track becomes the sorted list of the members its ``(frame, id)``
    pairs name. Pairs naming no fitted row (a table older than the
    current fits) are skipped, and a track left empty by that is
    dropped. Track order is the table's row order, which is upstream's
    (smallest member first).

    Raises ``ValueError`` when the table lacks the 0.1.11 field names,
    i.e. it was written by an older mlgidbase.
    """
    names = tracked.dtype.names or ()
    if "frame_list" not in names or "peak_list" not in names:
        raise ValueError(
            "the tracked_peaks table predates the frames/peak_ids -> "
            "frame_list/peak_list rename (mlgidbase < 0.1.11); run "
            "tracking again to rebuild it."
        )
    members = read_fitted_members(file_path, entry)
    index = {
        (int(fr), int(pid)): i
        for i, (fr, pid) in enumerate(
            zip(members["frame"], members["peak_id"])
        )
    }
    components: list = []
    unresolved = 0
    for row in tracked:
        keys = [
            (int(fr), int(pid))
            for fr, pid in zip(
                np.atleast_1d(row["frame_list"]),
                np.atleast_1d(row["peak_list"]),
            )
        ]
        comp = sorted(index[key] for key in keys if key in index)
        unresolved += len(keys) - len(comp)
        if comp:
            components.append(comp)
    logger.info(
        "tracking: %s fitted peaks across the entry, %d track(s)",
        f"{members['frame'].size:,}", len(components),
    )
    return TrackingPayload(
        entry=str(entry),
        threshold=float(threshold),
        length=int(length),
        q_xy=members["q_xy"],
        q_z=members["q_z"],
        frame_num=members["frame"],
        amplitude=members["amplitude"],
        components=components,
        peak_id=members["peak_id"],
        window_size=None if window_size is None else int(window_size),
        n_unresolved=unresolved,
    )


@dataclass
class SavedTracking:
    """The tracking table an entry carries in the file, with the settings
    mlgidbase recorded beside it (``process/peak_tracking``). Settings
    are None when the file has no such record."""

    table: np.ndarray
    threshold: "float | None" = None
    length: "int | None" = None
    window_size: "int | None" = None
    program: str = ""
    version: str = ""
    date: str = ""


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def read_saved_tracking(file_path, entry: str) -> "SavedTracking | None":
    """The entry's saved ``tracked_peaks`` table, or None when there is
    none or it was written before the 0.1.11 field rename (the caller
    cannot use it either way; ``saved_tracking_problem`` says which)."""
    with h5py.File(file_path, "r") as f:
        ds = f.get(f"{entry}/{TRACKED_PEAKS_REL}")
        if not isinstance(ds, h5py.Dataset):
            return None
        table = ds[()]
        names = table.dtype.names or ()
        if "frame_list" not in names or "peak_list" not in names:
            return None
        meta = f.get(f"{entry}/{TRACKING_PROCESS_REL}")
        out = SavedTracking(table=table)
        if isinstance(meta, h5py.Group):
            def _get(key):
                item = meta.get(key)
                if isinstance(item, h5py.Dataset):
                    return item[()]
                return meta.attrs.get(key)
            thr, length, window = _get("threshold"), _get("length"), _get("window_size")
            out.threshold = None if thr is None else float(thr)
            out.length = None if length is None else int(length)
            # mlgidbase stores "no window" as the string/None-ish value;
            # anything that is not a positive integer means unlimited.
            try:
                out.window_size = int(window) if window is not None and int(window) > 0 else None
            except (TypeError, ValueError):
                out.window_size = None
            out.program = _text(_get("program") or "")
            out.version = _text(_get("version") or "")
            out.date = _text(_get("date") or "")
        return out


def saved_tracking_problem(file_path, entry: str) -> str:
    """Why ``read_saved_tracking`` returned None: ``"none"`` (no table),
    ``"old_layout"`` (a pre-0.1.11 table) or ``""`` (it is readable)."""
    with h5py.File(file_path, "r") as f:
        ds = f.get(f"{entry}/{TRACKED_PEAKS_REL}")
        if not isinstance(ds, h5py.Dataset):
            return "none"
        names = ds.dtype.names or ()
        if "frame_list" not in names or "peak_list" not in names:
            return "old_layout"
    return ""


@dataclass
class TrackingState:
    """Everything the Scan-tracking panel and the phase views need from one
    tracking result: the payload plus what is derived from it. Built by
    ``build_tracking_state`` (thread-safe: h5py + numpy only), applied by
    the GUI thread."""

    payload: TrackingPayload
    ids: list
    tables: dict
    ring_tracks: set
    track_phases: dict
    member_phases: dict
    matched_tables: dict


def ring_track_set(payload: TrackingPayload, ids: list, tables: dict) -> set:
    """Indices of the tracks whose members are rings (any ring member;
    rings and spots never share a component)."""
    rings: set = set()
    for k, comp in enumerate(payload.components):
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            t = tables.get(int(tagged[0]))
            if t is None or len(t) == 0:
                continue
            hits = np.flatnonzero(np.asarray(t.ids, dtype=int) == int(tagged[1]))
            if hits.size == 1 and bool(t.is_ring[int(hits[0])]):
                rings.add(k)
                break
    return rings


def read_fitted_tables(file_path, entry: str, payload: TrackingPayload) -> dict:
    """``{frame: fitted PeakTable}`` for every frame the payload touches,
    read through one handle ({} on any error)."""
    from mlgidgui import file_model

    tables: dict = {}
    try:
        with h5py.File(file_path, "r") as f:
            for frame in sorted({int(x) for x in payload.frame_num}):
                tables[frame] = file_model.read_peaks(f, entry, frame)["fitted"]
    except Exception:
        logger.debug("read_fitted_tables failed", exc_info=True)
        return {}
    return tables


def read_matched_tables(
    file_path, entry: str, payload: TrackingPayload, fitted_tables: dict,
) -> dict:
    """``{frame: [MatchedStructure]}`` for the payload's frames (one
    handle; {} on any error or before Matching has run)."""
    from mlgidgui import file_model

    out: dict = {}
    try:
        with h5py.File(file_path, "r") as f:
            for frame in sorted({int(x) for x in payload.frame_num}):
                out[frame] = file_model.read_matched_peaks(
                    f, entry, frame, fitted_tables.get(frame),
                )
    except Exception:
        logger.debug("read_matched_tables failed", exc_info=True)
        return {}
    return out


def build_tracking_state(
    file_path, entry: str, payload: TrackingPayload,
) -> TrackingState:
    """Derive a payload's display state: member ids, the GUI-side ring
    tracks, and the matched phases.

    Rings are tracked here by 1-D radial IoU (rings span the whole arc,
    so radius intervals are the right overlap measure) and appended to
    the payload. Upstream links rings natively since pygidfit 0.1.8
    stores them with angle 45; those components are dropped first so
    every physical ring yields exactly ONE track. Mutates
    ``payload.components`` accordingly.
    """
    tables = read_fitted_tables(file_path, entry, payload)
    ids = (
        member_ids(payload, tables) if tables else [None] * payload.n_members
    )
    if tables:
        native_ring = ring_track_set(payload, ids, tables)
        if native_ring:
            payload.components = [
                c for k, c in enumerate(payload.components)
                if k not in native_ring
            ]
        ring_comps = track_rings(
            payload, ids, tables, payload.threshold, payload.length,
        )
        if ring_comps:
            payload.components = payload.components + ring_comps
    ring_tracks = ring_track_set(payload, ids, tables)
    matched_tables = read_matched_tables(file_path, entry, payload, tables)
    return TrackingState(
        payload=payload,
        ids=ids,
        tables=tables,
        ring_tracks=ring_tracks,
        track_phases=match_tracks_to_structures(payload, ids, matched_tables),
        member_phases=member_matched_phases(payload, ids, matched_tables),
        matched_tables=matched_tables,
    )


def load_saved_tracking_state(
    file_path, entry: str, *, force: bool = False,
) -> tuple:
    """Build the tracking state from the table saved in the file.

    Returns ``(state, info)``. ``state`` is None when there is nothing to
    show; ``info`` always says why or what was loaded:
    ``{"status": "loaded" | "none" | "old_layout" | "stale" | "error",
    "saved": SavedTracking | None, "n_tracks": int, "unresolved": int,
    "members": int, "message": str}``.

    A table whose members name fitted peaks that are no longer there
    (``payload.n_unresolved`` > 0: written for other fits, or edited
    outside the GUI) is ``"stale"`` and not loaded, unless ``force``.
    Never raises on bad data. Thread-safe.
    """
    info: dict = {
        "status": "none", "saved": None, "n_tracks": 0, "unresolved": 0,
        "members": 0, "message": "",
    }
    try:
        saved = read_saved_tracking(file_path, entry)
        if saved is None:
            info["status"] = saved_tracking_problem(file_path, entry)
            return None, info
        info["saved"] = saved
        payload = payload_from_tracked_table(
            file_path, entry, saved.table,
            0.5 if saved.threshold is None else saved.threshold,
            10 if saved.length is None else saved.length,
            saved.window_size,
        )
        members = int(sum(len(np.atleast_1d(r["peak_list"])) for r in saved.table))
        info.update(unresolved=payload.n_unresolved, members=members)
        if payload.n_unresolved and not force:
            info["status"] = "stale"
            info["message"] = (
                f"{payload.n_unresolved} of {members} saved track members "
                "no longer match a fitted peak."
            )
            return None, info
        state = build_tracking_state(file_path, entry, payload)
        info.update(status="loaded", n_tracks=payload.n_tracks)
        return state, info
    except Exception as exc:
        logger.debug("load_saved_tracking_state failed", exc_info=True)
        info.update(status="error", message=str(exc))
        return None, info


#: Layout of the saved table, as mlgidbase >= 0.1.11 writes it
#: (``peak_operations.build_tracked_peaks_dtype``): one row per track,
#: ``frame_list`` / ``peak_list`` the parallel vlen arrays of each member's
#: frame and fitted-peak id. Kept here so writing needs no backend.
def tracked_peaks_dtype() -> np.dtype:
    vlen = h5py.vlen_dtype(np.int32)
    return np.dtype([
        ("id", "i4"), ("CIF", "S64"), ("h", "f4"), ("k", "f4"), ("l", "f4"),
        ("frame_list", vlen), ("peak_list", vlen),
    ])


def tracking_table_from_payload(
    payload: TrackingPayload, ids: list, matched_tables: dict,
) -> np.ndarray:
    """The saved-table rows for the payload's CURRENT tracks.

    Every track becomes one row (ids renumbered ``0..n-1`` in track
    order, as upstream numbers them): members as the exact
    ``(frame, fitted id)`` pairs of ``ids`` sorted by frame (members
    without an id are left out), and the phase columns from a majority
    vote over the members' matched structures - the structure that claims
    most members wins, ties go to the higher probability, ``unmatched`` /
    NaN when none does. This is upstream's own attribution rule, so a
    table the GUI wrote reads like one mlgidbase wrote.
    """
    from collections import Counter

    claims: dict = {}  # (frame, fitted id) -> [(key, probability)]
    for frame, structures in matched_tables.items():
        for st in structures or ():
            key = (f"{st.cif}.cif", int(st.h), int(st.k), int(st.l))
            for pid in np.asarray(st.peaks.ids, dtype=int):
                claims.setdefault((int(frame), int(pid)), []).append(
                    (key, float(st.probability))
                )
    rows = np.empty(len(payload.components), dtype=tracked_peaks_dtype())
    for k, comp in enumerate(payload.components):
        pairs = sorted(
            (int(ids[int(i)][0]), int(ids[int(i)][1]))
            for i in comp if ids[int(i)] is not None
        )
        votes: Counter = Counter()
        best: dict = {}
        for pair in pairs:
            for key, prob in claims.get(pair, ()):
                votes[key] += 1
                best[key] = max(prob, best.get(key, -1.0))
        rows["id"][k] = k
        rows["CIF"][k] = b"unmatched"
        rows["h"][k] = rows["k"][k] = rows["l"][k] = np.nan
        if votes:
            cif, h, kk, ll = max(votes, key=lambda key: (votes[key], best[key]))
            rows["CIF"][k] = cif.encode()[:64]
            rows["h"][k], rows["k"][k], rows["l"][k] = h, kk, ll
        rows["frame_list"][k] = np.asarray([p[0] for p in pairs], dtype=np.int32)
        rows["peak_list"][k] = np.asarray([p[1] for p in pairs], dtype=np.int32)
    return rows


def write_saved_tracking(
    file_path, entry: str, payload: TrackingPayload, ids: list,
    matched_tables: dict,
) -> int:
    """Replace the entry's saved ``tracked_peaks`` table with the payload's
    current tracks (``tracking_table_from_payload``) and note in
    ``process/peak_tracking`` that mlgidGUI edited it. Returns the number
    of tracks written. Needs the file free of read handles (``r+``)."""
    from mlgidgui import __version__

    table = tracking_table_from_payload(payload, ids, matched_tables)
    with h5py.File(file_path, "r+") as f:
        ana = f.require_group(f"{entry}/{ANALYSIS_REL}")
        general = ana.require_group("general")
        if "tracked_peaks" in general:
            del general["tracked_peaks"]
        general.create_dataset(
            "tracked_peaks", data=table, dtype=table.dtype,
        )
        meta = f.require_group(f"{entry}/{TRACKING_PROCESS_REL}")
        meta.attrs.setdefault("NX_class", "NXprocess")
        for key, value in (
            ("threshold", float(payload.threshold)),
            ("length", int(payload.length)),
            ("window_size", int(payload.window_size or 0)),
        ):
            if key in meta:
                del meta[key]
            meta.create_dataset(key, data=value)
        meta.attrs["edited_by"] = f"mlgidGUI {__version__}"
    return int(len(table))


def smooth_normalize_amplitude(
    frames: np.ndarray,
    amplitudes: np.ndarray,
    window: int = 7,
    normalize: bool = True,
) -> tuple:
    """Frame-sorted, median-smoothed, optionally nanmax-normalized
    amplitude trace for one track — the processing the prototype
    notebook applies before plotting amplitude evolution.

    ``window`` is the median-filter size (1 = no smoothing;
    mode='nearest' like the notebook). Returns
    ``(sorted_frames, processed_amplitudes)``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    frames = np.asarray(frames)
    amps = np.asarray(amplitudes, dtype=float)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    amps = amps[order].copy()
    if window > 1 and amps.size:
        amps = median_filter(amps, size=min(window, amps.size), mode="nearest")
    if normalize and amps.size:
        peak = np.nanmax(amps)
        if np.isfinite(peak) and peak > 0:
            amps = amps / peak
    return frames, amps


# Fit-box geometry for an interpolated RING: pygidfit classifies a box
# as a ring geometrically (its angular extent covers the reachable
# span — ``find_box_type``), so the injected box must span the whole
# 0-90 deg quadrant with FINITE numbers (an ``inf`` width breaks
# ``boxes_preprocessing``'s integer grid math). The stored fitted ring
# row still uses mlgidGUI's ring convention (``angle_width = inf``).
RING_BOX_ANGLE = 45.0
RING_BOX_ANGLE_WIDTH = 90.0


def plan_gap_fills(
    payload: TrackingPayload, ids: list, fitted_tables: dict,
    ring_tracks=frozenset(),
) -> dict:
    """Plan the detected-box injections that fill each track's GAP frames.

    A gap frame is an integer frame strictly between two of a track's
    real (anchor) frames where the track has no fitted member. For each
    such frame this returns an injection spec:

    * position (``radius`` — and ``angle`` for spots): linearly
      interpolated between the bracketing anchors (frame-distance
      weighted);
    * box size (``radius_width`` — and ``angle_width`` for spots): the
      mean of the two bracketing anchors' sizes — a seed only; the
      subsequent 2D fit refines both position and size against the
      actual image;
    * ``is_ring``: ``True`` for tracks in ``ring_tracks`` (component
      indices — the GUI-side ring tracks). A ring has no azimuthal
      position, so its spec carries the quadrant-spanning fit box
      (``RING_BOX_ANGLE`` / ``RING_BOX_ANGLE_WIDTH``) instead of an
      interpolated angle, and only radius / radius_width interpolate.

    Anchor geometry comes from the fitted tables via the reconstructed
    ``ids`` (same-frame members average into one anchor, mirroring the
    old display interpolation). Tracks with fewer than two usable
    anchors have no bracketed gaps and contribute nothing. Anchor
    frames themselves are never emitted — the real fitted peak is
    already there.

    Returns ``{frame: [{"track", "radius", "angle", "radius_width",
    "angle_width", "is_ring"}, ...]}`` — pure planning data for the
    ``interpolate_tracks`` pipeline op; nothing is read from or written
    to any file here.
    """
    fills: dict = {}
    for comp_idx, comp in enumerate(payload.components):
        is_ring_track = comp_idx in ring_tracks
        anchors: dict = {}
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            frame, pid = int(tagged[0]), int(tagged[1])
            table = fitted_tables.get(frame)
            if table is None or len(table) == 0:
                continue
            hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == pid)
            if hits.size != 1:
                continue
            j = int(hits[0])
            if bool(table.is_ring[j]) != is_ring_track:
                continue  # defensive: never mix ring and spot anchors
            if is_ring_track:
                # Rings: radial geometry only. Stored ring rows carry
                # angle 45 / angle_width inf (or NaN from the pipeline
                # fit), neither of which may leak into interpolation.
                anchors.setdefault(frame, []).append(np.array([
                    float(table.radius[j]),
                    RING_BOX_ANGLE,
                    float(table.radius_width[j]),
                    RING_BOX_ANGLE_WIDTH,
                ]))
            else:
                anchors.setdefault(frame, []).append(np.array([
                    float(table.radius[j]),
                    float(table.angle[j]),
                    float(table.radius_width[j]),
                    float(table.angle_width[j]),
                ]))
        if len(anchors) < 2:
            continue
        seq = sorted(
            (frame, np.mean(np.vstack(rows), axis=0))
            for frame, rows in anchors.items()
        )
        for (f0, v0), (f1, v1) in zip(seq, seq[1:]):
            for frame in range(f0 + 1, f1):   # strictly between = gaps
                t = (frame - f0) / (f1 - f0)
                fills.setdefault(int(frame), []).append({
                    "track": int(comp_idx),
                    "radius": float(v0[0] + t * (v1[0] - v0[0])),
                    "angle": (
                        RING_BOX_ANGLE if is_ring_track
                        else float(v0[1] + t * (v1[1] - v0[1]))
                    ),
                    "radius_width": float((v0[2] + v1[2]) / 2.0),
                    "angle_width": (
                        RING_BOX_ANGLE_WIDTH if is_ring_track
                        else float((v0[3] + v1[3]) / 2.0)
                    ),
                    "is_ring": is_ring_track,
                })
    return fills


def _connected_components(n: int, pairs) -> list:
    """Union-find connected components over ``n`` nodes and an iterable
    of ``(a, b)`` edges. Returns a list of node-index lists.

    Kept local (no networkx) so ring tracking runs on the CI box that
    lacks the private backend — unlike upstream's spot tracking, which
    needs mlgidbase + networkx.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def track_rings(
    payload: TrackingPayload,
    ids: list,
    fitted_tables: dict,
    threshold: float,
    length: int,
) -> list:
    """Track ring peaks across frames by 1-D radial IoU.

    mlgidBASE's ``track_peaks`` cannot track rings at all: a ring's
    ``angle_width`` is infinite, so the (angle, radius) box it builds is
    infinitely wide, every IoU against it evaluates to NaN, and
    ``_track_peaks`` zeroes NaNs — each ring ends as an isolated
    single-member component the ``length`` cut discards. Rings have no
    azimuthal position (they span the whole arc), so the correct
    overlap measure is the 1-D IoU of their radius intervals
    ``[radius - radius_width/2, radius + radius_width/2]``.

    This mirrors upstream's spot algorithm on that 1-D measure:
    all-ring-against-all-ring radial IoU, thresholded into a graph,
    connected components, components with MORE than ``length`` members
    kept. Ring members are located in the payload via the reconstructed
    ``ids`` (their ``is_ring`` flag and radius come from
    ``fitted_tables``).

    Returns a list of components (each a sorted list of payload member
    indices) to append to ``payload.components`` — so tracked rings
    flow through the table, the views, and the only-tracked filter
    exactly like tracked spots. Ambiguous / unidentifiable ring members
    are skipped (best-effort, like ``member_ids``).
    """
    members = []  # (member_index, r_lo, r_hi)
    for i in range(payload.n_members):
        tagged = ids[int(i)]
        if tagged is None:
            continue
        frame, pid = int(tagged[0]), int(tagged[1])
        table = fitted_tables.get(frame)
        if table is None or len(table) == 0:
            continue
        hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == pid)
        if hits.size != 1:
            continue
        j = int(hits[0])
        if not bool(table.is_ring[j]):
            continue
        r = float(table.radius[j])
        rw = float(table.radius_width[j])
        members.append((int(i), r - rw / 2.0, r + rw / 2.0))
    if len(members) < 2:
        return []

    member_idx = np.array([m[0] for m in members])
    lo = np.array([m[1] for m in members])
    hi = np.array([m[2] for m in members])
    inter = np.clip(
        np.minimum(hi[:, None], hi[None, :])
        - np.maximum(lo[:, None], lo[None, :]),
        0.0, None,
    )
    len_i = hi - lo
    union = len_i[:, None] + len_i[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    pairs = np.argwhere(np.triu(iou >= float(threshold), k=1))
    out = []
    for comp in _connected_components(len(members), pairs):
        if len(comp) > int(length):
            out.append(sorted(int(member_idx[k]) for k in comp))
    return out


def member_ids(
    payload: TrackingPayload, fitted_tables: dict
) -> list:
    """``(frame, fitted_peak_id)`` per payload member.

    A payload built from upstream's table (``payload_from_tracked_table``)
    carries each member's id, so this is exact: a member whose frame has
    no fitted table, or whose id is not in it, yields ``None``.

    A hand-built payload without ``peak_id`` (tests) falls back to
    matching each member's ``(q_xy, q_z)`` — plus its amplitude as a
    duplicate-breaker — against the frame's fitted ``PeakTable``
    (``fitted_tables`` maps frame -> PeakTable or None): exact float
    equality first, ``np.isclose`` second, the amplitude only when it is
    finite. A member with no match or an ambiguous match yields
    ``None`` (the caller degrades to jump-to-frame-only).
    """
    out: list = []
    if payload.peak_id is not None:
        id_sets: dict = {}
        for i in range(payload.n_members):
            frame = int(payload.frame_num[i])
            pid = int(payload.peak_id[i])
            if frame not in id_sets:
                table = fitted_tables.get(frame)
                id_sets[frame] = (
                    set() if table is None or len(table) == 0
                    else {int(v) for v in table.ids}
                )
            out.append((frame, pid) if pid in id_sets[frame] else None)
        return out
    for i in range(payload.n_members):
        frame = int(payload.frame_num[i])
        table = fitted_tables.get(frame)
        if table is None or len(table) == 0:
            out.append(None)
            continue
        amp = payload.amplitude[i]
        use_amp = bool(np.isfinite(amp))
        exact = (
            (np.asarray(table.q_xy) == payload.q_xy[i])
            & (np.asarray(table.q_z) == payload.q_z[i])
        )
        if use_amp:
            exact &= np.asarray(table.amplitude) == amp
        hits = np.flatnonzero(exact)
        if hits.size != 1:
            close = (
                np.isclose(table.q_xy, payload.q_xy[i], rtol=1e-9, atol=1e-12)
                & np.isclose(table.q_z, payload.q_z[i], rtol=1e-9, atol=1e-12)
            )
            if use_amp:
                close &= np.isclose(
                    table.amplitude, amp, rtol=1e-9, atol=1e-12,
                )
            hits = np.flatnonzero(close)
        if hits.size == 1:
            out.append((frame, int(table.ids[hits[0]])))
        else:
            out.append(None)
    return out


def _matched_claim_index(matched_tables: dict) -> dict:
    """Per-frame claim index over the matched structures:
    ``{frame: {fitted_id: {cif, ...}}}`` — which CIFs claim which fitted
    peak on which frame. Shared by the track-level and member-level
    phase attributions below."""
    per_frame: dict = {}
    for frame, structures in matched_tables.items():
        if not structures:
            continue
        claim: dict = {}
        for s in structures:
            cif = str(s.cif)
            for pid in np.asarray(s.peaks.ids, dtype=int):
                claim.setdefault(int(pid), set()).add(cif)
        per_frame[int(frame)] = claim
    return per_frame


def member_matched_phases(
    payload: TrackingPayload, ids: list, matched_tables: dict
) -> dict:
    """Per-MEMBER matched phases: ``{global member index: [cif, ...]}``.

    The member-level companion of ``match_tracks_to_structures``: a
    member appears only when at least one matched structure claims its
    ``(frame, fitted_id)`` ON THAT FRAME, with the claiming CIFs sorted
    by name. Members never claimed are absent, so the map doubles as
    the per-frame matched/unmatched split of every track — a track
    fitted on frames x..y but matched only from frame z carries entries
    only for its z..y members. Pure; feeds the phase views' per-frame
    phase attribution.
    """
    per_frame = _matched_claim_index(matched_tables)
    out: dict = {}
    for comp in payload.components:
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            cifs = per_frame.get(int(tagged[0]), {}).get(int(tagged[1]))
            if cifs:
                out[int(i)] = sorted(cifs)
    return out


def match_tracks_to_structures(
    payload: TrackingPayload, ids: list, matched_tables: dict
) -> dict:
    """Map each track to the matched crystal phases (CIFs) it belongs to.

    For every track (component index ``k``), each member's reconstructed
    ``(frame, fitted_id)`` (from ``ids``) is checked against the frame's
    matched structures (``matched_tables`` maps frame ->
    ``list[MatchedStructure]``): a structure claims the member when its
    ``peaks.ids`` contain ``fitted_id``. The CIFs of all claiming
    structures are tallied across the track's members.

    Returns ``{k: [cif, ...]}`` with the CIFs sorted by descending
    member-match count then name (dominant phase first). Tracks with no
    matched member are omitted, so the map doubles as the set of
    phase-identified tracks. Pure; feeds the phase-views q-map colouring.
    """
    from collections import Counter

    per_frame = _matched_claim_index(matched_tables)
    out: dict = {}
    for k, comp in enumerate(payload.components):
        counts: Counter = Counter()
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            frame, pid = int(tagged[0]), int(tagged[1])
            for cif in per_frame.get(frame, {}).get(pid, ()):
                counts[cif] += 1
        if counts:
            # Dominant first: descending count, then name for stability.
            out[k] = [
                cif for cif, _n in sorted(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )
            ]
    return out


