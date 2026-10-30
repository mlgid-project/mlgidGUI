"""HDF5 editing primitives behind the Structure tab.

No Qt, no mlgidBASE — h5py and numpy only, so every operation here is
testable on its own and the module stays inside the backend-free
contract ``tests/test_hermetic_guard.py`` pins.

Three things in here are load-bearing and worth reading before touching
this file.

**The long-lived write handle.** ``EditHandle`` keeps one ``r+`` handle
open on the session's temp working copy for as long as the user is
editing. That is deliberate and it is safe only because of four HDF5
behaviours, each pinned by ``tests/test_h5_edit.py`` so an h5py or HDF5
upgrade that breaks one fails loudly:

1. opening ``r+`` while an ``r`` handle is open on the same file fails;
2. opening ``r`` while an ``r+`` handle is open succeeds, and the two
   share one underlying file object;
3. the ``r`` handle sees writes made through ``r+`` immediately,
   including groups created after it was opened;
4. a second ``r+`` open succeeds, and closing it leaves the first valid.

(2) and (3) are what let silx's read-only browser stay attached and
correct while the editor writes. (1) is why the handle may only be
acquired inside a detached window. (4) is why the peak-writing paths,
which open ``r+`` of their own, cannot be broken by an open editor
handle. Every commit calls ``flush()``, so ``File > Save``'s plain
``shutil.copy2`` of the temp copy is always consistent.

**Links are never followed by accident.** Anything that walks the file
(``walk_search``, ``node_nbytes``, ``validate``) inspects links with
``get(..., getlink=True)`` and descends only into hard links. A master
file whose 226 entries are external links to multi-GB scans can
therefore be searched and validated without opening a single one.
``node_info`` follows soft links (in-file, cheap) but reports an
external link rather than resolving it, unless asked with
``follow=True``.

**Protection is advisory.** ``protection_reason`` names the nodes the
viewer and the pipeline read. Nothing here refuses to act on them; the
GUI uses the reason string to warn before a destructive edit, which is
the behaviour the user asked for.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import h5py
import numpy as np

from mlgidgui.file_model import (
    ANALYSIS_REL,
    IMG_REL,
    QXY_REL,
    QZ_REL,
    is_entry_group_name,
)

import logging

logger = logging.getLogger(__name__)


class EditError(Exception):
    """An edit that cannot be performed, with a message fit for a dialog."""


#: Sentinel for "the attribute did not exist", so an undo can tell
#: "restore the old value" from "remove it again". ``None`` cannot serve:
#: an attribute may legitimately hold a null-ish value.
class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


MISSING = _Missing()

#: dtypes offered when creating a dataset. ``str`` maps to a variable
#: length UTF-8 string, which is what NeXus text fields use.
DTYPE_CHOICES: tuple[str, ...] = (
    "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool", "str",
)

#: Datasets at or below this many cells open directly in the editable
#: grid. Anything larger opens read-only with an explicit opt-in, so a
#: 6000 x 2000 detector image is never accidentally painted cell by
#: cell. 250k cells is a 500 x 500 array — comfortably interactive.
EDITABLE_CELL_LIMIT = 250_000

#: Above this many cells a dataset is not loaded into the grid at all.
#: The grid holds the whole array in memory and in a table model; 5M
#: cells is ~20 MB of float32 and already an unreasonable thing to edit
#: by hand. Bigger datasets are viewed in the Data tab, which streams.
VIEW_CELL_LIMIT = 5_000_000

#: Deletes whose payload is at most this large are snapshotted in memory
#: and can be undone. Above it the GUI warns that the delete is final.
#: 64 MB holds any metadata group and any realistic peak table, while
#: staying small enough that a long editing session cannot exhaust RAM.
UNDO_SNAPSHOT_LIMIT = 64 * 1024 * 1024

LinkKind = Literal["hard", "soft", "external"]
NodeKind = Literal["group", "dataset", "link", "missing"]


# -- paths -----------------------------------------------------------------


def normalize_path(path: str) -> str:
    """``a/b/`` -> ``/a/b``; the root in every spelling -> ``/``."""
    cleaned = "/" + str(path).strip().strip("/")
    return "/" if cleaned == "/" else cleaned


def split_path(path: str) -> tuple[str, str]:
    """``/a/b/c`` -> ``("/a/b", "c")``. The root raises."""
    full = normalize_path(path)
    if full == "/":
        raise EditError("The file root has no parent.")
    parent, _, name = full.rpartition("/")
    return (parent or "/"), name


def join_path(parent: str, name: str) -> str:
    """``("/a", "b")`` -> ``/a/b``; ``("/", "b")`` -> ``/b``."""
    base = normalize_path(parent)
    if not name:
        return base
    return f"{'' if base == '/' else base}/{name}"


def entry_of(path: str) -> str | None:
    """The ``entry_*`` group a path lives in, or None."""
    parts = normalize_path(path).strip("/").split("/")
    if parts and parts[0] and is_entry_group_name(parts[0]):
        return parts[0]
    return None


# -- protection ------------------------------------------------------------

#: Paths *relative to an entry group* that the viewer or the pipeline
#: reads, mapped to why they matter. Derived from ``file_model``'s
#: constants so there is one definition of the layout, not two.
_PROTECTED_RELATIVE: dict[str, str] = {
    "": "the entry the viewer and the pipeline work on",
    "data": "the group the viewer resolves an entry through",
    IMG_REL: "the detector image stack the viewer displays",
    QXY_REL: "the q_xy axis the viewer and every fit read",
    QZ_REL: "the q_z axis the viewer and every fit read",
    ANALYSIS_REL: "where every detected, fitted and matched peak is stored",
}


def protection_reason(path: str) -> str | None:
    """Why ``path`` matters to the rest of the app, or None if it is free.

    Advisory only. The GUI shows the reason in the details pane and
    repeats it in the confirm dialog before a destructive edit; nothing
    in this module consults it.
    """
    full = normalize_path(path)
    entry = entry_of(full)
    if entry is None:
        return None
    rel = full.strip("/")[len(entry):].strip("/")
    if rel in _PROTECTED_RELATIVE:
        return _PROTECTED_RELATIVE[rel]
    if rel == ANALYSIS_REL or rel.startswith(ANALYSIS_REL + "/"):
        return _PROTECTED_RELATIVE[ANALYSIS_REL]
    if rel.rsplit("/", 1)[-1] == "angle_of_incidence":
        return (
            "the incidence angle pygid indexes per frame; the pipeline "
            "fails on a wrong shape"
        )
    return None


def affects_viewer(path: str) -> bool:
    """Whether editing ``path`` changes what the image viewer displays.

    Narrower than ``protection_reason`` on purpose. The entry group
    itself is protected — deleting it takes the scan with it — but
    putting a ``title`` on it changes nothing the viewer draws, and
    reloading the entry for that would be a visible stutter for no
    reason. What the viewer actually reads is the image stack, the two q
    axes, the peak tables and the incidence angle.
    """
    reason = protection_reason(path)
    if reason is None:
        return False
    entry = entry_of(path)
    # The bare entry group: protected, but nothing the viewer re-reads.
    return normalize_path(path).strip("/") != entry


def attr_protection_reason(path: str, attr: str) -> str | None:
    """Why an attribute matters, or None.

    Only one attribute is load-bearing today: ``signal`` on an entry's
    ``data`` group is how ``file_model.list_entries`` decides an entry
    is displayable at all.
    """
    entry = entry_of(path)
    if entry is None:
        return None
    rel = normalize_path(path).strip("/")[len(entry):].strip("/")
    if rel == "data" and attr == "signal":
        return "how the viewer finds an entry's image stack"
    return None


# -- the handle ------------------------------------------------------------


class EditHandle:
    """One ``r+`` handle on a working copy, held across many edits.

    Acquire only from inside a detached window — see behaviour (1) in
    the module docstring. ``release`` is idempotent, so the GUI can hook
    it into the same teardown path every other handle uses.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None

    # -- lifecycle
    def acquire(self) -> h5py.File:
        """Open ``r+``. Returns the live handle; re-acquiring is a no-op."""
        if self._file is not None:
            return self._file
        try:
            self._file = h5py.File(self.path, "r+")
        except OSError as exc:
            raise EditError(
                f"Could not open {self.path.name} for editing: {exc}"
            ) from exc
        return self._file

    def release(self) -> None:
        """Close the handle if open. Safe to call any number of times."""
        f, self._file = self._file, None
        if f is None:
            return
        try:
            f.close()
        except Exception:
            # A handle whose file was already torn down under us (temp dir
            # removed on session close) raises here. Nothing to recover:
            # the point of the call is that we stop holding it.
            logger.debug("suppressed exception closing edit handle", exc_info=True)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    @property
    def is_open(self) -> bool:
        return self._file is not None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise EditError("No file is open for editing.")
        return self._file

    def relocate(self, path: Path | str) -> None:
        """Point at a new path, releasing any handle on the old one.

        Mirrors ``FrameSource.relocate``: Save As renames the temp file,
        and a handle opened against the old basename would be stale.
        """
        self.release()
        self.path = Path(path)


# -- reading ---------------------------------------------------------------


@dataclass(frozen=True)
class LinkInfo:
    kind: LinkKind
    target: str = ""
    filename: str = ""

    def describe(self) -> str:
        if self.kind == "external":
            return f"external link -> {self.filename}::{self.target}"
        if self.kind == "soft":
            return f"soft link -> {self.target}"
        return "hard link"


@dataclass(frozen=True)
class NodeInfo:
    path: str
    kind: NodeKind
    link: LinkInfo
    nx_class: str = ""
    dtype: str = ""
    shape: tuple[int, ...] = ()
    nbytes: int = 0
    chunks: tuple[int, ...] | None = None
    compression: str | None = None
    maxshape: tuple[int | None, ...] | None = None
    fields: tuple[str, ...] = ()
    n_children: int | None = None
    n_attrs: int = 0
    protection: str | None = None

    @property
    def is_compound(self) -> bool:
        return bool(self.fields)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def link_info(f: h5py.File, path: str) -> LinkInfo | None:
    """The link at ``path``, without resolving it. None if nothing is there."""
    full = normalize_path(path)
    if full == "/":
        return LinkInfo("hard")
    parent_path, name = split_path(full)
    try:
        parent = f[parent_path]
    except (KeyError, OSError):
        return None
    if not isinstance(parent, h5py.Group):
        return None
    # ``keys()`` lists links without following them, which is the whole
    # point here: a dangling soft or external link must be reported so
    # the user can retarget it, not hidden because its target is gone.
    # (h5py's ``in`` happens to agree — it tests link existence, not the
    # target's — but asking keys() says what we actually mean.)
    if name not in parent.keys():
        return None
    link = parent.get(name, getlink=True)
    if isinstance(link, h5py.ExternalLink):
        return LinkInfo("external", target=link.path, filename=link.filename)
    if isinstance(link, h5py.SoftLink):
        return LinkInfo("soft", target=link.path)
    return LinkInfo("hard")


def node_info(f: h5py.File, path: str, *, follow: bool = False) -> NodeInfo:
    """Describe the node at ``path``.

    External links are described as links and **not** opened unless
    ``follow`` is set — that one rule is what keeps the Structure tab
    usable on a 123 GB master whose entries are external links. Soft
    links are followed: they stay inside the file and cost nothing.
    """
    full = normalize_path(path)
    link = link_info(f, full)
    if link is None:
        return NodeInfo(full, "missing", LinkInfo("hard"))
    if link.kind == "external" and not follow:
        return NodeInfo(
            full, "link", link, protection=protection_reason(full)
        )
    try:
        obj = f[full]
    except (KeyError, OSError):
        # A broken soft or external link: the link exists, the target
        # does not. Report the link so the user can retarget it.
        return NodeInfo(full, "link", link, protection=protection_reason(full))

    protection = protection_reason(full)
    if isinstance(obj, h5py.Group):
        return NodeInfo(
            full, "group", link,
            nx_class=_as_text(obj.attrs.get("NX_class", "")),
            n_children=len(obj),
            n_attrs=len(obj.attrs),
            protection=protection,
        )
    return NodeInfo(
        full, "dataset", link,
        dtype=str(obj.dtype),
        shape=tuple(obj.shape or ()),
        nbytes=int(obj.nbytes),
        chunks=tuple(obj.chunks) if obj.chunks else None,
        compression=obj.compression,
        maxshape=tuple(obj.maxshape) if obj.maxshape else None,
        fields=tuple(obj.dtype.names or ()),
        n_attrs=len(obj.attrs),
        protection=protection,
    )


@dataclass(frozen=True)
class ChildInfo:
    """One row of a group's contents, classified without resolving it.

    ``kind`` is what to draw: a hard link is ``group`` or ``dataset``, a
    soft link that resolves in-file is reported as ``soft`` (with the
    resolved kind in ``target_kind``, which costs nothing - it never
    leaves the file), an external link is ``external`` and its target is
    NOT opened, and ``broken`` is a link whose target is gone.

    ``expandable`` is whether the row gets a chevron. An external link
    gets one even though nothing has been read: expanding it is the
    user's own decision to open that file, the same bargain the Link
    card's Follow button makes.
    """

    name: str
    kind: str
    expandable: bool = False
    detail: str = ""
    target_kind: str = ""


def list_children(f: h5py.File, path: str) -> list[ChildInfo]:
    """Every child of the group at ``path``, in file order, links intact.

    The listing rule the whole Structure tab rests on: ``keys()`` names
    the links and ``get(..., getlink=True)`` classifies each one, so a
    master whose 226 entries are external links to multi-GB scans lists
    as fast as any other file and opens none of them.

    Each child is classified inside its own guard. One unreadable row -
    a link into a file that has been moved, a corrupt entry - must cost
    that row and not the listing, because the tree it fills is the only
    way the user can reach the row that would fix it.

    Returns an empty list for anything that is not a group, which is
    what a caller expanding a dataset by mistake should see.
    """
    full = normalize_path(path)
    try:
        group = f[full]
    except (KeyError, OSError):
        return []
    if not isinstance(group, h5py.Group):
        return []

    children: list[ChildInfo] = []
    for name in group.keys():
        try:
            children.append(_classify_child(group, str(name)))
        except (KeyError, OSError, RuntimeError, ValueError):
            logger.debug("child %s of %s unreadable", name, full, exc_info=True)
            children.append(ChildInfo(str(name), "broken", detail="unreadable"))
    return children


def _classify_child(group: h5py.Group, name: str) -> ChildInfo:
    """Classify one child of an open group. Never opens an external file."""
    link = group.get(name, getlink=True)

    if isinstance(link, h5py.ExternalLink):
        return ChildInfo(
            name, "external", expandable=True,
            detail=f"external link -> {link.filename}::{link.path}",
        )

    if isinstance(link, h5py.SoftLink):
        # A soft link stays inside this file, so resolving it is a
        # pointer chase, not an open. Worth doing: the tree can then
        # show whether it lands on a group or a dataset.
        try:
            target = group[name]
        except (KeyError, OSError):
            return ChildInfo(
                name, "broken", detail=f"soft link -> {link.path} (missing)")
        kind = "group" if isinstance(target, h5py.Group) else "dataset"
        return ChildInfo(
            name, "soft", expandable=kind == "group",
            detail=f"soft link -> {link.path}", target_kind=kind,
        )

    obj = group[name]
    if isinstance(obj, h5py.Group):
        nx_class = _as_text(obj.attrs.get("NX_class", ""))
        n = len(obj)
        return ChildInfo(
            name, "group", expandable=n > 0,
            detail=nx_class or ("group" if n else "empty group"),
        )
    shape = " x ".join(str(d) for d in obj.shape) if obj.shape else "scalar"
    return ChildInfo(name, "dataset", detail=f"{obj.dtype} - {shape}")


def _as_text(value: Any) -> str:
    """Decode an HDF5 string-ish value to ``str`` without guessing."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO" and value.size == 1:
        return _as_text(value.reshape(-1)[0])
    return "" if value is None else str(value)


def read_attrs(f: h5py.File, path: str) -> dict[str, Any]:
    """Every attribute of ``path``, in file order. Links are not followed."""
    full = normalize_path(path)
    link = link_info(f, full)
    if link is not None and link.kind == "external":
        return {}
    try:
        return dict(f[full].attrs)
    except (KeyError, OSError) as exc:
        raise EditError(f"{full} cannot be read: {exc}") from exc


def value_preview(f: h5py.File, path: str, *, max_items: int = 12) -> str:
    """A short rendering of what a dataset holds.

    Reads at most ``max_items`` elements, never the whole array: the
    slice walks index 0 along every axis but the last, so previewing a
    6000 x 2000 x 2000 stack costs one row, not the file. Groups and
    links return an empty string — there is nothing to preview.
    """
    full = normalize_path(path)
    link = link_info(f, full)
    if link is not None and link.kind == "external":
        return ""
    try:
        obj = f[full]
    except (KeyError, OSError):
        return ""
    if not isinstance(obj, h5py.Dataset):
        return ""
    try:
        if obj.ndim == 0:
            return format_value(obj[()])
        index = tuple([0] * (obj.ndim - 1) + [slice(0, max_items)])
        chunk = np.asarray(obj[index])
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("value preview failed for %s", full, exc_info=True)
        return f"(unreadable: {exc})"
    total = int(np.prod(obj.shape))
    if obj.dtype.names:
        rows = [
            ", ".join(f"{name}={format_value(row[name], limit=24)}"
                      for name in obj.dtype.names)
            for row in chunk[:2]
        ]
        text = " | ".join(rows)
    else:
        text = ", ".join(format_value(v, limit=24) for v in chunk.reshape(-1))
    if total > chunk.size:
        text += f", … ({total} values)"
    return text


def format_value(value: Any, *, limit: int = 120) -> str:
    """A one-line rendering of an attribute or scalar, elided at ``limit``."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return format_value(value.reshape(-1)[0], limit=limit)
        text = np.array2string(value, threshold=16, separator=", ")
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -- attributes ------------------------------------------------------------


def set_attr(f: h5py.File, path: str, name: str, value: Any) -> Any:
    """Write one attribute. Returns the previous value, or ``MISSING``."""
    if not name:
        raise EditError("An attribute needs a name.")
    obj = _require(f, path)
    old = obj.attrs[name] if name in obj.attrs else MISSING
    try:
        obj.attrs[name] = value
    except (TypeError, ValueError, OSError) as exc:
        raise EditError(f"Cannot store {name!r}: {exc}") from exc
    return old


def delete_attr(f: h5py.File, path: str, name: str) -> Any:
    """Remove one attribute. Returns its value so the delete can be undone."""
    obj = _require(f, path)
    if name not in obj.attrs:
        raise EditError(f"{path} has no attribute {name!r}.")
    old = obj.attrs[name]
    del obj.attrs[name]
    return old


def rename_attr(f: h5py.File, path: str, old_name: str, new_name: str) -> None:
    """Rename in place, keeping the value and refusing to clobber."""
    if old_name == new_name:
        return
    obj = _require(f, path)
    if old_name not in obj.attrs:
        raise EditError(f"{path} has no attribute {old_name!r}.")
    if new_name in obj.attrs:
        raise EditError(f"{path} already has an attribute {new_name!r}.")
    obj.attrs[new_name] = obj.attrs[old_name]
    del obj.attrs[old_name]


def attr_type_label(value: Any) -> str:
    """How an attribute's type reads in the UI: ``str``, ``float64``, …"""
    if value is MISSING:
        return "str"
    if isinstance(value, (bytes, str)):
        return "str"
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SOU":
            return "str" if value.size == 1 else f"str[{value.size}]"
        return (str(value.dtype) if value.size == 1
                else f"{value.dtype}[{value.size}]")
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    return str(np.asarray(value).dtype)


def is_inline_editable(value: Any) -> bool:
    """Whether an attribute can be edited as one line of text.

    Scalars, strings and flat lists can. A 2-D attribute cannot be typed
    into a single field without inventing a syntax, so the UI shows it
    read-only rather than pretending.
    """
    if isinstance(value, np.ndarray):
        return value.ndim <= 1
    return True


def parse_attr_value(text: str, template: Any) -> Any:
    """Parse ``text`` into the shape and dtype of an existing attribute.

    An attribute keeps its type when edited: typing ``2.5`` into a
    float64 attribute stores a float64, not a string. ``template`` is the
    current value, or ``MISSING`` for one being created (which becomes
    text). A flat array attribute is written comma-separated and may
    change length — HDF5 attributes are rewritten wholesale anyway.
    """
    if template is MISSING or template is None:
        return text
    if isinstance(template, (bytes, str)):
        return text
    if isinstance(template, np.ndarray) and template.dtype.kind in "SOU":
        parts = [p.strip() for p in text.split(",")]
        return parts[0] if template.size == 1 and len(parts) == 1 else parts
    if isinstance(template, np.ndarray) and template.size != 1:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise EditError("Expected one or more comma-separated values.")
        return np.array(
            [parse_scalar(p, str(template.dtype)) for p in parts],
            dtype=template.dtype,
        )
    dtype = str(np.asarray(template).dtype)
    if np.asarray(template).dtype.kind == "b":
        dtype = "bool"
    return parse_scalar(text, dtype)


def parse_scalar(text: str, dtype: str) -> Any:
    """Turn user-typed text into a value of ``dtype``.

    Raises ``EditError`` with a message naming what was expected, which
    is what the details pane shows next to the field.
    """
    text = text.strip()
    if dtype in ("str", "string") or dtype.startswith(("|S", "|O", "<U", "object")):
        return text
    if dtype == "bool":
        low = text.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise EditError("Expected true or false.")
    try:
        return np.dtype(dtype).type(text)
    except (TypeError, ValueError) as exc:
        raise EditError(f"Expected a value of type {dtype}: {exc}") from exc


# -- structure -------------------------------------------------------------


def _require(f: h5py.File, path: str) -> Any:
    full = normalize_path(path)
    try:
        return f[full]
    except (KeyError, OSError) as exc:
        raise EditError(f"{full} does not exist in this file.") from exc


def _require_group(f: h5py.File, path: str) -> h5py.Group:
    obj = _require(f, path)
    if not isinstance(obj, h5py.Group):
        raise EditError(f"{normalize_path(path)} is a dataset, not a group.")
    return obj


def _check_free(parent: h5py.Group, name: str) -> None:
    if not name or "/" in name:
        raise EditError("A name cannot be empty or contain '/'.")
    if name in parent:
        raise EditError(f"{parent.name}/{name} already exists.")


def create_group(
    f: h5py.File,
    parent: str,
    name: str,
    *,
    nx_class: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> str:
    """Create one group. Returns its path.

    ``track_order=True`` matches how pygid writes its files, so a group
    added here keeps insertion order like every group around it.
    """
    group = _require_group(f, parent)
    _check_free(group, name)
    child = group.create_group(name, track_order=True)
    if nx_class:
        child.attrs["NX_class"] = nx_class
    for key, value in (attrs or {}).items():
        child.attrs[key] = value
    return child.name


def create_dataset(
    f: h5py.File,
    parent: str,
    name: str,
    *,
    dtype: str = "float32",
    shape: tuple[int, ...] | None = None,
    data: Any = None,
    fill: Any = None,
    attrs: dict[str, Any] | None = None,
    resizable: bool = False,
) -> str:
    """Create one dataset. Returns its path.

    With ``data`` the shape and dtype come from it. Otherwise ``shape``
    (``()`` for a scalar) and ``dtype`` are used, with every cell set to
    ``fill``. ``dtype="str"`` creates a variable-length UTF-8 string,
    which is what NeXus text fields are.

    ``resizable`` leaves the first axis unbounded, so rows can be
    appended later without rewriting the dataset — the same shape
    pygid's own append path relies on. Ignored for a scalar, which has
    no axis to grow.
    """
    group = _require_group(f, parent)
    _check_free(group, name)
    kwargs: dict[str, Any] = {"track_order": True}
    if resizable and shape:
        kwargs["maxshape"] = (None,) + tuple(shape[1:])
    elif resizable and data is not None and np.ndim(data) > 0:
        kwargs["maxshape"] = (None,) + tuple(np.shape(data)[1:])
    try:
        if data is not None:
            if dtype == "str":
                kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
            child = group.create_dataset(name, data=data, **kwargs)
        elif dtype == "str":
            child = group.create_dataset(
                name,
                shape=shape if shape is not None else (),
                dtype=h5py.string_dtype(encoding="utf-8"),
                **kwargs,
            )
        else:
            if fill is not None:
                kwargs["fillvalue"] = fill
            child = group.create_dataset(
                name,
                shape=shape if shape is not None else (),
                dtype=np.dtype(dtype),
                **kwargs,
            )
    except (TypeError, ValueError, OSError) as exc:
        raise EditError(f"Cannot create {name!r}: {exc}") from exc
    for key, value in (attrs or {}).items():
        child.attrs[key] = value
    return child.name


def delete_node(f: h5py.File, path: str) -> None:
    """Unlink ``path``.

    HDF5 does not shrink the file: the blocks become reusable *within*
    this file and its size on disk is unchanged. The working copy
    therefore only grows during a session, which is accepted -- it is
    temporary, and repacking it after every delete would rewrite the
    whole file each time.

    The saved file is a different matter. An earlier version of this
    note claimed the save "is written from it in full" and left it
    there, but ``NexusSession.save`` was a byte copy, so the holes were
    copied faithfully into the user's file and deleting never freed
    anything. ``h5_repack.copy_or_repack`` now handles that at save
    time.
    """
    full = normalize_path(path)
    if full == "/":
        raise EditError("The file root cannot be deleted.")
    parent_path, name = split_path(full)
    parent = _require_group(f, parent_path)
    if name not in parent:
        raise EditError(f"{full} does not exist in this file.")
    del parent[name]


def rename_node(f: h5py.File, path: str, new_name: str) -> str:
    """Rename in place, keeping the node's position. Returns the new path."""
    full = normalize_path(path)
    parent_path, old_name = split_path(full)
    if old_name == new_name:
        return full
    parent = _require_group(f, parent_path)
    _check_free(parent, new_name)
    parent.move(old_name, new_name)
    return join_path(parent_path, new_name)


def move_node(
    f: h5py.File, path: str, new_parent: str, new_name: str | None = None
) -> str:
    """Move a node under a different parent. Returns the new path."""
    full = normalize_path(path)
    dest_parent = normalize_path(new_parent)
    name = new_name or split_path(full)[1]
    if dest_parent == full or dest_parent.startswith(full.rstrip("/") + "/"):
        raise EditError("A group cannot be moved inside itself.")
    parent = _require_group(f, dest_parent)
    _check_free(parent, name)
    _require(f, full)
    f.move(full, join_path(dest_parent, name))
    return join_path(dest_parent, name)


# -- links -----------------------------------------------------------------


def create_link(
    f: h5py.File,
    parent: str,
    name: str,
    kind: LinkKind,
    target: str,
    *,
    filename: str | None = None,
) -> str:
    """Create a hard, soft or external link. Returns its path.

    Nothing is resolved: a soft or external link may point at something
    that does not exist yet, which is legal HDF5 and occasionally what
    the user means.
    """
    group = _require_group(f, parent)
    _check_free(group, name)
    if kind == "external":
        if not filename:
            raise EditError("An external link needs a file to point at.")
        group[name] = h5py.ExternalLink(filename, target)
    elif kind == "soft":
        group[name] = h5py.SoftLink(normalize_path(target))
    elif kind == "hard":
        group[name] = _require(f, target)
    else:  # pragma: no cover - guarded by the caller's combo
        raise EditError(f"Unknown link kind {kind!r}.")
    return join_path(group.name, name)


def retarget_link(
    f: h5py.File,
    path: str,
    kind: LinkKind,
    target: str,
    *,
    filename: str | None = None,
) -> str:
    """Point an existing link somewhere else, keeping its name."""
    parent_path, name = split_path(path)
    existing = link_info(f, path)
    if existing is None:
        raise EditError(f"{normalize_path(path)} does not exist in this file.")
    if existing.kind == "hard":
        raise EditError(
            "A hard link is the object itself, not a pointer — it cannot "
            "be retargeted. Delete it, or create a new link beside it."
        )
    del _require_group(f, parent_path)[name]
    return create_link(f, parent_path, name, kind, target, filename=filename)


# -- copying ---------------------------------------------------------------


def copy_node(
    dst: h5py.File,
    dst_parent: str,
    name: str,
    src: h5py.File,
    src_path: str,
    *,
    expand_external: bool = False,
) -> str:
    """Copy a group or dataset, within a file or between two open files.

    Attributes come along; a group is copied whole. External links
    *inside* the copied subtree stay links unless ``expand_external`` is
    set, so copying an entry out of a master does not silently drag a
    multi-GB scan along with it.

    When the source path **is itself** a soft or external link, the link
    is recreated rather than followed. h5py's own ``copy`` dereferences
    in that case — it would turn a paste of a 4 GB external link into a
    4 GB copy, which is never what the user meant by copying a link. A
    relative external filename is resolved against the source file's
    directory first, so the pasted link still points at the same file
    when the destination lives somewhere else.
    """
    parent = _require_group(dst, dst_parent)
    _check_free(parent, name)
    source = normalize_path(src_path)
    link = link_info(src, source)
    if link is not None and link.kind != "hard" and not expand_external:
        if link.kind == "soft":
            return create_link(dst, dst_parent, name, "soft", link.target)
        filename = link.filename
        if not Path(filename).is_absolute():
            filename = str((Path(src.filename).parent / filename).resolve())
        return create_link(
            dst, dst_parent, name, "external", link.target, filename=filename
        )
    try:
        src.copy(
            source, parent, name=name,
            expand_soft=False, expand_external=expand_external, expand_refs=False,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise EditError(f"Cannot copy {source}: {exc}") from exc
    return join_path(parent.name, name)


def copy_attrs(
    dst: h5py.File, dst_path: str, src: h5py.File, src_path: str,
    *, names: list[str] | None = None, overwrite: bool = False,
) -> list[str]:
    """Copy attributes between two nodes. Returns the names written."""
    target = _require(dst, dst_path)
    source = _require(src, src_path)
    written: list[str] = []
    for key in (names if names is not None else list(source.attrs)):
        if key in target.attrs and not overwrite:
            continue
        target.attrs[key] = source.attrs[key]
        written.append(key)
    return written


# -- values ----------------------------------------------------------------


def read_block(f: h5py.File, path: str, index: Any = ...) -> np.ndarray:
    """Read a slice of a dataset as an array."""
    obj = _require(f, path)
    if not isinstance(obj, h5py.Dataset):
        raise EditError(f"{normalize_path(path)} is a group, not a dataset.")
    return np.asarray(obj[index])


def write_cells(
    f: h5py.File, path: str, updates: dict[Any, Any]
) -> dict[Any, Any]:
    """Write individual cells. Returns the previous values, for undo.

    ``updates`` maps an index — anything h5py accepts, including
    ``(row, "field")`` for a compound dataset — to its new value.
    """
    obj = _require(f, path)
    if not isinstance(obj, h5py.Dataset):
        raise EditError(f"{normalize_path(path)} is a group, not a dataset.")
    previous: dict[Any, Any] = {}
    for index, value in updates.items():
        if isinstance(index, tuple) and index and isinstance(index[-1], str):
            row, field_name = index[:-1], index[-1]
            row_index = row[0] if len(row) == 1 else row
            record = obj[row_index]
            previous[index] = record[field_name]
            record[field_name] = value
            obj[row_index] = record
        else:
            previous[index] = obj[index]
            obj[index] = value
    return previous


def rewrite_dataset(f: h5py.File, path: str, array: np.ndarray) -> None:
    """Replace a dataset's contents, keeping its name and attributes.

    Resizes in place when the dataset is resizable and only its length
    changed; otherwise deletes and recreates, which is what
    ``file_model._add_peak_row`` has always done for the peak tables.
    """
    full = normalize_path(path)
    obj = _require(f, full)
    if not isinstance(obj, h5py.Dataset):
        raise EditError(f"{full} is a group, not a dataset.")
    if (
        obj.maxshape
        and obj.maxshape[0] is None
        and array.shape[1:] == obj.shape[1:]
        and array.dtype == obj.dtype
    ):
        obj.resize((array.shape[0],) + obj.shape[1:])
        obj[...] = array
        return
    attrs = dict(obj.attrs)
    parent_path, name = split_path(full)
    parent = _require_group(f, parent_path)
    del parent[name]
    new = parent.create_dataset(name, data=array, track_order=True)
    for key, value in attrs.items():
        new.attrs[key] = value


def insert_rows(
    f: h5py.File, path: str, at: int, count: int = 1, template: Any = None
) -> None:
    """Insert ``count`` rows into a 1-D or compound dataset at ``at``.

    New rows are copies of ``template``, of the row at ``at - 1``, or
    zeros — in that order of preference, so duplicating a row is the
    same operation as inserting one.
    """
    array = read_block(f, path)
    if array.ndim == 0:
        raise EditError("A scalar dataset has no rows.")
    at = max(0, min(int(at), array.shape[0]))
    if template is not None:
        filler = np.array([template] * count, dtype=array.dtype)
    elif at > 0:
        filler = np.repeat(array[at - 1: at], count, axis=0)
    else:
        filler = np.zeros((count,) + array.shape[1:], dtype=array.dtype)
    rewrite_dataset(f, path, np.concatenate([array[:at], filler, array[at:]]))


def delete_rows(f: h5py.File, path: str, rows: list[int]) -> np.ndarray:
    """Remove rows. Returns them, so the delete can be undone."""
    array = read_block(f, path)
    if array.ndim == 0:
        raise EditError("A scalar dataset has no rows.")
    keep = np.ones(array.shape[0], dtype=bool)
    wanted = [r for r in rows if 0 <= r < array.shape[0]]
    if not wanted:
        raise EditError("No such rows in this dataset.")
    keep[wanted] = False
    removed = array[~keep].copy()
    rewrite_dataset(f, path, array[keep])
    return removed


# -- snapshots -------------------------------------------------------------


@dataclass
class Snapshot:
    """An in-memory HDF5 image of one node, held so a delete can be undone."""

    name: str
    image: bytes
    nbytes: int

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.image)


def node_nbytes(f: h5py.File, path: str, cap: int | None = None) -> int:
    """Payload size of a node, summed over hard-linked children only.

    Stops early once ``cap`` is exceeded and returns a value above it,
    so asking "is this too big to snapshot?" never walks a 226-entry
    master. Soft and external links contribute nothing: copying the
    subtree keeps them as links.
    """
    total = 0
    for _, obj in _walk_hard(f, path):
        if isinstance(obj, h5py.Dataset):
            total += int(obj.nbytes)
            if cap is not None and total > cap:
                return total
    return total


def snapshot_node(f: h5py.File, path: str) -> Snapshot:
    """Copy a node into an in-memory HDF5 image.

    Bytes rather than a live handle: the snapshot outlives the undo
    entry that holds it, and a memory file left open would be one more
    handle to release on teardown.
    """
    full = normalize_path(path)
    _, name = split_path(full)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as mem:
        try:
            f.copy(full, mem, name=name, expand_soft=False,
                   expand_external=False, expand_refs=False)
        except (KeyError, ValueError, OSError) as exc:
            raise EditError(f"Cannot snapshot {full}: {exc}") from exc
    image = buffer.getvalue()
    return Snapshot(name=name, image=image, nbytes=len(image))


def restore_snapshot(f: h5py.File, parent: str, snap: Snapshot) -> str:
    """Put a snapshotted node back under ``parent``. Returns its path."""
    with h5py.File(io.BytesIO(snap.image), "r") as mem:
        return copy_node(f, parent, snap.name, mem, "/" + snap.name)


# -- walking, searching, validating ---------------------------------------


def _walk_hard(f: h5py.File, path: str = "/") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, object)`` for a node and its hard-linked children.

    Soft and external links are yielded as their *link*, never resolved,
    so a walk of a master file never opens a linked scan.
    """
    full = normalize_path(path)
    if full != "/":
        parent_path, name = split_path(full)
        parent = f.get(parent_path)
        if isinstance(parent, h5py.Group) and name in parent.keys():
            raw = parent.get(name, getlink=True)
            if isinstance(raw, (h5py.SoftLink, h5py.ExternalLink)):
                yield full, raw
                return
    try:
        obj = f[full]
    except (KeyError, OSError):
        return
    yield full, obj
    if not isinstance(obj, h5py.Group):
        return
    for name in obj.keys():
        child = join_path(full, name)
        sub = obj.get(name, getlink=True)
        if isinstance(sub, (h5py.SoftLink, h5py.ExternalLink)):
            yield child, sub
            continue
        yield from _walk_hard(f, child)


@dataclass(frozen=True)
class SearchHit:
    path: str
    #: ``name`` the node's own name, ``data`` a small dataset's value,
    #: ``attribute`` an attribute's name, ``value`` an attribute's value.
    where: Literal["name", "data", "attribute", "value"]
    detail: str


#: How many elements a dataset may hold before its value is skipped by
#: a search. NeXus metadata is small strings and scalars — a sample
#: name, a chemical formula, a start time — and those are exactly what
#: someone searching a file is looking for. A detector stack is not, and
#: reading one per node would turn a search into a file read. ``size``
#: comes from the dataspace, so the big ones are skipped without ever
#: being touched.
VALUE_SCAN_MAX = 16


def parse_search(query: str) -> tuple[tuple[str, ...], str]:
    """``"sample/material/Si"`` -> ``(("sample", "material"), "Si")``.

    Everything before the last ``/`` is where to look; the last part is
    what to look for. A query with no ``/`` is the plain form and comes
    back with an empty lead.
    """
    parts = [part.strip() for part in query.split("/")]
    parts = [part for part in parts if part]
    if not parts:
        return (), ""
    return tuple(parts[:-1]), parts[-1]


def _lead_matches(components: list[str], lead: tuple[str, ...]) -> bool:
    """Whether ``lead`` appears along ``components``, in order.

    A subsequence, not a prefix and not adjacency: ``sample/material``
    finds ``/entry_0000/instrument/sample/x/material`` because what the
    user knows is the shape of the file, not every group between.
    Both sides arrive already case-folded.
    """
    remaining = iter(components)
    for want in lead:
        for component in remaining:
            if want in component:
                break
        else:
            return False
    return True


def _dataset_text(obj: Any) -> str:
    """A small dataset's value as one line, or ``""``.

    Bounded by ``VALUE_SCAN_MAX`` and defended: a search must not raise
    on the one dataset in a file with an exotic dtype.
    """
    if not isinstance(obj, h5py.Dataset):
        return ""
    try:
        if obj.size > VALUE_SCAN_MAX:
            return ""
        return format_value(obj[()])
    except Exception:
        logger.debug("reading a dataset value for search failed", exc_info=True)
        return ""


def walk_search(
    f: h5py.File,
    query: str,
    *,
    root: str = "/",
    max_nodes: int = 20_000,
    case_sensitive: bool = False,
    include_attrs: bool = True,
) -> tuple[list[SearchHit], bool]:
    """Find nodes matching ``query`` by name, attribute or small value.

    ``query`` may carry slashes, and then it is *also* read as a place
    followed by a term: ``material/Si`` is "something matching Si,
    somewhere under a group matching material". That is what makes a
    common word usable — plain ``Si`` matches every ``signal`` in the
    file too, and no amount of reading the list fixes that.

    Also, not instead. A slash is a legitimate thing to search *for*:
    ``1/Angstrom`` is a unit, ``m/s`` is a unit, and reading those as a
    path would quietly stop finding them. So both readings run over one
    walk and a node matches on either. They do not compete for hits in
    practice: nothing in an HDF5 file is literally named
    ``material/Si``, and nothing lives under a group called ``1``.

    ``case_sensitive`` is off by default because someone searching a
    file they did not write does not know how its author capitalised
    things. It is offered because sometimes they do, and in a file about
    silicon ``Si`` is not ``si``.

    Returns ``(hits, truncated)``. ``truncated`` says the walk hit
    ``max_nodes`` and stopped, so the UI can say so rather than implying
    the result is complete. Never follows a soft or external link.
    """
    fold = (lambda text: text) if case_sensitive else (lambda text: text.lower())
    whole = fold(query.strip())
    if not whole:
        return [], False
    lead, term = parse_search(query)
    lead = tuple(fold(part) for part in lead)
    needle = fold(term)

    def matches(text: str, scoped: bool) -> bool:
        """The two readings, in one test. ``scoped`` says the node's
        path satisfied the lead, so the term alone is enough for it."""
        hay = fold(text)
        return whole in hay or (scoped and needle in hay)

    hits: list[SearchHit] = []
    seen = 0
    for path, obj in _walk_hard(f, root):
        seen += 1
        if seen > max_nodes:
            return hits, True
        scoped = bool(lead) and _lead_matches(
            [fold(part) for part in path.split("/") if part], lead
        )
        name = path.rsplit("/", 1)[-1]
        if matches(name, scoped):
            hits.append(SearchHit(path, "name", name))
        if isinstance(obj, (h5py.SoftLink, h5py.ExternalLink)):
            continue
        text = _dataset_text(obj)
        if text and matches(text, scoped):
            hits.append(SearchHit(path, "data", text))
        if not include_attrs:
            continue
        for key, value in obj.attrs.items():
            text = format_value(value)
            if matches(key, scoped):
                hits.append(SearchHit(path, "attribute", f"{key} = {text}"))
            elif matches(text, scoped):
                hits.append(SearchHit(path, "value", f"{key} = {text}"))
    return hits, False


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning", "info"]
    path: str
    message: str
    fix: str = ""


def validate(f: h5py.File) -> list[Issue]:
    """Report what is wrong or risky about the file's layout.

    Three checks, each one a failure mode this GUI has actually hit:

    * a top-level group with no ``data`` child, or a ``signal`` naming a
      dataset that is not there — the two lookups pygid's
      ``get_entry_type`` performs unconditionally, and the reason
      ``file_model.list_pygid_incompatible_top_level`` exists;
    * an ``NXdata`` group with no ``signal`` attribute, which the viewer
      cannot resolve an entry through;
    * a group with no ``NX_class``, reported as information only —
      common, legal enough, and worth knowing about.

    Unlike ``list_pygid_incompatible_top_level`` this reads through the
    already-open handle and **skips entries that are external links**
    rather than resolving them. On a master of 226 linked scans the
    published helper would open all 226; this returns in milliseconds
    and says which entries it did not check.
    """
    issues: list[Issue] = []
    for name in f.keys():
        path = "/" + name
        link = link_info(f, path)
        if link is not None and link.kind == "external":
            issues.append(Issue(
                "info", path,
                "external link — not checked, following it would open "
                f"{link.filename}",
            ))
            continue
        obj = f.get(name)
        if not isinstance(obj, h5py.Group):
            issues.append(Issue(
                "error", path,
                "a top-level dataset makes pygid's reader fail on this file",
                fix="Move it inside a group.",
            ))
            continue
        data = obj.get("data")
        if not isinstance(data, h5py.Group):
            issues.append(Issue(
                "error", path,
                "no 'data' group — pygid's reader indexes it unconditionally "
                "and the whole file fails to open",
                fix="Add a 'data' group, or move this group inside an entry.",
            ))
            continue
        signal = data.attrs.get("signal")
        signal = _as_text(signal) if signal is not None else ""
        if not signal:
            issues.append(Issue(
                "warning", path + "/data",
                "no 'signal' attribute — the viewer cannot tell which "
                "dataset holds the image stack",
                fix="Set 'signal' to the name of the image dataset.",
            ))
        elif signal not in data:
            issues.append(Issue(
                "error", path + "/data",
                f"'signal' names {signal!r}, which is not in this group",
                fix=f"Set 'signal' to one of: {', '.join(list(data)) or 'nothing yet'}.",
            ))
        if "NX_class" not in obj.attrs:
            issues.append(Issue(
                "info", path, "no NX_class attribute",
                fix="Set NX_class to NXentry.",
            ))
    return issues
