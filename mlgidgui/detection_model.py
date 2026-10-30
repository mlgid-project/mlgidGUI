"""Detection-model cache pre-flight for the GUI.

mlgidDETECT ships no ``.onnx`` weights: the first detection run
downloads them into a per-user cache (``appdirs.user_data_dir(
'mlgiddetect')``) via ``mlgiddetect.utils.path_utils``. Two properties
of that downloader make it unsafe to drive from a GUI process, and this
module exists to work around both without touching the backend.

1. ``path_utils.download`` renders a progress bar with
   ``sys.stdout.write``. A GUI launched through the ``gui-scripts``
   entry point runs under ``pythonw.exe`` on Windows, where
   ``sys.stdout`` is ``None`` — so the very first progress tick raises
   ``AttributeError: 'NoneType' object has no attribute 'write'``. The
   downloader swallows it in a bare ``except``, returns ``None``, and
   ``Inference.__init__`` then calls ``sys.exit()``. mlgidbase's
   ``load_inference`` catches even that (bare ``except:`` sees
   ``SystemExit``) and re-raises the generic "Detection failed.
   Couldn't load the model." The user never learns a download was
   attempted. ``safe_console`` closes this hole.

2. ``urllib.request.urlretrieve`` writes straight to the *final* path,
   so any interruption leaves a truncated — often zero-byte — file
   behind. ``path_utils.check_filepath`` only tests that the path
   exists and ends in ``.onnx``, so that corpse is accepted forever
   after: no re-download is attempted and onnxruntime fails with an
   opaque ``[ONNXRuntimeError] ... Load model ... failed``. The install
   is then permanently broken even once (1) is fixed.
   ``ensure_detection_model`` downloads through a unique temp file and
   only ``os.replace``s it into place after the byte count matches
   ``Content-Length``, and it deletes previously poisoned cache entries
   so existing broken installs self-heal.

Everything here degrades to a no-op when mlgiddetect is absent (the
headless CI suite does not ship the backend), and never raises for a
merely *unverifiable* cache — an offline user with a good model file
must keep working.

No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

import contextlib
import logging
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Any genuine mlgidDETECT model is ~94 MB (frcnn) to ~1.1 GB (dino), so
# anything below this floor is a failed download rather than a real
# file. Deliberately generous: the point is to catch 0-byte and
# obviously truncated corpses, not to guess exact sizes.
_MIN_SANE_BYTES = 1 << 20  # 1 MiB

# Partial downloads carry this suffix. mlgidDETECT's own
# ``check_filepath`` requires an ``.onnx`` extension, so a leftover
# ``*.onnx.part`` is invisible to it and can never be mistaken for a
# model — it only wastes disk.
_PART_SUFFIX = ".onnx.part"

# A ``.part`` file younger than this may belong to a download running
# right now in another mlgidGUI instance, so leave it alone; older ones
# are abandoned and get cleaned up.
_STALE_PART_SECONDS = 6 * 60 * 60

_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB
_CONNECT_TIMEOUT = 60

# ``MODEL_TYPE`` -> the ``MODEL_URLS`` key (and therefore the cached
# ``<key>.onnx`` filename) that mlgidDETECT resolves it to. Mirrors
# ``path_utils.get_model_path``; kept as data so a mismatch degrades to
# "skip the pre-flight" rather than downloading the wrong file.
_MODEL_TYPE_TO_KEY = {
    "dino": "dino",
    "faster_rcnn": "frcnn",
}

# Paths whose size we already reconciled against the server in this
# process. The check costs one HTTP request, so do it once per session
# per model rather than on every detection run.
_size_verified: set[str] = set()


def _stream_is_usable(stream: Any) -> bool:
    """True when ``stream`` can actually absorb a ``.write`` call.

    ``pythonw`` hands out ``None``; a detached/closed stream raises on
    write. Both must be treated as unusable.
    """
    if stream is None:
        return False
    write = getattr(stream, "write", None)
    if not callable(write):
        return False
    return not getattr(stream, "closed", False)


@contextlib.contextmanager
def safe_console():
    """Guarantee ``sys.stdout`` / ``sys.stderr`` accept writes.

    Only substitutes streams that are *already* unusable, so a GUI
    started from a terminal keeps its real console and nothing is
    swallowed. The substitute is ``os.devnull``: the only backend code
    that writes to these streams directly is mlgidDETECT's download
    progress bar, and this module logs its own progress through the
    ``logging`` machinery the GUI's log panel is attached to.

    ``print()`` is safe either way (CPython no-ops when ``sys.stdout``
    is ``None``); it is bare ``sys.stdout.write`` that needs this.
    """
    devnull = None
    saved: dict[str, Any] = {}
    try:
        for name in ("stdout", "stderr"):
            if _stream_is_usable(getattr(sys, name, None)):
                continue
            if devnull is None:
                devnull = open(os.devnull, "w")
            saved[name] = getattr(sys, name, None)
            setattr(sys, name, devnull)
        yield
    finally:
        for name, stream in saved.items():
            setattr(sys, name, stream)
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                logger.debug("suppressed exception closing devnull", exc_info=True)


def _path_utils():
    """mlgidDETECT's path helpers, or ``None`` when the backend is absent.

    Imported lazily and defensively: the GUI must start, and its test
    suite must run, on a box without the private backend stack.
    """
    try:
        from mlgiddetect.utils import path_utils
        return path_utils
    except Exception:
        logger.debug("mlgiddetect.utils.path_utils unavailable", exc_info=True)
        return None


def cache_dir() -> Path | None:
    """The directory mlgidDETECT caches downloaded models in."""
    path_utils = _path_utils()
    if path_utils is None:
        return None
    try:
        return Path(path_utils.get_data_dir())
    except Exception:
        logger.debug("could not resolve the mlgiddetect data dir", exc_info=True)
        return None


def _model_urls() -> dict[str, str] | None:
    """mlgidDETECT's ``MODEL_URLS`` table.

    Read from the installed package rather than duplicated here so a
    backend that adds or re-points a model stays authoritative.
    """
    path_utils = _path_utils()
    if path_utils is None:
        return None
    urls = getattr(path_utils, "MODEL_URLS", None)
    return dict(urls) if isinstance(urls, dict) else None


def _model_type_from_yaml(config_path: str) -> str | None:
    """Read ``MODEL_TYPE`` out of a mlgidDETECT YAML config.

    Parsed directly instead of through ``mlgiddetect.Config`` on
    purpose: that constructor calls ``sys.exit()`` on a missing or
    malformed file and resets the root logger level as a side effect,
    neither of which is acceptable inside a GUI worker. The nesting
    mirrors ``Config.load_config``, which flattens ``{SECTION: {KEY:
    value}}`` into ``SECTION_KEY``.

    PyYAML is used when present but is NOT a dependency of the GUI-only
    install profile (it arrives with the pipeline extras), so a
    minimal line parser covers its absence — precisely the profile
    this pre-flight exists for.
    """
    try:
        with open(config_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        logger.debug(
            "could not read MODEL_TYPE from %s; falling back to the "
            "mlgidDETECT default", config_path, exc_info=True,
        )
        return None
    try:
        import yaml
    except ImportError:
        return _model_type_without_pyyaml(text)
    try:
        data = yaml.safe_load(text)
        section = (data or {}).get("MODEL")
        if isinstance(section, dict):
            value = section.get("TYPE")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        logger.debug(
            "could not read MODEL_TYPE from %s; falling back to the "
            "mlgidDETECT default", config_path, exc_info=True,
        )
    return None


def _model_type_without_pyyaml(text: str) -> str | None:
    """``MODEL:`` / ``TYPE:`` from the flat two-level block layout
    mlgidDETECT configs use, without a YAML library. Deliberately
    minimal: an exotic config (flow style, anchors) simply resolves to
    None, which leaves the download decision to mlgidDETECT — the same
    safe fallback as any other parse failure here."""
    in_model = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line[:1] not in (" ", "\t"):
            in_model = line.strip() == "MODEL:"
            continue
        if not in_model:
            continue
        key, sep, value = line.strip().partition(":")
        if sep and key.strip() == "TYPE":
            value = value.strip().strip("'\"")
            return value or None
    return None


def resolve_model_key(
    model_type: str | None = None,
    config_detect: Any = None,
) -> str | None:
    """Which ``MODEL_URLS`` entry this detection run will load.

    Mirrors ``mlgidbase.mlgiddetect_functions.load_config`` followed by
    ``path_utils.get_model_path``. An explicit ``model_type`` wins over
    everything else: since mlgidbase 0.1.8 ``load_config`` ends with a
    plain ``config.MODEL_TYPE = model_type`` that runs on every branch,
    so the combo overrides a config file rather than the other way
    round. Up to 0.1.6 the str-config branch dropped the argument (the
    assignment there was a ``==`` comparison), which is why this used
    to read the file first.

    A ``dict`` config is the 0.1.8 addition: mlgidbase flattens
    ``{SECTION: {KEY: value}}`` onto ``SECTION_KEY`` attributes, so
    ``{"MODEL": {"TYPE": ...}}`` is where a dict keeps its model type.

    ``MODEL_ONNX_BASE`` is deliberately ignored — it is only consulted
    by ``inference.load_sessions``, which mlgidbase never calls; its
    ``load_inference`` constructs ``Inference(config)`` directly and so
    always lands in ``get_model_path``.

    Returns ``None`` for a model type we do not recognise, which leaves
    the download to mlgidDETECT (safe, because ``safe_console`` is
    active by then).
    """
    if model_type:
        resolved = model_type
    elif isinstance(config_detect, str) and config_detect.strip():
        resolved = _model_type_from_yaml(config_detect.strip()) or "dino"
    elif isinstance(config_detect, dict):
        section = config_detect.get("MODEL")
        resolved = (
            section.get("TYPE") if isinstance(section, dict) else None
        ) or "dino"
    elif config_detect is not None:
        resolved = getattr(config_detect, "MODEL_TYPE", None) or "dino"
    else:
        resolved = "dino"
    return _MODEL_TYPE_TO_KEY.get(str(resolved).strip())


def purge_broken_downloads(directory: Path) -> list[Path]:
    """Delete cache entries that can only ever fail to load.

    Two kinds: ``.onnx`` files too small to be a real model (the
    zero-byte residue of an interrupted ``urlretrieve``, which
    mlgidDETECT would otherwise accept forever) and abandoned
    ``.part`` files from our own downloader. Every removal is logged at
    INFO so the self-heal is visible in the GUI log panel rather than
    happening behind the user's back.
    """
    removed: list[Path] = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        return removed

    now = time.time()
    for path in entries:
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
            if path.name.endswith(_PART_SUFFIX):
                if now - path.stat().st_mtime < _STALE_PART_SECONDS:
                    continue  # possibly an in-flight download elsewhere
                reason = "abandoned partial download"
            elif path.suffix == ".onnx" and size < _MIN_SANE_BYTES:
                reason = f"truncated model file ({size} bytes)"
            else:
                continue
            path.unlink()
        except OSError:
            logger.debug("could not remove %s", path, exc_info=True)
            continue
        removed.append(path)
        logger.info(
            "Removed %s from the detection-model cache: %s. It will be "
            "re-downloaded on demand.", path.name, reason,
        )
    return removed


def _urlopen(url: str, timeout: int = _CONNECT_TIMEOUT):
    """Open ``url`` with certificate verification, falling back once.

    mlgidDETECT's downloader disables TLS verification process-wide
    (``ssl._create_default_https_context = _create_unverified_context``).
    We verify by default instead, but keep a one-shot unverified retry
    so users behind a TLS-inspecting corporate proxy — who worked
    before precisely *because* of that global — are not regressed by
    this fix. The fallback is loud.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "mlgidGUI"})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise
        logger.warning(
            "TLS certificate verification failed for %s (%s). Retrying "
            "without verification — this matches what mlgidDETECT's own "
            "downloader always does, but if you are not behind a known "
            "TLS-inspecting proxy, treat the download as untrusted.",
            url, exc.reason,
        )
        return urllib.request.urlopen(
            request, timeout=timeout, context=ssl._create_unverified_context()
        )


def _remote_size(url: str) -> int | None:
    """Content length of ``url``, or ``None`` when it cannot be learned.

    Best effort by design: no network (offline laptop, blocked proxy)
    must never turn a perfectly good cached model into an error.
    """
    try:
        with _urlopen(url, timeout=15) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        logger.debug("could not query the size of %s", url, exc_info=True)
        return None


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _download(url: str, destination: Path, key: str) -> Path:
    """Fetch ``url`` to ``destination`` atomically, or raise.

    Writes to a unique temp file in the same directory (so the rename
    is atomic and two GUI instances cannot interleave into one another's
    buffer), verifies the byte count against ``Content-Length``, and
    only then moves it into place. A failure therefore leaves the cache
    exactly as it was — never a corpse for the next run to trip over.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f"{key}.", suffix=_PART_SUFFIX,
    )
    os.close(handle)
    temp_path = Path(temp_name)

    try:
        with _urlopen(url) as response:
            length = response.headers.get("Content-Length")
            expected = int(length) if length else 0
            logger.info(
                "Downloading detection model %r (%s) to %s. The first "
                "detection run has to fetch this once; later runs reuse "
                "the cached copy.",
                key, _human(expected) if expected else "unknown size",
                destination.parent,
            )
            written = 0
            next_report = 10
            with open(temp_path, "wb") as sink:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    sink.write(chunk)
                    written += len(chunk)
                    if expected:
                        percent = written * 100 // expected
                        if percent >= next_report:
                            logger.info(
                                "Detection model %r: %d%% (%s / %s)",
                                key, percent, _human(written), _human(expected),
                            )
                            next_report = percent - (percent % 10) + 10

        if expected and written != expected:
            raise OSError(
                f"incomplete download: got {written} bytes, expected {expected}"
            )
        if written < _MIN_SANE_BYTES:
            raise OSError(
                f"implausibly small download ({written} bytes) — the server "
                f"did not return a model file"
            )

        os.replace(temp_path, destination)
        logger.info(
            "Detection model %r ready: %s (%s).",
            key, destination, _human(written),
        )
        _size_verified.add(str(destination))
        return destination
    except BaseException as exc:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise RuntimeError(
            f"Could not download the {key!r} detection model from {url}: "
            f"{exc}. mlgidDETECT ships no model weights — they are fetched "
            f"on first use into {destination.parent}. Check the machine's "
            f"network/proxy access to huggingface.co, or download the file "
            f"manually and save it as {destination}."
        ) from exc


def _cached_copy_is_sound(destination: Path, url: str) -> bool:
    """Whether an existing cache entry can be trusted without re-fetching.

    Size is the check that matters: a truncated file is the one failure
    mode mlgidDETECT's own existence test cannot see. Reconciled against
    the server once per process, and treated as sound whenever the
    server cannot be reached.
    """
    try:
        local = destination.stat().st_size
    except OSError:
        return False
    if local < _MIN_SANE_BYTES:
        return False
    if str(destination) in _size_verified:
        return True

    remote = _remote_size(url)
    _size_verified.add(str(destination))
    if remote is None or remote == local:
        return True
    logger.warning(
        "Cached detection model %s is %s but the server reports %s — the "
        "file is incomplete. Re-downloading it.",
        destination.name, _human(local), _human(remote),
    )
    return False


def _has_explicit_onnx_path(config_detect: Any) -> bool:
    """Does this config point mlgidDETECT at an ``.onnx`` file of its own?

    Only the dict form is inspected, because that is the form the
    Pipeline dock sends (``mlgidgui.detection_config``). ``"base"`` is
    mlgidDETECT's own default for ``MODEL_ONNX_BASE`` and means "the
    weights you download for MODEL_TYPE", so it is not an explicit
    path; neither is an empty string, which the panel maps back to
    ``"base"`` anyway.

    ``resolve_model_key`` deliberately ignores ``MODEL_ONNX_BASE`` (see
    its docstring: the field is read by ``inference.load_sessions``,
    which mlgidbase never calls), so the decision belongs here, at the
    pre-flight, and not in the key resolution.
    """
    if not isinstance(config_detect, dict):
        return False
    section = config_detect.get("MODEL")
    if not isinstance(section, dict):
        return False
    base = section.get("ONNX_BASE")
    return isinstance(base, str) and base.strip() not in ("", "base")


def ensure_detection_model(
    model_type: str | None = None,
    config_detect: Any = None,
) -> Path | None:
    """Make sure the model this run needs is cached and complete.

    Called before ``run_detection`` reaches mlgidbase so the download
    happens with progress in the GUI log, an actionable error on
    failure, and no chance of poisoning the cache — instead of inside
    ``Inference.__init__``, where any problem collapses into "Detection
    failed. Couldn't load the model."

    Returns the model path, or ``None`` when the pre-flight does not
    apply (backend absent, an explicit ``.onnx`` path configured, or an
    unrecognised model type left to mlgidDETECT). Raises
    ``RuntimeError`` only when a download was genuinely required and
    genuinely failed.
    """
    if _has_explicit_onnx_path(config_detect):
        # Nothing to fetch and nothing to verify: this whole pre-flight
        # exists to get the weights mlgidDETECT would download for
        # ``MODEL_TYPE`` on disk and intact. A user-supplied .onnx is
        # neither downloaded nor named by ``MODEL_URLS``, so the
        # pre-flight would at best check an unrelated file and at worst
        # download one that is never loaded.
        logger.info(
            "detection config sets MODEL_ONNX_BASE to an explicit .onnx "
            "path, skipping the model pre-flight"
        )
        return None

    urls = _model_urls()
    directory = cache_dir()
    if urls is None or directory is None:
        logger.debug("mlgiddetect unavailable — skipping the model pre-flight")
        return None

    purge_broken_downloads(directory)

    key = resolve_model_key(model_type, config_detect)
    if key is None or key not in urls:
        logger.debug(
            "no pre-flight for model type %r — leaving the download to "
            "mlgidDETECT", model_type,
        )
        return None

    destination = directory / f"{key}.onnx"
    url = urls[key]
    if destination.exists() and _cached_copy_is_sound(destination, url):
        return destination
    return _download(url, destination, key)
