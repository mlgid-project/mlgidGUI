"""Detection-model cache pre-flight.

Guards the release bug where a GUI-only install could never obtain the
mlgidDETECT weights. Three distinct failures had to line up, and each
gets a test here:

1. A GUI launched through the ``gui-scripts`` entry point runs under
   ``pythonw.exe`` on Windows, where ``sys.stdout is None``.
   mlgidDETECT's downloader renders a progress bar with a bare
   ``sys.stdout.write``, so the download died on its first tick with
   ``AttributeError: 'NoneType' object has no attribute 'write'``.
2. That aborted download left a zero-byte ``.onnx`` in the cache, and
   mlgidDETECT's ``check_filepath`` only tests existence plus the file
   extension — so the corpse was accepted forever after and no
   re-download was ever attempted.
3. Every one of those failures surfaced as the same opaque "Detection
   failed. Couldn't load the model.", because ``Inference.__init__``
   calls ``sys.exit()`` and mlgidbase catches it in a bare ``except:``.

No network: the HTTP layer is monkeypatched throughout.
"""
from __future__ import annotations

import io
import sys
import time

import pytest

from mlgidgui import detection_model


@pytest.fixture(autouse=True)
def _clear_verify_cache():
    """``_size_verified`` is process-global memoisation — reset it so
    tests cannot leak a "already checked" verdict into one another."""
    detection_model._size_verified.clear()
    yield
    detection_model._size_verified.clear()


@pytest.fixture
def small_models(monkeypatch):
    """Shrink the plausibility floor so tests move bytes, not megabytes."""
    monkeypatch.setattr(detection_model, "_MIN_SANE_BYTES", 16)
    return 16


class _FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, payload: bytes, declared_length: int | None = None):
        self._buffer = io.BytesIO(payload)
        length = len(payload) if declared_length is None else declared_length
        self.headers = {"Content-Length": str(length)} if length is not None else {}

    def read(self, size=-1):
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _serve(monkeypatch, payload: bytes, declared_length: int | None = None):
    """Point the module's HTTP layer at a fixed payload; count calls."""
    calls: list[str] = []

    def _fake_urlopen(url, timeout=None):
        calls.append(url)
        return _FakeResponse(payload, declared_length)

    monkeypatch.setattr(detection_model, "_urlopen", _fake_urlopen)
    return calls


def _wire_cache(monkeypatch, tmp_path, key="dino"):
    """Redirect the model cache into ``tmp_path`` with one known model."""
    monkeypatch.setattr(detection_model, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        detection_model, "_model_urls",
        lambda: {key: f"https://example.invalid/{key}.onnx"},
    )


# --- safe_console -----------------------------------------------------

def test_safe_console_substitutes_a_none_stdout(monkeypatch):
    """The pythonw case: a bare ``sys.stdout.write`` — exactly what
    mlgidDETECT's progress bar does — must not raise inside the guard."""
    monkeypatch.setattr(sys, "stdout", None)
    with detection_model.safe_console():
        assert sys.stdout is not None
        sys.stdout.write("progress 50%")  # the original crash site
        sys.stdout.flush()


def test_safe_console_restores_the_original_stream(monkeypatch):
    """The substitution is scoped: whatever was there comes back, so a
    later failure can't be blamed on the guard."""
    monkeypatch.setattr(sys, "stdout", None)
    with detection_model.safe_console():
        pass
    assert sys.stdout is None


def test_safe_console_leaves_a_working_stdout_alone(monkeypatch):
    """A GUI started from a terminal keeps its real console — output
    must not be silently diverted to devnull."""
    real = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real)
    with detection_model.safe_console():
        assert sys.stdout is real
        sys.stdout.write("visible")
    assert real.getvalue() == "visible"


def test_safe_console_substitutes_a_closed_stream(monkeypatch):
    """A closed stream is as unusable as ``None`` (a detached console
    on Windows lands here rather than on ``None``)."""
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stdout", closed)
    with detection_model.safe_console():
        sys.stdout.write("no raise")


# --- model-key resolution --------------------------------------------

def test_resolve_model_key_defaults_to_dino():
    """No model_type means mlgidDETECT's own default, MODEL_TYPE='dino'."""
    assert detection_model.resolve_model_key() == "dino"


def test_resolve_model_key_maps_faster_rcnn_to_its_cache_name():
    """The GUI's combo says 'faster_rcnn'; the cached file is frcnn.onnx."""
    assert detection_model.resolve_model_key("faster_rcnn") == "frcnn"


def test_resolve_model_key_reads_a_yaml_config(tmp_path):
    """With the combo left on "(default)" the config file's MODEL_TYPE
    is what actually runs, so it is what we must pre-fetch."""
    config = tmp_path / "detect.yaml"
    config.write_text("MODEL:\n  TYPE: faster_rcnn\n", encoding="utf-8")
    assert detection_model.resolve_model_key(None, str(config)) == "frcnn"


def test_resolve_model_key_lets_the_combo_override_a_config_file(tmp_path):
    """mlgidbase 0.1.8 applies ``model_type`` on every branch of
    ``load_config``, including the str-config one that used to drop it,
    so an explicit combo choice beats the file."""
    config = tmp_path / "detect.yaml"
    config.write_text("MODEL:\n  TYPE: faster_rcnn\n", encoding="utf-8")
    assert detection_model.resolve_model_key("dino", str(config)) == "dino"


def test_resolve_model_key_reads_a_dict_config():
    """0.1.8 also accepts a dict, flattened onto ``SECTION_KEY``."""
    assert detection_model.resolve_model_key(
        None, {"MODEL": {"TYPE": "faster_rcnn"}}
    ) == "frcnn"
    assert detection_model.resolve_model_key(None, {"MODEL": {}}) == "dino"


def test_resolve_model_key_reads_yaml_without_pyyaml(tmp_path, monkeypatch):
    """PyYAML is not part of the GUI-only install (it arrives with the
    pipeline extras), and that profile is exactly who this pre-flight
    serves — the config must still be honoured through the fallback
    line parser. Regression: CI (no extras) resolved every config to
    'dino'."""
    import sys

    monkeypatch.setitem(sys.modules, "yaml", None)  # import yaml -> ImportError
    config = tmp_path / "detect.yaml"
    config.write_text(
        "# comment\nMODEL:\n  ONNX_BASE: dino_old\n  TYPE: 'faster_rcnn'\n"
        "OTHER:\n  TYPE: not_this_one\n",
        encoding="utf-8",
    )
    assert detection_model.resolve_model_key(None, str(config)) == "frcnn"
    # Still safe on a missing file / unknown type.
    assert detection_model.resolve_model_key(
        None, str(tmp_path / "gone.yaml")
    ) == "dino"


def test_resolve_model_key_survives_an_unreadable_config(tmp_path):
    """A missing config must not raise here — mlgidDETECT's own Config
    would ``sys.exit()`` on it, which is precisely why we parse the
    YAML ourselves. Fall back to the default model."""
    assert detection_model.resolve_model_key(None, str(tmp_path / "gone.yaml")) == "dino"


def test_resolve_model_key_declines_unknown_types():
    """An unrecognised type yields None, leaving the download to
    mlgidDETECT rather than guessing at the wrong file."""
    assert detection_model.resolve_model_key("some_future_net") is None


# --- cache hygiene ----------------------------------------------------

def test_purge_removes_a_zero_byte_model(tmp_path, small_models):
    """The exact corpse the old failure left behind. Without this, the
    install stays broken forever: the file exists and ends in .onnx, so
    mlgidDETECT accepts it and onnxruntime then fails to load it."""
    corpse = tmp_path / "dino.onnx"
    corpse.touch()
    removed = detection_model.purge_broken_downloads(tmp_path)
    assert corpse in removed
    assert not corpse.exists()


def test_purge_keeps_a_complete_model(tmp_path, small_models):
    """A real cached model must never be deleted — re-downloading 1.1 GB
    because of an over-eager cleanup would be its own bug."""
    good = tmp_path / "dino.onnx"
    good.write_bytes(b"x" * 512)
    assert detection_model.purge_broken_downloads(tmp_path) == []
    assert good.exists()


def test_purge_removes_an_abandoned_part_file(tmp_path, small_models):
    """Our own temp files are cleaned up on failure, but a killed
    process leaves them; sweep the old ones."""
    stale = tmp_path / f"dino.abc{detection_model._PART_SUFFIX}"
    stale.write_bytes(b"x" * 512)
    old = time.time() - detection_model._STALE_PART_SECONDS - 60
    import os
    os.utime(stale, (old, old))
    assert stale in detection_model.purge_broken_downloads(tmp_path)


def test_purge_spares_a_fresh_part_file(tmp_path, small_models):
    """A recent .part may be an in-flight download in a second mlgidGUI
    instance — deleting it would corrupt that run."""
    fresh = tmp_path / f"dino.abc{detection_model._PART_SUFFIX}"
    fresh.write_bytes(b"x" * 512)
    assert detection_model.purge_broken_downloads(tmp_path) == []
    assert fresh.exists()


# --- ensure_detection_model ------------------------------------------

def test_downloads_when_the_cache_is_empty(tmp_path, monkeypatch, small_models):
    """The headline fix: a fresh GUI-only install really does fetch the
    weights, and the bytes land intact."""
    payload = b"onnx-model-bytes" * 8
    _wire_cache(monkeypatch, tmp_path)
    _serve(monkeypatch, payload)

    result = detection_model.ensure_detection_model()

    assert result == tmp_path / "dino.onnx"
    assert result.read_bytes() == payload


def test_download_works_without_a_console(tmp_path, monkeypatch, small_models):
    """End-to-end regression for the reported bug: under pythonw
    (``sys.stdout is None``) the download must still complete. This is
    the exact combination that produced a zero-byte cache file in the
    released build."""
    payload = b"onnx-model-bytes" * 8
    _wire_cache(monkeypatch, tmp_path)
    _serve(monkeypatch, payload)
    monkeypatch.setattr(sys, "stdout", None)

    with detection_model.safe_console():
        result = detection_model.ensure_detection_model()

    assert result.read_bytes() == payload


def test_reuses_a_complete_cached_model(tmp_path, monkeypatch, small_models):
    """A verified model is reused; the second run must not re-fetch
    1.1 GB. One HEAD-equivalent size check per process is the budget."""
    payload = b"onnx-model-bytes" * 8
    _wire_cache(monkeypatch, tmp_path)
    calls = _serve(monkeypatch, payload)
    (tmp_path / "dino.onnx").write_bytes(payload)

    detection_model.ensure_detection_model()
    detection_model.ensure_detection_model()

    assert len(calls) == 1  # the size probe only, and only once


def test_redownloads_a_truncated_cached_model(tmp_path, monkeypatch, small_models):
    """Self-heal: a short file is what an interrupted download leaves,
    and it is invisible to mlgidDETECT's existence-only check."""
    payload = b"onnx-model-bytes" * 8
    _wire_cache(monkeypatch, tmp_path)
    _serve(monkeypatch, payload)
    truncated = tmp_path / "dino.onnx"
    truncated.write_bytes(payload[:32])

    result = detection_model.ensure_detection_model()

    assert result.read_bytes() == payload


def test_zero_byte_cache_entry_is_replaced(tmp_path, monkeypatch, small_models):
    """The precise state an affected user's machine is in right now:
    running detection again must fix it rather than fail again."""
    payload = b"onnx-model-bytes" * 8
    _wire_cache(monkeypatch, tmp_path)
    _serve(monkeypatch, payload)
    (tmp_path / "dino.onnx").touch()

    assert detection_model.ensure_detection_model().read_bytes() == payload


def test_failed_download_leaves_no_corpse(tmp_path, monkeypatch, small_models):
    """The property that made the original bug permanent. A failure
    must leave the cache untouched, so the next run retries cleanly
    instead of inheriting a file that can only fail to load."""
    _wire_cache(monkeypatch, tmp_path)

    def _boom(url, timeout=None):
        raise OSError("network unreachable")

    monkeypatch.setattr(detection_model, "_urlopen", _boom)

    with pytest.raises(RuntimeError, match="Could not download"):
        detection_model.ensure_detection_model()

    assert list(tmp_path.iterdir()) == []


def test_truncated_transfer_is_rejected(tmp_path, monkeypatch, small_models):
    """A server that closes early must not produce a cached model:
    the byte count is reconciled against Content-Length before the
    file is moved into place."""
    _wire_cache(monkeypatch, tmp_path)
    _serve(monkeypatch, b"short", declared_length=4096)

    with pytest.raises(RuntimeError, match="incomplete download"):
        detection_model.ensure_detection_model()

    assert not (tmp_path / "dino.onnx").exists()


def test_error_names_the_cache_dir_and_url(tmp_path, monkeypatch, small_models):
    """The old failure said only "Couldn't load the model". The
    replacement has to tell the user where the file belongs and where
    it comes from, so a manual download is possible on an air-gapped
    or proxied machine."""
    _wire_cache(monkeypatch, tmp_path)

    def _boom(url, timeout=None):
        raise OSError("blocked by proxy")

    monkeypatch.setattr(detection_model, "_urlopen", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        detection_model.ensure_detection_model()

    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "https://example.invalid/dino.onnx" in message


def test_is_a_noop_without_the_backend(monkeypatch):
    """The GUI and its CI suite must run with no mlgidDETECT installed."""
    monkeypatch.setattr(detection_model, "_model_urls", lambda: None)
    monkeypatch.setattr(detection_model, "cache_dir", lambda: None)
    assert detection_model.ensure_detection_model() is None


def test_unknown_model_type_defers_to_mlgiddetect(tmp_path, monkeypatch):
    """We only pre-fetch what we can name with certainty; anything else
    falls through to mlgidDETECT (safe, since safe_console is active)."""
    _wire_cache(monkeypatch, tmp_path)
    assert detection_model.ensure_detection_model("some_future_net") is None
