"""Detection parameters the Pipeline dock exposes, and their defaults.

Detection used to be the one pipeline stage whose behaviour could not be
dialled in from the panel: Fitting and Matching expose their full kwarg
surface as spin boxes and combos, while Detection offered only a picker
for a YAML config file, which meant leaving the GUI, learning
mlgidDETECT's config schema, and editing a file to try a different score
threshold. The hook that removes all of that already exists in the
backend: ``mlgidbase.mlgiddetect_functions.load_config`` accepts a
**dict** shaped ``{SECTION: {KEY: value}}`` and applies it with
``setattr(config, f"{SECTION}_{KEY}", value)`` onto a fresh
``mlgiddetect`` ``Config``. So the panel can build that dict from
widgets and hand it over as ``config_detect`` with no backend change and
no file on disk.

This module is the Qt-free half of that: the defaults and the nesting
rule, so both are unit-testable without constructing a widget.

Every value in ``DETECT_DEFAULTS`` is copied verbatim from
``mlgiddetect.configuration.Config.init_default``. That is what makes
the form safe to send unconditionally: an untouched panel reproduces
today's detection behaviour exactly, rather than quietly imposing the
GUI's opinion of a good threshold. ``tests/test_detection_params.py``
carries the drift guard that compares every key below against
mlgidDETECT's own defaults, so an upstream change of heart fails a test
instead of silently changing results.

``MODEL_TYPE`` is deliberately absent: the panel's **Model** combo keeps
travelling as mlgidBASE's separate ``model_type`` kwarg, which
``load_config`` assigns *after* the dict. Keeping it out of the dict
leaves one source of truth and leaves
``detection_model.resolve_model_key``'s precedence untouched.

Not every ``Config`` field is exposed, and the omissions are deliberate:

* ``GEO_RECIPROCAL_SHAPE`` / ``GEO_PIXELPERANGSTROEM`` are overwritten
  per frame by ``mlgiddetect.dataloader.imgcontainer.from_pygid``, which
  sets them from the image shape and the q_z maximum. A value typed in
  the panel would survive exactly until the first frame is loaded.
* ``GEO_QMAX`` is likewise recomputed from the data, in
  ``mlgiddetect.preprocessing.geometry`` and ``dataloader.pygidloader``.
* ``INPUT_*`` / ``OUTPUT_*`` are unused on the NeXus path: mlgidbase
  hands mlgidDETECT the image array directly and writes the results back
  into the open file, so nothing reads an input path or an output folder.
* ``PREPROCESSING_CUDA`` needs cupy plus a CUDA-enabled OpenCV build;
  ``Config.check_cuda_support`` resets it to False when either is
  missing, so a checkbox for it would do nothing on most installs and
  would not say why.
* ``PREPROCESSING_POLAR_CONVERSION`` / ``PREPROCESSING_QUAZIPOLAR`` /
  ``PREPROCESSING_POLAR_SHAPE`` decide the grid the model sees. The
  weights are trained on the 512x1024 polar grid, and postprocessing
  maps the predicted boxes back to q through
  ``PREPROCESSING_POLAR_SHAPE`` (``postprocessing.geometry``), so
  changing any of them silently invalidates the box coordinate
  round-trip rather than producing a differently-tuned run.
"""
from __future__ import annotations

# Mirrors ``Config.init_default`` key for key (mlgiddetect 0.2.8).
DETECT_DEFAULTS: dict[str, dict[str, object]] = {
    "GENERAL":        {"DEBUG": False},
    "MODEL":          {"REDOWNLOAD": False, "FORCE_CPU": False,
                       "ENSEMBLE_ENABLED": False,
                       "ONNX_BASE": "base", "ONNX_ENSEMBLE": None},
    "PREPROCESSING":  {"LOG": True, "HISTOGRAMEQUALIZATION": True,
                       "PERFORMCLIPPING": True,
                       "HIGHERCLIPPINGPERCENTILE": 99.5,
                       "LOWERCLIPPINGPERCENTILE": 5.0,
                       "FLIPHORIZONTAL": False},
    "POSTPROCESSING": {"SCORE": 0.4, "NMSIOU": 0.4,
                       "CLASSAWARE_NMS": False,
                       "NMSIOU_RING": 0.1, "NMSIOU_SEG": 0.4},
}

# Section names, longest first so prefix matching cannot be fooled by a
# future section whose name starts with another's.
DETECT_SECTIONS: tuple[str, ...] = tuple(
    sorted(DETECT_DEFAULTS, key=len, reverse=True)
)


def flatten_defaults() -> dict[str, object]:
    """``DETECT_DEFAULTS`` as a flat ``{"SECTION_KEY": value}`` map.

    The flat form is what the panel's widget registry is keyed by (one
    key per widget), so seeding a widget and resetting it both read the
    same numbers the nested table holds.
    """
    return {
        f"{section}_{key}": value
        for section, settings in DETECT_DEFAULTS.items()
        for key, value in settings.items()
    }


def build_config_detect(flat: dict[str, object]) -> dict[str, dict[str, object]]:
    """Nest a flat ``{"SECTION_KEY": value}`` map into ``load_config``'s shape.

    Splitting is done by matching against the known section names, not
    by ``split("_", 1)``: several keys contain underscores themselves
    (``CLASSAWARE_NMS``, ``ONNX_BASE``, ``NMSIOU_RING``), so a naive
    split would file ``POSTPROCESSING_CLASSAWARE_NMS`` under the key
    ``CLASSAWARE_NMS``' first half and lose the rest.

    ``MODEL_ONNX_ENSEMBLE`` is dropped when it is ``None``, which is
    mlgidDETECT's own default for it: sending it explicitly says nothing
    and only makes the dict in the log noisier.

    Raises ``ValueError`` for a key that belongs to no known section,
    which catches a typo at the call site rather than letting it reach
    ``setattr`` and invent a ``Config`` attribute nothing reads.
    """
    nested: dict[str, dict[str, object]] = {}
    for flat_key, value in flat.items():
        for section in DETECT_SECTIONS:
            prefix = f"{section}_"
            if flat_key.startswith(prefix):
                key = flat_key[len(prefix):]
                break
        else:
            raise ValueError(
                f"{flat_key!r} does not start with a known detection "
                f"config section ({', '.join(sorted(DETECT_DEFAULTS))})"
            )
        if section == "MODEL" and key == "ONNX_ENSEMBLE" and value is None:
            continue
        nested.setdefault(section, {})[key] = value
    return nested
