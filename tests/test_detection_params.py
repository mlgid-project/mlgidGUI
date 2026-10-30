"""Interactive detection parameters in the Pipeline dock.

Detection used to be configurable only through a YAML file picker; it
now has the same kind of widget surface as Fitting and Matching, and
sends it as mlgidBASE's dict-form ``config_detect``. What these tests
hold down:

* the dict an untouched form sends is mlgidDETECT's own defaults, so
  growing these fields cannot have changed what a run does;
* the nesting from flat widget keys back to ``{SECTION: {KEY: value}}``,
  including the keys that contain underscores themselves;
* the YAML picker is really gone, and the Model combo still travels as
  the separate ``model_type`` kwarg;
* the download pre-flight stands aside for a user-supplied ``.onnx``.

The drift guard (``test_defaults_match_mlgiddetects_own``) needs the
real backend and skips without it; everything else runs backend-free,
which is the CI profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit

from mlgidgui import detection_model
from mlgidgui.detection_config import (
    DETECT_DEFAULTS,
    build_config_detect,
    flatten_defaults,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qtbot, monkeypatch):
    # The detection form is pure Qt (no backend call happens until a
    # command actually RUNS), but the panel builds only an install hint
    # without mlgidbase. Force availability so these tests cover the
    # real widgets on a backend-free box too.
    from mlgidgui import pipeline_panel as panel_mod

    monkeypatch.setattr(panel_mod, "is_mlgidbase_available", lambda: True)
    p = panel_mod.PipelinePanel()
    qtbot.addWidget(p)
    return p


def _commands(panel):
    got: list = []
    panel.runRequested.connect(got.append)
    return got


def _defaults_as_sent() -> dict:
    """``DETECT_DEFAULTS`` in the shape the panel sends it: unchanged,
    minus the ``None``-valued ONNX ensemble key."""
    expected = {
        section: dict(settings)
        for section, settings in DETECT_DEFAULTS.items()
    }
    del expected["MODEL"]["ONNX_ENSEMBLE"]
    return expected


# --- what an untouched form sends ------------------------------------

def test_untouched_form_sends_mlgiddetects_defaults(panel):
    """The reason the dict can be sent unconditionally: it is a no-op."""
    got = _commands(panel)
    panel._on_run_detection()

    assert len(got) == 1
    assert got[0].op_name == "run_detection"
    assert got[0].kwargs["config_detect"] == _defaults_as_sent()
    # The combo sits on "(default)", so the model type is mlgidbase's
    # business and must not be forced from here.
    assert "model_type" not in got[0].kwargs


def test_the_model_combo_still_travels_as_its_own_kwarg(panel):
    """``MODEL.TYPE`` deliberately stays out of the dict: load_config
    assigns ``model_type`` after it, so one source of truth is the only
    way the pre-flight and the run agree on which weights load."""
    got = _commands(panel)
    panel.det_model_type.setCurrentText("faster_rcnn")
    panel._on_run_detection()

    assert got[0].kwargs["model_type"] == "faster_rcnn"
    assert "TYPE" not in got[0].kwargs["config_detect"]["MODEL"]


# --- rows follow the selected model ----------------------------------

# What the dino branch of mlgidDETECT reads and the faster_rcnn branch
# does not: the five post-processing settings, plus the ensemble pair
# (faster_rcnn never fuses models).
_DINO_ONLY = (
    "det_score",
    "det_nms_iou",
    "det_classaware_nms",
    "det_nms_iou_ring",
    "det_nms_iou_seg",
    "det_ensemble",
    "det_onnx_ensemble",
)
# Read whichever model is chosen.
_ALWAYS_SHOWN = (
    "det_force_cpu",
    "det_onnx_base",
    "det_redownload",
    "det_log",
    "det_clip_upper",
    "det_debug",
)


def _hidden(panel, names) -> list[bool]:
    return [getattr(panel, n).isHidden() for n in names]


def test_dino_only_rows_hide_for_faster_rcnn_and_return(panel):
    panel.det_model_type.setCurrentText("faster_rcnn")
    assert all(_hidden(panel, _DINO_ONLY))
    assert not any(_hidden(panel, _ALWAYS_SHOWN))

    panel.det_model_type.setCurrentText("dino")
    assert not any(_hidden(panel, _DINO_ONLY))
    assert not any(_hidden(panel, _ALWAYS_SHOWN))


def test_the_default_model_shows_every_row(panel):
    """``(default)`` resolves to dino, so nothing is hidden on it, at
    startup or after coming back from faster_rcnn."""
    assert panel.det_model_type.currentText() == "(default)"
    assert not any(_hidden(panel, _DINO_ONLY + _ALWAYS_SHOWN))

    panel.det_model_type.setCurrentText("faster_rcnn")
    panel.det_model_type.setCurrentText("(default)")
    assert not any(_hidden(panel, _DINO_ONLY + _ALWAYS_SHOWN))


def test_a_hidden_row_hides_its_label_too(panel):
    form_labels = []
    for form, field in panel._det_dino_rows:
        label = form.labelForField(field)
        assert label is not None
        form_labels.append(label)
    assert len(form_labels) == len(_DINO_ONLY)

    panel.det_model_type.setCurrentText("faster_rcnn")
    assert all(label.isHidden() for label in form_labels)
    panel.det_model_type.setCurrentText("dino")
    assert not any(label.isHidden() for label in form_labels)


def test_hidden_rows_still_send_their_values(panel):
    """Hiding is presentation only: the dict is the one dino gets, and
    an edit made before switching survives the round trip."""
    got = _commands(panel)
    panel.det_score.setValue(0.15)
    panel.det_model_type.setCurrentText("faster_rcnn")
    panel._on_run_detection()
    panel.det_model_type.setCurrentText("dino")
    panel._on_run_detection()

    expected = _defaults_as_sent()
    expected["POSTPROCESSING"]["SCORE"] = pytest.approx(0.15)
    assert got[0].kwargs["config_detect"] == expected
    assert got[1].kwargs["config_detect"] == expected
    assert got[0].kwargs["model_type"] == "faster_rcnn"
    assert got[1].kwargs["model_type"] == "dino"


def test_reset_reaches_hidden_rows(panel):
    got = _commands(panel)
    panel.det_score.setValue(0.9)
    panel.det_ensemble.setChecked(True)
    panel.det_model_type.setCurrentText("faster_rcnn")
    panel._reset_detection_params()
    panel._on_run_detection()

    assert got[0].kwargs["config_detect"] == _defaults_as_sent()


# --- edits ------------------------------------------------------------

def test_edits_land_in_the_right_section(panel):
    got = _commands(panel)
    panel.det_score.setValue(0.15)
    panel.det_classaware_nms.setChecked(True)
    panel._on_run_detection()

    sent = got[0].kwargs["config_detect"]
    assert sent["POSTPROCESSING"]["SCORE"] == pytest.approx(0.15)
    assert sent["POSTPROCESSING"]["CLASSAWARE_NMS"] is True
    # Nothing else moved.
    expected = _defaults_as_sent()
    expected["POSTPROCESSING"]["SCORE"] = pytest.approx(0.15)
    expected["POSTPROCESSING"]["CLASSAWARE_NMS"] = True
    assert sent == expected


def test_reset_restores_every_widget(panel):
    """Edits spanning both the inline rows and the Advanced section."""
    got = _commands(panel)
    panel.det_score.setValue(0.9)
    panel.det_nms_iou.setValue(0.75)
    panel.det_force_cpu.setChecked(True)
    panel.det_classaware_nms.setChecked(True)
    panel.det_nms_iou_ring.setValue(0.5)
    panel.det_clip_upper.setValue(80.0)
    panel.det_log.setChecked(False)
    panel.det_debug.setChecked(True)
    panel.det_onnx_base.setText("/models/custom.onnx")
    panel.det_onnx_ensemble.setText("/models/second.onnx")

    panel._reset_detection_params()
    panel._on_run_detection()

    assert got[0].kwargs["config_detect"] == _defaults_as_sent()
    # The text fields are empty again, not filled with the literal
    # default: in this form an empty field IS the default.
    assert panel.det_onnx_base.text() == ""
    assert panel.det_onnx_ensemble.text() == ""


# --- the YAML picker is gone -----------------------------------------

def test_the_yaml_config_row_is_gone(panel):
    assert not hasattr(panel, "det_config_path")
    assert not hasattr(panel, "_browse_detect_config")


# --- the nesting, without Qt -----------------------------------------

def test_build_config_detect_nests_flat_keys():
    nested = build_config_detect(
        {"POSTPROCESSING_SCORE": 0.25, "GENERAL_DEBUG": True}
    )
    assert nested == {
        "POSTPROCESSING": {"SCORE": 0.25},
        "GENERAL": {"DEBUG": True},
    }


def test_build_config_detect_keeps_underscored_keys_whole():
    """What a naive ``split("_", 1)`` gets wrong: the key itself has
    underscores, so only a known-section prefix match works."""
    nested = build_config_detect({
        "POSTPROCESSING_CLASSAWARE_NMS": True,
        "POSTPROCESSING_NMSIOU_RING": 0.2,
        "MODEL_ONNX_BASE": "base",
    })
    assert nested == {
        "POSTPROCESSING": {"CLASSAWARE_NMS": True, "NMSIOU_RING": 0.2},
        "MODEL": {"ONNX_BASE": "base"},
    }


def test_build_config_detect_rejects_an_unknown_section():
    with pytest.raises(ValueError):
        build_config_detect({"GEOMETRY_QMAX": 3.0})


def test_flatten_defaults_round_trips():
    assert build_config_detect(flatten_defaults()) == _defaults_as_sent()


# --- the two text fields ---------------------------------------------

def test_empty_onnx_fields_mean_the_defaults(panel):
    got = _commands(panel)
    panel._on_run_detection()

    model = got[0].kwargs["config_detect"]["MODEL"]
    assert model["ONNX_BASE"] == "base"
    # None is mlgidDETECT's own default for the ensemble path, so
    # spelling it out would only add noise.
    assert "ONNX_ENSEMBLE" not in model


def test_onnx_paths_are_passed_through_verbatim(panel):
    got = _commands(panel)
    panel.det_onnx_base.setText("/models/base.onnx")
    panel.det_onnx_ensemble.setText("/models/ensemble.onnx")
    panel._on_run_detection()

    model = got[0].kwargs["config_detect"]["MODEL"]
    assert model["ONNX_BASE"] == "/models/base.onnx"
    assert model["ONNX_ENSEMBLE"] == "/models/ensemble.onnx"


# --- the chained run -------------------------------------------------

def test_run_all_carries_the_same_dict(panel):
    """``_on_run_all`` delegates to ``_on_run_detection``, so the
    chained run must inherit the edited parameters."""
    got = _commands(panel)
    # Run-all is gated on a matching source being set.
    panel.cif_path.setText("/data/cifs/one.cif")
    panel.det_score.setValue(0.2)
    panel._on_run_all()

    detection = [c for c in got if c.op_name == "run_detection"]
    assert len(detection) == 1
    sent = detection[0].kwargs["config_detect"]
    assert sent["POSTPROCESSING"]["SCORE"] == pytest.approx(0.2)


# --- widget plumbing -------------------------------------------------

def test_every_default_has_exactly_one_widget(panel):
    """The registry is what the dict and the reset both walk, so a key
    with no widget would be sent as a stale default forever."""
    assert set(panel._det_param_widgets) == set(flatten_defaults())
    for key, widget in panel._det_param_widgets.items():
        assert isinstance(widget, (QCheckBox, QDoubleSpinBox, QLineEdit)), key


# --- drift guard -----------------------------------------------------

def test_defaults_match_mlgiddetects_own():
    """Every exposed default must still equal mlgidDETECT's.

    The whole design rests on it: the panel sends the dict on every run,
    so a default that drifts from the backend's silently changes
    detection results for anyone who never opens the section.

    ``Config.__new__`` + ``init_default`` rather than ``Config()``: the
    real constructor also runs ``check_cuda_support`` (which touches
    ``cv2.cuda``) and ``set_logging_level`` (which mutates the root
    logger's level for the rest of the session).
    """
    pytest.importorskip("mlgiddetect")
    from mlgiddetect.configuration import Config

    cfg = Config.__new__(Config)
    Config.init_default(cfg)

    for section, settings in DETECT_DEFAULTS.items():
        for key, value in settings.items():
            attribute = f"{section}_{key}"
            assert hasattr(cfg, attribute), attribute
            assert getattr(cfg, attribute) == value, attribute


# --- the pre-flight stands aside for an explicit .onnx ---------------

@pytest.fixture
def preflight(monkeypatch, tmp_path):
    """Point the model pre-flight at an empty temp cache and stub the
    download, so nothing reaches the network. Returns the list of
    destinations a download was attempted for."""
    attempted: list[Path] = []

    def _fake_download(url, destination, key):
        attempted.append(destination)
        return destination

    monkeypatch.setattr(detection_model, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        detection_model, "_model_urls",
        lambda: {"dino": "https://example.invalid/dino.onnx"},
    )
    monkeypatch.setattr(detection_model, "_download", _fake_download)
    return attempted


def test_an_explicit_onnx_path_skips_the_preflight(preflight, caplog):
    config = {"MODEL": {"ONNX_BASE": "/models/custom.onnx"}}

    with caplog.at_level("INFO", logger="mlgidgui.detection_model"):
        assert detection_model.ensure_detection_model(config_detect=config) is None

    assert preflight == []
    assert "skipping the model pre-flight" in caplog.text


def test_the_default_onnx_base_still_runs_the_preflight(preflight):
    config = {"MODEL": {"ONNX_BASE": "base", "FORCE_CPU": False}}

    detection_model.ensure_detection_model(config_detect=config)

    assert preflight  # the weights are still fetched and verified


def test_a_dict_without_onnx_base_still_runs_the_preflight(preflight):
    detection_model.ensure_detection_model(
        config_detect={"POSTPROCESSING": {"SCORE": 0.4}}
    )

    assert preflight
