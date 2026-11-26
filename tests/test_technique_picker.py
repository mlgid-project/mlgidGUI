"""Conversion: techniques are picked from the bundled PaNET list.

Source: panet.py (bundled list and search), technique_picker.py
(``TechniquePickerDialog``), conversion_panel.py ``_select_technique`` /
``add_technique``, tools/update_panet.py (builds the list).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QPushButton,
)

from mlgidgui import conversion_panel, panet
from mlgidgui.conversion_config import ConversionConfig
from mlgidgui.technique_picker import TechniquePickerDialog

pytestmark = pytest.mark.gui

BASE = "http://purl.org/pan-science/PaNET/"
# The two techniques pygid records by itself (pygid datasaver.py).
XRD = ("x-ray diffraction", BASE + "PaNET01216")
GIWAXS = (
    "grazing incidence wide angle x-ray scattering", BASE + "PaNET01316",
)
XRR = ("x-ray reflectivity", BASE + "PaNET01149")


def _pairs(techniques):
    return [(t.name, t.pid) for t in techniques]


def _find(query):
    return panet.search(panet.load_catalogue().techniques, query)


# --- the bundled list -------------------------------------------------------

def test_bundled_list_is_complete_and_unambiguous():
    catalogue = panet.load_catalogue()
    pairs = _pairs(catalogue.techniques)
    assert len(pairs) > 300
    assert len({pid for _n, pid in pairs}) == len(pairs)
    assert len({name for name, _p in pairs}) == len(pairs)
    assert all(pid.startswith(BASE + "PaNET") for _n, pid in pairs)
    # pygid's own names and PIDs are in it verbatim, so picking one of
    # them cannot add a near-duplicate to the file.
    assert {XRD, GIWAXS, XRR} <= set(pairs)
    # The abstract grouping nodes and the specifier branch are left out.
    names = {name for name, _p in pairs}
    assert not any(name.startswith("defined by") for name in names)
    assert names.isdisjoint({"photon and neutron technique", "photon", "probe"})
    assert catalogue.released and catalogue.version
    assert "creativecommons.org/licenses/by/4.0" in catalogue.license


@pytest.mark.parametrize("query, first", [
    ("giwaxs", GIWAXS),
    ("XRR", XRR),
    ("xrd", XRD),
    ("x ray reflectivity", XRR),
    ("xray diffraction", XRD),
    ("X-Ray Diffraction", XRD),
    ("grazing wide scattering x-ray", GIWAXS),
    ("PaNET01316", GIWAXS),
    ("01316", GIWAXS),
    (GIWAXS[1], GIWAXS),
])
def test_search_puts_the_expected_technique_first(query, first):
    assert _pairs(_find(query))[0] == first


def test_search_empty_and_unmatched():
    everything = panet.load_catalogue().techniques
    assert _find("   ") == list(everything)
    assert _find("zzzz-no-such-technique") == []


def test_search_needs_every_word():
    names = [t.name for t in _find("grazing neutron")]
    assert names and all("grazing" in n and "neutron" in n for n in names)


def test_update_script_keeps_techniques_only():
    path = Path(__file__).resolve().parents[1] / "tools" / "update_panet.py"
    spec = importlib.util.spec_from_file_location("update_panet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def cls(ident, label, parent="", extra=""):
        sup = (f'<rdfs:subClassOf rdf:resource="{BASE}{parent}"/>'
               if parent else "")
        return (f'<owl:Class rdf:about="{BASE}{ident}">{sup}'
                f"<rdfs:label>{label}</rdfs:label>{extra}</owl:Class>")

    owl = (
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        ' xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"'
        ' xmlns:owl="http://www.w3.org/2002/07/owl#"'
        ' xmlns:skos="http://www.w3.org/2004/02/skos/core#"'
        ' xmlns:terms="http://purl.org/dc/terms/">'
        '<owl:Ontology rdf:about="x"><terms:title>T</terms:title>'
        "<terms:created>2030-01-02</terms:created>"
        "<owl:versionInfo>9.9</owl:versionInfo>"
        '<terms:license rdf:resource="https://lic"/></owl:Ontology>'
        + cls("PaNET00001", "photon and neutron technique")
        + cls("PaNET00003", "defined by process", "PaNET00001")
        + cls("PaNET10", "b technique", "PaNET00003",
              "<skos:altLabel>BT</skos:altLabel>")
        + cls("PaNET11", "a technique", "PaNET10")
        + cls("PaNET2000000", "specifiers")
        + cls("PaNET2000001", "photon", "PaNET2000000")
        + "</rdf:RDF>"
    )
    payload = module.build(owl.encode())
    assert payload["version"] == "9.9"
    assert payload["released"] == "2030-01-02"
    assert payload["techniques"] == [
        {"name": "a technique", "pid": BASE + "PaNET11"},
        {"name": "b technique", "pid": BASE + "PaNET10", "synonyms": ["BT"]},
    ]


# --- the dialog -------------------------------------------------------------

def _ok(dialog):
    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)


def test_dialog_lists_everything_without_preselecting(qtbot):
    dialog = TechniquePickerDialog()
    qtbot.addWidget(dialog)
    total = len(panet.load_catalogue().techniques)
    assert dialog.tree.topLevelItemCount() == total
    assert dialog.selected() is None
    assert not _ok(dialog).isEnabled()
    dialog.accept()  # nothing chosen: stays open
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_dialog_search_selects_first_match_and_shows_pid(qtbot):
    dialog = TechniquePickerDialog()
    qtbot.addWidget(dialog)
    dialog.search_edit.setText("giwaxs")
    assert dialog.tree.topLevelItemCount() == 1
    assert (dialog.selected().name, dialog.selected().pid) == GIWAXS
    assert GIWAXS[1] in dialog.detail_label.text()
    assert _ok(dialog).isEnabled()
    dialog.search_edit.setText("zzzz")
    assert dialog.selected() is None
    assert not _ok(dialog).isEnabled()
    assert "No technique" in dialog.count_label.text()


def test_dialog_keyboard_from_the_search_field(qtbot):
    dialog = TechniquePickerDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.search_edit.setText("xrd")
    assert dialog.selected().pid == XRD[1]
    qtbot.keyClick(dialog.search_edit, Qt.Key.Key_Down)
    second = dialog.selected()
    assert second.pid != XRD[1]
    qtbot.keyClick(dialog.search_edit, Qt.Key.Key_Return)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selected() == second


# --- the Conversion panel ---------------------------------------------------

@pytest.fixture
def panel(qtbot):
    widget = conversion_panel.ConversionPanel()
    qtbot.addWidget(widget)
    return widget


def _collected(panel):
    cfg = ConversionConfig()
    panel._collect_nexus_info(cfg)
    return cfg.technique_info


def _pick(monkeypatch, query):
    """Make the panel's dialog search ``query`` and accept (None: cancel)."""
    def run(dialog):
        if query is None:
            return QDialog.DialogCode.Rejected
        dialog.search_edit.setText(query)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(TechniquePickerDialog, "exec", run)


def test_select_button_adds_name_and_pid(panel, monkeypatch):
    assert panel.technique_table.rowCount() == 0
    assert panel.select_technique_btn.text() == "Select technique…"
    assert "Add technique" not in [
        b.text() for b in panel.technique_info_section.findChildren(QPushButton)
    ]
    _pick(monkeypatch, "xrr")
    panel.select_technique_btn.click()
    _pick(monkeypatch, "gisaxs")
    panel.select_technique_btn.click()
    assert _collected(panel) == [
        {"name": XRR[0], "pid": XRR[1]},
        {"name": "grazing incidence small angle x-ray scattering",
         "pid": BASE + "PaNET01162"},
    ]


def test_cancel_adds_nothing_and_repeat_is_not_added(panel, monkeypatch):
    _pick(monkeypatch, None)
    panel.select_technique_btn.click()
    assert panel.technique_table.rowCount() == 0
    _pick(monkeypatch, "xrr")
    panel.select_technique_btn.click()
    panel.select_technique_btn.click()
    assert panel.technique_table.rowCount() == 1


def test_cells_are_read_only_and_rows_removable(panel):
    table = panel.technique_table
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    panel.add_technique(*XRR)
    panel.add_technique(*GIWAXS)
    table.selectRow(0)
    panel._remove_technique_rows()
    assert _collected(panel) == [{"name": GIWAXS[0], "pid": GIWAXS[1]}]


def test_unreadable_list_is_logged(panel, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("gone")

    monkeypatch.setattr(conversion_panel, "TechniquePickerDialog", broken)
    logged = []
    monkeypatch.setattr(panel, "append_log", logged.append)
    panel.select_technique_btn.click()
    assert panel.technique_table.rowCount() == 0
    assert logged and "gone" in logged[0]
