#!/usr/bin/env python3
"""Rebuild the bundled PaNET technique list from the published ontology.

The Conversion panel's technique picker reads
``mlgidgui/assets/panet/techniques.json``; the GUI never goes online for
it. Run this before a release when PaNET has published a new version::

    python tools/update_panet.py                    # downloads the ontology
    python tools/update_panet.py path/to/ontology.owl

Only technique classes are kept: everything below "photon and neutron
technique" (PaNET00001) or "defined by experimental probe" (PaNET00002),
without the six abstract grouping nodes at the top. The "photon and
neutron specifiers" branch (probe, process, purpose, ...) holds the
building blocks PaNET defines techniques with, not techniques.

PaNET is published under CC BY 4.0; the attribution is stored in the
JSON and shown in the picker.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_URL = "https://pan-ontologies.github.io/PaNET/ontology.owl"
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "mlgidgui" / "assets" / "panet" / "techniques.json"
)
BASE = "http://purl.org/pan-science/PaNET/"
# Where the technique branches start ...
ROOTS = ("PaNET00001", "PaNET00002")
# ... and the nodes that only group them ("photon and neutron technique"
# and the five "defined by ..." headings).
GROUPING = tuple(f"PaNET0000{n}" for n in range(1, 7))

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "obo": "http://purl.obolibrary.org/obo/",
    "terms": "http://purl.org/dc/terms/",
}
_ABOUT = "{%s}about" % NS["rdf"]
_RESOURCE = "{%s}resource" % NS["rdf"]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def build(owl_bytes: bytes) -> dict:
    """The JSON payload for one RDF/XML serialization of PaNET."""
    root = ET.fromstring(owl_bytes)
    classes = {
        el.get(_ABOUT): el
        for el in root.findall("owl:Class", NS)
        if (el.get(_ABOUT) or "").startswith(BASE)
    }
    children: dict[str, list[str]] = {}
    for iri, el in classes.items():
        for parent in el.findall("rdfs:subClassOf", NS):
            if parent.get(_RESOURCE):
                children.setdefault(parent.get(_RESOURCE), []).append(iri)

    wanted: set[str] = set()
    stack = [BASE + r for r in ROOTS]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in wanted:
                wanted.add(child)
                stack.append(child)
    wanted -= {BASE + g for g in GROUPING}

    techniques = []
    for iri in wanted:
        el = classes[iri]
        entry = {
            "name": _clean(el.findtext("rdfs:label", None, NS)),
            "pid": iri,
        }
        if not entry["name"]:
            raise ValueError(f"{iri} has no label")
        synonyms = sorted(
            {_clean(s.text) for s in el.findall("skos:altLabel", NS)} - {""},
            key=str.lower,
        )
        if synonyms:
            entry["synonyms"] = synonyms
        definition = _clean(el.findtext("obo:IAO_0000115", None, NS))
        if definition:
            entry["definition"] = definition
        techniques.append(entry)
    techniques.sort(key=lambda t: (t["name"].lower(), t["pid"]))

    header = root.find("owl:Ontology", NS)
    license_el = header.find("terms:license", NS)
    return {
        "title": _clean(header.findtext("terms:title", None, NS)),
        "version": _clean(header.findtext("owl:versionInfo", None, NS)),
        "released": _clean(header.findtext("terms:created", None, NS)),
        "license": license_el.get(_RESOURCE) if license_el is not None else "",
        "source": SOURCE_URL,
        "techniques": techniques,
    }


def main(argv: list[str]) -> int:
    if argv:
        data = Path(argv[0]).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as reply:
            data = reply.read()
    payload = build(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{OUTPUT}: {len(payload['techniques'])} techniques, PaNET "
        f"{payload['version']} ({payload['released']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
