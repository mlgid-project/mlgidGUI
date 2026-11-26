"""The bundled PaNET technique list and its search.

PaNET (Photon and Neutron Experimental Techniques ontology) fixes the
name and persistent identifier of each technique; the PIDs have the form
``http://purl.org/pan-science/PaNET/PaNET01316``, the form pygid writes
for the techniques it records itself. The list ships as package data
(``assets/panet/techniques.json``, rebuilt by ``tools/update_panet.py``)
so the picker works without network access. Qt-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_ASSET = "assets/panet/techniques.json"


@dataclass(frozen=True)
class Technique:
    name: str
    pid: str
    synonyms: tuple[str, ...] = ()
    definition: str = ""

    @property
    def short_id(self) -> str:
        """The last part of the PID, e.g. ``PaNET01316``."""
        return self.pid.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Catalogue:
    version: str
    released: str
    license: str
    techniques: tuple[Technique, ...]


@lru_cache(maxsize=1)
def load_catalogue() -> Catalogue:
    """Read the bundled list. Raises ``OSError`` / ``ValueError`` when
    the file is missing or damaged (the caller reports it)."""
    text = resources.files("mlgidgui").joinpath(_ASSET).read_text(
        encoding="utf-8"
    )
    data = json.loads(text)
    return Catalogue(
        version=data["version"],
        released=data["released"],
        license=data["license"],
        techniques=tuple(
            Technique(
                name=t["name"],
                pid=t["pid"],
                synonyms=tuple(t.get("synonyms", ())),
                definition=t.get("definition", ""),
            )
            for t in data["techniques"]
        ),
    )


def _joined(text: str) -> str:
    """Lower case without hyphens: ``X-ray`` -> ``xray``."""
    return " ".join(text.lower().replace("-", "").split())


def _spaced(text: str) -> str:
    """Lower case with hyphens as spaces: ``X-ray`` -> ``x ray``."""
    return " ".join(text.lower().replace("-", " ").split())


def _rank(technique: Technique, query: str, tokens: list[str]) -> int | None:
    """How well ``technique`` matches, lower is better; None for no match."""
    lowered = query.strip().lower()
    if lowered in (technique.pid.lower(), technique.short_id.lower()):
        return 0
    labels = (technique.name, *technique.synonyms)
    joined = [_joined(label) for label in labels]
    spaced = [_spaced(label) for label in labels]
    haystack = " | ".join(joined + spaced)
    if not all(token in haystack for token in tokens):
        # The number of an identifier on its own, e.g. "01316".
        digits = lowered.isdigit() and len(lowered) >= 4
        return 4 if digits and lowered in technique.short_id else None
    wanted = _joined(query)
    if wanted in joined:
        return 0
    if any(label.startswith(wanted) for label in joined):
        return 1
    words = {word for label in joined + spaced for word in label.split()}
    if all(any(word.startswith(token) for word in words) for token in tokens):
        return 2
    return 3


def search(
    techniques: tuple[Technique, ...] | list[Technique], query: str
) -> list[Technique]:
    """The techniques matching ``query``, best first.

    Every word of the query must occur in the name or a synonym (case and
    hyphens ignored); a PID or its ``PaNET…`` part also matches. Exact
    names and synonyms come first, then labels starting with the query,
    then word starts. Within one rank the specific techniques come before
    the "…-based techniques" groups, shorter names first. An empty query
    returns everything in the list's (alphabetical) order.
    """
    tokens = _spaced(query).split()
    if not tokens:
        return list(techniques)
    hits = []
    for technique in techniques:
        rank = _rank(technique, query, tokens)
        if rank is not None:
            hits.append((
                rank,
                technique.name.endswith("-based techniques"),
                len(technique.name),
                technique.name,
                technique,
            ))
    hits.sort(key=lambda hit: hit[:4])
    return [hit[4] for hit in hits]
