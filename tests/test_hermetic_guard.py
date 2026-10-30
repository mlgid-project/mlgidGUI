"""Locks in the backend-free claim for the pure-logic path.

Importing ``file_model`` / ``fit`` must not drag in the heavy analysis
stack — those are imported lazily only by ``pipeline.py`` when a
pipeline command actually runs (and by ``manual_fit.py`` when the
"Add to fitted" path's 2D fit is invoked).

The assertion is order-independent: it snapshots ``sys.modules``
before the imports and confirms the two specific imports add no
heavy backend modules. This survives running the test suite in any
order (earlier tests may have already loaded pygid via
``test_manual_fit.py`` or ``test_energy_guard.py``).
"""

from __future__ import annotations

import sys


_HEAVY_BACKENDS = {"mlgidbase", "pygid", "pygidfit", "pygidsim", "torch"}


def test_pure_path_has_no_heavy_backend():
    before = set(sys.modules)
    import mlgidgui.file_model  # noqa: F401
    import mlgidgui.fit  # noqa: F401
    newly_loaded = set(sys.modules) - before
    heavy_added = _HEAVY_BACKENDS & newly_loaded
    assert not heavy_added, (
        f"Importing file_model + fit pulled in heavy backend modules "
        f"{heavy_added!r} — both modules must stay backend-free so the "
        f"pure-logic path runs on environments without the private "
        f"upstream stack installed."
    )


def test_structure_editor_primitives_have_no_heavy_backend():
    """The Structure tab's primitives must run on a GUI-only install too.

    ``h5_edit`` imports ``file_model`` for the NeXus layout constants and
    nothing else; ``nexus_schema`` is pure data. Neither may reach the
    private analysis stack, or editing a file would stop working on
    exactly the installs that most need a plain HDF5 editor.
    """
    before = set(sys.modules)
    import mlgidgui.h5_edit  # noqa: F401
    import mlgidgui.nexus_schema  # noqa: F401
    heavy_added = _HEAVY_BACKENDS & (set(sys.modules) - before)
    assert not heavy_added, (
        f"Importing h5_edit + nexus_schema pulled in {heavy_added!r}."
    )


def test_the_structure_tree_has_no_heavy_backend_and_no_h5py():
    """The tab's tree is a view over strings, and must stay one.

    No analysis stack, for the same reason as the primitives above — and
    no h5py either, which is the design rather than an accident: a tree
    that could open a file would sooner or later hold one open, and an
    open read handle is exactly what stops the editor taking its ``r+``.
    """
    before = set(sys.modules)
    import mlgidgui.structure_tree  # noqa: F401
    newly_loaded = set(sys.modules) - before
    assert not (_HEAVY_BACKENDS & newly_loaded), (
        f"structure_tree pulled in {_HEAVY_BACKENDS & newly_loaded!r}."
    )
    assert "h5py" not in newly_loaded, (
        "structure_tree imported h5py — it reads through the lister "
        "MainWindow installs, and must not be able to open a file itself."
    )
