"""Matched-structure colour + visibility are keyed to a cross-frame
identity (CIF + hkl), so they stay consistent as the user scrubs frames.

Before this, both were keyed by per-frame position: the palette was
allocated by the structure's index in the frame's (name-sorted) list, so
the same structure could change colour between frames, and the show/hide
checkbox was stored per ``(frame, unique_id)`` so unchecking it did not
persist to other frames.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidgui.file_model import MatchedStructure, PeakTable

pytestmark = pytest.mark.gui


def _structure(local_idx: int, cif: str, hkl: tuple) -> MatchedStructure:
    z = np.zeros(1, dtype=float)
    peaks = PeakTable(
        q_xy=z.copy(), q_z=z.copy(), angle=z.copy(), radius=z.copy(),
        angle_width=z.copy(), radius_width=z.copy(),
        is_ring=np.zeros(1, dtype=bool), ids=np.array([0]),
        score=z.copy(), amplitude=z.copy(),
    )
    return MatchedStructure(
        solution_field="matched_segments_0000", local_idx=local_idx,
        cif=cif, h=hkl[0], k=hkl[1], l=hkl[2],
        probability=0.9, peaks=peaks, peak_list=np.array([0]),
    )


@pytest.fixture
def viewer(qtbot):
    from mlgidgui.image_viewer import GIWAXSImageViewer
    v = GIWAXSImageViewer()
    qtbot.addWidget(v)
    return v


def _by_cif(structs, cif):
    return next(s for s in structs if s.cif == cif)


def test_matched_color_stable_across_frames_despite_order(viewer):
    """Two structures whose order flips between frames keep their colours
    (identity-keyed, not position-keyed)."""
    v = viewer
    # Frame 0: Aaa(110) first, Zzz(200) second.
    f0 = [_structure(0, "Aaa", (1, 1, 0)), _structure(1, "Zzz", (2, 0, 0))]
    # Frame 1: order reversed.
    f1 = [_structure(0, "Zzz", (2, 0, 0)), _structure(1, "Aaa", (1, 1, 0))]

    v._frame_index = 0
    v.set_matched_structures(0, f0)
    c_aaa = v.matched_color(_by_cif(f0, "Aaa"))
    c_zzz = v.matched_color(_by_cif(f0, "Zzz"))
    assert c_aaa != c_zzz

    v._frame_index = 1
    v.set_matched_structures(1, f1)
    # Same identities -> same colours, even though positions swapped.
    assert v.matched_color(_by_cif(f1, "Aaa")) == c_aaa
    assert v.matched_color(_by_cif(f1, "Zzz")) == c_zzz


def test_matched_visibility_persists_across_frames(viewer):
    """Unchecking a structure on one frame hides it on every frame it
    appears (its per-frame unique_id differs, but the identity is the
    same)."""
    v = viewer
    f0 = [_structure(0, "Aaa", (1, 1, 0)), _structure(1, "Zzz", (2, 0, 0))]
    f1 = [_structure(0, "Zzz", (2, 0, 0)), _structure(1, "Aaa", (1, 1, 0))]
    v._frame_index = 0
    v.set_matched_structures(0, f0)
    v._frame_index = 1
    v.set_matched_structures(1, f1)

    # Hide Zzz while on frame 0.
    v._frame_index = 0
    zzz_uid_f0 = _by_cif(f0, "Zzz").unique_id
    v.set_matched_structure_visible(zzz_uid_f0, False)
    assert not v.matched_visibility(0, zzz_uid_f0)

    # On frame 1 its unique_id differs, but the identity match keeps it
    # hidden; Aaa stays visible.
    zzz_uid_f1 = _by_cif(f1, "Zzz").unique_id
    assert zzz_uid_f1 != zzz_uid_f0
    assert not v.matched_visibility(1, zzz_uid_f1)
    assert v.matched_visibility(1, _by_cif(f1, "Aaa").unique_id)


def _struct_with_peaks(cif: str, rows: list) -> MatchedStructure:
    """rows: [(radius, angle, is_ring), ...]."""
    n = len(rows)
    radius = np.array([r for r, _, _ in rows], dtype=float)
    angle = np.array([a for _, a, _ in rows], dtype=float)
    is_ring = np.array([bool(rg) for _, _, rg in rows])
    aw = np.where(is_ring, np.inf, 5.0)
    z = np.zeros(n, dtype=float)
    peaks = PeakTable(
        q_xy=radius * np.cos(np.deg2rad(angle)),
        q_z=radius * np.sin(np.deg2rad(angle)),
        angle=angle, radius=radius, angle_width=aw,
        radius_width=np.full(n, 0.2), is_ring=is_ring,
        ids=np.arange(n), score=z.copy(), amplitude=z.copy(),
    )
    return MatchedStructure(
        solution_field="matched_segments_0000", local_idx=0, cif=cif,
        h=1, k=1, l=0, probability=0.9, peaks=peaks,
        peak_list=np.arange(n),
    )


def test_matched_display_style_defaults_to_boxes(viewer):
    from mlgidgui.image_viewer import _PeakShapeItem
    v = viewer
    v._frame_index = 0
    v.set_matched_structures(0, [_struct_with_peaks("PbI2", [(1.0, 30.0, False)])])
    assert v._matched_display_style == "boxes"
    items = [it for _uid, it in v._matched_items]
    assert len(items) == 1 and isinstance(items[0], _PeakShapeItem)


def test_matched_marker_style_circles_and_ring_arcs(viewer):
    import pyqtgraph as pg
    from mlgidgui.image_viewer import MODE_CARTESIAN
    v = viewer
    v._frame_index = 0
    v._mode = MODE_CARTESIAN
    # 2 spots + 1 ring.
    v.set_matched_structures(0, [_struct_with_peaks(
        "PbI2", [(1.0, 30.0, False), (2.0, 60.0, False), (3.0, 45.0, True)],
    )])
    v.set_matched_display_style("markers")
    items = [it for _uid, it in v._matched_items]
    scatters = [i for i in items if isinstance(i, pg.ScatterPlotItem)]
    curves = [i for i in items if isinstance(i, pg.PlotCurveItem)]
    assert len(scatters) == 1 and len(curves) == 1     # spots grouped, one ring
    assert len(scatters[0].getData()[0]) == 2          # two spot markers
    xd, yd = curves[0].getData()
    r = np.hypot(np.asarray(xd), np.asarray(yd))
    assert np.allclose(r, 3.0)                          # ring arc at constant |q|
    # Switching back restores boxes.
    from mlgidgui.image_viewer import _PeakShapeItem
    v.set_matched_display_style("boxes")
    assert all(
        isinstance(it, _PeakShapeItem) for _uid, it in v._matched_items
    )


def test_matched_marker_style_visibility_toggle(viewer):
    v = viewer
    v._frame_index = 0
    s = _struct_with_peaks("PbI2", [(1.0, 30.0, False), (3.0, 45.0, True)])
    v.set_matched_structures(0, [s])
    v.set_matched_display_style("markers")
    # Every marker item of the structure follows its checkbox.
    v.set_matched_structure_visible(s.unique_id, False)
    assert all(not it.isVisible() for _uid, it in v._matched_items)
    v.set_matched_structure_visible(s.unique_id, True)
    assert all(it.isVisible() for _uid, it in v._matched_items)


def test_matched_marker_ring_shares_structure_colour(viewer):
    """In marker mode a ring arc and the peak circles share the
    structure's colour; the SHAPE (dashed arc vs small circles), not a
    second colour, tells them apart, so one distinguishable colour per
    structure is enough."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from mlgidgui.image_viewer import MODE_CARTESIAN
    v = viewer
    v._frame_index = 0
    v._mode = MODE_CARTESIAN
    v.set_matched_structures(0, [_struct_with_peaks(
        "PbI2", [(1.0, 30.0, False), (3.0, 45.0, True)],
    )])
    v.set_matched_display_style("markers")
    items = [it for _uid, it in v._matched_items]
    scat = next(i for i in items if isinstance(i, pg.ScatterPlotItem))
    ring = next(i for i in items if isinstance(i, pg.PlotCurveItem))
    peak_c = pg.mkPen(scat.opts["pen"]).color().name()
    ring_c = ring.opts["pen"].color().name()
    assert peak_c == ring_c                                   # one colour per structure
    assert ring.opts["pen"].style() == Qt.PenStyle.DashLine   # ring distinguished by dash


def _fitted_table(rows: list) -> PeakTable:
    """rows: [(id, radius, angle, is_ring), ...] -> fitted PeakTable."""
    n = len(rows)
    radius = np.array([r[1] for r in rows], dtype=float)
    angle = np.array([r[2] for r in rows], dtype=float)
    is_ring = np.array([bool(r[3]) for r in rows])
    z = np.zeros(n, dtype=float)
    return PeakTable(
        q_xy=radius * np.cos(np.deg2rad(angle)),
        q_z=radius * np.sin(np.deg2rad(angle)),
        angle=angle, radius=radius,
        angle_width=np.where(is_ring, np.inf, 5.0),
        radius_width=np.full(n, 0.2), is_ring=is_ring,
        ids=np.array([r[0] for r in rows], dtype=int),
        score=z.copy(), amplitude=z.copy() + 10.0,
    )


def test_unmatched_fitted_pseudo_group(viewer):
    """Fitted rows no structure claims render as a grey pseudo-group
    through the matched machinery: hidden by default, own checkbox,
    marker style, master toggle, and the tracked-peak filter."""
    import pyqtgraph as pg
    from mlgidgui.image_viewer import (
        MODE_CARTESIAN, UNMATCHED_COLOR, UNMATCHED_UID, _PeakShapeItem,
    )
    v = viewer
    v._frame_index = 0
    v._mode = MODE_CARTESIAN
    # Fitted ids 0/1/2 (2 spots + 1 ring); a structure claims id 0 only.
    v.set_peaks(0, {
        "detected": None,
        "fitted": _fitted_table(
            [(0, 1.0, 30.0, False), (1, 2.0, 60.0, False), (2, 3.0, 45.0, True)]
        ),
    })
    v.set_matched_structures(0, [_struct_with_peaks("PbI2", [(1.0, 30.0, False)])])

    def un_items():
        return [it for uid, it in v._matched_items if uid == UNMATCHED_UID]

    # Boxes mode: one grey _PeakShapeItem, hidden by default.
    assert v.has_unmatched_fitted(0)
    items = un_items()
    assert len(items) == 1 and isinstance(items[0], _PeakShapeItem)
    assert not items[0].isVisible()
    assert v.unmatched_visible() is False
    v.set_unmatched_visible(True)
    assert un_items()[0].isVisible()

    # Markers mode: unmatched ids 1 (spot) + 2 (ring) -> one scatter
    # with a single circle + one grey dashed ring arc at |q|=3.
    v.set_matched_display_style("markers")
    scats = [i for i in un_items() if isinstance(i, pg.ScatterPlotItem)]
    curves = [i for i in un_items() if isinstance(i, pg.PlotCurveItem)]
    assert len(scats) == 1 and len(curves) == 1
    assert len(scats[0].getData()[0]) == 1
    assert pg.mkPen(scats[0].opts["pen"]).color().name() == UNMATCHED_COLOR
    assert np.allclose(
        np.hypot(*[np.asarray(d) for d in curves[0].getData()]), 3.0
    )
    assert all(i.isVisible() for i in un_items())

    # Matched master toggle reaches the pseudo-group.
    v.set_matched_master_visible(False)
    assert all(not i.isVisible() for i in un_items())
    v.set_matched_master_visible(True)

    # Tracked-peak filter subsets it: only fitted id 1 whitelisted ->
    # the ring (id 2) drops out of the unmatched subset.
    v.set_fitted_visible_only({0: {1}})
    scats = [i for i in un_items() if isinstance(i, pg.ScatterPlotItem)]
    curves = [i for i in un_items() if isinstance(i, pg.PlotCurveItem)]
    assert len(scats) == 1 and len(scats[0].getData()[0]) == 1
    assert curves == []
    # Whitelist without any unmatched row -> the group disappears.
    v.set_fitted_visible_only({0: {0}})
    assert not v.has_unmatched_fitted(0)
    assert un_items() == []


def test_unmatched_group_renders_without_structures(viewer):
    """With NO matched structures at all, every fitted row is unmatched
    and the pseudo-group still renders (helps tracking visualization)."""
    from mlgidgui.image_viewer import UNMATCHED_UID
    v = viewer
    v._frame_index = 0
    v.set_peaks(0, {
        "detected": None,
        "fitted": _fitted_table([(0, 1.0, 30.0, False)]),
    })
    v.set_matched_structures(0, [])
    v.set_unmatched_visible(True)
    items = [it for uid, it in v._matched_items if uid == UNMATCHED_UID]
    assert len(items) == 1 and items[0].isVisible()


def test_matched_color_and_visibility_reset_on_clear(viewer):
    v = viewer
    f0 = [_structure(0, "Aaa", (1, 1, 0))]
    v._frame_index = 0
    v.set_matched_structures(0, f0)
    v.set_matched_structure_visible(_by_cif(f0, "Aaa").unique_id, False)
    assert v._matched_color_index and v._matched_visibility
    v.clear()
    assert v._matched_color_index == {} and v._matched_visibility == {}


def test_custom_matched_color_overrides_palette(
    viewer, clean_matched_colors,
):
    """A picked colour replaces the palette colour for that identity
    only; line style/width and every other structure stay automatic,
    and ``None`` returns to the palette."""
    v = viewer
    f0 = [_structure(0, "Aaa", (1, 1, 0)), _structure(1, "Zzz", (2, 0, 0))]
    v._frame_index = 0
    v.set_matched_structures(0, f0)
    aaa, zzz = _by_cif(f0, "Aaa"), _by_cif(f0, "Zzz")
    pen_before = dict(v.matched_pen(aaa))
    zzz_before = dict(v.matched_pen(zzz))

    v.set_matched_color(aaa.color_key, "#123456")
    pen = v.matched_pen(aaa)
    assert pen["color"] == "#123456"
    assert pen["style"] == pen_before["style"]
    assert pen["width"] == pen_before["width"]
    assert v.matched_pen(zzz) == zzz_before

    v.set_matched_color(aaa.color_key, None)
    assert v.matched_pen(aaa) == pen_before


def test_custom_matched_color_reemits_structures(
    viewer, clean_matched_colors,
):
    """``set_matched_color`` re-emits ``matchedStructuresChanged`` for
    the current frame so both legends rebuild their swatches."""
    v = viewer
    f0 = [_structure(0, "Aaa", (1, 1, 0))]
    v._frame_index = 0
    v.set_matched_structures(0, f0)
    got = []
    v.matchedStructuresChanged.connect(lambda f, s: got.append((f, s)))
    v.set_matched_color(("Aaa", 1, 1, 0), "#123456")
    assert len(got) == 1
    frame, structs = got[0]
    assert frame == 0 and [s.cif for s in structs] == ["Aaa"]


def test_custom_matched_color_survives_clear_and_restart(
    viewer, qtbot, clean_matched_colors,
):
    """Overrides are preferences, not file state: they outlive
    ``clear()`` and, via QSettings, a fresh viewer instance (the
    restart stand-in)."""
    from mlgidgui.image_viewer import GIWAXSImageViewer
    v = viewer
    key = ("Aaa", 1, 1, 0)
    v.set_matched_color(key, "#123456")
    v.clear()
    assert v._pen_for_key(key)["color"] == "#123456"

    v2 = GIWAXSImageViewer()
    qtbot.addWidget(v2)
    assert v2._pen_for_key(key)["color"] == "#123456"
    # Resetting in the new instance clears the stored value too.
    v2.set_matched_color(key, None)
    v3 = GIWAXSImageViewer()
    qtbot.addWidget(v3)
    assert v3._matched_pen_overrides == {}


def test_a_matched_pen_can_set_style_and_width_too(
    viewer, clean_matched_colors,
):
    """Colour was the only editable property; now all three are."""
    from PySide6.QtCore import Qt

    v = viewer
    key = ("Aaa", 1, 1, 0)
    auto = dict(v._pen_for_key(key))

    v.set_matched_pen(key, {"style": Qt.PenStyle.DotLine, "width": 3.0})
    pen = v._pen_for_key(key)
    assert pen["style"] == Qt.PenStyle.DotLine
    assert pen["width"] == pytest.approx(3.0)
    # Untouched properties stay palette-driven, which is what keeps a
    # widened structure re-cycling to a distinguishable hue.
    assert pen["color"] == auto["color"]

    v.set_matched_pen(key, None)
    assert v._pen_for_key(key) == auto


def test_a_width_only_override_is_not_a_colour_choice(
    viewer, clean_matched_colors,
):
    """The phase views key off colour; a width must not claim a CIF."""
    v = viewer
    v.set_matched_pen(("Aaa", 1, 1, 0), {"width": 3.0})
    assert v.cif_color_overrides() == {}
    v.set_matched_pen(("Aaa", 1, 1, 0), {"width": 3.0, "color": "#123456"})
    assert v.cif_color_overrides() == {"Aaa": "#123456"}


def test_a_matched_pen_survives_a_restart(viewer, qtbot, clean_matched_colors):
    from PySide6.QtCore import Qt
    from mlgidgui.image_viewer import GIWAXSImageViewer

    key = ("Aaa", 1, 1, 0)
    viewer.set_matched_pen(
        key, {"color": "#123456", "style": Qt.PenStyle.DashDotLine,
              "width": 2.5},
    )
    fresh = GIWAXSImageViewer()
    qtbot.addWidget(fresh)
    pen = fresh._pen_for_key(key)
    assert pen["color"] == "#123456"
    assert pen["style"] == Qt.PenStyle.DashDotLine
    assert pen["width"] == pytest.approx(2.5)


def test_colours_stored_before_pens_are_still_honoured(
    viewer, qtbot, clean_matched_colors,
):
    """An upgrading user keeps the colours they picked.

    The pre-pen key held ``[[key, "#hex"], ...]``; it is read as a
    colour-only pen when the new key is absent.
    """
    import json

    from PySide6.QtCore import QSettings
    from mlgidgui.image_viewer import GIWAXSImageViewer

    QSettings().setValue(
        "matchedColors", json.dumps([[["Aaa", 1, 1, 0], "#abcdef"]]),
    )
    fresh = GIWAXSImageViewer()
    qtbot.addWidget(fresh)
    assert fresh._pen_for_key(("Aaa", 1, 1, 0))["color"] == "#abcdef"
    # ...and only the colour: style and width stay automatic.
    assert fresh.matched_pen_override(("Aaa", 1, 1, 0)) == {"color": "#abcdef"}


def test_the_detected_and_fitted_overlays_take_a_pen(
    viewer, clean_matched_colors,
):
    """The peaks overlays are customisable now, and default to the preset."""
    from PySide6.QtCore import Qt
    from mlgidgui.viewer_styles import OVERLAY_STYLE

    v = viewer
    assert v.overlay_pen("detected") == OVERLAY_STYLE["detected"]
    assert v.overlay_pen_override("detected") == {}

    seen: list[str] = []
    v.overlayPenChanged.connect(seen.append)
    v.set_overlay_pen(
        "fitted", {"color": "#ff00ff", "style": Qt.PenStyle.DotLine,
                   "width": 2.5},
    )
    assert seen == ["fitted"]
    pen = v.overlay_pen("fitted")
    assert pen["color"] == "#ff00ff"
    assert pen["width"] == pytest.approx(2.5)
    # The live item follows, not just the bookkeeping.
    assert v._fitted._pen.color().name() == "#ff00ff"
    assert v._fitted._pen.style() == Qt.PenStyle.DotLine
    # Detected is untouched by a change to fitted.
    assert v.overlay_pen("detected") == OVERLAY_STYLE["detected"]

    v.set_overlay_pen("fitted", None)
    assert v.overlay_pen("fitted") == OVERLAY_STYLE["fitted"]
    assert v._fitted._pen.color().name() == OVERLAY_STYLE["fitted"]["color"]
    assert seen == ["fitted", "fitted"]


def test_an_overlay_pen_survives_a_restart(viewer, qtbot, clean_matched_colors):
    from mlgidgui.image_viewer import GIWAXSImageViewer

    viewer.set_overlay_pen("detected", {"color": "#00ff00", "width": 3.0})
    fresh = GIWAXSImageViewer()
    qtbot.addWidget(fresh)
    assert fresh.overlay_pen("detected")["color"] == "#00ff00"
    # Built from the effective pen, so it is right from the FIRST render
    # rather than flicking to it after the first change.
    assert fresh._detected._pen.color().name() == "#00ff00"
    assert fresh._detected._pen.widthF() == pytest.approx(3.0)


def test_an_unknown_overlay_kind_is_refused(viewer, clean_matched_colors):
    with pytest.raises(ValueError, match="Unknown overlay kind"):
        viewer.set_overlay_pen("nonsense", {"color": "#ffffff"})


def test_cif_color_overrides_smallest_hkl_wins(
    viewer, clean_matched_colors,
):
    """The CIF-level view (for the phase views) is deterministic when
    several hkl rows of one CIF are overridden."""
    v = viewer
    v.set_matched_color(("Aaa", 2, 0, 0), "#222222")
    v.set_matched_color(("Aaa", 1, 1, 0), "#111111")
    v.set_matched_color(("Bbb", 0, 0, 1), "#333333")
    assert v.cif_color_overrides() == {"Aaa": "#111111", "Bbb": "#333333"}
