"""The HDF5 editing primitives, and the HDF5 behaviours they rest on.

No Qt here on purpose: everything ``mlgidgui.h5_edit`` does is plain
h5py, so it is testable without a QApplication and runs in the
backend-less CI job unchanged.

The first class of tests is unusual and deliberate. ``EditHandle`` keeps
a ``r+`` handle open while silx keeps ``r`` handles on the same file,
which is only safe because of four specific HDF5 behaviours. Those are
pinned here so an h5py or HDF5 upgrade that changes one fails this file
loudly, instead of corrupting an editing session in the field.
"""
from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from mlgidgui import h5_edit
from mlgidgui.h5_edit import EditError, EditHandle


@pytest.fixture
def simple_h5(tmp_path) -> Path:
    """A small file with a group, a few datasets, and both link kinds."""
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as f:
        f.create_dataset("far", data=np.arange(4, dtype="i4"))

    path = tmp_path / "simple.h5"
    with h5py.File(path, "w", track_order=True) as f:
        meta = f.create_group("meta", track_order=True)
        meta.attrs["NX_class"] = "NXcollection"
        meta.attrs["comment"] = "written at the beamline"
        meta.create_dataset("counts", data=np.arange(6, dtype="i4"))
        meta.create_dataset("name", data="sample A")
        meta.create_dataset("temperature", data=np.float32(297.5))
        table = np.zeros(3, dtype=np.dtype([("a", "f4"), ("b", "i4")]))
        table["a"] = [1.0, 2.0, 3.0]
        table["b"] = [10, 20, 30]
        meta.create_dataset("table", data=table)
        f["soft"] = h5py.SoftLink("/meta/counts")
        f["ext"] = h5py.ExternalLink(str(external), "/far")
        f["broken"] = h5py.SoftLink("/nowhere")
    return path


# -- the four HDF5 behaviours EditHandle depends on ------------------------


def test_r_plus_fails_while_a_read_handle_is_open(simple_h5):
    """Behaviour 1: this is why the handle is acquired while detached."""
    reader = h5py.File(simple_h5, "r")
    try:
        with pytest.raises(OSError):
            h5py.File(simple_h5, "r+")
    finally:
        reader.close()


def test_read_handle_opens_after_a_write_handle_and_sees_its_writes(simple_h5):
    """Behaviours 2 and 3: silx can reattach over an open edit handle."""
    writer = h5py.File(simple_h5, "r+")
    try:
        writer.create_group("added_before_reader")
        writer.flush()
        reader = h5py.File(simple_h5, "r")
        try:
            assert "added_before_reader" in reader
            writer.create_group("added_after_reader")
            writer["meta"].attrs["comment"] = "edited"
            writer.flush()
            # The two handles share one underlying file object, so the
            # reader sees structure and attribute changes with no reopen.
            assert "added_after_reader" in reader
            assert reader["meta"].attrs["comment"] == "edited"
        finally:
            reader.close()
    finally:
        writer.close()


def test_a_second_write_handle_coexists(simple_h5):
    """Behaviour 4: a peak write elsewhere cannot break an open editor."""
    editor = h5py.File(simple_h5, "r+")
    try:
        with h5py.File(simple_h5, "r+") as other:
            other["meta/counts"][0] = 99
        assert editor.id.valid
        assert editor["meta/counts"][0] == 99
    finally:
        editor.close()


# -- EditHandle ------------------------------------------------------------


def test_edit_handle_acquire_release_is_idempotent(simple_h5):
    handle = EditHandle(simple_h5)
    assert not handle.is_open
    first = handle.acquire()
    assert handle.acquire() is first, "re-acquiring must reuse the handle"
    handle.flush()
    handle.release()
    handle.release()
    assert not handle.is_open


def test_edit_handle_file_raises_when_closed(simple_h5):
    handle = EditHandle(simple_h5)
    with pytest.raises(EditError):
        handle.file


def test_edit_handle_relocate_drops_the_old_handle(simple_h5, tmp_path):
    """Save As renames the temp file; the old handle must not survive."""
    handle = EditHandle(simple_h5)
    handle.acquire()
    moved = tmp_path / "renamed.h5"
    simple_h5.replace(moved)
    handle.relocate(moved)
    assert not handle.is_open
    assert handle.acquire()["meta"] is not None
    handle.release()


def test_edit_handle_reports_a_readable_error_for_a_missing_file(tmp_path):
    with pytest.raises(EditError, match="Could not open"):
        EditHandle(tmp_path / "nope.h5").acquire()


# -- paths and protection --------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("a/b", "/a/b"), ("/a/b/", "/a/b"), ("/", "/"), ("", "/"), ("///", "/")],
)
def test_normalize_path(raw, expected):
    assert h5_edit.normalize_path(raw) == expected


def test_split_and_join_round_trip():
    parent, name = h5_edit.split_path("/entry_0000/data/q_xy")
    assert (parent, name) == ("/entry_0000/data", "q_xy")
    assert h5_edit.join_path(parent, name) == "/entry_0000/data/q_xy"
    assert h5_edit.join_path("/", "entry_0000") == "/entry_0000"


def test_split_path_refuses_the_root():
    with pytest.raises(EditError):
        h5_edit.split_path("/")


@pytest.mark.parametrize(
    "path",
    [
        "/entry_0000",
        "/entry_0000/data",
        "/entry_0000/data/img_gid_q",
        "/entry_0000/data/q_xy",
        "/entry_0000/data/q_z",
        "/entry_0000/data/analysis",
        "/entry_0000/data/analysis/frame00000/detected_peaks",
        "/entry/data/angle_of_incidence",
    ],
)
def test_protected_paths_are_flagged(path):
    assert h5_edit.protection_reason(path)


@pytest.mark.parametrize(
    "path",
    ["/", "/meta", "/meta/counts", "/entry_0000/sample/name", "/entry_0000/title"],
)
def test_free_paths_are_not_flagged(path):
    assert h5_edit.protection_reason(path) is None


def test_signal_attribute_is_protected_only_on_an_entry_data_group():
    assert h5_edit.attr_protection_reason("/entry_0000/data", "signal")
    assert h5_edit.attr_protection_reason("/entry_0000/data", "units") is None
    assert h5_edit.attr_protection_reason("/meta", "signal") is None


# -- reading ---------------------------------------------------------------


def test_node_info_describes_a_dataset(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/meta/counts")
    assert info.kind == "dataset"
    assert info.shape == (6,)
    assert info.dtype.startswith("int")
    assert info.n_cells == 6
    assert not info.is_compound


def test_node_info_describes_a_group_with_its_class(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/meta")
    assert info.kind == "group"
    assert info.nx_class == "NXcollection"
    assert info.n_children == 4
    assert info.n_attrs == 2


def test_node_info_lists_compound_fields(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/meta/table")
    assert info.is_compound
    assert info.fields == ("a", "b")


def test_node_info_does_not_open_an_external_link(simple_h5):
    """The rule that keeps a 123 GB master browsable."""
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/ext")
        assert info.kind == "link"
        assert info.link.kind == "external"
        assert info.link.target == "/far"
        followed = h5_edit.node_info(f, "/ext", follow=True)
    assert followed.kind == "dataset"
    assert followed.shape == (4,)


def test_node_info_follows_a_soft_link(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/soft")
    assert info.kind == "dataset"
    assert info.link.kind == "soft"
    assert info.link.target == "/meta/counts"


def test_node_info_reports_a_broken_link_rather_than_raising(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        info = h5_edit.node_info(f, "/broken")
    assert info.kind == "link"
    assert info.link.kind == "soft"


def test_node_info_reports_a_missing_path(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit.node_info(f, "/not/here").kind == "missing"


def test_read_attrs_and_format_value(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        attrs = h5_edit.read_attrs(f, "/meta")
    assert h5_edit.format_value(attrs["comment"]) == "written at the beamline"
    assert h5_edit.format_value(np.arange(3)) == "[0, 1, 2]"
    assert h5_edit.format_value("x" * 200).endswith("…")


# -- attributes ------------------------------------------------------------


def test_set_attr_returns_the_previous_value(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        old = h5_edit.set_attr(f, "/meta", "comment", "new text")
        assert old == "written at the beamline"
        assert h5_edit.set_attr(f, "/meta", "brand_new", 3) is h5_edit.MISSING
        assert f["meta"].attrs["comment"] == "new text"


def test_delete_and_rename_attr(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        value = h5_edit.delete_attr(f, "/meta", "comment")
        assert value == "written at the beamline"
        assert "comment" not in f["meta"].attrs
        h5_edit.rename_attr(f, "/meta", "NX_class", "NX_class_old")
        assert f["meta"].attrs["NX_class_old"] == "NXcollection"
        with pytest.raises(EditError):
            h5_edit.delete_attr(f, "/meta", "gone")


def test_rename_attr_refuses_to_clobber(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="already has"):
            h5_edit.rename_attr(f, "/meta", "comment", "NX_class")


@pytest.mark.parametrize(
    "text,dtype,expected",
    [("2.5", "float32", 2.5), ("7", "int32", 7), ("true", "bool", True),
     ("off", "bool", False), (" hi ", "str", "hi")],
)
def test_parse_scalar(text, dtype, expected):
    assert h5_edit.parse_scalar(text, dtype) == expected


def test_parse_scalar_reports_what_it_wanted():
    with pytest.raises(EditError, match="float32"):
        h5_edit.parse_scalar("not a number", "float32")
    with pytest.raises(EditError, match="true or false"):
        h5_edit.parse_scalar("maybe", "bool")


# -- structure -------------------------------------------------------------


def test_create_group_stamps_the_class(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        path = h5_edit.create_group(f, "/", "sample", nx_class="NXsample")
        assert path == "/sample"
        assert f["sample"].attrs["NX_class"] == "NXsample"


def test_create_group_refuses_a_duplicate_or_a_bad_name(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="already exists"):
            h5_edit.create_group(f, "/", "meta")
        with pytest.raises(EditError, match="cannot be empty"):
            h5_edit.create_group(f, "/", "a/b")


def test_create_dataset_shapes_dtypes_and_fill(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.create_dataset(
            f, "/meta", "grid", dtype="float32", shape=(2, 3), fill=1.5,
            attrs={"units": "1/Angstrom"},
        )
        assert f["meta/grid"].shape == (2, 3)
        assert f["meta/grid"][0, 0] == pytest.approx(1.5)
        assert f["meta/grid"].attrs["units"] == "1/Angstrom"

        h5_edit.create_dataset(f, "/meta", "note", dtype="str", data="hello")
        assert h5_edit._as_text(f["meta/note"][()]) == "hello"

        h5_edit.create_dataset(f, "/meta", "scalar", dtype="float64")
        assert f["meta/scalar"].shape == ()


def test_create_dataset_on_a_dataset_parent_is_refused(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="dataset, not a group"):
            h5_edit.create_dataset(f, "/meta/counts", "child")


def test_delete_rename_and_move(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.delete_node(f, "/meta/name")
        assert "name" not in f["meta"]

        new_path = h5_edit.rename_node(f, "/meta/counts", "counts_raw")
        assert new_path == "/meta/counts_raw"
        assert "counts_raw" in f["meta"]

        h5_edit.create_group(f, "/", "archive")
        moved = h5_edit.move_node(f, "/meta/counts_raw", "/archive")
        assert moved == "/archive/counts_raw"
        assert "counts_raw" not in f["meta"]


def test_delete_refuses_the_root_and_a_missing_node(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="root"):
            h5_edit.delete_node(f, "/")
        with pytest.raises(EditError, match="does not exist"):
            h5_edit.delete_node(f, "/meta/absent")


def test_move_refuses_moving_a_group_into_itself(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="inside itself"):
            h5_edit.move_node(f, "/meta", "/meta")


# -- links -----------------------------------------------------------------


def test_create_each_link_kind(simple_h5, tmp_path):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.create_link(f, "/", "soft2", "soft", "/meta/counts")
        assert f.get("soft2", getlink=True).path == "/meta/counts"

        h5_edit.create_link(f, "/", "hard2", "hard", "/meta/counts")
        assert isinstance(f.get("hard2", getlink=True), h5py.HardLink)

        h5_edit.create_link(
            f, "/", "ext2", "external", "/far", filename=str(tmp_path / "external.h5")
        )
        link = f.get("ext2", getlink=True)
        assert isinstance(link, h5py.ExternalLink)


def test_external_link_without_a_file_is_refused(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="needs a file"):
            h5_edit.create_link(f, "/", "bad", "external", "/far")


def test_retarget_a_soft_link_and_refuse_a_hard_one(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.retarget_link(f, "/soft", "soft", "/meta/temperature")
        assert f.get("soft", getlink=True).path == "/meta/temperature"
        h5_edit.create_link(f, "/", "hard2", "hard", "/meta/counts")
        with pytest.raises(EditError, match="hard link"):
            h5_edit.retarget_link(f, "/hard2", "soft", "/meta/counts")


# -- copying ---------------------------------------------------------------


def test_copy_within_one_file_brings_attributes(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        path = h5_edit.copy_node(f, "/", "meta_copy", f, "/meta")
        assert path == "/meta_copy"
        assert f["meta_copy"].attrs["NX_class"] == "NXcollection"
        assert list(f["meta_copy"]) == list(f["meta"])


def test_copy_between_two_open_files(simple_h5, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("target")
    with h5py.File(other, "r+") as dst, h5py.File(simple_h5, "r") as src:
        h5_edit.copy_node(dst, "/target", "meta", src, "/meta")
        assert dst["target/meta/counts"][2] == 2
        assert dst["target/meta"].attrs["NX_class"] == "NXcollection"


def test_copying_a_link_copies_the_link_not_its_target(simple_h5, tmp_path):
    """h5py's own copy dereferences here; ours must not, or pasting a
    link to a 4 GB scan would paste 4 GB."""
    other = tmp_path / "other.h5"
    with h5py.File(other, "w"):
        pass
    with h5py.File(other, "r+") as dst, h5py.File(simple_h5, "r") as src:
        h5_edit.copy_node(dst, "/", "ext_copy", src, "/ext")
        h5_edit.copy_node(dst, "/", "soft_copy", src, "/soft")
        ext = dst.get("ext_copy", getlink=True)
        assert isinstance(ext, h5py.ExternalLink)
        # The filename was made absolute, so the paste still resolves
        # from a destination in a different directory.
        assert Path(ext.filename).is_absolute()
        assert isinstance(dst.get("soft_copy", getlink=True), h5py.SoftLink)


def test_copying_a_group_keeps_the_links_inside_it(simple_h5, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w"):
        pass
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.create_group(f, "/", "bundle")
        h5_edit.move_node(f, "/ext", "/bundle")
    with h5py.File(other, "r+") as dst, h5py.File(simple_h5, "r") as src:
        h5_edit.copy_node(dst, "/", "bundle", src, "/bundle")
        assert isinstance(dst["bundle"].get("ext", getlink=True), h5py.ExternalLink)


def test_copy_can_be_asked_to_follow_a_link(simple_h5, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w"):
        pass
    with h5py.File(other, "r+") as dst, h5py.File(simple_h5, "r") as src:
        h5_edit.copy_node(dst, "/", "materialised", src, "/ext", expand_external=True)
        assert isinstance(dst["materialised"], h5py.Dataset)
        assert list(dst["materialised"][()]) == [0, 1, 2, 3]


def test_copy_attrs_skips_existing_unless_told_otherwise(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.create_group(f, "/", "dest")
        f["dest"].attrs["comment"] = "keep me"
        written = h5_edit.copy_attrs(f, "/dest", f, "/meta")
        assert "NX_class" in written and "comment" not in written
        assert f["dest"].attrs["comment"] == "keep me"
        h5_edit.copy_attrs(f, "/dest", f, "/meta", overwrite=True)
        assert f["dest"].attrs["comment"] == "written at the beamline"


# -- values ----------------------------------------------------------------


def test_write_cells_returns_the_previous_values(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        previous = h5_edit.write_cells(f, "/meta/counts", {0: 100, 3: 300})
        assert previous == {0: 0, 3: 3}
        assert list(f["meta/counts"][()]) == [100, 1, 2, 300, 4, 5]


def test_write_cells_on_a_compound_field(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        previous = h5_edit.write_cells(f, "/meta/table", {(1, "b"): 999})
        assert previous[(1, "b")] == 20
        assert f["meta/table"]["b"][1] == 999


def test_write_cells_on_a_group_is_refused(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="group, not a dataset"):
            h5_edit.write_cells(f, "/meta", {0: 1})


def test_rewrite_dataset_keeps_name_and_attributes(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        f["meta/counts"].attrs["units"] = "counts"
        h5_edit.rewrite_dataset(f, "/meta/counts", np.arange(10, dtype="i4"))
        assert f["meta/counts"].shape == (10,)
        assert f["meta/counts"].attrs["units"] == "counts"


def test_insert_and_delete_rows_on_a_table(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        h5_edit.insert_rows(f, "/meta/table", at=1)
        assert f["meta/table"].shape == (4,)
        # A new row duplicates the one above it, which is what makes
        # "duplicate row" and "insert row" the same operation.
        assert f["meta/table"]["b"][1] == 10

        removed = h5_edit.delete_rows(f, "/meta/table", [0, 1])
        assert len(removed) == 2
        assert f["meta/table"].shape == (2,)


def test_delete_rows_with_no_valid_row_is_refused(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="No such rows"):
            h5_edit.delete_rows(f, "/meta/table", [99])


def test_row_operations_refuse_a_scalar(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        with pytest.raises(EditError, match="scalar"):
            h5_edit.insert_rows(f, "/meta/temperature", at=0)


# -- snapshots -------------------------------------------------------------


def test_snapshot_and_restore_a_group(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        snap = h5_edit.snapshot_node(f, "/meta")
        h5_edit.delete_node(f, "/meta")
        assert "meta" not in f
        path = h5_edit.restore_snapshot(f, "/", snap)
        assert path == "/meta"
        assert f["meta"].attrs["NX_class"] == "NXcollection"
        assert list(f["meta/counts"][()]) == [0, 1, 2, 3, 4, 5]
        assert f["meta/table"]["a"][2] == pytest.approx(3.0)


def test_snapshot_of_a_dataset_round_trips(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        snap = h5_edit.snapshot_node(f, "/meta/counts")
        h5_edit.delete_node(f, "/meta/counts")
        h5_edit.restore_snapshot(f, "/meta", snap)
        assert list(f["meta/counts"][()]) == [0, 1, 2, 3, 4, 5]


def test_node_nbytes_stops_at_the_cap(simple_h5):
    with h5py.File(simple_h5, "r+") as f:
        f.create_dataset("big", data=np.zeros(100_000, dtype="f8"))
        capped = h5_edit.node_nbytes(f, "/", cap=1024)
    assert capped > 1024


def test_node_nbytes_ignores_linked_payloads(simple_h5):
    """An external link contributes nothing: the copy keeps it a link."""
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit.node_nbytes(f, "/ext") == 0


# -- walking, searching, validating ---------------------------------------


def test_walk_never_resolves_a_soft_or_external_link(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        seen = dict(h5_edit._walk_hard(f, "/"))
    assert isinstance(seen["/ext"], h5py.ExternalLink)
    assert isinstance(seen["/soft"], h5py.SoftLink)
    # ...and nothing from inside the linked file leaked into the walk.
    assert not any(p.startswith("/ext/") for p in seen)


def test_search_matches_names_attributes_and_values(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        hits, truncated = h5_edit.walk_search(f, "counts")
        assert not truncated
        assert any(h.path == "/meta/counts" and h.where == "name" for h in hits)

        hits, _ = h5_edit.walk_search(f, "beamline")
        assert [(h.path, h.where) for h in hits] == [("/meta", "value")]

        hits, _ = h5_edit.walk_search(f, "nx_class")
        assert any(h.where == "attribute" for h in hits)


def test_search_reads_a_small_dataset_value(simple_h5):
    """NeXus metadata is small string datasets, and those are exactly
    what someone searching a file is looking for."""
    with h5py.File(simple_h5, "r") as f:
        hits, _ = h5_edit.walk_search(f, "sample A")
    assert [(h.path, h.where) for h in hits] == [("/meta/name", "data")]


def test_search_leaves_a_big_dataset_unread(simple_h5, monkeypatch):
    """The bound is what keeps a search from becoming a file read.

    ``size`` is answered by the dataspace, so a detector stack is
    skipped without a byte of it being touched. Pinned with a cap below
    the file's own six-element dataset, so the same file has one
    dataset on each side of the line.
    """
    monkeypatch.setattr(h5_edit, "VALUE_SCAN_MAX", 2)
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit._dataset_text(f["/meta/counts"]) == ""
        assert h5_edit._dataset_text(f["/meta/name"]) == "sample A"
        # ...and the walk therefore cannot match on what it did not read.
        assert h5_edit.walk_search(
            f, "0, 1, 2", include_attrs=False)[0] == []


def test_a_slash_query_is_a_place_and_a_term(simple_h5):
    """``meta/counts``: something matching counts, under something
    matching meta."""
    with h5py.File(simple_h5, "r") as f:
        hits, _ = h5_edit.walk_search(f, "meta/counts")
        assert [h.path for h in hits] == ["/meta/counts"]

        # The lead has to be there. ``nowhere`` matches no component.
        assert h5_edit.walk_search(f, "nowhere/counts")[0] == []


def test_the_lead_need_not_be_adjacent(simple_h5):
    """What the user knows is the shape of the file, not every group
    between one landmark and the next."""
    with h5py.File(simple_h5, "r+") as f:
        f.create_group("/meta/a/b").create_dataset("counts", data=1)
    with h5py.File(simple_h5, "r") as f:
        hits, _ = h5_edit.walk_search(f, "meta/counts")
    assert "/meta/a/b/counts" in [h.path for h in hits]


def test_a_slash_is_still_searchable_as_itself(simple_h5):
    """``1/Angstrom`` is a unit, ``m/s`` is a unit. Reading every slash
    as a path would quietly stop finding them."""
    with h5py.File(simple_h5, "r+") as f:
        f["/meta/counts"].attrs["units"] = "1/Angstrom"
    with h5py.File(simple_h5, "r") as f:
        hits, _ = h5_edit.walk_search(f, "1/Angstrom")
    assert [(h.path, h.where) for h in hits] == [("/meta/counts", "value")]


def test_case_sensitivity_is_a_choice(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit.walk_search(f, "COUNTS")[0]
        assert not h5_edit.walk_search(f, "COUNTS", case_sensitive=True)[0]
        assert h5_edit.walk_search(f, "counts", case_sensitive=True)[0]


def test_case_sensitivity_applies_to_a_slash_query(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit.walk_search(f, "META/counts")[0]
        assert not h5_edit.walk_search(
            f, "META/counts", case_sensitive=True)[0]


def test_search_reports_truncation(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        _, truncated = h5_edit.walk_search(f, "meta", max_nodes=1)
    assert truncated


def test_search_of_an_empty_query_finds_nothing(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        assert h5_edit.walk_search(f, "") == ([], False)


def test_validate_flags_a_group_with_no_data(simple_h5):
    with h5py.File(simple_h5, "r") as f:
        issues = h5_edit.validate(f)
    meta = [i for i in issues if i.path == "/meta"]
    assert meta and meta[0].level == "error"
    assert "data" in meta[0].message


def test_validate_flags_a_signal_that_names_nothing(tmp_path):
    path = tmp_path / "bad_signal.h5"
    with h5py.File(path, "w") as f:
        data = f.create_group("entry_0000/data")
        data.attrs["signal"] = "img_gid_q"
    with h5py.File(path, "r") as f:
        issues = h5_edit.validate(f)
    assert any(
        i.level == "error" and "not in this group" in i.message for i in issues
    )


def test_validate_passes_a_well_formed_entry(synthetic_nexus):
    with h5py.File(synthetic_nexus, "r") as f:
        issues = h5_edit.validate(f)
    assert not [i for i in issues if i.level in ("error", "warning")]


def test_validate_does_not_open_an_external_entry(tmp_path):
    """The check that would otherwise resolve 226 linked scans."""
    linked = tmp_path / "linked.h5"
    with h5py.File(linked, "w") as f:
        f.create_group("entry_0000/data")
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as f:
        f["entry_0000"] = h5py.ExternalLink(str(linked), "/entry_0000")
    linked.unlink()  # following the link now would raise
    with h5py.File(master, "r") as f:
        issues = h5_edit.validate(f)
    assert [i.level for i in issues] == ["info"]
    assert "not checked" in issues[0].message
