"""Linux desktop integration, so the shell can find the app's icon.

``setWindowIcon`` fills the window's own icon and, on X11, ``_NET_WM_ICON``.
It does not reach the dash, the window switcher or the top bar: GNOME
resolves a window to an *application* first and takes the icon from that
application's ``.desktop`` file. The app shipped no such file, so there
was nothing to resolve to.

The fix is two names that have to agree -- ``setDesktopFileName`` (the
Wayland ``app_id`` / X11 ``WM_CLASS`` instance) and a ``.desktop`` file of
that basename -- so most of what is worth testing here is agreement, plus
the property that a broken ``$HOME`` cannot take the startup down with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mlgidgui import desktop_entry


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    """Point XDG_DATA_HOME at a temp dir so nothing touches the real one."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return tmp_path


def _entries(text: str) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in text.splitlines()
        if "=" in line and not line.startswith("[")
    )


# -- the three names must agree -----------------------------------------


def test_desktop_name_is_the_file_basename(xdg):
    """The whole mechanism is the shell matching an app id to a file of
    that name, so a mismatch here silently disables the feature."""
    assert desktop_entry.desktop_file_path().name == (
        f"{desktop_entry.DESKTOP_NAME}.desktop"
    )


def test_icon_key_matches_the_installed_icon_name(xdg):
    fields = _entries(desktop_entry.desktop_file_contents())
    assert fields["Icon"] == desktop_entry.DESKTOP_NAME
    assert desktop_entry.icon_path(48).stem == desktop_entry.DESKTOP_NAME


def test_startup_wm_class_matches_the_desktop_name(xdg):
    """Qt derives WM_CLASS from setDesktopFileName, so these agree today.
    StartupWMClass is what keeps the match if they ever stop agreeing."""
    fields = _entries(desktop_entry.desktop_file_contents())
    assert fields["StartupWMClass"] == desktop_entry.DESKTOP_NAME


def test_main_hands_qt_the_same_name(monkeypatch):
    """``main`` must call setDesktopFileName with DESKTOP_NAME, before a
    window exists. Read from the source rather than by running ``main``,
    which would need a real QApplication and a display."""
    src = Path(desktop_entry.__file__).with_name("__init__.py").read_text()
    assert "app.setDesktopFileName(desktop_entry.DESKTOP_NAME)" in src
    assert src.index("setDesktopFileName") < src.index("window = MainWindow()")


# -- the entry itself ----------------------------------------------------


def test_entry_is_a_valid_application_entry(xdg):
    fields = _entries(desktop_entry.desktop_file_contents())
    assert fields["Type"] == "Application"
    assert fields["Name"] == "mlgidGUI"
    assert fields["Terminal"] == "false"
    # Exactly one main category: desktop-file-validate warns that a
    # second one lists the app twice in a categorised menu.
    assert fields["Categories"] == "Science;Physics;"


def test_no_mimetype_claim(xdg):
    """Claiming application/x-hdf would put mlgidGUI in "Open with" for
    every HDF5 file on the machine, which is a bigger claim than a
    viewer for one flavour of NeXus should make."""
    assert "MimeType" not in _entries(desktop_entry.desktop_file_contents())


def test_exec_is_absolute(xdg):
    """A bare ``mlgidgui`` resolves for the person who typed it in their
    activated env and fails for the same person clicking the dash icon --
    the launcher runs with the session's PATH, which usually has no conda
    env on it."""
    exec_line = _entries(desktop_entry.desktop_file_contents())["Exec"]
    assert Path(exec_line.split()[0]).is_absolute()


def test_exec_takes_a_whole_selection(xdg):
    """``%F`` (a list), not ``%f`` (one path). ``main`` opens every extra
    argv entry that exists, so the launcher may hand it a selection; with
    ``%f`` the shell starts one process per file and opening three files
    would give three copies of the app."""
    exec_line = _entries(desktop_entry.desktop_file_contents())["Exec"]
    assert exec_line.endswith(" %F")


def test_exec_prefers_the_script_beside_the_interpreter(tmp_path, monkeypatch):
    script = tmp_path / desktop_entry.DESKTOP_NAME
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    assert desktop_entry._exec_command() == str(script)


def test_exec_falls_back_to_module_invocation(tmp_path, monkeypatch):
    """A source checkout with no installed gui-script still gets a line
    that works."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(desktop_entry.shutil, "which", lambda _n: None)
    assert desktop_entry._exec_command() == (
        f"{tmp_path / 'python'} -m mlgidgui"
    )


# -- installation --------------------------------------------------------


def test_install_writes_the_entry_and_icons(xdg):
    assert desktop_entry.install() is True
    assert desktop_entry.desktop_file_path().is_file()
    installed = [s for s in desktop_entry.ICON_SIZES
                 if desktop_entry.icon_path(s).is_file()]
    assert installed == list(desktop_entry.ICON_SIZES)


def test_installed_icons_are_the_shipped_bytes(xdg):
    from importlib import resources
    desktop_entry.install()
    source = resources.files("mlgidgui").joinpath("assets/app/mlgidgui_48.png")
    with resources.as_file(source) as real:
        assert desktop_entry.icon_path(48).read_bytes() == Path(real).read_bytes()


def test_install_is_idempotent(xdg):
    """Second run writes nothing, so a launch does not churn the file for
    every backup tool and menu watcher on the box."""
    assert desktop_entry.install() is True
    assert desktop_entry.install() is False


def test_a_moved_environment_rewrites_the_exec_line(xdg, monkeypatch):
    """"Skip if the file exists" would leave a stale Exec pointing at a
    deleted env forever, which is worse than no entry at all."""
    desktop_entry.install()
    monkeypatch.setattr(
        desktop_entry, "_exec_command", lambda: "/somewhere/else/mlgidlab")
    assert desktop_entry.install() is True
    assert "/somewhere/else/mlgidlab" in (
        desktop_entry.desktop_file_path().read_text()
    )


def test_xdg_data_home_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert desktop_entry.desktop_file_path().parent == tmp_path / "applications"


def test_a_relative_xdg_data_home_is_ignored(tmp_path, monkeypatch):
    """The spec says a relative XDG_DATA_HOME is invalid; honouring one
    would scatter the entry wherever the app happened to be started."""
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")
    assert desktop_entry.desktop_file_path().is_absolute()


# -- a broken $HOME must not cost the startup ---------------------------


def test_install_survives_an_unwritable_home(xdg, monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("read-only file system")
    monkeypatch.setattr(Path, "mkdir", _boom)
    assert desktop_entry.install() is False


def test_install_survives_missing_icon_assets(xdg, monkeypatch):
    """A packaging slip should cost the dash icon, not the launch. The
    entry is still written -- the shell falls back to the window icon."""
    def _boom(*_a, **_k):
        raise FileNotFoundError("no assets in this wheel")
    monkeypatch.setattr(desktop_entry.resources, "files", _boom)
    assert desktop_entry.install() is True
    assert desktop_entry.desktop_file_path().is_file()


def test_install_is_a_no_op_off_linux(xdg, monkeypatch):
    monkeypatch.setattr(desktop_entry, "is_supported", lambda: False)
    assert desktop_entry.install() is False
    assert not desktop_entry.desktop_file_path().exists()


def test_is_supported_tracks_the_platform(monkeypatch):
    for platform, expected in (
        ("linux", True), ("win32", False), ("darwin", False),
    ):
        monkeypatch.setattr(sys, "platform", platform)
        assert desktop_entry.is_supported() is expected


# -- the entry the mlgidLAB alphas installed ------------------------------


def _alpha_entry(data_home):
    """What an mlgidLAB alpha left in the user's data dir."""
    entry = data_home / "applications" / "mlgidlab.desktop"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "[Desktop Entry]\nName=mlgidLAB\nExec=/env/bin/mlgidlab %F\n"
        "Icon=mlgidlab\nStartupWMClass=mlgidlab\n", encoding="utf-8")
    icons = []
    for size in desktop_entry.ICON_SIZES:
        icon = (data_home / "icons" / "hicolor" / f"{size}x{size}" / "apps"
                / "mlgidlab.png")
        icon.parent.mkdir(parents=True)
        icon.write_bytes(b"png")
        icons.append(icon)
    return entry, icons


def test_install_removes_the_alpha_entry_and_icons(xdg):
    entry, icons = _alpha_entry(xdg)
    assert desktop_entry.install() is True
    assert not entry.exists()
    assert not any(icon.exists() for icon in icons)
    assert desktop_entry.desktop_file_path().is_file()


def test_a_foreign_file_with_the_old_name_is_left_alone(xdg):
    """Only an entry that is recognisably ours goes; someone else's
    ``mlgidlab.desktop`` is not ours to delete."""
    entry = xdg / "applications" / "mlgidlab.desktop"
    entry.parent.mkdir(parents=True)
    entry.write_text("[Desktop Entry]\nName=Something else\n", encoding="utf-8")
    desktop_entry.install()
    assert entry.exists()


def test_removing_the_alpha_entry_twice_is_harmless(xdg):
    _alpha_entry(xdg)
    assert desktop_entry.remove_legacy() == 1 + len(desktop_entry.ICON_SIZES)
    assert desktop_entry.remove_legacy() == 0
