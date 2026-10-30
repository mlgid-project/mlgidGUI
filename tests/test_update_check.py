"""Startup update check + post-update changelog.

Covers the Qt-free logic in ``mlgidgui.update_check`` (version parsing,
release picking, changelog slicing, network fetch with a mocked urlopen)
and the MainWindow wiring (banner shown when outdated, up-to-date status
message, changelog dialog gated by the persisted last-seen version).

Source: update_check.py; main_window.py ``_on_update_check_finished`` /
``_maybe_show_changelog`` / ``_UpdateBanner``.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from mlgidgui import update_check as uc

pytestmark = pytest.mark.gui


# --------------------------- pure logic --------------------------- #
def test_parse_version_strips_v_and_rejects_junk():
    assert str(uc.parse_version("v0.1.0a9")) == "0.1.0a9"
    assert str(uc.parse_version("0.1.0a9")) == "0.1.0a9"
    assert uc.parse_version("not-a-version") is None
    assert uc.parse_version("") is None
    assert uc.parse_version(None) is None


def test_pick_latest_release_prefers_highest_and_skips_drafts():
    releases = [
        {"tag_name": "v0.1.0a8", "html_url": "u8"},
        {"tag_name": "v0.1.0a10", "html_url": "u10"},  # newest (alpha kept)
        {"tag_name": "v0.2.0", "html_url": "draft", "draft": True},  # skipped
        {"tag_name": "v0.1.0a9", "html_url": "u9"},
        {"name": "weird", "html_url": "bad"},  # unparsable -> ignored
    ]
    assert uc.pick_latest_release(releases) == ("v0.1.0a10", "u10")


def test_pick_latest_release_empty_and_non_list():
    assert uc.pick_latest_release([]) is None
    assert uc.pick_latest_release(None) is None
    assert uc.pick_latest_release({"tag_name": "v1"}) is None


def test_is_outdated():
    assert uc.is_outdated("0.1.0a9", "v0.1.0a10") is True
    assert uc.is_outdated("0.1.0a9", "v0.1.0a9") is False
    assert uc.is_outdated("0.1.0a10", "v0.1.0a9") is False
    assert uc.is_outdated("0.1.0a9", "garbage") is False


def test_parse_changelog_sections():
    text = (
        "# Changelog\n\nintro\n\n"
        "## 0.1.0a10 — tenth (2026-07-01)\n\nnew stuff\n\n"
        "## 0.1.0a9 — ninth (2026-06-30)\n\nold stuff\n"
    )
    secs = uc.parse_changelog_sections(text)
    assert [str(v) for v, _, _ in secs] == ["0.1.0a10", "0.1.0a9"]
    assert secs[0][2] == "new stuff"
    assert secs[1][1].startswith("0.1.0a9")


def test_whats_new_returns_only_newer_sections(tmp_path, monkeypatch):
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(
        "# Changelog\n\n"
        "## 0.1.0a10 — ten\n\nfeature ten\n\n"
        "## 0.1.0a9 — nine\n\nfeature nine\n\n"
        "## 0.1.0a8 — eight\n\nfeature eight\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(uc, "_changelog_path", lambda: cl)
    # updated a8 -> a10: show a10 and a9 (newest first), not a8
    out = uc.whats_new("0.1.0a10", "0.1.0a8")
    assert [h.split()[0] for h, _ in out] == ["0.1.0a10", "0.1.0a9"]
    # no update (same version) -> nothing
    assert uc.whats_new("0.1.0a10", "0.1.0a10") is None
    # fresh install (no last-seen) -> nothing
    assert uc.whats_new("0.1.0a10", None) is None
    # downgrade / last-seen newer -> nothing
    assert uc.whats_new("0.1.0a9", "0.1.0a10") is None


def test_whats_new_none_when_changelog_missing(monkeypatch):
    monkeypatch.setattr(uc, "_changelog_path", lambda: None)
    assert uc.whats_new("0.1.0a10", "0.1.0a8") is None


def test_changelog_path_prefers_the_copy_inside_the_package(tmp_path, monkeypatch):
    """A wheel carries CHANGELOG.md inside ``mlgidgui/``; a checkout has
    it one level up. The package copy wins, the repo root is the
    fallback, and with neither the lookup is ``None``."""
    import mlgidgui

    pkg = tmp_path / "mlgidgui"
    pkg.mkdir()
    monkeypatch.setattr(mlgidgui, "__file__", str(pkg / "__init__.py"))
    assert uc._changelog_path() is None
    (tmp_path / "CHANGELOG.md").write_text("root", encoding="utf-8")
    assert uc._changelog_path() == (tmp_path / "CHANGELOG.md").resolve()
    (pkg / "CHANGELOG.md").write_text("packaged", encoding="utf-8")
    assert uc._changelog_path() == (pkg / "CHANGELOG.md").resolve()


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_latest_release_parses_and_survives_offline(monkeypatch):
    payload = json.dumps(
        [{"tag_name": "v0.1.0a11", "html_url": "u11"}]
    ).encode()
    monkeypatch.setattr(
        uc.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(payload),
    )
    assert uc.latest_release() == ("v0.1.0a11", "u11")

    def _boom(req, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(uc.urllib.request, "urlopen", _boom)
    assert uc.latest_release() is None


# --------------------------- self-update logic --------------------------- #
def test_build_upgrade_command_with_and_without_pipeline():
    exe = "/some/python"
    plain = uc.build_upgrade_command(
        "v1.0.1", executable=exe, with_pipeline=False
    )
    # An exact pin from PyPI: no git URL and no repo name anywhere.
    assert plain == [exe, "-m", "pip", "install", "mlgidgui==1.0.1"]
    full = uc.build_upgrade_command(
        "v1.0.1", executable=exe, with_pipeline=True
    )
    assert full[-1] == "mlgidgui[pipeline]==1.0.1"


def test_build_upgrade_command_pins_prereleases_exactly():
    # `pip install mlgidgui==1.0.1rc1` needs no --pre; a plain upgrade would
    # skip it. The tag's `v` is stripped and the version normalised.
    cmd = uc.build_upgrade_command(
        "v1.0.1rc1", executable="py", with_pipeline=False
    )
    assert cmd[-1] == "mlgidgui==1.0.1rc1"
    assert "--pre" not in cmd and "--upgrade" not in cmd


def test_build_upgrade_command_defaults(monkeypatch):
    import sys
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: True)
    cmd = uc.build_upgrade_command("v1.2.3")
    assert cmd[0] == sys.executable
    assert cmd[-1] == "mlgidgui[pipeline]==1.2.3"
    with pytest.raises(ValueError):
        uc.build_upgrade_command("not-a-version")


def test_the_updater_talks_to_the_renamed_project():
    assert uc.DIST_NAME == "mlgidgui"
    assert uc.RELEASES_PAGE == "https://github.com/mlgid-project/mlgidGUI/releases"
    assert "mlgidGUI" in uc.RELEASES_API
    # The bridge to the renamed app is gone: nothing left of the handoff.
    for name in ("SUCCESSOR_DIST", "is_successor_release", "restart_module",
                 "build_migration_commands", "build_update_commands",
                 "REPO_GIT_URL"):
        assert not hasattr(uc, name), name


def test_is_editable_install(monkeypatch):
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": True}},
    )
    assert uc.is_editable_install() is True
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": False}},
    )
    assert uc.is_editable_install() is False
    # No direct_url.json at all (plain wheel install) -> not editable.
    monkeypatch.setattr(uc, "_read_direct_url", lambda dist_name=uc.DIST_NAME: None)
    assert uc.is_editable_install() is False


def test_self_update_supported(monkeypatch):
    # Editable -> unsupported with an explanatory reason.
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": True}},
    )
    ok, reason = uc.self_update_supported()
    assert ok is False and "editable" in reason.lower()
    # Installed from a URL or a wheel file (direct_url present, not
    # editable) -> supported.
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"url": "x", "dir_info": {"editable": False}},
    )
    assert uc.self_update_supported() == (True, "")


@pytest.mark.parametrize(
    "installer, supported",
    [("pip", True), ("conda", False), ("dpkg", False), (None, False)],
)
def test_an_install_from_pypi_updates_in_app_only_when_pip_made_it(
    monkeypatch, installer, supported
):
    """A PyPI install has no direct_url.json (pip writes one only for URLs
    and local files), so the INSTALLER file is what tells a pip install
    from a conda or distro one that pip must not touch."""
    monkeypatch.setattr(uc, "_read_direct_url", lambda dist_name=uc.DIST_NAME: None)
    monkeypatch.setattr(uc, "_installer_name", lambda dist_name=uc.DIST_NAME: installer)
    ok, reason = uc.self_update_supported()
    assert ok is supported
    assert (reason == "") is supported


def test_installer_name_reads_the_metadata_file(monkeypatch):
    import importlib.metadata as md

    class _Dist:
        def __init__(self, text):
            self._text = text

        def read_text(self, name):
            assert name == "INSTALLER"
            return self._text

    monkeypatch.setattr(md, "distribution", lambda name: _Dist("pip\n"))
    assert uc._installer_name() == "pip"
    monkeypatch.setattr(md, "distribution", lambda name: _Dist(None))
    assert uc._installer_name() is None
    monkeypatch.setattr(md, "distribution", lambda name: _Dist("  \n"))
    assert uc._installer_name() is None

    def _missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "distribution", _missing)
    assert uc._installer_name() is None


def test_pipeline_extra_installed(monkeypatch):
    monkeypatch.setattr(uc, "_dist_present", lambda name: name == "mlgidbase")
    assert uc.pipeline_extra_installed() is True
    monkeypatch.setattr(uc, "_dist_present", lambda name: False)
    assert uc.pipeline_extra_installed() is False


# ----------------- Windows launcher rename (WinError 32) ----------------- #
# A running gui-scripts launcher (Scripts\mlgidgui.exe) is locked on
# Windows, so pip's uninstall of the old version dies with WinError 32.
# free_locked_launchers renames it aside first; restore_launchers puts it
# back if pip fails. The path logic is OS-independent, so these run
# everywhere by forcing platform="nt" against a tmp scripts dir.
def test_free_locked_launchers_noop_off_windows(tmp_path):
    exe = tmp_path / "mlgidgui.exe"
    exe.write_bytes(b"launcher")
    assert uc.free_locked_launchers(
        platform="posix", scripts_dir=tmp_path
    ) == []
    assert exe.exists()


def test_free_locked_launchers_renames_exe_aside(tmp_path):
    exe = tmp_path / "mlgidgui.exe"
    exe.write_bytes(b"launcher")
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    backup = tmp_path / "mlgidgui.exe.old"
    assert renames == [(exe, backup)]
    assert not exe.exists() and backup.read_bytes() == b"launcher"
    # No launcher at all (e.g. console_scripts removed by hand) -> no-op.
    assert uc.free_locked_launchers(
        platform="nt", scripts_dir=tmp_path
    ) == []


def test_free_locked_launchers_sweeps_stale_backups(tmp_path):
    exe = tmp_path / "mlgidgui.exe"
    exe.write_bytes(b"new")
    (tmp_path / "mlgidgui.exe.old").write_bytes(b"stale")
    (tmp_path / "mlgidgui.exe.old2").write_bytes(b"staler")
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    # Both leftovers deleted, so the rename lands on plain ".old" again.
    assert renames == [(exe, tmp_path / "mlgidgui.exe.old")]
    assert (tmp_path / "mlgidgui.exe.old").read_bytes() == b"new"
    assert not (tmp_path / "mlgidgui.exe.old2").exists()


def test_free_locked_launchers_skips_undeletable_backup(
    tmp_path, monkeypatch
):
    # A backup from the PREVIOUS update whose process is still running
    # can be neither deleted nor overwritten; the rename must land on
    # the next free name instead.
    from pathlib import Path

    exe = tmp_path / "mlgidgui.exe"
    exe.write_bytes(b"new")
    locked = tmp_path / "mlgidgui.exe.old"
    locked.write_bytes(b"still running")
    monkeypatch.setattr(
        Path, "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked")),
    )
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    assert renames == [(exe, tmp_path / "mlgidgui.exe.old2")]
    assert locked.read_bytes() == b"still running"
    assert (tmp_path / "mlgidgui.exe.old2").read_bytes() == b"new"


def test_restore_launchers(tmp_path):
    exe = tmp_path / "mlgidgui.exe"
    backup = tmp_path / "mlgidgui.exe.old"
    # pip failed before installing a replacement -> backup renamed back.
    backup.write_bytes(b"launcher")
    uc.restore_launchers([(exe, backup)])
    assert exe.read_bytes() == b"launcher" and not backup.exists()
    # pip already wrote the new exe (or rolled back) -> keep it, drop
    # the backup.
    exe.write_bytes(b"fresh")
    backup.write_bytes(b"old")
    uc.restore_launchers([(exe, backup)])
    assert exe.read_bytes() == b"fresh" and not backup.exists()


def test_install_worker_frees_launchers_and_restores_on_failure(
    qapp, monkeypatch
):
    import subprocess
    import types
    from mlgidgui.main_window import _UpdateInstallWorker

    calls: list = []
    renames = [("orig", "backup")]
    monkeypatch.setattr(uc, "free_locked_launchers", lambda: renames)
    monkeypatch.setattr(
        uc, "restore_launchers", lambda r: calls.append(("restore", r))
    )

    ran: list = []

    def _fake_run(cmd, **kwargs):
        ran.append((cmd, kwargs))
        return types.SimpleNamespace(
            returncode=_fake_run.rc, stdout="out", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    results: list = []

    # Success: launcher freed, restore NOT called.
    _fake_run.rc = 0
    worker = _UpdateInstallWorker(["pip", "install"])
    worker.finished.connect(lambda rc, out: results.append(rc))
    worker.run()
    assert results == [0] and calls == []
    # The pip child must not flash a console window under pythonw.
    assert ran[0][1]["creationflags"] == getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )

    # Failure: the renamed launcher is put back.
    _fake_run.rc = 1
    worker = _UpdateInstallWorker(["pip", "install"])
    worker.finished.connect(lambda rc, out: results.append(rc))
    worker.run()
    assert results == [0, 1]
    assert calls == [("restore", renames)]


def _worker_with_fake_pip(monkeypatch, returncode, *, cannot_launch=False):
    """An ``_UpdateInstallWorker`` whose ``subprocess.run`` answers with
    ``returncode`` and records the commands it was given."""
    import subprocess
    import types

    ran: list = []
    restored: list = []
    renames = [("orig", "backup")]
    monkeypatch.setattr(uc, "free_locked_launchers", lambda: renames)
    monkeypatch.setattr(uc, "restore_launchers", restored.append)

    def _fake_run(cmd, **kwargs):
        ran.append(cmd)
        if cannot_launch:
            raise OSError("cannot launch")
        return types.SimpleNamespace(
            returncode=returncode, stdout="out\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return ran, restored, renames


def _run_install_worker(command):
    from mlgidgui.update_ui import _UpdateInstallWorker

    results: list = []
    worker = _UpdateInstallWorker(command)
    worker.finished.connect(lambda rc, out: results.append((rc, out)))
    worker.run()
    return results


def test_install_worker_runs_the_command_and_reports_pips_output(
    qapp, monkeypatch, tmp_path
):
    from mlgidgui import diagnostics

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    ran, restored, _ = _worker_with_fake_pip(monkeypatch, 0)
    results = _run_install_worker(["pip", "install", "new"])
    assert ran == [["pip", "install", "new"]]
    assert results == [(0, "out\n")]
    assert restored == []
    # One started and one finished block in the update log.
    text = (tmp_path / diagnostics.UPDATE_LOG_NAME).read_text(encoding="utf-8")
    assert text.count("=== update started") == 1
    assert text.count("=== update finished") == 1


def test_install_worker_failure_puts_the_launcher_back(qapp, monkeypatch):
    ran, restored, renames = _worker_with_fake_pip(monkeypatch, 1)
    results = _run_install_worker(["pip", "install", "new"])
    assert results == [(1, "out\n")]
    assert restored == [renames]


def test_install_worker_install_that_cannot_launch_fails(qapp, monkeypatch):
    ran, restored, renames = _worker_with_fake_pip(
        monkeypatch, 0, cannot_launch=True
    )
    results = _run_install_worker(["pip", "install", "new"])
    assert results == [(1, "Failed to launch pip: cannot launch")]
    assert len(ran) == 1 and restored == [renames]


# --------------------------- MainWindow wiring --------------------------- #
def test_banner_shown_only_when_outdated(main_window):
    from mlgidgui import __version__ as current

    assert main_window._update_banner.isHidden()
    # up to date -> banner stays hidden
    main_window._on_update_check_finished((f"v{current}", "http://x"))
    assert main_window._update_banner.isHidden()
    # newer available -> banner shown with the version
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert "99.0.0" in main_window._update_banner._label.text()


def test_up_to_date_status_message_only_when_requested(main_window):
    main_window._update_notify_uptodate = True
    main_window._on_update_check_finished(None)
    assert "up to date" in main_window.statusBar().currentMessage().lower()


def test_maybe_show_changelog_gated_by_last_seen(main_window, monkeypatch):
    from mlgidgui import __version__ as current
    from PySide6.QtCore import QSettings

    shown: list[list[tuple[str, str]]] = []
    monkeypatch.setattr(
        main_window, "_show_changelog_dialog",
        lambda cur, sections: shown.append(sections),
    )
    monkeypatch.setattr(
        uc, "whats_new", lambda cur, last: [("0.1.0aX — x", "body")]
    )
    QSettings().setValue(main_window._LAST_SEEN_VERSION_KEY, "0.0.1")
    main_window._maybe_show_changelog()
    assert shown  # dialog shown because last-seen was older
    # last-seen recorded as current so it won't show again
    assert str(QSettings().value(main_window._LAST_SEEN_VERSION_KEY)) == current

    # whats_new returns nothing -> no dialog, still records version
    shown.clear()
    monkeypatch.setattr(uc, "whats_new", lambda cur, last: None)
    main_window._maybe_show_changelog()
    assert not shown


def test_update_button_shown_only_when_self_update_supported(
    main_window, monkeypatch
):
    from PySide6.QtCore import QSettings

    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)
    # Supported install -> banner offers the "Update now" button.
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert not main_window._update_banner._update_btn.isHidden()
    # Editable / unsupported -> banner shown but no in-app install button.
    monkeypatch.setattr(
        uc, "self_update_supported", lambda *a, **k: (False, "editable")
    )
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert main_window._update_banner._update_btn.isHidden()


def test_auto_update_installs_on_launch_when_enabled(main_window, monkeypatch):
    from PySide6.QtCore import QSettings

    requested: list[bool] = []
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    # Auto-update on launch goes through the same confirm-then-install flow as
    # the button (never a silent install), so it routes to
    # _on_install_update_requested, not _start_update_install directly.
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: requested.append(True),
    )
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, True)
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert requested == [True]
    assert main_window._update_banner.isHidden()
    # Reset so the setting doesn't leak into other tests in this process.
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)


def test_update_now_menu_confirms_install_when_newer(main_window, monkeypatch):
    # Help -> Update now… runs a check with install_when_found; on finding a
    # newer supported release it goes straight to the confirm dialog, not the
    # banner.
    confirmed: list[bool] = []
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: confirmed.append(True),
    )
    main_window._update_install_when_found = True
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert confirmed == [True]
    assert main_window._update_banner.isHidden()


def test_auto_update_ignored_for_unsupported_install(main_window, monkeypatch):
    from PySide6.QtCore import QSettings

    requested: list[bool] = []
    # Toggle is on, but the install is editable -> never auto-install; the
    # banner is shown without the "Update now" button.
    monkeypatch.setattr(
        uc, "self_update_supported", lambda *a, **k: (False, "editable")
    )
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: requested.append(True),
    )
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, True)
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert requested == []
    assert not main_window._update_banner.isHidden()
    assert main_window._update_banner._update_btn.isHidden()
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)


# ------------------------- MainWindow wiring: the install ------------------------- #
def test_banner_names_the_new_version_and_the_running_one(main_window):
    banner = main_window._update_banner
    banner.show_update("1.0.0", "v1.0.1", "http://x", can_install=True)
    text = banner._label.text()
    assert "newer version of mlgidGUI" in text and "v1.0.1" in text
    assert "mlgidLAB" not in text
    assert not banner._update_btn.isHidden()


def _confirm_texts(main_window, monkeypatch, tag):
    """Open the confirm dialog for ``tag`` (answering Cancel); return the
    ``(title, text)`` it showed."""
    from PySide6.QtWidgets import QMessageBox

    shown: list = []

    def _question(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", _question)
    main_window._latest_tag = tag
    main_window._on_install_update_requested()
    return shown[0]


def test_confirm_dialog_names_the_release_and_the_restart(
    main_window, monkeypatch
):
    title, text = _confirm_texts(main_window, monkeypatch, "v1.0.1")
    assert title == "Update mlgidGUI"
    assert "v1.0.1" in text and "pip-install" in text and "restart" in text


def _start_install_with_stubs(main_window, monkeypatch, tag):
    """Run ``_start_update_install(tag)`` with a stub thread and worker, so
    nothing actually runs (real thread teardown is covered by
    ``test_update_threads_survive_an_instant_worker``).
    Returns ``(commands_seen, progress_calls)``."""
    import mlgidgui.main_window_update as mwu

    seen: list = []
    progress: list = []

    class _Signal:
        def connect(self, slot):
            pass

    class _StubThread:
        def __init__(self, parent=None):
            self.started = _Signal()
            self.finished = _Signal()

        def start(self):
            pass

        def quit(self):
            pass

        def deleteLater(self):
            pass

    class _StubWorker:
        def __init__(self, commands):
            seen.append(commands)
            self.finished = _Signal()

        def moveToThread(self, thread):
            pass

        def run(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(mwu, "QThread", _StubThread)
    monkeypatch.setattr(mwu, "_UpdateInstallWorker", _StubWorker)
    monkeypatch.setattr(
        mwu, "make_progress_dialog",
        lambda parent, text, title="": progress.append((title, text)),
    )
    main_window._install_thread = None
    main_window._latest_tag = tag
    main_window._start_update_install(tag)
    return seen, progress


def test_start_install_hands_the_worker_the_pinned_pypi_command(
    main_window, monkeypatch
):
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    seen, progress = _start_install_with_stubs(main_window, monkeypatch, "v1.0.1")
    (command,) = seen
    assert command[-1] == "mlgidgui==1.0.1"
    assert command[1:4] == ["-m", "pip", "install"]
    assert progress == [("Updating mlgidGUI", "Updating mlgidGUI to v1.0.1…")]


# Child-process body for the thread-teardown test below: real QThreads,
# workers that finish the moment they start, many start/finish cycles.
_INSTANT_WORKERS_SCRIPT = r'''
import os
import sys
import time

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

import mlgidgui.main_window_update as mwu


class InstantCheck(QObject):
    finished = Signal(object)

    @Slot()
    def run(self):
        self.finished.emit(None)


class InstantInstall(QObject):
    finished = Signal(int, str)

    def __init__(self, commands):
        super().__init__()

    @Slot()
    def run(self):
        self.finished.emit(0, "")


mwu._UpdateCheckWorker = InstantCheck
mwu._UpdateInstallWorker = InstantInstall
mwu.make_progress_dialog = lambda *args, **kwargs: None


class Host(QObject, mwu.UpdateMixin):
    done = 0

    @Slot(object)
    def _on_update_check_finished(self, result):
        Host.done += 1

    @Slot(int, str)
    def _on_update_install_finished(self, returncode, output):
        Host.done += 1


def drain(attr):
    deadline = time.monotonic() + 10
    while getattr(host, attr) is not None and time.monotonic() < deadline:
        app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


app = QApplication([])
host = Host()
for _ in range(int(sys.argv[1])):
    host._start_update_check()
    drain("_update_thread")
    host._start_update_install("v0.1.0a99")
    drain("_install_thread")
print("finished", Host.done)
sys.stdout.flush()
os._exit(0)
'''


def test_update_threads_survive_an_instant_worker(tmp_path):
    """Starting and finishing the check / install threads must not crash.

    The worker used to be deleted twice: by a ``deleteLater`` that runs
    on the worker thread as it finishes, and by the GUI thread dropping
    the last Python reference in ``_clear_*_thread`` at the same moment.
    That segfaulted within the first few cycles. Run in a child process
    so a crash fails this test instead of ending the whole session.
    """
    import subprocess
    import sys

    script = tmp_path / "instant_update_workers.py"
    script.write_text(_INSTANT_WORKERS_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), "100"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "finished 200" in proc.stdout


def test_failure_dialog_shows_the_manual_command(main_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: shown.append(self.informativeText())
    )
    main_window._latest_tag = "v1.0.1"
    main_window._on_update_install_finished(1, "pip exploded")
    assert any(
        line.endswith("pip install mlgidgui==1.0.1")
        for line in shown[0].splitlines()
    )


def test_restart_relaunches_the_app_and_logs_it(main_window, monkeypatch, tmp_path):
    import sys

    from PySide6.QtWidgets import QMessageBox

    import mlgidgui.main_window_update as mwu
    from mlgidgui import diagnostics

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    asked: list = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda parent, title, text, *a, **k: (
            asked.append(text) or QMessageBox.StandardButton.Yes
        ),
    )
    launched: list = []
    monkeypatch.setattr(
        mwu.QProcess, "startDetached",
        lambda exe, args: launched.append((exe, args)),
    )
    monkeypatch.setattr(mwu.QApplication, "quit", lambda: None)
    main_window._latest_tag = "v1.0.1"
    main_window._prompt_restart()
    assert "mlgidGUI was updated" in asked[0]
    assert launched == [(sys.executable, ["-m", "mlgidgui"])]
    text = (tmp_path / diagnostics.UPDATE_LOG_NAME).read_text("utf-8")
    assert "-m mlgidgui ===" in text


def test_there_is_no_rename_notice_any_more(main_window):
    """The one-time "mlgidLAB is becoming mlgidGUI" notice did its job in
    the alphas; the renamed app neither shows it nor offers it."""
    assert not hasattr(main_window, "_maybe_show_rename_notice")
    assert not hasattr(main_window, "action_rename_notice")
    titles = [a.text() for a in main_window.menuBar().actions()]
    help_menu = main_window.menuBar().actions()[titles.index("&Help")].menu()
    assert not any("becoming" in a.text() for a in help_menu.actions())


def test_startup_checks_show_the_changelog_before_the_network_check(
    main_window, monkeypatch
):
    order: list[str] = []
    monkeypatch.setattr(
        main_window, "_maybe_show_changelog", lambda: order.append("changelog")
    )
    monkeypatch.setattr(
        main_window, "_start_update_check", lambda **k: order.append("check")
    )
    main_window._run_startup_update_checks()
    assert order == ["changelog", "check"]


# ------------- Windows: the launcher the move from mlgidLAB left ------------- #
def test_sweep_legacy_launchers_deletes_what_the_move_left(tmp_path):
    (tmp_path / "mlgidlab.exe.old").write_bytes(b"x")
    (tmp_path / "mlgidlab.exe.old2").write_bytes(b"y")
    keep = [tmp_path / "mlgidgui.exe", tmp_path / "mlgidgui.exe.old",
            tmp_path / "other.exe.old"]
    for path in keep:
        path.write_bytes(b"z")
    assert uc.sweep_legacy_launchers(tmp_path, platform="nt") == 2
    assert not list(tmp_path.glob("mlgidlab.exe*"))
    assert all(path.exists() for path in keep)


def test_sweep_legacy_launchers_does_nothing_off_windows(tmp_path):
    stale = tmp_path / "mlgidlab.exe.old"
    stale.write_bytes(b"x")
    assert uc.sweep_legacy_launchers(tmp_path, platform="posix") == 0
    assert stale.exists()


def test_sweep_legacy_launchers_leaves_a_locked_file_for_next_time(
    tmp_path, monkeypatch
):
    from pathlib import Path

    stale = tmp_path / "mlgidlab.exe.old"
    stale.write_bytes(b"still running")
    monkeypatch.setattr(
        Path, "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked")),
    )
    assert uc.sweep_legacy_launchers(tmp_path, platform="nt") == 0
    assert stale.exists()


def test_main_sweeps_the_legacy_launcher_after_dropping_its_own():
    import inspect

    import mlgidgui

    src = inspect.getsource(mlgidgui.main)
    assert (src.index("drop_launcher_from_sys_path()")
            < src.index("sweep_legacy_launchers()"))


# ------- the app must not hold its own launcher open (Windows updates) ------- #
def _fds_open_on(path) -> bool:
    """Linux only: does this process hold ``path`` open?"""
    import os

    target = os.path.realpath(path)
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.readlink(f"/proc/self/fd/{name}")) == target:
                return True
        except OSError:
            continue
    return False


@pytest.fixture
def launcher_zip(tmp_path, monkeypatch):
    """A zip named like a pip launcher, first on ``sys.path`` as it is when
    the interpreter runs ``Scripts\\mlgidgui.exe``; ``sys.path`` and the
    metadata caches are restored afterwards."""
    import importlib.metadata
    import sys
    import zipfile

    exe = tmp_path / "mlgidgui.exe"
    with zipfile.ZipFile(exe, "w") as z:
        z.writestr("__main__.py", "")
    monkeypatch.setattr(sys, "path", [str(exe)] + list(sys.path))
    yield exe
    importlib.metadata.FastPath.__new__.cache_clear()


def test_drop_launcher_from_sys_path_only_removes_executables(monkeypatch):
    import sys

    keep = ["/a/site-packages", "/b/lib.zip", "", "/c/tools"]
    monkeypatch.setattr(
        sys, "path", ["C:\\v\\Scripts\\mlgidgui.exe"] + keep + ["D:\\x\\Y.EXE"]
    )
    removed = uc.drop_launcher_from_sys_path()
    assert removed == ["C:\\v\\Scripts\\mlgidgui.exe", "D:\\x\\Y.EXE"]
    assert sys.path == keep
    assert uc.drop_launcher_from_sys_path() == []


@pytest.mark.skipif(
    not __import__("os").path.isdir("/proc/self/fd"), reason="needs /proc"
)
def test_release_launcher_handles_closes_the_archive(launcher_zip):
    import importlib.metadata

    # What the app does on startup: any importlib.metadata lookup scans the
    # whole sys.path, including the launcher zip.
    list(importlib.metadata.distributions())
    if not _fds_open_on(launcher_zip):
        pytest.skip("this Python does not keep the archive open")
    uc.release_launcher_handles()
    assert not _fds_open_on(launcher_zip)


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows file locking")
def test_launcher_can_be_renamed_after_the_handles_are_released(launcher_zip):
    import importlib.metadata

    list(importlib.metadata.distributions())
    uc.release_launcher_handles()
    aside = launcher_zip.with_name("mlgidgui.exe.old")
    launcher_zip.rename(aside)  # a held handle makes this a PermissionError
    assert aside.exists() and not launcher_zip.exists()


def test_free_locked_launchers_releases_our_own_handle_first(
    tmp_path, monkeypatch
):
    order: list[str] = []
    monkeypatch.setattr(
        uc, "release_launcher_handles", lambda: order.append("release")
    )
    exe = tmp_path / "mlgidgui.exe"
    exe.write_bytes(b"x")
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    order.append("renamed" if renames else "not renamed")
    assert order == ["release", "renamed"]
    # Off Windows there is nothing to release.
    order.clear()
    uc.free_locked_launchers(platform="posix", scripts_dir=tmp_path)
    assert order == []


def test_main_drops_the_launcher_before_importing_qt():
    import inspect

    import mlgidgui

    source = inspect.getsource(mlgidgui.main)
    drop = source.index("drop_launcher_from_sys_path")
    assert drop < source.index("from PySide6")
