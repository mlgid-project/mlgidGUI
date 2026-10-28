"""Startup update check + post-update changelog.

Covers the Qt-free logic in ``mlgidlab.update_check`` (version parsing,
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

from mlgidlab import update_check as uc

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
    """A wheel carries CHANGELOG.md inside ``mlgidlab/``; a checkout has
    it one level up. The package copy wins, the repo root is the
    fallback, and with neither the lookup is ``None``."""
    import mlgidlab

    pkg = tmp_path / "mlgidlab"
    pkg.mkdir()
    monkeypatch.setattr(mlgidlab, "__file__", str(pkg / "__init__.py"))
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
        "v0.1.0a10", executable=exe, with_pipeline=False
    )
    assert plain == [
        exe, "-m", "pip", "install", "--upgrade",
        "mlgidlab @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a10",
    ]
    full = uc.build_upgrade_command(
        "v0.1.0a10", executable=exe, with_pipeline=True
    )
    assert full[-1] == (
        "mlgidlab[pipeline] @ "
        "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a10"
    )


def test_build_upgrade_command_defaults_to_running_interpreter(monkeypatch):
    import sys
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    cmd = uc.build_upgrade_command("v1.2.3")
    assert cmd[0] == sys.executable
    assert cmd[-1].endswith("@v1.2.3")


# ------------------- handoff to the renamed app (1.0.0) ------------------- #
def test_is_successor_release_only_for_major_one_and_up():
    # The alphas keep the in-place upgrade; the handoff stays dormant until
    # a 1.x release exists.
    for tag in ("v0.1.0a20", "v0.1.0a21", "0.9.9", "v0.6.2"):
        assert not uc.is_successor_release(tag), tag
    for tag in ("v1.0.0", "1.0.0", "v1.0.0rc1", "v1.2.3", "v2.0.0"):
        assert uc.is_successor_release(tag), tag
    # Unparsable input never triggers a migration.
    for bad in ("", None, "not-a-version"):
        assert not uc.is_successor_release(bad)


def test_restart_module_follows_the_installed_release():
    assert uc.restart_module("v0.1.0a21") == "mlgidlab"
    assert uc.restart_module("v1.0.0") == "mlgidgui"
    assert uc.restart_module(None) == "mlgidlab"


def test_build_migration_commands_install_first_then_uninstall():
    exe = "/some/python"
    plain, drop = uc.build_migration_commands(
        "v1.0.0", executable=exe, with_pipeline=False
    )
    # Exact pin from PyPI, no git URL and no repo name anywhere.
    assert plain == [exe, "-m", "pip", "install", "mlgidgui==1.0.0"]
    # The old distribution goes only after the install, non-interactively.
    assert drop == [exe, "-m", "pip", "uninstall", "-y", "mlgidlab"]

    full, _ = uc.build_migration_commands(
        "v1.0.0", executable=exe, with_pipeline=True
    )
    assert full[-1] == "mlgidgui[pipeline]==1.0.0"


def test_build_migration_commands_pins_prereleases_exactly():
    # `pip install mlgidgui==1.0.0rc1` needs no --pre; a plain upgrade would
    # skip it. The tag's `v` is stripped and the version normalised.
    install, _ = uc.build_migration_commands(
        "v1.0.0rc1", executable="py", with_pipeline=False
    )
    assert install[-1] == "mlgidgui==1.0.0rc1"
    assert "--pre" not in install and "--upgrade" not in install


def test_build_migration_commands_defaults(monkeypatch):
    import sys
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: True)
    install, drop = uc.build_migration_commands("v1.0.0")
    assert install[0] == drop[0] == sys.executable
    assert install[-1] == "mlgidgui[pipeline]==1.0.0"
    with pytest.raises(ValueError):
        uc.build_migration_commands("not-a-version")


def test_build_update_commands_picks_upgrade_or_migration():
    exe = "py"
    # An alpha target: exactly the old single in-place upgrade.
    alpha = uc.build_update_commands(
        "v0.1.0a22", executable=exe, with_pipeline=False
    )
    assert alpha == [
        uc.build_upgrade_command(
            "v0.1.0a22", executable=exe, with_pipeline=False
        )
    ]
    # A 1.x target: install the successor, then drop this distribution.
    migrated = uc.build_update_commands(
        "v1.0.0", executable=exe, with_pipeline=False
    )
    assert [c[3] for c in migrated] == ["install", "uninstall"]
    assert migrated[0][-1] == "mlgidgui==1.0.0"


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
    # Normal pip install (direct_url present, not editable) -> supported.
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"url": "x", "dir_info": {"editable": False}},
    )
    assert uc.self_update_supported() == (True, "")
    # Unknown install (no direct_url) -> unsupported.
    monkeypatch.setattr(uc, "_read_direct_url", lambda dist_name=uc.DIST_NAME: None)
    ok, reason = uc.self_update_supported()
    assert ok is False and reason


def test_pipeline_extra_installed(monkeypatch):
    monkeypatch.setattr(uc, "_dist_present", lambda name: name == "mlgidbase")
    assert uc.pipeline_extra_installed() is True
    monkeypatch.setattr(uc, "_dist_present", lambda name: False)
    assert uc.pipeline_extra_installed() is False


# ----------------- Windows launcher rename (WinError 32) ----------------- #
# A running gui-scripts launcher (Scripts\mlgidlab.exe) is locked on
# Windows, so pip's uninstall of the old version dies with WinError 32.
# free_locked_launchers renames it aside first; restore_launchers puts it
# back if pip fails. The path logic is OS-independent, so these run
# everywhere by forcing platform="nt" against a tmp scripts dir.
def test_free_locked_launchers_noop_off_windows(tmp_path):
    exe = tmp_path / "mlgidlab.exe"
    exe.write_bytes(b"launcher")
    assert uc.free_locked_launchers(
        platform="posix", scripts_dir=tmp_path
    ) == []
    assert exe.exists()


def test_free_locked_launchers_renames_exe_aside(tmp_path):
    exe = tmp_path / "mlgidlab.exe"
    exe.write_bytes(b"launcher")
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    backup = tmp_path / "mlgidlab.exe.old"
    assert renames == [(exe, backup)]
    assert not exe.exists() and backup.read_bytes() == b"launcher"
    # No launcher at all (e.g. console_scripts removed by hand) -> no-op.
    assert uc.free_locked_launchers(
        platform="nt", scripts_dir=tmp_path
    ) == []


def test_free_locked_launchers_sweeps_stale_backups(tmp_path):
    exe = tmp_path / "mlgidlab.exe"
    exe.write_bytes(b"new")
    (tmp_path / "mlgidlab.exe.old").write_bytes(b"stale")
    (tmp_path / "mlgidlab.exe.old2").write_bytes(b"staler")
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    # Both leftovers deleted, so the rename lands on plain ".old" again.
    assert renames == [(exe, tmp_path / "mlgidlab.exe.old")]
    assert (tmp_path / "mlgidlab.exe.old").read_bytes() == b"new"
    assert not (tmp_path / "mlgidlab.exe.old2").exists()


def test_free_locked_launchers_skips_undeletable_backup(
    tmp_path, monkeypatch
):
    # A backup from the PREVIOUS update whose process is still running
    # can be neither deleted nor overwritten; the rename must land on
    # the next free name instead.
    from pathlib import Path

    exe = tmp_path / "mlgidlab.exe"
    exe.write_bytes(b"new")
    locked = tmp_path / "mlgidlab.exe.old"
    locked.write_bytes(b"still running")
    monkeypatch.setattr(
        Path, "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked")),
    )
    renames = uc.free_locked_launchers(platform="nt", scripts_dir=tmp_path)
    assert renames == [(exe, tmp_path / "mlgidlab.exe.old2")]
    assert locked.read_bytes() == b"still running"
    assert (tmp_path / "mlgidlab.exe.old2").read_bytes() == b"new"


def test_restore_launchers(tmp_path):
    exe = tmp_path / "mlgidlab.exe"
    backup = tmp_path / "mlgidlab.exe.old"
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
    from mlgidlab.main_window import _UpdateInstallWorker

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


def _worker_with_fake_pip(monkeypatch, returncodes, *, raise_on=None):
    """An ``_UpdateInstallWorker`` whose ``subprocess.run`` answers from
    ``returncodes`` (one per call) and records the commands it was given.
    ``raise_on`` is a call index that raises instead of returning."""
    import subprocess
    import types

    ran: list = []
    restored: list = []
    renames = [("orig", "backup")]
    monkeypatch.setattr(uc, "free_locked_launchers", lambda: renames)
    monkeypatch.setattr(uc, "restore_launchers", restored.append)

    def _fake_run(cmd, **kwargs):
        n = len(ran)
        ran.append(cmd)
        if raise_on == n:
            raise OSError("cannot launch")
        return types.SimpleNamespace(
            returncode=returncodes[n], stdout=f"out{n}\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return ran, restored, renames


def _run_install_worker(commands):
    from mlgidlab.update_ui import _UpdateInstallWorker

    results: list = []
    worker = _UpdateInstallWorker(commands)
    worker.finished.connect(lambda rc, out: results.append((rc, out)))
    worker.run()
    return results


def test_install_worker_runs_the_commands_in_order(qapp, monkeypatch):
    ran, restored, _ = _worker_with_fake_pip(monkeypatch, [0, 0])
    results = _run_install_worker([["pip", "install", "new"], ["pip", "uninstall"]])
    assert ran == [["pip", "install", "new"], ["pip", "uninstall"]]
    # One result for the whole update, pip's output from both steps, and
    # nothing to put back: both steps worked.
    assert results == [(0, "out0\nout1\n")]
    assert restored == []


def test_install_worker_stops_when_the_install_fails(qapp, monkeypatch):
    ran, restored, renames = _worker_with_fake_pip(monkeypatch, [1, 0])
    results = _run_install_worker([["pip", "install", "new"], ["pip", "uninstall"]])
    # The uninstall must never run after a failed install: the old app is
    # the only working copy left.
    assert ran == [["pip", "install", "new"]]
    assert results == [(1, "out0\n")]
    assert restored == [renames]


def test_install_worker_failed_cleanup_does_not_fail_the_update(
    qapp, monkeypatch, tmp_path
):
    from mlgidlab import diagnostics

    monkeypatch.setenv(diagnostics.LOG_DIR_ENV, str(tmp_path))
    ran, restored, renames = _worker_with_fake_pip(monkeypatch, [0, 1])
    results = _run_install_worker([["pip", "install", "new"], ["pip", "uninstall"]])
    assert len(ran) == 2
    # The new app is installed, so the update succeeded; the leftover is
    # reported, and the old launcher is put back because it still belongs
    # to a distribution that is still installed.
    (rc, out), = results
    assert rc == 0
    assert out.startswith("out0\nout1\n")
    assert "mlgidlab could not be removed (exit 1)" in out
    assert restored == [renames]
    # Each pip call gets its own started/finished block in the update log.
    text = (tmp_path / diagnostics.UPDATE_LOG_NAME).read_text(encoding="utf-8")
    assert text.count("=== update started") == 2
    assert text.count("=== update finished") == 2
    assert "return code 1 ===" in text


def test_install_worker_cleanup_that_cannot_launch_is_not_fatal(
    qapp, monkeypatch
):
    ran, _, _ = _worker_with_fake_pip(monkeypatch, [0, 0], raise_on=1)
    results = _run_install_worker([["pip", "install", "new"], ["pip", "uninstall"]])
    (rc, out), = results
    assert rc == 0 and "could not be removed" in out
    assert len(ran) == 2


def test_install_worker_install_that_cannot_launch_fails(qapp, monkeypatch):
    ran, restored, renames = _worker_with_fake_pip(
        monkeypatch, [0, 0], raise_on=0
    )
    results = _run_install_worker([["pip", "install", "new"], ["pip", "uninstall"]])
    assert results == [(1, "Failed to launch pip: cannot launch")]
    assert len(ran) == 1 and restored == [renames]


# --------------------------- MainWindow wiring --------------------------- #
def test_banner_shown_only_when_outdated(main_window):
    from mlgidlab import __version__ as current

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
    from mlgidlab import __version__ as current
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


# --------------- MainWindow wiring: the move to the renamed app --------------- #
def test_display_version_strips_the_tag_prefix():
    assert uc.display_version("v1.0.0") == "1.0.0"
    assert uc.display_version("v1.0.0rc1") == "1.0.0rc1"
    assert uc.display_version("garbage") == "garbage"
    assert uc.display_version(None) == ""


def test_banner_names_the_rename_only_for_the_renamed_release(main_window):
    banner = main_window._update_banner
    banner.show_update("0.1.0a21", "v0.1.0a22", "http://x", can_install=True)
    assert "A newer version of mlgidLAB" in banner._label.text()
    banner.show_update("0.1.0a21", "v1.0.0", "http://x", can_install=True)
    text = banner._label.text()
    assert "mlgidLAB is now called mlgidGUI" in text and "mlgidGUI 1.0.0" in text
    assert "remove mlgidLAB" in banner._update_btn.toolTip()
    # Back to an alpha: the plain tooltip returns.
    banner.show_update("0.1.0a21", "v0.1.0a22", "http://x", can_install=True)
    assert "remove mlgidLAB" not in banner._update_btn.toolTip()


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


def test_confirm_dialog_explains_the_move_for_the_renamed_release(
    main_window, monkeypatch
):
    title, text = _confirm_texts(main_window, monkeypatch, "v1.0.0")
    assert title == "Move to mlgidGUI"
    assert "mlgidGUI 1.0.0" in text and "remove mlgidLAB" in text
    assert "installed first" in text and "data files are not touched" in text
    # An alpha keeps the old wording.
    title, text = _confirm_texts(main_window, monkeypatch, "v0.1.0a22")
    assert title == "Update mlgidLAB"
    assert "v0.1.0a22" in text and "pip-install" in text


def _start_install_with_stubs(main_window, monkeypatch, tag):
    """Run ``_start_update_install(tag)`` with a stub thread and worker, so
    nothing actually runs (real thread teardown is covered by
    ``test_update_threads_survive_an_instant_worker``).
    Returns ``(commands_seen, progress_calls)``."""
    import mlgidlab.main_window_update as mwu

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


def test_start_install_hands_the_worker_the_whole_command_list(
    main_window, monkeypatch
):
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    seen, progress = _start_install_with_stubs(main_window, monkeypatch, "v1.0.0")
    (commands,) = seen
    assert [c[3] for c in commands] == ["install", "uninstall"]
    assert commands[0][-1] == "mlgidgui==1.0.0"
    assert progress == [("Moving to mlgidGUI", "Installing mlgidGUI 1.0.0…")]

    seen, progress = _start_install_with_stubs(
        main_window, monkeypatch, "v0.1.0a22"
    )
    (commands,) = seen
    assert len(commands) == 1 and commands[0][-1].endswith("@v0.1.0a22")
    assert progress == [("Updating mlgidLAB", "Updating mlgidLAB to v0.1.0a22…")]


# Child-process body for the thread-teardown test below: real QThreads,
# workers that finish the moment they start, many start/finish cycles.
_INSTANT_WORKERS_SCRIPT = r'''
import os
import sys
import time

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

import mlgidlab.main_window_update as mwu


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


def test_failure_dialog_lists_every_manual_command(main_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: shown.append(self.informativeText())
    )
    main_window._latest_tag = "v1.0.0"
    main_window._on_update_install_finished(1, "pip exploded")
    lines = shown[0].splitlines()
    assert any(line.endswith("pip install mlgidgui==1.0.0") for line in lines)
    assert any(line.endswith("pip uninstall -y mlgidlab") for line in lines)


@pytest.mark.parametrize(
    "tag, module, wording",
    [
        ("v1.0.0", "mlgidgui", "mlgidGUI 1.0.0 was installed"),
        ("v0.1.0a22", "mlgidlab", "mlgidLAB was updated"),
    ],
)
def test_restart_relaunches_the_installed_apps_module(
    main_window, monkeypatch, tmp_path, tag, module, wording
):
    import sys

    from PySide6.QtWidgets import QMessageBox

    import mlgidlab.main_window_update as mwu
    from mlgidlab import diagnostics

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
    main_window._latest_tag = tag
    main_window._prompt_restart()
    assert wording in asked[0]
    assert launched == [(sys.executable, ["-m", module])]
    text = (tmp_path / diagnostics.UPDATE_LOG_NAME).read_text("utf-8")
    assert f"-m {module} ===" in text


# ------------- one-time notice: mlgidLAB is becoming mlgidGUI ------------- #
@pytest.fixture
def rename_notice_unseen(main_window):
    """A profile that has not seen the notice, and one that is left clean."""
    from PySide6.QtCore import QSettings

    QSettings().remove(main_window._RENAME_NOTICE_KEY)
    yield
    QSettings().remove(main_window._RENAME_NOTICE_KEY)


def _record_notice_dialogs(monkeypatch, press=None):
    """Replace ``QMessageBox.exec``: record the box's texts and, when
    ``press`` names a button, click it. Returns the list of records."""
    from PySide6.QtWidgets import QMessageBox

    shown: list = []

    def _exec(self):
        shown.append((self.windowTitle(), self.text(), self.informativeText()))
        if press is not None:
            next(b for b in self.buttons() if b.text() == press).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _exec)
    return shown


def test_rename_notice_shows_once_per_profile(
    main_window, rename_notice_unseen, monkeypatch
):
    from PySide6.QtCore import QSettings

    shown = _record_notice_dialogs(monkeypatch)
    main_window._maybe_show_rename_notice()
    assert len(shown) == 1
    assert str(QSettings().value(main_window._RENAME_NOTICE_KEY)).lower() == "true"
    # Never again on later launches.
    main_window._maybe_show_rename_notice()
    assert len(shown) == 1


def test_rename_notice_is_recorded_even_if_the_dialog_fails(
    main_window, rename_notice_unseen, monkeypatch
):
    from PySide6.QtCore import QSettings

    def _boom():
        raise RuntimeError("dialog failed")

    monkeypatch.setattr(main_window, "_show_rename_notice", _boom)
    with pytest.raises(RuntimeError):
        main_window._maybe_show_rename_notice()
    # A failing dialog must not come back on every launch.
    assert str(QSettings().value(main_window._RENAME_NOTICE_KEY)).lower() == "true"


def test_rename_notice_says_what_to_do_and_what_changes(
    main_window, monkeypatch
):
    shown = _record_notice_dialogs(monkeypatch)
    main_window._show_rename_notice()
    (title, text, info), = shown
    assert title == "mlgidLAB is becoming mlgidGUI"
    assert "1.0.0" in text and "same application" in text
    assert "nothing now" in info and "Update now" in info
    assert "mlgidgui instead of mlgidlab" in info
    assert "no compatibility layer" in info


def test_rename_notice_more_information_opens_the_releases_page(
    main_window, monkeypatch
):
    import mlgidlab.main_window_update as mwu

    opened: list = []
    monkeypatch.setattr(
        mwu.QDesktopServices, "openUrl", lambda url: opened.append(url.toString())
    )
    _record_notice_dialogs(monkeypatch, press="More information")
    main_window._show_rename_notice()
    assert opened == [uc.RELEASES_PAGE]
    # OK just closes it.
    opened.clear()
    _record_notice_dialogs(monkeypatch, press="OK")
    main_window._show_rename_notice()
    assert opened == []


def test_rename_notice_is_reachable_from_the_help_menu(
    main_window, monkeypatch
):
    shown = _record_notice_dialogs(monkeypatch)
    action = main_window.action_rename_notice
    assert "mlgidGUI" in action.text()
    action.trigger()
    assert len(shown) == 1


def test_startup_checks_show_the_notice_before_the_network_check(
    main_window, monkeypatch
):
    order: list[str] = []
    monkeypatch.setattr(
        main_window, "_maybe_show_changelog", lambda: order.append("changelog")
    )
    monkeypatch.setattr(
        main_window, "_maybe_show_rename_notice", lambda: order.append("notice")
    )
    monkeypatch.setattr(
        main_window, "_start_update_check", lambda **k: order.append("check")
    )
    main_window._run_startup_update_checks()
    assert order == ["changelog", "notice", "check"]


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
    the interpreter runs ``Scripts\\mlgidlab.exe``; ``sys.path`` and the
    metadata caches are restored afterwards."""
    import importlib.metadata
    import sys
    import zipfile

    exe = tmp_path / "mlgidlab.exe"
    with zipfile.ZipFile(exe, "w") as z:
        z.writestr("__main__.py", "")
    monkeypatch.setattr(sys, "path", [str(exe)] + list(sys.path))
    yield exe
    importlib.metadata.FastPath.__new__.cache_clear()


def test_drop_launcher_from_sys_path_only_removes_executables(monkeypatch):
    import sys

    keep = ["/a/site-packages", "/b/lib.zip", "", "/c/tools"]
    monkeypatch.setattr(
        sys, "path", ["C:\\v\\Scripts\\mlgidlab.exe"] + keep + ["D:\\x\\Y.EXE"]
    )
    removed = uc.drop_launcher_from_sys_path()
    assert removed == ["C:\\v\\Scripts\\mlgidlab.exe", "D:\\x\\Y.EXE"]
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
    aside = launcher_zip.with_name("mlgidlab.exe.old")
    launcher_zip.rename(aside)  # a held handle makes this a PermissionError
    assert aside.exists() and not launcher_zip.exists()


def test_free_locked_launchers_releases_our_own_handle_first(
    tmp_path, monkeypatch
):
    order: list[str] = []
    monkeypatch.setattr(
        uc, "release_launcher_handles", lambda: order.append("release")
    )
    exe = tmp_path / "mlgidlab.exe"
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

    import mlgidlab

    source = inspect.getsource(mlgidlab.main)
    drop = source.index("drop_launcher_from_sys_path")
    assert drop < source.index("from PySide6")
