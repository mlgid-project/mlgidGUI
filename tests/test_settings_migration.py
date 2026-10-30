"""The settings the mlgidLAB alphas stored move to the mlgidGUI store, once.

QSettings keys on organisation and application name, and both changed with
the rename. ``settings_migration.migrate_legacy_settings`` copies the old
store into the new one on the first start. These tests point Qt's settings
path at a temp directory (``QSettings.setPath``) and use the real names, so
what they exercise is the production lookup, not a stand-in.
"""
from __future__ import annotations

import os

import pytest
from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QApplication

from mlgidgui import settings_migration as sm

pytestmark = pytest.mark.gui


@pytest.fixture
def profile(qapp, tmp_path):
    """A fresh settings root, with the app named as ``main`` names it."""
    previous = (qapp.organizationName(), qapp.applicationName())
    root = os.environ.get("XDG_CONFIG_HOME", "")
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(tmp_path))
    qapp.setOrganizationName(sm.ORGANIZATION)
    qapp.setApplicationName(sm.APPLICATION)
    yield tmp_path
    qapp.setOrganizationName(previous[0])
    qapp.setApplicationName(previous[1])
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, root)


def _legacy() -> QSettings:
    return QSettings(sm.LEGACY_ORGANIZATION, sm.LEGACY_APPLICATION)


def test_the_old_settings_arrive_in_the_new_store(profile):
    old = _legacy()
    old.setValue("theme", "light")
    old.setValue("recentFiles", '["/data/a.h5"]')
    old.setValue("windowState", QByteArray(b"\x00\x01layout"))
    old.sync()

    assert sm.migrate_legacy_settings() == 3

    new = QSettings()
    assert new.value("theme") == "light"
    assert new.value("recentFiles") == '["/data/a.h5"]'
    assert bytes(new.value("windowState")) == b"\x00\x01layout"


def test_it_runs_once_and_never_overwrites(profile):
    old = _legacy()
    old.setValue("theme", "light")
    old.sync()
    assert sm.migrate_legacy_settings() == 1

    # The user switches theme in the new app; the old store changes too.
    QSettings().setValue("theme", "dark")
    old.setValue("theme", "light")
    old.setValue("later", "x")
    old.sync()

    assert sm.migrate_legacy_settings() is None
    assert QSettings().value("theme") == "dark"
    assert not QSettings().contains("later")


def test_a_key_the_new_store_already_has_is_kept(profile):
    QSettings().setValue("theme", "dark")
    QSettings().sync()
    old = _legacy()
    old.setValue("theme", "light")
    old.setValue("rail", "false")
    old.sync()

    assert sm.migrate_legacy_settings() == 1
    assert QSettings().value("theme") == "dark"
    assert QSettings().value("rail") == "false"


def test_a_fresh_profile_just_gets_the_marker(profile):
    assert sm.migrate_legacy_settings() == 0
    assert QSettings().contains(sm.MIGRATED_KEY)
    assert sm.migrate_legacy_settings() is None


def test_the_old_store_is_left_as_it_was(profile):
    old = _legacy()
    old.setValue("theme", "light")
    old.sync()
    sm.migrate_legacy_settings()
    assert _legacy().value("theme") == "light"
    assert not _legacy().contains(sm.MIGRATED_KEY)


def test_the_pyfai_calibration_settings_move_too(profile):
    old = sm.calibration_settings(sm.LEGACY_ORGANIZATION)
    old.setValue("calibrant/last", "AgBh")
    old.sync()

    sm.migrate_legacy_settings()

    assert sm.calibration_settings().value("calibrant/last") == "AgBh"


def test_the_stores_live_under_the_new_names(profile):
    sm.migrate_legacy_settings()
    QSettings().sync()
    names = sorted(p.name for p in profile.iterdir())
    assert "mlgidGUI" in names
    assert QApplication.instance().organizationName() == "mlgidGUI"


def test_main_migrates_after_naming_the_app_and_before_reading_a_setting():
    """``main`` needs a display, so read the order from its source: the new
    names first (they pick the store), then the copy, then the first read."""
    import inspect

    import mlgidgui

    src = inspect.getsource(mlgidgui.main)
    named = src.index('app.setApplicationName("mlgidGUI")')
    copied = src.index("migrate_legacy_settings()")
    first_read = src.index('QSettings().value("theme"')
    assert named < copied < first_read
