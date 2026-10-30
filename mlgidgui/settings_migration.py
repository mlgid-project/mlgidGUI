"""One-time copy of the settings the mlgidLAB alphas stored.

QSettings keys its store on the organisation and application name, and both
changed with the rename (``mlgidLAB`` -> ``mlgidGUI``). Without a copy a
user who updates would silently lose the recent-files list, the dock
layout, the theme, the pen colours and the "last version seen" value.

On the first start the new app copies every key of the old store into its
own. A key the new store already has is kept, so running this twice, or
after the user changed something, never overwrites anything. The old store
is left untouched: an mlgidLAB alpha installed next to the new app keeps
working, and nothing is lost if the user goes back. A marker key in the new
store records that the copy ran, so later starts do not look at the old
store again.

pyFAI's calibration dialog keeps its own ini file (``<org>/pyFAI-calib``,
see ``calibration_dialog``); that one moves with the same rule.

The location of a store is Qt's: on Linux ``~/.config/<org>/<app>.conf``
(``$XDG_CONFIG_HOME`` respected), on Windows the registry key
``HKCU\\Software\\<org>\\<app>``, on macOS the ``.plist`` for the org and
app. Tests point ``QSettings.setPath`` at a temp directory.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSettings

logger = logging.getLogger(__name__)

LEGACY_ORGANIZATION = "mlgidLAB"
LEGACY_APPLICATION = "mlgidLAB"
ORGANIZATION = "mlgidGUI"
APPLICATION = "mlgidGUI"
#: The pyFAI calibration dialog's settings file name (without extension).
CALIBRATION_APPLICATION = "pyFAI-calib"

#: Present in the new store once the copy has run.
MIGRATED_KEY = "settingsMigratedFromMlgidLab"


def _copy(source: QSettings, target: QSettings) -> int:
    """Copy every key ``target`` lacks; return how many were copied."""
    copied = 0
    for key in source.allKeys():
        if target.contains(key):
            continue
        target.setValue(key, source.value(key))
        copied += 1
    target.sync()
    return copied


def calibration_settings(organization: str = ORGANIZATION) -> QSettings:
    """The ini store pyFAI's calibration widgets use (shared with the dialog)."""
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        organization,
        CALIBRATION_APPLICATION,
        None,
    )


def migrate_legacy_settings() -> int | None:
    """Copy the mlgidLAB settings into the mlgidGUI store, once.

    Call after ``QApplication.setOrganizationName`` / ``setApplicationName``
    with the new names and before anything reads a setting. Returns the
    number of keys copied, or ``None`` when the copy had already run.
    """
    current = QSettings()
    if current.contains(MIGRATED_KEY):
        return None
    copied = _copy(QSettings(LEGACY_ORGANIZATION, LEGACY_APPLICATION), current)
    copied += _copy(calibration_settings(LEGACY_ORGANIZATION),
                    calibration_settings())
    current.setValue(MIGRATED_KEY, True)
    current.sync()
    if copied:
        logger.info("Copied %d setting(s) from the mlgidLAB profile.", copied)
    return copied
