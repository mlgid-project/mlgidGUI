"""Theme switch persistence via QSettings.

``MainWindow.__init__`` does not re-read the persisted theme — only
``mlgidgui.__init__.main`` does (it also sets the org/app name). So
this asserts only what ``_set_theme`` guarantees (main_window.py:
1149-1178): invalid clamps to "dark", ``_current_theme`` updates, and
``QSettings().setValue("theme", ...)`` round-trips (the same key
``main()`` reads). XDG_CONFIG_HOME is redirected by conftest so the
store is a throwaway temp dir.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.gui


def _settings_theme():
    s = QSettings()
    s.sync()
    return s.value("theme")


def test_set_theme_persists_and_clamps(main_window):
    # Mirror production QSettings resolution (set in mlgidgui.main).
    app = QApplication.instance()
    app.setOrganizationName("mlgidGUI")
    app.setApplicationName("mlgidGUI")

    main_window._set_theme("light")
    assert main_window._current_theme == "light"
    assert _settings_theme() == "light"

    main_window._set_theme("dark")
    assert main_window._current_theme == "dark"
    assert _settings_theme() == "dark"

    # Anything not in {dark, light} clamps to dark.
    main_window._set_theme("bogus")
    assert main_window._current_theme == "dark"
    assert _settings_theme() == "dark"
