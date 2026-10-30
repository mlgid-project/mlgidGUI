"""Every picker in the app starts where the last one finished.

The alpha-test complaint this locks down: opening a scan, then
picking a mask, then saving a PONI used to walk through three
unrelated starting directories. ``mlgidgui.file_dialogs`` is now the
only way a dialog is built, so the rule is testable in one place --
plus a source guard so the next Browse button cannot quietly
reintroduce its own ``QFileDialog``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QFileDialog

from mlgidgui import file_dialogs


@pytest.fixture(autouse=True)
def _clean_last_dir():
    """The shared directory is persisted, so it must not leak between tests."""
    settings = QSettings()
    for key in (file_dialogs.SETTINGS_KEY, file_dialogs.LEGACY_KEY):
        settings.remove(key)
    settings.sync()
    yield
    for key in (file_dialogs.SETTINGS_KEY, file_dialogs.LEGACY_KEY):
        settings.remove(key)
    settings.sync()


@pytest.fixture
def picker(monkeypatch, qtbot):
    """Drive the shared pickers without showing a dialog.

    Records the directory each dialog opened in, and answers with the
    next queued path (``None`` cancels), the same way
    ``test_fabio_gui`` fakes the Open dialog.
    """

    class Picker:
        def __init__(self) -> None:
            self.opened: list[str] = []
            self.answers: list = []

        def answer(self, *paths) -> None:
            self.answers.extend(paths)

    state = Picker()

    def fake_exec(dlg):
        state.opened.append(dlg.directory().absolutePath())
        answer = state.answers.pop(0) if state.answers else None
        if answer is None:
            return 0
        dlg.selectFile(str(answer))
        return 1

    monkeypatch.setattr(QFileDialog, "exec", fake_exec)
    return state


# --- The stored directory --------------------------------------------


def test_last_dir_defaults_to_home():
    assert file_dialogs.last_dir() == str(Path.home())


def test_last_dir_skips_a_directory_that_disappeared(tmp_path):
    gone = tmp_path / "unmounted"
    gone.mkdir()
    file_dialogs.remember(gone)
    gone.rmdir()
    # Handing Qt a dead path makes it fall back to the *working*
    # directory, which is what this module exists to avoid.
    assert file_dialogs.last_dir() == str(Path.home())


def test_a_legacy_open_dir_is_still_honoured(tmp_path):
    """Upgrading users keep the directory the old Open dialog stored."""
    QSettings().setValue(file_dialogs.LEGACY_KEY, str(tmp_path))
    assert file_dialogs.last_dir() == str(tmp_path)


def test_remember_takes_a_file_or_a_directory(tmp_path):
    target = tmp_path / "scan.h5"
    target.write_bytes(b"")
    file_dialogs.remember(target)
    assert file_dialogs.last_dir() == str(tmp_path)

    other = tmp_path / "sub"
    other.mkdir()
    file_dialogs.remember(other)
    assert file_dialogs.last_dir() == str(other)


def test_remember_ignores_a_path_that_leads_nowhere(tmp_path):
    file_dialogs.remember(tmp_path)
    file_dialogs.remember("")
    file_dialogs.remember(tmp_path / "no" / "such" / "dir" / "f.h5")
    assert file_dialogs.last_dir() == str(tmp_path)


def test_start_at_keeps_only_the_name_of_a_suggested_path(tmp_path):
    """A caller's absolute path contributes its basename, nothing more.

    Honouring one caller's directory is precisely how the starting
    directory used to jump around between buttons.
    """
    file_dialogs.remember(tmp_path)
    suggested = "/somewhere/else/entirely/peak_tracking.png"
    assert file_dialogs.start_at(suggested) == str(
        tmp_path / "peak_tracking.png"
    )
    assert file_dialogs.start_at("") == str(tmp_path)


# --- The pickers ------------------------------------------------------


def test_every_picker_opens_where_the_last_one_ended(picker, tmp_path):
    """The headline claim, walked across all four picker kinds."""
    first = tmp_path / "beamtime"
    second = tmp_path / "results"
    third = tmp_path / "cifs"
    for d in (first, second, third):
        d.mkdir()
    scan = first / "scan.h5"
    scan.write_bytes(b"")
    mask = first / "mask.npy"
    mask.write_bytes(b"")

    picker.answer(scan)
    assert file_dialogs.open_file(None, "Open") == str(scan)

    # ...the mask picker now starts in the scan's directory...
    picker.answer(mask)
    assert file_dialogs.open_files(None, "Mask") == [str(mask)]
    assert picker.opened[-1] == str(first)

    # ...saving somewhere else moves the shared directory with it...
    picker.answer(second / "out.poni")
    path, _filter = file_dialogs.save_file(None, "Save PONI")
    assert path == str(second / "out.poni")

    # ...and the next picker of any kind follows.
    picker.answer(third)
    assert file_dialogs.existing_directory(None, "CIF folder") == str(third)
    assert picker.opened[-1] == str(second)


def test_a_cancelled_picker_changes_nothing(picker, tmp_path):
    file_dialogs.remember(tmp_path)
    picker.answer(None)
    assert file_dialogs.open_file(None, "Open") == ""
    assert file_dialogs.last_dir() == str(tmp_path)


def test_a_suggested_name_survives_but_its_directory_does_not(
    picker, tmp_path
):
    here = tmp_path / "here"
    here.mkdir()
    file_dialogs.remember(here)
    picker.answer(None)
    file_dialogs.save_file(
        None, "Save As", suggested_name="/elsewhere/original.h5",
    )
    assert picker.opened[-1] == str(here)


def test_pickers_are_non_native_and_use_the_fast_icon_provider(
    monkeypatch, qtbot, tmp_path
):
    """Now that every picker starts in the last-used directory, any of
    them can land in a folder of thousands of detector images -- the
    case the constant-time provider was written for."""
    from mlgidgui.browser_widgets import _FastFileIconProvider

    seen: dict = {}

    def fake_exec(dlg):
        seen["non_native"] = dlg.testOption(
            QFileDialog.Option.DontUseNativeDialog
        )
        seen["provider"] = dlg.iconProvider()
        return 0

    monkeypatch.setattr(QFileDialog, "exec", fake_exec)
    file_dialogs.open_file(None, "Open")

    assert seen["non_native"] is True
    assert isinstance(seen["provider"], _FastFileIconProvider)


# --- Dialogs built elsewhere -----------------------------------------


def test_adopt_seeds_and_records_a_foreign_dialog(qtbot, tmp_path):
    """pyFAI builds its own pickers; ``adopt`` is how they join in."""
    start = tmp_path / "start"
    end = tmp_path / "end"
    for d in (start, end):
        d.mkdir()
    picked = end / "calibration.poni"
    picked.write_bytes(b"")
    file_dialogs.remember(start)

    dlg = QFileDialog()
    qtbot.addWidget(dlg)
    file_dialogs.adopt(dlg)
    assert dlg.directory().absolutePath() == str(start)
    assert dlg.testOption(QFileDialog.Option.DontUseNativeDialog) is True

    dlg.selectFile(str(picked))
    dlg.done(QDialog.DialogCode.Accepted)
    assert file_dialogs.last_dir() == str(end)


def test_adopt_ignores_a_rejected_dialog(qtbot, tmp_path):
    start = tmp_path / "start"
    start.mkdir()
    file_dialogs.remember(start)

    dlg = QFileDialog()
    qtbot.addWidget(dlg)
    file_dialogs.adopt(dlg)
    dlg.setDirectory(str(tmp_path))
    dlg.done(QDialog.DialogCode.Rejected)
    assert file_dialogs.last_dir() == str(start)


def test_adopt_survives_a_dialog_that_does_not_cooperate():
    """A third-party dialog must still open if it answers nothing."""

    class Awkward:
        pass

    file_dialogs.adopt(Awkward())  # must not raise


# --- The guard --------------------------------------------------------

_PICKER_CALLS = (
    "QFileDialog.getOpenFileName",
    "QFileDialog.getOpenFileNames",
    "QFileDialog.getSaveFileName",
    "QFileDialog.getExistingDirectory",
    "QFileDialog(",
)


def test_no_module_builds_its_own_file_picker():
    """One shared directory only holds if there is one shared builder.

    A new Browse button that reaches for ``QFileDialog`` directly gets
    its own starting directory back, which is the bug this whole
    module exists to remove -- so the source is checked instead of
    trusting review.
    """
    package = Path(file_dialogs.__file__).parent
    offenders: list[str] = []
    for module in sorted(package.glob("*.py")):
        if module.name == "file_dialogs.py":
            continue
        text = module.read_text()
        for call in _PICKER_CALLS:
            if call in text:
                offenders.append(f"{module.name}: {call}")
    assert not offenders, (
        "these build a picker outside mlgidgui.file_dialogs, so they get "
        f"their own starting directory: {offenders}"
    )
