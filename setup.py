"""Build hook: ship the repo's ``CHANGELOG.md`` inside the wheel.

The project is configured in ``pyproject.toml``; this file only adds one
build step. ``CHANGELOG.md`` lives at the repository root (where GitHub
readers and the release checklist expect it), and package data cannot
reach outside the package directory, so the wheel built from a checkout
or an sdist would not contain it and the post-update "What's new" dialog
(``update_check.whats_new``) never found anything to show. The step
copies the file into ``mlgidgui/`` of the build tree only; the source
tree keeps a single copy.
"""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class _BuildPyWithChangelog(build_py):
    def run(self):
        super().run()
        src = Path(__file__).resolve().parent / "CHANGELOG.md"
        if src.is_file() and not self.dry_run:
            dest = Path(self.build_lib) / "mlgidgui"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest / "CHANGELOG.md")


setup(cmdclass={"build_py": _BuildPyWithChangelog})
