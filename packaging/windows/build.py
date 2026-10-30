"""Build the Windows app: a PyInstaller folder, then an Inno Setup installer.

    python packaging/windows/build.py [--name mlgidGUI] [--module mlgidgui]
                                      [--lite] [--skip-installer]

Run it in an environment where the app is installed NON-editable, with
PyInstaller next to it. On Windows, with Inno Setup installed (``iscc`` on
PATH or in its default folder), it writes the installer too. On any other
platform it stops after the PyInstaller folder, which is enough to check the
spec on a development machine.

The numbers the packaging decision needs (folder size, installer size, build
time, SHA-256) go to ``dist/build_report.json``, to stdout and, in GitHub
Actions, to the job summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIST = ROOT / "dist"
INNO_DEFAULTS = (
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
)


def project_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def file_version(version: str) -> tuple[int, int, int, int]:
    """``0.1.0a23`` -> (0, 1, 0, 23): the four numbers a Windows version
    resource holds (the pre-release number goes last)."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:[a-z]+(\d+))?", version)
    if not match:
        return (0, 0, 0, 0)
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def write_version_resource(name: str, version: str) -> Path:
    """The version resource of the exe (Details tab, SmartScreen's publisher
    line): product and company name and the version."""
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo,
        VarStruct, VSVersionInfo)

    numbers = file_version(version)
    info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", "mlgid-project"),
                StringStruct("FileDescription", name),
                StringStruct("FileVersion", version),
                StringStruct("InternalName", name),
                StringStruct("OriginalFilename", f"{name}.exe"),
                StringStruct("ProductName", name),
                StringStruct("ProductVersion", version),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 0x04B0])]),
        ],
    )
    path = HERE / "version_info.txt"
    path.write_text(str(info), encoding="utf-8")
    return path


def folder_stats(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_iscc() -> str | None:
    found = shutil.which("iscc")
    if found:
        return found
    return next((p for p in INNO_DEFAULTS if Path(p).is_file()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="mlgidGUI", help="app and exe name")
    parser.add_argument("--module", default="mlgidgui", help="importable package")
    parser.add_argument("--lite", action="store_true",
                        help="leave out the pipeline backends")
    parser.add_argument("--skip-installer", action="store_true")
    args = parser.parse_args()

    version = project_version()
    variant = "lite" if args.lite else "full"
    env = {**os.environ, "APP_NAME": args.name, "APP_MODULE": args.module,
           "WITH_PIPELINE": "0" if args.lite else "1"}
    if sys.platform == "win32":
        write_version_resource(args.name, version)

    started = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
         str(HERE / "app.spec")],
        cwd=ROOT, env=env, check=True,
    )
    freeze_seconds = time.monotonic() - started
    app_dir = DIST / args.name
    report: dict = {
        "app": args.name, "version": version, "variant": variant,
        "python": sys.version.split()[0], "platform": sys.platform,
        "folder": folder_stats(app_dir), "freeze_seconds": round(freeze_seconds),
    }

    iscc = find_iscc() if sys.platform == "win32" and not args.skip_installer else None
    if iscc:
        base = f"{args.name}-{version}-windows-x64-{variant}-setup"
        started = time.monotonic()
        subprocess.run(
            [iscc, f"/DAppName={args.name}", f"/DAppVersion={version}",
             f"/DSourceDir={app_dir}", f"/DOutputDir={DIST}", f"/DOutputBase={base}",
             str(HERE / "installer.iss")],
            cwd=HERE, check=True,
        )
        installer = DIST / f"{base}.exe"
        report["installer"] = {
            "file": installer.name, "bytes": installer.stat().st_size,
            "sha256": sha256(installer),
            "seconds": round(time.monotonic() - started),
        }

    (DIST / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    gib = 1024 ** 3
    lines = [
        f"### {args.name} {version} ({variant}) on {sys.platform}",
        f"- folder: {report['folder']['files']} files, "
        f"{report['folder']['bytes'] / gib:.2f} GiB, frozen in {report['freeze_seconds']} s",
    ]
    if "installer" in report:
        inst = report["installer"]
        lines += [
            f"- installer: `{inst['file']}`, {inst['bytes'] / gib:.2f} GiB, "
            f"built in {inst['seconds']} s",
            f"- SHA-256: `{inst['sha256']}`",
        ]
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
