"""Build the desktop app and its installer for the computer this runs on.

    uv sync --extra gui --group build
    uv run python -m installers.build              # build, self-test, package
    uv run python -m installers.build --no-package # only dist/arabic-ocr/

Steps:
1. PyInstaller (arabic-ocr.spec) puts Python, the libraries, the code and the two
   models in one folder: dist/arabic-ocr/ (macOS: dist/Arabic OCR.app).
2. Self-test: runs the built programs (not the source code) on the sample image:
   `arabic-ocr-gui --self-test` and `arabic-ocr sample.png`.
3. Package it as the usual installer for the OS:
     Windows  dist/arabic-ocr-<version>-windows-x64-setup.exe  (Inno Setup 6)
     macOS    dist/arabic-ocr-<version>-macos-arm64.dmg        (hdiutil)
     Linux    dist/arabic-ocr_<version>_amd64.deb              (dpkg-deb)
   and check its size against the PRD's 250 MB limit (NFR-6).

Takes 2-5 minutes; needs no GPU. A build only works on the OS it was made on,
which is why GitHub Actions builds on all three (.github/workflows/release.yml).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WORK = ROOT / "build" / "pyinstaller"
HERE = ROOT / "installers"
SAMPLE = ROOT / "src" / "arabic_ocr" / "gui" / "sample.png"
MAX_MB = 250  # PRD NFR-6
VERSION = version("arabic-ocr")


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# ----------------------------------------------------------------------- build
def pyinstaller() -> None:
    import PyInstaller.__main__

    for old in (DIST / "arabic-ocr", DIST / "Arabic OCR.app"):
        shutil.rmtree(old, ignore_errors=True)
    PyInstaller.__main__.run(
        [str(HERE / "arabic-ocr.spec"), "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(WORK)]
    )  # fmt: skip


def app_dir() -> Path:
    """The folder that holds the two programs."""
    if sys.platform == "darwin":
        return DIST / "Arabic OCR.app" / "Contents" / "MacOS"
    return DIST / "arabic-ocr"


def program(name: str, folder: Path | None = None) -> Path:
    return (folder or app_dir()) / (name + (".exe" if sys.platform == "win32" else ""))


# ------------------------------------------------------------------- self-test
def self_test(folder: Path | None = None) -> None:
    """Run the built programs. Raises if anything fails."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The built app must not lean on this computer's Python or venv.
    for var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(var, None)
    cli = program("arabic-ocr", folder)
    out = run([cli, "--version"], env=env, capture_output=True, text=True).stdout
    print(out.strip())
    text = run(
        [cli, SAMPLE], env=env, capture_output=True, encoding="utf-8", errors="replace"
    ).stdout
    print(text.strip())
    if "العربية" not in text:
        raise SystemExit("CLI self-test failed: the sample text was not read")

    serve_test(cli, env)

    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.txt"
        code = subprocess.run(
            [str(program("arabic-ocr-gui", folder)), "--self-test", str(report)],
            env=env, timeout=600,
        ).returncode  # fmt: skip
        print(report.read_text(encoding="utf-8") if report.exists() else "(no report)")
        if code != 0:
            raise SystemExit(f"GUI self-test failed (exit code {code})")


def serve_test(cli: Path, env: dict) -> None:
    """Start `arabic-ocr serve`, upload the sample, check the JSON, stop it."""
    import json
    import socket
    import time
    import urllib.request
    import uuid

    with socket.socket() as s:  # a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([str(cli), "serve", "--port", str(port)], env=env)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    print("serve /health:", r.read().decode())
                break
            except OSError:
                if server.poll() is not None:
                    raise SystemExit("arabic-ocr serve exited") from None
                time.sleep(0.5)
        boundary = uuid.uuid4().hex
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="s.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        )
        body = head.encode() + SAMPLE.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            url + "/ocr", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(request, timeout=120) as r:
            result = json.loads(r.read().decode("utf-8"))
        if "العربية" not in json.dumps(result, ensure_ascii=False):
            raise SystemExit("serve self-test failed: the sample text was not read")
        print("serve POST /ocr: OK")
    finally:
        server.terminate()
        server.wait(30)


# -------------------------------------------------------------------- packages
def find_iscc() -> Path:
    """Inno Setup's compiler."""
    found = shutil.which("iscc")
    candidates = [Path(found)] if found else []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA", "") + "/Programs"):  # fmt: skip
        if base:
            candidates.append(Path(base) / "Inno Setup 6" / "ISCC.exe")
    if os.environ.get("INNO_SETUP"):
        candidates.insert(0, Path(os.environ["INNO_SETUP"]) / "ISCC.exe")
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit("Inno Setup 6 not found: install it (jrsoftware.org) or set INNO_SETUP")


def package_windows() -> Path:
    run([find_iscc(), f"/DVERSION={VERSION}", f"/DSOURCE={app_dir()}", f"/DOUTPUT={DIST}",
         HERE / "arabic-ocr.iss"])  # fmt: skip
    return DIST / f"arabic-ocr-{VERSION}-windows-x64-setup.exe"


def package_macos() -> Path:
    out = DIST / f"arabic-ocr-{VERSION}-macos-arm64.dmg"
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "Arabic OCR"
        stage.mkdir()
        # symlinks=True keeps the app's internal links (Qt frameworks) intact
        shutil.copytree(DIST / "Arabic OCR.app", stage / "Arabic OCR.app", symlinks=True)
        (stage / "Applications").symlink_to("/Applications")  # drag the app here
        run(["hdiutil", "create", "-volname", "Arabic OCR", "-srcfolder", stage,
             "-ov", "-format", "UDZO", out])  # fmt: skip
    return out


def linux_depends(folder: Path) -> list[str]:
    """The Ubuntu packages that provide the system libraries the app links to.

    The bundle leaves system libraries out (arabic-ocr.spec). Here `ldd` lists
    what every program, library and Qt plugin in it needs from the system, and
    `dpkg -S` names the package each of those files comes from. Those become
    the .deb's dependencies, so `apt install` adds whatever a computer lacks.
    """
    internal = folder / "_internal"
    env = {**os.environ, "LD_LIBRARY_PATH": str(internal)}  # as the app runs
    needed: set[str] = set()
    missing: set[str] = set()
    for f in folder.rglob("*"):
        if f.is_symlink() or not f.is_file() or f.read_bytes()[:4] != b"\x7fELF":
            continue
        out = subprocess.run(["ldd", str(f)], capture_output=True, text=True, env=env).stdout
        for line in out.splitlines():
            name, _, target = line.strip().partition(" => ")
            if target.startswith("not found"):
                missing.add(f"{name} (needed by {f.relative_to(folder)})")
            elif target.startswith("/"):
                path = target.split(" (")[0]
                if not Path(path).resolve().is_relative_to(folder.resolve()):
                    needed.add(path)
    if missing:
        raise SystemExit(
            "Libraries not found on this build machine; install them first:\n  "
            + "\n  ".join(sorted(missing))
        )
    packages = {"libc6"}
    for path in sorted(needed):
        # Ubuntu's /lib is a link to /usr/lib; dpkg knows a file by one of them
        other = path.replace("/usr/lib/", "/lib/", 1)
        if other == path:
            other = path.replace("/lib/", "/usr/lib/", 1)
        for c in (path, other):
            r = subprocess.run(["dpkg", "-S", c], capture_output=True, text=True)
            if r.returncode == 0:
                packages.add(r.stdout.split(":")[0].strip())
                break
        else:
            raise SystemExit(f"No package provides {path}")
    print("Depends:", ", ".join(sorted(packages)))
    return [p if p != "libc6" else "libc6 (>= 2.35)" for p in sorted(packages)]


DESKTOP = """[Desktop Entry]
Type=Application
Name=Arabic OCR
Name[ar]=القارئ العربي
Comment=Read Arabic, Persian and Urdu text from images and PDFs
Comment[ar]=قراءة النصوص العربية والفارسية والأردية من الصور وملفات PDF
Exec=/opt/arabic-ocr/arabic-ocr-gui %F
Icon=arabic-ocr
Terminal=false
Categories=Office;Graphics;OCR;
MimeType=image/png;image/jpeg;image/tiff;image/bmp;image/webp;application/pdf;
"""


def package_linux() -> Path:
    out = DIST / f"arabic-ocr_{VERSION}_amd64.deb"
    depends = linux_depends(app_dir())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pkg"
        shutil.copytree(app_dir(), root / "opt" / "arabic-ocr", symlinks=True)
        bin_dir = root / "usr" / "bin"
        bin_dir.mkdir(parents=True)
        for name in ("arabic-ocr", "arabic-ocr-gui"):
            (bin_dir / name).symlink_to(f"/opt/arabic-ocr/{name}")
        apps = root / "usr" / "share" / "applications"
        apps.mkdir(parents=True)
        (apps / "arabic-ocr.desktop").write_text(DESKTOP, encoding="utf-8")
        icons = root / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
        icons.mkdir(parents=True)
        shutil.copy(HERE / "icons" / "arabic-ocr.png", icons / "arabic-ocr.png")
        size_kb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) // 1024
        control = root / "DEBIAN" / "control"
        control.parent.mkdir()
        control.write_text(
            f"Package: arabic-ocr\nVersion: {VERSION}\nArchitecture: amd64\n"
            "Maintainer: arabic-ocr <https://github.com/youssef-mhmd/arabic-ocr>\n"
            f"Installed-Size: {size_kb}\nDepends: {', '.join(depends)}\n"
            "Section: graphics\nPriority: optional\n"
            "Homepage: https://github.com/youssef-mhmd/arabic-ocr\n"
            "Description: Offline OCR for Arabic-script text\n"
            " Reads Arabic, Persian and Urdu text from images and PDFs: a desktop\n"
            " app (arabic-ocr-gui) and a command-line tool (arabic-ocr).\n",
            encoding="utf-8",
        )
        run(["dpkg-deb", "--build", "--root-owner-group", "-Zxz", root, out])
    return out


def _folder() -> Path:
    return DIST / ("Arabic OCR.app" if sys.platform == "darwin" else "arabic-ocr")


def _folder_mb() -> float:
    files = (f for f in _folder().rglob("*") if f.is_file() and not f.is_symlink())
    return sum(f.stat().st_size for f in files) / 1e6


def report_size(top: int = 20) -> None:
    """Print the app folder's size and its largest files (to see what to trim)."""
    files = sorted(
        (f for f in _folder().rglob("*") if f.is_file() and not f.is_symlink()),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    print(f"App folder: {_folder_mb():.0f} MB. Largest files:")
    lines = [f"{f.stat().st_size / 1e6:7.1f} MB  {f.relative_to(_folder())}" for f in files[:top]]
    print(*lines, sep="\n")
    notice(f"app folder {_folder_mb():.0f} MB", lines)


def notice(title: str, lines: list[str]) -> None:
    """On GitHub Actions: show this on the run's summary page as a notice."""
    if os.environ.get("GITHUB_ACTIONS"):
        body = "%0A".join(line.replace("%", "%25") for line in lines)
        print(f"::notice title={sys.platform} {title}::{body}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-build", action="store_true", help="reuse dist/ from the last build")
    p.add_argument("--no-test", action="store_true", help="skip the self-test")
    p.add_argument("--no-package", action="store_true", help="stop after the app folder")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Arabic in logs

    print(f"arabic-ocr {VERSION} on {sys.platform} {platform.machine()}")
    if not args.no_build:
        pyinstaller()
    report_size()
    if not args.no_test:
        self_test()
    if args.no_package:
        return
    package = {"win32": package_windows, "darwin": package_macos}.get(sys.platform, package_linux)()
    mb = package.stat().st_size / 1e6
    print(f"Installer: {package.name}, {mb:.0f} MB (limit {MAX_MB} MB)")
    print(f"SHA-256: {sha256(package)}")
    notice(f"{package.name}: {mb:.0f} MB", [f"app folder {_folder_mb():.0f} MB"])
    if mb > MAX_MB:
        raise SystemExit(f"Installer is {mb:.0f} MB, over the {MAX_MB} MB limit (PRD NFR-6)")


if __name__ == "__main__":
    main()
