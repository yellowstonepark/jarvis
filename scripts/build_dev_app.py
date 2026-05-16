from __future__ import annotations

from pathlib import Path
import plistlib
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "Jarvis"
BUNDLE_ID = "com.otzarjaffe.jarvis"
APP_PATH = ROOT / "dist" / f"{APP_NAME}.app"
CONTENTS = APP_PATH / "Contents"
MACOS = CONTENTS / "MacOS"


def main() -> int:
    if APP_PATH.exists():
        shutil.rmtree(APP_PATH)

    MACOS.mkdir(parents=True)
    write_info_plist()
    compile_native_app()
    ad_hoc_sign()

    print(f"Built {APP_PATH}")
    print(f"Open with: open {APP_PATH}")
    return 0


def write_info_plist() -> None:
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "13.0",
        "LSUIElement": True,
    }

    with (CONTENTS / "Info.plist").open("wb") as file:
        plistlib.dump(plist, file)


def compile_native_app() -> None:
    result = subprocess.run(
        [
            "swiftc",
            str(ROOT / "app" / "JarvisApp.swift"),
            "-o",
            str(MACOS / APP_NAME),
            "-framework",
            "AppKit",
            "-framework",
            "ApplicationServices",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)


def ad_hoc_sign() -> None:
    result = subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(APP_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Warning: ad hoc signing failed.")
        print((result.stderr or result.stdout).strip())


if __name__ == "__main__":
    raise SystemExit(main())
