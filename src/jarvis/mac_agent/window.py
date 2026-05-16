from __future__ import annotations

import subprocess

from jarvis.common.models import WindowSnapshot


class ActiveWindowError(RuntimeError):
    """Raised when macOS does not return active-window information."""


APPLE_SCRIPT = """
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set windowTitle to ""
    try
        set windowTitle to name of front window of frontApp
    end try
end tell
return appName & linefeed & windowTitle
""".strip()


def get_active_window(source: str = "local-mac") -> WindowSnapshot:
    """Return the current frontmost macOS application and window title."""

    try:
        result = subprocess.run(
            ["osascript", "-e", APPLE_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired as error:
        raise ActiveWindowError(
            "Timed out while asking macOS for the active window."
        ) from error

    if result.returncode != 0:
        message = result.stderr.strip() or "Unable to read the active window."
        raise ActiveWindowError(message)

    return parse_active_window(result.stdout, source=source)


def parse_active_window(output: str, source: str = "local-mac") -> WindowSnapshot:
    lines = output.splitlines()
    app_name = lines[0].strip() if lines else ""
    window_title = lines[1].strip() if len(lines) > 1 else ""

    if not app_name:
        raise ActiveWindowError("macOS returned an empty active application name.")

    return WindowSnapshot.now(
        app_name=app_name,
        window_title=window_title or None,
        source=source,
    )

