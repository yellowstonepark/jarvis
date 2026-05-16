from __future__ import annotations

import argparse
import sys
import time

from jarvis.mac_agent.client import (
    AskClient,
    AskStreamError,
    WindowEventClient,
    WindowEventSendError,
    default_receiver_endpoint,
)
from jarvis.mac_agent.window import ActiveWindowError, get_active_window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Report the current macOS foreground application and window.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running and print the active window every interval.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds when using --watch. Default: 1.0.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print newline-delimited JSON snapshots instead of display text.",
    )
    parser.add_argument(
        "--source",
        default="local-mac",
        help="Name to include in JSON snapshots. Default: local-mac.",
    )
    parser.add_argument(
        "--send-to",
        help="POST each snapshot to a Jarvis receiver URL, for example http://mac-mini:8765/v1/window/events.",
    )
    parser.add_argument(
        "--send-timeout",
        type=float,
        default=3.0,
        help="HTTP send timeout in seconds when using --send-to. Default: 3.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] == "ask":
        return ask(raw_args[1:])

    args = build_parser().parse_args(raw_args)

    if args.interval <= 0:
        print("--interval must be greater than 0.", file=sys.stderr)
        return 2

    if args.send_timeout <= 0:
        print("--send-timeout must be greater than 0.", file=sys.stderr)
        return 2

    sender = WindowEventClient(args.send_to, args.send_timeout) if args.send_to else None

    try:
        if args.watch:
            return watch_active_window(args.interval, args.json, args.source, sender)

        snapshot = get_active_window(source=args.source)
        print_snapshot(snapshot, args.json)
        send_snapshot(snapshot, sender)
        return 0
    except ActiveWindowError as error:
        print(format_error(error), file=sys.stderr)
        return 1


def ask(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="jarvis ask",
        description="Ask the Mac mini Jarvis receiver and stream the response.",
    )
    parser.add_argument("prompt", nargs="+", help="Prompt to send to Jarvis.")
    parser.add_argument(
        "--ask-url",
        help="Jarvis ask endpoint. Defaults to JARVIS_ASK_URL or ~/.jarvis/receiver-url converted to /v1/ask.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Receiver timeout in seconds. Default: 60.0.",
    )
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        print("--timeout must be greater than 0.", file=sys.stderr)
        return 2

    endpoint = args.ask_url or default_receiver_endpoint()
    if endpoint is None:
        print(
            "No Jarvis ask endpoint configured. Set ~/.jarvis/receiver-url or pass --ask-url.",
            file=sys.stderr,
        )
        return 2

    prompt = " ".join(args.prompt)
    client = AskClient(endpoint, args.timeout)

    def write_stream(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    try:
        client.stream(prompt, write_stream)
        print(flush=True)
        return 0
    except AskStreamError as error:
        print(f"failed to ask Jarvis: {error}", file=sys.stderr)
        return 1


def watch_active_window(
    interval: float,
    as_json: bool,
    source: str,
    sender: WindowEventClient | None,
) -> int:
    while True:
        snapshot = get_active_window(source=source)
        print_snapshot(snapshot, as_json)
        send_snapshot(snapshot, sender)
        time.sleep(interval)


def send_snapshot(snapshot, sender: WindowEventClient | None) -> None:
    if sender is None:
        return

    try:
        sender.send(snapshot)
    except WindowEventSendError as error:
        print(f"failed to send window event: {error}", file=sys.stderr, flush=True)


def print_snapshot(snapshot, as_json: bool) -> None:
    if as_json:
        print(snapshot.to_json(), flush=True)
        return

    print(snapshot.display(), flush=True)


def format_error(error: ActiveWindowError) -> str:
    return (
        f"{error}\n\n"
        "Jarvis needs permission to inspect the current UI. For development, "
        "you can allow Terminal in System Settings > Privacy & Security > "
        "Accessibility. For regular use, prefer running Jarvis as its own app "
        "or launch agent so only Jarvis receives this permission. If macOS "
        "shows an Automation prompt for System Events, allow that too."
    )


if __name__ == "__main__":
    raise SystemExit(main())

