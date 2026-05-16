from __future__ import annotations

import argparse
import sys
import time

from jarvis.mac_agent.client import (
    AskClient,
    AskStreamError,
    MemoryClient,
    MemoryInspectError,
    WindowEventClient,
    WindowEventSendError,
    default_memory_endpoint,
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
    if raw_args and raw_args[0] == "memory":
        return memory(raw_args[1:])

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
    parser.add_argument(
        "--history-minutes",
        type=float,
        default=30.0,
        help="Minutes of recent window history to include. Default: 30.",
    )
    parser.add_argument(
        "--max-history-events",
        type=int,
        default=80,
        help="Maximum raw window events to consider before compaction. Default: 80.",
    )
    parser.add_argument(
        "--no-window-history",
        action="store_true",
        help="Ask without injecting recent window history.",
    )
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        print("--timeout must be greater than 0.", file=sys.stderr)
        return 2

    if args.history_minutes <= 0:
        print("--history-minutes must be greater than 0.", file=sys.stderr)
        return 2

    if args.max_history_events <= 0:
        print("--max-history-events must be greater than 0.", file=sys.stderr)
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
        client.stream(
            prompt,
            write_stream,
            with_window_history=not args.no_window_history,
            history_minutes=args.history_minutes,
            max_history_events=args.max_history_events,
        )
        print(flush=True)
        return 0
    except AskStreamError as error:
        print(f"failed to ask Jarvis: {error}", file=sys.stderr)
        return 1



def memory(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="jarvis memory",
        description="Inspect Jarvis memory stored on the Mac mini.",
    )
    parser.add_argument(
        "--memory-url",
        help="Jarvis memory endpoint. Defaults to receiver URL converted to /v1/memory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Receiver timeout in seconds. Default: 10.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recent_parser = subparsers.add_parser("recent", help="Show recent sessions.")
    recent_parser.add_argument("--hours", type=float, default=4.0)

    search_parser = subparsers.add_parser("search", help="Search session memory.")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--limit", type=int, default=10)

    session_parser = subparsers.add_parser("session", help="Show one session with raw events.")
    session_parser.add_argument("id", type=int)

    subparsers.add_parser("stats", help="Show memory database stats.")

    args = parser.parse_args(argv)
    if args.timeout <= 0:
        print("--timeout must be greater than 0.", file=sys.stderr)
        return 2

    endpoint = args.memory_url or default_memory_endpoint()
    if endpoint is None:
        print(
            "No Jarvis memory endpoint configured. Set ~/.jarvis/receiver-url or pass --memory-url.",
            file=sys.stderr,
        )
        return 2

    client = MemoryClient(endpoint, args.timeout)
    try:
        if args.command == "recent":
            if args.hours <= 0:
                print("--hours must be greater than 0.", file=sys.stderr)
                return 2
            print_sessions(client.recent(args.hours).get("sessions", []))
            return 0

        if args.command == "search":
            if args.limit <= 0:
                print("--limit must be greater than 0.", file=sys.stderr)
                return 2
            print_sessions(client.search(" ".join(args.query), args.limit).get("sessions", []))
            return 0

        if args.command == "session":
            print_session_detail(client.session(args.id))
            return 0

        if args.command == "stats":
            print_stats(client.stats())
            return 0
    except MemoryInspectError as error:
        print(f"failed to inspect Jarvis memory: {error}", file=sys.stderr)
        return 1

    parser.error("unknown memory command")
    return 2


def print_sessions(sessions: list[dict]) -> None:
    if not sessions:
        print("No sessions found.")
        return

    for session in sessions:
        print(format_session_header(session))
        print(f"  Summary: {session.get('summary') or ''}")
        print(f"  Key actions: {format_list(session.get('key_actions', []))}")
        print(f"  Open loops: {format_list(session.get('open_loops', []))}")
        print(
            "  "
            f"confidence={session.get('confidence')} "
            f"source={session.get('summary_source')} "
            f"events={session.get('event_count')}"
        )
        print()


def print_session_detail(payload: dict) -> None:
    session = payload.get("session") or {}
    print(format_session_header(session))
    print(f"Summary: {session.get('summary') or ''}")
    print(f"Category: {session.get('category') or ''}")
    print(f"Confidence: {session.get('confidence')}")
    print(f"Summary source: {session.get('summary_source')}")
    print(f"Event count: {session.get('event_count')}")
    print(f"Updated at: {session.get('updated_at') or ''}")
    print(f"Key actions: {format_list(session.get('key_actions', []))}")
    print(f"Open loops: {format_list(session.get('open_loops', []))}")
    print(f"Evidence windows: {format_list(session.get('evidence_windows', []))}")
    print("Raw compacted events:")
    raw_events = payload.get("raw_events") or []
    if not raw_events:
        print("  No raw events found for this session.")
        return
    for event in raw_events:
        title = f" - {event.get('window_title')}" if event.get("window_title") else ""
        print(f"  - {event.get('observed_at')} {event.get('app_name')}{title}")


def print_stats(stats: dict) -> None:
    for key in [
        "db_path",
        "oldest_event_time",
        "newest_event_time",
        "window_event_count",
        "session_count",
        "ollama_summary_count",
        "heuristic_summary_count",
        "fts_available",
        "fts_row_count",
    ]:
        print(f"{key}: {stats.get(key)}")


def format_session_header(session: dict) -> str:
    session_id = session.get("id")
    start = session.get("start_at") or "?"
    end = session.get("end_at") or "?"
    label = session.get("label") or ""
    category = session.get("category") or ""
    return f"[{session_id}] {start} -> {end}  {label} ({category})"


def format_list(values) -> str:
    if not values:
        return "-"
    return "; ".join(str(value) for value in values)

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

