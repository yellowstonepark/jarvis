from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from jarvis.common.models import WindowSnapshot


class State:
    def __init__(self, event_log: Path) -> None:
        self._lock = threading.Lock()
        self._latest_window: WindowSnapshot | None = None
        self._event_log = event_log.expanduser()

    def set_latest_window(self, snapshot: WindowSnapshot) -> None:
        with self._lock:
            self._latest_window = snapshot
            self._append_event(snapshot)

    def get_latest_window(self) -> WindowSnapshot | None:
        with self._lock:
            return self._latest_window

    @property
    def event_log(self) -> Path:
        return self._event_log

    def _append_event(self, snapshot: WindowSnapshot) -> None:
        self._event_log.parent.mkdir(parents=True, exist_ok=True)
        with self._event_log.open("a", encoding="utf-8") as file:
            file.write(snapshot.to_json())
            file.write("\n")


def build_handler(state: State):
    class JarvisMiniHandler(BaseHTTPRequestHandler):
        server_version = "JarvisMini/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self.write_json(HTTPStatus.OK, {"ok": True})
                return

            if self.path == "/v1/window/latest":
                latest = state.get_latest_window()
                if latest is None:
                    self.write_json(HTTPStatus.NOT_FOUND, {"error": "no window snapshots"})
                    return
                self.write_json(HTTPStatus.OK, json.loads(latest.to_json()))
                return

            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/window/events":
                self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")

            try:
                snapshot = WindowSnapshot.from_json(raw_body)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                self.write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid window snapshot: {error}"},
                )
                return

            state.set_latest_window(snapshot)
            self.write_json(HTTPStatus.ACCEPTED, {"ok": True})

        def log_message(self, format: str, *args) -> None:
            return

        def write_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return JarvisMiniHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis-mini",
        description="Run the Jarvis Mac mini coordination service.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--event-log",
        type=Path,
        default=Path("~/.jarvis/window-events.jsonl"),
        help="Path for newline-delimited window events. Default: ~/.jarvis/window-events.jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = State(args.event_log)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))

    print(f"jarvis-mini listening on http://{args.host}:{args.port}", flush=True)
    print(f"writing window events to {state.event_log}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

