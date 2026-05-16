from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.memory import MemoryStore, format_session_context, render_recap


class OllamaChatError(Exception):
    """Raised when the Mac mini cannot stream a response from Ollama."""


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    timeout: float

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"


class State:
    def __init__(self, event_log: Path, db_path: Path) -> None:
        self._lock = threading.Lock()
        self._latest_window: WindowSnapshot | None = None
        self._event_log = event_log.expanduser()
        self._memory = MemoryStore(db_path)
        self._memory.ingest_jsonl(self._event_log)

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

    @property
    def db_path(self) -> Path:
        return self._memory.path

    def recent_window_events(
        self,
        minutes: float,
    ) -> list[WindowSnapshot]:
        with self._lock:
            return self._memory.recent_window_events(minutes)

    def recap(self, minutes: float) -> str:
        end_at = datetime.now(timezone.utc)
        start_at = end_at - timedelta(minutes=minutes)
        with self._lock:
            sessions = self._memory.build_sessions(start_at, end_at)
        return render_recap(sessions)

    def recent_session_context(self, minutes: float) -> str:
        end_at = datetime.now(timezone.utc)
        start_at = end_at - timedelta(minutes=minutes)
        with self._lock:
            sessions = self._memory.build_sessions(start_at, end_at)
        return format_session_context(sessions)

    def _append_event(self, snapshot: WindowSnapshot) -> None:
        self._event_log.parent.mkdir(parents=True, exist_ok=True)
        with self._event_log.open("a", encoding="utf-8") as file:
            file.write(snapshot.to_json())
            file.write("\n")
        self._memory.insert_window_event(snapshot)


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_handler(state: State, ollama: OllamaConfig):
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
            if self.path == "/v1/window/events":
                self.handle_window_event()
                return

            if self.path == "/v1/ask":
                self.handle_ask()
                return

            if self.path == "/v1/recap":
                self.handle_recap()
                return

            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def handle_window_event(self) -> None:
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

        def handle_recap(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")

            try:
                payload = json.loads(raw_body) if raw_body else {}
                minutes = float(payload.get("minutes", 120))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid recap payload: {error}"})
                return

            if minutes <= 0:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "minutes must be greater than 0"})
                return

            self.write_json(HTTPStatus.OK, {"recap": state.recap(minutes)})

        def handle_ask(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")

            try:
                payload = json.loads(raw_body)
                prompt = payload["prompt"]
                with_window_history = payload.get("with_window_history", True)
                history_minutes = float(payload.get("history_minutes", 30))
                max_history_events = int(payload.get("max_history_events", 80))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid ask payload: {error}"})
                return

            if not isinstance(prompt, str) or not prompt.strip():
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "prompt must be a non-empty string"})
                return

            if not isinstance(with_window_history, bool):
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "with_window_history must be a boolean"})
                return

            if history_minutes <= 0:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "history_minutes must be greater than 0"})
                return

            if max_history_events <= 0:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "max_history_events must be greater than 0"})
                return

            if with_window_history:
                events = state.recent_window_events(history_minutes)
                session_context = state.recent_session_context(history_minutes)
                prompt = build_ask_prompt(
                    prompt,
                    events,
                    history_minutes,
                    max_history_events,
                    session_context,
                )

            try:
                ollama_response = open_ollama_chat(prompt, ollama)
            except OllamaChatError as error:
                self.write_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
                return

            with ollama_response:
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "text/plain; charset=utf-8")
                self.end_headers()

                for chunk in iter_ollama_content(ollama_response):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()

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




def build_ask_prompt(
    question: str,
    events: list[WindowSnapshot],
    history_minutes: float,
    max_segments: int = 80,
    session_context: str = "",
) -> str:
    timeline = format_window_timeline(events, max_segments)
    if not timeline:
        timeline = "- No recent window events were recorded."

    if not session_context:
        session_context = "- No session summaries are available yet."

    return (
        "You are Jarvis, a concise local assistant. Answer the user using only "
        "the session summaries and recent window timeline below when the question "
        "asks about activity, focus, apps, projects, or recent work. Prefer "
        "session summaries for higher-level answers and raw events for details. "
        "If the context is insufficient, say what is missing. Do not invent details.\n\n"
        f"User question:\n{question.strip()}\n\n"
        f"Recent session summaries, last {history_minutes:g} minutes:\n{session_context}\n\n"
        f"Recent raw window timeline, last {history_minutes:g} minutes:\n{timeline}\n\n"
        "Answer concisely."
    )


def format_window_timeline(events: list[WindowSnapshot], max_segments: int = 80) -> str:
    if not events:
        return ""

    sorted_events = sorted(events, key=lambda event: parse_timestamp(event.observed_at))
    segments: list[tuple[datetime, datetime, WindowSnapshot]] = []

    for event in sorted_events:
        observed_at = parse_timestamp(event.observed_at)
        if not segments:
            segments.append((observed_at, observed_at, event))
            continue

        start, _, previous = segments[-1]
        if (previous.app_name, previous.window_title) == (event.app_name, event.window_title):
            segments[-1] = (start, observed_at, previous)
        else:
            segments.append((observed_at, observed_at, event))

    if len(segments) > max_segments:
        segments = segments[-max_segments:]

    lines = []
    for start, end, event in segments:
        time_label = format_time_range(start, end)
        title = f" - {event.window_title}" if event.window_title else ""
        lines.append(f"- {time_label} {event.app_name}{title}")

    return "\n".join(lines)


def format_time_range(start: datetime, end: datetime) -> str:
    start_label = start.strftime("%H:%M")
    end_label = end.strftime("%H:%M")
    if start_label == end_label:
        return start_label
    return f"{start_label}-{end_label}"


def open_ollama_chat(prompt: str, config: OllamaConfig):
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        config.chat_url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )

    try:
        return urlopen(request, timeout=config.timeout)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaChatError(f"ollama returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise OllamaChatError(f"could not reach ollama: {error.reason}") from error
    except TimeoutError as error:
        raise OllamaChatError("timed out connecting to ollama") from error


def iter_ollama_content(response):
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue

        payload = json.loads(line)
        if "error" in payload:
            raise OllamaChatError(str(payload["error"]))

        content = payload.get("message", {}).get("content")
        if content:
            yield content


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
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("~/.jarvis/jarvis.sqlite"),
        help="Path for SQLite memory store. Default: ~/.jarvis/jarvis.sqlite.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="gemma4.e4b")
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=30.0,
        help="Ollama connection timeout in seconds. Default: 30.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = State(args.event_log, args.db_path)
    ollama = OllamaConfig(args.ollama_url, args.ollama_model, args.ollama_timeout)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, ollama))

    print(f"jarvis-mini listening on http://{args.host}:{args.port}", flush=True)
    print(f"writing window events to {state.event_log}", flush=True)
    print(f"writing memory database to {state.db_path}", flush=True)
    print(f"ollama chat model {ollama.model} via {ollama.chat_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

