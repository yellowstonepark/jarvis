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
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.memory import (
    ActivitySession,
    MemoryStore,
    format_events_for_summary,
    format_session_context,
    replace_session_summary,
    should_ignore_window_event,
)


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


@dataclass(frozen=True)
class SummaryConfig:
    interval_seconds: float
    lookback_minutes: float
    stable_after_minutes: float


class State:
    def __init__(
        self,
        event_log: Path,
        db_path: Path,
        profile_path: Path = Path("~/.jarvis/profile.json"),
    ) -> None:
        self._lock = threading.Lock()
        self._latest_window: WindowSnapshot | None = None
        self._event_log = event_log.expanduser()
        self._memory = MemoryStore(db_path)
        self._profile_path = profile_path.expanduser()
        self._memory.ingest_jsonl(self._event_log)
        self._ask_active = threading.Event()
        self._summary_lock = threading.Lock()

    def set_latest_window(self, snapshot: WindowSnapshot) -> None:
        if should_ignore_window_event(snapshot):
            return

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

    @property
    def profile_path(self) -> Path:
        return self._profile_path

    def user_profile_context(self) -> str:
        return format_user_profile(self._profile_path)

    def recent_ask_context(self) -> str:
        with self._lock:
            interactions = self._memory.recent_ask_interactions(limit=5)
        return format_ask_interactions(interactions)

    def record_ask_interaction(
        self,
        prompt: str,
        timezone_name: str | None,
        location: str | None,
    ) -> None:
        with self._lock:
            self._memory.record_ask_interaction(prompt, timezone_name, location, keep_latest=5)

    def oldest_memory_context(self) -> str:
        with self._lock:
            oldest = self._memory.oldest_window_event()
        return format_oldest_memory(oldest)

    def memory_recent(self, hours: float) -> list[ActivitySession]:
        with self._lock:
            return self._memory.recent_sessions_by_hours(hours)

    def memory_search(self, query: str, limit: int = 10) -> list[ActivitySession]:
        with self._lock:
            return self._memory.search_sessions(query, limit=limit)

    def memory_session_detail(self, session_id: int) -> dict | None:
        with self._lock:
            session = self._memory.session_by_id(session_id)
            if session is None:
                return None
            events = self._memory.window_events_for_session(session)
        return {
            "session": session_to_dict(session),
            "raw_events": [window_snapshot_to_dict(event) for event in compact_window_events(events)],
        }

    def memory_stats(self) -> dict[str, object]:
        with self._lock:
            return self._memory.memory_stats()

    def recent_window_events(
        self,
        minutes: float,
    ) -> list[WindowSnapshot]:
        with self._lock:
            return self._memory.recent_window_events(minutes)

    def begin_interactive_request(self) -> None:
        self._ask_active.set()

    def end_interactive_request(self) -> None:
        self._ask_active.clear()

    def ask_memory_context(self, question: str, minutes: float) -> tuple[str, str]:
        end_at = datetime.now(timezone.utc)
        start_at = end_at - timedelta(minutes=minutes)
        with self._lock:
            recent_sessions = self._memory.build_sessions(start_at, end_at)
            relevant_sessions = self._memory.search_sessions(question, limit=6)

        recent_keys = session_keys(recent_sessions)
        older_relevant_sessions = [
            session for session in relevant_sessions if session_key(session) not in recent_keys
        ]
        return (
            format_session_context(recent_sessions),
            format_session_context(older_relevant_sessions),
        )

    def refresh_summaries(
        self,
        ollama: OllamaConfig,
        lookback_minutes: float,
        stable_after_minutes: float,
    ) -> int:
        if self._ask_active.is_set():
            return 0

        if not self._summary_lock.acquire(blocking=False):
            return 0

        try:
            end_at = datetime.now(timezone.utc) - timedelta(minutes=stable_after_minutes)
            start_at = end_at - timedelta(minutes=lookback_minutes)
            with self._lock:
                sessions = self._memory.build_sessions(start_at, end_at)

            updated: list[ActivitySession] = []
            for session in sessions:
                if self._ask_active.is_set():
                    break
                if session_has_structured_summary(session):
                    continue

                try:
                    events = self.window_events_for_session(session)
                    updated.append(summarize_session_with_ollama(session, events, ollama))
                except OllamaChatError:
                    continue

            if updated:
                with self._lock:
                    self._memory.replace_sessions(updated)

            return len(updated)
        finally:
            self._summary_lock.release()

    def window_events_for_session(self, session: ActivitySession) -> list[WindowSnapshot]:
        with self._lock:
            return self._memory.window_events_for_session(session)

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
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/health":
                self.write_json(HTTPStatus.OK, {"ok": True})
                return

            if path == "/v1/memory/recent":
                self.handle_memory_recent(parsed.query)
                return

            if path == "/v1/memory/search":
                self.handle_memory_search(parsed.query)
                return

            if path.startswith("/v1/memory/session/"):
                self.handle_memory_session(path)
                return

            if path == "/v1/memory/stats":
                self.write_json(HTTPStatus.OK, state.memory_stats())
                return

            if path == "/v1/window/latest":
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

            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})


        def handle_memory_recent(self, raw_query: str) -> None:
            try:
                query = parse_qs(raw_query)
                hours = float(query.get("hours", ["4"])[0])
            except (TypeError, ValueError):
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "hours must be a number"})
                return

            if hours <= 0:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "hours must be greater than 0"})
                return

            sessions = state.memory_recent(hours)
            self.write_json(
                HTTPStatus.OK,
                {"sessions": [session_to_dict(session) for session in sessions]},
            )

        def handle_memory_search(self, raw_query: str) -> None:
            query = parse_qs(raw_query)
            term = query.get("q", [""])[0].strip()
            try:
                limit = int(query.get("limit", ["10"])[0])
            except (TypeError, ValueError):
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
                return

            if limit <= 0:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be greater than 0"})
                return

            sessions = state.memory_search(term, limit=limit)
            self.write_json(
                HTTPStatus.OK,
                {"sessions": [session_to_dict(session) for session in sessions]},
            )

        def handle_memory_session(self, path: str) -> None:
            raw_id = path.removeprefix("/v1/memory/session/")
            try:
                session_id = int(unquote(raw_id))
            except ValueError:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "session id must be an integer"})
                return

            detail = state.memory_session_detail(session_id)
            if detail is None:
                self.write_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
                return
            self.write_json(HTTPStatus.OK, detail)

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

        def handle_ask(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")

            try:
                payload = json.loads(raw_body)
                prompt = payload["prompt"]
                with_window_history = payload.get("with_window_history", True)
                history_minutes = float(payload.get("history_minutes", 30))
                max_history_events = int(payload.get("max_history_events", 80))
                timezone_name = payload.get("timezone")
                location = payload.get("location")
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

            if timezone_name is not None and not isinstance(timezone_name, str):
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "timezone must be a string"})
                return

            if location is not None and not isinstance(location, str):
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "location must be a string"})
                return

            state.begin_interactive_request()
            try:
                if with_window_history:
                    events = state.recent_window_events(history_minutes)
                    session_context, relevant_session_context = state.ask_memory_context(
                        prompt, history_minutes
                    )
                    recent_ask_context = state.recent_ask_context()
                    user_profile_context = state.user_profile_context()
                    window_stats_context = format_window_stats(events, history_minutes)
                    oldest_memory_context = state.oldest_memory_context()
                    state.record_ask_interaction(prompt, timezone_name, location)
                    prompt = build_ask_prompt(
                        prompt,
                        events,
                        history_minutes,
                        max_history_events,
                        session_context,
                        relevant_session_context,
                        recent_ask_context=recent_ask_context,
                        user_profile_context=user_profile_context,
                        window_stats_context=window_stats_context,
                        oldest_memory_context=oldest_memory_context,
                        timezone_name=timezone_name,
                        location=location,
                    )

                if not with_window_history:
                    state.record_ask_interaction(prompt, timezone_name, location)

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
            finally:
                state.end_interactive_request()

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







def session_to_dict(session: ActivitySession) -> dict[str, object]:
    return {
        "id": session.id,
        "start_at": session.start_at,
        "end_at": session.end_at,
        "label": session.label,
        "category": session.category,
        "summary": session.summary,
        "key_actions": list(session.key_actions),
        "open_loops": list(session.open_loops),
        "evidence_windows": list(session.evidence_windows),
        "confidence": session.confidence,
        "summary_source": session.summary_source,
        "event_count": session.event_count,
        "updated_at": session.updated_at,
    }


def window_snapshot_to_dict(snapshot: WindowSnapshot) -> dict[str, str | None]:
    return {
        "observed_at": snapshot.observed_at,
        "source": snapshot.source,
        "app_name": snapshot.app_name,
        "window_title": snapshot.window_title,
    }


def compact_window_events(events: list[WindowSnapshot]) -> list[WindowSnapshot]:
    compacted: list[WindowSnapshot] = []
    previous_key: tuple[str, str | None] | None = None
    for event in sorted(events, key=lambda item: parse_timestamp(item.observed_at)):
        key = (event.app_name, event.window_title)
        if key == previous_key:
            continue
        compacted.append(event)
        previous_key = key
    return compacted

def session_key(session: ActivitySession) -> tuple[str, str]:
    return (session.start_at, session.end_at)


def session_keys(sessions: list[ActivitySession]) -> set[tuple[str, str]]:
    return {session_key(session) for session in sessions}


def session_has_structured_summary(session: ActivitySession) -> bool:
    return session.summary_source == "ollama" and bool(session.evidence_windows)


def summarize_session_with_ollama(
    session: ActivitySession,
    events: list[WindowSnapshot],
    config: OllamaConfig,
) -> ActivitySession:
    prompt = build_session_summary_prompt(session, events)
    response = complete_ollama_chat(prompt, config).strip()
    return parse_session_summary_response(response, session)


def build_session_summary_prompt(
    session: ActivitySession,
    events: list[WindowSnapshot],
) -> str:
    timeline = format_events_for_summary(events)
    start = parse_timestamp(session.start_at).strftime("%H:%M")
    end = parse_timestamp(session.end_at).strftime("%H:%M")
    return (
        "You summarize a short Mac activity session for Jarvis memory. "
        "Use only the window timeline. Do not invent file contents, code changes, "
        "project decisions, or tasks that are not visible from app/window titles. "
        "Prefer concrete work descriptions over app lists. If the timeline is vague, say so.\n\n"
        "Allowed labels/categories: Jarvis, Actuate, Shwaz, School, Admin, Coding, "
        "Browsing, Mixed/context switching, Unknown.\n\n"
        f"Time range: {start}-{end}\n"
        f"Heuristic label: {session.label}\n"
        f"Window timeline:\n{timeline}\n\n"
        "Return exactly one JSON object with keys: label, category, summary, "
        "key_actions, open_loops, evidence_windows, confidence. "
        "key_actions, open_loops, and evidence_windows must be arrays of short strings. "
        "key_actions should describe visible activity, not inferred implementation details. "
        "confidence must be a number from 0 to 1. Use empty arrays when unsure. "
        "The summary must be one concise sentence. Example: "
        '{"label":"Jarvis memory work","category":"Jarvis",'
        '"summary":"Worked on Jarvis memory code in Terminal and Code.",'
        '"key_actions":["Worked in Code on memory.py"],"open_loops":[],'
        '"evidence_windows":["Terminal - jarvis","Code - memory.py"],"confidence":0.82}'
    )


def parse_session_summary_response(
    response: str,
    fallback: ActivitySession,
) -> ActivitySession:
    try:
        data = json.loads(response)
        label = clean_text(data["label"])
        category = clean_category(clean_text(data.get("category", fallback.category)), fallback.category)
        summary = clean_text(data["summary"])
        key_actions = clean_string_array(data.get("key_actions", []), max_items=5)
        open_loops = clean_string_array(data.get("open_loops", []), max_items=5)
        evidence_windows = clean_string_array(data.get("evidence_windows", []), max_items=8)
        confidence = clean_confidence(data.get("confidence", 0.85), fallback.confidence)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback

    if not label or not summary:
        return fallback

    return replace_session_summary(
        fallback,
        label=label[:80],
        summary=summary[:300],
        confidence=confidence,
        summary_source="ollama",
        category=category,
        key_actions=key_actions,
        open_loops=open_loops,
        evidence_windows=evidence_windows or fallback.evidence_windows,
    )


def clean_text(value) -> str:
    return str(value).strip()


def clean_category(category: str, fallback: str) -> str:
    allowed = {
        "Jarvis",
        "Actuate",
        "Shwaz",
        "School",
        "Admin",
        "Coding",
        "Browsing",
        "Mixed/context switching",
        "Unknown",
    }
    normalized = category.strip()
    return normalized if normalized in allowed else fallback


def clean_string_array(value, max_items: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned: list[str] = []
    for item in value:
        text = clean_text(item)
        if text:
            cleaned.append(text[:160])
        if len(cleaned) >= max_items:
            break
    return tuple(cleaned)


def clean_confidence(value, fallback: float) -> float:
    confidence = float(value)
    if confidence < 0 or confidence > 1:
        return fallback
    return round(confidence, 2)


def complete_ollama_chat(prompt: str, config: OllamaConfig) -> str:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
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
        with urlopen(request, timeout=config.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaChatError(f"ollama returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise OllamaChatError(f"could not reach ollama: {error.reason}") from error
    except (KeyError, json.JSONDecodeError) as error:
        raise OllamaChatError(f"invalid ollama response: {error}") from error
    except TimeoutError as error:
        raise OllamaChatError("timed out connecting to ollama") from error

    try:
        return payload["message"]["content"]
    except KeyError as error:
        raise OllamaChatError(f"invalid ollama response: {error}") from error

def format_user_profile(profile_path: Path) -> str:
    if not profile_path.exists():
        return ""
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "- User profile exists but could not be read as JSON."
    if not isinstance(data, dict) or not data:
        return ""

    lines = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)


def format_ask_interactions(interactions: list[dict[str, str | None]]) -> str:
    if not interactions:
        return ""
    lines = []
    for item in interactions:
        asked_at = parse_timestamp(str(item["asked_at"])).strftime("%H:%M")
        lines.append(f"- {asked_at} {item['prompt']}")
    return "\n".join(lines)


def format_oldest_memory(snapshot: WindowSnapshot | None) -> str:
    if snapshot is None:
        return ""
    observed_at = parse_timestamp(snapshot.observed_at).isoformat()
    title = f" - {snapshot.window_title}" if snapshot.window_title else ""
    return f"- Oldest recorded window event: {observed_at} {snapshot.app_name}{title}"


def format_window_stats(events: list[WindowSnapshot], history_minutes: float) -> str:
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

    switches = max(0, len(segments) - 1)
    observed_minutes = max(
        (parse_timestamp(sorted_events[-1].observed_at) - parse_timestamp(sorted_events[0].observed_at)).total_seconds() / 60,
        min(history_minutes, 1),
    )
    switches_per_minute = switches / observed_minutes if observed_minutes else 0
    unique_apps = sorted({event.app_name for event in sorted_events})
    longest = max((end - start for start, end, _ in segments), default=timedelta(0))
    longest_minutes = longest.total_seconds() / 60

    if switches_per_minute >= 1.5:
        switching_level = "high"
    elif switches_per_minute >= 0.5:
        switching_level = "moderate"
    else:
        switching_level = "low"

    return (
        f"- Switches: {switches}\n"
        f"- Switches per minute: {switches_per_minute:.2f}\n"
        f"- Switching level: {switching_level}\n"
        f"- Unique apps: {len(unique_apps)} ({', '.join(unique_apps[:8])})\n"
        f"- Longest continuous same-window stretch: {longest_minutes:.1f} minutes"
    )

def build_ask_prompt(
    question: str,
    events: list[WindowSnapshot],
    history_minutes: float,
    max_segments: int = 80,
    session_context: str = "",
    relevant_session_context: str = "",
    recent_ask_context: str = "",
    user_profile_context: str = "",
    window_stats_context: str = "",
    oldest_memory_context: str = "",
    timezone_name: str | None = None,
    location: str | None = None,
) -> str:
    timeline = format_window_timeline(events, max_segments)
    if not timeline:
        timeline = "- No recent window events were recorded."

    if not session_context:
        session_context = "- No recent session summaries are available yet."

    if not relevant_session_context:
        relevant_session_context = "- No older matching session memories were found."

    if not recent_ask_context:
        recent_ask_context = "- No previous Jarvis ask commands are available."

    if not user_profile_context:
        user_profile_context = "- No user profile is configured."

    if not window_stats_context:
        window_stats_context = "- No window switching stats are available."

    if not oldest_memory_context:
        oldest_memory_context = "- No oldest memory marker is available."

    context = build_environment_context(timezone_name, location)

    return (
        "You are Jarvis, a concise local assistant. Answer the user using only "
        "the session memories and recent window timeline below when the question "
        "asks about activity, focus, apps, projects, or recent work. Prefer "
        "relevant older session memories when the question names a topic from the past, "
        "recent session summaries for current context, recent Jarvis ask commands for "
        "follow-up questions, window stats for focus/switching questions, and raw events for details. "
        "If the context is insufficient, say what is missing. Do not invent details.\n\n"
        f"Current context:\n{context}\n\n"
        f"User profile:\n{user_profile_context}\n\n"
        f"Previous Jarvis ask commands:\n{recent_ask_context}\n\n"
        f"Oldest memory marker:\n{oldest_memory_context}\n\n"
        f"Window switching stats, last {history_minutes:g} minutes:\n{window_stats_context}\n\n"
        f"User question:\n{question.strip()}\n\n"
        f"Recent session summaries, last {history_minutes:g} minutes:\n{session_context}\n\n"
        f"Relevant older session memories:\n{relevant_session_context}\n\n"
        f"Recent raw window timeline, last {history_minutes:g} minutes:\n{timeline}\n\n"
        "Answer concisely."
    )



def build_environment_context(
    timezone_name: str | None,
    location: str | None,
) -> str:
    now_utc = datetime.now(timezone.utc)
    tz = timezone.utc
    timezone_label = "UTC"

    if timezone_name:
        try:
            tz = ZoneInfo(timezone_name)
            timezone_label = timezone_name
        except ZoneInfoNotFoundError:
            timezone_label = timezone_name

    now_local = now_utc.astimezone(tz)
    location_label = location.strip() if location and location.strip() else "unknown"

    return (
        f"- Current UTC time: {now_utc.isoformat()}\n"
        f"- Current local time: {now_local.isoformat()}\n"
        f"- Timezone: {timezone_label}\n"
        f"- Location: {location_label}"
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



def start_summary_worker(
    state: State,
    ollama: OllamaConfig,
    config: SummaryConfig,
) -> threading.Thread | None:
    if config.interval_seconds <= 0:
        return None

    def run() -> None:
        pause = threading.Event()
        while True:
            pause.wait(config.interval_seconds)
            try:
                updated = state.refresh_summaries(
                    ollama,
                    config.lookback_minutes,
                    config.stable_after_minutes,
                )
                if updated:
                    print(f"summary worker updated {updated} sessions", flush=True)
            except Exception as error:
                print(f"summary worker error: {error}", flush=True)

    thread = threading.Thread(target=run, name="jarvis-summary-worker", daemon=True)
    thread.start()
    return thread


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
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path("~/.jarvis/profile.json"),
        help="Path for user profile JSON. Default: ~/.jarvis/profile.json.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="gemma4.e4b")
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=30.0,
        help="Ollama connection timeout in seconds. Default: 30.0.",
    )
    parser.add_argument(
        "--summary-interval",
        type=float,
        default=300.0,
        help="Background summary interval in seconds. Use 0 to disable. Default: 300.",
    )
    parser.add_argument(
        "--summary-lookback-minutes",
        type=float,
        default=24 * 60,
        help="How far back the summary worker refreshes. Default: 1440.",
    )
    parser.add_argument(
        "--summary-stable-after-minutes",
        type=float,
        default=5.0,
        help="Do not summarize sessions that ended more recently than this. Default: 5.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = State(args.event_log, args.db_path, args.profile_path)
    ollama = OllamaConfig(args.ollama_url, args.ollama_model, args.ollama_timeout)
    summary_config = SummaryConfig(
        interval_seconds=args.summary_interval,
        lookback_minutes=args.summary_lookback_minutes,
        stable_after_minutes=args.summary_stable_after_minutes,
    )
    start_summary_worker(state, ollama, summary_config)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, ollama))

    print(f"jarvis-mini listening on http://{args.host}:{args.port}", flush=True)
    print(f"writing window events to {state.event_log}", flush=True)
    print(f"writing memory database to {state.db_path}", flush=True)
    print(f"reading user profile from {state.profile_path}", flush=True)
    print(f"ollama chat model {ollama.model} via {ollama.chat_url}", flush=True)
    print(
        f"summary worker interval {summary_config.interval_seconds:g}s "
        f"lookback {summary_config.lookback_minutes:g}m "
        f"stable_after {summary_config.stable_after_minutes:g}m",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

