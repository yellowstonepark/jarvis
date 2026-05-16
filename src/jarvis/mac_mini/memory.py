from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3

from jarvis.common.models import WindowSnapshot


@dataclass(frozen=True)
class ActivitySession:
    start_at: str
    end_at: str
    label: str
    summary: str
    event_count: int
    confidence: float
    summary_source: str = "heuristic"
    category: str = "Unknown"
    key_actions: tuple[str, ...] = ()
    open_loops: tuple[str, ...] = ()
    evidence_windows: tuple[str, ...] = ()
    updated_at: str = ""
    id: int | None = None


def should_ignore_window_event(snapshot: WindowSnapshot) -> bool:
    app_name = snapshot.app_name.lower()
    title = (snapshot.window_title or "").lower()
    terminal_apps = {"terminal", "iterm2", "warp"}
    return app_name in terminal_apps and "jarvis ask" in title


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS window_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    window_title TEXT,
                    raw_json TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_window_events_observed_at
                ON window_events(observed_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    label TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    summary_source TEXT NOT NULL DEFAULT 'heuristic',
                    category TEXT NOT NULL DEFAULT 'Unknown',
                    key_actions TEXT NOT NULL DEFAULT '[]',
                    open_loops TEXT NOT NULL DEFAULT '[]',
                    evidence_windows TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(start_at, end_at)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_start_at
                ON sessions(start_at)
                """
            )
            ensure_column(connection, "sessions", "summary_source", "TEXT NOT NULL DEFAULT 'heuristic'")
            ensure_column(connection, "sessions", "category", "TEXT NOT NULL DEFAULT 'Unknown'")
            ensure_column(connection, "sessions", "key_actions", "TEXT NOT NULL DEFAULT '[]'")
            ensure_column(connection, "sessions", "open_loops", "TEXT NOT NULL DEFAULT '[]'")
            ensure_column(connection, "sessions", "evidence_windows", "TEXT NOT NULL DEFAULT '[]'")
            ensure_column(connection, "sessions", "updated_at", "TEXT NOT NULL DEFAULT ''")
            self._initialize_session_fts(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ask_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asked_at TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    timezone TEXT,
                    location TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ask_interactions_asked_at
                ON ask_interactions(asked_at)
                """
            )

    def _initialize_session_fts(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                    session_id UNINDEXED,
                    start_at UNINDEXED,
                    end_at UNINDEXED,
                    label,
                    category,
                    summary,
                    key_actions,
                    open_loops,
                    evidence_windows
                )
                """
            )
        except sqlite3.OperationalError:
            self._fts_available = False
            return

        self._fts_available = True
        self._rebuild_session_fts(connection)

    def _rebuild_session_fts(self, connection: sqlite3.Connection) -> None:
        if not self._fts_available:
            return

        connection.execute("DELETE FROM session_fts")
        rows = connection.execute(
            """
            SELECT id, start_at, end_at, label, summary, category,
                   key_actions, open_loops, evidence_windows
            FROM sessions
            """
        ).fetchall()
        for row in rows:
            insert_session_fts_row(connection, row)

    def ingest_jsonl(self, path: Path) -> int:
        expanded_path = path.expanduser()
        if not expanded_path.exists():
            return 0

        inserted = 0
        with expanded_path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshot = WindowSnapshot.from_json(line)
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                inserted += self.insert_window_event(snapshot)
        return inserted

    def insert_window_event(self, snapshot: WindowSnapshot) -> int:
        if should_ignore_window_event(snapshot):
            return 0

        raw_json = snapshot.to_json()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO window_events (
                    observed_at, source, app_name, window_title, raw_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    parse_timestamp(snapshot.observed_at).isoformat(),
                    snapshot.source,
                    snapshot.app_name,
                    snapshot.window_title,
                    raw_json,
                ),
            )
            return cursor.rowcount

    def record_ask_interaction(
        self,
        prompt: str,
        timezone_name: str | None = None,
        location: str | None = None,
        keep_latest: int = 5,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ask_interactions (asked_at, prompt, timezone, location)
                VALUES (?, ?, ?, ?)
                """,
                (utc_now().isoformat(), prompt.strip(), timezone_name, location),
            )
            connection.execute(
                """
                DELETE FROM ask_interactions
                WHERE id NOT IN (
                    SELECT id FROM ask_interactions
                    ORDER BY asked_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (keep_latest,),
            )

    def recent_ask_interactions(self, limit: int = 5) -> list[dict[str, str | None]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT asked_at, prompt, timezone, location
                FROM ask_interactions
                ORDER BY asked_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "asked_at": row["asked_at"],
                "prompt": row["prompt"],
                "timezone": row["timezone"],
                "location": row["location"],
            }
            for row in reversed(rows)
        ]

    def oldest_window_event(self) -> WindowSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT app_name, window_title, observed_at, source
                FROM window_events
                ORDER BY observed_at ASC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None
        return WindowSnapshot(
            app_name=row["app_name"],
            window_title=row["window_title"],
            observed_at=row["observed_at"],
            source=row["source"],
        )

    def window_events_between(
        self,
        start_at: datetime,
        end_at: datetime,
    ) -> list[WindowSnapshot]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT app_name, window_title, observed_at, source
                FROM window_events
                WHERE observed_at >= ? AND observed_at <= ?
                ORDER BY observed_at ASC
                """,
                (start_at.isoformat(), end_at.isoformat()),
            ).fetchall()

        return [
            WindowSnapshot(
                app_name=row["app_name"],
                window_title=row["window_title"],
                observed_at=row["observed_at"],
                source=row["source"],
            )
            for row in rows
        ]

    def recent_window_events(self, minutes: float) -> list[WindowSnapshot]:
        end_at = utc_now()
        return self.window_events_between(end_at - timedelta(minutes=minutes), end_at)

    def window_events_for_session(self, session: ActivitySession) -> list[WindowSnapshot]:
        return self.window_events_between(
            parse_timestamp(session.start_at),
            parse_timestamp(session.end_at),
        )

    def build_sessions(
        self,
        start_at: datetime,
        end_at: datetime,
        gap_minutes: float = 8,
    ) -> list[ActivitySession]:
        events = self.window_events_between(start_at, end_at)
        sessions = sessionize_events(events, gap_minutes=gap_minutes)
        self.replace_sessions(sessions)
        return sessions

    def recent_sessions(self, minutes: float) -> list[ActivitySession]:
        start_at = utc_now() - timedelta(minutes=minutes)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, start_at, end_at, label, summary, event_count, confidence, summary_source, category, key_actions, open_loops, evidence_windows, updated_at
                FROM sessions
                WHERE end_at >= ?
                ORDER BY start_at ASC
                """,
                (start_at.isoformat(),),
            ).fetchall()

        return dedupe_sessions([session_from_row(row) for row in rows])

    def stored_sessions_between(
        self,
        start_at: datetime,
        end_at: datetime,
    ) -> list[ActivitySession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, start_at, end_at, label, summary, event_count, confidence, summary_source, category, key_actions, open_loops, evidence_windows, updated_at
                FROM sessions
                WHERE end_at >= ? AND start_at <= ?
                ORDER BY start_at ASC
                """,
                (start_at.isoformat(), end_at.isoformat()),
            ).fetchall()

        return dedupe_sessions([session_from_row(row) for row in rows])

    def replace_sessions(self, sessions: list[ActivitySession]) -> None:
        if not sessions:
            return

        with self._connect() as connection:
            for session in sessions:
                existing = existing_smart_session(connection, session)
                if existing is not None and session.summary_source == "heuristic":
                    session = existing

                connection.execute(
                    "DELETE FROM sessions WHERE start_at = ? AND end_at = ?",
                    (session.start_at, session.end_at),
                )
                cursor = connection.execute(
                    """
                    INSERT OR REPLACE INTO sessions (
                        start_at, end_at, label, summary, event_count, confidence,
                        summary_source, category, key_actions, open_loops, evidence_windows, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.start_at,
                        session.end_at,
                        session.label,
                        session.summary,
                        session.event_count,
                        session.confidence,
                        session.summary_source,
                        session.category,
                        encode_json_array(session.key_actions),
                        encode_json_array(session.open_loops),
                        encode_json_array(session.evidence_windows),
                        session.updated_at,
                    ),
                )
                if self._fts_available:
                    connection.execute(
                        "DELETE FROM session_fts WHERE start_at = ? AND end_at = ?",
                        (session.start_at, session.end_at),
                    )
                    insert_session_fts_session(connection, cursor.lastrowid, session)


    def session_by_id(self, session_id: int) -> ActivitySession | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, start_at, end_at, label, summary, event_count, confidence,
                       summary_source, category, key_actions, open_loops,
                       evidence_windows, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return session_from_row(row)

    def recent_sessions_by_hours(self, hours: float) -> list[ActivitySession]:
        start_at = utc_now() - timedelta(hours=hours)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, start_at, end_at, label, summary, event_count, confidence,
                       summary_source, category, key_actions, open_loops,
                       evidence_windows, updated_at
                FROM sessions
                WHERE end_at >= ?
                ORDER BY start_at ASC
                """,
                (start_at.isoformat(),),
            ).fetchall()

        return dedupe_sessions([session_from_row(row) for row in rows])

    def memory_stats(self) -> dict[str, object]:
        with self._connect() as connection:
            oldest_event = connection.execute(
                "SELECT observed_at FROM window_events ORDER BY observed_at ASC LIMIT 1"
            ).fetchone()
            newest_event = connection.execute(
                "SELECT observed_at FROM window_events ORDER BY observed_at DESC LIMIT 1"
            ).fetchone()
            window_event_count = connection.execute(
                "SELECT COUNT(*) AS count FROM window_events"
            ).fetchone()["count"]
            session_count = connection.execute(
                "SELECT COUNT(*) AS count FROM sessions"
            ).fetchone()["count"]
            ollama_summary_count = connection.execute(
                "SELECT COUNT(*) AS count FROM sessions WHERE summary_source = 'ollama'"
            ).fetchone()["count"]
            heuristic_summary_count = connection.execute(
                "SELECT COUNT(*) AS count FROM sessions WHERE summary_source != 'ollama'"
            ).fetchone()["count"]
            fts_row_count = None
            if self._fts_available:
                try:
                    fts_row_count = connection.execute(
                        "SELECT COUNT(*) AS count FROM session_fts"
                    ).fetchone()["count"]
                except sqlite3.OperationalError:
                    fts_row_count = None

        return {
            "db_path": str(self.path),
            "oldest_event_time": oldest_event["observed_at"] if oldest_event else None,
            "newest_event_time": newest_event["observed_at"] if newest_event else None,
            "window_event_count": window_event_count,
            "session_count": session_count,
            "ollama_summary_count": ollama_summary_count,
            "heuristic_summary_count": heuristic_summary_count,
            "fts_available": self._fts_available,
            "fts_row_count": fts_row_count,
        }

    def search_sessions(self, query: str, limit: int = 6) -> list[ActivitySession]:
        if not self._fts_available:
            return []

        match_query = build_fts_query(query)
        if not match_query:
            return []

        with self._connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT s.id, s.start_at, s.end_at, s.label, s.summary, s.event_count,
                           s.confidence, s.summary_source, s.category, s.key_actions,
                           s.open_loops, s.evidence_windows, s.updated_at
                    FROM session_fts
                    JOIN sessions s ON s.id = session_fts.session_id
                    WHERE session_fts MATCH ?
                    ORDER BY bm25(session_fts), s.end_at DESC
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

        return dedupe_sessions([session_from_row(row) for row in rows])


def ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def insert_session_fts_row(connection: sqlite3.Connection, row: sqlite3.Row) -> None:
    connection.execute(
        """
        INSERT INTO session_fts (
            session_id, start_at, end_at, label, category, summary,
            key_actions, open_loops, evidence_windows
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row["start_at"],
            row["end_at"],
            row["label"],
            row["category"],
            row["summary"],
            row["key_actions"],
            row["open_loops"],
            row["evidence_windows"],
        ),
    )


def insert_session_fts_session(
    connection: sqlite3.Connection,
    session_id: int,
    session: ActivitySession,
) -> None:
    connection.execute(
        """
        INSERT INTO session_fts (
            session_id, start_at, end_at, label, category, summary,
            key_actions, open_loops, evidence_windows
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            session.start_at,
            session.end_at,
            session.label,
            session.category,
            session.summary,
            " ".join(session.key_actions),
            " ".join(session.open_loops),
            " ".join(session.evidence_windows),
        ),
    )


def build_fts_query(query: str) -> str:
    stop_words = {
        "about",
        "again",
        "could",
        "doing",
        "from",
        "have",
        "leave",
        "left",
        "like",
        "what",
        "when",
        "where",
        "with",
        "work",
        "worked",
        "working",
    }
    tokens = []
    for token in re.findall(r"[A-Za-z0-9_]+", query.lower()):
        if len(token) < 3 or token in stop_words:
            continue
        tokens.append(token)

    unique_tokens = ordered_unique(tokens)[:8]
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def existing_smart_session(
    connection: sqlite3.Connection,
    session: ActivitySession,
) -> ActivitySession | None:
    row = connection.execute(
        """
        SELECT id, start_at, end_at, label, summary, event_count, confidence, summary_source, category, key_actions, open_loops, evidence_windows, updated_at
        FROM sessions
        WHERE start_at = ? AND end_at = ? AND summary_source = 'ollama'
        """,
        (session.start_at, session.end_at),
    ).fetchone()
    if row is None:
        return None
    return session_from_row(row)


def dedupe_sessions(sessions: list[ActivitySession]) -> list[ActivitySession]:
    by_range: dict[tuple[str, str], ActivitySession] = {}
    for session in sessions:
        key = (session.start_at, session.end_at)
        existing = by_range.get(key)
        if existing is None or (
            existing.summary_source != "ollama" and session.summary_source == "ollama"
        ):
            by_range[key] = session
    return sorted(by_range.values(), key=lambda session: session.start_at)


def session_from_row(row: sqlite3.Row) -> ActivitySession:
    return ActivitySession(
        start_at=row["start_at"],
        end_at=row["end_at"],
        label=row["label"],
        summary=row["summary"],
        event_count=row["event_count"],
        confidence=row["confidence"],
        summary_source=row["summary_source"],
        category=row["category"],
        key_actions=decode_json_array(row["key_actions"]),
        open_loops=decode_json_array(row["open_loops"]),
        evidence_windows=decode_json_array(row["evidence_windows"]),
        updated_at=row["updated_at"],
        id=row["id"] if "id" in row.keys() else None,
    )


def sessionize_events(
    events: list[WindowSnapshot],
    gap_minutes: float = 8,
) -> list[ActivitySession]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda event: parse_timestamp(event.observed_at))
    chunks: list[list[WindowSnapshot]] = []
    current: list[WindowSnapshot] = []
    current_label: str | None = None
    previous_at: datetime | None = None

    for event in sorted_events:
        observed_at = parse_timestamp(event.observed_at)
        label = classify_event(event)

        starts_new = False
        if previous_at is not None and observed_at - previous_at > timedelta(minutes=gap_minutes):
            starts_new = True
        elif current and label != current_label and len(current) >= 3:
            starts_new = True

        if starts_new:
            chunks.append(current)
            current = []

        current.append(event)
        current_label = dominant_label(current)
        previous_at = observed_at

    if current:
        chunks.append(current)

    return [summarize_chunk(chunk) for chunk in chunks if chunk]


def classify_event(event: WindowSnapshot) -> str:
    text = f"{event.app_name} {event.window_title or ''}".lower()

    if "jarvis" in text or "codex" in text:
        return "Jarvis"
    if "actuate" in text:
        return "Actuate"
    if "shwaz" in text:
        return "Shwaz"
    if "school" in text or "canvas" in text or "gradescope" in text:
        return "School"
    if "calendar" in text or "mail" in text or "messages" in text or "slack" in text:
        return "Admin"
    if event.app_name.lower() in {"code", "xcode", "terminal", "iterm2"}:
        return "Coding"
    if event.app_name.lower() in {"safari", "chrome", "arc", "firefox"}:
        return "Browsing"
    return "Unknown"


def dominant_label(events: list[WindowSnapshot]) -> str:
    return dominant_counted_label([classify_event(event) for event in events])


def summarize_chunk(events: list[WindowSnapshot]) -> ActivitySession:
    start_at = parse_timestamp(events[0].observed_at)
    end_at = parse_timestamp(events[-1].observed_at)
    labels = [classify_event(event) for event in events]
    label_counts = count_labels(labels)
    label = dominant_counted_label(labels)
    app_names = ordered_unique(event.app_name for event in events)
    title_samples = ordered_unique(
        title for title in (event.window_title for event in events) if title
    )[:3]

    app_switches = sum(
        1
        for previous, current in zip(events, events[1:])
        if previous.app_name != current.app_name
    )
    confidence = round(label_counts[label] / len(events), 2)

    if len(app_names) >= 4 and app_switches >= max(3, len(events) // 3):
        label = "Mixed/context switching"
        confidence = min(confidence, 0.6)

    summary = build_summary(label, app_names, title_samples)

    return ActivitySession(
        start_at=start_at.isoformat(),
        end_at=end_at.isoformat(),
        label=label,
        summary=summary,
        event_count=len(events),
        confidence=confidence,
        summary_source="heuristic",
        category=label,
        evidence_windows=tuple(format_evidence_windows(events)),
        updated_at=utc_now().isoformat(),
    )


def count_labels(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def dominant_counted_label(labels: list[str]) -> str:
    counts = count_labels(labels)
    best_label = labels[0]
    best_count = counts[best_label]
    for label in labels:
        count = counts[label]
        if count > best_count:
            best_label = label
            best_count = count
    return best_label


def build_summary(
    label: str,
    app_names: list[str],
    title_samples: list[str],
) -> str:
    apps = ", ".join(app_names[:4])
    if label == "Mixed/context switching":
        base = f"Moved between {apps} with frequent context switches."
    elif label == "Jarvis":
        base = f"Worked on Jarvis using {apps}."
    elif label in {"Actuate", "Shwaz", "School", "Admin"}:
        base = f"Worked on {label} activity using {apps}."
    elif label == "Coding":
        base = f"Worked in coding tools: {apps}."
    elif label == "Browsing":
        base = f"Browsed or researched in {apps}."
    else:
        base = f"Worked in {apps}."

    if title_samples:
        return f"{base} Notable windows: {'; '.join(title_samples)}."
    return base


def ordered_unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def encode_json_array(values: tuple[str, ...] | list[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def decode_json_array(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    cleaned = []
    for item in parsed:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def format_evidence_windows(events: list[WindowSnapshot], max_windows: int = 6) -> list[str]:
    windows = []
    for event in events:
        title = f" - {event.window_title}" if event.window_title else ""
        windows.append(f"{event.app_name}{title}")
    return ordered_unique(windows)[:max_windows]


def render_sessions(sessions: list[ActivitySession]) -> str:
    if not sessions:
        return "No window activity found for that range."

    lines: list[str] = []
    for session in sessions:
        start = parse_timestamp(session.start_at).strftime("%H:%M")
        end = parse_timestamp(session.end_at).strftime("%H:%M")
        lines.append(f"{start}-{end}  {session.label}")
        lines.append(session.summary)
    return "\n".join(lines)


def format_session_context(sessions: list[ActivitySession], max_sessions: int = 12) -> str:
    if not sessions:
        return ""

    lines: list[str] = []
    for session in sessions[-max_sessions:]:
        start = parse_timestamp(session.start_at).strftime("%H:%M")
        end = parse_timestamp(session.end_at).strftime("%H:%M")
        lines.append(
            f"- {start}-{end} {session.label} "
            f"(category: {session.category}, confidence: {session.confidence:.2f}): "
            f"{session.summary}"
        )
        if session.key_actions:
            lines.append(f"  Key actions: {'; '.join(session.key_actions)}")
        if session.open_loops:
            lines.append(f"  Open loops: {'; '.join(session.open_loops)}")
        if session.evidence_windows:
            lines.append(f"  Evidence windows: {'; '.join(session.evidence_windows)}")
    return "\n".join(lines)


def replace_session_summary(
    session: ActivitySession,
    label: str,
    summary: str,
    confidence: float = 0.85,
    summary_source: str = "ollama",
    category: str | None = None,
    key_actions: tuple[str, ...] = (),
    open_loops: tuple[str, ...] = (),
    evidence_windows: tuple[str, ...] = (),
) -> ActivitySession:
    return ActivitySession(
        start_at=session.start_at,
        end_at=session.end_at,
        label=label,
        summary=summary,
        event_count=session.event_count,
        confidence=confidence,
        summary_source=summary_source,
        category=category or session.category,
        key_actions=key_actions,
        open_loops=open_loops,
        evidence_windows=evidence_windows or session.evidence_windows,
        updated_at=utc_now().isoformat(),
        id=session.id,
    )


def format_events_for_summary(events: list[WindowSnapshot], max_events: int = 80) -> str:
    if not events:
        return "- No events"

    compacted: list[WindowSnapshot] = []
    previous_key: tuple[str, str | None] | None = None
    for event in sorted(events, key=lambda item: parse_timestamp(item.observed_at)):
        key = (event.app_name, event.window_title)
        if key == previous_key:
            continue
        compacted.append(event)
        previous_key = key

    compacted = compacted[-max_events:]
    lines = []
    for event in compacted:
        observed_at = parse_timestamp(event.observed_at).strftime("%H:%M")
        title = f" - {event.window_title}" if event.window_title else ""
        lines.append(f"- {observed_at} {event.app_name}{title}")
    return "\n".join(lines)
