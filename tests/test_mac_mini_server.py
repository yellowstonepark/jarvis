import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.server import (
    OllamaConfig,
    State,
    build_ask_prompt,
    build_handler,
    format_user_profile,
    format_window_stats,
    format_window_timeline,
    open_ollama_chat,
    parse_session_summary_response,
)
from jarvis.mac_mini.memory import ActivitySession, MemoryStore, format_session_context, render_sessions


class StateTests(unittest.TestCase):
    def test_set_latest_window_appends_jsonl_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = Path(tmpdir) / ".jarvis" / "window-events.jsonl"
            db_path = Path(tmpdir) / ".jarvis" / "jarvis.sqlite"
            state = State(event_log, db_path)
            snapshot = WindowSnapshot(
                app_name="Safari",
                window_title="OpenAI",
                observed_at="2026-05-16T12:00:00+00:00",
                source="macbook",
            )

            state.set_latest_window(snapshot)

            self.assertEqual(state.get_latest_window(), snapshot)
            lines = event_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), json.loads(snapshot.to_json()))

    def test_ask_memory_context_dedupes_recent_and_retrieved_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = Path(tmpdir) / ".jarvis" / "window-events.jsonl"
            db_path = Path(tmpdir) / ".jarvis" / "jarvis.sqlite"
            state = State(event_log, db_path)
            state.set_latest_window(
                WindowSnapshot(
                    "Code",
                    "SQLite memory.py",
                    datetime.now(timezone.utc).isoformat(),
                    "macbook",
                )
            )
            older = ActivitySession(
                start_at="2026-05-15T18:00:00+00:00",
                end_at="2026-05-15T18:05:00+00:00",
                label="Older Jarvis SQLite memory",
                category="Jarvis",
                summary="Worked on older SQLite recall.",
                event_count=5,
                confidence=0.9,
                summary_source="ollama",
                key_actions=("Designed recall search",),
                evidence_windows=("Code - server.py",),
            )
            state._memory.replace_sessions([older])

            recent_context, relevant_context = state.ask_memory_context("SQLite memory", 30)

            self.assertIn("SQLite memory.py", recent_context)
            self.assertIn("Older Jarvis SQLite memory", relevant_context)
            self.assertNotIn("SQLite memory.py", relevant_context)


    def test_set_latest_window_ignores_terminal_jarvis_ask_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = Path(tmpdir) / ".jarvis" / "window-events.jsonl"
            db_path = Path(tmpdir) / ".jarvis" / "jarvis.sqlite"
            state = State(event_log, db_path)

            state.set_latest_window(
                WindowSnapshot(
                    "Terminal",
                    "jarvis ask what was I doing",
                    "2026-05-16T12:00:00+00:00",
                    "macbook",
                )
            )

            self.assertIsNone(state.get_latest_window())
            self.assertFalse(event_log.exists())
            self.assertEqual(state.recent_window_events(60), [])

    def test_record_ask_interaction_keeps_latest_five(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = Path(tmpdir) / ".jarvis" / "window-events.jsonl"
            db_path = Path(tmpdir) / ".jarvis" / "jarvis.sqlite"
            state = State(event_log, db_path)

            for index in range(7):
                state.record_ask_interaction(
                    f"question {index}",
                    "America/Los_Angeles",
                    None,
                )

            context = state.recent_ask_context()

            self.assertNotIn("question 0", context)
            self.assertNotIn("question 1", context)
            self.assertIn("question 2", context)
            self.assertIn("question 6", context)


class MemoryStoreTests(unittest.TestCase):
    def test_memory_store_persists_window_events_and_builds_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir) / "jarvis.sqlite")
            store.insert_window_event(
                WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:00:00+00:00", "macbook")
            )
            store.insert_window_event(
                WindowSnapshot("Code", "server.py", "2026-05-16T18:01:00+00:00", "macbook")
            )

            sessions = store.build_sessions(
                datetime.fromisoformat("2026-05-16T17:59:00+00:00"),
                datetime.fromisoformat("2026-05-16T18:02:00+00:00"),
            )

            rendered_sessions = render_sessions(sessions)
            self.assertIn("18:00-18:01", rendered_sessions)
            self.assertIn("Jarvis", rendered_sessions)

    def test_memory_store_persists_structured_session_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir) / "jarvis.sqlite")
            session = ActivitySession(
                start_at="2026-05-16T18:00:00+00:00",
                end_at="2026-05-16T18:05:00+00:00",
                label="Jarvis memory work",
                category="Jarvis",
                summary="Worked on structured memory.",
                event_count=5,
                confidence=0.9,
                summary_source="ollama",
                key_actions=("Added structured fields",),
                open_loops=("Verify persistence",),
                evidence_windows=("Code - memory.py",),
                updated_at="2026-05-16T18:06:00+00:00",
            )

            store.replace_sessions([session])
            stored = store.stored_sessions_between(
                datetime.fromisoformat("2026-05-16T17:59:00+00:00"),
                datetime.fromisoformat("2026-05-16T18:06:00+00:00"),
            )

            self.assertEqual(stored[0].key_actions, ("Added structured fields",))
            self.assertEqual(stored[0].open_loops, ("Verify persistence",))
            self.assertEqual(stored[0].evidence_windows, ("Code - memory.py",))

    def test_memory_store_indexes_sessions_into_fts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir) / "jarvis.sqlite")
            store.replace_sessions([
                ActivitySession(
                    start_at="2026-05-15T18:00:00+00:00",
                    end_at="2026-05-15T18:20:00+00:00",
                    label="Jarvis SQLite memory",
                    category="Jarvis",
                    summary="Worked on SQLite-backed searchable session memory.",
                    event_count=20,
                    confidence=0.9,
                    summary_source="ollama",
                    key_actions=("Added FTS indexing",),
                    open_loops=("Verify recall from jarvis ask",),
                    evidence_windows=("Code - memory.py",),
                )
            ])

            matches = store.search_sessions("where did I leave off on SQLite memory?")

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].label, "Jarvis SQLite memory")

    def test_memory_store_search_empty_query_returns_no_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir) / "jarvis.sqlite")

            self.assertEqual(store.search_sessions("what was I doing?"), [])

    def test_ingest_jsonl_skips_when_checkpoint_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "window-events.jsonl"
            log_path.write_text(
                WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:00:00+00:00", "macbook").to_json()
                + "\n",
                encoding="utf-8",
            )
            store = MemoryStore(Path(tmpdir) / "jarvis.sqlite")

            self.assertEqual(store.ingest_jsonl(log_path), 1)
            self.assertEqual(store.ingest_jsonl(log_path), 0)

            with log_path.open("a", encoding="utf-8") as file:
                file.write(
                    WindowSnapshot("Code", "server.py", "2026-05-16T18:01:00+00:00", "macbook").to_json()
                    + "\n"
                )

            self.assertEqual(store.ingest_jsonl(log_path), 1)


class TimelinePromptTests(unittest.TestCase):
    def test_format_window_timeline_compacts_repeated_windows(self) -> None:
        events = [
            WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:00:00+00:00", "macbook"),
            WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:00:30+00:00", "macbook"),
            WindowSnapshot("Codex", "Codex", "2026-05-16T18:01:00+00:00", "macbook"),
        ]

        self.assertEqual(
            format_window_timeline(events),
            "- 18:00 Terminal - jarvis\n- 18:01 Codex - Codex",
        )

    def test_build_ask_prompt_includes_question_and_timeline(self) -> None:
        events = [
            WindowSnapshot("Safari", "Ollama docs", "2026-05-16T18:00:00+00:00", "macbook"),
        ]

        session_context = format_session_context([
            ActivitySession(
                start_at="2026-05-16T17:50:00+00:00",
                end_at="2026-05-16T18:00:00+00:00",
                label="Jarvis memory work",
                category="Jarvis",
                summary="Worked on structured session memory.",
                key_actions=("Added key action memory",),
                open_loops=("Verify prompt context",),
                evidence_windows=("Code - server.py",),
                event_count=10,
                confidence=0.88,
                summary_source="ollama",
            )
        ])

        relevant_session_context = format_session_context([
            ActivitySession(
                start_at="2026-05-15T17:50:00+00:00",
                end_at="2026-05-15T18:00:00+00:00",
                label="Older Jarvis SQLite work",
                category="Jarvis",
                summary="Worked on SQLite recall for Jarvis memory.",
                key_actions=("Added FTS search",),
                open_loops=("Try jarvis ask recall",),
                evidence_windows=("Code - memory.py",),
                event_count=10,
                confidence=0.84,
                summary_source="ollama",
            )
        ])

        prompt = build_ask_prompt(
            "what was I doing?",
            events,
            30,
            session_context=session_context,
            relevant_session_context=relevant_session_context,
            timezone_name="America/Los_Angeles",
            location="unknown",
        )

        self.assertIn("Current context:", prompt)
        self.assertIn("Clock:", prompt)
        self.assertIn("Timezone reference: America/Los_Angeles", prompt)
        self.assertIn("User question:\nwhat was I doing?", prompt)
        self.assertIn("Recent raw window timeline, last 30 minutes", prompt)
        self.assertIn("Safari - Ollama docs", prompt)
        self.assertIn("Jarvis memory work", prompt)
        self.assertIn("category: Jarvis", prompt)
        self.assertIn("Key actions: Added key action memory", prompt)
        self.assertIn("Open loops: Verify prompt context", prompt)
        self.assertIn("Evidence windows: Code - server.py", prompt)
        self.assertIn("Relevant older session memories", prompt)
        self.assertIn("Older Jarvis SQLite work", prompt)
        self.assertIn("Key actions: Added FTS search", prompt)
        self.assertIn("Do not invent details", prompt)

    def test_build_ask_prompt_falls_back_when_no_relevant_memories(self) -> None:
        prompt = build_ask_prompt(
            "what was I doing?",
            [],
            30,
            session_context="- 18:00-18:05 Jarvis: Worked on Jarvis.",
        )

        self.assertIn("Relevant older session memories", prompt)
        self.assertIn("No older matching session memories were found", prompt)


    def test_build_ask_prompt_includes_profile_previous_asks_stats_and_oldest_marker(self) -> None:
        prompt = build_ask_prompt(
            "but was that a lot?",
            [],
            20,
            recent_ask_context="- 12:00 did I switch windows a lot?",
            user_profile_context="- name: Otzar",
            window_stats_context="- Switches: 12\n- Switching level: moderate",
            oldest_memory_context="- Oldest recorded window event: 2026-05-16T12:00:00+00:00 ChatGPT",
        )

        self.assertIn("User profile:\n- name: Otzar", prompt)
        self.assertIn("Previous Jarvis ask commands", prompt)
        self.assertIn("did I switch windows a lot?", prompt)
        self.assertIn("Switching level: moderate", prompt)
        self.assertIn("Oldest recorded window event", prompt)

    def test_format_window_stats_reports_switch_rate(self) -> None:
        events = [
            WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:00:00+00:00", "macbook"),
            WindowSnapshot("Code", "server.py", "2026-05-16T18:01:00+00:00", "macbook"),
            WindowSnapshot("Safari", "Docs", "2026-05-16T18:02:00+00:00", "macbook"),
        ]

        stats = format_window_stats(events, 20)

        self.assertIn("Switches: 2", stats)
        self.assertIn("Switches per minute: 1.00", stats)
        self.assertIn("Unique apps: 3", stats)

    def test_format_user_profile_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(
                json.dumps({"name": "Otzar", "projects": ["Jarvis"]}),
                encoding="utf-8",
            )

            profile = format_user_profile(profile_path)

            self.assertIn("- name: Otzar", profile)
            self.assertIn('- projects: ["Jarvis"]', profile)


class SmartSummaryTests(unittest.TestCase):
    def test_parse_session_summary_response_uses_valid_structured_json(self) -> None:
        fallback = render_fallback_session()

        session = parse_session_summary_response(
            json.dumps({
                "label": "Jarvis memory work",
                "category": "Jarvis",
                "summary": "Worked on Jarvis session memory.",
                "key_actions": ["Added structured summary fields"],
                "open_loops": ["Check ask prompt output"],
                "evidence_windows": ["Code - memory.py"],
                "confidence": 0.82,
            }),
            fallback,
        )

        self.assertEqual(session.label, "Jarvis memory work")
        self.assertEqual(session.category, "Jarvis")
        self.assertEqual(session.summary, "Worked on Jarvis session memory.")
        self.assertEqual(session.key_actions, ("Added structured summary fields",))
        self.assertEqual(session.open_loops, ("Check ask prompt output",))
        self.assertEqual(session.evidence_windows, ("Code - memory.py",))
        self.assertEqual(session.confidence, 0.82)
        self.assertEqual(session.summary_source, "ollama")

    def test_parse_session_summary_response_falls_back_on_invalid_json(self) -> None:
        fallback = render_fallback_session()

        session = parse_session_summary_response("not json", fallback)

        self.assertEqual(session, fallback)


def render_fallback_session():
    return ActivitySession(
        start_at="2026-05-16T18:00:00+00:00",
        end_at="2026-05-16T18:05:00+00:00",
        label="Coding",
        summary="Worked in coding tools: Terminal.",
        event_count=5,
        confidence=0.8,
        category="Coding",
        evidence_windows=("Terminal - jarvis",),
    )


class MemoryEndpointTests(unittest.TestCase):
    def test_memory_recent_returns_sessions(self) -> None:
        with memory_state() as state:
            session = sample_session("Jarvis timezone memory")
            state._memory.replace_sessions([session])

            sessions = state.memory_recent(100000)

            self.assertEqual(sessions[0].label, "Jarvis timezone memory")
            self.assertIsNotNone(sessions[0].id)

    def test_memory_search_uses_fts(self) -> None:
        with memory_state() as state:
            state._memory.replace_sessions([
                sample_session("Jarvis timezone memory", summary="Worked on timezone profile recall."),
                sample_session(
                    "School work",
                    start_at="2026-05-16T19:00:00+00:00",
                    end_at="2026-05-16T19:05:00+00:00",
                    summary="Reviewed course notes.",
                ),
            ])

            sessions = state.memory_search("timezone")

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].label, "Jarvis timezone memory")

    def test_memory_session_includes_compacted_raw_events(self) -> None:
        with memory_state() as state:
            session = sample_session("Jarvis raw events")
            state._memory.replace_sessions([session])
            session_id = state._memory.search_sessions("raw events")[0].id
            for snapshot in [
                WindowSnapshot("Code", "memory.py", "2026-05-16T18:00:00+00:00", "macbook"),
                WindowSnapshot("Code", "memory.py", "2026-05-16T18:01:00+00:00", "macbook"),
                WindowSnapshot("Terminal", "jarvis", "2026-05-16T18:02:00+00:00", "macbook"),
            ]:
                state._memory.insert_window_event(snapshot)

            payload = state.memory_session_detail(session_id)

            self.assertEqual(payload["session"]["label"], "Jarvis raw events")
            self.assertEqual(len(payload["raw_events"]), 2)
            self.assertEqual(payload["raw_events"][0]["app_name"], "Code")
            self.assertEqual(payload["raw_events"][1]["app_name"], "Terminal")

    def test_memory_stats_returns_counts(self) -> None:
        with memory_state() as state:
            state._memory.insert_window_event(
                WindowSnapshot("Code", "memory.py", "2026-05-16T18:00:00+00:00", "macbook")
            )
            state._memory.replace_sessions([
                sample_session("Ollama summary", summary_source="ollama"),
                sample_session("Heuristic summary", start_at="2026-05-16T19:00:00+00:00", end_at="2026-05-16T19:05:00+00:00"),
            ])

            payload = state.memory_stats()

            self.assertEqual(payload["window_event_count"], 1)
            self.assertEqual(payload["session_count"], 2)
            self.assertEqual(payload["ollama_summary_count"], 1)
            self.assertEqual(payload["heuristic_summary_count"], 1)
            self.assertTrue(payload["fts_available"])

    def test_memory_empty_db_is_graceful(self) -> None:
        with memory_state() as state:
            recent = state.memory_recent(4)
            search = state.memory_search("timezone")
            stats = state.memory_stats()

            self.assertEqual(recent, [])
            self.assertEqual(search, [])
            self.assertIsNone(stats["oldest_event_time"])
            self.assertEqual(stats["window_event_count"], 0)


def sample_session(
    label: str,
    start_at: str = "2026-05-16T18:00:00+00:00",
    end_at: str = "2026-05-16T18:05:00+00:00",
    summary: str = "Worked on Jarvis memory inspection.",
    summary_source: str = "heuristic",
) -> ActivitySession:
    return ActivitySession(
        start_at=start_at,
        end_at=end_at,
        label=label,
        category="Jarvis",
        summary=summary,
        event_count=5,
        confidence=0.8,
        summary_source=summary_source,
        key_actions=("Inspected memory",),
        open_loops=("Verify CLI output",),
        evidence_windows=("Code - memory.py",),
    )


class memory_state:
    def __enter__(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.state = State(root / "events.jsonl", root / "memory.sqlite")
        return self.state

    def __exit__(self, exc_type, exc, tb):
        self.tmpdir.cleanup()


class OllamaChatTests(unittest.TestCase):
    @unittest.mock.patch("jarvis.mac_mini.server.urlopen")
    def test_open_ollama_chat_disables_thinking(self, mock_urlopen) -> None:
        config = OllamaConfig("http://127.0.0.1:11434", "gemma4.e4b", 12.0)

        open_ollama_chat("hello", config)

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "gemma4.e4b")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertIs(payload["stream"], True)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 12.0)


if __name__ == "__main__":
    unittest.main()
