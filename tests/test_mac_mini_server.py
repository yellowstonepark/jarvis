import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime
from pathlib import Path

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.server import OllamaConfig, State, build_ask_prompt, format_window_timeline, open_ollama_chat, parse_session_summary_response
from jarvis.mac_mini.memory import MemoryStore, render_sessions


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

        prompt = build_ask_prompt(
            "what was I doing?",
            events,
            30,
            timezone_name="America/Los_Angeles",
            location="unknown",
        )

        self.assertIn("Current context:", prompt)
        self.assertIn("Timezone: America/Los_Angeles", prompt)
        self.assertIn("Location: unknown", prompt)
        self.assertIn("User question:\nwhat was I doing?", prompt)
        self.assertIn("Recent raw window timeline, last 30 minutes", prompt)
        self.assertIn("Safari - Ollama docs", prompt)
        self.assertIn("Do not invent details", prompt)


class SmartSummaryTests(unittest.TestCase):
    def test_parse_session_summary_response_uses_valid_json(self) -> None:
        fallback = render_fallback_session()

        label, summary = parse_session_summary_response(
            '{"label":"Jarvis","summary":"Worked on Jarvis session memory."}',
            fallback,
        )

        self.assertEqual(label, "Jarvis")
        self.assertEqual(summary, "Worked on Jarvis session memory.")

    def test_parse_session_summary_response_falls_back_on_invalid_json(self) -> None:
        fallback = render_fallback_session()

        label, summary = parse_session_summary_response("not json", fallback)

        self.assertEqual(label, fallback.label)
        self.assertEqual(summary, fallback.summary)


def render_fallback_session():
    from jarvis.mac_mini.memory import ActivitySession

    return ActivitySession(
        start_at="2026-05-16T18:00:00+00:00",
        end_at="2026-05-16T18:05:00+00:00",
        label="Coding",
        summary="Worked in coding tools: Terminal.",
        event_count=5,
        confidence=0.8,
    )


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
