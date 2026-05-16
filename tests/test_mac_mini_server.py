import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.server import OllamaConfig, State, build_ask_prompt, format_window_timeline, open_ollama_chat


class StateTests(unittest.TestCase):
    def test_set_latest_window_appends_jsonl_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_log = Path(tmpdir) / ".jarvis" / "window-events.jsonl"
            state = State(event_log)
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

        prompt = build_ask_prompt("what was I doing?", events, 30)

        self.assertIn("User question:\nwhat was I doing?", prompt)
        self.assertIn("Recent window timeline, last 30 minutes", prompt)
        self.assertIn("Safari - Ollama docs", prompt)
        self.assertIn("Do not invent details", prompt)


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
