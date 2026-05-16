import json
import tempfile
import unittest
from pathlib import Path

from jarvis.common.models import WindowSnapshot
from jarvis.mac_mini.server import State


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


if __name__ == "__main__":
    unittest.main()
