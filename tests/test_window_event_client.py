import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from jarvis.common.models import WindowSnapshot
from jarvis.mac_agent.client import AskClient, WindowEventClient, WindowEventOutbox, WindowEventSendError, ask_endpoint_from_receiver_url


def snapshot(app_name: str, observed_at: str = "2026-05-16T12:00:00+00:00") -> WindowSnapshot:
    return WindowSnapshot(
        app_name=app_name,
        window_title=None,
        observed_at=observed_at,
        source="macbook",
    )


class WindowEventClientTests(unittest.TestCase):
    @patch("jarvis.mac_agent.client.urlopen")
    def test_send_posts_snapshot_json(self, mock_urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value.status = 202
        mock_urlopen.return_value = response
        item = snapshot("Finder")

        with tempfile.TemporaryDirectory() as tmpdir:
            outbox = WindowEventOutbox(Path(tmpdir) / "outbox.jsonl")
            WindowEventClient(
                "http://mini:8765/v1/window/events",
                timeout=1.5,
                outbox=outbox,
            ).send(item)

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://mini:8765/v1/window/events")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, item.to_json().encode("utf-8"))
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 1.5)

    @patch("jarvis.mac_agent.client.urlopen")
    def test_failed_send_appends_snapshot_to_outbox(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = URLError("offline")
        item = snapshot("Codex")

        with tempfile.TemporaryDirectory() as tmpdir:
            outbox = WindowEventOutbox(Path(tmpdir) / "outbox.jsonl")
            client = WindowEventClient("http://mini:8765/v1/window/events", outbox=outbox)

            with self.assertRaises(WindowEventSendError):
                client.send(item)

            self.assertEqual(outbox.read_all(), [item])

    @patch("jarvis.mac_agent.client.urlopen")
    def test_send_flushes_queued_snapshots_before_current_snapshot(self, mock_urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value.status = 202
        mock_urlopen.return_value = response
        queued = snapshot("Safari", "2026-05-16T12:00:00+00:00")
        current = snapshot("Codex", "2026-05-16T12:00:01+00:00")

        with tempfile.TemporaryDirectory() as tmpdir:
            outbox = WindowEventOutbox(Path(tmpdir) / "outbox.jsonl")
            outbox.append(queued)
            client = WindowEventClient("http://mini:8765/v1/window/events", outbox=outbox)

            client.send(current)

            sent_bodies = [call.args[0].data for call in mock_urlopen.call_args_list]
            self.assertEqual(
                sent_bodies,
                [queued.to_json().encode("utf-8"), current.to_json().encode("utf-8")],
            )
            self.assertEqual(outbox.read_all(), [])


class AskEndpointTests(unittest.TestCase):
    def test_ask_endpoint_from_window_event_receiver_url(self) -> None:
        self.assertEqual(
            ask_endpoint_from_receiver_url("http://100.110.15.28:8765/v1/window/events"),
            "http://100.110.15.28:8765/v1/ask",
        )

    def test_ask_endpoint_from_base_receiver_url(self) -> None:
        self.assertEqual(
            ask_endpoint_from_receiver_url("http://100.110.15.28:8765"),
            "http://100.110.15.28:8765/v1/ask",
        )


class AskClientTests(unittest.TestCase):
    @patch("jarvis.mac_agent.client.urlopen")
    def test_stream_sends_window_history_settings(self, mock_urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = [b"o", b"k", b""]
        mock_urlopen.return_value = response
        chunks: list[str] = []

        AskClient("http://mini:8765/v1/ask").stream(
            "what was I doing?",
            chunks.append,
            with_window_history=True,
            history_minutes=45,
            max_history_events=120,
            timezone_name="America/Los_Angeles",
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://mini:8765/v1/ask")
        self.assertEqual(payload["prompt"], "what was I doing?")
        self.assertIs(payload["with_window_history"], True)
        self.assertEqual(payload["history_minutes"], 45)
        self.assertEqual(payload["max_history_events"], 120)
        self.assertEqual(payload["timezone"], "America/Los_Angeles")
        self.assertEqual("".join(chunks), "ok")


if __name__ == "__main__":
    unittest.main()
