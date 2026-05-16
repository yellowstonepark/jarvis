import unittest
from unittest.mock import MagicMock, patch

from jarvis.common.models import WindowSnapshot
from jarvis.mac_agent.client import WindowEventClient


class WindowEventClientTests(unittest.TestCase):
    @patch("jarvis.mac_agent.client.urlopen")
    def test_send_posts_snapshot_json(self, mock_urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value.status = 202
        mock_urlopen.return_value = response
        snapshot = WindowSnapshot(
            app_name="Finder",
            window_title=None,
            observed_at="2026-05-16T12:00:00+00:00",
            source="macbook",
        )

        WindowEventClient("http://mini:8765/v1/window/events", timeout=1.5).send(snapshot)

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://mini:8765/v1/window/events")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, snapshot.to_json().encode("utf-8"))
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 1.5)


if __name__ == "__main__":
    unittest.main()
