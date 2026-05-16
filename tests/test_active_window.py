import unittest
from unittest.mock import patch

from jarvis.mac_agent.window import ActiveWindowError, get_active_window, parse_active_window
import subprocess


class ParseActiveWindowTests(unittest.TestCase):
    def test_parse_active_window_with_title(self) -> None:
        active_window = parse_active_window("Safari\nOpenAI\n")

        self.assertEqual(active_window.app_name, "Safari")
        self.assertEqual(active_window.window_title, "OpenAI")
        self.assertEqual(active_window.display(), "Safari - OpenAI")

    def test_parse_active_window_without_title(self) -> None:
        active_window = parse_active_window("Finder\n\n")

        self.assertEqual(active_window.app_name, "Finder")
        self.assertIsNone(active_window.window_title)
        self.assertEqual(active_window.display(), "Finder")

    def test_parse_active_window_rejects_empty_output(self) -> None:
        with self.assertRaisesRegex(ActiveWindowError, "empty active application"):
            parse_active_window("")

    @patch("jarvis.mac_agent.window.subprocess.run")
    def test_get_active_window_reports_timeout(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(["osascript"], timeout=3)

        with self.assertRaisesRegex(ActiveWindowError, "Timed out"):
            get_active_window()


if __name__ == "__main__":
    unittest.main()
