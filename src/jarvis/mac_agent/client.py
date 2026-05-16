from __future__ import annotations

from dataclasses import dataclass, field
import codecs
import json
import os
from pathlib import Path
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from jarvis.common.models import WindowSnapshot


class WindowEventSendError(Exception):
    """Raised when a window snapshot cannot be sent to the receiver."""


def default_outbox_path() -> Path:
    return Path.home() / ".jarvis" / "window-outbox.jsonl"


@dataclass(frozen=True)
class WindowEventOutbox:
    path: Path = field(default_factory=default_outbox_path)

    def append(self, snapshot: WindowSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(snapshot.to_json())
            file.write("\n")

    def read_all(self) -> list[WindowSnapshot]:
        if not self.path.exists():
            return []

        snapshots: list[WindowSnapshot] = []
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    snapshots.append(WindowSnapshot.from_json(line))
        return snapshots

    def replace(self, snapshots: list[WindowSnapshot]) -> None:
        if not snapshots:
            self.clear()
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            for snapshot in snapshots:
                file.write(snapshot.to_json())
                file.write("\n")
        temporary_path.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class WindowEventClient:
    endpoint: str
    timeout: float = 3.0
    outbox: WindowEventOutbox = field(default_factory=WindowEventOutbox)

    def send(self, snapshot: WindowSnapshot) -> None:
        try:
            self.flush_outbox()
            self._post(snapshot)
        except WindowEventSendError:
            self.outbox.append(snapshot)
            raise

    def flush_outbox(self) -> int:
        queued_snapshots = self.outbox.read_all()
        if not queued_snapshots:
            return 0

        sent_count = 0
        for index, snapshot in enumerate(queued_snapshots):
            try:
                self._post(snapshot)
            except WindowEventSendError:
                self.outbox.replace(queued_snapshots[index:])
                raise
            sent_count += 1

        self.outbox.clear()
        return sent_count

    def _post(self, snapshot: WindowSnapshot) -> None:
        body = snapshot.to_json().encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.status >= 400:
                    raise WindowEventSendError(
                        f"receiver returned HTTP {response.status}"
                    )
        except HTTPError as error:
            raise WindowEventSendError(
                f"receiver returned HTTP {error.code}"
            ) from error
        except URLError as error:
            raise WindowEventSendError(f"could not reach receiver: {error.reason}") from error
        except TimeoutError as error:
            raise WindowEventSendError("timed out sending window event") from error



class AskStreamError(Exception):
    """Raised when Jarvis cannot stream an ask response from the receiver."""


class MemoryInspectError(Exception):
    """Raised when Jarvis memory inspection fails."""


def default_receiver_endpoint() -> str | None:
    explicit_ask_url = os.environ.get("JARVIS_ASK_URL")
    if explicit_ask_url:
        return explicit_ask_url

    receiver_url = os.environ.get("JARVIS_RECEIVER_URL")
    if receiver_url:
        return ask_endpoint_from_receiver_url(receiver_url)

    receiver_file = Path.home() / ".jarvis" / "receiver-url"
    if not receiver_file.exists():
        return None

    receiver_url = receiver_file.read_text(encoding="utf-8").strip()
    if not receiver_url:
        return None

    return ask_endpoint_from_receiver_url(receiver_url)


def local_timezone_name() -> str:
    local_tz = datetime.now().astimezone().tzinfo
    key = getattr(local_tz, "key", None)
    if key:
        return key
    name = datetime.now().astimezone().tzname()
    return name or "UTC"


def ask_endpoint_from_receiver_url(receiver_url: str) -> str:
    if receiver_url.endswith("/v1/window/events"):
        return receiver_url[: -len("/v1/window/events")] + "/v1/ask"
    if receiver_url.endswith("/v1/memory"):
        return receiver_url[: -len("/v1/memory")] + "/v1/ask"
    return receiver_url.rstrip("/") + "/v1/ask"


def memory_endpoint_from_receiver_url(receiver_url: str) -> str:
    if receiver_url.endswith("/v1/window/events"):
        return receiver_url[: -len("/v1/window/events")] + "/v1/memory"
    if receiver_url.endswith("/v1/ask"):
        return receiver_url[: -len("/v1/ask")] + "/v1/memory"
    if receiver_url.endswith("/v1/memory"):
        return receiver_url
    return receiver_url.rstrip("/") + "/v1/memory"


def default_memory_endpoint() -> str | None:
    ask_endpoint = default_receiver_endpoint()
    if ask_endpoint is None:
        return None
    return memory_endpoint_from_receiver_url(ask_endpoint)


@dataclass(frozen=True)
class AskClient:
    endpoint: str
    timeout: float = 60.0

    def stream(
        self,
        prompt: str,
        write,
        with_window_history: bool = True,
        history_minutes: float = 30,
        max_history_events: int = 80,
        timezone_name: str | None = None,
    ) -> None:
        body = json.dumps(
            {
                "prompt": prompt,
                "with_window_history": with_window_history,
                "history_minutes": history_minutes,
                "max_history_events": max_history_events,
                "timezone": timezone_name or local_timezone_name(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                while True:
                    chunk = response.read(1)
                    if not chunk:
                        final_text = decoder.decode(b"", final=True)
                        if final_text:
                            write(final_text)
                        break
                    text = decoder.decode(chunk)
                    if text:
                        write(text)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise AskStreamError(f"receiver returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise AskStreamError(f"could not reach receiver: {error.reason}") from error
        except TimeoutError as error:
            raise AskStreamError("timed out waiting for receiver") from error


@dataclass(frozen=True)
class MemoryClient:
    endpoint: str
    timeout: float = 10.0

    def recent(self, hours: float) -> dict:
        return self._get_json(f"/recent?{urlencode({'hours': hours})}")

    def search(self, query: str, limit: int = 10) -> dict:
        return self._get_json(f"/search?{urlencode({'q': query, 'limit': limit})}")

    def session(self, session_id: int) -> dict:
        return self._get_json(f"/session/{session_id}")

    def stats(self) -> dict:
        return self._get_json("/stats")

    def _get_json(self, path: str) -> dict:
        request = Request(
            self.endpoint.rstrip("/") + path,
            headers={"accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise MemoryInspectError(f"receiver returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise MemoryInspectError(f"could not reach receiver: {error.reason}") from error
        except TimeoutError as error:
            raise MemoryInspectError("timed out waiting for receiver") from error
        except json.JSONDecodeError as error:
            raise MemoryInspectError(f"invalid memory response: {error}") from error
