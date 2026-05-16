from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jarvis.common.models import WindowSnapshot


class WindowEventSendError(Exception):
    """Raised when a window snapshot cannot be sent to the receiver."""


@dataclass(frozen=True)
class WindowEventClient:
    endpoint: str
    timeout: float = 3.0

    def send(self, snapshot: WindowSnapshot) -> None:
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
