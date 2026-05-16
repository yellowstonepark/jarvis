from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json


@dataclass(frozen=True)
class WindowSnapshot:
    app_name: str
    window_title: str | None
    observed_at: str
    source: str

    @classmethod
    def now(
        cls,
        app_name: str,
        window_title: str | None,
        source: str = "local-mac",
    ) -> "WindowSnapshot":
        return cls(
            app_name=app_name,
            window_title=window_title,
            observed_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        )

    def display(self) -> str:
        if self.window_title:
            return f"{self.app_name} - {self.window_title}"
        return self.app_name

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "WindowSnapshot":
        data = json.loads(payload)
        return cls(
            app_name=data["app_name"],
            window_title=data.get("window_title"),
            observed_at=data["observed_at"],
            source=data["source"],
        )

