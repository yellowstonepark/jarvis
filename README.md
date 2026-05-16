# jarvis
an AI assistant running on a Mac Mini

## Repo layout

- `src/jarvis/mac_agent`: code that runs on the Mac being observed.
- `src/jarvis/mac_mini`: code that runs on the Mac mini coordinator.
- `src/jarvis/common`: shared models and protocol data.
- `docs/architecture.md`: communication and permission notes.

## Active window prototype

This first prototype reports the macOS app and window that are currently in
front.

Run once:

```sh
PYTHONPATH=src python3 -m jarvis
```

Print once per second until stopped:

```sh
PYTHONPATH=src python3 -m jarvis --watch
```

Print newline-delimited JSON:

```sh
PYTHONPATH=src python3 -m jarvis --watch --json
```

Run the Mac mini receiver prototype:

```sh
PYTHONPATH=src python3 -m jarvis.mac_mini.server --host MAC_MINI_TAILSCALE_IP --port 8765
```

The Mac mini receiver also exposes `POST /v1/ask`, which streams responses from local Ollama. It defaults to `gemma4.e4b` with Ollama `think: false` and temperature `0`:

```sh
ollama pull gemma4.e4b
uv run jarvis-mini --host 100.110.15.28 --port 8765 --ollama-model gemma4.e4b
```

Ask Jarvis from the MacBook:

```sh
jarvis ask "what is the fastest way to test this?"
```

If `~/.jarvis/receiver-url` is not configured, pass the ask endpoint explicitly:

```sh
jarvis ask --ask-url http://100.110.15.28:8765/v1/ask "what is the fastest way to test this?"
```

By default, received window events are appended on the Mac mini to:

```sh
~/.jarvis/window-events.jsonl
```

You can choose a different JSONL log path:

```sh
PYTHONPATH=src python3 -m jarvis.mac_mini.server --event-log ~/.jarvis/macbook-window-events.jsonl
```

Stream MacBook window events to the Mac mini over Tailscale from the Python CLI:

```sh
PYTHONPATH=src python3 -m jarvis --watch --json --source macbook --send-to http://MAC_MINI_TAILSCALE_NAME:8765/v1/window/events
```

Configure the MacBook `Jarvis.app` to stream to the Mac mini:

```sh
mkdir -p ~/.jarvis
printf "%s\n" "http://MAC_MINI_TAILSCALE_NAME:8765/v1/window/events" > ~/.jarvis/receiver-url
```

If the Mac mini is unreachable, the MacBook CLI/app queues unsent events here and flushes them later in order:

```sh
~/.jarvis/window-outbox.jsonl
```

Build the dedicated macOS app:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_dev_app.py
open dist/Jarvis.app
```

The app writes logs to `~/Library/Logs/Jarvis/jarvis.log`.

Stop any running Jarvis app process:

```sh
killall Jarvis
```

If you previously ran the old Python wrapper, stop that stale process too:

```sh
pkill -f jarvis_app.py
```

If macOS blocks the query from the CLI, Terminal needs Accessibility access.
For regular use, open `dist/Jarvis.app` and grant Accessibility to Jarvis
instead, then remove Terminal from Accessibility.

Install locally on the Mac mini as a Python script/CLI:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
jarvis-mini --host MAC_MINI_TAILSCALE_IP --port 8765
```

Run tests:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests
```
