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

Ask Jarvis from the MacBook. By default, `/v1/ask` injects a compact timeline from the Mac mini's recent `~/.jarvis/window-events.jsonl` events:

```sh
jarvis ask "what was I doing recently?"
jarvis ask --history-minutes 60 "summarize my last hour"
jarvis ask --no-window-history "what is the fastest way to test this?"
```

The Mac mini runs a background summary worker that periodically compresses stable window sessions with Ollama and caches them in SQLite. `jarvis ask` uses cached recent summaries, searchable older session memories, the latest five explicit Jarvis ask commands, window switching stats, your optional `~/.jarvis/profile.json`, and recent raw window events. It also includes current UTC/local time plus the MacBook timezone, so the interactive path stays fast and relative-time questions work:

```sh
jarvis ask "what did I work on recently?"
jarvis ask "summarize my last two hours"
jarvis ask "was I focused or bouncing around?"
jarvis ask "what time is it right now?"
```

Optional user profile context can live at:

```sh
~/.jarvis/profile.json
```

Example:

```json
{
  "name": "Otzar",
  "timezone": "America/Los_Angeles",
  "projects": ["Jarvis", "Actuate", "Shwaz"]
}
```


Inspect Jarvis memory without asking Ollama:

```sh
jarvis memory recent --hours 4
jarvis memory search "Jarvis memory"
jarvis memory session 12
jarvis memory stats
```

These commands call the Mac mini over HTTP and show the SQLite-backed sessions, structured summaries, source evidence, compacted raw events, and database stats.

Use plain chatbot mode when you do not want Jarvis to include window memory:

```sh
jarvis ask --no-window-history "what is the fastest way to test this?"
```

If `~/.jarvis/receiver-url` is not configured, pass the ask endpoint explicitly:

```sh
jarvis ask --ask-url http://100.110.15.28:8765/v1/ask "what was I doing recently?"
```

By default, received window events are appended on the Mac mini to JSONL and also written into SQLite. Terminal windows whose title contains `jarvis ask` are ignored so Jarvis interactions do not pollute activity history:

```sh
~/.jarvis/window-events.jsonl
~/.jarvis/jarvis.sqlite
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


Build the push-to-talk macOS hotkey app:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_hotkey_app.py
open dist/JarvisHotkey.app
```

`JarvisHotkey.app` is a separate accessory app. Hold `Caps Lock` to record, release to stop, then it transcribes with Apple Speech, sends the transcript to the Mac mini `/v1/ask`, and posts the finished answer as a macOS notification. It reads the same `~/.jarvis/receiver-url` file as the main app/CLI.

Permissions needed on first run:

- Input Monitoring for the global shortcut event tap.
- Microphone for recording.
- Speech Recognition for Apple Speech transcription.
- Notifications for the final answer.

Hotkey logs are written to:

```sh
~/Library/Logs/Jarvis/jarvis-hotkey.log
```

Stop the hotkey app:

```sh
killall JarvisHotkey
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
# watchman-agentic
