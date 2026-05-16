# Jarvis Architecture

Jarvis is organized as one repo with separate runtime roles:

- `jarvis.mac_agent`: code that runs on the Mac being observed or controlled.
- `jarvis.mac_mini`: code that runs on the Mac mini coordinator.
- `jarvis.common`: shared models and protocol shapes used by both sides.

## Current Prototype

The local Mac agent can read the foreground macOS app/window and print it once
or once per second.

```sh
PYTHONPATH=src python3 -m jarvis --watch
```

For machine-readable output:

```sh
PYTHONPATH=src python3 -m jarvis --watch --json
```

The Mac mini service is a minimal HTTP process with no external dependencies. It can run as a plain Python process; no macOS app bundle is needed on the Mac mini:

```sh
PYTHONPATH=src python3 -m jarvis.mac_mini.server --host MAC_MINI_TAILSCALE_IP --port 8765
```

Received window snapshots are appended as newline-delimited JSON at `~/.jarvis/window-events.jsonl` by default. Use `--event-log` to point it somewhere else.

It currently exposes:

- `GET /health`
- `POST /v1/window/events`
- `GET /v1/window/latest`
- `POST /v1/ask`

`POST /v1/ask` proxies a prompt to local Ollama via `/api/chat` and streams plain text back to the caller. By default it reads recent SQLite-backed session summaries plus recent raw window events, then injects both into the prompt. The default Ollama request uses `model: gemma4.e4b`, `think: false`, `stream: true`, and `temperature: 0`.

The Mac mini keeps JSONL as an append-only backup and stores queryable memory in SQLite at `~/.jarvis/jarvis.sqlite` by default. Current tables:

- `window_events`: one row per observed active-window event.
- `sessions`: on-demand session chunks with label, summary, event count, and confidence.

A background summary worker periodically builds stable session chunks, removes consecutive duplicate windows from the summary prompt, asks Ollama for concise JSON summaries, and caches the result with `summary_source = 'ollama'`. Event ingestion never calls Ollama. `jarvis ask` does not generate summaries synchronously; it consumes cached summaries and recent raw events. If an ask is active, the background worker skips starting new summary work and stops before the next session.

## Communication Direction

The clean default is agent-to-mini push:

1. The observed Mac runs `jarvis`.
2. It reads local context, such as the active window.
3. It sends small JSON events to the Mac mini over Tailscale with `--send-to http://<mac-mini-tailscale-name>:8765/v1/window/events`, or the MacBook app reads the same receiver endpoint from `~/.jarvis/receiver-url`.
4. If the Mac mini is unreachable, the Mac agent queues unsent events at `~/.jarvis/window-outbox.jsonl` and retries later.
5. The Mac mini stores the latest event in memory and appends all received events to JSONL for later context.

This keeps macOS permissions local to the machine being observed. The Mac mini
does not need Accessibility permission for another Mac.

For command/control later, use a second channel from mini-to-agent:

1. Mac agent keeps a connection open to the Mac mini.
2. Mac mini sends requested actions.
3. Agent validates and executes only actions it supports.
4. Agent returns a result event.

That can be implemented as polling first, then upgraded to Server-Sent Events,
WebSockets, or gRPC when the protocol stabilizes.

## Tailscale Notes

Bind services to the Tailscale IP or localhost during development. Avoid binding
to all interfaces until authentication is added.

Recommended first pass:

- Tailscale MagicDNS names for addressing.
- Tailscale ACLs so only your trusted devices can reach the Mac mini service.
- A shared bearer token before any command/control endpoint exists.
- HTTPS is less urgent inside Tailscale, but request authentication still matters.

## macOS Permission Model

Granting Accessibility to Terminal means every process launched from that
Terminal can potentially use that permission. That is convenient for development
but too broad for regular use.

Better long-term options:

- Package Jarvis as its own signed `.app` and grant Accessibility only to that
  app.
- Run Jarvis through a dedicated launch agent or small wrapper app instead of a
  general-purpose shell.
- Keep the permissioned component small: one local agent process reads UI state
  and exposes only the narrow events/actions Jarvis needs.

The current AppleScript/System Events approach is fine for a prototype. A more
native future implementation can use macOS Accessibility APIs directly through a
small Swift helper, but it will still need Accessibility permission. The main
security win is giving that permission to a dedicated Jarvis binary/app, not to
Terminal.

## Phase 1 App Packaging

Phase 1 packages the local sensing loop as `Jarvis.app` with a stable bundle
identifier: `com.otzarjaffe.jarvis`.

The app uses a native Swift executable so macOS sees the permissioned process as
Jarvis, not Terminal or `python3.12`.

Build it with:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_dev_app.py
```

Run it with:

```sh
open dist/Jarvis.app
```

Logs are written to:

```sh
~/Library/Logs/Jarvis/jarvis.log
```

Stop the app with:

```sh
killall Jarvis
```

Once `Jarvis.app` is working, remove Terminal from Accessibility and grant
Accessibility to Jarvis instead. This narrows the permission from a general shell
to the Jarvis app identity.

This is still not the final security boundary. It is ad hoc signed for local
development, not signed with an Apple Developer ID. Keep the app limited to
local sensing and strictly allowlisted actions.

Later, this can move from a development wrapper to a fully standalone signed
bundle with a stable Developer ID signature.
