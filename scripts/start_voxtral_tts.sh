#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "python3 not found" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

VOICE="${JARVIS_TTS_VOICE:-neutral_male}"
SEED="${JARVIS_TTS_SEED:-42}"
GAIN="${JARVIS_TTS_GAIN:-1.0}"
MODEL="${JARVIS_TTS_MODEL:-mlx-community/Voxtral-4B-TTS-2603-mlx-4bit}"

exec "$PYTHON" -m jarvis.tts_server \
  --host 127.0.0.1 \
  --port 28766 \
  --model "$MODEL" \
  --voice "$VOICE" \
  --seed "$SEED" \
  --gain "$GAIN" \
  --preload \
  "$@"
