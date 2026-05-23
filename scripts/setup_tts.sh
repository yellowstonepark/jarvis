#!/usr/bin/env bash
# Install deps, download Kokoro weights, write ~/.jarvis TTS config, and smoke-test synthesis.
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

MODEL="${JARVIS_TTS_MODEL:-mlx-community/Kokoro-82M-bf16}"
VOICE="${JARVIS_TTS_VOICE:-am_adam}"
LANG="${JARVIS_TTS_LANG:-a}"
TTS_URL="${JARVIS_TTS_URL:-http://127.0.0.1:28766/v1/speak}"
CONFIG_DIR="${HOME}/.jarvis"

echo "==> Installing Python dependencies (mlx-audio + misaki)..."
if command -v uv >/dev/null 2>&1; then
  uv sync
else
  echo "uv not found; ensure jarvis .venv has mlx-audio and misaki[en] installed." >&2
fi

mkdir -p "$CONFIG_DIR"
printf "%s\n" "$TTS_URL" >"$CONFIG_DIR/tts-url"
printf "%s\n" "$VOICE" >"$CONFIG_DIR/tts-voice"
printf "%s\n" "$LANG" >"$CONFIG_DIR/tts-lang"

echo "==> Wrote ${CONFIG_DIR}/tts-url       -> ${TTS_URL}"
echo "==> Wrote ${CONFIG_DIR}/tts-voice     -> ${VOICE}"
echo "==> Wrote ${CONFIG_DIR}/tts-lang      -> ${LANG} (a=US English, b=British)"

echo "==> Downloading Kokoro model and running warmup synthesis (first run may take a minute)..."
"$PYTHON" <<PY
import io
import soundfile as sf

from jarvis.tts_server import _model_settings, load_tts_model, synthesize_wav

model_path = "${MODEL}"
voice = "${VOICE}"
lang = "${LANG}"

_model_settings.update(
    {
        "model_path": model_path,
        "voice": voice,
        "lang_code": lang,
        "speed": 1.0,
        "gain": 1.0,
    }
)

load_tts_model(model_path)
wav = synthesize_wav("Good evening. Kokoro is ready.", voice)
data, sr = sf.read(io.BytesIO(wav))
print(f"warmup ok: samples={len(data)} sample_rate={sr}", flush=True)
PY

chmod +x "$ROOT/scripts/start_tts.sh" "$ROOT/scripts/setup_tts.sh" 2>/dev/null || true

echo ""
echo "Setup complete."
echo ""
echo "Start TTS:"
echo "  ./scripts/start_tts.sh"
echo ""
echo "Hotkey app:"
echo "  PYTHONPATH=src .venv/bin/python scripts/build_hotkey_app.py"
echo "  open dist/JarvisHotkey.app"
echo ""
echo "Other male voices: am_echo, am_michael, bm_george (set ~/.jarvis/tts-voice)"
