"""Local Kokoro TTS server for Jarvis (Apple Silicon via mlx-audio)."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger("jarvis-tts")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 28766
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "am_adam"
DEFAULT_LANG = "a"
VOICE_CONFIG_FILE = Path("~/.jarvis/tts-voice").expanduser()
GAIN_CONFIG_FILE = Path("~/.jarvis/tts-gain").expanduser()
LANG_CONFIG_FILE = Path("~/.jarvis/tts-lang").expanduser()
MAX_TEXT_CHARS = 4000
FADE_IN_MS = 30
WARMUP_TEXT = "Good evening."
STREAM_AUDIO_HEADER = "pcm_s16le;rate=24000;channels=1"
DEFAULT_OUTPUT_GAIN = float(os.environ.get("JARVIS_TTS_GAIN", "1.0"))
OUTPUT_TARGET_PEAK = float(os.environ.get("JARVIS_TTS_TARGET_PEAK", "0.18"))
STREAM_PCM_SCALE = float(os.environ.get("JARVIS_TTS_STREAM_SCALE", "0.24"))

_model = None
_model_lock = threading.Lock()
_model_settings: dict[str, str | int | float] = {}


def configured_voice(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if VOICE_CONFIG_FILE.exists():
        value = VOICE_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return DEFAULT_VOICE


def configured_lang_code(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if LANG_CONFIG_FILE.exists():
        value = LANG_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return DEFAULT_LANG


def configured_gain(explicit: float | None = None) -> float:
    if explicit is not None and explicit > 0:
        return explicit
    if GAIN_CONFIG_FILE.exists():
        raw = GAIN_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if raw:
            try:
                value = float(raw)
                if value > 0:
                    return value
            except ValueError:
                LOGGER.warning("invalid tts gain in %s: %r", GAIN_CONFIG_FILE, raw)
    return DEFAULT_OUTPUT_GAIN


def prepare_output_levels(audio: np.ndarray) -> np.ndarray:
    """Attenuate loud output only — never amplify."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio

    gain = float(_model_settings.get("gain", DEFAULT_OUTPUT_GAIN))
    target_peak = min(0.5, OUTPUT_TARGET_PEAK * gain)
    if peak <= target_peak:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def load_tts_model(model_path: str):
    from mlx_audio.tts.utils import load

    LOGGER.info("loading model %s (first run downloads weights)", model_path)
    return load(model_path)


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_tts_model(_model_settings["model_path"])
    return _model


def apply_fade_in(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = min(audio.size, int(sample_rate * FADE_IN_MS / 1000))
    if fade_samples <= 1:
        return audio

    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    audio = audio.copy()
    audio[:fade_samples] *= fade
    return audio


def generate_waveform(model, text: str, voice: str) -> np.ndarray:
    lang_code = str(_model_settings.get("lang_code", DEFAULT_LANG))
    speed = float(_model_settings.get("speed", 1.0))
    segments: list[np.ndarray] = []
    for result in model.generate(
        text=text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
    ):
        chunk = np.asarray(result.audio, dtype=np.float32).squeeze()
        if chunk.size:
            segments.append(chunk)
    if not segments:
        raise RuntimeError("kokoro returned no audio")
    return np.concatenate(segments)


def pcm16_bytes(audio: np.ndarray, *, stream_chunk: bool = False) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.size == 0:
        return b""
    if stream_chunk:
        audio = audio * STREAM_PCM_SCALE
    else:
        audio = prepare_output_levels(audio)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def iter_pcm_chunks(model, text: str, voice: str) -> Iterator[bytes]:
    """Yield PCM s16le chunks per Kokoro pipeline segment."""
    sample_rate = int(getattr(model, "sample_rate", 24_000))
    lang_code = str(_model_settings.get("lang_code", DEFAULT_LANG))
    speed = float(_model_settings.get("speed", 1.0))
    first_chunk = True
    for result in model.generate(
        text=text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
    ):
        audio = np.asarray(result.audio, dtype=np.float32).squeeze()
        if audio.size == 0:
            continue
        if first_chunk:
            audio = apply_fade_in(audio, sample_rate)
            first_chunk = False
        chunk = pcm16_bytes(audio, stream_chunk=True)
        if chunk:
            yield chunk


def warmup_model(voice: str) -> None:
    try:
        synthesize_wav(WARMUP_TEXT, voice)
        LOGGER.info("warmup synthesis complete")
    except Exception as error:
        LOGGER.warning("warmup synthesis failed: %s", error)


def prepare_synthesis(text: str, voice: str) -> tuple[Any, str, str]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must be a non-empty string")
    if len(cleaned) > MAX_TEXT_CHARS:
        cleaned = cleaned[:MAX_TEXT_CHARS]
    return get_model(), cleaned, voice.strip()


def synthesize_wav(text: str, voice: str) -> bytes:
    model, cleaned, voice_name = prepare_synthesis(text, voice)
    sample_rate = int(getattr(model, "sample_rate", 24_000))

    audio = generate_waveform(model, cleaned, voice_name)
    if audio.ndim != 1:
        raise RuntimeError(f"unexpected audio shape: {audio.shape}")

    audio = apply_fade_in(audio, sample_rate)
    audio = prepare_output_levels(audio)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def build_handler():
    class JarvisTTSHandler(BaseHTTPRequestHandler):
        server_version = "JarvisTTS/0.2"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.write_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "engine": "kokoro",
                        "model": _model_settings["model_path"],
                        "voice": _model_settings["voice"],
                        "lang_code": _model_settings["lang_code"],
                    },
                )
                return
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path in {"/v1/speak", "/v1/audio/speech", "/v1/speak/stream"}:
                self.handle_speak(stream=path.endswith("/stream"))
                return
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def write_http_chunk(self, data: bytes) -> None:
            if not data:
                return
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def handle_speak(self, stream: bool = False) -> None:
            length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")

            try:
                payload = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError as error:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid json: {error}"})
                return

            text = payload.get("text") or payload.get("input")
            if not isinstance(text, str) or not text.strip():
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "text or input is required"})
                return

            voice = payload.get("voice", _model_settings["voice"])
            if not isinstance(voice, str) or not voice.strip():
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": "voice must be a string"})
                return

            use_stream = stream or payload.get("stream") is True
            if use_stream:
                self.handle_speak_stream(text, voice.strip())
                return

            try:
                wav = synthesize_wav(text, voice.strip())
            except ValueError as error:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:
                LOGGER.exception("synthesis failed")
                self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "audio/wav")
            self.send_header("content-length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)

        def handle_speak_stream(self, text: str, voice: str) -> None:
            try:
                model, cleaned, voice_name = prepare_synthesis(text, voice)
            except ValueError as error:
                self.write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("x-jarvis-audio-format", STREAM_AUDIO_HEADER)
            self.send_header("transfer-encoding", "chunked")
            self.send_header("cache-control", "no-cache")
            self.end_headers()

            try:
                for chunk in iter_pcm_chunks(model, cleaned, voice_name):
                    self.write_http_chunk(chunk)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception as error:
                LOGGER.exception("stream synthesis failed")
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
                LOGGER.error("stream ended with error: %s", error)

        def log_message(self, format: str, *args) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return JarvisTTSHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis-tts",
        description="Run local Kokoro TTS for Jarvis on Apple Silicon.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice preset (e.g. am_adam, am_echo, am_michael, bm_george).",
    )
    parser.add_argument(
        "--lang-code",
        default=None,
        help="Language code (a=US English, b=British).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = normal).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the model before accepting requests (recommended).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Scales output peak cap (default 1.0; values >1 are louder, <1 quieter).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    global _model_settings
    voice = configured_voice(args.voice) if args.voice else configured_voice(DEFAULT_VOICE)
    _model_settings = {
        "model_path": args.model,
        "voice": voice,
        "gain": configured_gain(args.gain),
        "lang_code": configured_lang_code(args.lang_code),
        "speed": args.speed,
    }

    if args.preload:
        get_model()
        warmup_model(str(_model_settings["voice"]))

    server = HTTPServer((args.host, args.port), build_handler())
    LOGGER.info(
        "jarvis-tts listening on http://%s:%s (kokoro model=%s voice=%s lang=%s)",
        args.host,
        args.port,
        args.model,
        _model_settings["voice"],
        _model_settings["lang_code"],
    )
    LOGGER.info(
        "endpoints: GET /health  POST /v1/speak  POST /v1/speak/stream  POST /v1/audio/speech"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
