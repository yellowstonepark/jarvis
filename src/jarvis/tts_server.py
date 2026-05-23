"""Local Voxtral TTS server for Jarvis (Apple Silicon via mlx-audio).

Uses the MLX conversion of mistralai/Voxtral-4B-TTS-2603:
https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections.abc import Iterator
from typing import Any
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger("jarvis-tts")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 28766
DEFAULT_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
# Mistral demo "Paul" / Jarvis-style calm English male preset on the model card.
DEFAULT_VOICE = "neutral_male"
VOICE_CONFIG_FILE = Path("~/.jarvis/tts-voice").expanduser()
GAIN_CONFIG_FILE = Path("~/.jarvis/tts-gain").expanduser()
MAX_TEXT_CHARS = 4000
# Voxtral decodes 1920 samples (80ms) per codec frame; the first frames are often "ah"/breath.
VOXTRAL_SAMPLES_PER_FRAME = 1920
# Time-domain trim (after decode). Kept milder now that semantic-frame trim runs first.
MIN_SKIP_LEADING_CODEC_FRAMES = 2
MAX_LEADING_TRIM_MS = 400
LEADING_RMS_THRESHOLD = 0.04
LEADING_STABLE_HOPS = 3
HIGHPASS_CUTOFF_HZ = 100
FADE_IN_MS = 30
TTS_SEED = 42
MAX_GENERATION_TOKENS = 4096
WARMUP_TEXT = "Good evening."
# Larger chunks = fewer vocoder passes and smoother playback (Voxtral frame = 80ms).
STREAM_CHUNK_SECONDS = float(os.environ.get("JARVIS_TTS_STREAM_CHUNK", "1.0"))
STREAM_CONTEXT_FRAMES = 16
STREAM_AUDIO_HEADER = "pcm_s16le;rate=24000;channels=1"
# Loudness: 1.0 = gentle normalize only. Values above ~1.2 clip and sound harsh.
DEFAULT_OUTPUT_GAIN = float(os.environ.get("JARVIS_TTS_GAIN", "1.0"))
OUTPUT_TARGET_PEAK = 0.85


def configured_voice(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if VOICE_CONFIG_FILE.exists():
        value = VOICE_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return DEFAULT_VOICE


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
    """Bring quiet Voxtral output to a comfortable level without clipping."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio

    gain = float(_model_settings.get("gain", DEFAULT_OUTPUT_GAIN))
    target_peak = min(0.92, OUTPUT_TARGET_PEAK * gain)
    audio = audio * (target_peak / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


_model = None
_model_lock = threading.Lock()
_model_settings: dict[str, str | int | float] = {}


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


def _chunk_rms(audio: np.ndarray, offset: int, hop: int) -> float:
    chunk = audio[offset : offset + hop]
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk * chunk)))


def trim_leading_voxtral_artifact(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Aggressively remove Voxtral's leading vowel filler ('ah') before real speech."""
    if audio.size == 0:
        return audio

    original_samples = audio.size
    min_skip = VOXTRAL_SAMPLES_PER_FRAME * MIN_SKIP_LEADING_CODEC_FRAMES
    max_skip = int(sample_rate * MAX_LEADING_TRIM_MS / 1000)
    if audio.size > min_skip:
        audio = audio[min_skip:]

    hop = max(1, int(sample_rate * 0.02))
    scan_limit = min(audio.size, max_skip)
    threshold = LEADING_RMS_THRESHOLD

    consecutive = 0
    onset_sample = 0
    for offset in range(0, max(0, scan_limit - hop), hop):
        if _chunk_rms(audio, offset, hop) >= threshold:
            consecutive += 1
            if consecutive >= LEADING_STABLE_HOPS:
                onset_sample = offset - (consecutive - 1) * hop
                break
        else:
            consecutive = 0

    if onset_sample > 0:
        audio = audio[onset_sample:]
    elif scan_limit > 0 and float(np.max(np.abs(audio[:scan_limit]))) >= threshold:
        audio = audio[scan_limit:]

    trimmed_ms = (original_samples - audio.size) / sample_rate * 1000
    if trimmed_ms >= 50:
        LOGGER.info("trimmed %.0fms from start of audio", trimmed_ms)

    return audio


def seed_mlx_rng(seed: int) -> None:
    import mlx.core as mx

    mx.random.seed(seed)


def semantic_code_from_frame(frame) -> int:
    """First semantic token in an acoustic frame (stored as (1, 1, C) or (1, C))."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 3:
        return int(arr[0, 0, 0])
    if arr.ndim == 2:
        return int(arr[0, 0])
    flat = arr.reshape(-1)
    if flat.size != 1:
        raise ValueError(f"unexpected frame shape {arr.shape}")
    return int(flat[0])


def trim_repeated_semantic_frames(all_codes: list) -> list:
    """Drop Voxtral warm-up frames that repeat the same semantic code (sounds like echo/ah).

    See mlx-audio discussion on PR #607: repeated codes 10/855 decode to noise bursts.
    """
    if len(all_codes) < 2:
        return all_codes

    first_semantic = semantic_code_from_frame(all_codes[0])
    repeat_end = 0
    for index, frame in enumerate(all_codes):
        semantic = semantic_code_from_frame(frame)
        if semantic != first_semantic:
            break
        repeat_end = index + 1

    if repeat_end <= 1:
        return all_codes

    if repeat_end >= len(all_codes):
        LOGGER.warning(
            "all %d frames shared semantic code %d; keeping last frame only",
            len(all_codes),
            first_semantic,
        )
        return all_codes[-1:]

    LOGGER.info(
        "dropped %d warm-up frames with repeated semantic code %d",
        repeat_end,
        first_semantic,
    )
    return all_codes[repeat_end:]


def generate_waveform(model, text: str, voice: str, max_tokens: int = MAX_GENERATION_TOKENS):
    """Generate audio with semantic-prefix trim before vocoder decode."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if model.tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")

    input_ids = model._encode_text(text, voice)
    input_ids_mx = mx.array(input_ids)[None, :]
    input_embeddings = model._build_input_embeddings(input_ids_mx, voice)
    lm_backbone = model.language_model.model.model

    cache = make_prompt_cache(model.language_model.model)
    hidden = lm_backbone(input_ids_mx, cache=cache, input_embeddings=input_embeddings)

    audio_tok_emb = model.language_model.embed_tokens(
        mx.array([[model.config.audio_token_id]])
    )
    hidden = lm_backbone(
        mx.array([[model.config.audio_token_id]]),
        cache=cache,
        input_embeddings=audio_tok_emb,
    )

    all_codes: list = []
    for step in range(max_tokens):
        codes = model.acoustic_transformer.decode_one_frame(hidden[:, -1, :])
        semantic_code = int(codes[0, 0].item())
        if semantic_code <= 1:
            break

        all_codes.append(codes[:, None, :])
        global_codes = model._codes_to_global_indices(codes)
        code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
        next_embedding = code_embeddings.sum(axis=1, keepdims=True)
        hidden = lm_backbone(
            mx.array([[model.config.audio_token_id]]),
            cache=cache,
            input_embeddings=next_embedding,
        )

        if step % 50 == 0:
            mx.clear_cache()

    if not all_codes:
        raise RuntimeError("model returned no audio frames")

    trimmed_codes = trim_repeated_semantic_frames(all_codes)
    audio_codes = mx.concatenate(trimmed_codes, axis=1)
    return model.audio_tokenizer.decode(audio_codes).squeeze(0)


def warmup_skip_frames(all_codes: list) -> int:
    """Frames to skip at the start (repeated semantic warm-up)."""
    if len(all_codes) < 2:
        return 0
    trimmed = trim_repeated_semantic_frames(all_codes)
    skip = len(all_codes) - len(trimmed)
    return skip if skip > 1 else 0


def decode_codes_window(
    model,
    all_codes: list,
    yielded_frames: int,
    warmup_skip: int,
) -> np.ndarray:
    """Decode new audio since ``yielded_frames``, with codec context overlap."""
    import mlx.core as mx

    ctx_start = max(warmup_skip, yielded_frames - STREAM_CONTEXT_FRAMES)
    chunk_codes = mx.concatenate(all_codes[ctx_start:], axis=1)
    waveform = model.audio_tokenizer.decode(chunk_codes).squeeze(0)
    mx.eval(waveform)
    waveform = np.asarray(waveform, dtype=np.float32)
    trim_samples = (yielded_frames - ctx_start) * VOXTRAL_SAMPLES_PER_FRAME
    if trim_samples > 0:
        waveform = waveform[trim_samples:]
    return waveform


def pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = prepare_output_levels(audio)
    if audio.size == 0:
        return b""
    pcm = (audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


def iter_pcm_chunks(
    model,
    text: str,
    voice: str,
    max_tokens: int = MAX_GENERATION_TOKENS,
    chunk_seconds: float = STREAM_CHUNK_SECONDS,
) -> Iterator[bytes]:
    """Yield PCM s16le chunks while tokens are generated (low time-to-first-audio)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if model.tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")

    input_ids = model._encode_text(text, voice)
    input_ids_mx = mx.array(input_ids)[None, :]
    input_embeddings = model._build_input_embeddings(input_ids_mx, voice)
    lm_backbone = model.language_model.model.model

    cache = make_prompt_cache(model.language_model.model)
    hidden = lm_backbone(input_ids_mx, cache=cache, input_embeddings=input_embeddings)

    audio_tok_emb = model.language_model.embed_tokens(
        mx.array([[model.config.audio_token_id]])
    )
    hidden = lm_backbone(
        mx.array([[model.config.audio_token_id]]),
        cache=cache,
        input_embeddings=audio_tok_emb,
    )

    sample_rate = int(getattr(model, "sample_rate", 24_000))
    frames_per_chunk = max(1, int(chunk_seconds / 0.08))
    all_codes: list = []
    yielded_frames = 0
    warmup_skip = 0
    first_chunk = True

    for step in range(max_tokens):
        codes = model.acoustic_transformer.decode_one_frame(hidden[:, -1, :])
        if int(codes[0, 0].item()) <= 1:
            break

        all_codes.append(codes[:, None, :])
        global_codes = model._codes_to_global_indices(codes)
        code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
        next_embedding = code_embeddings.sum(axis=1, keepdims=True)
        hidden = lm_backbone(
            mx.array([[model.config.audio_token_id]]),
            cache=cache,
            input_embeddings=next_embedding,
        )

        if step % 150 == 0:
            mx.clear_cache()

        warmup_skip = max(warmup_skip, warmup_skip_frames(all_codes))
        if len(all_codes) - max(yielded_frames, warmup_skip) < frames_per_chunk:
            continue

        audio = decode_codes_window(model, all_codes, yielded_frames, warmup_skip)
        if first_chunk:
            audio = apply_fade_in(audio, sample_rate)
            first_chunk = False
        chunk = pcm16_bytes(audio)
        if chunk:
            yield chunk
        yielded_frames = len(all_codes)

    if not all_codes:
        raise RuntimeError("model returned no audio frames")

    if len(all_codes) > yielded_frames:
        audio = decode_codes_window(model, all_codes, yielded_frames, warmup_skip)
        if first_chunk:
            audio = apply_fade_in(audio, sample_rate)
        chunk = pcm16_bytes(audio)
        if chunk:
            yield chunk


def apply_highpass(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    if audio.size < sample_rate // 10:
        return audio

    sos = butter(2, HIGHPASS_CUTOFF_HZ, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def apply_fade_in(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = min(audio.size, int(sample_rate * FADE_IN_MS / 1000))
    if fade_samples <= 1:
        return audio

    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    audio = audio.copy()
    audio[:fade_samples] *= fade
    return audio


def polish_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = trim_leading_voxtral_artifact(audio, sample_rate)
    audio = apply_highpass(audio, sample_rate)
    audio = apply_fade_in(audio, sample_rate)
    return audio


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
        cleaned = cleaned[: MAX_TEXT_CHARS]

    seed = int(_model_settings.get("seed", TTS_SEED))
    seed_mlx_rng(seed)
    return get_model(), cleaned, voice.strip()


def synthesize_wav(text: str, voice: str) -> bytes:
    model, cleaned, voice_name = prepare_synthesis(text, voice)
    sample_rate = int(getattr(model, "sample_rate", 24_000))

    waveform = generate_waveform(model, cleaned, voice_name)
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim != 1:
        raise RuntimeError(f"unexpected audio shape: {audio.shape}")

    audio = polish_waveform(audio, sample_rate)
    audio = prepare_output_levels(audio)

    buffer = io.BytesIO()
    sf.write(buffer, audio, int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def build_handler():
    class JarvisTTSHandler(BaseHTTPRequestHandler):
        server_version = "JarvisTTS/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.write_json(HTTPStatus.OK, {"ok": True, "model": _model_settings["model_path"]})
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
                # Response already started; client will see truncated stream.
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
        description="Run local Voxtral TTS for Jarvis on Apple Silicon.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Preset voice (e.g. neutral_male, casual_male, cheerful_female).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the model before accepting requests (recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TTS_SEED,
        help="MLX RNG seed for reproducible speech (default: 42).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Output loudness multiplier after normalization (default: JARVIS_TTS_GAIN or 1.0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    global _model_settings
    _model_settings = {
        "model_path": args.model,
        "voice": configured_voice(args.voice),
        "seed": args.seed,
        "gain": configured_gain(args.gain),
    }

    if args.preload:
        get_model()
        seed_mlx_rng(args.seed)
        warmup_model(str(_model_settings["voice"]))

    # MLX models must load and run on the same thread (no ThreadingHTTPServer).
    server = HTTPServer((args.host, args.port), build_handler())
    LOGGER.info(
        "jarvis-tts listening on http://%s:%s (model=%s voice=%s)",
        args.host,
        args.port,
        args.model,
        _model_settings["voice"],
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
