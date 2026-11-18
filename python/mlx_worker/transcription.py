"""Whisper speech-to-text serving through ``mlx-audio``.

The worker runs each transcription to completion on its model thread. While a
transcription runs, the worker does not advance text or VLM scheduling, so a
long upload delays concurrent chat requests by its inference time.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import shutil
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .ipc import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionWord,
    WorkerCommandError,
)

SAMPLE_RATE = 16_000
"""Whisper input sample rate in Hz."""

FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Temperatures tried in order when a request asks for temperature 0."""

HOMEBREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
"""Homebrew prefixes searched for ffmpeg when it is not on ``PATH``.

A detached instance runs under launchd, whose ``PATH`` omits Homebrew.
"""

AudioReader = Callable[[bytes], np.ndarray]


def ensure_ffmpeg_on_path(
    candidates: tuple[str, ...] = HOMEBREW_BIN_DIRS,
) -> bool:
    """Append a Homebrew directory holding ffmpeg to ``PATH`` when needed.

    Args:
        candidates: Directories to search, in order.

    Returns:
        Whether ``ffmpeg`` is reachable through ``PATH`` afterwards.
    """

    if shutil.which("ffmpeg") is not None:
        return True
    for directory in candidates:
        if os.access(os.path.join(directory, "ffmpeg"), os.X_OK):
            current = os.environ.get("PATH", "")
            os.environ["PATH"] = (
                f"{current}{os.pathsep}{directory}" if current else directory
            )
            return True
    return False


ModelLoader = Callable[[str], Any]


def read_audio(data: bytes) -> np.ndarray:
    """Decode an uploaded audio file into 16 kHz mono float32 samples.

    WAV, MP3, and FLAC decode in-process with ``miniaudio``. M4A/MP4, Ogg,
    WebM, and other containers need ``ffmpeg`` on ``PATH``.

    Args:
        data: Raw uploaded file bytes.

    Returns:
        A one-dimensional float32 array sampled at 16 kHz.

    Raises:
        ValueError: The bytes are not a decodable audio file, or decoding needs
            ``ffmpeg`` and it is not installed.
    """

    from mlx_audio.audio_io import read

    ensure_ffmpeg_on_path()
    try:
        samples, _ = read(
            io.BytesIO(data),
            dtype="float32",
            sample_rate=SAMPLE_RATE,
            nchannels=1,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        message = " ".join(str(exc).split())
        if "ffmpeg" in message.lower() and shutil.which("ffmpeg") is None:
            raise ValueError(
                "decoding this audio format requires ffmpeg on PATH "
                "(install it with `brew install ffmpeg`), or upload WAV, MP3, "
                "or FLAC"
            ) from exc
        raise ValueError(f"unable to decode audio: {message}") from exc
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("uploaded audio contains no samples")
    return samples


def load_whisper_model(model_ref: str) -> Any:
    """Load a Whisper checkpoint with ``mlx_audio.stt.load``.

    Args:
        model_ref: Hugging Face repository id or local checkpoint path.

    Returns:
        The loaded ``mlx-audio`` Whisper model.

    Raises:
        RuntimeError: The checkpoint is not a Whisper model.
    """

    from mlx_audio.stt import load

    model = load(model_ref)
    if ".whisper." not in type(model).__module__:
        raise RuntimeError(
            f"transcription model {model_ref!r} loaded as "
            f"{type(model).__module__}.{type(model).__name__}; only Whisper "
            "checkpoints are supported"
        )
    return model


def _whisper_languages() -> tuple[dict[str, str], dict[str, str]]:
    from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    return LANGUAGES, TO_LANGUAGE_CODE


class WhisperTranscriptionEngine:
    """Lazily loaded Whisper model that serves transcription requests.

    Args:
        model_ref: Hugging Face repository id or local checkpoint path.
        loader: Loads the model; defaults to :func:`load_whisper_model`.
        audio_reader: Decodes uploads; defaults to :func:`read_audio`.
        languages: ``(code -> name, name -> code)`` tables; default to
            Whisper's tokenizer tables.
    """

    def __init__(
        self,
        model_ref: str,
        *,
        loader: ModelLoader | None = None,
        audio_reader: AudioReader | None = None,
        languages: tuple[dict[str, str], dict[str, str]] | None = None,
    ) -> None:
        self.model_ref = model_ref
        self._loader = loader or load_whisper_model
        self._audio_reader = audio_reader or read_audio
        self._languages = languages
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._loader(self.model_ref)
        return self._model

    def _language_tables(self) -> tuple[dict[str, str], dict[str, str]]:
        if self._languages is None:
            self._languages = _whisper_languages()
        return self._languages

    def _language_code(self, language: str | None) -> str | None:
        if language is None:
            return None
        names, codes = self._language_tables()
        normalized = language.strip().lower()
        if normalized in names:
            return normalized
        if normalized in codes:
            return codes[normalized]
        raise ValueError(f"unsupported language {language!r} for Whisper")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Run one transcription or translation to completion.

        Args:
            request: The decoded gateway request.

        Returns:
            The transcription result with segment and optional word timings.

        Raises:
            ValueError: The request is invalid (bad base64, undecodable audio,
                unsupported language).
        """

        language = self._language_code(request.language)
        try:
            data = base64.b64decode(request.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio_base64 is not valid base64") from exc
        if not data:
            raise ValueError("uploaded audio file is empty")

        decode_started = time.perf_counter()
        samples = self._audio_reader(data)
        audio_decode_latency_ms = _elapsed_ms(decode_started)

        model = self._ensure_model()
        inference_started = time.perf_counter()
        output = model.generate(
            samples,
            task=request.task,
            language=language,
            initial_prompt=request.prompt,
            temperature=(
                FALLBACK_TEMPERATURES
                if request.temperature <= 0.0
                else float(request.temperature)
            ),
            word_timestamps=request.word_timestamps,
            verbose=None,
        )
        inference_latency_ms = _elapsed_ms(inference_started)

        raw_segments = list(getattr(output, "segments", None) or [])
        segments = tuple(
            TranscriptionSegment(
                id=index,
                seek=int(segment.get("seek", 0)),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")),
                tokens=tuple(int(token) for token in segment.get("tokens", [])),
                temperature=float(segment.get("temperature", 0.0)),
                avg_logprob=float(segment.get("avg_logprob", 0.0)),
                compression_ratio=float(segment.get("compression_ratio", 0.0)),
                no_speech_prob=float(segment.get("no_speech_prob", 0.0)),
            )
            for index, segment in enumerate(raw_segments)
        )
        words = tuple(
            TranscriptionWord(
                word=str(word.get("word", "")),
                start=float(word.get("start", 0.0)),
                end=float(word.get("end", 0.0)),
            )
            for segment in raw_segments
            for word in segment.get("words", None) or []
        )
        names, _ = self._language_tables()
        detected = str(getattr(output, "language", None) or language or "en")
        return TranscriptionResponse(
            request_id=request.request_id,
            model=request.model,
            task=request.task,
            text=str(getattr(output, "text", "")),
            language=names.get(detected, detected),
            duration=round(samples.size / SAMPLE_RATE, 3),
            segments=segments,
            words=words if request.word_timestamps else (),
            audio_decode_latency_ms=audio_decode_latency_ms,
            inference_latency_ms=inference_latency_ms,
        )


class TranscriptionService:
    """Routes transcription commands to the configured Whisper model.

    The engine is created on the first request, normally the gateway's
    startup warmup, so a worker without a transcription model never imports
    ``mlx-audio``.

    Args:
        model_ref: Configured transcription model, or ``None`` when audio
            endpoints are disabled.
        engine_factory: Builds the engine; defaults to
            :class:`WhisperTranscriptionEngine`.
    """

    def __init__(
        self,
        model_ref: str | None,
        *,
        engine_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_ref = model_ref
        self._engine_factory = engine_factory or WhisperTranscriptionEngine
        self._engine: Any | None = None

    def handle(
        self, request: TranscriptionRequest
    ) -> TranscriptionResponse | WorkerCommandError:
        """Serve one request and return the event to send to the gateway.

        Args:
            request: The decoded gateway request.

        Returns:
            The transcription result, or an ``INVALID_REQUEST`` error for bad
            input and a ``WORKER_ERROR`` error for model failures.
        """

        if self.model_ref is None or request.model != self.model_ref:
            served = self.model_ref or "<none>"
            return WorkerCommandError(
                code="INVALID_REQUEST",
                request_id=request.request_id,
                message=(
                    f"model '{request.model}' is not served by this worker "
                    f"(serves transcription='{served}')"
                ),
            )
        try:
            if self._engine is None:
                self._engine = self._engine_factory(self.model_ref)
            return self._engine.transcribe(request)
        except ValueError as exc:
            code = "INVALID_REQUEST"
            message = str(exc)
        except Exception as exc:
            code = "WORKER_ERROR"
            message = f"transcription failed: {exc}"
        return WorkerCommandError(
            code=code, request_id=request.request_id, message=message
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
