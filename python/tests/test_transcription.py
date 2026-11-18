from __future__ import annotations

import base64
import io
import json
import os
import shutil
import socket
import threading
import wave
from types import SimpleNamespace
from typing import Any, Self

import numpy as np
import pytest

from mlx_worker.ipc import (
    TranscriptionRequest,
    TranscriptionResponse,
    WorkerCommandError,
    decode_command,
    decode_event,
    encode_event,
)
from mlx_worker.transcription import (
    FALLBACK_TEMPERATURES,
    TranscriptionService,
    WhisperTranscriptionEngine,
    ensure_ffmpeg_on_path,
    read_audio,
)

LANGUAGES = ({"en": "english", "fr": "french"}, {"english": "en", "french": "fr"})


def _request(**overrides: Any) -> TranscriptionRequest:
    values: dict[str, Any] = {
        "request_id": "audio-1",
        "model": "whisper",
        "task": "transcribe",
        "audio_base64": base64.b64encode(b"RIFF....WAVE").decode(),
    }
    values.update(overrides)
    return TranscriptionRequest(**values)


def _command_line(request: TranscriptionRequest) -> bytes:
    payload = {
        "type": "transcription",
        "request": {
            "request_id": request.request_id,
            "model": request.model,
            "task": request.task,
            "audio_base64": request.audio_base64,
            "temperature": request.temperature,
        },
    }
    return (json.dumps(payload) + "\n").encode()


class FakeWhisper:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, audio: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"samples": audio.size, **kwargs})
        return SimpleNamespace(
            text=" Bonjour.",
            language="fr",
            segments=[
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Bonjour.",
                    "tokens": [50365, 12],
                    "temperature": 0.0,
                    "avg_logprob": -0.25,
                    "compression_ratio": 0.7,
                    "no_speech_prob": 0.02,
                    "words": [
                        {
                            "word": " Bonjour.",
                            "start": 0.1,
                            "end": 0.9,
                            "probability": 1,
                        }
                    ],
                }
            ],
        )


def _engine(model: FakeWhisper, reader: Any = None) -> WhisperTranscriptionEngine:
    return WhisperTranscriptionEngine(
        "whisper",
        loader=lambda _ref: model,
        audio_reader=reader or (lambda _data: np.zeros(32_000, dtype=np.float32)),
        languages=LANGUAGES,
    )


def _wav_bytes(sample_rate: int, channels: int, frames: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * frames * channels)
    return buffer.getvalue()


def test_transcription_command_and_event_round_trip() -> None:
    line = (
        b'{"type":"transcription","request":{"request_id":"audio-7","model":"w",'
        b'"task":"translate","audio_base64":"AAAA","filename":"a.wav",'
        b'"prompt":"names","temperature":0.4,"word_timestamps":true}}\n'
    )
    command = decode_command(line)
    assert command == TranscriptionRequest(
        request_id="audio-7",
        model="w",
        task="translate",
        audio_base64="AAAA",
        filename="a.wav",
        prompt="names",
        temperature=0.4,
        word_timestamps=True,
    )

    response = _engine(FakeWhisper()).transcribe(_request(word_timestamps=True))
    encoded = encode_event(response)
    payload = json.loads(encoded)
    assert payload["type"] == "transcription"
    assert payload["response"]["segments"][0]["tokens"] == [50365, 12]
    assert decode_event(encoded) == response


def test_decode_command_rejects_unknown_task() -> None:
    line = (
        b'{"type":"transcription","request":{"request_id":"a","model":"w",'
        b'"task":"speak","audio_base64":"AAAA"}}\n'
    )
    with pytest.raises(ValueError, match="unsupported transcription task"):
        decode_command(line)


def test_engine_maps_openai_options_to_whisper() -> None:
    model = FakeWhisper()
    engine = _engine(model)

    response = engine.transcribe(
        _request(language="French", prompt="Paris", word_timestamps=True)
    )
    engine.transcribe(_request(temperature=0.3))

    first, second = model.calls
    assert first["task"] == "transcribe"
    assert first["language"] == "fr"
    assert first["initial_prompt"] == "Paris"
    assert first["temperature"] == FALLBACK_TEMPERATURES
    assert first["word_timestamps"] is True
    assert first["samples"] == 32_000
    assert second["temperature"] == pytest.approx(0.3)
    assert second["language"] is None
    assert response.language == "french"
    assert response.duration == 2.0
    assert response.text == " Bonjour."
    assert response.segments[0].no_speech_prob == pytest.approx(0.02)
    assert [word.word for word in response.words] == [" Bonjour."]


def test_engine_omits_words_unless_requested() -> None:
    response = _engine(FakeWhisper()).transcribe(_request(task="translate"))
    assert response.task == "translate"
    assert response.words == ()


def test_engine_rejects_bad_input_before_loading_the_model() -> None:
    loads: list[str] = []
    engine = WhisperTranscriptionEngine(
        "whisper",
        loader=lambda ref: loads.append(ref) or FakeWhisper(),
        audio_reader=lambda _data: np.zeros(10, dtype=np.float32),
        languages=LANGUAGES,
    )

    with pytest.raises(ValueError, match="base64"):
        engine.transcribe(_request(audio_base64="not base64!"))
    with pytest.raises(ValueError, match="unsupported language"):
        engine.transcribe(_request(language="klingon"))
    assert loads == []


def test_service_returns_typed_errors_and_builds_engine_once() -> None:
    built: list[str] = []

    class FailingEngine:
        def __init__(self, model_ref: str) -> None:
            built.append(model_ref)

        def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
            if request.request_id == "bad-input":
                raise ValueError("unable to decode audio: junk")
            raise RuntimeError("metal exploded")

    service = TranscriptionService("whisper", engine_factory=FailingEngine)

    wrong = service.handle(_request(model="other"))
    bad = service.handle(_request(request_id="bad-input"))
    crash = service.handle(_request(request_id="crash"))

    assert isinstance(wrong, WorkerCommandError)
    assert wrong.code == "INVALID_REQUEST"
    assert "not served" in wrong.message
    assert isinstance(bad, WorkerCommandError)
    assert (bad.code, bad.request_id) == ("INVALID_REQUEST", "bad-input")
    assert isinstance(crash, WorkerCommandError)
    assert crash.code == "WORKER_ERROR"
    assert "metal exploded" in crash.message
    assert built == ["whisper"]


def test_service_without_configured_model_rejects_requests() -> None:
    event = TranscriptionService(None).handle(_request())
    assert isinstance(event, WorkerCommandError)
    assert "transcription='<none>'" in event.message


def test_read_audio_resamples_wav_to_16k_mono() -> None:
    samples = read_audio(_wav_bytes(sample_rate=8_000, channels=2, frames=8_000))

    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert abs(samples.size - 16_000) <= 32


def test_read_audio_rejects_non_audio_bytes() -> None:
    with pytest.raises(ValueError, match="unable to decode audio"):
        read_audio(b"definitely not audio")


def test_read_audio_names_ffmpeg_when_a_container_needs_it(monkeypatch) -> None:
    from mlx_worker import transcription as transcription_module

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(transcription_module, "HOMEBREW_BIN_DIRS", ())
    monkeypatch.setattr(transcription_module, "ensure_ffmpeg_on_path", lambda: False)
    ogg = b"OggS" + b"\x00" * 60

    with pytest.raises(ValueError, match="requires ffmpeg"):
        read_audio(ogg)


def test_ensure_ffmpeg_on_path_adds_homebrew_directory(monkeypatch, tmp_path) -> None:
    empty = tmp_path / "empty"
    brew = tmp_path / "brew"
    empty.mkdir()
    brew.mkdir()
    ffmpeg = brew / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\n")
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "ffmpeg" else "/x"
    )

    assert ensure_ffmpeg_on_path((str(empty), str(brew))) is True
    assert os.environ["PATH"] == f"/usr/bin:/bin:{brew}"
    assert ensure_ffmpeg_on_path((str(empty),)) is False


def test_config_reads_transcription_model(monkeypatch) -> None:
    from mlx_worker.config import load_config

    monkeypatch.setenv("MLX_RUNTIME_TRANSCRIPTION_MODEL", "  ")
    assert load_config().transcription_model is None
    monkeypatch.setenv("MLX_RUNTIME_TRANSCRIPTION_MODEL", "whisper-model")
    assert load_config().transcription_model == "whisper-model"


class _FakeRuntime:
    last_warmup_latency_ms = 1

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.closed = False

    def warmup(self) -> None:
        return None

    def idle(self) -> bool:
        return True

    def submit(self, command: object) -> None:
        self.submitted.append(command)

    def cancel(self, _request_id: str) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


def test_native_worker_serves_multi_megabyte_transcription_line(monkeypatch) -> None:
    from mlx_worker import transcription as transcription_module
    from mlx_worker.config import WorkerConfig
    from mlx_worker.native_mlx.worker import run_native_worker

    seen: list[TranscriptionRequest] = []

    class RecordingEngine:
        def __init__(self, model_ref: str) -> None:
            self.model_ref = model_ref

        def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
            seen.append(request)
            return TranscriptionResponse(
                request_id=request.request_id,
                model=request.model,
                task=request.task,
                text="ok",
                language="english",
                duration=1.0,
            )

    monkeypatch.setattr(
        transcription_module, "WhisperTranscriptionEngine", RecordingEngine
    )
    worker_end, gateway_end = socket.socketpair()
    runtime = _FakeRuntime()
    audio = base64.b64encode(b"\x01" * (3 * 1024 * 1024)).decode()
    line = _command_line(_request(audio_base64=audio))
    received = bytearray()

    def gateway() -> None:
        gateway_end.sendall(line)
        gateway_end.shutdown(socket.SHUT_WR)
        while chunk := gateway_end.recv(65536):
            received.extend(chunk)

    thread = threading.Thread(target=gateway)
    thread.start()
    status = run_native_worker(
        worker_end,
        WorkerConfig(
            socket_path="unused",
            model="text-model",
            transcription_model="whisper",
        ),
        native_worker_factory=lambda _config: runtime,
    )
    worker_end.close()
    thread.join(timeout=10)
    gateway_end.close()

    assert status == 0
    assert runtime.closed
    assert runtime.submitted == []
    assert len(seen) == 1
    assert seen[0].audio_base64 == audio
    events = [
        decode_event(raw)
        for raw in bytes(received).splitlines()
        if raw.startswith(b"{")
    ]
    responses = [event for event in events if isinstance(event, TranscriptionResponse)]
    assert [response.text for response in responses] == ["ok"]


def test_v1_worker_routes_transcription_without_touching_text_engine(
    monkeypatch,
) -> None:
    from mlx_worker import main as worker_main
    from mlx_worker import transcription as transcription_module

    class FakeSocket:
        def __init__(self, data: bytes) -> None:
            self.reader = io.BytesIO(data)
            self.sent: list[bytes] = []

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def connect(self, _path: str) -> None:
            return None

        def sendall(self, data: bytes) -> None:
            self.sent.append(data)

        def recv(self, size: int) -> bytes:
            return self.reader.read(size)

        def shutdown(self, _how: int) -> None:
            return None

    class TextEngine:
        def __init__(self, _model: str) -> None:
            return None

        def warmup(self) -> None:
            return None

        def complete_chat(self, _request: object) -> object:
            raise AssertionError("text engine must not serve audio")

    class RecordingEngine:
        def __init__(self, _model_ref: str) -> None:
            return None

        def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
            return TranscriptionResponse(
                request_id=request.request_id,
                model=request.model,
                task=request.task,
                text="v1 ok",
                language="english",
                duration=0.5,
            )

    monkeypatch.setattr(
        transcription_module, "WhisperTranscriptionEngine", RecordingEngine
    )
    fake_socket = FakeSocket(
        _command_line(_request()) + _command_line(_request(model="other"))
    )
    monkeypatch.setattr(worker_main.socket, "socket", lambda *_a, **_k: fake_socket)
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            backend="v1",
            model="text-model",
            transcription_model="whisper",
        ),
    )

    assert worker_main.main(engine_factory=TextEngine) == 0

    events = [decode_event(raw) for raw in fake_socket.sent if raw.startswith(b"{")]
    assert isinstance(events[0], TranscriptionResponse)
    assert events[0].text == "v1 ok"
    assert isinstance(events[1], WorkerCommandError)
    assert events[1].code == "INVALID_REQUEST"
