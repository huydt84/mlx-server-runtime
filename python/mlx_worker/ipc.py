"""Line-based IPC helpers for bootstrap and Phase 1 inference frames."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


ModelState = Literal[
    "not_loaded",
    "downloading",
    "verifying",
    "loading_weights",
    "initializing_runtime",
    "warming_up",
    "ready",
    "degraded",
    "failed",
    "unloading",
]


@dataclass(frozen=True)
class WorkerReady:
    """Worker ready signal."""


@dataclass(frozen=True)
class WorkerError:
    """Worker startup error."""

    message: str


@dataclass(frozen=True)
class ModelError:
    """Model lifecycle error information."""

    code: str
    message: str
    at: int


@dataclass(frozen=True)
class ModelLoadProgress:
    """Optional model loading progress details."""

    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    loaded_tensors: int | None = None
    total_tensors: int | None = None
    current_phase: str | None = None


@dataclass(frozen=True)
class ModelStatus:
    """Detailed model lifecycle status."""

    model: str
    revision: str | None
    state: ModelState
    ready: bool
    servable: bool
    progress: ModelLoadProgress | None
    device: str | None
    dtype: str | None
    loaded_at: int | None
    started_loading_at: int | None
    last_transition_at: int
    last_error: ModelError | None
    warmup_passed: bool
    last_warmup_at: int | None
    last_warmup_latency_ms: int | None


@dataclass(frozen=True)
class ChatMessage:
    """One chat message from the OpenAI-style request."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    """A non-streaming worker request."""

    request_id: str
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool = False


@dataclass(frozen=True)
class ChatCompletionResponse:
    """A non-streaming worker response."""

    request_id: str
    model: str
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ChatCompletionDelta:
    """A streamed completion delta."""

    request_id: str
    delta: str


@dataclass(frozen=True)
class WorkerCommandError:
    """A worker-side request failure."""

    request_id: str
    message: str


def encode_bootstrap_message(
    message: WorkerReady | WorkerError | ModelStatus,
) -> bytes:
    """Encode a worker bootstrap message."""

    if isinstance(message, WorkerReady):
        return b"READY\n"
    if isinstance(message, WorkerError):
        sanitized = message.message.replace("\n", " ")
        return f"ERROR\t{sanitized}\n".encode("utf-8")

    return f"STATUS\t{json.dumps(asdict(message))}\n".encode("utf-8")


def _model_error_from_dict(data: dict[str, Any] | None) -> ModelError | None:
    if data is None:
        return None
    return ModelError(
        code=data["code"],
        message=data["message"],
        at=data["at"],
    )


def _model_progress_from_dict(
    data: dict[str, Any] | None,
) -> ModelLoadProgress | None:
    if data is None:
        return None
    return ModelLoadProgress(
        downloaded_bytes=data.get("downloaded_bytes"),
        total_bytes=data.get("total_bytes"),
        loaded_tensors=data.get("loaded_tensors"),
        total_tensors=data.get("total_tensors"),
        current_phase=data.get("current_phase"),
    )


def decode_bootstrap_message(
    raw_line: bytes,
) -> WorkerReady | WorkerError | ModelStatus | None:
    """Decode a worker bootstrap message."""

    line = raw_line.decode("utf-8", errors="replace").strip()
    if line == "READY":
        return WorkerReady()

    if line.startswith("ERROR\t"):
        return WorkerError(message=line.split("\t", 1)[1])

    if line.startswith("STATUS\t"):
        payload = json.loads(line.split("\t", 1)[1])
        return ModelStatus(
            model=payload["model"],
            revision=payload.get("revision"),
            state=payload["state"],
            ready=payload["ready"],
            servable=payload["servable"],
            progress=_model_progress_from_dict(payload.get("progress")),
            device=payload.get("device"),
            dtype=payload.get("dtype"),
            loaded_at=payload.get("loaded_at"),
            started_loading_at=payload.get("started_loading_at"),
            last_transition_at=payload["last_transition_at"],
            last_error=_model_error_from_dict(payload.get("last_error")),
            warmup_passed=payload["warmup_passed"],
            last_warmup_at=payload.get("last_warmup_at"),
            last_warmup_latency_ms=payload.get("last_warmup_latency_ms"),
        )

    return None


def encode_command(request: ChatCompletionRequest) -> bytes:
    """Encode a Phase 1 gateway command."""

    payload = {
        "type": "chat_completion",
        "request": {
            "request_id": request.request_id,
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        },
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_command(raw_line: bytes) -> ChatCompletionRequest | None:
    """Decode a Phase 1 gateway command."""

    payload = json.loads(raw_line.decode("utf-8"))
    if payload.get("type") != "chat_completion":
        return None
    request = payload["request"]
    return ChatCompletionRequest(
        request_id=request["request_id"],
        model=request["model"],
        messages=[
            ChatMessage(role=message["role"], content=message["content"])
            for message in request["messages"]
        ],
        max_tokens=request["max_tokens"],
        temperature=request["temperature"],
        top_p=request["top_p"],
        stream=request.get("stream", False),
    )


def encode_event(
    event: ChatCompletionResponse | ChatCompletionDelta | WorkerCommandError,
) -> bytes:
    """Encode a Phase 1 worker event."""

    payload: dict[str, Any]
    if isinstance(event, ChatCompletionResponse):
        payload = {
            "type": "chat_completion",
            "response": {
                "request_id": event.request_id,
                "model": event.model,
                "text": event.text,
                "finish_reason": event.finish_reason,
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
            },
        }
    elif isinstance(event, ChatCompletionDelta):
        payload = {
            "type": "chat_completion_delta",
            "delta": {
                "request_id": event.request_id,
                "delta": event.delta,
            },
        }
    else:
        payload = {
            "type": "error",
            "request_id": event.request_id,
            "message": event.message.replace("\n", " "),
        }

    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_event(
    raw_line: bytes,
) -> ChatCompletionResponse | ChatCompletionDelta | WorkerCommandError | None:
    """Decode a Phase 1 worker event."""

    payload = json.loads(raw_line.decode("utf-8"))
    if payload.get("type") == "chat_completion":
        response = payload["response"]
        return ChatCompletionResponse(
            request_id=response["request_id"],
            model=response["model"],
            text=response["text"],
            finish_reason=response["finish_reason"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
        )
    if payload.get("type") == "chat_completion_delta":
        delta = payload["delta"]
        return ChatCompletionDelta(
            request_id=delta["request_id"],
            delta=delta["delta"],
        )
    if payload.get("type") == "error":
        return WorkerCommandError(
            request_id=payload["request_id"],
            message=payload["message"],
        )
    return None


encode_message = encode_bootstrap_message
decode_message = decode_bootstrap_message
