"""Line-based IPC helpers for bootstrap and Phase 1 inference frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class WorkerReady:
    """Worker ready signal."""


@dataclass(frozen=True)
class WorkerError:
    """Worker startup error."""

    message: str


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
class WorkerCommandError:
    """A worker-side request failure."""

    request_id: str
    message: str


def encode_bootstrap_message(message: WorkerReady | WorkerError) -> bytes:
    """Encode a worker bootstrap message."""

    if isinstance(message, WorkerReady):
        return b"READY\n"
    sanitized = message.message.replace("\n", " ")
    return f"ERROR\t{sanitized}\n".encode("utf-8")


def decode_bootstrap_message(raw_line: bytes) -> WorkerReady | WorkerError | None:
    """Decode a worker bootstrap message."""

    line = raw_line.decode("utf-8", errors="replace").strip()
    if line == "READY":
        return WorkerReady()

    if line.startswith("ERROR\t"):
        return WorkerError(message=line.split("\t", 1)[1])

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
    )


def encode_event(event: ChatCompletionResponse | WorkerCommandError) -> bytes:
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
    else:
        payload = {
            "type": "error",
            "request_id": event.request_id,
            "message": event.message.replace("\n", " "),
        }

    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_event(raw_line: bytes) -> ChatCompletionResponse | WorkerCommandError | None:
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
    if payload.get("type") == "error":
        return WorkerCommandError(
            request_id=payload["request_id"],
            message=payload["message"],
        )
    return None


encode_message = encode_bootstrap_message
decode_message = decode_bootstrap_message
