"""Line-based IPC helpers for bootstrap and Phase 1 inference frames,
including Phase 8 VLM image content support."""

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
    error: ModelError | None = None


@dataclass(frozen=True)
class ModelError:
    """Model lifecycle error information."""

    code: str
    message: str
    at: int
    backend: str | None = None
    stage: str | None = None
    category: str | None = None
    detail: str | None = None


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
class ImageContent:
    """An image URL or local path for VLM requests."""

    url: str
    detail: str = "auto"


@dataclass(frozen=True)
class TextContent:
    """A text segment within a multi-part message."""

    text: str


ContentPart = ImageContent | TextContent


@dataclass(frozen=True)
class ChatMessage:
    """One chat message from the OpenAI-style request.

    For text-only requests, *content* is a plain string (backward compatible).
    For VLM requests, *content* is a list of ``ContentPart`` items (text or image).
    """

    role: Literal["system", "user", "assistant"]
    content: str | tuple[ContentPart, ...]


@dataclass(frozen=True)
class ChatCompletionRequest:
    """A non-streaming worker request."""

    request_id: str
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float
    top_p: float
    max_prompt_tokens: int
    max_completion_tokens: int
    max_total_tokens_per_request: int
    stop: tuple[str, ...] = ()
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
    image_count: int | None = None
    image_preprocess_latency_ms: int | None = None
    prompt_template_latency_ms: int | None = None
    prompt_cache_hit: bool | None = None
    cached_tokens: int | None = None
    prompt_cache_bytes: int | None = None
    active_batch_cache_bytes: int | None = None
    prompt_batch_size: int | None = None
    decode_batch_size: int | None = None
    configured_prompt_batch_size: int | None = None
    configured_decode_batch_size: int | None = None
    backend: str | None = None
    modality: str | None = None
    apc_mode: str | None = None
    scheduler_stage: str | None = None
    cancellation_stage: str | None = None
    queue_time_ms: int | None = None
    prefill_time_ms: int | None = None
    ttft_ms: int | None = None
    decode_time_ms: int | None = None
    completion_time_ms: int | None = None
    scheduler_tick_latency_ms: int | None = None
    arbitration_delay_ms: int | None = None
    worker_cancellation_count: int | None = None
    worker_error_count: int | None = None
    vision_feature_cache_hit: bool | None = None
    vision_feature_cache_bytes: int | None = None
    vision_feature_cache_entries: int | None = None
    vision_feature_cache_evictions: int | None = None
    vision_encoder_latency_ms: int | None = None
    embedding_latency_ms: int | None = None
    prompt_cache_entries: int | None = None
    prompt_cache_evictions: int | None = None
    peak_memory_bytes: int | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True)
class ChatCompletionDelta:
    """A streamed completion delta."""

    request_id: str
    delta: str


@dataclass(frozen=True)
class CancelRequest:
    """A cancellation request for an in-flight completion."""

    request_id: str


@dataclass(frozen=True)
class WorkerCommandError:
    """A worker-side request failure."""

    code: str
    request_id: str
    message: str


@dataclass(frozen=True)
class SchedulerMetricsEvent:
    """Per-step native scheduler metrics event."""

    backend: str
    modality: str
    phase: Literal["prefill", "decode"]
    scheduled_tokens: int
    batch_size: int
    waiting_requests: int
    running_requests: int
    scheduler_tick_latency_ms: int
    forward_mode: str | None = None
    physical_batch_size: int | None = None
    model_forward_count: int | None = None
    cache_backend: str | None = None
    attention_backend: str | None = None
    attention_mode: str | None = None
    attention_time_ms: int | None = None
    executor_prepare_ms: int | None = None
    executor_reserve_ms: int | None = None
    executor_forward_ms: int | None = None
    executor_sample_ms: int | None = None
    executor_eval_ms: int | None = None
    executor_commit_ms: int | None = None
    total_pages: int | None = None
    used_pages: int | None = None
    free_pages: int | None = None
    pinned_pages: int | None = None
    internal_fragmentation_tokens: int | None = None
    active_kv_bytes: int | None = None
    allocation_failures: int | None = None
    page_size: int | None = None
    prefix_strategy: str | None = None
    prefix_queries: int | None = None
    prefix_hits: int | None = None
    prefix_misses: int | None = None
    prefix_reused_tokens: int | None = None
    prefix_reused_pages: int | None = None
    prefix_entries: int | None = None
    prefix_bytes: int | None = None
    prefix_pinned_pages: int | None = None
    prefix_collisions_rejected: int | None = None
    prefix_evictions: int | None = None


def encode_bootstrap_message(
    message: WorkerReady | WorkerError | ModelStatus,
) -> bytes:
    """Encode a worker bootstrap message."""

    if isinstance(message, WorkerReady):
        return b"READY\n"
    if isinstance(message, WorkerError):
        if message.error is None:
            sanitized = message.message.replace("\n", " ")
            return f"ERROR\t{sanitized}\n".encode("utf-8")
        payload = {
            "message": message.message.replace("\n", " "),
            "error": asdict(message.error),
        }
        return f"ERROR\t{json.dumps(payload)}\n".encode("utf-8")

    return f"STATUS\t{json.dumps(asdict(message))}\n".encode("utf-8")


def _model_error_from_dict(data: dict[str, Any] | None) -> ModelError | None:
    if data is None:
        return None
    return ModelError(
        code=data["code"],
        message=data["message"],
        at=data["at"],
        backend=data.get("backend"),
        stage=data.get("stage"),
        category=data.get("category"),
        detail=data.get("detail"),
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
        payload = line.split("\t", 1)[1]
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return WorkerError(message=payload)
        if not isinstance(decoded, dict):
            return WorkerError(message=payload)
        return WorkerError(
            message=str(decoded.get("message", payload)),
            error=_model_error_from_dict(decoded.get("error")),
        )

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
                {
                    "role": message.role,
                    "content": _encode_content(message.content),
                }
                for message in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_prompt_tokens": request.max_prompt_tokens,
            "max_completion_tokens": request.max_completion_tokens,
            "max_total_tokens_per_request": request.max_total_tokens_per_request,
            "stop": list(request.stop),
            "stream": request.stream,
        },
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def _decode_content(
    raw: str | list[dict[str, Any]],
) -> str | tuple[ContentPart, ...]:
    """Decode message content that may be plain text or multi-part VLM content."""
    if isinstance(raw, str):
        return raw
    parts: list[ContentPart] = []
    for item in raw:
        item_type = item.get("type")
        if item_type == "text":
            parts.append(TextContent(text=item["text"]))
        elif item_type == "image_url":
            url_data = item["image_url"]
            parts.append(
                ImageContent(
                    url=url_data["url"],
                    detail=url_data.get("detail", "auto"),
                )
            )
        else:
            parts.append(TextContent(text=str(item)))
    return tuple(parts)


def _encode_content(
    content: str | tuple[ContentPart, ...],
) -> str | list[dict[str, Any]]:
    """Encode message content for JSON serialization."""
    if isinstance(content, str):
        return content
    serialized: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextContent):
            serialized.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            serialized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.url, "detail": part.detail},
                }
            )
    return serialized


def decode_command(raw_line: bytes) -> ChatCompletionRequest | CancelRequest | None:
    """Decode a Phase 1 gateway command."""

    payload = json.loads(raw_line.decode("utf-8"))
    if payload.get("type") != "chat_completion":
        if payload.get("type") == "cancel_request":
            return CancelRequest(request_id=payload["request_id"])
        return None
    request = payload["request"]
    return ChatCompletionRequest(
        request_id=request["request_id"],
        model=request["model"],
        messages=[
            ChatMessage(
                role=message["role"],
                content=_decode_content(message["content"]),
            )
            for message in request["messages"]
        ],
        max_tokens=request["max_tokens"],
        temperature=request["temperature"],
        top_p=request["top_p"],
        max_prompt_tokens=request["max_prompt_tokens"],
        max_completion_tokens=request["max_completion_tokens"],
        max_total_tokens_per_request=request["max_total_tokens_per_request"],
        stop=tuple(str(item) for item in request.get("stop", [])),
        stream=request.get("stream", False),
    )


def encode_event(
    event: (
        ChatCompletionResponse
        | ChatCompletionDelta
        | WorkerCommandError
        | SchedulerMetricsEvent
    ),
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
                "prompt_cache_hit": event.prompt_cache_hit,
                "cached_tokens": event.cached_tokens,
                "prompt_cache_bytes": event.prompt_cache_bytes,
                "active_batch_cache_bytes": event.active_batch_cache_bytes,
                "prompt_batch_size": event.prompt_batch_size,
                "decode_batch_size": event.decode_batch_size,
                "backend": event.backend,
                "scheduler_stage": event.scheduler_stage,
                "cancellation_stage": event.cancellation_stage,
                "queue_time_ms": event.queue_time_ms,
                "prefill_time_ms": event.prefill_time_ms,
                "ttft_ms": event.ttft_ms,
                "decode_time_ms": event.decode_time_ms,
                "completion_time_ms": event.completion_time_ms,
                "vision_feature_cache_hit": event.vision_feature_cache_hit,
                "vision_feature_cache_bytes": event.vision_feature_cache_bytes,
                "vision_encoder_latency_ms": event.vision_encoder_latency_ms,
                "embedding_latency_ms": event.embedding_latency_ms,
            },
            "image_count": event.image_count,
            "image_preprocess_latency_ms": event.image_preprocess_latency_ms,
            "prompt_template_latency_ms": event.prompt_template_latency_ms,
            "prompt_cache_hit": event.prompt_cache_hit,
            "cached_tokens": event.cached_tokens,
            "prompt_cache_bytes": event.prompt_cache_bytes,
            "active_batch_cache_bytes": event.active_batch_cache_bytes,
            "prompt_batch_size": event.prompt_batch_size,
            "decode_batch_size": event.decode_batch_size,
            "configured_prompt_batch_size": event.configured_prompt_batch_size,
            "configured_decode_batch_size": event.configured_decode_batch_size,
            "backend": event.backend,
            "modality": event.modality,
            "apc_mode": event.apc_mode,
            "scheduler_stage": event.scheduler_stage,
            "cancellation_stage": event.cancellation_stage,
            "queue_time_ms": event.queue_time_ms,
            "prefill_time_ms": event.prefill_time_ms,
            "ttft_ms": event.ttft_ms,
            "decode_time_ms": event.decode_time_ms,
            "completion_time_ms": event.completion_time_ms,
            "scheduler_tick_latency_ms": event.scheduler_tick_latency_ms,
            "arbitration_delay_ms": event.arbitration_delay_ms,
            "worker_cancellation_count": event.worker_cancellation_count,
            "worker_error_count": event.worker_error_count,
            "vision_feature_cache_hit": event.vision_feature_cache_hit,
            "vision_feature_cache_bytes": event.vision_feature_cache_bytes,
            "vision_feature_cache_entries": event.vision_feature_cache_entries,
            "vision_feature_cache_evictions": event.vision_feature_cache_evictions,
            "vision_encoder_latency_ms": event.vision_encoder_latency_ms,
            "embedding_latency_ms": event.embedding_latency_ms,
            "prompt_cache_entries": event.prompt_cache_entries,
            "prompt_cache_evictions": event.prompt_cache_evictions,
            "peak_memory_bytes": event.peak_memory_bytes,
            "image_width": event.image_width,
            "image_height": event.image_height,
        }
    elif isinstance(event, ChatCompletionDelta):
        payload = {
            "type": "chat_completion_delta",
            "delta": {
                "request_id": event.request_id,
                "delta": event.delta,
            },
        }
    elif isinstance(event, WorkerCommandError):
        payload = {
            "type": "error",
            "code": event.code,
            "request_id": event.request_id,
            "message": event.message.replace("\n", " "),
        }
    else:
        payload = {
            "type": "scheduler_metrics",
            "metrics": asdict(event),
        }

    return (json.dumps(payload) + "\n").encode("utf-8")


def encode_cancel_request(request_id: str) -> bytes:
    """Encode a cancellation command."""

    payload = {"type": "cancel_request", "request_id": request_id}
    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_event(
    raw_line: bytes,
) -> (
    ChatCompletionResponse
    | ChatCompletionDelta
    | WorkerCommandError
    | SchedulerMetricsEvent
    | None
):
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
            prompt_cache_hit=response.get("prompt_cache_hit"),
            cached_tokens=response.get("cached_tokens"),
            prompt_cache_bytes=response.get("prompt_cache_bytes"),
            active_batch_cache_bytes=response.get("active_batch_cache_bytes"),
            prompt_batch_size=response.get("prompt_batch_size"),
            decode_batch_size=response.get("decode_batch_size"),
            backend=response.get("backend"),
            scheduler_stage=response.get("scheduler_stage"),
            cancellation_stage=response.get("cancellation_stage"),
            queue_time_ms=response.get("queue_time_ms"),
            prefill_time_ms=response.get("prefill_time_ms"),
            ttft_ms=response.get("ttft_ms"),
            decode_time_ms=response.get("decode_time_ms"),
            completion_time_ms=response.get("completion_time_ms"),
            vision_feature_cache_hit=response.get("vision_feature_cache_hit"),
            vision_feature_cache_bytes=response.get("vision_feature_cache_bytes"),
            vision_encoder_latency_ms=response.get("vision_encoder_latency_ms"),
            embedding_latency_ms=response.get("embedding_latency_ms"),
            image_count=payload.get("image_count"),
            image_preprocess_latency_ms=payload.get("image_preprocess_latency_ms"),
            prompt_template_latency_ms=payload.get("prompt_template_latency_ms"),
        )
    if payload.get("type") == "chat_completion_delta":
        delta = payload["delta"]
        return ChatCompletionDelta(
            request_id=delta["request_id"],
            delta=delta["delta"],
        )
    if payload.get("type") == "error":
        return WorkerCommandError(
            code=payload["code"],
            request_id=payload["request_id"],
            message=payload["message"],
        )
    if payload.get("type") == "scheduler_metrics":
        metrics = payload["metrics"]
        return SchedulerMetricsEvent(
            backend=metrics["backend"],
            modality=metrics["modality"],
            phase=metrics["phase"],
            scheduled_tokens=metrics["scheduled_tokens"],
            batch_size=metrics["batch_size"],
            waiting_requests=metrics["waiting_requests"],
            running_requests=metrics["running_requests"],
            scheduler_tick_latency_ms=metrics["scheduler_tick_latency_ms"],
            forward_mode=metrics.get("forward_mode"),
            physical_batch_size=metrics.get("physical_batch_size"),
            model_forward_count=metrics.get("model_forward_count"),
            cache_backend=metrics.get("cache_backend"),
            attention_backend=metrics.get("attention_backend"),
            attention_mode=metrics.get("attention_mode"),
            attention_time_ms=metrics.get("attention_time_ms"),
            total_pages=metrics.get("total_pages"),
            used_pages=metrics.get("used_pages"),
            free_pages=metrics.get("free_pages"),
            pinned_pages=metrics.get("pinned_pages"),
            internal_fragmentation_tokens=metrics.get("internal_fragmentation_tokens"),
            active_kv_bytes=metrics.get("active_kv_bytes"),
            allocation_failures=metrics.get("allocation_failures"),
            page_size=metrics.get("page_size"),
            prefix_strategy=metrics.get("prefix_strategy"),
            prefix_queries=metrics.get("prefix_queries"),
            prefix_hits=metrics.get("prefix_hits"),
            prefix_misses=metrics.get("prefix_misses"),
            prefix_reused_tokens=metrics.get("prefix_reused_tokens"),
            prefix_reused_pages=metrics.get("prefix_reused_pages"),
            prefix_entries=metrics.get("prefix_entries"),
            prefix_bytes=metrics.get("prefix_bytes"),
            prefix_pinned_pages=metrics.get("prefix_pinned_pages"),
            prefix_collisions_rejected=metrics.get("prefix_collisions_rejected"),
            prefix_evictions=metrics.get("prefix_evictions"),
        )
    return None


def has_image_content(content: str | tuple[ContentPart, ...]) -> bool:
    """Return True when the message content includes at least one image."""
    if isinstance(content, str):
        return False
    return any(isinstance(part, ImageContent) for part in content)


def request_has_images(request: ChatCompletionRequest) -> bool:
    """Return True when any message in the request contains image content."""
    return any(has_image_content(msg.content) for msg in request.messages)


encode_message = encode_bootstrap_message
decode_message = decode_bootstrap_message
