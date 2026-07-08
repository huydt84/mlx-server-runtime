"""Native request lifecycle, tokenization, text, and terminal semantics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..ipc import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    SchedulerMetricsEvent,
    WorkerCommandError,
)
from .interfaces import (
    RuntimeEvent,
    SamplingParams,
    SchedulableRequest,
    SchedulerEvent,
)
from .scheduler import NativeContinuousScheduler


@dataclass
class _StopTextAssembler:
    stop_sequences: tuple[str, ...]
    pending: str = ""
    emitted: str = ""

    def push(self, text: str) -> tuple[str, bool]:
        self.pending += text
        matches = [
            index
            for stop in self.stop_sequences
            if (index := self.pending.find(stop)) >= 0
        ]
        if matches:
            index = min(matches)
            chunk = self.pending[:index]
            self.emitted += chunk
            self.pending = ""
            return chunk, True
        hold = max((len(stop) - 1 for stop in self.stop_sequences), default=0)
        if len(self.pending) <= hold:
            return "", False
        chunk = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[-hold:] if hold else ""
        self.emitted += chunk
        return chunk, False

    def finish(self) -> str:
        chunk = self.pending
        self.emitted += chunk
        self.pending = ""
        return chunk

    def text(self) -> str:
        return self.emitted + self.pending


@dataclass
class _RuntimeState:
    request: ChatCompletionRequest
    prompt_token_ids: tuple[int, ...]
    stop: _StopTextAssembler
    enqueued_at: float
    completion_token_ids: list[int]
    decoded_prefix: str = ""
    prefill_time_ms: int = 0
    decode_time_ms: int = 0
    prompt_batch_size: int | None = None
    decode_batch_size: int | None = None
    ttft_ms: int | None = None
    cancellation_stage: str | None = None
    queue_time_ms: int = 0


class NativeRuntime:
    """Own public request normalization and terminal request lifecycle."""

    def __init__(
        self,
        scheduler: NativeContinuousScheduler,
        *,
        model_ref: str,
        prompt_tokenizer: Any,
        decode_target: Any,
        eos_token_ids: tuple[int, ...],
    ) -> None:
        self._scheduler = scheduler
        self._model_ref = model_ref
        self._prompt_tokenizer = prompt_tokenizer
        self._decode_target = decode_target
        self._eos_token_ids = eos_token_ids
        self._requests: dict[str, _RuntimeState] = {}
        self.last_warmup_latency_ms = 0
        self._cancellation_count = 0
        self._error_count = 0

    def warmup(self) -> None:
        started = time.perf_counter()
        from ..ipc import ChatMessage

        request = ChatCompletionRequest(
            request_id="warmup",
            model=self._model_ref,
            messages=[ChatMessage(role="user", content="ping")],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=64,
            max_completion_tokens=64,
            max_total_tokens_per_request=128,
        )
        self.submit(request)
        while not self.idle():
            self.tick()
        self.last_warmup_latency_ms = max(
            1, int((time.perf_counter() - started) * 1000)
        )

    def submit(self, request: ChatCompletionRequest) -> None:
        prompt = self._build_prompt(request)
        self._validate(request, prompt)
        state = _RuntimeState(
            request=request,
            prompt_token_ids=prompt,
            stop=_StopTextAssembler(tuple(stop for stop in request.stop if stop)),
            enqueued_at=time.perf_counter(),
            completion_token_ids=[],
        )
        self._requests[request.request_id] = state
        self._scheduler.submit(
            SchedulableRequest(
                request_id=request.request_id,
                prompt_token_ids=prompt,
                sampling=SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
                enqueued_at=state.enqueued_at,
            )
        )

    def cancel(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        if state is not None:
            state.cancellation_stage = (
                "decode" if state.completion_token_ids else "prefill"
            )
        return self._scheduler.cancel(request_id)

    def tick(self) -> tuple[RuntimeEvent, ...]:
        output: list[RuntimeEvent] = []
        for event in self._scheduler.tick():
            self._apply_scheduler_event(event, output)
        return tuple(output)

    def idle(self) -> bool:
        return self._scheduler.idle()

    def close(self) -> None:
        self._scheduler.close()
        self._requests.clear()

    def _apply_scheduler_event(
        self,
        event: SchedulerEvent,
        output: list[RuntimeEvent],
    ) -> None:
        if event.kind == "metrics":
            metrics = event.metrics or {}
            output.append(
                RuntimeEvent(
                    kind="metrics",
                    payload=SchedulerMetricsEvent(
                        backend="native-mlx",
                        modality="text",
                        phase=str(metrics.get("phase", "decode")),
                        scheduled_tokens=int(metrics.get("scheduled_tokens", 0)),
                        batch_size=int(metrics.get("batch_size", 0)),
                        waiting_requests=int(metrics.get("waiting_requests", 0)),
                        running_requests=int(metrics.get("running_requests", 0)),
                        scheduler_tick_latency_ms=int(
                            metrics.get("scheduler_tick_latency_ms", 0)
                        ),
                        forward_mode=_optional_str(metrics, "forward_mode"),
                        physical_batch_size=_optional_int(
                            metrics, "physical_batch_size"
                        ),
                        model_forward_count=_optional_int(
                            metrics, "model_forward_count"
                        ),
                        cache_backend=_optional_str(metrics, "cache_backend"),
                        attention_backend=_optional_str(metrics, "attention_backend"),
                        attention_mode=_optional_str(metrics, "attention_mode"),
                        attention_time_ms=_optional_int(metrics, "attention_time_ms"),
                        total_pages=_optional_int(metrics, "total_pages"),
                        used_pages=_optional_int(metrics, "used_pages"),
                        free_pages=_optional_int(metrics, "free_pages"),
                        pinned_pages=_optional_int(metrics, "pinned_pages"),
                        internal_fragmentation_tokens=_optional_int(
                            metrics, "internal_fragmentation_tokens"
                        ),
                        active_kv_bytes=_optional_int(metrics, "active_kv_bytes"),
                        allocation_failures=_optional_int(
                            metrics, "allocation_failures"
                        ),
                        page_size=_optional_int(metrics, "page_size"),
                        prefix_strategy=_optional_str(metrics, "prefix_strategy"),
                        prefix_queries=_optional_int(metrics, "prefix_queries"),
                        prefix_hits=_optional_int(metrics, "prefix_hits"),
                        prefix_misses=_optional_int(metrics, "prefix_misses"),
                        prefix_reused_tokens=_optional_int(
                            metrics, "prefix_reused_tokens"
                        ),
                        prefix_reused_pages=_optional_int(
                            metrics, "prefix_reused_pages"
                        ),
                        prefix_entries=_optional_int(metrics, "prefix_entries"),
                        prefix_bytes=_optional_int(metrics, "prefix_bytes"),
                        prefix_pinned_pages=_optional_int(
                            metrics, "prefix_pinned_pages"
                        ),
                        prefix_collisions_rejected=_optional_int(
                            metrics, "prefix_collisions_rejected"
                        ),
                        prefix_evictions=_optional_int(metrics, "prefix_evictions"),
                    ),
                )
            )
            return
        if event.request_id is None:
            return
        state = self._requests.get(event.request_id)
        if state is None:
            return
        if event.kind == "prefill_progress":
            metrics = event.metrics or {}
            state.prefill_time_ms += int(metrics.get("step_time_ms", 0))
            state.prompt_batch_size = int(metrics.get("batch_size", 0))
            state.queue_time_ms = int(metrics.get("queue_time_ms", 0))
            return
        if event.kind == "token":
            if event.metrics:
                state.decode_time_ms += int(event.metrics.get("step_time_ms", 0))
                state.decode_batch_size = int(event.metrics.get("batch_size", 0))
            if event.token_id is None:
                self._fail(
                    state, "WORKER_ERROR", "model step returned no token", output
                )
            else:
                self._accept_token(state, event.token_id, output)
            return
        if event.kind == "cancelled":
            self._cancellation_count += 1
            self._finish(state, "cancelled", output)
            return
        if event.kind == "execution_error":
            self._fail(
                state,
                event.error_code or "WORKER_ERROR",
                event.message or "native execution failed",
                output,
            )

    def _accept_token(
        self,
        state: _RuntimeState,
        token_id: int,
        output: list[RuntimeEvent],
    ) -> None:
        state.completion_token_ids.append(token_id)
        if token_id in self._eos_token_ids:
            self._finish(state, "stop", output)
            return
        text = str(
            self._decode_target.decode(
                state.completion_token_ids,
                skip_special_tokens=False,
            )
        )
        delta = (
            text[len(state.decoded_prefix) :]
            if text.startswith(state.decoded_prefix)
            else text
        )
        state.decoded_prefix = text
        emitted, stop_hit = state.stop.push(delta)
        if emitted and state.request.stream:
            output.append(
                RuntimeEvent(
                    kind="delta",
                    payload=ChatCompletionDelta(
                        request_id=state.request.request_id,
                        delta=emitted,
                    ),
                )
            )
        if emitted and state.ttft_ms is None:
            state.ttft_ms = max(
                1, int((time.perf_counter() - state.enqueued_at) * 1000)
            )
        if stop_hit:
            self._finish(state, "stop", output)
        elif len(state.completion_token_ids) >= state.request.max_tokens:
            self._finish(state, "length", output)

    def _finish(
        self,
        state: _RuntimeState,
        reason: str,
        output: list[RuntimeEvent],
    ) -> None:
        trailing = "" if reason == "cancelled" else state.stop.finish()
        if trailing and state.request.stream:
            output.append(
                RuntimeEvent(
                    kind="delta",
                    payload=ChatCompletionDelta(
                        request_id=state.request.request_id,
                        delta=trailing,
                    ),
                )
            )
        request_id = state.request.request_id
        self._scheduler.finish(request_id)
        self._requests.pop(request_id, None)
        output.append(
            RuntimeEvent(
                kind="response",
                payload=ChatCompletionResponse(
                    request_id=request_id,
                    model=self._model_ref,
                    text=state.stop.text(),
                    finish_reason=reason,
                    prompt_tokens=len(state.prompt_token_ids),
                    completion_tokens=len(state.completion_token_ids),
                    prompt_batch_size=state.prompt_batch_size,
                    decode_batch_size=state.decode_batch_size,
                    backend="native-mlx",
                    modality="text",
                    scheduler_stage="cancelled"
                    if reason == "cancelled"
                    else "completed",
                    cancellation_stage=state.cancellation_stage,
                    queue_time_ms=state.queue_time_ms,
                    prefill_time_ms=state.prefill_time_ms,
                    ttft_ms=state.ttft_ms,
                    decode_time_ms=state.decode_time_ms,
                    completion_time_ms=max(
                        1, int((time.perf_counter() - state.enqueued_at) * 1000)
                    ),
                    worker_cancellation_count=self._cancellation_count,
                    worker_error_count=self._error_count,
                ),
            )
        )

    def _fail(
        self,
        state: _RuntimeState,
        code: str,
        message: str,
        output: list[RuntimeEvent],
    ) -> None:
        self._error_count += 1
        request_id = state.request.request_id
        self._scheduler.finish(request_id)
        self._requests.pop(request_id, None)
        output.append(
            RuntimeEvent(
                kind="error",
                payload=WorkerCommandError(
                    code=code,
                    request_id=request_id,
                    message=message,
                ),
            )
        )

    def _build_prompt(self, request: ChatCompletionRequest) -> tuple[int, ...]:
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if not isinstance(message.content, str):
                raise ValueError(
                    "native-mlx text backend does not support image content"
                )
            messages.append({"role": message.role, "content": message.content})
        tokens = self._prompt_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return tuple(int(token) for token in tokens)

    def _validate(
        self,
        request: ChatCompletionRequest,
        prompt: tuple[int, ...],
    ) -> None:
        if request.model != self._model_ref:
            raise ValueError("requested model does not match loaded model")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise ValueError("native-mlx currently supports greedy decoding only")
        if any(not value for value in request.stop):
            raise ValueError("stop sequences must not be empty")
        if len(prompt) > request.max_prompt_tokens:
            raise ValueError("prompt exceeds max_prompt_tokens")
        if request.max_tokens > request.max_completion_tokens:
            raise ValueError("completion exceeds max_completion_tokens")
        if len(prompt) + request.max_tokens > request.max_total_tokens_per_request:
            raise ValueError("request exceeds max_total_tokens_per_request")


def _optional_int(metrics: dict[str, object], name: str) -> int | None:
    value = metrics.get(name)
    return None if value is None else int(value)


def _optional_str(metrics: dict[str, object], name: str) -> str | None:
    value = metrics.get(name)
    return None if value is None else str(value)
