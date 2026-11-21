"""Native request lifecycle, tokenization, text, and terminal semantics."""

from __future__ import annotations

import math
import secrets
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

_WARMUP_DECODE_TOKENS = 4
_WARMUP_MIN_PROMPT_TOKENS = 48
_WARMUP_PAIR_LENGTH_GAP = 8
_WARMUP_PROBE_WORDS = 64
_WARMUP_WORD = "warm"
_WARMUP_SAMPLED_TEMPERATURE = 0.7
_WARMUP_SAMPLED_TOP_P = 0.9


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
    detokenizer: Any | None = None
    decoded_prefix: str = ""
    prefill_time_ms: int = 0
    decode_time_ms: int = 0
    prompt_batch_size: int | None = None
    decode_batch_size: int | None = None
    ttft_ms: int | None = None
    cancellation_stage: str | None = None
    scheduler_queue_wait_ms: int = 0
    cancellation_latency_ms: int | None = None
    cached_tokens: int = 0


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
        profiler: Any | None = None,
        max_total_tokens: int | None = None,
        warmup_prompt_lengths: tuple[int, ...] = (_WARMUP_MIN_PROMPT_TOKENS,),
    ) -> None:
        self._scheduler = scheduler
        self._max_total_tokens = max_total_tokens
        self._model_ref = model_ref
        self._prompt_tokenizer = prompt_tokenizer
        self._decode_target = decode_target
        self._eos_token_ids = eos_token_ids
        self._profiler = profiler
        self._warmup_prompt_lengths = warmup_prompt_lengths
        self._warming_up = False
        self._requests: dict[str, _RuntimeState] = {}
        self.last_warmup_latency_ms = 0
        self._cancellation_count = 0
        self._error_count = 0

    def warmup(self) -> None:
        """Compile the GPU kernels a first public request would otherwise compile.

        MLX compiles a Metal kernel variant the first time a step uses it, which
        added hundreds of milliseconds to the first requests after startup.
        Warmup therefore runs, to completion:

        - one greedy request per ``warmup_prompt_lengths`` entry, each decoding
          a few tokens, so decode reaches every key-count range the paged
          decode kernel specializes on. The prompts share a prefix, so the
          prefix cache spares later ones most of their prefill;
        - two requests at once with different prompt lengths, one sampled with
          ``top_p``, which compiles padded multi-row prefill, multi-row decode,
          and top-p sampling.

        Lengths that the KV pool cannot hold are skipped.
        """

        started = time.perf_counter()
        self._warming_up = True
        try:
            lengths = sorted(set(self._warmup_prompt_lengths)) or [
                _WARMUP_MIN_PROMPT_TOKENS
            ]
            for index, length in enumerate(lengths):
                request = self._warmup_request(f"warmup-{index}", length)
                if request is not None:
                    self._run_to_completion((request,))
            pair = (
                self._warmup_request("warmup-greedy", lengths[0]),
                self._warmup_request(
                    "warmup-sampled",
                    lengths[0] + _WARMUP_PAIR_LENGTH_GAP,
                    temperature=_WARMUP_SAMPLED_TEMPERATURE,
                    top_p=_WARMUP_SAMPLED_TOP_P,
                ),
            )
            self._run_to_completion(
                tuple(request for request in pair if request is not None)
            )
        finally:
            self._warming_up = False
        self.last_warmup_latency_ms = max(
            1, int((time.perf_counter() - started) * 1000)
        )

    def _run_to_completion(self, requests: tuple[ChatCompletionRequest, ...]) -> None:
        for request in requests:
            self.submit(request)
        while not self.idle():
            self.tick()

    def _warmup_request(
        self,
        request_id: str,
        prompt_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> ChatCompletionRequest | None:
        """Return a request whose prompt is about ``prompt_tokens`` long.

        Returns ``None`` when the request would not fit the KV pool.
        """

        words = self._warmup_word_count(prompt_tokens)
        prompt_limit = len(self._build_prompt(self._warmup_chat(request_id, words)))
        total = prompt_limit + _WARMUP_DECODE_TOKENS
        if self._max_total_tokens is not None and total > self._max_total_tokens:
            return None
        return self._warmup_chat(
            request_id,
            words,
            temperature=temperature,
            top_p=top_p,
            max_prompt_tokens=prompt_limit,
            max_total_tokens_per_request=total,
        )

    def _warmup_word_count(self, prompt_tokens: int) -> int:
        """Return how many filler words make a prompt about ``prompt_tokens`` long."""

        empty = len(self._build_prompt(self._warmup_chat("warmup-size", 0)))
        probe = len(
            self._build_prompt(self._warmup_chat("warmup-size", _WARMUP_PROBE_WORDS))
        )
        tokens_per_word = (probe - empty) / _WARMUP_PROBE_WORDS
        if tokens_per_word <= 0:
            return max(1, prompt_tokens)
        return max(1, round((prompt_tokens - empty) / tokens_per_word))

    def _warmup_chat(
        self,
        request_id: str,
        words: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_prompt_tokens: int = 0,
        max_total_tokens_per_request: int = 0,
    ) -> ChatCompletionRequest:
        from ..ipc import ChatMessage

        return ChatCompletionRequest(
            request_id=request_id,
            model=self._model_ref,
            messages=[
                ChatMessage(role="user", content=" ".join([_WARMUP_WORD] * words))
            ],
            max_tokens=_WARMUP_DECODE_TOKENS,
            temperature=temperature,
            top_p=top_p,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=_WARMUP_DECODE_TOKENS,
            max_total_tokens_per_request=max_total_tokens_per_request,
        )

    def submit(self, request: ChatCompletionRequest) -> None:
        tokenization_started_ns = time.perf_counter_ns()
        prompt = self._build_prompt(request)
        self._validate(request, prompt)
        state = _RuntimeState(
            request=request,
            prompt_token_ids=prompt,
            stop=_StopTextAssembler(tuple(stop for stop in request.stop if stop)),
            enqueued_at=time.perf_counter(),
            completion_token_ids=[],
            detokenizer=_new_streaming_detokenizer(self._prompt_tokenizer),
        )
        self._requests[request.request_id] = state
        if self._profiler is not None and not self._warming_up:
            self._profiler.begin_metal_capture(request.request_id)
            self._profiler.record(
                request.request_id,
                "runtime",
                "tokenization",
                started_ns=tokenization_started_ns,
                duration_us=(time.perf_counter_ns() - tokenization_started_ns) // 1_000,
                prompt_tokens=len(prompt),
                state="submitted",
                details={"stream": request.stream},
            )
        self._scheduler.submit(
            SchedulableRequest(
                request_id=request.request_id,
                prompt_token_ids=prompt,
                sampling=SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    seed=secrets.randbits(63),
                ),
                enqueued_at=state.enqueued_at,
                max_tokens=request.max_tokens,
                priority=0,
            )
        )

    def cancel(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        if state is not None:
            state.cancellation_stage = (
                "decode"
                if state.completion_token_ids
                else (
                    "python_waiting"
                    if state.prefill_time_ms == 0 and state.prompt_batch_size is None
                    else "prefill"
                )
            )
        handled = self._scheduler.cancel(request_id)
        if handled and self._profiler is not None:
            self._profiler.record(request_id, "runtime", "cancel_requested")
        return handled

    def tick(self) -> tuple[RuntimeEvent, ...]:
        output: list[RuntimeEvent] = []
        for event in self._scheduler.tick():
            self._apply_scheduler_event(event, output)
        return tuple(output)

    def idle(self) -> bool:
        return self._scheduler.idle()

    def benchmark_reset(
        self, *, clear_cache: bool, reset_counters: bool
    ) -> dict[str, Any]:
        """Reset idle benchmark state while preserving weights and graphs."""

        if not self.idle() or self._requests:
            raise RuntimeError("benchmark reset requires an idle scheduler")
        if reset_counters:
            self._cancellation_count = 0
            self._error_count = 0
        return self._scheduler.benchmark_reset(
            clear_cache=clear_cache,
            reset_counters=reset_counters,
        )

    def close(self) -> None:
        self._scheduler.close()
        self._requests.clear()

    def record_transport(
        self, request_id: str, stage: str, *, started_ns: int | None = None
    ) -> None:
        """Record a worker-transport boundary without moving IPC ownership."""

        if self._profiler is None:
            return
        self._profiler.record(
            request_id,
            "transport",
            stage,
            started_ns=started_ns,
            duration_us=(time.perf_counter_ns() - started_ns) // 1_000
            if started_ns is not None
            else 0,
        )

    def flush_profile(self) -> None:
        """Persist newly recorded transport events after IPC emission."""

        if self._profiler is not None:
            self._profiler.write_artifacts()

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
                        scheduler_select_ms=_optional_int(
                            metrics, "scheduler_select_ms"
                        ),
                        scheduler_cache_probe_ms=_optional_int(
                            metrics, "scheduler_cache_probe_ms"
                        ),
                        scheduler_cache_acquire_ms=_optional_int(
                            metrics, "scheduler_cache_acquire_ms"
                        ),
                        scheduler_cache_publish_ms=_optional_int(
                            metrics, "scheduler_cache_publish_ms"
                        ),
                        scheduler_apply_ms=_optional_int(metrics, "scheduler_apply_ms"),
                        scheduler_queue_wait_ms=_optional_int(
                            metrics, "scheduler_queue_wait_ms"
                        ),
                        cancellation_latency_ms=_optional_int(
                            metrics, "cancellation_latency_ms"
                        ),
                        scheduling_policy=_optional_str(metrics, "scheduling_policy"),
                        execution_mode=_optional_str(metrics, "execution_mode"),
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
                        executor_prepare_ms=_optional_int(
                            metrics, "executor_prepare_ms"
                        ),
                        executor_reserve_ms=_optional_int(
                            metrics, "executor_reserve_ms"
                        ),
                        executor_forward_ms=_optional_int(
                            metrics, "executor_forward_ms"
                        ),
                        executor_sample_ms=_optional_int(metrics, "executor_sample_ms"),
                        executor_dispatch_ms=_optional_int(
                            metrics, "executor_dispatch_ms"
                        ),
                        executor_eval_ms=_optional_int(metrics, "executor_eval_ms"),
                        executor_commit_ms=_optional_int(metrics, "executor_commit_ms"),
                        model_graph_embedding_ms=_optional_int(
                            metrics, "model_graph_embedding_ms"
                        ),
                        model_graph_projection_total_ms=_optional_int(
                            metrics, "model_graph_projection_total_ms"
                        ),
                        model_graph_attention_ms=_optional_int(
                            metrics, "model_graph_attention_ms"
                        ),
                        model_graph_mlp_total_ms=_optional_int(
                            metrics, "model_graph_mlp_total_ms"
                        ),
                        model_graph_norm_ms=_optional_int(
                            metrics, "model_graph_norm_ms"
                        ),
                        model_graph_lm_head_ms=_optional_int(
                            metrics, "model_graph_lm_head_ms"
                        ),
                        model_graph_layer_total_ms=_optional_int(
                            metrics, "model_graph_layer_total_ms"
                        ),
                        model_graph_worst_layer_ms=_optional_int(
                            metrics, "model_graph_worst_layer_ms"
                        ),
                        model_graph_worst_layer_index=_optional_int(
                            metrics, "model_graph_worst_layer_index"
                        ),
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
                        radix_nodes=_optional_int(metrics, "radix_nodes"),
                        radix_splits=_optional_int(metrics, "radix_splits"),
                        radix_shared_pages=_optional_int(metrics, "radix_shared_pages"),
                        radix_protected_pages=_optional_int(
                            metrics, "radix_protected_pages"
                        ),
                        radix_evictable_pages=_optional_int(
                            metrics, "radix_evictable_pages"
                        ),
                        radix_tree_depth=_optional_int(metrics, "radix_tree_depth"),
                        radix_leaf_evictions=_optional_int(
                            metrics, "radix_leaf_evictions"
                        ),
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
            state.cached_tokens = max(
                state.cached_tokens, int(metrics.get("cached_tokens", 0))
            )
            state.prefill_time_ms += int(metrics.get("step_time_ms", 0))
            state.prompt_batch_size = int(metrics.get("batch_size", 0))
            state.scheduler_queue_wait_ms = int(
                metrics.get(
                    "scheduler_queue_wait_ms",
                    metrics.get("queue_time_ms", 0),
                )
            )
            if self._profiler is not None:
                self._profiler.record(
                    event.request_id,
                    "runtime",
                    "prefill",
                    duration_us=int(metrics.get("step_time_ms", 0)) * 1_000,
                    phase="prefill",
                    prompt_tokens=len(state.prompt_token_ids),
                    details={
                        "scheduler_queue_wait_ms": state.scheduler_queue_wait_ms,
                        "batch_size": state.prompt_batch_size,
                    },
                )
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
            metrics = event.metrics or {}
            if metrics.get("cancellation_stage") is not None:
                state.cancellation_stage = str(metrics["cancellation_stage"])
            state.cancellation_latency_ms = _optional_int(
                metrics, "cancellation_latency_ms"
            )
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
        detokenize_started_ns = time.perf_counter_ns()
        if state.detokenizer is not None:
            state.detokenizer.add_token(token_id)
            delta = str(state.detokenizer.last_segment)
        else:
            text = str(
                self._decode_target.decode(
                    state.completion_token_ids,
                    skip_special_tokens=False,
                )
            )
            delta = text.removeprefix(state.decoded_prefix)
            state.decoded_prefix = text
        emitted, stop_hit = state.stop.push(delta)
        if self._profiler is not None:
            self._profiler.record(
                state.request.request_id,
                "runtime",
                "detokenization",
                started_ns=detokenize_started_ns,
                duration_us=(time.perf_counter_ns() - detokenize_started_ns) // 1_000,
                phase="decode",
                completion_tokens=len(state.completion_token_ids),
            )
        if emitted and state.request.stream:
            stream_started_ns = time.perf_counter_ns()
            output.append(
                RuntimeEvent(
                    kind="delta",
                    payload=ChatCompletionDelta(
                        request_id=state.request.request_id,
                        delta=emitted,
                    ),
                )
            )
            if self._profiler is not None:
                self._profiler.record(
                    state.request.request_id,
                    "streaming",
                    "emit_delta",
                    started_ns=stream_started_ns,
                    duration_us=(time.perf_counter_ns() - stream_started_ns) // 1_000,
                    phase="decode",
                    details={"chunk_chars": len(emitted)},
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
        final_emitted = ""
        if reason != "cancelled" and state.detokenizer is not None:
            state.detokenizer.finalize()
            final_segment = str(state.detokenizer.last_segment)
            if final_segment:
                final_emitted, _ = state.stop.push(final_segment)
        if final_emitted and state.request.stream:
            output.append(
                RuntimeEvent(
                    kind="delta",
                    payload=ChatCompletionDelta(
                        request_id=state.request.request_id,
                        delta=final_emitted,
                    ),
                )
            )
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
        if self._profiler is not None:
            self._profiler.end_metal_capture(request_id)
            self._profiler.record(
                request_id,
                "streaming",
                "response_assembly",
                details={
                    "stream": state.request.stream,
                    "chunks": len(state.completion_token_ids)
                    if state.request.stream
                    else 0,
                },
            )
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
                    prompt_cache_hit=state.cached_tokens > 0,
                    cached_tokens=state.cached_tokens or None,
                    prompt_batch_size=state.prompt_batch_size,
                    decode_batch_size=state.decode_batch_size,
                    backend="native-mlx",
                    modality="text",
                    scheduler_stage="cancelled"
                    if reason == "cancelled"
                    else "completed",
                    cancellation_stage=state.cancellation_stage,
                    queue_time_ms=state.scheduler_queue_wait_ms,
                    scheduler_queue_wait_ms=state.scheduler_queue_wait_ms,
                    cancellation_latency_ms=state.cancellation_latency_ms,
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
        if self._profiler is not None:
            self._profiler.end_metal_capture(request_id)
            self._profiler.record(
                request_id,
                "runtime",
                "terminal",
                duration_us=max(
                    0, int((time.perf_counter() - state.enqueued_at) * 1_000_000)
                ),
                prompt_tokens=len(state.prompt_token_ids),
                completion_tokens=len(state.completion_token_ids),
                state=reason,
                details={
                    "ttft_ms": state.ttft_ms,
                    "scheduler_queue_wait_ms": state.scheduler_queue_wait_ms,
                    "prefill_time_ms": state.prefill_time_ms,
                    "decode_time_ms": state.decode_time_ms,
                },
            )
            self._profiler.write_artifacts()

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
        if self._profiler is not None:
            self._profiler.record(
                request_id,
                "runtime",
                "terminal",
                state="error",
                error=f"{code}: {message}",
            )
            self._profiler.write_artifacts()

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
        if not math.isfinite(request.temperature) or request.temperature < 0.0:
            raise ValueError("temperature must be a non-negative number")
        if not math.isfinite(request.top_p) or not 0.0 < request.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if any(not value for value in request.stop):
            raise ValueError("stop sequences must not be empty")
        if len(prompt) > request.max_prompt_tokens:
            raise ValueError("prompt exceeds max_prompt_tokens")
        if request.max_tokens > request.max_completion_tokens:
            raise ValueError("completion exceeds max_completion_tokens")
        if len(prompt) + request.max_tokens > request.max_total_tokens_per_request:
            raise ValueError("request exceeds max_total_tokens_per_request")
        if (
            self._max_total_tokens is not None
            and len(prompt) + request.max_tokens > self._max_total_tokens
        ):
            raise ValueError(
                "request exceeds max_total_tokens_per_request "
                f"{self._max_total_tokens} derived from the KV cache pool; raise "
                "--memory-utilization or --kv-cache-memory"
            )


def _optional_int(metrics: dict[str, object], name: str) -> int | None:
    value = metrics.get(name)
    return None if value is None else int(value)


def _optional_str(metrics: dict[str, object], name: str) -> str | None:
    value = metrics.get(name)
    return None if value is None else str(value)


def _new_streaming_detokenizer(tokenizer: Any) -> Any | None:
    """Create one request-local streaming detokenizer when supported."""

    try:
        detokenizer = tokenizer.detokenizer
    except (AttributeError, TypeError):
        return None
    return detokenizer if callable(getattr(detokenizer, "add_token", None)) else None
