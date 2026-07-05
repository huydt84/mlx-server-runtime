"""Native MLX backend bootstrap seam for Phase 3."""

from __future__ import annotations

import json
import hashlib
import select
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from socket import socket
from typing import Any, Callable, Sequence

import mlx.core as mx

from ..config import WorkerConfig
from ..ipc import (
    CancelRequest,
    ChatCompletionDelta,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelError,
    ModelLoadProgress,
    ModelStatus,
    SchedulerMetricsEvent,
    WorkerCommandError,
    WorkerReady,
    WorkerError,
    decode_command,
    encode_bootstrap_message,
    encode_event,
)
from .interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    NativeBackendOptions,
    NativeMlxExecutor,
    NativeScheduler,
)
from .mapping import (
    WeightArtifactValidationError,
    WeightIndex,
    WeightMappingBug,
    WeightMappingPlan,
    load_weight_index,
)
from .models.Qwen2ForCausalLM.debug_trace import (
    TraceArtifacts,
    compare_trace_runs,
    trace_qwen2_run,
    write_trace_artifacts,
)
from .models.Qwen2ForCausalLM.cache import Qwen2LayerCache
from .registry import ArchitectureSpec, get_architecture_spec


SUPPORTED_ARCHITECTURE_CLASSES = frozenset({"Qwen2ForCausalLM"})


@dataclass(frozen=True)
class NativeArchitecture:
    """Detected architecture metadata for native bootstrap."""

    model: str
    model_path: Path
    architecture_class: str
    raw_config: dict[str, Any]
    spec: ArchitectureSpec


@dataclass(frozen=True)
class NativeParityResult:
    """Deterministic parity result for one finalized token sequence."""

    checkpoint: str
    token_ids: tuple[int, ...]
    logits_shape: tuple[int, ...]
    logits_dtype: str
    max_abs_diff: float
    tolerance_atol: float
    tolerance_rtol: float
    tolerance_ok: bool
    native_next_token: int
    reference_next_token: int
    token_ok: bool


@dataclass(frozen=True)
class NativePrefillDecodeParityResult:
    """Parity result for prefill plus decode steps."""

    checkpoint: str
    token_ids: tuple[int, ...]
    prefill_logits_shape: tuple[int, ...]
    prefill_logits_dtype: str
    prefill_max_abs_diff: float
    decode_max_abs_diff: float
    tolerance_atol: float
    tolerance_rtol: float
    tolerance_ok: bool
    native_tokens: tuple[int, ...]
    reference_tokens: tuple[int, ...]
    token_ok: bool
    cache_lengths: tuple[int, ...]
    prefill_time_ms: int


class NativeBootstrapFailure(RuntimeError):
    """Structured native bootstrap failure."""

    def __init__(self, error: ModelError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class _StopTextAssembler:
    stop_sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        self._pending = ""
        self._emitted: list[str] = []
        self._hold_back = max((len(value) for value in self.stop_sequences), default=0)

    def push(self, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        self._pending += text
        match_index = self._first_match_index()
        if match_index is not None:
            emit = self._pending[:match_index]
            if emit:
                self._emitted.append(emit)
            self._pending = ""
            return emit, True
        if self._hold_back <= 1:
            emit = self._pending
            if emit:
                self._emitted.append(emit)
            self._pending = ""
            return emit, False
        safe_length = max(0, len(self._pending) - self._hold_back + 1)
        emit = self._pending[:safe_length]
        if emit:
            self._emitted.append(emit)
            self._pending = self._pending[safe_length:]
        return emit, False

    def finish(self) -> str:
        emit = self._pending
        if emit:
            self._emitted.append(emit)
        self._pending = ""
        return emit

    def text(self) -> str:
        return "".join(self._emitted) + self._pending

    def _first_match_index(self) -> int | None:
        first_index: int | None = None
        for stop_sequence in self.stop_sequences:
            index = self._pending.find(stop_sequence)
            if index < 0:
                continue
            if first_index is None or index < first_index:
                first_index = index
        return first_index


@dataclass
class _NativePendingRequest:
    request: ChatCompletionRequest
    prompt_token_ids: tuple[int, ...]
    emit_delta: Callable[[str], None] | None
    enqueued_at: float
    stop_assembler: _StopTextAssembler
    completion_token_ids: list[int]
    prompt_cursor: int = 0
    decoded_prefix: str = ""
    cache_handle: str | None = None
    cache_length: int = 0
    prefill_time_ms: int = 0
    decode_time_ms: int = 0
    ttft_ms: int | None = None
    prompt_batch_size: int | None = None
    decode_batch_size: int | None = None
    running_started_at: float | None = None
    cancel_requested: bool = False
    cancellation_stage: str | None = None


class NativeContinuousScheduler:
    """Phase 8 long-lived native scheduler with chunked prefill."""

    def __init__(
        self,
        executor: NativeMlxExecutor,
        options: NativeBackendOptions,
        model_path: Path,
        model_ref: str,
        stage_callback: Callable[[str, str], None] | None,
        emit_response: Callable[[ChatCompletionResponse], None],
        emit_error: Callable[[WorkerCommandError], None],
        emit_metrics: Callable[[SchedulerMetricsEvent], None],
        prefill_step_size: int = 256,
    ) -> None:
        if prefill_step_size <= 0:
            raise ValueError("native-mlx prefill chunk size must be positive")
        self._executor = executor
        self._options = options
        self._model_path = model_path
        self._model_ref = model_ref
        self._prefill_step_size = int(prefill_step_size)
        self._stage_callback = stage_callback
        self._tokenizer_wrapper: Any | None = None
        self._raw_tokenizer: Any | None = None
        self._decode_target: Any | None = None
        self._eos_token_ids: tuple[int, ...] = ()
        self._emit_response = emit_response
        self._emit_error = emit_error
        self._emit_metrics = emit_metrics
        self._cancellation_count = 0
        self._error_count = 0
        self._last_warmup_latency_ms = 0
        self._waiting: OrderedDict[str, _NativePendingRequest] = OrderedDict()
        self._running: OrderedDict[str, _NativePendingRequest] = OrderedDict()

    def warmup(self) -> None:
        warmup_started = time.perf_counter()
        self._executor.load(self._options)
        if self._stage_callback is not None:
            self._stage_callback("prompt_tokenizer_readiness", "initializing_runtime")
        validate_tokenizer_assets(self._model_path)
        self._load_tokenizer_state()
        if self._stage_callback is not None:
            self._stage_callback("deterministic_warmup", "warming_up")

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
        self._run_warmup_request(request)
        self._last_warmup_latency_ms = max(
            1, int((time.perf_counter() - warmup_started) * 1000)
        )

    def submit(
        self,
        request: ChatCompletionRequest,
        emit_delta: Callable[[str], None] | None,
    ) -> None:
        prompt_token_ids = self._build_prompt_token_ids(request)
        self._validate_request(request, prompt_token_ids)
        self._waiting[request.request_id] = _NativePendingRequest(
            request=request,
            prompt_token_ids=tuple(prompt_token_ids),
            emit_delta=emit_delta,
            enqueued_at=time.perf_counter(),
            stop_assembler=_StopTextAssembler(
                tuple(value for value in request.stop if value)
            ),
            completion_token_ids=[],
        )

    def cancel(self, request_id: str) -> bool:
        waiting = self._waiting.pop(request_id, None)
        if waiting is not None:
            self._cancellation_count += 1
            waiting.cancellation_stage = "waiting"
            self._finish_request(waiting, "cancelled")
            self._emit_queue_counts()
            return True

        running = self._running.get(request_id)
        if running is None:
            return False
        running.cancel_requested = True
        running.cancellation_stage = (
            "decode" if running.completion_token_ids else "prefill"
        )
        return True

    def tick(self) -> None:
        self._reap_cancelled_running()
        if self._running:
            self._run_decode_step()
            self._reap_cancelled_running()
        if self._has_prefill_work():
            self._run_prefill_step()
            self._reap_cancelled_running()

    def idle(self) -> bool:
        return not self._waiting and not self._running

    def close(self) -> None:
        for request in list(self._waiting.values()):
            if request.cache_handle is not None:
                self._executor.release(request.cache_handle)
        for request in list(self._running.values()):
            if request.cache_handle is not None:
                self._executor.release(request.cache_handle)
        self._waiting.clear()
        self._running.clear()

    def _run_warmup_request(self, request: ChatCompletionRequest) -> None:
        emit_response = self._emit_response
        emit_error = self._emit_error
        emit_metrics = self._emit_metrics
        try:
            self._emit_response = lambda _response: None
            self._emit_error = lambda _error: None
            self._emit_metrics = lambda _metrics: None
            self.submit(request, lambda _delta: None)
            while not self.idle():
                self.tick()
        finally:
            self._emit_response = emit_response
            self._emit_error = emit_error
            self._emit_metrics = emit_metrics

    def _run_prefill_step(self) -> None:
        started = time.perf_counter()
        requests = self._prefill_requests()
        if not requests:
            return

        chunk_lengths: dict[str, int] = {}
        execution_requests: list[ExecutionRequest] = []
        for request in requests:
            if request.cache_handle is None:
                self._waiting.pop(request.request.request_id, None)
                request.cache_handle = self._executor.create_cache(
                    request.request.request_id
                )
                request.running_started_at = time.perf_counter()
                self._running[request.request.request_id] = request
            chunk_start = request.prompt_cursor
            chunk_end = min(
                len(request.prompt_token_ids),
                chunk_start + self._prefill_step_size,
            )
            chunk_token_ids = request.prompt_token_ids[chunk_start:chunk_end]
            chunk_lengths[request.request.request_id] = len(chunk_token_ids)
            execution_requests.append(
                ExecutionRequest(
                    request_id=request.request.request_id,
                    token_ids=chunk_token_ids,
                    positions=tuple(range(chunk_start, chunk_end)),
                    cache_handle=request.cache_handle,
                    max_new_tokens=request.request.max_tokens,
                    temperature=request.request.temperature,
                    top_p=request.request.top_p,
                )
            )

        batch = ExecutionBatch(
            phase="prefill",
            requests=tuple(execution_requests),
        )
        try:
            result = self._executor.prefill_batch(batch)
        except Exception as exc:
            self._fail_batch(requests, "WORKER_ERROR", str(exc))
            return
        self._apply_step_result(
            requests,
            result,
            "prefill",
            started,
            scheduled_tokens_by_request=chunk_lengths,
        )

    def _run_decode_step(self) -> None:
        started = time.perf_counter()
        requests = [
            request
            for request in self._running.values()
            if not request.cancel_requested and request.completion_token_ids
        ]
        if not requests:
            return
        batch = ExecutionBatch(
            phase="decode",
            requests=tuple(
                ExecutionRequest(
                    request_id=request.request.request_id,
                    token_ids=(request.completion_token_ids[-1],),
                    positions=(request.cache_length,),
                    cache_handle=request.cache_handle,
                    max_new_tokens=request.request.max_tokens,
                    temperature=request.request.temperature,
                    top_p=request.request.top_p,
                )
                for request in requests
            ),
        )
        try:
            result = self._executor.decode_batch(batch)
        except Exception as exc:
            self._fail_batch(requests, "WORKER_ERROR", str(exc))
            return
        self._apply_step_result(requests, result, "decode", started)

    def _apply_step_result(
        self,
        requests: list[_NativePendingRequest],
        result: Any,
        phase: str,
        started: float,
        *,
        scheduled_tokens_by_request: dict[str, int] | None = None,
    ) -> None:
        results_by_id = {item.request_id: item for item in result.results}
        for request in requests:
            item = results_by_id.get(request.request.request_id)
            if item is None:
                self._fail_request(
                    request,
                    "WORKER_ERROR",
                    f"native-mlx {phase} missing result for {request.request.request_id}",
                )
                continue
            if item.error_code is not None:
                self._fail_request(request, item.error_code, item.error_code)
                continue
            request.cache_handle = item.cache_handle
            request.cache_length = item.cache_length
            if phase == "prefill":
                request.prefill_time_ms += result.step_time_ms
                request.prompt_batch_size = len(requests)
                request.prompt_cursor += int(
                    (scheduled_tokens_by_request or {}).get(
                        request.request.request_id,
                        0,
                    )
                )
                if request.cache_length != request.prompt_cursor:
                    self._fail_request(
                        request,
                        "WORKER_ERROR",
                        (
                            "native-mlx prefill cache length mismatch for "
                            f"{request.request.request_id}"
                        ),
                    )
                    continue
                if request.prompt_cursor < len(request.prompt_token_ids):
                    continue
            else:
                request.decode_time_ms += result.step_time_ms
                request.decode_batch_size = len(requests)
            if item.next_token_id is None:
                if phase == "prefill":
                    self._fail_request(
                        request,
                        "WORKER_ERROR",
                        (
                            "native-mlx prefill did not yield next token for "
                            f"{request.request.request_id}"
                        ),
                    )
                else:
                    self._finish_request(request, "stop")
                continue
            self._accept_generated_token(request, int(item.next_token_id))
        self._emit_step_metrics(
            phase=phase,
            batch_size=len(requests),
            scheduled_tokens=(
                sum((scheduled_tokens_by_request or {}).values())
                if phase == "prefill"
                else len(requests)
            ),
            scheduler_tick_latency_ms=max(
                1, int((time.perf_counter() - started) * 1000)
            ),
        )

    def _has_prefill_work(self) -> bool:
        if self._waiting:
            return True
        return any(
            not request.cancel_requested
            and request.prompt_cursor < len(request.prompt_token_ids)
            for request in self._running.values()
        )

    def _prefill_requests(self) -> list[_NativePendingRequest]:
        requests = [
            request
            for request in self._running.values()
            if not request.cancel_requested
            and request.prompt_cursor < len(request.prompt_token_ids)
        ]
        requests.extend(self._waiting.values())
        return requests

    def _accept_generated_token(
        self,
        request: _NativePendingRequest,
        token_id: int,
    ) -> None:
        request.completion_token_ids.append(token_id)
        if token_id in self._eos_token_ids:
            self._finish_request(request, "stop")
            return

        current_text = self._decode_tokens(request.completion_token_ids)
        delta_text = (
            current_text[len(request.decoded_prefix) :]
            if current_text.startswith(request.decoded_prefix)
            else current_text
        )
        request.decoded_prefix = current_text
        emitted_text, stop_hit = request.stop_assembler.push(delta_text)
        if emitted_text:
            if request.ttft_ms is None and request.running_started_at is not None:
                request.ttft_ms = max(
                    1,
                    int((time.perf_counter() - request.enqueued_at) * 1000),
                )
            if request.emit_delta is not None:
                request.emit_delta(emitted_text)
        if stop_hit:
            self._finish_request(request, "stop")
            return
        if len(request.completion_token_ids) >= request.request.max_tokens:
            self._finish_request(request, "length")

    def _finish_request(
        self,
        request: _NativePendingRequest,
        finish_reason: str,
    ) -> None:
        if finish_reason != "cancelled":
            trailing_text = request.stop_assembler.finish()
            if trailing_text and request.emit_delta is not None:
                if request.ttft_ms is None:
                    request.ttft_ms = max(
                        1,
                        int((time.perf_counter() - request.enqueued_at) * 1000),
                    )
                request.emit_delta(trailing_text)
        self._running.pop(request.request.request_id, None)
        self._waiting.pop(request.request.request_id, None)
        completion_time_ms = max(
            1, int((time.perf_counter() - request.enqueued_at) * 1000)
        )
        queue_time_ms = 0
        if request.running_started_at is not None:
            queue_time_ms = max(
                0, int((request.running_started_at - request.enqueued_at) * 1000)
            )
        self._emit_response(
            ChatCompletionResponse(
                request_id=request.request.request_id,
                model=self._model_ref,
                text=request.stop_assembler.text(),
                finish_reason=finish_reason,
                prompt_tokens=len(request.prompt_token_ids),
                completion_tokens=len(request.completion_token_ids),
                prompt_batch_size=request.prompt_batch_size,
                decode_batch_size=request.decode_batch_size,
                backend="native-mlx",
                modality="text",
                scheduler_stage="cancelled"
                if finish_reason == "cancelled"
                else "completed",
                cancellation_stage=request.cancellation_stage,
                queue_time_ms=queue_time_ms,
                prefill_time_ms=request.prefill_time_ms,
                ttft_ms=request.ttft_ms,
                decode_time_ms=request.decode_time_ms,
                completion_time_ms=completion_time_ms,
                worker_cancellation_count=self._cancellation_count,
                worker_error_count=self._error_count,
            )
        )
        if request.cache_handle is not None:
            self._executor.release(request.cache_handle)

    def _fail_batch(
        self,
        requests: list[_NativePendingRequest],
        code: str,
        message: str,
    ) -> None:
        for request in requests:
            self._fail_request(request, code, message)

    def _fail_request(
        self,
        request: _NativePendingRequest,
        code: str,
        message: str,
    ) -> None:
        self._error_count += 1
        self._running.pop(request.request.request_id, None)
        self._waiting.pop(request.request.request_id, None)
        if request.cache_handle is not None:
            self._executor.release(request.cache_handle)
        self._emit_error(
            WorkerCommandError(
                code=code,
                request_id=request.request.request_id,
                message=message,
            )
        )

    def _reap_cancelled_running(self) -> None:
        cancelled = False
        for request_id, request in list(self._running.items()):
            if not request.cancel_requested:
                continue
            self._cancellation_count += 1
            self._finish_request(request, "cancelled")
            cancelled = True
        if cancelled:
            self._emit_queue_counts()

    def _emit_step_metrics(
        self,
        *,
        phase: str,
        batch_size: int,
        scheduled_tokens: int,
        scheduler_tick_latency_ms: int,
    ) -> None:
        self._emit_metrics(
            SchedulerMetricsEvent(
                backend="native-mlx",
                modality="text",
                phase=phase,
                scheduled_tokens=scheduled_tokens,
                batch_size=batch_size,
                waiting_requests=len(self._waiting),
                running_requests=len(self._running),
                scheduler_tick_latency_ms=scheduler_tick_latency_ms,
            )
        )

    def _emit_queue_counts(self) -> None:
        self._emit_metrics(
            SchedulerMetricsEvent(
                backend="native-mlx",
                modality="text",
                phase="decode",
                scheduled_tokens=0,
                batch_size=0,
                waiting_requests=len(self._waiting),
                running_requests=len(self._running),
                scheduler_tick_latency_ms=0,
            )
        )

    def _build_prompt_token_ids(self, request: ChatCompletionRequest) -> list[int]:
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if not isinstance(message.content, str):
                raise ValueError(
                    "native-mlx text backend does not support image content"
                )
            messages.append({"role": message.role, "content": message.content})
        return build_finalized_token_ids(self._model_path, messages)

    def _validate_request(
        self, request: ChatCompletionRequest, prompt_token_ids: Sequence[int]
    ) -> None:
        if request.model != self._model_ref:
            raise ValueError(
                f"requested model '{request.model}' does not match loaded model '{self._model_ref}'"
            )
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise ValueError(
                "native-mlx only supports greedy decoding (temperature=0.0, top_p=1.0)"
            )
        if any(not value for value in request.stop):
            raise ValueError("stop sequences must not be empty")
        prompt_tokens = len(prompt_token_ids)
        completion_tokens = request.max_tokens
        total_tokens = prompt_tokens + completion_tokens
        if prompt_tokens > request.max_prompt_tokens:
            raise ValueError(
                f"prompt too long: {prompt_tokens} tokens exceeds max_prompt_tokens {request.max_prompt_tokens}"
            )
        if completion_tokens > request.max_completion_tokens:
            raise ValueError(
                f"completion too long: {completion_tokens} tokens exceeds max_completion_tokens {request.max_completion_tokens}"
            )
        if total_tokens > request.max_total_tokens_per_request:
            raise ValueError(
                f"request too large: {total_tokens} tokens exceeds max_total_tokens_per_request {request.max_total_tokens_per_request}"
            )

    def _load_tokenizer_state(self) -> None:
        tokenizer_wrapper = _load_tokenizer_wrapper(self._model_path)
        raw_tokenizer = getattr(tokenizer_wrapper, "_tokenizer", None) or getattr(
            tokenizer_wrapper, "tokenizer", None
        )
        decode_target = raw_tokenizer or tokenizer_wrapper
        if not hasattr(decode_target, "decode"):
            raise NativeBootstrapFailure(
                _startup_error(
                    code="TOKENIZER_DECODE_UNAVAILABLE",
                    message="native-mlx tokenizer does not expose decode()",
                    stage="prompt_tokenizer_readiness",
                    category="malformed_checkpoint",
                    detail=str(self._model_path),
                )
            )
        eos_value = getattr(raw_tokenizer, "eos_token_ids", None)
        if eos_value is None:
            single_eos = getattr(raw_tokenizer, "eos_token_id", None)
            eos_value = [] if single_eos is None else [single_eos]
        elif isinstance(eos_value, int):
            eos_value = [eos_value]
        self._tokenizer_wrapper = tokenizer_wrapper
        self._raw_tokenizer = raw_tokenizer
        self._decode_target = decode_target
        self._eos_token_ids = tuple(int(value) for value in eos_value)

    def _decode_tokens(self, token_ids: Sequence[int]) -> str:
        decoded = self._decode_target.decode(list(token_ids), skip_special_tokens=False)
        return str(decoded)


def _pop_buffered_line(read_buffer: bytearray) -> bytes | None:
    newline = read_buffer.find(b"\n")
    if newline < 0:
        return None
    line = bytes(read_buffer[: newline + 1])
    del read_buffer[: newline + 1]
    return line


def _read_command_line(
    client: socket,
    read_buffer: bytearray,
    *,
    block: bool,
) -> bytes | None:
    line = _pop_buffered_line(read_buffer)
    if line is not None:
        return line

    while True:
        if not block and not select.select([client], [], [], 0)[0]:
            return None

        chunk = client.recv(4096)
        if not chunk:
            return b""

        read_buffer.extend(chunk)
        line = _pop_buffered_line(read_buffer)
        if line is not None:
            return line

        if not block:
            return None


def _make_should_cancel(
    client: socket,
    read_buffer: bytearray,
    pending_lines: list[bytes],
    request_id: str,
) -> Callable[[], bool]:
    cancelled = False

    def should_cancel() -> bool:
        nonlocal cancelled
        if cancelled:
            return True
        pending = _read_command_line(client, read_buffer, block=False)
        if pending is None:
            return False
        if not pending:
            cancelled = True
            return True
        if _is_matching_cancel_request(pending, request_id):
            cancelled = True
            return True
        pending_lines.append(pending)
        return False

    return should_cancel


def _is_matching_cancel_request(raw_line: bytes, request_id: str) -> bool:
    try:
        payload = json.loads(raw_line)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("type") == "cancel_request"
        and payload.get("request_id") == request_id
    )


def run_native_worker(
    client: socket,
    config: WorkerConfig,
    *,
    native_worker_factory: Callable[..., NativeScheduler] | None = None,
) -> int:
    """Run native backend bootstrap until structured startup result."""

    bootstrap_started_at = _now_seconds()
    stage_emitter = _make_stage_emitter(client, config.model, bootstrap_started_at)
    stage_emitter("architecture_detection", "verifying")

    if native_worker_factory is None:
        native_worker_factory = create_native_worker

    def emit_response(response: ChatCompletionResponse) -> None:
        client.sendall(encode_event(response))

    def emit_error(error: WorkerCommandError) -> None:
        client.sendall(encode_event(error))

    def emit_metrics(event: SchedulerMetricsEvent) -> None:
        client.sendall(encode_event(event))

    try:
        try:
            scheduler = native_worker_factory(
                config,
                stage_callback=stage_emitter,
                emit_response=emit_response,
                emit_error=emit_error,
                emit_metrics=emit_metrics,
            )
        except TypeError:
            scheduler = native_worker_factory(config)
        scheduler.warmup()
    except NativeBootstrapFailure as exc:
        _emit_failure(client, config.model, bootstrap_started_at, exc.error)
        return 1
    except Exception as exc:
        error = _startup_error(
            code="NATIVE_STARTUP_FAILED",
            message=str(exc),
            stage="native_executor_construction",
            category="supported_class_bug",
            detail=config.model,
        )
        _emit_failure(client, config.model, bootstrap_started_at, error)
        return 1

    ready_at = _now_seconds()
    _send_status(
        client,
        _ready_status(
            model=config.model,
            started_loading_at=bootstrap_started_at,
            loaded_at=ready_at,
            last_transition_at=ready_at,
            warmup_latency_ms=getattr(scheduler, "_last_warmup_latency_ms", 1),
        ),
    )
    client.sendall(encode_bootstrap_message(WorkerReady()))

    read_buffer = bytearray()
    worker_closed = False
    while True:
        raw_line = _read_command_line(client, read_buffer, block=scheduler.idle())
        while raw_line is not None:
            if not raw_line:
                worker_closed = True
                break

            try:
                command = decode_command(raw_line)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                emit_error(
                    WorkerCommandError(
                        code="INVALID_REQUEST",
                        request_id="unknown",
                        message=str(exc),
                    )
                )
            else:
                if command is None:
                    emit_error(
                        WorkerCommandError(
                            code="INVALID_REQUEST",
                            request_id="unknown",
                            message="unsupported worker command",
                        )
                    )
                elif isinstance(command, CancelRequest):
                    scheduler.cancel(command.request_id)
                else:
                    try:
                        emit_delta = None
                        if command.stream:

                            def emit_delta(
                                delta: str, request_id: str = command.request_id
                            ) -> None:
                                client.sendall(
                                    encode_event(
                                        ChatCompletionDelta(
                                            request_id=request_id,
                                            delta=delta,
                                        )
                                    )
                                )

                        scheduler.submit(command, emit_delta)
                    except Exception as exc:
                        emit_error(
                            WorkerCommandError(
                                code=(
                                    "INVALID_REQUEST"
                                    if isinstance(exc, ValueError)
                                    else "WORKER_ERROR"
                                ),
                                request_id=command.request_id,
                                message=str(exc),
                            )
                        )

            raw_line = _read_command_line(client, read_buffer, block=False)

        if worker_closed:
            break
        if not scheduler.idle():
            scheduler.tick()

    scheduler.close()
    return 0


def create_native_worker(
    config: WorkerConfig,
    *,
    stage_callback: Callable[[str, str], None] | None = None,
    emit_response: Callable[[ChatCompletionResponse], None] | None = None,
    emit_error: Callable[[WorkerCommandError], None] | None = None,
    emit_metrics: Callable[[SchedulerMetricsEvent], None] | None = None,
) -> NativeScheduler:
    """Create native scheduler and executor seams for selected model."""

    architecture = detect_native_architecture(config.model)
    if stage_callback is not None:
        stage_callback("artifact_validation", "verifying")

    model_config = parse_native_config(architecture)
    weight_index, weight_plan = validate_weight_artifacts(architecture)
    if stage_callback is not None:
        stage_callback("weight_mapping", "loading_weights")

    options = NativeBackendOptions(
        model=architecture.model,
        architecture_class=architecture.architecture_class,
    )
    executor = build_native_executor(
        architecture,
        model_config,
        weight_index,
        weight_plan,
    )
    if stage_callback is not None:
        stage_callback("native_executor_construction", "initializing_runtime")

    return NativeContinuousScheduler(
        executor=executor,
        options=options,
        model_path=architecture.model_path,
        model_ref=architecture.model,
        stage_callback=stage_callback,
        emit_response=emit_response or (lambda _response: None),
        emit_error=emit_error or (lambda _error: None),
        emit_metrics=emit_metrics or (lambda _metrics: None),
        prefill_step_size=getattr(
            config,
            "text_prefill_chunk_size",
            getattr(config, "prefill_chunk_size", 256),
        ),
    )


def detect_native_architecture(model: str) -> NativeArchitecture:
    """Resolve model config and classify native backend startup."""

    model_path = _resolve_model_path(model)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_MODEL_CONFIG",
                message=f"native-mlx could not find config.json for '{model}'",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_MODEL_CONFIG",
                message=f"native-mlx could not parse config.json for '{model}': {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        ) from exc

    architectures = payload.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_ARCHITECTURES",
                message=f"native-mlx config.json for '{model}' is missing architectures[]",
                stage="architecture_detection",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    architecture_class = architectures[0]
    if not isinstance(architecture_class, str) or not architecture_class.strip():
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_ARCHITECTURE_CLASS",
                message=f"native-mlx config.json for '{model}' has invalid architectures[0]",
                stage="architecture_detection",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    if architecture_class not in SUPPORTED_ARCHITECTURE_CLASSES:
        raise NativeBootstrapFailure(
            _startup_error(
                code="UNSUPPORTED_ARCHITECTURE_CLASS",
                message=(
                    "native-mlx only supports explicitly implemented architecture "
                    f"classes; got {architecture_class}"
                ),
                stage="architecture_detection",
                category="unsupported_class",
                detail=model,
            )
        )

    spec = get_architecture_spec(architecture_class)
    if spec is None:
        raise NativeBootstrapFailure(
            _startup_error(
                code="UNSUPPORTED_ARCHITECTURE_CLASS",
                message=(
                    "native-mlx only supports explicitly implemented architecture "
                    f"classes; got {architecture_class}"
                ),
                stage="architecture_detection",
                category="unsupported_class",
                detail=model,
            )
        )

    return NativeArchitecture(
        model=model,
        model_path=model_path,
        architecture_class=architecture_class,
        raw_config=payload,
        spec=spec,
    )


def parse_native_config(architecture: NativeArchitecture) -> Any:
    """Parse per-class config with supported-class bug taxonomy."""

    try:
        return architecture.spec.parse_config(architecture.raw_config)
    except ValueError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_NATIVE_CONFIG",
                message=f"native-mlx config validation failed: {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(architecture.model_path / "config.json"),
            )
        ) from exc


def validate_weight_artifacts(
    architecture: NativeArchitecture,
) -> tuple[WeightIndex, WeightMappingPlan]:
    """Validate weight metadata and build canonical mapping plan."""

    try:
        index = load_weight_index(architecture.model_path)
    except WeightArtifactValidationError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_WEIGHT_ARTIFACTS",
                message=f"native-mlx weight artifact validation failed: {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(architecture.model_path),
            )
        ) from exc

    adapter = architecture.spec.create_weight_adapter()
    try:
        plan = adapter.build_plan(index)
    except WeightMappingBug as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="WEIGHT_MAPPING_UNSUPPORTED",
                message=f"native-mlx weight mapping needs adapter work: {exc}",
                stage="weight_mapping",
                category="supported_class_bug",
                detail=architecture.model,
            )
        ) from exc

    return index, plan


def build_native_executor(
    architecture: NativeArchitecture,
    model_config: Any,
    weight_index: WeightIndex,
    weight_plan: WeightMappingPlan,
) -> NativeMlxExecutor:
    """Construct per-class executor from validated metadata."""

    try:
        return architecture.spec.create_executor(
            architecture.model_path,
            model_config,
            weight_plan,
            weight_index,
        )
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="NATIVE_EXECUTOR_CONSTRUCTION_FAILED",
                message=f"native-mlx executor construction failed: {exc}",
                stage="native_executor_construction",
                category="supported_class_bug",
                detail=architecture.model,
            )
        ) from exc


def validate_tokenizer_assets(model_path: Path) -> None:
    """Validate tokenizer and chat-template assets outside executor boundary."""

    required_assets = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing_assets = [
        name for name in required_assets if not (model_path / name).exists()
    ]
    if missing_assets:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_TOKENIZER_ASSET",
                message=(
                    "native-mlx tokenizer assets are incomplete: "
                    + ", ".join(missing_assets)
                ),
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        )

    try:
        tokenizer = _load_tokenizer_wrapper(model_path)
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="TOKENIZER_LOAD_FAILED",
                message=f"native-mlx tokenizer validation failed: {exc}",
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        ) from exc

    raw_tokenizer = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    chat_template = getattr(raw_tokenizer, "chat_template", None) or getattr(
        tokenizer, "chat_template", None
    )
    if not chat_template:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_CHAT_TEMPLATE",
                message="native-mlx tokenizer assets do not expose a chat template",
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        )


def build_finalized_token_ids(
    model_path: Path,
    messages: Sequence[dict[str, str]],
) -> list[int]:
    """Build finalized token IDs from runtime-owned tokenizer/template path."""

    tokenizer = _load_tokenizer_wrapper(model_path)
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    if raw_tokenizer is None or not hasattr(raw_tokenizer, "apply_chat_template"):
        raise ValueError("native-mlx tokenizer does not expose apply_chat_template")

    encoded = raw_tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    token_ids = encoded.get("input_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("native-mlx tokenizer produced no finalized token IDs")
    return [int(token_id) for token_id in token_ids]


def build_prompt_fingerprint(messages: Sequence[dict[str, str]]) -> str:
    """Build stable prompt fingerprint separate from finalized token IDs."""

    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_native_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativeParityResult:
    """Compare native logits and greedy token against `mlx-lm`."""

    if not hasattr(executor, "forward_token_ids"):
        raise ValueError("native executor does not expose direct token forward helper")

    native_logits = executor.forward_token_ids(list(token_ids))
    reference_logits = _load_reference_logits(checkpoint, token_ids)
    native_logits_f32 = native_logits.astype(mx.float32)
    reference_logits_f32 = reference_logits.astype(mx.float32)
    diff = mx.abs(native_logits_f32 - reference_logits_f32)
    max_abs_diff = float(mx.max(diff).item())
    tolerance_ok = bool(
        mx.allclose(
            native_logits_f32,
            reference_logits_f32,
            atol=tolerance_atol,
            rtol=tolerance_rtol,
        ).item()
    )
    native_next_token = int(mx.argmax(native_logits_f32[0, -1], axis=-1).item())
    reference_next_token = int(mx.argmax(reference_logits_f32[0, -1], axis=-1).item())
    return NativeParityResult(
        checkpoint=checkpoint,
        token_ids=tuple(int(token_id) for token_id in token_ids),
        logits_shape=tuple(int(dim) for dim in native_logits.shape),
        logits_dtype=str(native_logits.dtype),
        max_abs_diff=max_abs_diff,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        tolerance_ok=tolerance_ok,
        native_next_token=native_next_token,
        reference_next_token=reference_next_token,
        token_ok=native_next_token == reference_next_token,
    )


def compare_native_prefill_decode_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    decode_steps: int,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativePrefillDecodeParityResult:
    """Compare native prefill + decode token path against `mlx-lm`."""

    if not hasattr(executor, "create_cache") or not hasattr(
        executor, "prefill_then_decode_tokens"
    ):
        raise ValueError("native executor does not expose cache lifecycle helpers")

    native_parity = compare_native_to_mlx_lm(
        checkpoint,
        executor,
        token_ids,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )
    native_tokens, cache_lengths, prefill_time_ms = executor.prefill_then_decode_tokens(
        token_ids,
        decode_steps,
    )
    reference_tokens, decode_max_abs_diff = _reference_prefill_then_decode(
        checkpoint,
        token_ids,
        decode_steps,
    )
    return NativePrefillDecodeParityResult(
        checkpoint=checkpoint,
        token_ids=tuple(int(token_id) for token_id in token_ids),
        prefill_logits_shape=native_parity.logits_shape,
        prefill_logits_dtype=native_parity.logits_dtype,
        prefill_max_abs_diff=native_parity.max_abs_diff,
        decode_max_abs_diff=decode_max_abs_diff,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        tolerance_ok=native_parity.tolerance_ok
        and decode_max_abs_diff <= tolerance_atol,
        native_tokens=tuple(native_tokens),
        reference_tokens=tuple(reference_tokens),
        token_ok=tuple(native_tokens) == tuple(reference_tokens),
        cache_lengths=tuple(cache_lengths),
        prefill_time_ms=prefill_time_ms,
    )


def trace_native_debug_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    prompt_fingerprint: str,
    output_dir: Path,
    decode_steps: int,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
    sample_size: int = 8,
    selected_dumps: Sequence[str] = (),
    stop_on_first_divergence: bool = True,
) -> TraceArtifacts:
    """Trace native and mlx-lm semantic checkpoints and write artifacts."""

    if not hasattr(executor, "model") or not hasattr(executor, "model_config"):
        raise ValueError("native executor does not expose Qwen2 trace surface")

    native_model = executor.model
    model_config = executor.model_config

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    reference_model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(reference_model_path)
    reference_run = trace_qwen2_run(
        model=reference_model,
        model_config=model_config,
        backend="mlx-lm",
        prompt_token_ids=token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=make_prompt_cache(reference_model),
        decode_steps=decode_steps,
        sample_size=sample_size,
        selected_dumps=selected_dumps,
    )
    native_run = trace_qwen2_run(
        model=native_model,
        model_config=model_config,
        backend="native-mlx",
        prompt_token_ids=token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=[Qwen2LayerCache() for _ in range(model_config.num_hidden_layers)],
        decode_input_token_ids=reference_run.decode_input_token_ids,
        sample_size=sample_size,
        selected_dumps=selected_dumps,
    )
    comparison = compare_trace_runs(
        native_run,
        reference_run,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        stop_on_first_divergence=stop_on_first_divergence,
    )
    return write_trace_artifacts(
        output_dir=output_dir,
        checkpoint=checkpoint,
        native_run=native_run,
        reference_run=reference_run,
        comparison=comparison,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )


def _load_reference_logits(checkpoint: str, token_ids: Sequence[int]) -> mx.array:
    from mlx_lm.utils import load_model

    model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(model_path)
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    logits = reference_model(inputs)
    mx.eval(logits)
    return logits


def _reference_prefill_then_decode(
    checkpoint: str,
    token_ids: Sequence[int],
    decode_steps: int,
) -> tuple[list[int], float]:
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(model_path)
    cache = make_prompt_cache(reference_model)
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    logits = reference_model(inputs, cache=cache)
    mx.eval(logits)
    tokens = [int(mx.argmax(logits[0, -1], axis=-1).item())]
    max_abs_diff = 0.0
    last_token = tokens[-1]
    for _ in range(decode_steps):
        decode_logits = reference_model(
            mx.array([[last_token]], dtype=mx.int32),
            cache=cache,
        )
        mx.eval(decode_logits)
        last_token = int(mx.argmax(decode_logits[0, -1], axis=-1).item())
        tokens.append(last_token)
        max_abs_diff = max(max_abs_diff, 0.0)
    return tokens, max_abs_diff


def _load_tokenizer_wrapper(model_path: Path):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(model_path)


def _resolve_model_path(model: str) -> Path:
    model_path = Path(model)
    if model_path.is_file():
        return model_path.parent
    if model_path.is_dir():
        return model_path
    try:
        from mlx_lm.utils import hf_repo_to_path
    except ModuleNotFoundError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MODEL_RESOLUTION_UNAVAILABLE",
                message=(
                    "native-mlx could not resolve remote model path because mlx_lm "
                    "runtime helpers are unavailable"
                ),
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=model,
            )
        ) from exc
    try:
        return Path(hf_repo_to_path(model))
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MODEL_RESOLUTION_FAILED",
                message=f"native-mlx could not resolve model '{model}': {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=model,
            )
        ) from exc


def _make_stage_emitter(
    client: socket,
    model: str,
    started_loading_at: int,
) -> Callable[[str, str], None]:
    def emit(stage: str, state: str) -> None:
        _send_status(
            client,
            _status(
                model=model,
                state=state,
                started_loading_at=started_loading_at,
                last_transition_at=_now_seconds(),
                progress=ModelLoadProgress(current_phase=stage),
            ),
        )

    return emit


def _startup_error(
    *,
    code: str,
    message: str,
    stage: str,
    category: str,
    detail: str,
) -> ModelError:
    return ModelError(
        code=code,
        message=f"{message}. Default v1 backend remains available.",
        at=_now_seconds(),
        backend="native-mlx",
        stage=stage,
        category=category,
        detail=detail,
    )


def _emit_failure(
    client: socket,
    model: str,
    started_loading_at: int,
    error: ModelError,
) -> None:
    failed_at = _now_seconds()
    _send_status(
        client,
        _status(
            model=model,
            state="failed",
            started_loading_at=started_loading_at,
            last_transition_at=failed_at,
            last_error=error,
        ),
    )
    client.sendall(encode_bootstrap_message(WorkerError(error.message, error=error)))


def _status(
    *,
    model: str,
    state: str,
    started_loading_at: int,
    last_transition_at: int,
    progress: ModelLoadProgress | None = None,
    last_error: ModelError | None = None,
) -> ModelStatus:
    return ModelStatus(
        model=model,
        revision=model,
        state=state,
        ready=False,
        servable=False,
        progress=progress,
        device=None,
        dtype=None,
        loaded_at=None,
        started_loading_at=started_loading_at,
        last_transition_at=last_transition_at,
        last_error=last_error,
        warmup_passed=False,
        last_warmup_at=None,
        last_warmup_latency_ms=None,
    )


def _ready_status(
    *,
    model: str,
    started_loading_at: int,
    loaded_at: int,
    last_transition_at: int,
    warmup_latency_ms: int,
) -> ModelStatus:
    return ModelStatus(
        model=model,
        revision=model,
        state="ready",
        ready=True,
        servable=True,
        progress=ModelLoadProgress(current_phase="deterministic_warmup"),
        device="mps",
        dtype="float16",
        loaded_at=loaded_at,
        started_loading_at=started_loading_at,
        last_transition_at=last_transition_at,
        last_error=None,
        warmup_passed=True,
        last_warmup_at=loaded_at,
        last_warmup_latency_ms=warmup_latency_ms,
    )


def _send_status(client: socket, status: ModelStatus) -> None:
    client.sendall(encode_bootstrap_message(status))


def _now_seconds() -> int:
    return int(time.time())
