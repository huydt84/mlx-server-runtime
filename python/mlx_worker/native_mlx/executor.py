"""Architecture-independent native MLX generation executor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .attention import AttentionBackend
from .cache import KVCacheBackend, RequestCache
from .interfaces import (
    BatchExecutionError,
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    ForwardMode,
    NativeBackendOptions,
    NativeModel,
    StepRequestResult,
    StepResult,
)


@dataclass(frozen=True)
class _BatchLayout:
    requests: tuple[ExecutionRequest, ...]
    source_indices: tuple[int, ...]
    request_caches: tuple[RequestCache, ...]
    token_lengths: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    input_ids: mx.array
    positions: mx.array
    forward_mode: ForwardMode


@dataclass
class PreparedStep:
    """CPU-prepared step metadata that is safe to build ahead of dispatch."""

    batch: ExecutionBatch
    layout: _BatchLayout | None
    ordered_results: list[StepRequestResult | None]
    started: float
    prepare_ms: int

    @property
    def has_work(self) -> bool:
        """Return whether model work remains after request-local preflight."""

        return self.layout is not None


@dataclass
class InFlightStep:
    """MLX graph awaiting one explicit result-resolution boundary."""

    prepared: PreparedStep
    reservation: Any
    logits: mx.array
    next_tokens: mx.array
    reserve_ms: int
    forward_ms: int
    sample_ms: int
    dispatch_ms: int
    dispatch_started_ns: int
    forward_mode: ForwardMode


@dataclass
class MlxGenerationExecutor:
    """Shared causal-model executor with no request-cache lifecycle methods."""

    architecture_class: str
    model: NativeModel
    cache_backend: KVCacheBackend
    attention_backend: AttentionBackend
    execution_mode: str = field(init=False, default="serial")

    def load(self, options: NativeBackendOptions) -> None:
        if options.architecture_class != self.architecture_class:
            raise ValueError("executor architecture does not match backend options")

    def execute_batch(self, batch: ExecutionBatch) -> StepResult:
        """Execute all valid scheduler-selected work in one model invocation."""

        return self.execute_prepared(self.prepare_batch(batch))

    def prepare_batch(self, batch: ExecutionBatch) -> PreparedStep:
        """Build request-independent tensors and cache metadata on the CPU."""

        started = time.perf_counter()
        layout, ordered_results = self._prepare(batch)
        return PreparedStep(
            batch=batch,
            layout=layout,
            ordered_results=ordered_results,
            started=started,
            prepare_ms=_elapsed_ms(started),
        )

    def execute_prepared(self, prepared: PreparedStep) -> StepResult:
        """Complete one already-prepared step synchronously."""

        if not prepared.has_work:
            return self._empty_result(prepared)
        return self.resolve_batch(self.dispatch_batch(prepared))

    def dispatch_batch(self, prepared: PreparedStep) -> InFlightStep:
        """Submit one MLX graph without reading device results."""

        layout = prepared.layout
        if layout is None:
            raise BatchExecutionError("EMPTY_EXECUTION_STEP", "nothing to dispatch")
        reserve_started = time.perf_counter()
        try:
            reservation = self.cache_backend.reserve_batch(
                layout.request_caches,
                layout.token_lengths,
            )
        except Exception as exc:
            raise BatchExecutionError("KV_RESERVATION_FAILED", str(exc)) from exc
        reserve_ms = _elapsed_ms(reserve_started)
        forward_batch = ForwardBatch(
            forward_mode=layout.forward_mode,
            token_lengths=layout.token_lengths,
            cache_lengths=layout.cache_lengths,
            attention_mask="causal",
            layer_attention=self.attention_backend.contexts(
                reservation,
                layout.forward_mode,
            ),
        )
        try:
            reset_graph_profile = getattr(self.model, "reset_graph_profile", None)
            if callable(reset_graph_profile):
                reset_graph_profile()
            forward_started = time.perf_counter()
            logits = self.model(layout.input_ids, layout.positions, forward_batch)
            forward_ms = _elapsed_ms(forward_started)
            sample_started = time.perf_counter()
            next_tokens = self._sample_last_logits(logits, layout.token_lengths)
            sample_ms = _elapsed_ms(sample_started)
        except Exception as exc:
            reservation.abort()
            raise BatchExecutionError("MODEL_EXECUTION_FAILED", str(exc)) from exc

        return InFlightStep(
            prepared=prepared,
            reservation=reservation,
            logits=logits,
            next_tokens=next_tokens,
            reserve_ms=reserve_ms,
            forward_ms=forward_ms,
            sample_ms=sample_ms,
            dispatch_ms=0,
            dispatch_started_ns=0,
            forward_mode=layout.forward_mode,
        )

    def resolve_batch(self, in_flight: InFlightStep) -> StepResult:
        """Resolve device outputs and commit cache state exactly once."""

        prepared = in_flight.prepared
        layout = prepared.layout
        assert layout is not None
        eval_started = time.perf_counter()
        eval_started_ns = time.perf_counter_ns()
        try:
            mx.eval(in_flight.logits, in_flight.next_tokens)
            token_ids = [int(value) for value in in_flight.next_tokens.tolist()]
        except Exception as exc:
            in_flight.reservation.abort()
            raise BatchExecutionError("MODEL_EXECUTION_FAILED", str(exc)) from exc
        eval_ms = _elapsed_ms(eval_started)
        eval_completed_ns = time.perf_counter_ns()
        try:
            commit_started = time.perf_counter()
            committed_lengths = in_flight.reservation.commit()
            commit_ms = _elapsed_ms(commit_started)
        except Exception as exc:
            in_flight.reservation.abort()
            raise BatchExecutionError("CACHE_COMMIT_FAILED", str(exc)) from exc
        graph_profile_metrics = _graph_profile_metrics(self.model)
        for source_index, request, cache_length, token_id in zip(
            layout.source_indices,
            layout.requests,
            committed_lengths,
            token_ids,
            strict=True,
        ):
            prepared.ordered_results[source_index] = StepRequestResult(
                request_id=request.request_id,
                phase=request.phase,
                token_ids=request.token_ids,
                cache_handle=request.cache_handle,
                cache_length=cache_length,
                next_token_id=token_id,
            )
        return StepResult(
            forward_mode=layout.forward_mode,
            results=tuple(_require_result(item) for item in prepared.ordered_results),
            step_time_ms=max(1, int((time.perf_counter() - prepared.started) * 1000)),
            physical_batch_size=len(layout.requests),
            model_forward_count=1,
            metrics=self._metrics(
                executor_prepare_ms=prepared.prepare_ms,
                executor_reserve_ms=in_flight.reserve_ms,
                executor_forward_ms=in_flight.forward_ms,
                executor_sample_ms=in_flight.sample_ms,
                executor_dispatch_ms=in_flight.dispatch_ms,
                executor_dispatch_started_ns=in_flight.dispatch_started_ns,
                executor_eval_started_ns=eval_started_ns,
                executor_eval_completed_ns=eval_completed_ns,
                executor_eval_ms=eval_ms,
                executor_commit_ms=commit_ms,
                **graph_profile_metrics,
            ),
        )

    def discard_batch(self, in_flight: InFlightStep) -> None:
        """Synchronize outstanding work, then release its uncommitted reservation."""

        try:
            mx.eval(in_flight.logits, in_flight.next_tokens)
        finally:
            in_flight.reservation.abort()

    def _empty_result(self, prepared: PreparedStep) -> StepResult:
        """Return request-local preflight errors without entering MLX."""

        return StepResult(
            forward_mode=prepared.batch.forward_mode,
            results=tuple(_require_result(item) for item in prepared.ordered_results),
            step_time_ms=max(1, int((time.perf_counter() - prepared.started) * 1000)),
            physical_batch_size=0,
            model_forward_count=0,
            metrics=self._metrics(
                executor_prepare_ms=prepared.prepare_ms,
                executor_reserve_ms=0,
                executor_forward_ms=0,
                executor_sample_ms=0,
                executor_dispatch_ms=0,
                executor_eval_ms=0,
                executor_commit_ms=0,
            ),
        )

    def _metrics(self, **execution_metrics: int) -> dict[str, object]:
        """Compose cache, attention, and executor metrics with one base mapping."""

        metrics = self.cache_backend.metrics()
        self.attention_backend.add_metrics(metrics)
        metrics.update(execution_metrics)
        return metrics

    def _prepare(
        self, batch: ExecutionBatch
    ) -> tuple[_BatchLayout | None, list[StepRequestResult | None]]:
        if not batch.requests:
            raise BatchExecutionError(
                "INVALID_EXECUTION_BATCH",
                "execution batch must contain at least one request",
            )
        request_ids = [request.request_id for request in batch.requests]
        if len(request_ids) != len(set(request_ids)):
            raise BatchExecutionError(
                "INVALID_EXECUTION_BATCH",
                "execution batch contains duplicate request IDs",
            )
        caches: list[RequestCache] = []
        valid_requests: list[ExecutionRequest] = []
        source_indices: list[int] = []
        token_lengths: list[int] = []
        cache_lengths: list[int] = []
        ordered_results: list[StepRequestResult | None] = [None] * len(batch.requests)

        for index, request in enumerate(batch.requests):
            cache, error = self._preflight_cache(request)
            if error is not None:
                ordered_results[index] = self._request_error(
                    request,
                    "INVALID_EXECUTION_REQUEST",
                    error,
                )
                continue
            assert cache is not None
            valid_requests.append(request)
            source_indices.append(index)
            caches.append(cache)
            token_lengths.append(len(request.token_ids))
            cache_lengths.append(cache.size())

        if valid_requests:
            capacity_errors = self.cache_backend.preflight(
                tuple(caches),
                tuple(token_lengths),
            )
            kept_requests: list[ExecutionRequest] = []
            kept_indices: list[int] = []
            kept_caches: list[RequestCache] = []
            kept_token_lengths: list[int] = []
            kept_cache_lengths: list[int] = []
            for request, source_index, cache, token_length, cache_length, error in zip(
                valid_requests,
                source_indices,
                caches,
                token_lengths,
                cache_lengths,
                capacity_errors,
                strict=True,
            ):
                if error is not None:
                    ordered_results[source_index] = self._request_error(
                        request,
                        "KV_CAPACITY_EXHAUSTED",
                        error,
                    )
                    continue
                kept_requests.append(request)
                kept_indices.append(source_index)
                kept_caches.append(cache)
                kept_token_lengths.append(token_length)
                kept_cache_lengths.append(cache_length)
            valid_requests = kept_requests
            source_indices = kept_indices
            caches = kept_caches
            token_lengths = kept_token_lengths
            cache_lengths = kept_cache_lengths

        if not valid_requests:
            return None, ordered_results

        max_tokens = max(token_lengths)
        token_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        for request in valid_requests:
            token_count = len(request.token_ids)
            token_rows.append(
                list(request.token_ids) + [0] * (max_tokens - token_count)
            )
            position_rows.append(
                list(request.positions)
                + [request.positions[-1]] * (max_tokens - token_count)
            )

        return (
            _BatchLayout(
                requests=tuple(valid_requests),
                source_indices=tuple(source_indices),
                request_caches=tuple(caches),
                token_lengths=tuple(token_lengths),
                cache_lengths=tuple(cache_lengths),
                input_ids=mx.array(token_rows, dtype=mx.int32),
                positions=mx.array(position_rows, dtype=mx.int32),
                forward_mode=ForwardMode.from_phases(
                    tuple(request.phase for request in valid_requests)
                ),
            ),
            ordered_results,
        )

    def _preflight_cache(
        self,
        request: ExecutionRequest,
    ) -> tuple[RequestCache | None, str | None]:
        token_count = len(request.token_ids)
        if request.phase == "decode" and token_count != 1:
            return None, "decode requires exactly one token"
        if request.phase == "prefill" and token_count == 0:
            return None, "prefill requires at least one token"
        if request.phase not in ("prefill", "decode"):
            return None, "unsupported execution phase"
        try:
            cache = self.cache_backend.get(
                request.cache_handle,
                request.request_id,
            )
        except ValueError as exc:
            return None, str(exc)
        cache_length = cache.size()
        if request.phase == "decode" and cache_length == 0:
            return None, "decode requires existing prefill state"
        expected_positions = tuple(range(cache_length, cache_length + token_count))
        if request.positions != expected_positions:
            return None, "positions do not match cache length"
        return cache, None

    def _request_error(
        self,
        request: ExecutionRequest,
        code: str,
        message: str,
    ) -> StepRequestResult:
        return StepRequestResult(
            request_id=request.request_id,
            phase=request.phase,
            token_ids=request.token_ids,
            cache_handle=request.cache_handle,
            cache_length=self.cache_backend.length(request.cache_handle),
            error_code=code,
            error_message=message,
        )

    @staticmethod
    def _sample_last_logits(
        logits: mx.array,
        token_lengths: tuple[int, ...],
    ) -> mx.array:
        rows = mx.stack(
            [logits[index, length - 1, :] for index, length in enumerate(token_lengths)]
        )
        return mx.argmax(rows, axis=-1)


@dataclass
class MlxOverlapGenerationExecutor(MlxGenerationExecutor):
    """Same-thread MLX executor that dispatches one bounded asynchronous step."""

    execution_mode: str = field(init=False, default="overlap")
    _stream: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not mx.metal.is_available():
            raise ValueError("native overlap execution requires MLX Metal")
        # Keep model, cache, and async evaluation on the owner thread's default
        # GPU stream.  A separate stream adds a command-queue handoff without
        # enabling additional dependency overlap for this one-model pipeline.
        self._stream = mx.default_stream(mx.gpu)

    def dispatch_batch(self, prepared: PreparedStep) -> InFlightStep:
        """Build and enqueue the graph on an owner-thread MLX stream."""

        with mx.stream(self._stream):
            in_flight = super().dispatch_batch(prepared)
            started = time.perf_counter()
            in_flight.dispatch_started_ns = time.perf_counter_ns()
            mx.async_eval(in_flight.logits, in_flight.next_tokens)
            in_flight.dispatch_ms = _elapsed_ms(started)
            return in_flight

    def execute_prepared_serial(self, prepared: PreparedStep) -> StepResult:
        """Run latency-sensitive pure prefill without an async handoff."""

        if not prepared.has_work:
            return self._empty_result(prepared)
        in_flight = MlxGenerationExecutor.dispatch_batch(self, prepared)
        return MlxGenerationExecutor.resolve_batch(self, in_flight)

    def resolve_batch(self, in_flight: InFlightStep) -> StepResult:
        """Synchronize only the previously dispatched owner-thread step."""

        with mx.stream(self._stream):
            return super().resolve_batch(in_flight)

    def discard_batch(self, in_flight: InFlightStep) -> None:
        """Synchronize before releasing an outstanding cache reservation."""

        with mx.stream(self._stream):
            super().discard_batch(in_flight)


def _require_result(result: StepRequestResult | None) -> StepRequestResult:
    if result is None:
        raise BatchExecutionError(
            "INCOMPLETE_EXECUTION_RESULT",
            "executor did not produce a result for every request",
        )
    return result


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _graph_profile_metrics(model: NativeModel) -> dict[str, int]:
    read_metrics = getattr(model, "graph_profile_metrics", None)
    if not callable(read_metrics):
        return {}
    metrics = read_metrics()
    if not isinstance(metrics, dict):
        return {}
    normalized = {
        str(name): int(value)
        for name, value in metrics.items()
        if str(name).startswith("model_graph_")
    }
    projection_total = normalized.get("model_graph_projection_ms", 0) + sum(
        normalized.get(name, 0)
        for name in (
            "model_graph_q_proj_ms",
            "model_graph_k_proj_ms",
            "model_graph_v_proj_ms",
            "model_graph_o_proj_ms",
            "model_graph_lm_head_ms",
        )
    )
    if projection_total:
        normalized["model_graph_projection_total_ms"] = projection_total
    mlp_total = normalized.get("model_graph_mlp_ms", 0) + sum(
        normalized.get(name, 0)
        for name in (
            "model_graph_mlp_gate_ms",
            "model_graph_mlp_up_ms",
            "model_graph_mlp_activation_ms",
            "model_graph_mlp_down_ms",
        )
    )
    if mlp_total:
        normalized["model_graph_mlp_total_ms"] = mlp_total
    return normalized
