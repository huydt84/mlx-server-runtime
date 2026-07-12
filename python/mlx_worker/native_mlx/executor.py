"""Architecture-independent native MLX generation executor."""

from __future__ import annotations

import time
from dataclasses import dataclass

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
class MlxGenerationExecutor:
    """Shared causal-model executor with no request-cache lifecycle methods."""

    architecture_class: str
    model: NativeModel
    cache_backend: KVCacheBackend
    attention_backend: AttentionBackend

    def load(self, options: NativeBackendOptions) -> None:
        if options.architecture_class != self.architecture_class:
            raise ValueError("executor architecture does not match backend options")

    def execute_batch(self, batch: ExecutionBatch) -> StepResult:
        """Execute all valid scheduler-selected work in one model invocation."""

        started = time.perf_counter()
        prepare_started = started
        layout, ordered_results = self._prepare(batch)
        prepare_ms = _elapsed_ms(prepare_started)
        if layout is None:
            return StepResult(
                forward_mode=batch.forward_mode,
                results=tuple(_require_result(item) for item in ordered_results),
                step_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
                physical_batch_size=0,
                model_forward_count=0,
                metrics=self._metrics(
                    executor_prepare_ms=prepare_ms,
                    executor_reserve_ms=0,
                    executor_forward_ms=0,
                    executor_sample_ms=0,
                    executor_eval_ms=0,
                    executor_commit_ms=0,
                ),
            )
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
            eval_started = time.perf_counter()
            mx.eval(logits, next_tokens)
            eval_ms = _elapsed_ms(eval_started)
            token_ids = [int(value) for value in next_tokens.tolist()]
        except Exception as exc:
            reservation.abort()
            raise BatchExecutionError("MODEL_EXECUTION_FAILED", str(exc)) from exc
        try:
            commit_started = time.perf_counter()
            committed_lengths = reservation.commit()
            commit_ms = _elapsed_ms(commit_started)
        except Exception as exc:
            reservation.abort()
            raise BatchExecutionError("CACHE_COMMIT_FAILED", str(exc)) from exc
        graph_profile_metrics = _graph_profile_metrics(self.model)
        for source_index, request, cache_length, token_id in zip(
            layout.source_indices,
            layout.requests,
            committed_lengths,
            token_ids,
            strict=True,
        ):
            ordered_results[source_index] = StepRequestResult(
                request_id=request.request_id,
                phase=request.phase,
                token_ids=request.token_ids,
                cache_handle=request.cache_handle,
                cache_length=cache_length,
                next_token_id=token_id,
            )
        return StepResult(
            forward_mode=layout.forward_mode,
            results=tuple(_require_result(item) for item in ordered_results),
            step_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
            physical_batch_size=len(layout.requests),
            model_forward_count=1,
            metrics=self._metrics(
                executor_prepare_ms=prepare_ms,
                executor_reserve_ms=reserve_ms,
                executor_forward_ms=forward_ms,
                executor_sample_ms=sample_ms,
                executor_eval_ms=eval_ms,
                executor_commit_ms=commit_ms,
                **graph_profile_metrics,
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
