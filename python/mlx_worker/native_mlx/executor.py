"""Architecture-independent native MLX generation executor."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from .cache import (
    DenseBatchLayerCache,
    DenseRequestCache,
    KVCacheBackend,
)
from .interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    NativeBackendOptions,
    NativeModel,
    StepRequestResult,
    StepResult,
)


@dataclass(frozen=True)
class _BatchLayout:
    requests: tuple[ExecutionRequest, ...]
    request_caches: tuple[DenseRequestCache, ...]
    token_lengths: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    input_ids: mx.array
    positions: mx.array
    attention_mask: mx.array


@dataclass
class MlxGenerationExecutor:
    """Shared dense causal-model executor.

    Schedulers select token work. This module validates and tensorizes that
    work, performs one model invocation, commits model-side cache state, and
    returns sampled token IDs.
    """

    architecture_class: str
    model: NativeModel
    cache_backend: KVCacheBackend

    def load(self, options: NativeBackendOptions) -> None:
        if options.architecture_class != self.architecture_class:
            raise ValueError("executor architecture does not match backend options")

    def create_cache(self, request_id: str) -> str:
        return self.cache_backend.create(request_id)

    def cache_len(self, cache_handle: str | None) -> int:
        return self.cache_backend.length(cache_handle)

    def release(self, cache_handle: str | None) -> None:
        self.cache_backend.release(cache_handle)

    def prefill_batch(self, batch: ExecutionBatch) -> StepResult:
        return self._execute(batch, require_decode=False)

    def decode_batch(self, batch: ExecutionBatch) -> StepResult:
        return self._execute(batch, require_decode=True)

    def _execute(self, batch: ExecutionBatch, *, require_decode: bool) -> StepResult:
        started = time.perf_counter()
        layout = self._prepare(batch, require_decode=require_decode)
        layer_caches = self.cache_backend.batch_layers(
            layout.request_caches,
            layout.token_lengths,
        )
        forward_batch = ForwardBatch(
            token_lengths=layout.token_lengths,
            cache_lengths=layout.cache_lengths,
            attention_mask=layout.attention_mask,
            layer_caches=tuple(layer_caches),
        )
        logits = self.model(layout.input_ids, layout.positions, forward_batch)
        next_tokens = self._sample_last_logits(logits, layout.token_lengths)
        mx.eval(logits, next_tokens)
        self._commit(
            layout.request_caches,
            layer_caches,
            layout.cache_lengths,
            layout.token_lengths,
        )
        token_ids = [int(value) for value in next_tokens.tolist()]
        results = tuple(
            StepRequestResult(
                request_id=request.request_id,
                token_ids=request.token_ids,
                cache_handle=request.cache_handle,
                cache_length=cache.size(),
                next_token_id=token_id,
            )
            for request, cache, token_id in zip(
                layout.requests,
                layout.request_caches,
                token_ids,
                strict=True,
            )
        )
        return StepResult(
            phase=batch.phase,
            results=results,
            step_time_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )

    def _prepare(
        self,
        batch: ExecutionBatch,
        *,
        require_decode: bool,
    ) -> _BatchLayout:
        if not batch.requests:
            raise ValueError("execution batch must contain at least one request")
        caches: list[DenseRequestCache] = []
        token_lengths: list[int] = []
        cache_lengths: list[int] = []
        token_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        max_tokens = max(len(request.token_ids) for request in batch.requests)

        for request in batch.requests:
            token_count = len(request.token_ids)
            if require_decode and token_count != 1:
                raise ValueError("decode requires exactly one token per request")
            if not require_decode and token_count == 0:
                raise ValueError("prefill requires at least one token per request")
            cache = self.cache_backend.get(request.cache_handle, request.request_id)
            cache_length = cache.size()
            if require_decode and cache_length == 0:
                raise ValueError("decode requires existing prefill state")
            expected_positions = tuple(range(cache_length, cache_length + token_count))
            if request.positions != expected_positions:
                raise ValueError("positions do not match cache length")
            caches.append(cache)
            token_lengths.append(token_count)
            cache_lengths.append(cache_length)
            token_rows.append(
                list(request.token_ids) + [0] * (max_tokens - token_count)
            )
            position_rows.append(
                list(request.positions)
                + [request.positions[-1]] * (max_tokens - token_count)
            )

        return _BatchLayout(
            requests=batch.requests,
            request_caches=tuple(caches),
            token_lengths=tuple(token_lengths),
            cache_lengths=tuple(cache_lengths),
            input_ids=mx.array(token_rows, dtype=mx.int32),
            positions=mx.array(position_rows, dtype=mx.int32),
            attention_mask=_causal_attention_mask(cache_lengths, token_lengths),
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

    @staticmethod
    def _commit(
        request_caches: tuple[DenseRequestCache, ...],
        layer_caches: list[DenseBatchLayerCache],
        old_lengths: tuple[int, ...],
        append_lengths: tuple[int, ...],
    ) -> None:
        for layer_cache in layer_caches:
            layer_cache.validate_commit()
        for layer_cache in layer_caches:
            layer_cache.commit()
        expected = tuple(
            old + append
            for old, append in zip(old_lengths, append_lengths, strict=True)
        )
        for cache, length in zip(request_caches, expected, strict=True):
            if cache.size() != length:
                raise ValueError("cache layer lengths diverged after commit")


def _causal_attention_mask(
    cache_lengths: list[int],
    token_lengths: list[int],
) -> mx.array:
    max_tokens = max(token_lengths)
    max_total = max(
        cache_length + token_length
        for cache_length, token_length in zip(cache_lengths, token_lengths, strict=True)
    )
    rows: list[list[list[list[float]]]] = []
    for cache_length, token_length in zip(cache_lengths, token_lengths, strict=True):
        query_rows: list[list[float]] = []
        for query_index in range(max_tokens):
            max_key = cache_length + query_index
            query_rows.append(
                [
                    0.0
                    if query_index < token_length
                    and key_index < cache_length + token_length
                    and key_index <= max_key
                    else -1e9
                    for key_index in range(max_total)
                ]
            )
        rows.append([query_rows])
    return mx.array(rows, dtype=mx.float32)
