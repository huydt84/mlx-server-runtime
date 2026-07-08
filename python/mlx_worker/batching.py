"""Batch-aware completion backend for the MLX worker."""

from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import suppress
import copy
from dataclasses import dataclass, field
import importlib
import importlib.metadata
import inspect
import time
from typing import Any, Callable, Literal, Protocol, Sequence

from .ipc import ChatCompletionRequest, ChatCompletionResponse


CacheMatchKind = Literal["exact", "shorter_prefix", "trimmed_longer_prefix"]


@dataclass(frozen=True)
class CachedPrompt:
    """A cached prompt prefix and its MLX prompt cache snapshot."""

    tokens: tuple[int, ...]
    prompt_cache: list[Any]
    cache_bytes: int
    match_kind: CacheMatchKind


@dataclass
class PromptCacheStoreStats:
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    shorter_prefix_hits: int = 0
    trimmed_longer_prefix_hits: int = 0
    entries: int = 0
    evictions: int = 0
    bytes: int = 0


class CacheBudgetManager:
    """Apply one stored-cache budget across several cache stores."""

    def __init__(self, total_budget_bytes: int) -> None:
        self._total_budget_bytes = max(0, total_budget_bytes)
        self._stores: list["PromptCacheStore"] = []
        self._active_batch_bytes: dict[str, int] = {}

    def register(self, store: "PromptCacheStore") -> None:
        if store not in self._stores:
            self._stores.append(store)

    def set_active_batch_bytes(self, backend: str, value: int) -> None:
        self._active_batch_bytes[backend] = max(0, int(value))
        self.enforce()

    @property
    def active_batch_bytes(self) -> int:
        return sum(self._active_batch_bytes.values())

    @property
    def stored_cache_target_bytes(self) -> int:
        return max(0, self._total_budget_bytes - self.active_batch_bytes)

    @property
    def stored_cache_bytes(self) -> int:
        return sum(store.total_bytes for store in self._stores)

    def enforce(self) -> None:
        target = self.stored_cache_target_bytes
        while self.stored_cache_bytes > target:
            evicted = False
            for store in self._stores:
                if store.evict_oldest():
                    evicted = True
                    if self.stored_cache_bytes <= target:
                        return
            if not evicted:
                return


class PromptCacheStore:
    """Store prompt caches for recent prompt prefixes."""

    def __init__(
        self,
        *,
        name: str = "prompt_cache",
        max_entries: int = 32,
        max_bytes: int = 8 * 1024 * 1024,
        trim_cache: Callable[[Sequence[Any], int], list[Any] | None] | None = None,
        budget_manager: CacheBudgetManager | None = None,
    ) -> None:
        self._name = name
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._trim_cache = trim_cache or _trim_mlx_prompt_cache
        self._entries: OrderedDict[
            tuple[tuple[Any, ...], tuple[int, ...]], CachedPrompt
        ] = OrderedDict()
        self._total_bytes = 0
        self._stats = PromptCacheStoreStats()
        self._budget_manager = budget_manager
        if self._budget_manager is not None:
            self._budget_manager.register(self)

    def lookup(
        self,
        prompt_tokens: Sequence[int],
        *,
        cache_key: Sequence[Any] | None = None,
    ) -> CachedPrompt | None:
        """Return the longest cached prefix for the prompt, if any."""

        namespace = tuple(cache_key or ())
        prompt_key = tuple(prompt_tokens)
        best: CachedPrompt | None = None
        for (cached_namespace, cached_tokens), cached_prompt in reversed(
            self._entries.items()
        ):
            if cached_namespace != namespace:
                continue
            cached_key = cached_tokens
            common_prefix = 0
            for cached_token, prompt_token in zip(cached_key, prompt_key):
                if cached_token != prompt_token:
                    break
                common_prefix += 1
            if cached_key == prompt_key:
                candidate = CachedPrompt(
                    tokens=prompt_key,
                    prompt_cache=copy.deepcopy(cached_prompt.prompt_cache),
                    cache_bytes=cached_prompt.cache_bytes,
                    match_kind="exact",
                )
            elif len(cached_key) < len(prompt_key):
                if common_prefix == len(cached_key):
                    candidate = CachedPrompt(
                        tokens=cached_key,
                        prompt_cache=copy.deepcopy(cached_prompt.prompt_cache),
                        cache_bytes=cached_prompt.cache_bytes,
                        match_kind="shorter_prefix",
                    )
                else:
                    reusable_length = min(common_prefix, len(prompt_key) - 1)
                    if reusable_length <= 0:
                        continue
                    trimmed_cache = self._trim_cache(
                        cached_prompt.prompt_cache,
                        len(cached_key) - reusable_length,
                    )
                    if trimmed_cache is None:
                        continue
                    candidate = CachedPrompt(
                        tokens=prompt_key[:reusable_length],
                        prompt_cache=trimmed_cache,
                        cache_bytes=_estimate_cache_bytes(trimmed_cache),
                        match_kind="trimmed_longer_prefix",
                    )
            else:
                reusable_length = min(common_prefix, len(prompt_key) - 1)
                if reusable_length <= 0:
                    continue
                trimmed_cache = self._trim_cache(
                    cached_prompt.prompt_cache,
                    len(cached_key) - reusable_length,
                )
                if trimmed_cache is None:
                    continue
                candidate = CachedPrompt(
                    tokens=prompt_key[:reusable_length],
                    prompt_cache=trimmed_cache,
                    cache_bytes=_estimate_cache_bytes(trimmed_cache),
                    match_kind="trimmed_longer_prefix",
                )
            if best is None or len(candidate.tokens) > len(best.tokens):
                best = candidate
        if best is None:
            self._stats.misses += 1
            return None
        self._stats.hits += 1
        if best.match_kind == "exact":
            self._stats.exact_hits += 1
        elif best.match_kind == "shorter_prefix":
            self._stats.shorter_prefix_hits += 1
        else:
            self._stats.trimmed_longer_prefix_hits += 1
        return best

    def remember(
        self,
        sequence_tokens: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        cache_key: Sequence[Any] | None = None,
    ) -> None:
        """Remember a prompt cache snapshot for later prefix reuse."""

        if not prompt_cache:
            return

        namespace = tuple(cache_key or ())
        key = (namespace, tuple(sequence_tokens))
        cache = CachedPrompt(
            key[1],
            copy.deepcopy(list(prompt_cache)),
            _estimate_cache_bytes(prompt_cache),
            "exact",
        )
        if cache.cache_bytes > self._max_bytes:
            return

        existing = self._entries.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing.cache_bytes

        self._entries[key] = cache
        self._total_bytes += cache.cache_bytes
        self._entries.move_to_end(key)
        self._stats.entries = len(self._entries)
        self._stats.bytes = self._total_bytes
        while (
            len(self._entries) > self._max_entries
            or self._total_bytes > self._max_bytes
        ):
            self.evict_oldest()
        if self._budget_manager is not None:
            self._budget_manager.enforce()
        self._stats.entries = len(self._entries)
        self._stats.bytes = self._total_bytes

    def evict_oldest(self) -> bool:
        if not self._entries:
            return False
        _, evicted = self._entries.popitem(last=False)
        self._total_bytes -= evicted.cache_bytes
        self._stats.evictions += 1
        self._stats.entries = len(self._entries)
        self._stats.bytes = self._total_bytes
        return True

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def stats_snapshot(self) -> PromptCacheStoreStats:
        self._stats.entries = len(self._entries)
        self._stats.bytes = self._total_bytes
        return copy.deepcopy(self._stats)


class BatchCompletionBackend(Protocol):
    """Backend contract for batched chat completions."""

    def complete_many(
        self,
        requests: Sequence[ChatCompletionRequest],
    ) -> list[ChatCompletionResponse]:
        """Complete a batch of chat requests."""


@dataclass(frozen=True)
class BatchBackendContext:
    """Inputs needed to build a batch completion backend."""

    model_id: str
    model: Any
    tokenizer: Any
    prompt_cache_store: PromptCacheStore
    build_prompt_tokens: Callable[[ChatCompletionRequest], list[int]]
    validate_token_limits: Callable[[ChatCompletionRequest, Sequence[int]], None]
    make_sampler: Callable[[float, float], Callable[[Any], Any]]


@dataclass(frozen=True)
class BatchEventSink:
    """Callbacks used by continuous batching to emit worker events."""

    emit_delta: Callable[[str, str], None]
    emit_response: Callable[[ChatCompletionResponse], None]
    emit_error: Callable[[str, str, str], None]


@dataclass
class RequestTiming:
    admitted_at: float = field(default_factory=time.perf_counter)
    decode_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None


def _duration_ms(start: float | None, end: float | None) -> int | None:
    if start is None or end is None or end < start:
        return None
    return max(0, int(round((end - start) * 1000.0)))


class MlxBatchCompletionBackend:
    """Batch completion backend powered by direct `mlx-lm` primitives."""

    def __init__(self, context: BatchBackendContext) -> None:
        self._context = context
        self._batch_generator_cls: type[Any] | None = None

    def complete_many(
        self,
        requests: Sequence[ChatCompletionRequest],
    ) -> list[ChatCompletionResponse]:
        if not requests:
            return []

        batch_generator_cls = self._get_batch_generator_cls()

        prompt_tokens: list[list[int]] = []
        effective_prompts: list[list[int]] = []
        samplers: list[Callable[[Any], Any]] = []
        max_tokens: list[int] = []
        cache_inputs: list[list[Any] | None] = []
        cached_prefixes: list[list[int]] = []
        prompt_cache_hits: list[bool] = []
        cached_token_counts: list[int] = []
        prompt_cache_bytes: list[int] = []

        for request in requests:
            if request.model != self._context.model_id:
                raise ValueError(
                    f"requested model '{request.model}' does not match loaded model '{self._context.model_id}'"
                )

            tokens = self._context.build_prompt_tokens(request)
            self._context.validate_token_limits(request, tokens)
            prompt_tokens.append(tokens)
            max_tokens.append(request.max_tokens)
            samplers.append(
                self._context.make_sampler(request.temperature, request.top_p)
            )

            cached_prompt = self._context.prompt_cache_store.lookup(tokens)
            if cached_prompt is None:
                effective_prompts.append(tokens)
                cache_inputs.append(None)
                cached_prefixes.append([])
                prompt_cache_hits.append(False)
                cached_token_counts.append(0)
                prompt_cache_bytes.append(0)
                continue

            suffix = tokens[len(cached_prompt.tokens) :]
            if not suffix:
                effective_prompts.append(tokens)
                cache_inputs.append(None)
                cached_prefixes.append([])
                prompt_cache_hits.append(False)
                cached_token_counts.append(0)
                prompt_cache_bytes.append(0)
                continue

            effective_prompts.append(suffix)
            cache_inputs.append(cached_prompt.prompt_cache)
            cached_prefixes.append(list(cached_prompt.tokens))
            prompt_cache_hits.append(True)
            cached_token_counts.append(len(cached_prompt.tokens))
            prompt_cache_bytes.append(cached_prompt.cache_bytes)

        generator = batch_generator_cls(
            self._context.model,
            stop_tokens=[[token] for token in self._context.tokenizer.eos_token_ids],
        )
        try:
            uids = generator.insert(
                effective_prompts,
                max_tokens=max_tokens,
                caches=cache_inputs,
                all_tokens=cached_prefixes,
                samplers=samplers,
            )

            if len(uids) != len(requests):
                raise RuntimeError(
                    f"batch generator returned {len(uids)} UIDs for {len(requests)} requests"
                )
            if len(set(uids)) != len(uids):
                raise RuntimeError(f"batch generator returned duplicate UIDs: {uids}")

            uid_to_index = {uid: index for index, uid in enumerate(uids)}
            generated_tokens: dict[int, list[int]] = {uid: [] for uid in uids}
            finish_reasons: dict[int, str] = {}
            prompt_caches: dict[int, tuple[Sequence[Any], Sequence[int] | None]] = {}
            pending_uids: set[int] = set(uids)

            peak_memory_bytes = 0
            with generator.stats() as stats:
                empty_polls = 0
                while pending_uids:
                    responses = generator.next_generated()
                    if not responses:
                        empty_polls += 1
                        if empty_polls >= 3:
                            raise RuntimeError(
                                f"batch generator stalled with pending UIDs: {pending_uids}"
                            )
                        continue
                    empty_polls = 0

                    for response in responses:
                        uid = response.uid
                        if uid not in pending_uids:
                            continue

                        if response.finish_reason is None:
                            generated_tokens[uid].append(response.token)
                            continue

                        if response.finish_reason != "stop":
                            generated_tokens[uid].append(response.token)
                        finish_reasons[uid] = response.finish_reason
                        if response.prompt_cache is not None:
                            prompt_caches[uid] = (
                                response.prompt_cache,
                                getattr(response, "all_tokens", None),
                            )
                        pending_uids.discard(uid)
                peak_memory_bytes = int(getattr(stats, "peak_memory", 0) or 0)
        finally:
            with suppress(Exception):
                generator.close()

        responses: list[ChatCompletionResponse] = []
        for uid in uids:
            index = uid_to_index[uid]
            text = self._context.tokenizer.decode(generated_tokens[uid])
            finish_reason = finish_reasons.get(uid, "stop")
            sequence_tokens = prompt_tokens[index] + generated_tokens[uid]
            response = ChatCompletionResponse(
                request_id=requests[index].request_id,
                model=self._context.model_id,
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=len(prompt_tokens[index]),
                completion_tokens=len(generated_tokens[uid]),
                prompt_cache_hit=prompt_cache_hits[index],
                cached_tokens=cached_token_counts[index] or None,
                prompt_cache_bytes=prompt_cache_bytes[index] or None,
                active_batch_cache_bytes=self._context.prompt_cache_store.total_bytes,
                prompt_batch_size=len(requests),
                decode_batch_size=len(requests),
                configured_prompt_batch_size=len(requests),
                configured_decode_batch_size=len(requests),
                backend="text",
                modality="text",
                apc_mode="none",
                peak_memory_bytes=peak_memory_bytes or None,
                prompt_cache_entries=self._context.prompt_cache_store.total_entries,
                prompt_cache_evictions=self._context.prompt_cache_store.stats_snapshot.evictions,
            )
            responses.append(response)

            completed_cache = prompt_caches.get(uid)
            if completed_cache is not None:
                cached_prompt, cached_sequence = completed_cache
                cache_key = prompt_tokens[index]
                cache_value = _prompt_cache_for_prompt_tokens(
                    cached_prompt,
                    cached_sequence or sequence_tokens,
                    cache_key,
                )
                if cache_value is not None:
                    self._context.prompt_cache_store.remember(cache_key, cache_value)

        return responses

    def _get_batch_generator_cls(self) -> type[Any]:
        if self._batch_generator_cls is not None:
            return self._batch_generator_cls

        batch_generator_cls = _load_batch_generator_cls(continuous=False)
        self._batch_generator_cls = batch_generator_cls
        return batch_generator_cls


def create_default_batch_backend(
    context: BatchBackendContext,
) -> BatchCompletionBackend:
    """Create the default batch completion backend for the worker."""

    return MlxBatchCompletionBackend(context)


@dataclass
class _ScheduledRequest:
    request: ChatCompletionRequest
    stream: bool
    prompt_tokens: list[int]
    cached_prompt: CachedPrompt | None
    prompt_batch_size: int = 0
    decode_batch_size: int = 0
    uid: int | None = None
    generated_tokens: list[int] = None  # type: ignore[assignment]
    rendered_text: str = ""
    cancelled: bool = False
    stage: str = "pending"
    cancelled_stage: str | None = None
    timing: RequestTiming = field(default_factory=RequestTiming)

    def __post_init__(self) -> None:
        if self.generated_tokens is None:
            self.generated_tokens = []


class ContinuousBatchScheduler:
    """Continuously admit text LLM requests into ``mlx_lm.BatchGenerator``."""

    def __init__(
        self,
        context: BatchBackendContext,
        sink: BatchEventSink,
        *,
        prompt_concurrency: int = 4,
        decode_concurrency: int = 4,
        prefill_step_size: int = 256,
        cache_budget_manager: CacheBudgetManager | None = None,
        configured_prompt_batch_size: int | None = None,
        configured_decode_batch_size: int | None = None,
    ) -> None:
        self._context = context
        self._sink = sink
        self._prompt_concurrency = prompt_concurrency
        self._decode_concurrency = decode_concurrency
        self._prefill_step_size = prefill_step_size
        self._cache_budget_manager = cache_budget_manager
        self._configured_prompt_batch_size = (
            configured_prompt_batch_size or prompt_concurrency
        )
        self._configured_decode_batch_size = (
            configured_decode_batch_size or decode_concurrency
        )
        self._batch_generator_cls: type[Any] | None = None
        self._generator: Any | None = None
        self._generator_stats_ctx: Any | None = None
        self._generator_stats: Any | None = None
        self._pending: deque[_ScheduledRequest] = deque()
        self._active: dict[int, _ScheduledRequest] = {}
        self._worker_cancellation_count = 0
        self._worker_error_count = 0
        self._last_tick_latency_ms = 0
        self._arbitration_delay_ms = 0

    def submit(self, request: ChatCompletionRequest, stream: bool) -> None:
        if request.model != self._context.model_id:
            self._worker_error_count += 1
            self._sink.emit_error(
                request.request_id,
                "INVALID_REQUEST",
                f"requested model '{request.model}' does not match loaded model '{self._context.model_id}'",
            )
            return

        try:
            tokens = self._context.build_prompt_tokens(request)
            self._context.validate_token_limits(request, tokens)
        except ValueError as exc:
            self._worker_error_count += 1
            self._sink.emit_error(request.request_id, "INVALID_REQUEST", str(exc))
            return

        self._pending.append(
            _ScheduledRequest(
                request=request,
                stream=stream,
                prompt_tokens=tokens,
                cached_prompt=self._context.prompt_cache_store.lookup(tokens),
            )
        )

    def cancel(self, request_id: str) -> bool:
        for job in list(self._pending):
            if job.request.request_id != request_id:
                continue
            self._pending.remove(job)
            job.cancelled_stage = job.stage
            job.stage = "cancelled"
            self._worker_cancellation_count += 1
            self._emit_cancelled(job)
            return True

        for uid, job in list(self._active.items()):
            if job.request.request_id != request_id:
                continue
            self._active.pop(uid, None)
            job.cancelled_stage = job.stage
            job.stage = "cancelled"
            self._worker_cancellation_count += 1
            self._emit_cancelled(job)
            self._maybe_cancel_generator(uid)
            return True

        return False

    def tick(self) -> None:
        started = time.perf_counter()
        if self._generator is None and not self._pending:
            return

        if self._active and self._generator is not None:
            self._step_generator()
            if self._cache_budget_manager is not None:
                self._cache_budget_manager.set_active_batch_bytes(
                    "text", self._active_batch_cache_bytes()
                )

        self._admit_pending()
        self._last_tick_latency_ms = max(
            0, int(round((time.perf_counter() - started) * 1000.0))
        )

    def set_arbitration_delay_ms(self, value_ms: int) -> None:
        self._arbitration_delay_ms = max(0, value_ms)

    def idle(self) -> bool:
        return not self._pending and not self._active

    def close(self) -> None:
        if self._generator is not None:
            with suppress(Exception):
                self._generator.close()
            if self._generator_stats_ctx is not None:
                with suppress(Exception):
                    self._generator_stats_ctx.__exit__(None, None, None)
            self._generator = None
            self._generator_stats_ctx = None
            self._generator_stats = None
        if self._cache_budget_manager is not None:
            self._cache_budget_manager.set_active_batch_bytes("text", 0)

    def _step_generator(self) -> None:
        responses = self._generator.next_generated()
        if not responses:
            return

        for response in responses:
            job = self._active.get(response.uid)
            if job is None:
                continue
            if job.timing.decode_started_at is None:
                now = time.perf_counter()
                job.timing.decode_started_at = now
                job.stage = "decoding"

            job.decode_batch_size = max(job.decode_batch_size, len(self._active))
            if response.finish_reason is None:
                job.generated_tokens.append(response.token)
                self._emit_delta(job)
                continue

            if response.finish_reason != "stop":
                job.generated_tokens.append(response.token)
                self._emit_delta(job)

            self._finish(
                job,
                response.finish_reason,
                response.prompt_cache,
                getattr(response, "all_tokens", None),
            )
            self._active.pop(response.uid, None)

    def _admit_pending(self) -> None:
        if not self._pending:
            return

        if self._generator is None:
            self._generator = self._make_batch_generator()
            stats_factory = getattr(self._generator, "stats", None)
            if callable(stats_factory):
                self._generator_stats_ctx = stats_factory()
                with suppress(Exception):
                    self._generator_stats = self._generator_stats_ctx.__enter__()

        if len(self._active) >= self._decode_concurrency:
            return

        batch: list[_ScheduledRequest] = []
        while self._pending and len(batch) < self._prompt_concurrency:
            if len(self._active) + len(batch) >= self._decode_concurrency:
                break
            batch.append(self._pending.popleft())

        if not batch:
            return

        prompts: list[list[int]] = []
        max_tokens: list[int] = []
        caches: list[list[Any] | None] = []
        cached_prefixes: list[list[int]] = []
        samplers: list[Callable[[Any], Any]] = []
        for job in batch:
            max_tokens.append(job.request.max_tokens)
            samplers.append(
                self._context.make_sampler(job.request.temperature, job.request.top_p)
            )
            if job.cached_prompt is None:
                prompts.append(job.prompt_tokens)
                caches.append(None)
                cached_prefixes.append([])
                continue

            suffix = job.prompt_tokens[len(job.cached_prompt.tokens) :]
            if not suffix:
                prompts.append(job.prompt_tokens)
                caches.append(None)
                cached_prefixes.append([])
                job.cached_prompt = None
                continue
            prompts.append(suffix)
            caches.append(job.cached_prompt.prompt_cache)
            cached_prefixes.append(list(job.cached_prompt.tokens))

        uids = self._generator.insert(
            prompts,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=cached_prefixes,
            samplers=samplers,
        )
        if len(uids) != len(batch):
            raise RuntimeError(
                f"batch generator returned {len(uids)} UIDs for {len(batch)} requests"
            )
        if len(set(uids)) != len(uids):
            raise RuntimeError(f"batch generator returned duplicate UIDs: {uids}")

        for job, uid in zip(batch, uids, strict=True):
            job.uid = uid
            job.prompt_batch_size = len(batch)
            job.decode_batch_size = len(self._active) + len(batch)
            job.stage = "prompt_processing"
            self._active[uid] = job
        if self._cache_budget_manager is not None:
            self._cache_budget_manager.set_active_batch_bytes(
                "text", self._active_batch_cache_bytes()
            )

    def _make_batch_generator(self) -> Any:
        batch_generator_cls = self._get_batch_generator_cls()
        return batch_generator_cls(
            self._context.model,
            stop_tokens=[[token] for token in self._context.tokenizer.eos_token_ids],
            completion_batch_size=self._decode_concurrency,
            prefill_batch_size=self._prompt_concurrency,
            prefill_step_size=self._prefill_step_size,
        )

    def _get_batch_generator_cls(self) -> type[Any]:
        if self._batch_generator_cls is not None:
            return self._batch_generator_cls

        batch_generator_cls = _load_batch_generator_cls(continuous=True)
        self._batch_generator_cls = batch_generator_cls
        return batch_generator_cls

    def _emit_delta(self, job: _ScheduledRequest) -> None:
        if not job.stream:
            job.rendered_text = self._context.tokenizer.decode(job.generated_tokens)
            return

        decoded = self._context.tokenizer.decode(job.generated_tokens)
        delta = (
            decoded[len(job.rendered_text) :]
            if decoded.startswith(job.rendered_text)
            else decoded
        )
        if delta:
            self._sink.emit_delta(job.request.request_id, delta)
            if job.timing.first_token_at is None:
                job.timing.first_token_at = time.perf_counter()
        job.rendered_text = decoded

    def _finish(
        self,
        job: _ScheduledRequest,
        finish_reason: str,
        prompt_cache: Sequence[Any] | None,
        all_tokens: Sequence[int] | None,
    ) -> None:
        if not job.stream:
            job.rendered_text = self._context.tokenizer.decode(job.generated_tokens)
            if job.generated_tokens and job.timing.first_token_at is None:
                now = time.perf_counter()
                job.timing.first_token_at = now
                if job.timing.decode_started_at is None:
                    job.timing.decode_started_at = now

        if prompt_cache is not None:
            cached_prefix_tokens = (
                list(job.cached_prompt.tokens) if job.cached_prompt is not None else []
            )
            full_prompt_tokens = cached_prefix_tokens + job.prompt_tokens
            sequence_tokens = (
                list(all_tokens) if all_tokens is not None else full_prompt_tokens
            )
            cache_value = _prompt_cache_for_prompt_tokens(
                prompt_cache,
                sequence_tokens,
                full_prompt_tokens,
            )
            if cache_value is not None:
                self._context.prompt_cache_store.remember(
                    full_prompt_tokens, cache_value
                )
        job.stage = "completed"
        job.timing.completed_at = time.perf_counter()

        self._sink.emit_response(
            ChatCompletionResponse(
                request_id=job.request.request_id,
                model=self._context.model_id,
                text=job.rendered_text,
                finish_reason=finish_reason,
                prompt_tokens=len(job.prompt_tokens),
                completion_tokens=len(job.generated_tokens),
                prompt_cache_hit=job.cached_prompt is not None,
                cached_tokens=len(job.cached_prompt.tokens)
                if job.cached_prompt
                else None,
                prompt_cache_bytes=job.cached_prompt.cache_bytes
                if job.cached_prompt
                else None,
                active_batch_cache_bytes=self._active_batch_cache_bytes(),
                prompt_batch_size=job.prompt_batch_size or None,
                decode_batch_size=job.decode_batch_size or None,
                configured_prompt_batch_size=self._configured_prompt_batch_size,
                configured_decode_batch_size=self._configured_decode_batch_size,
                backend="text",
                modality="text",
                apc_mode="none",
                scheduler_stage=job.stage,
                cancellation_stage=job.cancelled_stage,
                queue_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.decode_started_at
                ),
                prefill_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.decode_started_at
                ),
                ttft_ms=_duration_ms(job.timing.admitted_at, job.timing.first_token_at),
                decode_time_ms=_duration_ms(
                    job.timing.decode_started_at, job.timing.completed_at
                ),
                completion_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.completed_at
                ),
                scheduler_tick_latency_ms=self._last_tick_latency_ms,
                arbitration_delay_ms=self._arbitration_delay_ms,
                worker_cancellation_count=self._worker_cancellation_count,
                worker_error_count=self._worker_error_count,
                prompt_cache_entries=self._context.prompt_cache_store.stats_snapshot.entries,
                prompt_cache_evictions=self._context.prompt_cache_store.stats_snapshot.evictions,
                peak_memory_bytes=self._peak_memory_bytes(),
            )
        )

    def _emit_cancelled(self, job: _ScheduledRequest) -> None:
        job.timing.completed_at = time.perf_counter()
        self._sink.emit_response(
            ChatCompletionResponse(
                request_id=job.request.request_id,
                model=self._context.model_id,
                text=job.rendered_text,
                finish_reason="cancelled",
                prompt_tokens=len(job.prompt_tokens),
                completion_tokens=len(job.generated_tokens),
                prompt_cache_hit=job.cached_prompt is not None,
                cached_tokens=len(job.cached_prompt.tokens)
                if job.cached_prompt
                else None,
                prompt_cache_bytes=job.cached_prompt.cache_bytes
                if job.cached_prompt
                else None,
                active_batch_cache_bytes=self._active_batch_cache_bytes(),
                prompt_batch_size=job.prompt_batch_size or None,
                decode_batch_size=job.decode_batch_size or None,
                configured_prompt_batch_size=self._configured_prompt_batch_size,
                configured_decode_batch_size=self._configured_decode_batch_size,
                backend="text",
                modality="text",
                apc_mode="none",
                scheduler_stage=job.stage,
                cancellation_stage=job.cancelled_stage,
                queue_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.decode_started_at
                ),
                prefill_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.decode_started_at
                ),
                ttft_ms=_duration_ms(job.timing.admitted_at, job.timing.first_token_at),
                decode_time_ms=_duration_ms(
                    job.timing.decode_started_at, job.timing.completed_at
                ),
                completion_time_ms=_duration_ms(
                    job.timing.admitted_at, job.timing.completed_at
                ),
                scheduler_tick_latency_ms=self._last_tick_latency_ms,
                arbitration_delay_ms=self._arbitration_delay_ms,
                worker_cancellation_count=self._worker_cancellation_count,
                worker_error_count=self._worker_error_count,
                prompt_cache_entries=self._context.prompt_cache_store.stats_snapshot.entries,
                prompt_cache_evictions=self._context.prompt_cache_store.stats_snapshot.evictions,
                peak_memory_bytes=self._peak_memory_bytes(),
            )
        )

    def _maybe_cancel_generator(self, uid: int) -> None:
        if self._generator is None:
            return

        remove = getattr(self._generator, "remove", None)
        if remove is None:
            raise RuntimeError("BatchGenerator does not provide remove(uids)")
        remove([uid])

    def _active_batch_cache_bytes(self) -> int:
        if self._generator is None:
            return 0
        value = getattr(self._generator, "prompt_cache_nbytes", 0)
        return value if isinstance(value, int) else 0

    def _peak_memory_bytes(self) -> int:
        if self._generator_stats is None:
            return 0
        value = getattr(self._generator_stats, "peak_memory", 0) or 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


def _trim_mlx_prompt_cache(
    prompt_cache: Sequence[Any], num_tokens: int
) -> list[Any] | None:
    """Copy and trim an MLX prompt cache by a token count when supported."""

    cache_module = importlib.import_module("mlx_lm.models.cache")
    trim = getattr(cache_module, "trim_prompt_cache", None)
    if trim is None:
        return None
    copied_cache = copy.deepcopy(list(prompt_cache))
    try:
        trim(copied_cache, num_tokens)
    except Exception:
        return None
    return copied_cache


def _prompt_cache_for_prompt_tokens(
    prompt_cache: Sequence[Any],
    sequence_tokens: Sequence[int],
    prompt_tokens: Sequence[int],
) -> list[Any] | None:
    """Return cache trimmed back to prompt-token boundary for reuse."""

    generated_tokens = len(sequence_tokens) - len(prompt_tokens)
    if generated_tokens < 0:
        return None
    if generated_tokens == 0:
        return copy.deepcopy(list(prompt_cache))
    return _trim_mlx_prompt_cache(prompt_cache, generated_tokens)


def validate_continuous_batching_backend() -> None:
    """Validate the installed text LLM batching API before worker readiness."""

    _validate_tested_minor_version("mlx-lm", expected_minor_prefix="0.31.")
    _load_batch_generator_cls(continuous=True)


def _load_batch_generator_cls(*, continuous: bool) -> type[Any]:
    module = importlib.import_module("mlx_lm.generate")
    batch_generator_cls = getattr(module, "BatchGenerator", None)
    if batch_generator_cls is None:
        raise RuntimeError(
            "mlx_lm.generate.BatchGenerator is unavailable; cannot batch text LLM requests"
        )
    _validate_batch_generator_contract(batch_generator_cls, continuous=continuous)
    return batch_generator_cls


def _validate_batch_generator_contract(
    batch_generator_cls: type[Any], *, continuous: bool
) -> None:
    """Fail fast when the installed mlx-lm batching API is incompatible."""

    required_parameters = {
        "insert": {"prompts", "max_tokens", "caches", "all_tokens", "samplers"},
        "next_generated": set(),
        "close": set(),
    }
    if continuous:
        required_parameters["__init__"] = {
            "completion_batch_size",
            "prefill_batch_size",
            "prefill_step_size",
        }
        required_parameters["remove"] = {"uids"}

    missing: list[str] = []
    for method_name, parameter_names in required_parameters.items():
        method = getattr(batch_generator_cls, method_name, None)
        if method is None:
            missing.append(method_name)
            continue
        available = set(inspect.signature(method).parameters)
        for parameter_name in sorted(parameter_names - available):
            missing.append(f"{method_name}({parameter_name}=...)")

    if missing:
        details = ", ".join(missing)
        raise RuntimeError(
            "installed mlx-lm BatchGenerator is incompatible with the runtime: "
            f"missing {details}"
        )


def _estimate_cache_bytes(prompt_cache: Sequence[Any]) -> int:
    total = 0
    for item in prompt_cache:
        nbytes = getattr(item, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
            continue
        if isinstance(item, (bytes, bytearray, memoryview, str, list, tuple)):
            total += len(item)
    return total


def _validate_tested_minor_version(
    package_name: str, *, expected_minor_prefix: str
) -> None:
    version = importlib.metadata.version(package_name)
    if not version.startswith(expected_minor_prefix):
        raise RuntimeError(
            f"{package_name} {version} is outside tested minor range "
            f"{expected_minor_prefix}x"
        )
