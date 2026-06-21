"""Batch-aware completion backend for the MLX worker."""

from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import suppress
import copy
from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Callable, Protocol, Sequence

from .ipc import ChatCompletionRequest, ChatCompletionResponse


@dataclass(frozen=True)
class CachedPrompt:
    """A cached prompt prefix and its MLX prompt cache snapshot."""

    tokens: tuple[int, ...]
    prompt_cache: list[Any]
    cache_bytes: int


class PromptCacheStore:
    """Store prompt caches for recent prompt prefixes."""

    def __init__(
        self,
        max_entries: int = 32,
        max_bytes: int = 8 * 1024 * 1024,
        trim_cache: Callable[[Sequence[Any], int], list[Any] | None] | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._trim_cache = trim_cache or _trim_mlx_prompt_cache
        self._entries: OrderedDict[
            tuple[tuple[Any, ...], tuple[int, ...]], CachedPrompt
        ] = OrderedDict()
        self._total_bytes = 0

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
                reusable_length = len(prompt_key) - 1
                if reusable_length <= 0:
                    continue
                trimmed_cache = self._trim_cache(cached_prompt.prompt_cache, 1)
                if trimmed_cache is None:
                    continue
                candidate = CachedPrompt(
                    tokens=prompt_key[:reusable_length],
                    prompt_cache=trimmed_cache,
                    cache_bytes=_estimate_cache_bytes(trimmed_cache),
                )
            elif len(cached_key) < len(prompt_key):
                if common_prefix == len(cached_key):
                    candidate = CachedPrompt(
                        tokens=cached_key,
                        prompt_cache=copy.deepcopy(cached_prompt.prompt_cache),
                        cache_bytes=cached_prompt.cache_bytes,
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
                )
            if best is None or len(candidate.tokens) > len(best.tokens):
                best = candidate
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
        )
        if cache.cache_bytes > self._max_bytes:
            return

        existing = self._entries.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing.cache_bytes

        self._entries[key] = cache
        self._total_bytes += cache.cache_bytes
        self._entries.move_to_end(key)
        while (
            len(self._entries) > self._max_entries
            or self._total_bytes > self._max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._total_bytes -= evicted.cache_bytes

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


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
                effective_prompts.append([])
                cache_inputs.append(cached_prompt.prompt_cache)
                cached_prefixes.append(list(cached_prompt.tokens))
                prompt_cache_hits.append(True)
                cached_token_counts.append(len(cached_prompt.tokens))
                prompt_cache_bytes.append(cached_prompt.cache_bytes)
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

            with generator.stats():
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
    ) -> None:
        self._context = context
        self._sink = sink
        self._prompt_concurrency = prompt_concurrency
        self._decode_concurrency = decode_concurrency
        self._prefill_step_size = prefill_step_size
        self._batch_generator_cls: type[Any] | None = None
        self._generator: Any | None = None
        self._pending: deque[_ScheduledRequest] = deque()
        self._active: dict[int, _ScheduledRequest] = {}

    def submit(self, request: ChatCompletionRequest, stream: bool) -> None:
        if request.model != self._context.model_id:
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
            self._emit_cancelled(job)
            return True

        for uid, job in list(self._active.items()):
            if job.request.request_id != request_id:
                continue
            self._active.pop(uid, None)
            self._emit_cancelled(job)
            self._maybe_cancel_generator(uid)
            return True

        return False

    def tick(self) -> None:
        if self._generator is None and not self._pending:
            return

        if self._active and self._generator is not None:
            self._step_generator()

        self._admit_pending()

    def idle(self) -> bool:
        return not self._pending and not self._active

    def close(self) -> None:
        if self._generator is not None:
            with suppress(Exception):
                self._generator.close()
            self._generator = None

    def _step_generator(self) -> None:
        responses = self._generator.next_generated()
        if not responses:
            return

        for response in responses:
            job = self._active.get(response.uid)
            if job is None:
                continue

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
            self._active[uid] = job

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
            )
        )

    def _emit_cancelled(self, job: _ScheduledRequest) -> None:
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
