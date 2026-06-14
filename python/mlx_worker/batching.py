"""Batch-aware completion backend for the MLX worker."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
import importlib
from typing import Any, Callable, Protocol, Sequence

from .ipc import ChatCompletionRequest, ChatCompletionResponse


@dataclass(frozen=True)
class CachedPrompt:
    """A cached prompt prefix and its MLX prompt cache snapshot."""

    tokens: tuple[int, ...]
    prompt_cache: list[Any]


class PromptCacheStore:
    """Store prompt caches for recent prompt prefixes."""

    def __init__(self, max_entries: int = 32) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], list[Any]] = OrderedDict()

    def lookup(self, prompt_tokens: Sequence[int]) -> CachedPrompt | None:
        """Return the longest cached prefix for the prompt, if any."""

        prompt_key = tuple(prompt_tokens)
        best: CachedPrompt | None = None
        for cached_key, cached_prompt in reversed(self._entries.items()):
            if len(cached_key) >= len(prompt_key):
                continue
            if prompt_key[: len(cached_key)] != cached_key:
                continue
            candidate = CachedPrompt(
                tokens=cached_key, prompt_cache=list(cached_prompt)
            )
            if best is None or len(candidate.tokens) > len(best.tokens):
                best = candidate
        return best

    def remember(
        self, prompt_tokens: Sequence[int], prompt_cache: Sequence[Any]
    ) -> None:
        """Remember a prompt cache snapshot for later prefix reuse."""

        if not prompt_cache:
            return

        key = tuple(prompt_tokens)
        self._entries[key] = list(prompt_cache)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


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
                continue

            suffix = tokens[len(cached_prompt.tokens) :]
            if not suffix:
                effective_prompts.append(tokens)
                cache_inputs.append(None)
                continue

            effective_prompts.append(suffix)
            cache_inputs.append(cached_prompt.prompt_cache)

        generator = batch_generator_cls(
            self._context.model,
            stop_tokens=[[token] for token in self._context.tokenizer.eos_token_ids],
        )
        try:
            uids = generator.insert(
                effective_prompts,
                max_tokens=max_tokens,
                caches=cache_inputs,
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
            prompt_caches: dict[int, Sequence[Any]] = {}
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
                            prompt_caches[uid] = response.prompt_cache
                        pending_uids.discard(uid)
        finally:
            with suppress(Exception):
                generator.close()

        responses: list[ChatCompletionResponse] = []
        for uid in uids:
            index = uid_to_index[uid]
            text = self._context.tokenizer.decode(generated_tokens[uid])
            finish_reason = finish_reasons.get(uid, "stop")
            response = ChatCompletionResponse(
                request_id=requests[index].request_id,
                model=self._context.model_id,
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=len(prompt_tokens[index]),
                completion_tokens=len(generated_tokens[uid]),
            )
            responses.append(response)

            cached_prompt = prompt_caches.get(uid)
            if cached_prompt is not None:
                self._context.prompt_cache_store.remember(
                    prompt_tokens[index],
                    cached_prompt,
                )

        return responses

    def _get_batch_generator_cls(self) -> type[Any]:
        if self._batch_generator_cls is not None:
            return self._batch_generator_cls

        module = importlib.import_module("mlx_lm.generate")
        batch_generator_cls = getattr(module, "BatchGenerator", None)
        if batch_generator_cls is None:
            raise RuntimeError(
                "mlx_lm.generate.BatchGenerator is unavailable; cannot batch requests"
            )
        self._batch_generator_cls = batch_generator_cls
        return batch_generator_cls


def create_default_batch_backend(
    context: BatchBackendContext,
) -> BatchCompletionBackend:
    """Create the default batch completion backend for the worker."""

    return MlxBatchCompletionBackend(context)
