"""Phase 1 worker engine with direct `mlx-lm` calls and batch completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .batching import (
    BatchBackendContext,
    BatchCompletionBackend,
    PromptCacheStore,
    create_default_batch_backend,
)
from .ipc import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


def _load_mlx_components() -> tuple[
    Callable[[str], tuple[Any, Any]],
    Callable[..., Any],
    Callable[[float, float], Callable[[Any], Any]],
]:
    """Load the runtime MLX components lazily."""

    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ModuleNotFoundError as exc:  # pragma: no cover - host-only dependency
        raise RuntimeError("mlx_lm is required for the default worker engine") from exc

    return load, stream_generate, make_sampler


def _noop_sampler(_temp: float, _top_p: float) -> Callable[[Any], Any]:
    """Fallback sampler used only when tests inject their own backend."""

    return lambda logits: logits


@dataclass
class MlxWorkerEngine:
    """A worker engine using direct `mlx-lm` generation primitives."""

    model_id: str
    model_loader: Callable[[str], tuple[Any, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    batch_backend_factory: (
        Callable[[BatchBackendContext], BatchCompletionBackend] | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Load the configured model and configure batching."""

        if self.model_loader is None:
            load_model, sg, ms = _load_mlx_components()
            self._model_loader = load_model
            self._stream_generate = sg
            self._make_sampler = ms
        else:
            self._model_loader = self.model_loader
            if self.batch_backend_factory is None:
                _, self._stream_generate, self._make_sampler = _load_mlx_components()
            else:
                self._stream_generate = None
                self._make_sampler = _noop_sampler

        self.model, self.tokenizer = self._model_loader(self.model_id)
        self._prompt_cache_store = PromptCacheStore()
        context = BatchBackendContext(
            model_id=self.model_id,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_cache_store=self._prompt_cache_store,
            build_prompt_tokens=self._build_prompt_tokens,
            validate_token_limits=self._validate_token_limits,
            make_sampler=self._make_sampler,
        )
        backend_factory = self.batch_backend_factory or create_default_batch_backend
        self._batch_backend = backend_factory(context)

    def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate one non-streaming chat completion."""

        return self.complete_many([request])[0]

    def complete_many(
        self, requests: Sequence[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        """Generate a batch of non-streaming chat completions."""

        for request in requests:
            if request.model != self.model_id:
                raise ValueError(
                    f"requested model '{request.model}' does not match loaded model '{self.model_id}'"
                )

        return self._batch_backend.complete_many(list(requests))

    def stream_chat(
        self,
        request: ChatCompletionRequest,
        emit_delta: Callable[[str], None],
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        """Generate a streaming chat completion and emit text deltas."""

        if request.model != self.model_id:
            raise ValueError(
                f"requested model '{request.model}' does not match loaded model '{self.model_id}'"
            )

        prompt_tokens = self._build_prompt_tokens(request)
        self._validate_token_limits(request, prompt_tokens)
        if self._stream_generate is None:
            raise RuntimeError("streaming requires the MLX runtime components")

        sampler = self._make_sampler(request.temperature, request.top_p)
        text_segments: list[str] = []
        final_response = None

        for response in self._stream_generate(
            self.model,
            self.tokenizer,
            prompt_tokens,
            max_tokens=request.max_tokens,
            sampler=sampler,
        ):
            if should_cancel is not None and should_cancel():
                return ChatCompletionResponse(
                    request_id=request.request_id,
                    model=self.model_id,
                    text="".join(text_segments),
                    finish_reason="cancelled",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            if response.text:
                text_segments.append(response.text)
                emit_delta(response.text)
            final_response = response
            if should_cancel is not None and should_cancel():
                return ChatCompletionResponse(
                    request_id=request.request_id,
                    model=self.model_id,
                    text="".join(text_segments),
                    finish_reason="cancelled",
                    prompt_tokens=0,
                    completion_tokens=0,
                )

        if final_response is None:
            raise RuntimeError("mlx-lm returned no completion response")

        return ChatCompletionResponse(
            request_id=request.request_id,
            model=self.model_id,
            text="".join(text_segments),
            finish_reason=final_response.finish_reason or "stop",
            prompt_tokens=int(final_response.prompt_tokens),
            completion_tokens=int(final_response.generation_tokens),
        )

    def warmup(self) -> ChatCompletionResponse:
        """Run a tiny warmup completion before reporting readiness."""

        return self.complete_chat(
            ChatCompletionRequest(
                request_id="warmup",
                model=self.model_id,
                messages=[ChatMessage(role="user", content="ping")],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                max_prompt_tokens=64,
                max_completion_tokens=64,
                max_total_tokens_per_request=128,
            )
        )

    def _build_prompt_tokens(self, request: ChatCompletionRequest) -> list[int]:
        """Convert chat messages into prompt token ids."""

        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

        if getattr(self.tokenizer, "has_chat_template", False):
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "\n".join(
                    f"{message.role.capitalize()}: {message.content}"
                    for message in request.messages
                )
                + "\nAssistant:"
            )
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return list(tokens)

    def _validate_token_limits(
        self,
        request: ChatCompletionRequest,
        prompt_tokens: Sequence[int],
    ) -> None:
        """Reject prompts that exceed gateway token caps."""

        prompt_token_count = len(prompt_tokens)
        completion_tokens = request.max_tokens
        total_tokens = prompt_token_count + completion_tokens

        if prompt_token_count > request.max_prompt_tokens:
            raise ValueError(
                f"prompt too long: {prompt_token_count} tokens exceeds max_prompt_tokens {request.max_prompt_tokens}"
            )
        if completion_tokens > request.max_completion_tokens:
            raise ValueError(
                f"completion too long: {completion_tokens} tokens exceeds max_completion_tokens {request.max_completion_tokens}"
            )
        if total_tokens > request.max_total_tokens_per_request:
            raise ValueError(
                f"request too large: {total_tokens} tokens exceeds max_total_tokens_per_request {request.max_total_tokens_per_request}"
            )
