"""Phase 1 non-streaming inference engine backed by direct `mlx-lm` calls."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from typing import Callable

from .ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


@dataclass
class MlxWorkerEngine:
    """A minimal Phase 1 worker engine using direct `mlx-lm` primitives."""

    model_id: str

    def __post_init__(self) -> None:
        """Load the configured model and tokenizer once at worker startup."""

        self.model, self.tokenizer = load(self.model_id)

    def complete_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate one non-streaming chat completion."""

        if request.model != self.model_id:
            raise ValueError(
                f"requested model '{request.model}' does not match loaded model '{self.model_id}'"
            )

        prompt = self._build_prompt(request)
        self._validate_token_limits(request, prompt)
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        text_segments: list[str] = []
        final_response = None

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
        ):
            text_segments.append(response.text)
            final_response = response

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

        prompt = self._build_prompt(request)
        self._validate_token_limits(request, prompt)
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        text_segments: list[str] = []
        final_response = None

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
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

    def _build_prompt(self, request: ChatCompletionRequest) -> list[int] | str:
        """Convert chat messages into the simplest supported prompt form."""

        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

        if getattr(self.tokenizer, "has_chat_template", False):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

        return (
            "\n".join(
                f"{message.role.capitalize()}: {message.content}"
                for message in request.messages
            )
            + "\nAssistant:"
        )

    def _validate_token_limits(
        self,
        request: ChatCompletionRequest,
        prompt: list[int] | str,
    ) -> None:
        """Reject prompts that exceed gateway token caps."""

        if isinstance(prompt, list):
            prompt_tokens = len(prompt)
        else:
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))

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
