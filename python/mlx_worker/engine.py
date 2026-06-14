"""Phase 1 non-streaming inference engine backed by direct `mlx-lm` calls."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from .ipc import ChatCompletionRequest, ChatCompletionResponse


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
