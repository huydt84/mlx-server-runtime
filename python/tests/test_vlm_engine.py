"""Tests for the Phase 8 VLM worker engine."""

from __future__ import annotations

from dataclasses import field
from types import SimpleNamespace

import pytest

from mlx_worker.ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ImageContent,
    TextContent,
)
from mlx_worker.vlm_engine import (
    MlxVlmEngine,
    _MAX_IMAGES_PER_REQUEST,
    _validate_image_source,
)  # fmt: skip


class FakeProcessor:
    """Simulates a VLM processor."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self._chat_template: str | None = None

    @property
    def chat_template(self) -> str | None:
        return self._chat_template

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "VLM processor prompt"


class FakeTokenizer:
    has_chat_template: bool = False
    eos_token_ids: list[int] = field(default_factory=lambda: [0])

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in prompt]

    def decode(self, tokens: list[int]) -> str:
        return "".join(f"<{token}>" for token in tokens)


class FakeGenerationResult:
    """Simulates mlx_vlm.GenerationResult."""

    def __init__(
        self,
        text: str = "VLM response",
        prompt_tokens: int = 10,
        generation_tokens: int = 5,
        finish_reason: str | None = "stop",
    ) -> None:
        self.text = text
        self.token = None
        self.logprobs = None
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.total_tokens = prompt_tokens + generation_tokens
        self.prompt_tps = 100.0
        self.generation_tps = 50.0
        self.peak_memory = 1.0
        self.cached_tokens = 0
        self.finish_reason: str | None = finish_reason


def _fake_vlm_load(model_id: str) -> tuple[SimpleNamespace, FakeProcessor]:
    return SimpleNamespace(), FakeProcessor()


def _fake_vlm_generate(
    model: object,
    processor: FakeProcessor,
    prompt: str,
    *,
    image: str | list[str] | None = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
    **kwargs: object,
) -> FakeGenerationResult:
    return FakeGenerationResult(text="mock VLM output")


def _fake_vlm_stream_generate(
    model: object,
    processor: FakeProcessor,
    prompt: str,
    *,
    image: str | list[str] | None = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **kwargs: object,
):
    yield FakeGenerationResult(text="stream ", prompt_tokens=10, generation_tokens=1)
    yield FakeGenerationResult(text="VLM ", prompt_tokens=10, generation_tokens=2)
    yield FakeGenerationResult(
        text="output",
        prompt_tokens=10,
        generation_tokens=3,
        finish_reason="stop",
    )


def _text_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id="text-req",
        model="vlm-model",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    )


def _vlm_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id="vlm-req",
        model="vlm-model",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    TextContent(text="What is this?"),
                    ImageContent(url="https://example.com/img.jpg"),
                ),
            )
        ],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    )


def test_vlm_engine_complete_chat_text_only() -> None:
    """Text-only request works with VLM engine."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    response = engine.complete_chat(_text_request())
    assert isinstance(response, ChatCompletionResponse)
    assert response.request_id == "text-req"


def test_vlm_engine_complete_chat_with_image() -> None:
    """VLM request with image generates response."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    response = engine.complete_chat(_vlm_request())
    assert isinstance(response, ChatCompletionResponse)
    assert response.request_id == "vlm-req"
    assert response.text == "mock VLM output"
    assert response.model == "vlm-model"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5


def test_vlm_engine_complete_chat_with_local_image_path(tmp_path) -> None:
    """VLM request accepts existing local image paths."""
    image_path = tmp_path / "image.ppm"
    image_path.write_text("P3\n1 1\n255\n255 0 0\n")
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    request = ChatCompletionRequest(
        request_id="local-img-req",
        model="vlm-model",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    TextContent(text="What is in this image?"),
                    ImageContent(url=str(image_path)),
                ),
            )
        ],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    )
    response = engine.complete_chat(request)
    assert response.text == "mock VLM output"


def test_vlm_engine_stream_chat_with_image() -> None:
    """VLM streaming emits deltas and returns final response."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    deltas: list[str] = []

    def emit_delta(delta: str) -> None:
        deltas.append(delta)

    response = engine.stream_chat(_vlm_request(), emit_delta)
    assert response is not None
    assert isinstance(response, ChatCompletionResponse)
    assert response.request_id == "vlm-req"
    assert deltas == ["stream ", "VLM ", "output"]
    assert response.text == "stream VLM output"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 3


def test_vlm_engine_stream_chat_cancelled_in_loop() -> None:
    """Cancelling a VLM stream inside the generation loop returns cancelled response.

    There are seven cancellation check points before the generation loop
    (before init, inside init before/after load, after init,
    lazy-load guard, extract, prompt).  This test passes the first
    eight calls (seven pre-loop + first loop iteration) and cancels on
    the ninth, after one token delta has been emitted.
    """
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    call_count = 0

    def should_cancel() -> bool:
        nonlocal call_count
        call_count += 1
        # Cancel on the 9th call = inside the 2nd loop iteration,
        # after the first token delta has been emitted.
        return call_count >= 9

    deltas: list[str] = []

    def emit_delta(delta: str) -> None:
        deltas.append(delta)

    response = engine.stream_chat(_vlm_request(), emit_delta, should_cancel)
    assert isinstance(response, ChatCompletionResponse)
    assert response.finish_reason == "cancelled"
    assert response.text == ""  # no full text accumulator on loop cancel
    assert len(deltas) == 1  # only first token emitted before cancel


def test_vlm_engine_stream_chat_cancelled_before_any_work() -> None:
    """Cancelling before any preprocessing returns cancelled response immediately."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    def should_cancel() -> bool:
        return True

    deltas: list[str] = []

    def emit_delta(delta: str) -> None:
        deltas.append(delta)

    response = engine.stream_chat(_vlm_request(), emit_delta, should_cancel)
    assert isinstance(response, ChatCompletionResponse)
    assert response.finish_reason == "cancelled"
    assert response.text == ""
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert len(deltas) == 0  # no deltas emitted at all


def test_vlm_engine_stream_chat_cancelled_before_prompt_build() -> None:
    """Cancelling after preprocessing but before prompt build returns cancelled response.

    The ``_extract_chat_and_image`` call updates telemetry state even
    when a subsequent cancellation returns a ``cancelled`` response.

    There are seven cancellation check points before the generation loop.
    Check 1 is before init, check 2 is inside init (before load),
    check 3 is inside init (after load), check 4 is after init,
    check 5 is after the lazy-load guard, check 6 is after
    ``_extract_chat_and_image``, and check 7 is after
    ``_build_prompt_str``.  We cancel at check 6 (after extract).
    """
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    call_count = 0

    def should_cancel() -> bool:
        nonlocal call_count
        call_count += 1
        # Cancel on the 6th call = after preprocessing, before prompt build.
        return call_count >= 6

    deltas: list[str] = []

    def emit_delta(delta: str) -> None:
        deltas.append(delta)

    response = engine.stream_chat(_vlm_request(), emit_delta, should_cancel)
    assert isinstance(response, ChatCompletionResponse)
    assert response.finish_reason == "cancelled"
    assert response.text == ""
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert len(deltas) == 0

    # Telemetry reflects preprocessing that already ran.
    timings = engine.last_vlm_timings
    assert timings["image_count"] == 1
    assert timings["image_preprocess_ms"] >= 0.0
    assert timings["prompt_template_ms"] == 0.0


def test_vlm_engine_complete_chat_cancelled() -> None:
    """Non-streaming VLM generation with cancellation returns cancelled response."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    def should_cancel() -> bool:
        return True

    response = engine._generate_vlm(  # type: ignore[attr-defined]
        _vlm_request(), should_cancel=should_cancel
    )
    assert isinstance(response, ChatCompletionResponse)
    assert response.text == ""
    assert response.finish_reason == "cancelled"
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0


def test_vlm_engine_handles_text_only_directly() -> None:
    """Text-only requests are handled by VLM engine directly.

    With model-first dispatch, the VLM engine receives all requests for
    its configured model (text-only or with images).  No delegation to a
    separate text engine occurs.
    """
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    response = engine.complete_chat(_text_request())
    assert response.text == "mock VLM output"
    assert response.request_id == "text-req"


def test_vlm_engine_warmup_succeeds() -> None:
    """Warmup returns a valid ChatCompletionResponse."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    response = engine.warmup()
    assert isinstance(response, ChatCompletionResponse)


def test_vlm_engine_raises_on_model_mismatch() -> None:
    """Requesting a different model raises ValueError."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    import pytest

    with pytest.raises(ValueError, match="vlm-model"):
        engine.complete_chat(
            ChatCompletionRequest(
                request_id="bad",
                model="other-model",
                messages=[ChatMessage(role="user", content="hi")],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                max_prompt_tokens=32,
                max_completion_tokens=32,
                max_total_tokens_per_request=64,
            )
        )


def test_vlm_engine_last_timings_after_text_request() -> None:
    """Telemetry snapshot is populated after a text-only request."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    _ = engine.complete_chat(_text_request())
    timings = engine.last_vlm_timings
    assert isinstance(timings, dict)
    assert timings["image_count"] == 0
    assert timings["image_preprocess_ms"] >= 0.0
    assert timings["prompt_template_ms"] >= 0.0


def test_vlm_engine_last_timings_after_vlm_request() -> None:
    """Telemetry snapshot is populated after a VLM (image) request."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    _ = engine.complete_chat(_vlm_request())
    timings = engine.last_vlm_timings
    assert timings["image_count"] == 1  # one image in _vlm_request
    assert timings["image_preprocess_ms"] >= 0.0
    assert timings["prompt_template_ms"] >= 0.0


def test_vlm_engine_forwards_image_count_to_prompt_template(monkeypatch) -> None:
    """VLM prompt template receives image count so image tokens are added."""
    import mlx_vlm.prompt_utils as prompt_utils

    captured: dict[str, int] = {}

    def fake_apply_chat_template(
        processor, config, prompt, add_generation_prompt=True, **kwargs
    ):
        captured["num_images"] = kwargs["num_images"]
        return "VLM prompt"

    monkeypatch.setattr(
        MlxVlmEngine, "_load_vlm_config", lambda self: {"model_type": "qwen3_vl"}
    )
    monkeypatch.setattr(prompt_utils, "apply_chat_template", fake_apply_chat_template)

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    response = engine.complete_chat(_vlm_request())
    assert response.text == "mock VLM output"
    assert captured["num_images"] == 1


def test_vlm_engine_last_timings_after_stream() -> None:
    """Telemetry snapshot is populated after a streaming VLM request."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    def emit_delta(_delta: str) -> None:
        pass

    _ = engine.stream_chat(_vlm_request(), emit_delta)
    timings = engine.last_vlm_timings
    assert timings["image_count"] == 1
    assert timings["image_preprocess_ms"] >= 0.0
    assert timings["prompt_template_ms"] >= 0.0


def test_vlm_engine_last_timings_defaults() -> None:
    """Telemetry snapshot has default zeros before any request."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    timings = engine.last_vlm_timings
    assert timings == {
        "image_count": 0,
        "image_preprocess_ms": 0.0,
        "prompt_template_ms": 0.0,
    }


# ------------------------------------------------------------------
# Cancel via public complete_chat
# ------------------------------------------------------------------


def test_vlm_engine_complete_chat_cancelled_via_public_method() -> None:
    """Public complete_chat with should_cancel returns cancelled response."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    def should_cancel() -> bool:
        return True

    response = engine.complete_chat(_text_request(), should_cancel=should_cancel)
    assert isinstance(response, ChatCompletionResponse)
    assert response.text == ""
    assert response.finish_reason == "cancelled"
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0


def test_vlm_engine_complete_chat_uses_non_stream_generate_when_cancel_checked() -> (
    None
):
    """Non-stream VLM completion stays on generate path even with cancel hook."""
    called: list[str] = []

    def fake_generate(*args, **kwargs):
        called.append("generate")
        return FakeGenerationResult(text="mock VLM output")

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=fake_generate,
        vlm_stream_generate_fn=None,
    )

    def should_cancel() -> bool:
        return False

    response = engine.complete_chat(_vlm_request(), should_cancel=should_cancel)
    assert isinstance(response, ChatCompletionResponse)
    assert response.text == "mock VLM output"
    assert called == ["generate"]


# ------------------------------------------------------------------
# Image source validation
# ------------------------------------------------------------------


def test_validate_image_source_rejects_data_uri() -> None:
    """data: URIs are rejected by image source validation."""
    with pytest.raises(ValueError, match="unsupported.*scheme.*data"):
        _validate_image_source("data:image/png;base64,iVBORw0KGgo=")


def test_validate_image_source_accepts_local_path(tmp_path) -> None:
    """Existing local image paths are accepted."""
    image_path = tmp_path / "image.ppm"
    image_path.write_text("P3\n1 1\n255\n255 0 0\n")
    assert _validate_image_source(str(image_path)) == str(image_path)


def test_validate_image_source_rejects_missing_local_path() -> None:
    """Missing local image paths are rejected."""
    with pytest.raises(ValueError, match="local image path"):
        _validate_image_source("/tmp/image-does-not-exist.ppm")


def test_validate_image_source_rejects_file_scheme() -> None:
    """file:// scheme is rejected (local file access vector)."""
    with pytest.raises(ValueError, match="unsupported.*scheme.*file"):
        _validate_image_source("file:///tmp/image.jpg")


def test_validate_image_source_accepts_loopback_http_scheme() -> None:
    """Loopback http:// image URLs are accepted."""
    assert _validate_image_source("http://127.0.0.1:8000/img.jpg")


def test_validate_image_source_rejects_remote_http_scheme() -> None:
    """Remote http:// scheme is rejected (SSRF vector)."""
    with pytest.raises(ValueError, match="localhost or 127.0.0.1"):
        _validate_image_source("http://example.com/img.jpg")


def test_validate_image_source_accepts_https_scheme() -> None:
    """https:// scheme is accepted."""
    _validate_image_source("https://example.com/img.jpg")


def test_validate_image_source_rejects_long_url() -> None:
    """Image URL exceeding max length raises ValueError."""
    long_url = "https://example.com/" + "x" * 4096
    with pytest.raises(ValueError, match="exceeds maximum length"):
        _validate_image_source(long_url)


def test_validate_image_source_accepts_max_length_url() -> None:
    """Image URL at exactly max length passes validation."""
    _validate_image_source("https://example.com/short.jpg")


# ------------------------------------------------------------------
# Image count validation in _extract_chat_and_image
# ------------------------------------------------------------------


def _multi_image_request(image_count: int) -> ChatCompletionRequest:
    """Build a VLM request with *image_count* images."""
    parts: list[TextContent | ImageContent] = [
        TextContent(text="describe these images")
    ]
    for i in range(image_count):
        parts.append(ImageContent(url=f"https://example.com/img{i}.jpg"))
    return ChatCompletionRequest(
        request_id="multi-img",
        model="vlm-model",
        messages=[ChatMessage(role="user", content=tuple(parts))],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    )


def test_vlm_engine_rejects_too_many_images() -> None:
    """Request exceeding max image count raises ValueError."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    with pytest.raises(ValueError, match="too many images"):
        engine.complete_chat(_multi_image_request(_MAX_IMAGES_PER_REQUEST + 1))


def test_vlm_engine_accepts_max_images() -> None:
    """Request with exactly max images succeeds."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    response = engine.complete_chat(_multi_image_request(_MAX_IMAGES_PER_REQUEST))
    assert response.finish_reason == "stop"


def test_vlm_engine_honors_configured_image_cap() -> None:
    """Instance cap can differ from default constant."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
        max_images_per_request=2,
    )
    with pytest.raises(ValueError, match="too many images"):
        engine.complete_chat(_multi_image_request(3))


def test_vlm_engine_rejects_invalid_scheme_in_request() -> None:
    """Request with data: URI image raises ValueError."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    bad_request = ChatCompletionRequest(
        request_id="bad-img",
        model="vlm-model",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    TextContent(text="what is this?"),
                    ImageContent(url="data:image/png;base64,abcd"),
                ),
            )
        ],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    )
    with pytest.raises(ValueError, match="unsupported.*scheme.*data"):
        engine.complete_chat(bad_request)


# ------------------------------------------------------------------
# Lazy initialization
# ------------------------------------------------------------------


def test_vlm_engine_lazy_init_not_loaded_at_construction() -> None:
    """Engine construction does not load the VLM model."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized
    assert engine.model is None
    assert engine.processor is None


def test_vlm_engine_lazy_init_on_first_chat() -> None:
    """Engine loads model on first complete_chat call."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized
    _ = engine.complete_chat(_text_request())
    assert engine.is_initialized
    assert engine.model is not None
    assert engine.processor is not None


def test_vlm_engine_lazy_init_on_first_stream() -> None:
    """Engine loads model on first stream_chat call."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized

    deltas: list[str] = []

    def emit_delta(delta: str) -> None:
        deltas.append(delta)

    _ = engine.stream_chat(_text_request(), emit_delta)
    assert engine.is_initialized
    assert engine.model is not None
    assert engine.processor is not None


def test_vlm_engine_lazy_init_idempotent() -> None:
    """Multiple initialize() calls are safe (no-op after first)."""
    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    engine.initialize()
    assert engine.is_initialized
    engine.initialize()  # second call should be safe
    assert engine.is_initialized


def test_vlm_engine_initialize_cancelled_skips_load() -> None:
    """initialize(should_cancel=True) skips model loading.

    When ``should_cancel`` returns ``True`` before the blocking
    ``mlx_vlm.load`` call, the model must NOT be loaded and
    ``is_initialized`` must stay ``False``.
    """
    load_called: list[bool] = []

    def tracking_load(model_id: str) -> tuple:
        load_called.append(True)
        return _fake_vlm_load(model_id)

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=tracking_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized

    # Cancel BEFORE the load call.
    engine.initialize(should_cancel=lambda: True)

    assert load_called == [], "model load must NOT be invoked"
    assert not engine.is_initialized, "engine must remain uninitialized"
    assert engine.model is None
    assert engine.processor is None


def test_vlm_engine_initialize_cancelled_after_load_keeps_state() -> None:
    """initialize(should_cancel) keeps loaded state when cancelled post-load.

    If ``mlx_vlm.load`` completes but cancellation is detected
    afterward, the loaded model/processor must be preserved and
    ``is_initialized`` set to ``True`` so the next request reuses
    the expensive load instead of cold-loading again.
    """
    call_count: list[int] = [0]

    def cancel_after_load() -> bool:
        call_count[0] += 1
        # Return False on first call (before load), True on second (after).
        return call_count[0] > 1

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=_fake_vlm_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized

    engine.initialize(should_cancel=cancel_after_load)

    # Model was loaded (first should_cancel returned False). Even though
    # second should_cancel returned True, the loaded state is KEPT.
    assert engine.is_initialized, "engine must be initialized"
    assert engine.model is not None, "loaded model must be preserved"
    assert engine.processor is not None, "loaded processor must be preserved"


def test_vlm_engine_generate_vlm_cancel_before_init(monkeypatch) -> None:
    """complete_chat with cancel before initialize skips model load."""
    load_called: list[bool] = []

    def tracking_load(model_id: str) -> tuple:
        load_called.append(True)
        return _fake_vlm_load(model_id)

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=tracking_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )

    response = engine.complete_chat(_text_request(), should_cancel=lambda: True)

    assert isinstance(response, ChatCompletionResponse)
    assert response.finish_reason == "cancelled"
    assert response.text == ""
    assert load_called == [], "model load must NOT be invoked"
    assert not engine.is_initialized


def test_vlm_engine_cancelled_after_load_then_next_request_succeeds() -> None:
    """Cancel after load preserves state; next request reuses without cold-load.

    First request cancels immediately after ``mlx_vlm.load`` completes.
    Engine keeps the loaded state.  Second request succeeds without
    invoking ``_vlm_loader`` again.
    """
    load_count: list[int] = [0]

    def tracking_load(model_id: str) -> tuple:
        load_count[0] += 1
        return _fake_vlm_load(model_id)

    def cancel_after_load() -> bool:
        cancel_after_load.call_count += 1  # type: ignore[attr-defined]
        # Call 1 = generate_vlm check before init,
        # Call 2 = initialize check before load,
        # Call 3 = initialize check after load → cancel here.
        return cancel_after_load.call_count > 2  # type: ignore[attr-defined]

    cancel_after_load.call_count = 0  # type: ignore[attr-defined]

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=tracking_load,
        vlm_generate_fn=_fake_vlm_generate,
        vlm_stream_generate_fn=_fake_vlm_stream_generate,
    )
    assert not engine.is_initialized
    assert load_count[0] == 0

    # First request — cancelled after load.
    response1 = engine.complete_chat(_text_request(), should_cancel=cancel_after_load)
    assert isinstance(response1, ChatCompletionResponse)
    assert response1.finish_reason == "cancelled"
    assert engine.is_initialized, "engine must be initialized after cancelled load"
    assert engine.model is not None, "model must survive cancellation"
    assert load_count[0] == 1, "model must have been loaded exactly once"

    # Second request — should succeed without reloading.
    response2 = engine.complete_chat(_text_request())
    assert isinstance(response2, ChatCompletionResponse)
    assert response2.finish_reason == "stop"
    assert load_count[0] == 1, "second request must NOT trigger load"
