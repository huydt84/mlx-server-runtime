"""Phase 8 VLM worker engine powered by direct ``mlx-vlm`` primitives."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ImageContent,
    TextContent,
)

# ---------------------------------------------------------------------------
# Image source validation constants
# ---------------------------------------------------------------------------

_MAX_IMAGES_PER_REQUEST = 5
"""Maximum allowed images in a single VLM request."""

_MAX_IMAGE_URL_LENGTH = 4096
"""Maximum length of an image URL string."""

_ALLOWED_IMAGE_SCHEMES = frozenset({"https", "http"})
"""Allowed URL schemes for image sources.

HTTPS web URLs, loopback HTTP URLs, and bare local filesystem paths are
supported.  Other schemes stay blocked.
"""

_ALLOWED_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""Allowed hosts for loopback HTTP image URLs."""


def _validate_image_source(url: str) -> str:
    """Validate a single image URL before passing upstream to ``mlx-vlm``.

    Checks:
      - URL string length does not exceed ``_MAX_IMAGE_URL_LENGTH``.
      - HTTPS URLs are allowed.
      - HTTP URLs are allowed only for loopback hosts.
      - Bare paths must exist and point at a file.

    Raises ``ValueError`` on validation failure.
    """
    if len(url) > _MAX_IMAGE_URL_LENGTH:
        raise ValueError(
            f"image URL exceeds maximum length of {_MAX_IMAGE_URL_LENGTH} characters"
        )
    parsed = urlparse(url)
    if parsed.scheme == "":
        path = Path(url).expanduser()
        if not path.is_file():
            raise ValueError(f"local image path does not exist or is not a file: {url}")
        return str(path)
    if parsed.scheme == "http":
        if parsed.hostname not in _ALLOWED_HTTP_HOSTS:
            raise ValueError("http image URLs must use localhost or 127.0.0.1")
        return url
    if parsed.scheme not in _ALLOWED_IMAGE_SCHEMES:
        raise ValueError(
            f"unsupported image URL scheme '{parsed.scheme}': "
            "must be one of https, http, or a local file path"
        )
    return url


def _load_vlm_components() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load the runtime MLX-VLM components lazily.

    Returns ``(load_fn, generate_fn)``.  ``stream_generate`` is loaded
    on first streaming call so unsupported streaming does not block
    non-streaming VLM startup.
    """
    try:
        from mlx_vlm import load as vlm_load
        from mlx_vlm import generate as vlm_generate
    except ModuleNotFoundError as exc:  # pragma: no cover - host-only dependency
        raise RuntimeError("mlx-vlm is required for the VLM worker engine") from exc

    return vlm_load, vlm_generate


@dataclass
class MlxVlmEngine:
    """A VLM worker engine using direct ``mlx-vlm`` primitives.

    Loads a vision-language model via ``mlx_vlm.load`` and generates
    responses using ``mlx_vlm.generate`` / ``mlx_vlm.stream_generate``.

    Handles both text-only and image requests directly through the VLM
    model.  Model-first dispatch is the caller's responsibility — this
    engine trusts it only receives requests for its configured model.
    """

    model_id: str
    vlm_loader: Callable[..., Any] | None = field(
        default=None, repr=False, compare=False
    )
    vlm_generate_fn: Callable[..., Any] | None = field(
        default=None, repr=False, compare=False
    )
    vlm_stream_generate_fn: Callable[..., Any] | None = field(
        default=None, repr=False, compare=False
    )
    max_images_per_request: int = field(
        default=_MAX_IMAGES_PER_REQUEST, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Initialize engine state without loading the VLM model.

        Model loading is deferred to ``initialize()`` (called on first
        ``complete_chat`` / ``stream_chat``) so construction is fast and
        the caller can observe the loading phase.
        """
        if self.vlm_loader is None:
            load_fn, gen_fn = _load_vlm_components()
            self._vlm_loader = load_fn
            self._vlm_generate = gen_fn
            self._vlm_stream_generate = None
        else:
            self._vlm_loader = self.vlm_loader
            self._vlm_generate = self.vlm_generate_fn
            self._vlm_stream_generate = self.vlm_stream_generate_fn

        # Deferred model state — populated by initialize().
        self.model: Any | None = None
        self.processor: Any | None = None
        self._vlm_config: dict[str, Any] | None = None
        self._initialized: bool = False

        # Telemetry state.
        self._last_image_count: int = 0
        self._last_image_preprocess_ms: float = 0.0
        self._last_prompt_template_ms: float = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, should_cancel: Callable[[], bool] | None = None) -> None:
        """Load the VLM model and processor on demand.

        Checks *should_cancel* before the blocking ``mlx_vlm.load`` call.
        When cancelled **before** load the model is NOT loaded and
        ``is_initialized`` stays ``False`` so a future request retries
        the cold-start load.

        When cancelled **after** load the loaded model/processor are
        preserved and ``is_initialized`` is set to ``True``.  The
        expensive load result is reused by the next request instead of
        forcing another cold-load (expensive memory churn).

        Safe to call multiple times — subsequent calls are no-ops once
        ``is_initialized`` is ``True``.
        """
        if self._initialized:
            return
        if should_cancel is not None and should_cancel():
            return  # caller handles cancelled response
        self.model, self.processor = self._vlm_loader(self.model_id)
        self._vlm_config = self._load_vlm_config()
        if should_cancel is not None and should_cancel():
            # Load completed but request was cancelled.
            # Keep loaded model/processor so next request reuses them
            # instead of cold-loading again (expensive memory churn).
            self._initialized = True
            return
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """True after the VLM model and processor have been loaded."""
        return self._initialized

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------

    def _load_vlm_config(self) -> dict[str, Any] | None:
        """Load model configuration via ``mlx_vlm.utils.load_config``.

        Returns ``None`` when the model artifact is not local yet (e.g. in
        test fixtures that inject bare ``SimpleNamespace`` objects).
        """
        try:
            from mlx_vlm.utils import load_config
        except ModuleNotFoundError:
            return None
        model_path = getattr(self.model, "model_path", None)
        if model_path is None:
            return None
        try:
            return load_config(str(model_path))
        except Exception:
            return None

    def complete_chat(
        self,
        request: ChatCompletionRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        """Generate a non-streaming chat completion.

        Handles both text-only and image requests directly through the VLM
        model.  The VLM dispatch is the caller's responsibility (model-first,
        not image-first) — this engine trusts it only receives requests for
        the configured VLM model.

        When *should_cancel* is provided the method checks it before image
        preprocessing, prompt building, and generation entry so disconnected
        or timed-out requests abort early.
        """
        if request.model != self.model_id:
            raise ValueError(
                f"requested model '{request.model}' does not match loaded model '{self.model_id}'"
            )

        return self._generate_vlm(request, should_cancel=should_cancel)

    def stream_chat(
        self,
        request: ChatCompletionRequest,
        emit_delta: Callable[[str], None],
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        """Generate a streaming chat completion with token deltas.

        Handles both text-only and image requests directly through the VLM
        model.  The VLM dispatch is the caller's responsibility (model-first,
        not image-first) — this engine trusts it only receives requests for
        the configured VLM model.

        Always returns a ``ChatCompletionResponse``.  When cancelled the
        response carries ``finish_reason="cancelled"`` so the gateway does
        not hang waiting for completion.
        """
        if request.model != self.model_id:
            raise ValueError(
                f"requested model '{request.model}' does not match loaded model '{self.model_id}'"
            )

        return self._stream_vlm(request, emit_delta, should_cancel)

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

    def _extract_chat_and_image(
        self, request: ChatCompletionRequest
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Split VLM messages into chat dicts and list of image URLs/paths.

        Validates image sources (URL scheme, length, count) before
        returning.  Records image count and preprocessing latency for
        telemetry.

        Returns (chat_messages, image_paths) where chat_messages uses the
        OpenAI content-part format for proper processor handling.

        Raises ``ValueError`` when image validation fails.
        """
        start = time.perf_counter()
        try:
            chat_messages: list[dict[str, Any]] = []
            image_paths: list[str] = []
            for message in request.messages:
                if isinstance(message.content, str):
                    chat_messages.append(
                        {"role": message.role, "content": message.content}
                    )
                else:
                    parts_list: list[dict[str, Any]] = []
                    for part in message.content:
                        if isinstance(part, TextContent):
                            parts_list.append({"type": "text", "text": part.text})
                        elif isinstance(part, ImageContent):
                            normalized_url = _validate_image_source(part.url)
                            parts_list.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": normalized_url,
                                        "detail": part.detail,
                                    },
                                }
                            )
                            image_paths.append(normalized_url)
                    chat_messages.append({"role": message.role, "content": parts_list})
            self._last_image_count = len(image_paths)
            if len(image_paths) > self.max_images_per_request:
                raise ValueError(
                    f"too many images: {len(image_paths)} exceeds "
                    f"maximum of {self.max_images_per_request}"
                )
            return chat_messages, image_paths
        finally:
            self._last_image_preprocess_ms = (time.perf_counter() - start) * 1000

    def _build_prompt_str(
        self, chat_messages: list[dict[str, Any]], *, num_images: int = 0
    ) -> str:
        """Build a VLM prompt string via ``mlx_vlm.prompt_utils``.

        Uses ``mlx_vlm.prompt_utils.apply_chat_template`` for model-type-aware
        prompt construction.  There is ``no`` manual string-concatenation
        fallback — the ``mlx-vlm`` utility handles every model type.
        """
        start = time.perf_counter()
        try:
            config = getattr(self, "_vlm_config", None)
            if config is not None:
                # Use the full mlx-vlm prompt utility stack.
                from mlx_vlm.prompt_utils import (
                    apply_chat_template,
                    get_chat_template,
                )

                result = apply_chat_template(
                    self.processor,
                    config,
                    chat_messages,
                    add_generation_prompt=True,
                    num_images=num_images,
                )
                if isinstance(result, str):
                    return result
                # Some model types return a dict or list; finalise through
                # the template.
                inner = [result] if isinstance(result, dict) else list(result)
                return get_chat_template(
                    self.processor, inner, add_generation_prompt=True
                )

            # No loaded config (test fixtures): use processor directly.
            if hasattr(self.processor, "apply_chat_template"):
                return self.processor.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            raise RuntimeError("VLM processor has no apply_chat_template method")
        except Exception as exc:
            raise RuntimeError(f"VLM prompt building failed: {exc}") from exc
        finally:
            self._last_prompt_template_ms = (time.perf_counter() - start) * 1000

    @staticmethod
    def _load_stream_generate() -> Callable[..., Any]:
        """Lazy-load ``mlx_vlm.stream_generate`` on first streaming call."""
        try:
            from mlx_vlm import stream_generate as vlm_stream_generate
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx-vlm is required for VLM streaming generation"
            ) from exc
        return vlm_stream_generate

    @staticmethod
    def _cancelled_response(
        request: ChatCompletionRequest, model_id: str
    ) -> ChatCompletionResponse:
        """Build a cancelled non-streaming response."""
        return ChatCompletionResponse(
            request_id=request.request_id,
            model=model_id,
            text="",
            finish_reason="cancelled",
            prompt_tokens=0,
            completion_tokens=0,
        )

    def _generate_vlm(
        self,
        request: ChatCompletionRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ChatCompletionResponse:
        """Generate a non-streaming VLM response.

        Ensures the model is initialized lazily, checks cancellation
        before expensive preprocessing, prompt building, and generation
        entry.
        """
        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        self.initialize(should_cancel)

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        chat_messages, image_paths = self._extract_chat_and_image(request)

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        prompt = self._build_prompt_str(chat_messages, num_images=len(image_paths))

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        try:
            result = self._vlm_generate(
                self.model,
                self.processor,
                prompt,
                image=image_paths if image_paths else None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"VLM generation failed: {exc}") from exc

        text = result.text if hasattr(result, "text") else str(result)
        finish_reason = getattr(result, "finish_reason", None) or "stop"
        prompt_tokens = int(getattr(result, "prompt_tokens", 0))
        completion_tokens = int(getattr(result, "generation_tokens", 0))

        return ChatCompletionResponse(
            request_id=request.request_id,
            model=self.model_id,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            image_count=len(image_paths),
            image_preprocess_latency_ms=max(
                1, math.ceil(self._last_image_preprocess_ms)
            ),
            prompt_template_latency_ms=max(1, math.ceil(self._last_prompt_template_ms)),
        )

    def _stream_vlm(
        self,
        request: ChatCompletionRequest,
        emit_delta: Callable[[str], None],
        should_cancel: Callable[[], bool] | None = None,
        *,
        check_cancel_before_work: bool = True,
    ) -> ChatCompletionResponse:
        """Generate a streaming VLM response, emitting token deltas.

        Checks cancellation before expensive preprocessing, prompt
        building, and generation entry so disconnected or timed-out
        requests abort early.

        Always returns a ``ChatCompletionResponse``.  When cancelled the
        response carries ``finish_reason="cancelled"`` so the gateway does
        not hang waiting for completion.
        """
        if check_cancel_before_work and should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        self.initialize(should_cancel)

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        if self._vlm_stream_generate is None:
            self._vlm_stream_generate = self._load_stream_generate()

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        chat_messages, image_paths = self._extract_chat_and_image(request)

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        prompt = self._build_prompt_str(chat_messages, num_images=len(image_paths))

        if should_cancel is not None and should_cancel():
            return self._cancelled_response(request, self.model_id)

        text_segments: list[str] = []
        final_response: object | None = None

        try:
            for response in self._vlm_stream_generate(
                self.model,
                self.processor,
                prompt,
                image=image_paths if image_paths else None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                if should_cancel is not None and should_cancel():
                    return self._cancelled_response(request, self.model_id)

                token_text = (
                    response.text if hasattr(response, "text") else str(response)
                )
                if token_text:
                    text_segments.append(token_text)
                    emit_delta(token_text)
                final_response = response
        except Exception as exc:
            raise RuntimeError(f"VLM streaming generation failed: {exc}") from exc

        finish_reason: str = "stop"
        prompt_tokens: int = 0
        completion_tokens: int = 0
        if final_response is not None:
            finish_reason = getattr(final_response, "finish_reason", None) or "stop"
            prompt_tokens = int(getattr(final_response, "prompt_tokens", 0))
            completion_tokens = int(getattr(final_response, "generation_tokens", 0))

        return ChatCompletionResponse(
            request_id=request.request_id,
            model=self.model_id,
            text="".join(text_segments),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            image_count=len(image_paths),
            image_preprocess_latency_ms=max(
                1, math.ceil(self._last_image_preprocess_ms)
            ),
            prompt_template_latency_ms=max(1, math.ceil(self._last_prompt_template_ms)),
        )

    # ------------------------------------------------------------------
    # Telemetry accessors
    # ------------------------------------------------------------------

    @property
    def last_vlm_timings(self) -> dict[str, int | float]:
        """Return the most-recent VLM request timing snapshot.

        Keys:
          ``image_count`` — number of images in the request
          ``image_preprocess_ms`` — content-extraction wall-clock (ms)
          ``prompt_template_ms`` — prompt-template application wall-clock (ms)

        Values reset to default on engine construction and are overwritten
        on every ``complete_chat`` / ``stream_chat`` call.
        """
        return {
            "image_count": self._last_image_count,
            "image_preprocess_ms": self._last_image_preprocess_ms,
            "prompt_template_ms": self._last_prompt_template_ms,
        }
