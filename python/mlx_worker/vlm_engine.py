"""Phase 8 VLM worker engine powered by direct ``mlx-vlm`` primitives."""

from __future__ import annotations

from collections import deque
import copy
import hashlib
import math
from contextlib import suppress
import time
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.metadata
import inspect
from typing import Sequence
from typing import Any, Callable
from urllib.parse import urlparse

from .batching import (
    BatchEventSink,
    CacheBudgetManager,
    CachedPrompt,
    PromptCacheStore,
    RequestTiming,
    _duration_ms,
    _prompt_cache_for_prompt_tokens,
)
from .ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ImageContent,
    TextContent,
)


def _make_sampler(temp: float, top_p: float) -> Callable[[Any], Any]:
    """Load the MLX sampler lazily for VLM continuous batching."""

    from mlx_lm.sample_utils import make_sampler

    return make_sampler(temp, top_p)


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


def _image_cache_key(image_paths: Sequence[str]) -> tuple[str, ...]:
    """Build media-safe cache namespace for image-bearing requests."""

    key_parts: list[str] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            key_parts.append(
                "remote:" + hashlib.sha256(image_path.encode("utf-8")).hexdigest()
            )
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key_parts.append(f"file:{digest}")
    return tuple(key_parts)


def _image_apc_hash(image_paths: Sequence[str], pixel_values: Any | None) -> int:
    """Build APC image hash for image-bearing requests."""

    try:
        from mlx_vlm import apc as _apc
    except (AttributeError, ImportError, ModuleNotFoundError):
        return 0

    return _apc.hash_image_payload(
        pixel_values=pixel_values,
        image_ref=list(image_paths) if image_paths else None,
    )


def _vlm_apc_cache_key(
    *,
    model_id: str,
    processor: Any,
    prompt_tokens: Sequence[int],
    image_paths: Sequence[str],
    apc_image_hash: int,
) -> tuple[Any, ...]:
    template_identity = getattr(processor, "chat_template", None)
    template_name = template_identity if isinstance(template_identity, str) else None
    adapter_identity = getattr(processor, "adapter_name", None)
    tenant_scope = getattr(processor, "tenant_scope", None) or "default"
    return (
        model_id,
        type(processor).__name__,
        template_name,
        adapter_identity,
        tuple(prompt_tokens),
        tuple(_image_cache_key(image_paths)),
        apc_image_hash,
        tenant_scope,
    )


def _estimate_array_bytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            item_bytes = _estimate_array_bytes(item)
            if item_bytes > 0:
                total += item_bytes
            elif isinstance(item, (int, float, complex, bool)):
                total += 1
            elif isinstance(item, (str, bytes, bytearray, memoryview)):
                total += len(item)
        return total or len(value)
    shape = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    dtype = getattr(value, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(size, int) and isinstance(itemsize, int):
        return size * itemsize
    if isinstance(shape, Sequence) and isinstance(itemsize, int):
        total = 1
        for dim in shape:
            if not isinstance(dim, int):
                return 0
            total *= dim
        return total * itemsize
    return 0


def _image_dimensions(model_inputs: dict[str, Any]) -> tuple[int | None, int | None]:
    grid = model_inputs.get("image_grid_thw")
    if not isinstance(grid, Sequence):
        tolist = getattr(grid, "tolist", None)
        if callable(tolist):
            grid = tolist()
    if isinstance(grid, Sequence) and grid:
        first = grid[0]
        if isinstance(first, Sequence) and len(first) >= 3:
            try:
                height = int(first[1])
                width = int(first[2])
            except (TypeError, ValueError):
                return None, None
            return width, height
    pixel_values = model_inputs.get("pixel_values")
    shape = getattr(pixel_values, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 4:
        try:
            height = int(shape[-2])
            width = int(shape[-1])
        except (TypeError, ValueError):
            return None, None
        return width, height
    return None, None


def _make_apc_manager() -> Any | None:
    """Create APC manager when mlx-vlm exposes it."""

    try:
        from mlx_vlm.apc import APCManager
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None

    return APCManager()


def _make_vision_feature_cache(max_entries: int) -> Any | None:
    """Create the upstream vision feature cache when available."""

    try:
        from mlx_vlm.vision_cache import VisionFeatureCache
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None
    return VisionFeatureCache(max_size=max_entries)


@dataclass
class _VisionCacheSnapshot:
    """Stored prompt-embedding state for repeated immutable image inputs."""

    prompt_signature: str
    inputs_embeds: Any
    prompt_kwargs: dict[str, Any]
    cache_bytes: int


class _VlmVisionFeatureCache:
    """Thin wrapper around upstream ``VisionFeatureCache`` with stats."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        cache: Any | None = None,
        budget_manager: CacheBudgetManager | None = None,
    ) -> None:
        self._cache = cache or _make_vision_feature_cache(max_entries)
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._budget_manager = budget_manager
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bytes = 0
        if self._budget_manager is not None:
            self._budget_manager.register(self)  # type: ignore[arg-type]

    def get(self, cache_key: str, prompt_signature: str) -> _VisionCacheSnapshot | None:
        if self._cache is None:
            self._misses += 1
            return None
        cached = self._cache.get(cache_key)
        if cached is None or cached.prompt_signature != prompt_signature:
            self._misses += 1
            return None
        self._hits += 1
        return cached

    def put(
        self,
        cache_key: str,
        prompt_signature: str,
        *,
        inputs_embeds: Any,
        prompt_kwargs: dict[str, Any],
    ) -> int:
        if self._cache is None:
            return 0
        cache_bytes = _estimate_array_bytes(inputs_embeds)
        if cache_bytes <= 0 or cache_bytes > self._max_bytes:
            return 0
        before = len(self._cache)
        self._cache.put(
            cache_key,
            _VisionCacheSnapshot(
                prompt_signature=prompt_signature,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=dict(prompt_kwargs),
                cache_bytes=cache_bytes,
            ),
        )
        after = len(self._cache)
        if after == before and cache_key not in self._cache:
            return 0
        if after == before and before >= self._max_entries:
            self._evictions += 1
        self._recompute_bytes()
        if self._budget_manager is not None:
            self._budget_manager.enforce()
        return cache_bytes

    def clear(self) -> None:
        if self._cache is not None:
            self._cache.clear()
        self._bytes = 0

    def evict_oldest(self) -> bool:
        if self._cache is None:
            return False
        internal = getattr(self._cache, "_cache", None)
        if not isinstance(internal, dict) or not internal:
            return False
        oldest_key = next(iter(internal))
        internal.pop(oldest_key, None)
        self._evictions += 1
        self._recompute_bytes()
        return True

    def _recompute_bytes(self) -> None:
        if self._cache is None:
            self._bytes = 0
            return
        internal = getattr(self._cache, "_cache", None)
        if isinstance(internal, dict):
            self._bytes = sum(
                getattr(value, "cache_bytes", 0) for value in internal.values()
            )

    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def total_bytes(self) -> int:
        return self._bytes

    @property
    def total_entries(self) -> int:
        if self._cache is None:
            return 0
        internal = getattr(self._cache, "_cache", None)
        return len(internal) if isinstance(internal, dict) else 0

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def entries(self) -> int:
        if self._cache is None:
            return 0
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


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
    Requests are generated individually; this engine does not use the text
    backend's ``mlx_lm.BatchGenerator`` continuous-batching path.

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
        self._last_vision_encoder_ms: float = 0.0
        self._last_embedding_ms: float = 0.0

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

    def _build_prompt_tokens(self, prompt: str) -> list[int]:
        """Tokenize a prepared VLM prompt string."""

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("VLM processor does not expose a tokenizer")

        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return list(tokens)

    @staticmethod
    def _validate_token_limits(
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
            backend="vlm",
            modality="vlm",
            apc_mode="direct",
            peak_memory_bytes=int(getattr(result, "peak_memory", 0) or 0) or None,
        )

    def complete_many(
        self,
        requests: Sequence[ChatCompletionRequest],
    ) -> list[ChatCompletionResponse]:
        """Generate a batch of VLM chat completions sequentially."""

        return [self.complete_chat(request) for request in requests]

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
            backend="vlm",
            modality="vlm",
            apc_mode="direct",
            peak_memory_bytes=int(getattr(final_response, "peak_memory", 0) or 0)
            if final_response is not None
            else None,
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
            "vision_encoder_ms": self._last_vision_encoder_ms,
            "embedding_ms": self._last_embedding_ms,
        }


@dataclass
class _ScheduledVlmRequest:
    request: ChatCompletionRequest
    stream: bool
    full_prompt_tokens: list[int]
    prompt_tokens: list[int]
    prompt: str
    prompt_kwargs: dict[str, Any]
    image_paths: tuple[str, ...]
    apc_cache_key: tuple[Any, ...]
    cached_prompt: CachedPrompt | None
    apc_extra_hash: int
    vision_feature_cache_hit: bool
    vision_feature_cache_bytes: int | None
    image_width: int | None
    image_height: int | None
    cached_tokens: int = 0
    prompt_batch_size: int = 0
    decode_batch_size: int = 0
    uid: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    rendered_text: str = ""
    stage: str = "pending"
    cancelled_stage: str | None = None
    timing: RequestTiming = field(default_factory=RequestTiming)


class _PreparationCancelled(Exception):
    """Raised when queued VLM request is cancelled during prompt prep."""


class VlmContinuousBatchScheduler:
    """Continuously admit autoregressive VLM requests into ``mlx_vlm``."""

    def __init__(
        self,
        engine: MlxVlmEngine,
        sink: BatchEventSink,
        *,
        prompt_concurrency: int = 4,
        decode_concurrency: int = 4,
        prefill_step_size: int = 256,
        prompt_cache_store: PromptCacheStore | None = None,
        request_cancelled: Callable[[str], bool] | None = None,
        cache_budget_manager: CacheBudgetManager | None = None,
        configured_prompt_batch_size: int | None = None,
        configured_decode_batch_size: int | None = None,
        apc_cache_max_entries: int = 32,
        apc_cache_max_bytes: int = 8 * 1024 * 1024,
        vision_feature_cache_max_entries: int = 20,
        vision_feature_cache_max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._engine = engine
        self._sink = sink
        self._prompt_concurrency = prompt_concurrency
        self._decode_concurrency = decode_concurrency
        self._prefill_step_size = prefill_step_size
        self._configured_prompt_batch_size = (
            configured_prompt_batch_size or prompt_concurrency
        )
        self._configured_decode_batch_size = (
            configured_decode_batch_size or decode_concurrency
        )
        self._prompt_cache_store = prompt_cache_store or PromptCacheStore(
            name="vlm_apc",
            max_entries=apc_cache_max_entries,
            max_bytes=apc_cache_max_bytes,
            budget_manager=cache_budget_manager,
        )
        self._request_cancelled = request_cancelled
        self._cache_budget_manager = cache_budget_manager
        self._apc_manager = _make_apc_manager()
        self._vision_feature_cache = _VlmVisionFeatureCache(
            max_entries=vision_feature_cache_max_entries,
            max_bytes=vision_feature_cache_max_bytes,
            budget_manager=cache_budget_manager,
        )
        self._batch_generator_cls: type[Any] | None = None
        self._generator: Any | None = None
        self._generator_stats_ctx: Any | None = None
        self._generator_stats: Any | None = None
        self._generator_sampling_key: tuple[float, float] | None = None
        self._pending: deque[_ScheduledVlmRequest | ChatCompletionRequest] = deque()
        self._active: dict[int, _ScheduledVlmRequest] = {}
        self._preparing: set[str] = set()
        self._cancelled_preparing: set[str] = set()
        self._worker_cancellation_count = 0
        self._worker_error_count = 0
        self._last_tick_latency_ms = 0
        self._arbitration_delay_ms = 0

    def submit(self, request: ChatCompletionRequest, stream: bool) -> bool:
        if request.model != self._engine.model_id:
            self._sink.emit_error(
                request.request_id,
                "INVALID_REQUEST",
                f"requested model '{request.model}' does not match loaded model '{self._engine.model_id}'",
            )
            self._worker_error_count += 1
            return False

        self._pending.append(request)
        return True

    def cancel(self, request_id: str) -> bool:
        for pending in list(self._pending):
            if pending.request_id != request_id:
                continue
            self._pending.remove(pending)
            self._worker_cancellation_count += 1
            self._sink.emit_response(
                ChatCompletionResponse(
                    request_id=request_id,
                    model=self._engine.model_id,
                    text="",
                    finish_reason="cancelled",
                    prompt_tokens=0,
                    completion_tokens=0,
                    configured_prompt_batch_size=self._configured_prompt_batch_size,
                    configured_decode_batch_size=self._configured_decode_batch_size,
                    backend="vlm",
                    modality="vlm",
                    apc_mode="apc_manager"
                    if self._apc_manager is not None
                    else "disabled",
                    scheduler_stage="cancelled",
                    cancellation_stage="pending",
                    completion_time_ms=0,
                    worker_cancellation_count=self._worker_cancellation_count,
                    worker_error_count=self._worker_error_count,
                )
            )
            return True

        for uid, job in list(self._active.items()):
            if job.request.request_id != request_id:
                continue
            self._active.pop(uid, None)
            job.cancelled_stage = job.stage
            job.stage = "cancelled"
            job.timing.completed_at = time.perf_counter()
            self._worker_cancellation_count += 1
            self._sink.emit_response(
                ChatCompletionResponse(
                    request_id=job.request.request_id,
                    model=self._engine.model_id,
                    text=job.rendered_text,
                    finish_reason="cancelled",
                    prompt_tokens=len(job.full_prompt_tokens),
                    completion_tokens=len(job.generated_tokens),
                    prompt_cache_hit=(
                        job.cached_prompt is not None or job.cached_tokens > 0
                    ),
                    cached_tokens=(
                        len(job.cached_prompt.tokens)
                        if job.cached_prompt is not None
                        else job.cached_tokens or None
                    ),
                    active_batch_cache_bytes=self._active_batch_cache_bytes(),
                    prompt_batch_size=job.prompt_batch_size or None,
                    decode_batch_size=job.decode_batch_size or None,
                    configured_prompt_batch_size=self._configured_prompt_batch_size,
                    configured_decode_batch_size=self._configured_decode_batch_size,
                    image_count=len(job.image_paths),
                    backend="vlm",
                    modality="vlm",
                    apc_mode="apc_manager"
                    if self._apc_manager is not None
                    else "disabled",
                    scheduler_stage=job.stage,
                    cancellation_stage=job.cancelled_stage,
                    queue_time_ms=_duration_ms(
                        job.timing.admitted_at, job.timing.decode_started_at
                    ),
                    prefill_time_ms=_duration_ms(
                        job.timing.admitted_at, job.timing.decode_started_at
                    ),
                    ttft_ms=_duration_ms(
                        job.timing.admitted_at, job.timing.first_token_at
                    ),
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
                    vision_feature_cache_hit=job.vision_feature_cache_hit,
                    vision_feature_cache_bytes=job.vision_feature_cache_bytes,
                    vision_feature_cache_entries=self._vision_feature_cache.total_entries,
                    vision_feature_cache_evictions=self._vision_feature_cache.evictions,
                    vision_encoder_latency_ms=max(
                        1, math.ceil(self._engine._last_vision_encoder_ms)
                    ),
                    embedding_latency_ms=max(
                        1, math.ceil(self._engine._last_embedding_ms)
                    ),
                    prompt_cache_entries=self._prompt_cache_store.total_entries,
                    prompt_cache_evictions=self._prompt_cache_store.stats_snapshot.evictions,
                    peak_memory_bytes=self._peak_memory_bytes(),
                    image_width=job.image_width,
                    image_height=job.image_height,
                )
            )
            self._maybe_cancel_generator(uid)
            return True

        if request_id in self._preparing:
            self._cancelled_preparing.add(request_id)
            self._worker_cancellation_count += 1
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
                    "vlm", self._active_batch_cache_bytes()
                )

        if self._generator is not None and not self._active and not self._pending:
            self.close()

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
            self._generator_sampling_key = None
            self._generator_stats_ctx = None
            self._generator_stats = None
        if self._cache_budget_manager is not None:
            self._cache_budget_manager.set_active_batch_bytes("vlm", 0)

    def _step_generator(self) -> None:
        prompt_responses, responses = self._generator.next()
        if prompt_responses:
            for progress in prompt_responses:
                job = self._active.get(progress.uid)
                if job is None:
                    continue
                job.stage = "prompt_processing"
                if (
                    job.cached_prompt is None
                    and getattr(progress, "prompt_cache", None) is not None
                ):
                    job.cached_prompt = CachedPrompt(
                        tokens=tuple(job.full_prompt_tokens),
                        prompt_cache=copy.deepcopy(list(progress.prompt_cache)),
                        cache_bytes=0,
                        match_kind="exact",
                    )
                job.cached_tokens = max(
                    job.cached_tokens,
                    int(getattr(progress, "cached_tokens", 0) or 0),
                )
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
            token = getattr(response, "token", None)
            if token is not None:
                job.generated_tokens.append(int(token))
            tokenizer = getattr(self._engine.processor, "tokenizer", None)
            if tokenizer is not None and job.generated_tokens:
                decoded = tokenizer.decode(job.generated_tokens)
                delta_text = (
                    decoded[len(job.rendered_text) :]
                    if decoded.startswith(job.rendered_text)
                    else decoded
                )
            else:
                delta_text = getattr(response, "text", "") or ""
            if delta_text:
                if job.stream:
                    self._sink.emit_delta(job.request.request_id, delta_text)
                if job.timing.first_token_at is None:
                    job.timing.first_token_at = time.perf_counter()
                job.rendered_text += delta_text

            finish_reason = getattr(response, "finish_reason", None)
            if finish_reason is None:
                continue

            self._finish(job, response, finish_reason)
            self._active.pop(response.uid, None)

    def _admit_pending(self) -> None:
        if not self._pending:
            return

        if len(self._active) >= self._decode_concurrency:
            return

        if self._generator is None:
            sampling_key = self._next_sampling_key()
            if sampling_key is None:
                return
            self._generator_sampling_key = sampling_key
            self._generator = self._make_batch_generator(sampling_key)
        elif self._generator_sampling_key is None:
            self._generator_sampling_key = self._next_sampling_key()
        elif self._active:
            sampling_key = self._next_sampling_key()
            if sampling_key is None:
                return
            if sampling_key != self._generator_sampling_key:
                return

        batch: list[_ScheduledVlmRequest] = []
        pending = deque()
        while self._pending:
            if len(self._active) + len(batch) >= self._decode_concurrency:
                break
            request = self._pending.popleft()
            if self._request_sampling_key(request) != self._generator_sampling_key:
                pending.append(request)
                continue
            request_id = request.request_id
            self._preparing.add(request_id)
            try:
                batch.append(
                    self._prepare_request(
                        request,
                        should_cancel=lambda request_id=request_id: (
                            request_id in self._cancelled_preparing
                            or (
                                self._request_cancelled is not None
                                and self._request_cancelled(request_id)
                            )
                        ),
                    )
                )
            except _PreparationCancelled:
                self._emit_preparing_cancelled(request)
            except Exception as exc:
                self._worker_error_count += 1
                self._sink.emit_error(request.request_id, "INVALID_REQUEST", str(exc))
            finally:
                self._preparing.discard(request_id)
                self._cancelled_preparing.discard(request_id)
            if len(batch) >= self._prompt_concurrency:
                break

        pending.extend(self._pending)
        self._pending = pending

        if not batch:
            if not self._active:
                self.close()
            return

        prompts: list[list[int]] = []
        max_tokens = [job.request.max_tokens for job in batch]
        prompt_kwargs = [job.prompt_kwargs for job in batch]
        insert_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "prompt_kwargs": prompt_kwargs,
        }
        for job in batch:
            if job.cached_prompt is None:
                prompts.append(job.prompt_tokens)
                continue
            cached_prefix_tokens = list(job.cached_prompt.tokens)
            prompts.append(job.prompt_tokens[len(cached_prefix_tokens) :])
        uids = self._generator.insert(prompts, **insert_kwargs)
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
                "vlm", self._active_batch_cache_bytes()
            )

    def _prepare_request(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> _ScheduledVlmRequest:
        def check_cancelled() -> None:
            if should_cancel is not None and should_cancel():
                raise _PreparationCancelled()

        check_cancelled()
        chat_messages, image_paths = self._engine._extract_chat_and_image(request)
        check_cancelled()
        prompt = self._engine._build_prompt_str(
            chat_messages, num_images=len(image_paths)
        )
        full_prompt_tokens = self._engine._build_prompt_tokens(prompt)
        self._engine._validate_token_limits(request, full_prompt_tokens)
        from mlx_vlm.utils import prepare_inputs

        check_cancelled()
        model_inputs = prepare_inputs(
            self._engine.processor,
            images=list(image_paths) if image_paths else None,
            prompts=prompt,
            padding=True,
            return_tensors="mlx",
        )
        check_cancelled()
        embedding_kwargs = {
            key: value
            for key, value in model_inputs.items()
            if key not in {"input_ids", "pixel_values", "attention_mask"}
            and value is not None
        }
        apc_image_hash = _image_apc_hash(
            image_paths,
            model_inputs.get("pixel_values"),
        )
        image_width, image_height = _image_dimensions(model_inputs)
        prompt_signature = hashlib.sha256(
            f"{self._engine.model_id}|{prompt}|{tuple(full_prompt_tokens)}".encode(
                "utf-8"
            )
        ).hexdigest()
        apc_cache_key = _vlm_apc_cache_key(
            model_id=self._engine.model_id,
            processor=self._engine.processor,
            prompt_tokens=full_prompt_tokens,
            image_paths=image_paths,
            apc_image_hash=apc_image_hash,
        )
        vision_cache_key = "|".join(_image_cache_key(image_paths))
        cached_embedding = (
            self._vision_feature_cache.get(vision_cache_key, prompt_signature)
            if image_paths
            else None
        )
        vision_feature_cache_hit = cached_embedding is not None
        cached_prompt = self._prompt_cache_store.lookup(
            full_prompt_tokens,
            cache_key=apc_cache_key,
        )
        if cached_embedding is not None:
            prompt_kwargs = dict(cached_embedding.prompt_kwargs)
            inputs_embeds = cached_embedding.inputs_embeds
        else:
            embedding_started = time.perf_counter()
            embedding_output = self._engine.model.get_input_embeddings(
                model_inputs["input_ids"],
                model_inputs.get("pixel_values"),
                mask=model_inputs.get("attention_mask"),
                **embedding_kwargs,
            )
            self._engine._last_vision_encoder_ms = (
                time.perf_counter() - embedding_started
            ) * 1000.0
            check_cancelled()
            if hasattr(embedding_output, "to_dict"):
                prompt_kwargs = {
                    key: value
                    for key, value in embedding_output.to_dict().items()
                    if key != "inputs_embeds" and value is not None
                }
            else:
                prompt_kwargs = {
                    key: value
                    for key, value in vars(embedding_output).items()
                    if key != "inputs_embeds" and value is not None
                }
            inputs_embeds = getattr(embedding_output, "inputs_embeds", None)
            if inputs_embeds is None:
                raise RuntimeError(
                    "VLM input embedding output did not include inputs_embeds"
                )
            self._engine._last_embedding_ms = self._engine._last_vision_encoder_ms
        if cached_prompt is not None:
            prompt_kwargs.setdefault(
                "prompt_cache", copy.deepcopy(cached_prompt.prompt_cache)
            )
            prompt_kwargs.setdefault("all_tokens", list(cached_prompt.tokens))
        prompt_kwargs["inputs_embeds"] = inputs_embeds
        prompt_kwargs["_apc_image_hash"] = apc_image_hash
        prompt_kwargs["_apc_tenant"] = (
            f"{self._engine.model_id}:{type(self._engine.processor).__name__}:{prompt_signature[:16]}"
        )
        vision_feature_cache_bytes: int | None = None
        if image_paths and not vision_feature_cache_hit:
            vision_feature_cache_bytes = self._vision_feature_cache.put(
                vision_cache_key,
                prompt_signature,
                inputs_embeds=inputs_embeds,
                prompt_kwargs={
                    key: value
                    for key, value in prompt_kwargs.items()
                    if key != "inputs_embeds"
                },
            )
        return _ScheduledVlmRequest(
            request=request,
            stream=request.stream,
            full_prompt_tokens=full_prompt_tokens,
            prompt_tokens=list(full_prompt_tokens),
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            image_paths=tuple(image_paths),
            apc_cache_key=apc_cache_key,
            cached_prompt=cached_prompt,
            apc_extra_hash=apc_image_hash,
            vision_feature_cache_hit=vision_feature_cache_hit,
            vision_feature_cache_bytes=vision_feature_cache_bytes,
            image_width=image_width,
            image_height=image_height,
        )

    def _emit_preparing_cancelled(self, request: ChatCompletionRequest) -> None:
        self._sink.emit_response(
            ChatCompletionResponse(
                request_id=request.request_id,
                model=self._engine.model_id,
                text="",
                finish_reason="cancelled",
                prompt_tokens=0,
                completion_tokens=0,
                configured_prompt_batch_size=self._configured_prompt_batch_size,
                configured_decode_batch_size=self._configured_decode_batch_size,
                backend="vlm",
                modality="vlm",
                apc_mode="apc_manager" if self._apc_manager is not None else "disabled",
                scheduler_stage="cancelled",
                cancellation_stage="prompt_processing",
                worker_cancellation_count=self._worker_cancellation_count,
                worker_error_count=self._worker_error_count,
            )
        )

    def _request_sampling_key(
        self, request: ChatCompletionRequest
    ) -> tuple[float, float]:
        return (float(request.temperature), float(request.top_p))

    def _next_sampling_key(self) -> tuple[float, float] | None:
        for request in self._pending:
            return self._request_sampling_key(request)
        return None

    def _make_batch_generator(self, sampling_key: tuple[float, float]) -> Any:
        batch_generator_cls = self._get_batch_generator_cls()
        self._engine.initialize()
        generator = batch_generator_cls(
            self._engine.model.language_model,
            self._engine.processor,
            sampler=_make_sampler(*sampling_key),
            completion_batch_size=self._decode_concurrency,
            prefill_batch_size=self._prompt_concurrency,
            prefill_step_size=self._prefill_step_size,
            apc_manager=self._apc_manager,
        )
        stats_factory = getattr(generator, "stats", None)
        if callable(stats_factory):
            self._generator_stats_ctx = stats_factory()
            with suppress(Exception):
                self._generator_stats = self._generator_stats_ctx.__enter__()
        return generator

    def _maybe_cancel_generator(self, uid: int) -> None:
        if self._generator is None:
            return

        remove = getattr(self._generator, "remove", None)
        if remove is None:
            raise RuntimeError("BatchGenerator does not provide remove(uid)")
        remove(uid)

    def _finish(
        self,
        job: _ScheduledVlmRequest,
        response: Any,
        finish_reason: str,
    ) -> None:
        tokenizer = getattr(self._engine.processor, "tokenizer", None)
        if tokenizer is not None and job.generated_tokens:
            job.rendered_text = tokenizer.decode(job.generated_tokens)
        elif not job.stream:
            job.rendered_text = job.rendered_text or getattr(response, "text", "")
        if job.generated_tokens and job.timing.first_token_at is None:
            now = time.perf_counter()
            job.timing.first_token_at = now
            if job.timing.decode_started_at is None:
                job.timing.decode_started_at = now
        job.stage = "completed"
        job.timing.completed_at = time.perf_counter()
        if self._cache_budget_manager is not None:
            self._cache_budget_manager.set_active_batch_bytes(
                "vlm", self._active_batch_cache_bytes()
            )

        if getattr(response, "prompt_cache", None) is not None:
            sequence_tokens = list(
                getattr(response, "all_tokens", None)
                or job.full_prompt_tokens + job.generated_tokens
            )
            cache_value = _prompt_cache_for_prompt_tokens(
                response.prompt_cache,
                sequence_tokens,
                job.full_prompt_tokens,
            )
            if cache_value is not None:
                self._prompt_cache_store.remember(
                    job.full_prompt_tokens,
                    cache_value,
                    cache_key=job.apc_cache_key,
                )

        self._sink.emit_response(
            ChatCompletionResponse(
                request_id=job.request.request_id,
                model=self._engine.model_id,
                text=job.rendered_text,
                finish_reason=finish_reason,
                prompt_tokens=int(
                    getattr(response, "prompt_tokens", len(job.full_prompt_tokens))
                ),
                completion_tokens=int(
                    getattr(
                        response,
                        "generation_tokens",
                        getattr(
                            response, "completion_tokens", len(job.generated_tokens)
                        ),
                    )
                ),
                image_count=len(job.image_paths),
                image_preprocess_latency_ms=max(
                    1, math.ceil(self._engine._last_image_preprocess_ms)
                ),
                prompt_template_latency_ms=max(
                    1, math.ceil(self._engine._last_prompt_template_ms)
                ),
                prompt_cache_hit=(
                    job.cached_prompt is not None or job.cached_tokens > 0
                ),
                cached_tokens=(
                    len(job.cached_prompt.tokens)
                    if job.cached_prompt is not None
                    else job.cached_tokens or None
                ),
                active_batch_cache_bytes=self._active_batch_cache_bytes(),
                prompt_batch_size=job.prompt_batch_size or None,
                decode_batch_size=job.decode_batch_size or None,
                configured_prompt_batch_size=self._configured_prompt_batch_size,
                configured_decode_batch_size=self._configured_decode_batch_size,
                backend="vlm",
                modality="vlm",
                apc_mode="apc_manager" if self._apc_manager is not None else "disabled",
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
                vision_feature_cache_hit=job.vision_feature_cache_hit,
                vision_feature_cache_bytes=job.vision_feature_cache_bytes,
                vision_feature_cache_entries=self._vision_feature_cache.total_entries,
                vision_feature_cache_evictions=self._vision_feature_cache.evictions,
                vision_encoder_latency_ms=max(
                    1, math.ceil(self._engine._last_vision_encoder_ms)
                ),
                embedding_latency_ms=max(1, math.ceil(self._engine._last_embedding_ms)),
                prompt_cache_entries=self._prompt_cache_store.total_entries,
                prompt_cache_evictions=self._prompt_cache_store.stats_snapshot.evictions,
                peak_memory_bytes=self._peak_memory_bytes(),
                image_width=job.image_width,
                image_height=job.image_height,
            )
        )

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

    def _get_batch_generator_cls(self) -> type[Any]:
        if self._batch_generator_cls is not None:
            return self._batch_generator_cls

        batch_generator_cls = _load_batch_generator_cls(continuous=True)
        self._batch_generator_cls = batch_generator_cls
        return batch_generator_cls


def validate_vlm_continuous_batching_backend() -> None:
    """Validate the installed VLM batching API before worker readiness."""

    _validate_tested_minor_version("mlx-vlm", expected_minor_prefix="0.6.")
    _load_batch_generator_cls(continuous=True)
    if _make_apc_manager() is None:
        raise RuntimeError(
            "mlx_vlm.apc.APCManager is unavailable; cannot batch VLM requests"
        )
    if _make_vision_feature_cache(1) is None:
        raise RuntimeError(
            "mlx_vlm.vision_cache.VisionFeatureCache is unavailable; cannot batch VLM requests"
        )


def _load_batch_generator_cls(*, continuous: bool) -> type[Any]:
    module = importlib.import_module("mlx_vlm.generate")
    batch_generator_cls = getattr(module, "BatchGenerator", None)
    if batch_generator_cls is None:
        raise RuntimeError(
            "mlx_vlm.generate.BatchGenerator is unavailable; cannot batch VLM requests"
        )
    _validate_batch_generator_contract(batch_generator_cls, continuous=continuous)
    return batch_generator_cls


def _validate_batch_generator_contract(
    batch_generator_cls: type[Any], *, continuous: bool
) -> None:
    """Fail fast when the installed mlx-vlm batching API is incompatible."""

    required_parameters = {
        "insert": {"prompts", "max_tokens", "prompt_kwargs"},
        "next": set(),
        "close": set(),
        "stats": set(),
        "remove": {"uid"},
    }
    if continuous:
        required_parameters["__init__"] = {
            "model",
            "processor",
            "sampler",
            "completion_batch_size",
            "prefill_batch_size",
            "prefill_step_size",
        }

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
            "installed mlx-vlm BatchGenerator is incompatible with the runtime: "
            f"missing {details}"
        )


def _validate_tested_minor_version(
    package_name: str, *, expected_minor_prefix: str
) -> None:
    version = importlib.metadata.version(package_name)
    if not version.startswith(expected_minor_prefix):
        raise RuntimeError(
            f"{package_name} {version} is outside tested minor range "
            f"{expected_minor_prefix}x"
        )
