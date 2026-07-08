"""Typed native MLX runtime, scheduler, cache, and executor boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

import mlx.core as mx

if TYPE_CHECKING:
    from ..ipc import ChatCompletionRequest


ExecutionPhase = Literal["prefill", "decode"]


class ForwardMode(str, Enum):
    """Physical model-forward shape selected for one scheduler step."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"

    @classmethod
    def from_phases(cls, phases: Sequence[ExecutionPhase]) -> "ForwardMode":
        """Derive one physical forward mode from request-local phases."""

        unique = set(phases)
        if unique == {"prefill"}:
            return cls.PREFILL
        if unique == {"decode"}:
            return cls.DECODE
        if unique == {"prefill", "decode"}:
            return cls.MIXED
        raise ValueError("execution batch must contain prefill or decode work")


@dataclass(frozen=True)
class SamplingParams:
    """Executor-visible sampling parameters."""

    temperature: float = 0.0
    top_p: float = 1.0


class LayerAttentionContext(Protocol):
    """Model-facing attention operation independent of physical KV layout."""

    def append_and_attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: str | None,
    ) -> mx.array:
        """Stage K/V append and attend over committed plus staged history."""


@dataclass(frozen=True)
class ForwardBatch:
    """Model-facing metadata for one physical tensor invocation."""

    forward_mode: ForwardMode
    token_lengths: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    attention_mask: str | None
    layer_attention: tuple[LayerAttentionContext, ...]


class NativeModel(Protocol):
    """Architecture model: token tensors and metadata in, logits out."""

    num_layers: int

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array: ...

    def load_weights(
        self, weights: Sequence[tuple[str, mx.array]], *, strict: bool = True
    ) -> None: ...


@dataclass(frozen=True)
class ExecutionRequest:
    """Executor-visible token work for one request.

    Raw prompts, HTTP state, queue policy, and SSE state stay outside this
    boundary.
    """

    request_id: str
    phase: ExecutionPhase
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    cache_handle: str | None
    sampling: SamplingParams = SamplingParams()


@dataclass(frozen=True)
class ExecutionBatch:
    """Scheduler-owned batch passed into one executor step."""

    requests: tuple[ExecutionRequest, ...]

    @property
    def forward_mode(self) -> ForwardMode:
        """Derive the model-forward mode without duplicating phase state."""

        return ForwardMode.from_phases(
            tuple(request.phase for request in self.requests)
        )


@dataclass(frozen=True)
class StepRequestResult:
    """Per-request executor output for one model step."""

    request_id: str
    phase: ExecutionPhase
    token_ids: tuple[int, ...]
    cache_handle: str | None
    cache_length: int
    next_token_id: int | None = None
    model_terminal: bool = False
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class StepResult:
    """Executor output returned to the scheduler."""

    forward_mode: ForwardMode
    results: tuple[StepRequestResult, ...]
    step_time_ms: int
    physical_batch_size: int
    model_forward_count: int
    metrics: dict[str, Any] = field(default_factory=dict)


class BatchExecutionError(RuntimeError):
    """Non-attributable failure for one physical executor invocation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NativeBackendOptions:
    """Native backend startup options owned by Python runtime."""

    model: str
    architecture_class: str


class NativeMlxExecutor(Protocol):
    """Model-step executor boundary for native MLX."""

    def load(self, options: NativeBackendOptions) -> None:
        """Load native model resources."""

    def execute_batch(self, batch: ExecutionBatch) -> StepResult:
        """Run one homogeneous or mixed model step."""


@dataclass(frozen=True)
class PrefixProbe:
    """Side-effect-free prefix lookup result."""

    matched_tokens: int = 0
    matched_pages: int = 0
    cache_handle: str | None = None


@dataclass(frozen=True)
class CacheAdmission:
    """Scheduler-visible cache acquisition result."""

    cache_handle: str
    cache_length: int
    reused_tokens: int = 0
    reused_pages: int = 0


@dataclass(frozen=True)
class CachePublication:
    """Result of publishing committed prefill state."""

    published_tokens: int = 0
    published_pages: int = 0


class CacheCoordinator(Protocol):
    """Only cache lifecycle surface available to the scheduler."""

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe: ...

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: PrefixProbe | None = None,
    ) -> CacheAdmission: ...

    def publish_committed(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication: ...

    def length(self, cache_handle: str | None) -> int: ...

    def release(self, cache_handle: str | None) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


class NativeMlxDiagnostics(Protocol):
    """Explicit diagnostics surface for parity and tracing."""

    def forward_token_ids(self, token_ids: Sequence[int]) -> Any:
        """Run direct model forward on finalized token IDs."""

    def prefill_then_decode_tokens(
        self, prompt_token_ids: Sequence[int], decode_steps: int
    ) -> tuple[list[int], list[int], int]:
        """Run deterministic prefill plus decode parity helper."""

    def trace_to_mlx_lm(
        self,
        checkpoint: str,
        token_ids: Sequence[int],
        *,
        prompt_fingerprint: str,
        output_dir: Any,
        decode_steps: int,
        tolerance_atol: float = 2e-2,
        tolerance_rtol: float = 2e-2,
        sample_size: int = 8,
        selected_dumps: Sequence[str] = (),
        stop_on_first_divergence: bool = True,
    ) -> Any:
        """Trace native and reference checkpoints through explicit adapter."""


class NativeScheduler(Protocol):
    """Token-level scheduler boundary for native MLX."""

    def submit(self, request: "SchedulableRequest") -> None:
        """Queue typed token work."""

    def cancel(self, request_id: str) -> bool: ...

    def finish(self, request_id: str) -> None:
        """Release scheduler-owned state and cache for a terminal request."""

    def tick(self) -> tuple["SchedulerEvent", ...]: ...

    def idle(self) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RuntimeRequest:
    """Normalized public request owned by the native runtime."""

    request_id: str
    model: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    stop: tuple[str, ...]
    sampling: SamplingParams


@dataclass(frozen=True)
class SchedulableRequest:
    """Model-independent token work accepted by the scheduler."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    enqueued_at: float


SchedulerEventKind = Literal[
    "token", "cancelled", "execution_error", "metrics", "prefill_progress"
]


@dataclass(frozen=True)
class SchedulerEvent:
    """Typed scheduler output consumed by the runtime."""

    kind: SchedulerEventKind
    request_id: str | None = None
    token_id: int | None = None
    cache_length: int | None = None
    phase: ExecutionPhase | None = None
    error_code: str | None = None
    message: str | None = None
    metrics: Any | None = None


@dataclass(frozen=True)
class RuntimeEvent:
    """Typed runtime output transported by the worker."""

    kind: Literal["delta", "response", "error", "metrics"]
    payload: Any


class NativeRuntime(Protocol):
    """Public-request lifecycle seam driven by worker transport."""

    last_warmup_latency_ms: int

    def warmup(self) -> None: ...

    def submit(self, request: ChatCompletionRequest) -> None: ...

    def cancel(self, request_id: str) -> bool: ...

    def tick(self) -> tuple[RuntimeEvent, ...]: ...

    def idle(self) -> bool: ...

    def close(self) -> None: ...


def execution_batch_field_names() -> tuple[str, ...]:
    """Expose field names for boundary tests."""

    return tuple(field.name for field in fields(ExecutionBatch))


def execution_request_field_names() -> tuple[str, ...]:
    """Expose field names for boundary tests."""

    return tuple(field.name for field in fields(ExecutionRequest))
