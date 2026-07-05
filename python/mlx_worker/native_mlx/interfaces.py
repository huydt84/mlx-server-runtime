"""Native MLX scheduler and executor boundaries for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

import mlx.core as mx

from .cache import LayerKVCache

if TYPE_CHECKING:
    from ..ipc import ChatCompletionRequest


ExecutionPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class SamplingParams:
    """Executor-visible sampling parameters."""

    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class ForwardBatch:
    """Model-facing metadata for one physical tensor invocation."""

    token_lengths: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    attention_mask: mx.array | str | None
    layer_caches: tuple[LayerKVCache, ...]


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
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    cache_handle: str | None
    sampling: SamplingParams = SamplingParams()


@dataclass(frozen=True)
class ExecutionBatch:
    """Scheduler-owned batch passed into one executor step."""

    phase: ExecutionPhase
    requests: tuple[ExecutionRequest, ...]


@dataclass(frozen=True)
class StepRequestResult:
    """Per-request executor output for one model step."""

    request_id: str
    token_ids: tuple[int, ...]
    cache_handle: str | None
    cache_length: int
    next_token_id: int | None = None
    model_terminal: bool = False
    error_code: str | None = None


@dataclass(frozen=True)
class StepResult:
    """Executor output returned to the scheduler."""

    phase: ExecutionPhase
    results: tuple[StepRequestResult, ...]
    step_time_ms: int


@dataclass(frozen=True)
class NativeBackendOptions:
    """Native backend startup options owned by Python runtime."""

    model: str
    architecture_class: str


class NativeMlxExecutor(Protocol):
    """Model-step executor boundary for native MLX."""

    def load(self, options: NativeBackendOptions) -> None:
        """Load native model resources."""

    def create_cache(self, request_id: str) -> str:
        """Create opaque Python-only cache handle for request."""

    def prefill_batch(self, batch: ExecutionBatch) -> StepResult:
        """Run one prefill step."""

    def decode_batch(self, batch: ExecutionBatch) -> StepResult:
        """Run one decode step."""

    def cache_len(self, cache_handle: str | None) -> int:
        """Return cache length for handle."""

    def release(self, cache_handle: str | None) -> None:
        """Release per-request cache resources."""


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
