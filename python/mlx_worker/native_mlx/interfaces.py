"""Native MLX scheduler and executor boundaries for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol


ExecutionPhase = Literal["prefill", "decode"]


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
    max_new_tokens: int
    temperature: float
    top_p: float
    trace_enabled: bool = False


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
    logits: Any | None
    cache_handle: str | None
    cache_length: int
    finished: bool
    next_token_id: int | None = None
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


class NativeScheduler(Protocol):
    """Runtime-owned scheduling boundary for native MLX."""

    def warmup(self) -> None:
        """Validate startup path before readiness."""


def execution_batch_field_names() -> tuple[str, ...]:
    """Expose field names for boundary tests."""

    return tuple(field.name for field in fields(ExecutionBatch))


def execution_request_field_names() -> tuple[str, ...]:
    """Expose field names for boundary tests."""

    return tuple(field.name for field in fields(ExecutionRequest))
