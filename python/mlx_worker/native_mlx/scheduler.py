"""Model-independent continuous scheduler for native MLX."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .interfaces import (
    BatchExecutionError,
    CacheCoordinator,
    ExecutionBatch,
    ExecutionRequest,
    NativeMlxExecutor,
    SchedulableRequest,
    SchedulerEvent,
    StepRequestResult,
    StepResult,
)


class SchedulingPolicy(str, Enum):
    """Inner waiting-queue policy owned by the native scheduler."""

    FCFS = "fcfs"
    LPM = "lpm"
    LOF = "lof"
    PRIORITY = "priority"


class SchedulerPolicy(Protocol):
    """Policy interface for ordering Python waiting-queue requests."""

    name: SchedulingPolicy

    def order(
        self,
        states: list["_ScheduledState"],
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> list["_ScheduledState"]:
        """Return states in preferred admission order."""


@dataclass
class _ScheduledState:
    work: SchedulableRequest
    arrival_order: int
    prompt_cursor: int = 0
    cache_handle: str | None = None
    cache_length: int = 0
    last_token_id: int | None = None
    cancel_requested: bool = False
    cancel_requested_at: float | None = None
    running_started_at: float | None = None
    cached_probe: object | None = None


@dataclass(frozen=True)
class _SelectedWork:
    state: _ScheduledState
    request: ExecutionRequest


@dataclass(frozen=True)
class _PolicyScore:
    state: _ScheduledState
    score: tuple[int, int, int, int]


class _BasePolicy:
    name = SchedulingPolicy.FCFS

    def order(
        self,
        states: list[_ScheduledState],
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> list[_ScheduledState]:
        del scheduler_round, cache_coordinator, timings
        return sorted(states, key=lambda state: state.arrival_order)


class _ScoredPolicy(_BasePolicy):
    """Shared deterministic ordering for non-FCFS policies."""

    starvation_rounds = 3

    def order(
        self,
        states: list[_ScheduledState],
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> list[_ScheduledState]:
        scored = [
            _PolicyScore(
                state=state,
                score=self._score(
                    state,
                    scheduler_round=scheduler_round,
                    cache_coordinator=cache_coordinator,
                    timings=timings,
                ),
            )
            for state in states
        ]
        return [
            item.state
            for item in sorted(
                scored,
                key=lambda item: item.score,
                reverse=True,
            )
        ]

    def _score(
        self,
        state: _ScheduledState,
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> tuple[int, int, int, int]:
        del cache_coordinator, timings
        age_rounds = max(0, scheduler_round - state.arrival_order)
        starvation_boost = int(age_rounds >= self.starvation_rounds)
        return (
            starvation_boost,
            self._policy_value(state),
            age_rounds,
            -state.arrival_order,
        )

    def _policy_value(self, state: _ScheduledState) -> int:
        del state
        return 0


class _LpmPolicy(_ScoredPolicy):
    name = SchedulingPolicy.LPM

    def _score(
        self,
        state: _ScheduledState,
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> tuple[int, int, int, int]:
        probe_started = time.perf_counter()
        probe = cache_coordinator.probe(state.work.prompt_token_ids)
        timings["scheduler_cache_probe_ms"] += _elapsed_ms(probe_started)
        state.cached_probe = probe
        age_rounds = max(0, scheduler_round - state.arrival_order)
        starvation_boost = int(age_rounds >= self.starvation_rounds)
        return (
            starvation_boost,
            int(getattr(probe, "matched_tokens", 0)),
            age_rounds,
            -state.arrival_order,
        )


class _LofPolicy(_ScoredPolicy):
    name = SchedulingPolicy.LOF

    def _policy_value(self, state: _ScheduledState) -> int:
        return int(state.work.max_tokens)


class _PriorityPolicy(_ScoredPolicy):
    name = SchedulingPolicy.PRIORITY

    def _policy_value(self, state: _ScheduledState) -> int:
        return int(state.work.priority)


def _policy_from_name(name: str) -> SchedulerPolicy:
    normalized = name.strip().lower()
    if normalized == SchedulingPolicy.FCFS.value:
        return _BasePolicy()
    if normalized == SchedulingPolicy.LPM.value:
        return _LpmPolicy()
    if normalized == SchedulingPolicy.LOF.value:
        return _LofPolicy()
    if normalized == SchedulingPolicy.PRIORITY.value:
        return _PriorityPolicy()
    raise ValueError("native-mlx scheduling policy must be fcfs, lpm, lof, or priority")


class NativeContinuousScheduler:
    """Select token work and dispatch shared executor batch primitives."""

    def __init__(
        self,
        executor: NativeMlxExecutor,
        cache_coordinator: CacheCoordinator,
        *,
        prefill_batch_size: int = 4,
        prefill_step_size: int = 256,
        prioritize_decode: bool = True,
        scheduling_policy: str = "fcfs",
        profiler: Any | None = None,
    ) -> None:
        if prefill_batch_size <= 0:
            raise ValueError("native-mlx prefill batch size must be positive")
        if prefill_step_size <= 0:
            raise ValueError("native-mlx prefill chunk size must be positive")
        self._executor = executor
        self._cache_coordinator = cache_coordinator
        self._prefill_batch_size = int(prefill_batch_size)
        self._prefill_step_size = int(prefill_step_size)
        self._prioritize_decode = bool(prioritize_decode)
        self._policy = _policy_from_name(scheduling_policy)
        self._profiler = profiler
        self._waiting: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._running: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._pending_events: list[SchedulerEvent] = []
        self._arrival_counter = 0
        self._scheduler_round = 0

    def submit(self, request: SchedulableRequest) -> None:
        if request.request_id in self._waiting or request.request_id in self._running:
            raise ValueError(f"duplicate native request {request.request_id!r}")
        self._arrival_counter += 1
        self._waiting[request.request_id] = _ScheduledState(
            work=request,
            arrival_order=self._arrival_counter,
        )

    def cancel(self, request_id: str) -> bool:
        state = self._waiting.get(request_id) or self._running.get(request_id)
        if state is None:
            return False
        state.cancel_requested = True
        state.cancel_requested_at = time.perf_counter()
        return True

    def finish(self, request_id: str) -> None:
        state = self._running.pop(request_id, None)
        if state is None:
            state = self._waiting.pop(request_id, None)
        if state is not None:
            self._cache_coordinator.release(state.cache_handle)

    def tick(self) -> tuple[SchedulerEvent, ...]:
        events = self._pending_events
        self._pending_events = []
        self._scheduler_round += 1
        self._reap_cancelled(events)
        select_started = time.perf_counter()
        timings: dict[str, int] = {
            "scheduler_cache_probe_ms": 0,
            "scheduler_cache_acquire_ms": 0,
            "scheduler_cache_publish_ms": 0,
            "scheduler_apply_ms": 0,
        }
        selected = self._select_work(timings)
        timings["scheduler_select_ms"] = _elapsed_ms(select_started)
        if self._profiler is not None:
            for item in selected:
                self._profiler.record(
                    item.request.request_id,
                    "scheduler",
                    "select_work",
                    started_ns=int(select_started * 1_000_000_000),
                    duration_us=timings["scheduler_select_ms"] * 1_000,
                    phase=item.request.phase,
                    details={"scheduled_tokens": len(item.request.token_ids)},
                )
        if selected:
            self._run_step(selected, events, timings)
            self._reap_cancelled(events)
        return tuple(events)

    def idle(self) -> bool:
        return not self._waiting and not self._running

    def close(self) -> None:
        for request_id in tuple(self._waiting) + tuple(self._running):
            self.finish(request_id)

    def _select_work(self, timings: dict[str, int]) -> list[_SelectedWork]:
        selected = [
            _SelectedWork(
                state=state,
                request=ExecutionRequest(
                    request_id=state.work.request_id,
                    phase="decode",
                    token_ids=(int(state.last_token_id),),
                    positions=(state.cache_length,),
                    cache_handle=state.cache_handle,
                    sampling=state.work.sampling,
                ),
            )
            for state in self._running.values()
            if not state.cancel_requested and state.last_token_id is not None
        ]
        prefill_states = [
            state
            for state in self._running.values()
            if not state.cancel_requested
            and state.prompt_cursor < len(state.work.prompt_token_ids)
        ]
        prefill_states.extend(
            state for state in self._waiting.values() if not state.cancel_requested
        )
        if selected and self._prioritize_decode:
            return selected
        prefill_states = self._policy.order(
            prefill_states,
            scheduler_round=self._scheduler_round,
            cache_coordinator=self._cache_coordinator,
            timings=timings,
        )
        for state in prefill_states[: self._prefill_batch_size]:
            request_id = state.work.request_id
            if state.cache_handle is None:
                self._waiting.pop(request_id, None)
                probe = state.cached_probe
                if probe is None:
                    probe_started = time.perf_counter()
                    probe = self._cache_coordinator.probe(state.work.prompt_token_ids)
                    timings["scheduler_cache_probe_ms"] += _elapsed_ms(probe_started)
                state.cached_probe = None
                acquire_started = time.perf_counter()
                admission = self._cache_coordinator.acquire(
                    request_id,
                    state.work.prompt_token_ids,
                    probe,  # type: ignore[arg-type]
                )
                timings["scheduler_cache_acquire_ms"] += _elapsed_ms(acquire_started)
                state.cache_handle = admission.cache_handle
                state.cache_length = admission.cache_length
                state.prompt_cursor = admission.reused_tokens
                state.running_started_at = time.perf_counter()
                self._running[request_id] = state
            start = state.prompt_cursor
            end = min(len(state.work.prompt_token_ids), start + self._prefill_step_size)
            tokens = state.work.prompt_token_ids[start:end]
            selected.append(
                _SelectedWork(
                    state=state,
                    request=ExecutionRequest(
                        request_id=request_id,
                        phase="prefill",
                        token_ids=tokens,
                        positions=tuple(range(start, end)),
                        cache_handle=state.cache_handle,
                        sampling=state.work.sampling,
                    ),
                )
            )
        return selected

    def _run_step(
        self,
        selected: list[_SelectedWork],
        events: list[SchedulerEvent],
        timings: dict[str, int],
    ) -> None:
        started = time.perf_counter()
        try:
            result = self._executor.execute_batch(
                ExecutionBatch(
                    requests=tuple(item.request for item in selected),
                )
            )
        except BatchExecutionError as exc:
            self._emit_failures(selected, events, exc.code, str(exc))
            return
        except Exception as exc:
            self._emit_failures(selected, events, "WORKER_ERROR", str(exc))
            return

        if self._profiler is not None:
            for selected_work in selected:
                request = selected_work.request
                shared = {
                    "forward_mode": result.forward_mode.value,
                    "physical_batch_size": result.physical_batch_size,
                    "model_forward_count": result.model_forward_count,
                }
                for component, stage, metric in (
                    ("executor", "batch_prepare", "executor_prepare_ms"),
                    ("cache", "reserve", "executor_reserve_ms"),
                    ("model", "forward_dispatch", "executor_forward_ms"),
                    ("sampling", "sample", "executor_sample_ms"),
                    ("mlx", "synchronize_eval", "executor_eval_ms"),
                    ("cache", "commit", "executor_commit_ms"),
                ):
                    self._profiler.record(
                        request.request_id,
                        component,
                        stage,
                        duration_us=int(result.metrics.get(metric, 0)) * 1_000,
                        phase=request.phase,
                        details=shared,
                    )
        results = {item.request_id: item for item in result.results}
        apply_started = time.perf_counter()
        for selected_work in selected:
            state = selected_work.state
            request = selected_work.request
            item = results.get(request.request_id)
            if item is None:
                self._emit_failure(
                    state,
                    events,
                    "INCOMPLETE_EXECUTION_RESULT",
                    "missing executor result",
                )
                continue
            if item.phase != request.phase:
                self._emit_failure(
                    state,
                    events,
                    "INVALID_EXECUTION_RESULT",
                    "executor result phase does not match scheduled work",
                )
                continue
            if item.error_code is not None:
                self._emit_failure(
                    state,
                    events,
                    item.error_code,
                    item.error_message or "native execution request failed",
                )
                continue
            if request.phase == "prefill":
                self._apply_prefill_result(
                    state,
                    request,
                    item,
                    result,
                    events,
                    timings,
                )
            else:
                self._apply_decode_result(state, item, result, events)
        timings["scheduler_apply_ms"] += _elapsed_ms(apply_started)

        for phase in ("decode", "prefill"):
            phase_work = [item for item in selected if item.request.phase == phase]
            if phase_work:
                self._emit_metrics(
                    events,
                    phase,
                    len(phase_work),
                    sum(len(item.request.token_ids) for item in phase_work),
                    started,
                    execution_metrics={
                        "forward_mode": result.forward_mode.value,
                        "physical_batch_size": result.physical_batch_size,
                        "model_forward_count": result.model_forward_count,
                        **self._cache_coordinator.metrics(),
                        **timings,
                        **result.metrics,
                    },
                )

    def _apply_prefill_result(
        self,
        state: _ScheduledState,
        request: ExecutionRequest,
        item: StepRequestResult,
        result: StepResult,
        events: list[SchedulerEvent],
        timings: dict[str, int],
    ) -> None:
        state.prompt_cursor += len(request.token_ids)
        state.cache_length = item.cache_length
        publish_started = time.perf_counter()
        self._cache_coordinator.publish_committed(
            item.cache_handle or "",
            state.work.prompt_token_ids,
            item.cache_length,
        )
        timings["scheduler_cache_publish_ms"] += _elapsed_ms(publish_started)
        events.append(
            SchedulerEvent(
                kind="prefill_progress",
                request_id=state.work.request_id,
                cache_length=item.cache_length,
                phase="prefill",
                metrics={
                    "step_time_ms": result.step_time_ms,
                    "batch_size": sum(
                        result_item.phase == "prefill" for result_item in result.results
                    ),
                    "scheduled_tokens": len(request.token_ids),
                    "prompt_complete": state.prompt_cursor
                    == len(state.work.prompt_token_ids),
                    "queue_time_ms": max(
                        0,
                        int(
                            (
                                (state.running_started_at or state.work.enqueued_at)
                                - state.work.enqueued_at
                            )
                            * 1000
                        ),
                    ),
                    "scheduler_queue_wait_ms": max(
                        0,
                        int(
                            (
                                (state.running_started_at or state.work.enqueued_at)
                                - state.work.enqueued_at
                            )
                            * 1000
                        ),
                    ),
                },
            )
        )
        if state.prompt_cursor == len(state.work.prompt_token_ids):
            state.last_token_id = item.next_token_id
            events.append(
                SchedulerEvent(
                    kind="token",
                    request_id=state.work.request_id,
                    token_id=item.next_token_id,
                    cache_length=item.cache_length,
                    phase="prefill",
                )
            )

    @staticmethod
    def _apply_decode_result(
        state: _ScheduledState,
        item: StepRequestResult,
        result: StepResult,
        events: list[SchedulerEvent],
    ) -> None:
        state.cache_length = item.cache_length
        state.last_token_id = item.next_token_id
        events.append(
            SchedulerEvent(
                kind="token",
                request_id=state.work.request_id,
                token_id=item.next_token_id,
                cache_length=item.cache_length,
                phase="decode",
                metrics={
                    "step_time_ms": result.step_time_ms,
                    "batch_size": sum(
                        result_item.phase == "decode" for result_item in result.results
                    ),
                },
            )
        )

    def _reap_cancelled(self, events: list[SchedulerEvent]) -> None:
        for state in tuple(self._waiting.values()) + tuple(self._running.values()):
            if state.cancel_requested:
                request_id = state.work.request_id
                stage = (
                    "python_waiting"
                    if request_id in self._waiting
                    else ("decode" if state.last_token_id is not None else "prefill")
                )
                self._waiting.pop(request_id, None)
                self._running.pop(request_id, None)
                self._cache_coordinator.release(state.cache_handle)
                if self._profiler is not None:
                    self._profiler.record(
                        request_id,
                        "scheduler",
                        "terminal",
                        state="cancelled",
                        details={"cancellation_stage": stage},
                    )
                events.append(
                    SchedulerEvent(
                        kind="cancelled",
                        request_id=request_id,
                        metrics={
                            "cancellation_stage": stage,
                            "cancellation_latency_ms": _elapsed_ms(
                                state.cancel_requested_at or time.perf_counter()
                            ),
                        },
                    )
                )

    @staticmethod
    def _emit_failure(
        state: _ScheduledState,
        events: list[SchedulerEvent],
        code: str,
        message: str,
    ) -> None:
        events.append(
            SchedulerEvent(
                kind="execution_error",
                request_id=state.work.request_id,
                error_code=code,
                message=message,
            )
        )

    def _emit_failures(
        self,
        selected: list[_SelectedWork],
        events: list[SchedulerEvent],
        code: str,
        message: str,
    ) -> None:
        for item in selected:
            self._emit_failure(item.state, events, code, message)

    def _emit_metrics(
        self,
        events: list[SchedulerEvent],
        phase: str,
        batch_size: int,
        scheduled_tokens: int,
        started: float,
        execution_metrics: dict[str, object] | None = None,
    ) -> None:
        events.append(
            SchedulerEvent(
                kind="metrics",
                phase=phase,  # type: ignore[arg-type]
                metrics={
                    "phase": phase,
                    "scheduled_tokens": scheduled_tokens,
                    "batch_size": batch_size,
                    "waiting_requests": len(self._waiting),
                    "running_requests": len(self._running),
                    "scheduler_tick_latency_ms": max(
                        1, int((time.perf_counter() - started) * 1000)
                    ),
                    "scheduling_policy": self._policy.name.value,
                    **(execution_metrics or {}),
                },
            )
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
