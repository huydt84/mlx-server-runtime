"""Model-independent continuous scheduler for native MLX."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, replace
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
        states: list[_ScheduledState],
        *,
        scheduler_round: int,
        cache_coordinator: CacheCoordinator,
        timings: dict[str, int],
    ) -> list[_ScheduledState]:
        """Return states in preferred admission order."""


@dataclass
class _ScheduledState:
    work: SchedulableRequest
    arrival_order: int
    prompt_cursor: int = 0
    cache_handle: str | None = None
    cache_length: int = 0
    reused_prompt_tokens: int = 0
    last_token_id: int | None = None
    generated_tokens: int = 0
    generation_complete: bool = False
    cancel_requested: bool = False
    finish_requested: bool = False
    cancel_requested_at: float | None = None
    running_started_at: float | None = None
    cached_probe: object | None = None


@dataclass(frozen=True)
class _SelectedWork:
    state: _ScheduledState
    request: ExecutionRequest


@dataclass
class _SchedulerInFlight:
    """One owner-thread MLX step retained across scheduler ticks."""

    selected: list[_SelectedWork]
    timings: dict[str, int]
    started: float
    executor_step: Any


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
        return states


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
        execution_mode: str = "serial",
        terminal_token_ids: tuple[int, ...] = (),
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
        if execution_mode not in {"serial", "overlap"}:
            raise ValueError("native-mlx execution mode must be serial or overlap")
        if execution_mode == "overlap":
            for name in (
                "prepare_batch",
                "execute_prepared",
                "dispatch_batch",
                "launch_batch",
                "resolve_batch",
                "discard_batch",
                "commit_dispatched",
                "device_next_tokens",
            ):
                if not callable(getattr(executor, name, None)):
                    raise TypeError(
                        f"overlap executor is missing required method {name}"
                    )
        self._execution_mode = execution_mode
        self._tick_impl = (
            self._tick_overlap if execution_mode == "overlap" else self._tick_serial
        )
        self._terminal_token_ids = frozenset(int(value) for value in terminal_token_ids)
        self._record_generated = (
            self._record_generated_overlap
            if execution_mode == "overlap"
            else self._record_generated_serial
        )
        self._profiler = profiler
        self._waiting: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._running: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._pending_events: list[SchedulerEvent] = []
        self._arrival_counter = 0
        self._scheduler_round = 0
        self._last_step_prefilled = False
        self._in_flight: _SchedulerInFlight | None = None
        self._kv_admission_waits = 0
        self._kv_prefix_reclaimed_pages = 0

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
        if self._in_flight is not None and any(
            item.request.request_id == request_id for item in self._in_flight.selected
        ):
            state = self._running.get(request_id) or self._waiting.get(request_id)
            if state is not None:
                state.finish_requested = True
            return
        state = self._running.pop(request_id, None)
        if state is None:
            state = self._waiting.pop(request_id, None)
        if state is not None:
            self._cache_coordinator.release(state.cache_handle)

    def tick(self) -> tuple[SchedulerEvent, ...]:
        return self._tick_impl()

    def _tick_serial(self) -> tuple[SchedulerEvent, ...]:
        """Run the existing synchronous scheduler behavior."""

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

    def _tick_overlap(self) -> tuple[SchedulerEvent, ...]:
        """Build step N+1, resolve step N, then return N's events.

        N+1 is normally built before N resolves: a decode step reads N's
        sampled tokens while they are still on the device, and a prefill step
        does not depend on N at all. The GPU then never waits for the CPU to
        read N's tokens and build N+1's graph. Every step, prefill included,
        is evaluated asynchronously and resolved on the following tick.

        A decode step that follows a decode step is enqueued at once. Any
        other chained step is enqueued at the start of the next tick, after
        N's events are returned: enqueueing blocks while a long step fills
        Metal's command queue, which would otherwise hold back N's tokens.
        A step that verifies proposed tokens needs N's result, so it waits for
        the next tick for the same reason.
        """

        if self._in_flight is not None:
            with suppress(Exception):
                self._executor.launch_batch(self._in_flight.executor_step)  # type: ignore[attr-defined]
        events = self._pending_events
        self._pending_events = []
        late_events: list[SchedulerEvent] = []
        self._scheduler_round += 1
        protected = self._in_flight_request_ids()
        self._reap_cancelled(events, protected_request_ids=protected)
        self._reap_finished(protected_request_ids=protected)

        resolved: (
            tuple[
                _SchedulerInFlight,
                StepResult,
                list[tuple[_SelectedWork, StepRequestResult]],
            ]
            | None
        ) = None
        immediate: (
            tuple[list[_SelectedWork], StepResult, dict[str, int], float] | None
        ) = None
        chained = False
        if self._in_flight is not None:
            step = self._in_flight
            self._in_flight = None
            chained, immediate = self._dispatch_chained(step, late_events)
            try:
                result = self._executor.resolve_batch(step.executor_step)  # type: ignore[attr-defined]
            except BatchExecutionError as exc:
                self._emit_failures(step.selected, events, exc.code, str(exc))
                self._mark_finished(step.selected)
            except Exception as exc:
                self._emit_failures(step.selected, events, "WORKER_ERROR", str(exc))
                self._mark_finished(step.selected)
            else:
                applied = self._advance_step_state(
                    step.selected,
                    result,
                    events,
                    step.timings,
                )
                resolved = (step, result, applied)
            protected = self._in_flight_request_ids()
            self._reap_cancelled(events, protected_request_ids=protected)
            self._reap_finished(protected_request_ids=protected)

        if not chained and not (resolved is not None and self._drafts_next()):
            select_started = time.perf_counter()
            timings = self._new_timings()
            selected = self._select_work(timings)
            timings["scheduler_select_ms"] = _elapsed_ms(select_started)
            self._record_selection(selected, select_started, timings)
            if selected:
                started = time.perf_counter()
                batch = ExecutionBatch(
                    requests=tuple(item.request for item in selected)
                )
                try:
                    prepared = self._executor.prepare_batch(batch)  # type: ignore[attr-defined]
                    if prepared.has_work:
                        executor_step = self._executor.dispatch_batch(prepared)  # type: ignore[attr-defined]
                        self._in_flight = _SchedulerInFlight(
                            selected=selected,
                            timings=timings,
                            started=started,
                            executor_step=executor_step,
                        )
                    else:
                        result = self._executor.execute_prepared(prepared)  # type: ignore[attr-defined]
                        immediate = (selected, result, timings, started)
                except BatchExecutionError as exc:
                    self._emit_failures(selected, events, exc.code, str(exc))
                    self._mark_finished(selected)
                except Exception as exc:
                    self._emit_failures(selected, events, "WORKER_ERROR", str(exc))
                    self._mark_finished(selected)
        if resolved is not None:
            step, result, applied = resolved
            self._emit_step_result(
                step.selected,
                result,
                applied,
                events,
                step.timings,
                step.started,
            )
        if immediate is not None:
            immediate_selected, result, immediate_timings, immediate_started = immediate
            self._apply_step_result(
                immediate_selected,
                result,
                events,
                immediate_timings,
                immediate_started,
            )
        events.extend(late_events)
        self._reap_finished(protected_request_ids=self._in_flight_request_ids())
        return tuple(events)

    def _dispatch_chained(
        self,
        step: _SchedulerInFlight,
        late_events: list[SchedulerEvent],
    ) -> tuple[
        bool, tuple[list[_SelectedWork], StepResult, dict[str, int], float] | None
    ]:
        """Dispatch the step that follows ``step`` before ``step`` resolves.

        Returns whether this tick's next step was handled here, plus a result
        to apply when the chained step had no model work. Rows known to stop
        after ``step`` (cancelled, finished, or at ``max_tokens``) are left
        out. A row that ``step`` ends with a terminal token still runs in a
        chained decode step; that extra token is dropped when it resolves, and
        the request's cache is released rather than published.
        """

        started = time.perf_counter()
        timings = self._new_timings()
        chained = self._chained_work(step, timings)
        if chained is None:
            return False, None
        selected, device_rows = chained
        pending = None
        if device_rows:
            request_ids = tuple(
                item.request.request_id for item in selected[:device_rows]
            )
            pending = self._executor.device_next_tokens(  # type: ignore[attr-defined]
                step.executor_step, request_ids
            )
            if pending is None:
                return False, None
        try:
            self._executor.commit_dispatched(step.executor_step)  # type: ignore[attr-defined]
        except Exception:
            return False, None
        batch = ExecutionBatch(
            requests=tuple(item.request for item in selected),
            pending_token_ids=pending,
        )
        try:
            prepared = self._executor.prepare_batch(batch)  # type: ignore[attr-defined]
            timings["scheduler_select_ms"] = _elapsed_ms(started)
            self._record_selection(selected, started, timings)
            if not prepared.has_work:
                result = self._executor.execute_prepared(prepared)  # type: ignore[attr-defined]
                return True, (selected, result, timings, started)
            launch = all(
                item.request.phase == "decode" for item in step.selected + selected
            )
            executor_step = self._executor.dispatch_batch(prepared, launch=launch)  # type: ignore[attr-defined]
        except BatchExecutionError as exc:
            self._emit_failures(selected, late_events, exc.code, str(exc))
            self._mark_finished(selected)
            return True, None
        except Exception as exc:
            self._emit_failures(selected, late_events, "WORKER_ERROR", str(exc))
            self._mark_finished(selected)
            return True, None
        self._in_flight = _SchedulerInFlight(
            selected=selected,
            timings=timings,
            started=started,
            executor_step=executor_step,
        )
        return True, None

    def _chained_work(
        self, step: _SchedulerInFlight, timings: dict[str, int]
    ) -> tuple[list[_SelectedWork], int] | None:
        """Select the step after ``step`` while ``step`` is still running.

        Follows the same decode/prefill alternation as ``_select_work``. A
        decode selection lists first the rows whose input token is ``step``'s
        unresolved sample, and returns how many there are. Returns ``None``
        when the next step depends on ``step``'s resolved state: another
        prefill chunk after a prefill step, or a step that verifies proposed
        tokens or follows one.
        """

        if any(item.request.draft_tokens for item in step.selected):
            return None
        in_step = {item.request.request_id for item in step.selected}
        continuing = [item for item in step.selected if self._feeds_next_decode(item)]
        resolved = [
            state
            for request_id, state in self._running.items()
            if request_id not in in_step and self._decode_ready(state)
        ]
        has_decode = bool(continuing or resolved)
        if all(item.request.phase == "decode" for item in step.selected):
            prefill_states = self._prefill_candidates()
            if prefill_states:
                selected = self._select_prefill(
                    prefill_states, decode_waiting=has_decode, timings=timings
                )
                return (selected, 0) if selected else None
        if not has_decode:
            return None
        if (
            len(continuing) == 1
            and not resolved
            and self._draft_tokens(continuing[0].state, unresolved=1)
        ):
            return None
        self._last_step_prefilled = False
        selected = [
            _SelectedWork(
                state=item.state,
                request=ExecutionRequest(
                    request_id=item.request.request_id,
                    phase="decode",
                    token_ids=(0,),
                    positions=(item.request.positions[-1] + 1,),
                    cache_handle=item.state.cache_handle,
                    sampling=item.state.work.sampling,
                ),
            )
            for item in continuing
        ]
        selected.extend(self._decode_work(state) for state in resolved)
        return selected, len(continuing)

    def _feeds_next_decode(self, item: _SelectedWork) -> bool:
        """Return whether ``item``'s sample is the input of its next decode."""

        state = item.state
        if not self._continues_after_step(state):
            return False
        if item.request.phase == "decode":
            return True
        return state.prompt_cursor + len(item.request.token_ids) == len(
            state.work.prompt_token_ids
        )

    @staticmethod
    def _decode_ready(state: _ScheduledState) -> bool:
        return (
            not state.cancel_requested
            and not state.finish_requested
            and not state.generation_complete
            and state.last_token_id is not None
        )

    @staticmethod
    def _continues_after_step(state: _ScheduledState) -> bool:
        """Return whether a row can still need a token after its pending one."""

        return (
            not state.cancel_requested
            and not state.finish_requested
            and not state.generation_complete
            and state.generated_tokens + 1 < state.work.max_tokens
        )

    def idle(self) -> bool:
        return not self._waiting and not self._running and self._in_flight is None

    def benchmark_reset(
        self, *, clear_cache: bool, reset_counters: bool
    ) -> dict[str, Any]:
        """Reset idle scheduler/cache state without rebuilding the executor."""

        if not self.idle():
            raise RuntimeError("benchmark reset requires an idle scheduler")
        self._pending_events.clear()
        self._last_step_prefilled = False
        if reset_counters:
            self._arrival_counter = 0
            self._scheduler_round = 0
        return self._cache_coordinator.reset(
            clear_cache=clear_cache,
            reset_counters=reset_counters,
        )

    def close(self) -> None:
        if self._in_flight is not None:
            try:
                self._executor.discard_batch(self._in_flight.executor_step)  # type: ignore[attr-defined]
            finally:
                self._in_flight = None
        for request_id in tuple(self._waiting) + tuple(self._running):
            self.finish(request_id)

    @staticmethod
    def _new_timings() -> dict[str, int]:
        return {
            "scheduler_cache_probe_ms": 0,
            "scheduler_cache_acquire_ms": 0,
            "scheduler_cache_publish_ms": 0,
            "scheduler_apply_ms": 0,
        }

    def _record_selection(
        self,
        selected: list[_SelectedWork],
        select_started: float,
        timings: dict[str, int],
    ) -> None:
        if self._profiler is None:
            return
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

    def _in_flight_request_ids(self) -> frozenset[str]:
        if self._in_flight is None:
            return frozenset()
        return frozenset(item.request.request_id for item in self._in_flight.selected)

    @staticmethod
    def _mark_finished(selected: list[_SelectedWork]) -> None:
        for item in selected:
            item.state.finish_requested = True

    def _select_work(self, timings: dict[str, int]) -> list[_SelectedWork]:
        """Select one phase-homogeneous step.

        The executor pads every row to the longest row, so a decode row sharing a
        step with a prefill chunk would run as many positions as the chunk.
        Decode and prefill therefore never share a step; when both have work,
        steps alternate between them.
        """

        decode_work = [
            self._decode_work(state)
            for state in self._running.values()
            if self._decode_ready(state)
        ]
        prefill_states = self._prefill_candidates()
        if decode_work and (not prefill_states or self._last_step_prefilled):
            self._last_step_prefilled = False
            if len(decode_work) == 1 and not prefill_states:
                return [self._with_draft(decode_work[0])]
            return decode_work
        selected = self._select_prefill(
            prefill_states, decode_waiting=bool(decode_work), timings=timings
        )
        if not selected and decode_work:
            self._last_step_prefilled = False
            return decode_work
        return selected

    @staticmethod
    def _decode_work(state: _ScheduledState) -> _SelectedWork:
        return _SelectedWork(
            state=state,
            request=ExecutionRequest(
                request_id=state.work.request_id,
                phase="decode",
                token_ids=(int(state.last_token_id),),  # type: ignore[arg-type]
                positions=(state.cache_length,),
                cache_handle=state.cache_handle,
                sampling=state.work.sampling,
            ),
        )

    def _drafts_next(self) -> bool:
        """Return whether the next step verifies proposed tokens."""

        ready = [state for state in self._running.values() if self._decode_ready(state)]
        return (
            len(ready) == 1
            and not self._prefill_candidates()
            and self._draft_tokens(ready[0]) > 0
        )

    def _with_draft(self, work: _SelectedWork) -> _SelectedWork:
        """Let a request that decodes alone verify proposed tokens."""

        count = self._draft_tokens(work.state)
        if not count:
            return work
        return replace(work, request=replace(work.request, draft_tokens=count))

    def _draft_tokens(self, state: _ScheduledState, *, unresolved: int = 0) -> int:
        """Return how many proposed tokens a request's next step verifies.

        Args:
            state: The only decoding request.
            unresolved: Its tokens that are still being computed.
        """

        choose = getattr(self._executor, "draft_tokens", None)
        if not callable(choose):
            return 0
        remaining = state.work.max_tokens - state.generated_tokens - unresolved
        return int(choose(state.work.request_id, state.cache_handle, remaining - 1))

    def _select_prefill(
        self,
        prefill_states: list[_ScheduledState],
        *,
        decode_waiting: bool,
        timings: dict[str, int],
    ) -> list[_SelectedWork]:
        """Admit and select the next prefill chunks."""

        prefill_limit = (
            1
            if decode_waiting and self._prioritize_decode
            else self._prefill_batch_size
        )
        selected: list[_SelectedWork] = []
        prefill_states = self._policy.order(
            prefill_states,
            scheduler_round=self._scheduler_round,
            cache_coordinator=self._cache_coordinator,
            timings=timings,
        )
        kv_available = self._kv_available_pages()
        window_available = self._window_available_pages()
        admission_blocked = False
        for state in prefill_states[:prefill_limit]:
            request_id = state.work.request_id
            if state.cache_handle is None:
                if admission_blocked:
                    continue
                probe = state.cached_probe
                if probe is None:
                    probe_started = time.perf_counter()
                    probe = self._cache_coordinator.probe(state.work.prompt_token_ids)
                    timings["scheduler_cache_probe_ms"] += _elapsed_ms(probe_started)
                if kv_available is not None:
                    pages, page_size = kv_available
                    needed = self._admission_pages(state, probe, page_size)
                    if needed > pages:
                        freed = self._reclaim_prefix_pages(needed - pages)
                        if freed:
                            probe = self._cache_coordinator.probe(
                                state.work.prompt_token_ids
                            )
                            pages += freed
                            needed = self._admission_pages(state, probe, page_size)
                    window_needed = self._window_pages(state)
                    if self._running and (
                        needed > pages or window_needed > window_available
                    ):
                        state.cached_probe = probe
                        admission_blocked = True
                        self._kv_admission_waits += 1
                        continue
                    kv_available = (pages - needed, page_size)
                    window_available -= window_needed
                self._waiting.pop(request_id, None)
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
                state.reused_prompt_tokens = admission.reused_tokens
                state.prompt_cursor = state.reused_prompt_tokens
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
        self._last_step_prefilled = True
        return selected

    def _kv_available_pages(self) -> tuple[int, int] | None:
        """Return free KV pages not promised to running requests.

        Each running request may still grow to its prompt plus ``max_tokens``;
        those pages stay promised so admitting new work never makes a running
        request fail for lack of KV capacity.
        """

        capacity = getattr(self._cache_coordinator, "kv_admission_capacity", None)
        if not callable(capacity):
            return None
        resolved = capacity()
        if resolved is None:
            return None
        free_pages, page_size = resolved
        promised = 0
        for state in self._running.values():
            if state.cancel_requested or state.finish_requested:
                continue
            final = len(state.work.prompt_token_ids) + max(0, state.work.max_tokens)
            promised += max(
                0,
                math.ceil(final / page_size)
                - math.ceil(state.cache_length / page_size),
            )
        return free_pages - promised, page_size

    def _window_available_pages(self) -> int:
        """Return windowed KV pages not promised to running requests.

        Only windowed cache families have them; a running request may use its
        whole promise at any time because its window slides.
        """

        total = getattr(self._cache_coordinator, "kv_window_pages", None)
        if not callable(total):
            return 0
        return int(total()) - sum(
            self._window_pages(state)
            for state in self._running.values()
            if not state.cancel_requested and not state.finish_requested
        )

    def _window_pages(self, state: _ScheduledState) -> int:
        """Return the most windowed KV pages a request holds at once."""

        pages_for = getattr(self._cache_coordinator, "kv_window_pages_for", None)
        if not callable(pages_for):
            return 0
        final = len(state.work.prompt_token_ids) + max(0, state.work.max_tokens)
        return int(pages_for(final))

    def _reclaim_prefix_pages(self, pages: int) -> int:
        """Free ``pages`` KV pages by evicting unpinned prefix-cache entries."""

        reclaim = getattr(self._cache_coordinator, "reclaim_pages", None)
        if not callable(reclaim):
            return 0
        freed = int(reclaim(pages))
        self._kv_prefix_reclaimed_pages += freed
        return freed

    @staticmethod
    def _admission_pages(state: _ScheduledState, probe: object, page_size: int) -> int:
        """Return pages a waiting request needs through its last token."""

        final = len(state.work.prompt_token_ids) + max(0, state.work.max_tokens)
        reused = int(getattr(probe, "matched_pages", 0) or 0)
        return max(0, math.ceil(final / page_size) - reused) + 1

    def _prefill_candidates(self) -> list[_ScheduledState]:
        """Return running and waiting states that still have prompt tokens."""

        states = [
            state
            for state in self._running.values()
            if not state.cancel_requested
            and not state.finish_requested
            and state.prompt_cursor < len(state.work.prompt_token_ids)
        ]
        states.extend(
            state
            for state in self._waiting.values()
            if not state.cancel_requested and not state.finish_requested
        )
        return states

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

        self._apply_step_result(selected, result, events, timings, started)

    def _apply_step_result(
        self,
        selected: list[_SelectedWork],
        result: StepResult,
        events: list[SchedulerEvent],
        timings: dict[str, int],
        started: float,
    ) -> None:
        """Apply one resolved executor result without dispatching model work."""

        applied = self._advance_step_state(selected, result, events, timings)
        self._emit_step_result(
            selected,
            result,
            applied,
            events,
            timings,
            started,
        )

    def _advance_step_state(
        self,
        selected: list[_SelectedWork],
        result: StepResult,
        events: list[SchedulerEvent],
        timings: dict[str, int],
    ) -> list[tuple[_SelectedWork, StepRequestResult]]:
        """Apply only state required to construct the next execution batch."""

        results = {item.request_id: item for item in result.results}
        applied: list[tuple[_SelectedWork, StepRequestResult]] = []
        apply_started = time.perf_counter()
        for selected_work in selected:
            state = selected_work.state
            request = selected_work.request
            if (
                state.cancel_requested
                or state.finish_requested
                or state.generation_complete
            ):
                continue
            item = results.get(request.request_id)
            if item is None:
                self._emit_failure(
                    state,
                    events,
                    "INCOMPLETE_EXECUTION_RESULT",
                    "missing executor result",
                )
                state.finish_requested = True
                continue
            if item.phase != request.phase:
                self._emit_failure(
                    state,
                    events,
                    "INVALID_EXECUTION_RESULT",
                    "executor result phase does not match scheduled work",
                )
                state.finish_requested = True
                continue
            if item.error_code is not None:
                self._emit_failure(
                    state,
                    events,
                    item.error_code,
                    item.error_message or "native execution request failed",
                )
                state.finish_requested = True
                continue
            state.cache_length = item.cache_length
            item = self._until_terminal(item)
            for token_id in item.accepted_token_ids:
                self._mark_generated(state, token_id)
            if request.phase == "prefill":
                state.prompt_cursor += len(request.token_ids)
                if state.prompt_cursor == len(state.work.prompt_token_ids):
                    state.last_token_id = item.next_token_id
                    self._mark_generated(state, item.next_token_id)
            else:
                state.last_token_id = item.next_token_id
                self._mark_generated(state, item.next_token_id)
            applied.append((selected_work, item))
        timings["scheduler_apply_ms"] += _elapsed_ms(apply_started)
        return applied

    def _until_terminal(self, item: StepRequestResult) -> StepRequestResult:
        """Drop verified tokens that follow a terminal token."""

        for index, token_id in enumerate(item.accepted_token_ids):
            if token_id in self._terminal_token_ids:
                return replace(
                    item,
                    accepted_token_ids=item.accepted_token_ids[:index],
                    next_token_id=token_id,
                )
        return item

    def _emit_step_result(
        self,
        selected: list[_SelectedWork],
        result: StepResult,
        applied: list[tuple[_SelectedWork, StepRequestResult]],
        events: list[SchedulerEvent],
        timings: dict[str, int],
        started: float,
    ) -> None:
        """Materialize events and metrics after the next MLX step is dispatched."""

        if self._profiler is not None:
            for selected_work in selected:
                request = selected_work.request
                shared = {
                    "forward_mode": result.forward_mode.value,
                    "physical_batch_size": result.physical_batch_size,
                    "model_forward_count": result.model_forward_count,
                    "execution_mode": self._execution_mode,
                }
                for component, stage, metric in (
                    ("executor", "batch_prepare", "executor_prepare_ms"),
                    ("cache", "reserve", "executor_reserve_ms"),
                    ("model", "forward_dispatch", "executor_forward_ms"),
                    ("sampling", "sample", "executor_sample_ms"),
                    ("mlx", "async_dispatch", "executor_dispatch_ms"),
                    ("mlx", "synchronize_eval", "executor_eval_ms"),
                    ("cache", "commit", "executor_commit_ms"),
                ):
                    started_ns = None
                    if stage == "async_dispatch":
                        started_ns = (
                            int(result.metrics.get("executor_dispatch_started_ns", 0))
                            or None
                        )
                    elif stage == "synchronize_eval":
                        started_ns = (
                            int(result.metrics.get("executor_eval_started_ns", 0))
                            or None
                        )
                    self._profiler.record(
                        request.request_id,
                        component,
                        stage,
                        started_ns=started_ns,
                        duration_us=int(result.metrics.get(metric, 0)) * 1_000,
                        phase=request.phase,
                        details=shared,
                    )
        for selected_work, item in applied:
            state = selected_work.state
            request = selected_work.request
            if request.phase == "prefill":
                self._emit_prefill_result(
                    state,
                    request,
                    item,
                    result,
                    events,
                    timings,
                )
            else:
                self._emit_decode_result(state, item, result, events)

        for phase in ("decode", "prefill"):
            phase_work = [
                item
                for item in selected
                if item.request.phase == phase
                and not item.state.cancel_requested
                and not item.state.finish_requested
            ]
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
                        "execution_mode": self._execution_mode,
                        **self._cache_coordinator.metrics(),
                        **timings,
                        **result.metrics,
                    },
                )

    def _emit_prefill_result(
        self,
        state: _ScheduledState,
        request: ExecutionRequest,
        item: StepRequestResult,
        result: StepResult,
        events: list[SchedulerEvent],
        timings: dict[str, int],
    ) -> None:
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
                    "cached_tokens": state.reused_prompt_tokens,
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
            events.append(
                SchedulerEvent(
                    kind="token",
                    request_id=state.work.request_id,
                    token_id=item.next_token_id,
                    cache_length=item.cache_length,
                    phase="prefill",
                )
            )

    def _emit_decode_result(
        self,
        state: _ScheduledState,
        item: StepRequestResult,
        result: StepResult,
        events: list[SchedulerEvent],
    ) -> None:
        metrics: dict[str, Any] | None = {
            "step_time_ms": result.step_time_ms,
            "batch_size": sum(
                result_item.phase == "decode" for result_item in result.results
            ),
        }
        for token_id in (*item.accepted_token_ids, item.next_token_id):
            events.append(
                SchedulerEvent(
                    kind="token",
                    request_id=state.work.request_id,
                    token_id=token_id,
                    cache_length=item.cache_length,
                    phase="decode",
                    metrics=metrics,
                )
            )
            metrics = None

    def _mark_generated(
        self,
        state: _ScheduledState,
        token_id: int | None,
    ) -> None:
        """Prevent known-terminal rows from entering the delayed next step."""

        state.generated_tokens += 1
        self._record_generated(state, token_id)

    @staticmethod
    def _record_generated_serial(
        state: _ScheduledState,
        token_id: int | None,
    ) -> None:
        del state, token_id

    def _record_generated_overlap(
        self,
        state: _ScheduledState,
        token_id: int | None,
    ) -> None:
        state.generation_complete = (
            state.generated_tokens >= state.work.max_tokens
            or token_id in self._terminal_token_ids
        )

    def _reap_cancelled(
        self,
        events: list[SchedulerEvent],
        *,
        protected_request_ids: frozenset[str] = frozenset(),
    ) -> None:
        for state in tuple(self._waiting.values()) + tuple(self._running.values()):
            if (
                state.cancel_requested
                and state.work.request_id not in protected_request_ids
            ):
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

    def _reap_finished(
        self,
        *,
        protected_request_ids: frozenset[str] = frozenset(),
    ) -> None:
        for state in tuple(self._waiting.values()) + tuple(self._running.values()):
            request_id = state.work.request_id
            if state.finish_requested and request_id not in protected_request_ids:
                self._waiting.pop(request_id, None)
                self._running.pop(request_id, None)
                self._cache_coordinator.release(state.cache_handle)

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
                    "kv_admission_waits": self._kv_admission_waits,
                    "kv_prefix_reclaimed_pages": self._kv_prefix_reclaimed_pages,
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
