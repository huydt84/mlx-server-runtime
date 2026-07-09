"""Model-independent continuous scheduler for native MLX."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

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


@dataclass
class _ScheduledState:
    work: SchedulableRequest
    prompt_cursor: int = 0
    cache_handle: str | None = None
    cache_length: int = 0
    last_token_id: int | None = None
    cancel_requested: bool = False
    running_started_at: float | None = None


@dataclass(frozen=True)
class _SelectedWork:
    state: _ScheduledState
    request: ExecutionRequest


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
        self._waiting: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._running: OrderedDict[str, _ScheduledState] = OrderedDict()
        self._pending_events: list[SchedulerEvent] = []

    def submit(self, request: SchedulableRequest) -> None:
        if request.request_id in self._waiting or request.request_id in self._running:
            raise ValueError(f"duplicate native request {request.request_id!r}")
        self._waiting[request.request_id] = _ScheduledState(work=request)

    def cancel(self, request_id: str) -> bool:
        state = self._waiting.get(request_id) or self._running.get(request_id)
        if state is None:
            return False
        state.cancel_requested = True
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
        for state in prefill_states[: self._prefill_batch_size]:
            request_id = state.work.request_id
            if state.cache_handle is None:
                self._waiting.pop(request_id, None)
                probe_started = time.perf_counter()
                probe = self._cache_coordinator.probe(state.work.prompt_token_ids)
                timings["scheduler_cache_probe_ms"] += _elapsed_ms(probe_started)
                acquire_started = time.perf_counter()
                admission = self._cache_coordinator.acquire(
                    request_id,
                    state.work.prompt_token_ids,
                    probe,
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
                events.append(
                    SchedulerEvent(
                        kind="cancelled",
                        request_id=state.work.request_id,
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
                    **(execution_metrics or {}),
                },
            )
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
