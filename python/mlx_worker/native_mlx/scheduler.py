"""Model-independent continuous scheduler for native MLX."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from .interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    NativeMlxExecutor,
    SchedulableRequest,
    SchedulerEvent,
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


class NativeContinuousScheduler:
    """Select token work and dispatch shared executor batch primitives."""

    def __init__(
        self,
        executor: NativeMlxExecutor,
        *,
        prefill_step_size: int = 256,
    ) -> None:
        if prefill_step_size <= 0:
            raise ValueError("native-mlx prefill chunk size must be positive")
        self._executor = executor
        self._prefill_step_size = int(prefill_step_size)
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
            self._executor.release(state.cache_handle)

    def tick(self) -> tuple[SchedulerEvent, ...]:
        events = self._pending_events
        self._pending_events = []
        self._reap_cancelled(events)
        if any(
            state.last_token_id is not None and not state.cancel_requested
            for state in self._running.values()
        ):
            self._run_decode(events)
            self._reap_cancelled(events)
        if self._waiting or any(
            state.prompt_cursor < len(state.work.prompt_token_ids)
            and not state.cancel_requested
            for state in self._running.values()
        ):
            self._run_prefill(events)
            self._reap_cancelled(events)
        return tuple(events)

    def idle(self) -> bool:
        return not self._waiting and not self._running

    def close(self) -> None:
        for request_id in tuple(self._waiting) + tuple(self._running):
            self.finish(request_id)

    def _run_prefill(self, events: list[SchedulerEvent]) -> None:
        started = time.perf_counter()
        states = [
            state
            for state in self._running.values()
            if not state.cancel_requested
            and state.prompt_cursor < len(state.work.prompt_token_ids)
        ]
        states.extend(
            state for state in self._waiting.values() if not state.cancel_requested
        )
        if not states:
            return
        requests: list[ExecutionRequest] = []
        chunk_lengths: dict[str, int] = {}
        for state in states:
            request_id = state.work.request_id
            if state.cache_handle is None:
                self._waiting.pop(request_id, None)
                state.cache_handle = self._executor.create_cache(request_id)
                state.running_started_at = time.perf_counter()
                self._running[request_id] = state
            start = state.prompt_cursor
            end = min(len(state.work.prompt_token_ids), start + self._prefill_step_size)
            tokens = state.work.prompt_token_ids[start:end]
            chunk_lengths[request_id] = len(tokens)
            requests.append(
                ExecutionRequest(
                    request_id=request_id,
                    token_ids=tokens,
                    positions=tuple(range(start, end)),
                    cache_handle=state.cache_handle,
                    sampling=state.work.sampling,
                )
            )
        try:
            result = self._executor.prefill_batch(
                ExecutionBatch(phase="prefill", requests=tuple(requests))
            )
        except Exception as exc:
            self._emit_failures(states, events, str(exc))
            return
        results = {item.request_id: item for item in result.results}
        for state in states:
            item = results.get(state.work.request_id)
            if item is None:
                self._emit_failure(state, events, "missing prefill result")
                continue
            state.prompt_cursor += chunk_lengths[state.work.request_id]
            state.cache_length = item.cache_length
            events.append(
                SchedulerEvent(
                    kind="prefill_progress",
                    request_id=state.work.request_id,
                    cache_length=item.cache_length,
                    phase="prefill",
                    metrics={
                        "step_time_ms": result.step_time_ms,
                        "batch_size": len(states),
                        "scheduled_tokens": chunk_lengths[state.work.request_id],
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
        self._emit_metrics(
            events, "prefill", len(states), sum(chunk_lengths.values()), started
        )

    def _run_decode(self, events: list[SchedulerEvent]) -> None:
        started = time.perf_counter()
        states = [
            state
            for state in self._running.values()
            if not state.cancel_requested and state.last_token_id is not None
        ]
        if not states:
            return
        batch = ExecutionBatch(
            phase="decode",
            requests=tuple(
                ExecutionRequest(
                    request_id=state.work.request_id,
                    token_ids=(int(state.last_token_id),),
                    positions=(state.cache_length,),
                    cache_handle=state.cache_handle,
                    sampling=state.work.sampling,
                )
                for state in states
            ),
        )
        try:
            result = self._executor.decode_batch(batch)
        except Exception as exc:
            self._emit_failures(states, events, str(exc))
            return
        results = {item.request_id: item for item in result.results}
        for state in states:
            item = results.get(state.work.request_id)
            if item is None:
                self._emit_failure(state, events, "missing decode result")
                continue
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
                        "batch_size": len(states),
                    },
                )
            )
        self._emit_metrics(events, "decode", len(states), len(states), started)

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
        message: str,
    ) -> None:
        events.append(
            SchedulerEvent(
                kind="execution_error",
                request_id=state.work.request_id,
                error_code="WORKER_ERROR",
                message=message,
            )
        )

    def _emit_failures(
        self,
        states: list[_ScheduledState],
        events: list[SchedulerEvent],
        message: str,
    ) -> None:
        for state in states:
            self._emit_failure(state, events, message)

    def _emit_metrics(
        self,
        events: list[SchedulerEvent],
        phase: str,
        batch_size: int,
        scheduled_tokens: int,
        started: float,
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
                },
            )
        )
