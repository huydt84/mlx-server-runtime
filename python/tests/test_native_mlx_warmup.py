"""Startup warmup covers every compiled decode, prefill, and sampling path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mlx_worker.native_mlx.interfaces import SchedulableRequest, SchedulerEvent
from mlx_worker.native_mlx.paged_decode import splits_for, warmup_key_counts
from mlx_worker.native_mlx.runtime import NativeRuntime


@pytest.mark.parametrize("kv_heads", [1, 2, 4, 8, 16, 32])
def test_warmup_key_counts_reach_every_single_row_split_count(kv_heads: int) -> None:
    counts = warmup_key_counts(kv_heads)

    splits = [splits_for(1, kv_heads, keys) for keys in counts]
    most = splits_for(1, kv_heads, 1 << 20)
    assert splits == [1 << index for index in range(len(counts))]
    assert splits[-1] == most
    for keys, count in zip(counts, splits, strict=True):
        assert splits_for(1, kv_heads, keys - 8) == count
        assert splits_for(1, kv_heads, keys + 8) == count


def test_more_rows_use_no_split_count_single_rows_miss() -> None:
    single = {splits_for(1, 8, keys) for keys in warmup_key_counts(8)}
    for rows in range(2, 9):
        for keys in (1, 40, 100, 300, 5000):
            assert splits_for(rows, 8, keys) in single


@dataclass
class _CompletingScheduler:
    """Emits one token per running request each tick."""

    running: list[SchedulableRequest] = field(default_factory=list)
    batches: list[tuple[str, ...]] = field(default_factory=list)
    submitted: list[SchedulableRequest] = field(default_factory=list)

    def submit(self, request: SchedulableRequest) -> None:
        self.running.append(request)
        self.submitted.append(request)

    def tick(self) -> tuple[SchedulerEvent, ...]:
        self.batches.append(tuple(item.request_id for item in self.running))
        return tuple(
            SchedulerEvent(
                kind="token",
                request_id=item.request_id,
                token_id=5,
                cache_length=len(item.prompt_token_ids) + 1,
                phase="decode",
                metrics={"step_time_ms": 1, "batch_size": len(self.running)},
            )
            for item in tuple(self.running)
        )

    def finish(self, request_id: str) -> None:
        self.running = [item for item in self.running if item.request_id != request_id]

    def idle(self) -> bool:
        return not self.running

    def close(self) -> None:
        self.running.clear()


class _WordTokenizer:
    """Three template tokens plus one token per word."""

    eos_token_id = 0

    def apply_chat_template(
        self, messages: Any, *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        return [1, 2, 3] + [4] * len(messages[0]["content"].split())

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "x" * len(token_ids)


@dataclass
class _CaptureProfiler:
    captured: list[str] = field(default_factory=list)

    def begin_metal_capture(self, request_id: str) -> None:
        self.captured.append(request_id)

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def _runtime(
    scheduler: _CompletingScheduler, **kwargs: Any
) -> tuple[NativeRuntime, _CaptureProfiler]:
    profiler = _CaptureProfiler()
    tokenizer = _WordTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
        profiler=profiler,
        **kwargs,
    )
    return runtime, profiler


def test_warmup_decodes_at_each_length_then_runs_a_sampled_pair() -> None:
    scheduler = _CompletingScheduler()
    runtime, profiler = _runtime(scheduler, warmup_prompt_lengths=(96, 48, 96))

    runtime.warmup()

    prompts = {
        item.request_id: len(item.prompt_token_ids) for item in scheduler.submitted
    }
    assert prompts == {
        "warmup-0": 48,
        "warmup-1": 96,
        "warmup-greedy": 48,
        "warmup-sampled": 56,
    }
    sampling = {item.request_id: item.sampling for item in scheduler.submitted}
    assert sampling["warmup-0"].temperature == 0.0
    assert sampling["warmup-sampled"].temperature > 0.0
    assert sampling["warmup-sampled"].top_p < 1.0
    assert ("warmup-greedy", "warmup-sampled") in scheduler.batches
    assert all(len(batch) == 1 for batch in scheduler.batches[:4])
    assert scheduler.batches.count(("warmup-0",)) == 4
    assert runtime.idle()
    assert runtime.last_warmup_latency_ms >= 1
    assert profiler.captured == []


def test_warmup_skips_lengths_the_kv_pool_cannot_hold() -> None:
    scheduler = _CompletingScheduler()
    runtime, _ = _runtime(
        scheduler, warmup_prompt_lengths=(48, 96, 192), max_total_tokens=120
    )

    runtime.warmup()

    assert [item.request_id for item in scheduler.submitted] == [
        "warmup-0",
        "warmup-1",
        "warmup-greedy",
        "warmup-sampled",
    ]


def test_metal_capture_resumes_after_warmup() -> None:
    scheduler = _CompletingScheduler()
    runtime, profiler = _runtime(scheduler)
    runtime.warmup()

    request = runtime._warmup_chat(
        "public",
        3,
        max_prompt_tokens=16,
        max_total_tokens_per_request=32,
    )
    runtime.submit(request)

    assert profiler.captured == ["public"]
