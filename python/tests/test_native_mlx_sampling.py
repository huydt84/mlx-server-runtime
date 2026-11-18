"""Native temperature and top-p sampling."""

from __future__ import annotations

import time
from collections import Counter

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.attention import DenseReferenceAttentionBackend
from mlx_worker.native_mlx.cache import DenseKVCacheBackend
from mlx_worker.native_mlx.cache_coordinator import NativeCacheCoordinator
from mlx_worker.native_mlx.executor import (
    MlxGenerationExecutor,
    MlxOverlapGenerationExecutor,
)
from mlx_worker.native_mlx.interfaces import SamplingParams, SchedulableRequest
from mlx_worker.native_mlx.prefix_cache import NoPrefixCache
from mlx_worker.native_mlx.sampling import sample_next_tokens
from mlx_worker.native_mlx.scheduler import NativeContinuousScheduler

from .test_native_mlx import _RecordingModel

_PROBS = mx.log(mx.array([0.4, 0.3, 0.2, 0.1], dtype=mx.float32))


def _draws(logits: mx.array, sampling: SamplingParams, count: int) -> Counter[int]:
    tokens = [
        int(
            sample_next_tokens(
                logits[None, :],
                [SamplingParams(sampling.temperature, sampling.top_p, seed)],
                [7],
            )[0].item()
        )
        for seed in range(count)
    ]
    return Counter(tokens)


def test_all_greedy_rows_match_argmax_exactly() -> None:
    logits = mx.array([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]], dtype=mx.float32)

    tokens = sample_next_tokens(logits, [SamplingParams(), SamplingParams()], [0, 0])

    assert tokens.tolist() == mx.argmax(logits, axis=-1).tolist()


def test_a_draw_depends_only_on_seed_and_position() -> None:
    logits = mx.zeros((64,), dtype=mx.float32)
    sampled = SamplingParams(temperature=1.0, seed=1234)
    alone = sample_next_tokens(logits[None, :], [sampled], [5])
    other = mx.arange(64, dtype=mx.float32)
    batched = sample_next_tokens(
        mx.stack([other, logits, other]),
        [SamplingParams(), sampled, SamplingParams(temperature=0.7, seed=9)],
        [3, 5, 11],
    )

    assert int(batched[1].item()) == int(alone[0].item())
    assert int(batched[0].item()) == 63


def test_different_positions_draw_different_tokens() -> None:
    logits = mx.zeros((1, 1000), dtype=mx.float32)
    sampled = [SamplingParams(temperature=1.0, seed=42)]

    tokens = {
        int(sample_next_tokens(logits, sampled, [position])[0].item())
        for position in range(8)
    }

    assert len(tokens) > 1


def test_temperature_sampling_follows_the_distribution() -> None:
    counts = _draws(_PROBS, SamplingParams(temperature=1.0), 2000)

    assert set(counts) == {0, 1, 2, 3}
    assert counts[0] > counts[1] > counts[2] > counts[3]
    assert 0.33 < counts[0] / 2000 < 0.47


def test_top_p_keeps_the_smallest_set_reaching_the_mass() -> None:
    # 0.4 + 0.3 reaches 0.5 only with the second token; the rest are dropped.
    counts = _draws(_PROBS, SamplingParams(temperature=1.0, top_p=0.5), 400)

    assert set(counts) == {0, 1}


def test_tiny_top_p_keeps_only_the_most_likely_token() -> None:
    counts = _draws(_PROBS, SamplingParams(temperature=1.5, top_p=0.01), 100)

    assert set(counts) == {0}


def _scheduler(executor_type: type, mode: str) -> NativeContinuousScheduler:
    model = _RecordingModel()
    cache_backend = DenseKVCacheBackend(num_layers=model.num_layers)
    return NativeContinuousScheduler(
        executor_type(
            architecture_class="FakeForCausalLM",
            model=model,
            cache_backend=cache_backend,
            attention_backend=DenseReferenceAttentionBackend(),
        ),
        NativeCacheCoordinator(cache_backend, NoPrefixCache()),
        execution_mode=mode,
    )


def _generate(scheduler: NativeContinuousScheduler, count: int) -> list[int]:
    scheduler.submit(
        SchedulableRequest(
            request_id="sampled",
            prompt_token_ids=(1, 2, 3),
            sampling=SamplingParams(temperature=1.5, top_p=0.95, seed=2024),
            enqueued_at=time.perf_counter(),
            max_tokens=count,
        )
    )
    tokens: list[int] = []
    for _ in range(count * 3):
        tokens.extend(
            int(event.token_id)
            for event in scheduler.tick()
            if event.kind == "token" and event.token_id is not None
        )
        if len(tokens) >= count:
            break
    scheduler.finish("sampled")
    return tokens[:count]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_serial_and_overlap_sample_identical_tokens() -> None:
    serial = _generate(_scheduler(MlxGenerationExecutor, "serial"), 12)
    overlap = _generate(_scheduler(MlxOverlapGenerationExecutor, "overlap"), 12)

    assert len(serial) == 12
    assert serial == overlap
    # The recording model's logits are [0, 1, 2, 3]; at temperature 1.5 more
    # than the greedy token appears.
    assert len(set(serial)) > 1
