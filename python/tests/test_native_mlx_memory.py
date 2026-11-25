"""KV pool sizing from memory utilization."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.bootstrap import _loading_buffers
from mlx_worker.native_mlx.memory import (
    KVMemoryError,
    activation_reserve_bytes,
    plan_kv_pool,
    resolve_max_total_tokens,
    step_reserve_bytes,
    weights_nbytes,
)
from mlx_worker.native_mlx.registry import StepMemory

GIB = 1024**3


class _Array:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


def test_utilization_pool_is_budget_minus_weights_and_reserve() -> None:
    plan = plan_kv_pool(
        kv_cache_bytes=None,
        utilization=0.5,
        working_set_bytes=20 * GIB,
        weights_bytes=5 * GIB,
        reserve_bytes=2 * GIB,
    )

    assert plan.pool_bytes == 3 * GIB
    assert plan.source == "memory_utilization"
    assert "kv_pool=3.00GiB" in plan.describe()


def test_absolute_pool_overrides_utilization() -> None:
    plan = plan_kv_pool(
        kv_cache_bytes=GIB,
        utilization=0.1,
        working_set_bytes=20 * GIB,
        weights_bytes=5 * GIB,
        reserve_bytes=2 * GIB,
    )

    assert plan.pool_bytes == GIB
    assert plan.source == "kv_cache_bytes"


def test_absolute_pool_works_without_a_known_working_set() -> None:
    plan = plan_kv_pool(
        kv_cache_bytes=GIB,
        utilization=0.5,
        working_set_bytes=None,
        weights_bytes=0,
        reserve_bytes=0,
    )

    assert plan.pool_bytes == GIB


def test_utilization_that_cannot_hold_the_weights_names_the_fix() -> None:
    with pytest.raises(KVMemoryError, match=r"raise --memory-utilization above 0\.35"):
        plan_kv_pool(
            kv_cache_bytes=None,
            utilization=0.3,
            working_set_bytes=20 * GIB,
            weights_bytes=5 * GIB,
            reserve_bytes=2 * GIB,
        )


def test_unknown_working_set_requires_an_absolute_pool() -> None:
    with pytest.raises(KVMemoryError, match="--kv-cache-memory"):
        plan_kv_pool(
            kv_cache_bytes=None,
            utilization=0.5,
            working_set_bytes=None,
            weights_bytes=0,
            reserve_bytes=0,
        )


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.01])
def test_out_of_range_utilization_is_rejected(utilization: float) -> None:
    with pytest.raises(KVMemoryError, match="utilization"):
        plan_kv_pool(
            kv_cache_bytes=None,
            utilization=utilization,
            working_set_bytes=20 * GIB,
            weights_bytes=0,
            reserve_bytes=0,
        )


def test_implicit_limit_is_lowered_to_pool_capacity() -> None:
    assert (
        resolve_max_total_tokens(
            capacity_tokens=24_000,
            configured=65_536,
            explicit=False,
            bytes_per_token=147_456,
        )
        == 24_000
    )


def test_limit_that_fits_is_kept() -> None:
    for explicit in (False, True):
        assert (
            resolve_max_total_tokens(
                capacity_tokens=24_000,
                configured=8_192,
                explicit=explicit,
                bytes_per_token=147_456,
            )
            == 8_192
        )


def test_explicit_limit_that_does_not_fit_refuses_startup() -> None:
    with pytest.raises(KVMemoryError, match="max_total_tokens_per_request=65536"):
        resolve_max_total_tokens(
            capacity_tokens=24_000,
            configured=65_536,
            explicit=True,
            bytes_per_token=147_456,
        )


def test_activation_reserve_counts_buffer_cache_prefill_and_logits() -> None:
    config = {"hidden_size": 4096, "intermediate_size": 12288, "vocab_size": 151936}

    reserve = activation_reserve_bytes(
        config,
        prefill_tokens=1024,
        prefill_rows=4,
        buffer_cache_limit_bytes=GIB,
    )

    prefill = 1024 * 2 * (4 * 4096 + 3 * 12288)
    logits = 4 * 151936 * 4
    assert reserve == GIB + prefill + logits + 256 * 1024 * 1024


def test_activation_reserve_reads_nested_text_config() -> None:
    nested = {
        "text_config": {"hidden_size": 8, "intermediate_size": 16},
        "vocab_size": 4,
    }
    flat = {"hidden_size": 8, "intermediate_size": 16, "vocab_size": 4}

    assert activation_reserve_bytes(
        nested, prefill_tokens=2, prefill_rows=1, buffer_cache_limit_bytes=0
    ) == activation_reserve_bytes(
        flat, prefill_tokens=2, prefill_rows=1, buffer_cache_limit_bytes=0
    )


def test_step_reserve_uses_the_measurements_of_a_family() -> None:
    reserve = step_reserve_bytes(
        StepMemory(base_bytes=GIB, bytes_per_token=1000, buffer_cache_bytes=GIB),
        prefill_tokens=512,
        buffer_cache_limit_bytes=77,
    )

    assert reserve == GIB + 512 * 1000 + 77


def test_loading_keeps_no_freed_buffers() -> None:
    before = mx.set_cache_limit(GIB)
    try:
        with _loading_buffers(GIB // 4):
            assert mx.set_cache_limit(0) == 0
        assert mx.set_cache_limit(GIB) == GIB // 4
        with _loading_buffers(None):
            pass
        assert mx.set_cache_limit(GIB) == GIB
    finally:
        mx.set_cache_limit(before)


def test_weights_nbytes_sums_loaded_arrays() -> None:
    assert weights_nbytes([("a", _Array(10)), ("b", _Array(32))]) == 42
