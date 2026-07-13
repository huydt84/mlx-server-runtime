"""Interface-level tests for the native MLX architecture."""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_worker.native_mlx.bootstrap import (
    NativeBootstrapFailure,
    build_native_artifacts,
    detect_native_architecture,
)
from mlx_worker.native_mlx.attention import (
    AttentionBackendCapabilities,
    DenseReferenceAttentionBackend,
    PagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.cache import (
    DenseBatchReservation,
    DenseKVCacheBackend,
    HybridPagedKVCacheBackend,
    KVCacheGeometry,
    PagedBatchReservation,
    PagedKVCacheBackend,
)
from mlx_worker.native_mlx.cache_coordinator import NativeCacheCoordinator
from mlx_worker.native_mlx.executor import (
    MlxGenerationExecutor,
    MlxOverlapGenerationExecutor,
)
from mlx_worker.native_mlx.execution_backends import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    available_native_execution_backends,
    build_native_execution_backend,
)
from mlx_worker.native_mlx.graph_profile import GraphProfiledModel
from mlx_worker.native_mlx.interfaces import (
    BatchExecutionError,
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    ForwardMode,
    RuntimeEvent,
    SamplingParams,
    SchedulableRequest,
    SchedulerEvent,
    StepRequestResult,
    StepResult,
)
from mlx_worker.native_mlx.models.qwen2 import (
    Qwen2ForCausalLM,
    Qwen2ModelConfig,
    Qwen2WeightAdapter,
)
from mlx_worker.native_mlx.prefix_cache import NoPrefixCache
from mlx_worker.native_mlx.prefix_cache import (
    BlockHashPrefixCache,
    PrefixCompatibilityFingerprint,
    RadixPrefixCache,
)
from mlx_worker.native_mlx.registry import get_architecture_spec
from mlx_worker.native_mlx.runtime import NativeRuntime
from mlx_worker.native_mlx.scheduler import NativeContinuousScheduler


def _tiny_qwen2_config() -> Qwen2ModelConfig:
    return Qwen2ModelConfig(
        architecture_class="Qwen2ForCausalLM",
        model_type="qwen2",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=True,
        quantization=None,
    )


def _request(
    request_id: str,
    token_ids: tuple[int, ...],
    positions: tuple[int, ...],
    cache_handle: str,
    *,
    phase: str = "prefill",
) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        phase=phase,  # type: ignore[arg-type]
        token_ids=token_ids,
        positions=positions,
        cache_handle=cache_handle,
        sampling=SamplingParams(),
    )


@dataclass
class _RecordingModel:
    """Second causal-model implementation proving executor reuse."""

    num_layers: int = 1
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    fail_after_cache_stage: bool = False

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        self.calls.append(
            (
                tuple(int(value) for value in input_ids.shape),
                forward_batch.token_lengths,
            )
        )
        batch, sequence = int(input_ids.shape[0]), int(input_ids.shape[1])
        queries = mx.zeros((batch, 1, sequence, 2), dtype=mx.float16)
        keys = mx.zeros((batch, 1, sequence, 2), dtype=mx.float16)
        values = mx.zeros((batch, 1, sequence, 2), dtype=mx.float16)
        for attention in forward_batch.layer_attention:
            attention.append_and_attend(
                queries,
                keys,
                values,
                scale=1.0,
                mask=forward_batch.attention_mask,
            )
        if self.fail_after_cache_stage:
            raise RuntimeError("model failed")
        row = mx.arange(4, dtype=mx.float32)
        return mx.broadcast_to(row, (batch, sequence, 4))

    def load_weights(
        self,
        weights: Any,
        *,
        strict: bool = True,
    ) -> None:
        del weights, strict


class _TinyGraphBlock(nn.Module):
    """Synthetic transformer-like block used by graph profiling tests."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)
        self.input_layernorm = nn.RMSNorm(4)
        self.post_attention_layernorm = nn.RMSNorm(4)

    def __call__(self, hidden: mx.array) -> mx.array:
        attention = self.self_attn(self.input_layernorm(hidden))
        return self.mlp(self.post_attention_layernorm(hidden + attention))


class _TinyGraphModel(nn.Module):
    """Minimal model tree with common transformer module names."""

    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 2
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = [_TinyGraphBlock(), _TinyGraphBlock()]
        self.norm = nn.RMSNorm(4)
        self.lm_head = nn.Linear(4, 8)

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        del positions, forward_batch
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))

    def load_weights(
        self,
        weights: Any,
        *,
        strict: bool = True,
    ) -> None:
        del weights, strict


@dataclass(frozen=True)
class _ExecutorFixture:
    executor: MlxGenerationExecutor
    cache_backend: DenseKVCacheBackend
    cache_coordinator: NativeCacheCoordinator

    def acquire(self, request_id: str, token_ids: tuple[int, ...] = ()) -> str:
        return self.cache_coordinator.acquire(request_id, token_ids).cache_handle


def _executor(model: _RecordingModel | None = None) -> _ExecutorFixture:
    active_model = model or _RecordingModel()
    cache_backend = DenseKVCacheBackend(num_layers=active_model.num_layers)
    cache_coordinator = NativeCacheCoordinator(cache_backend, NoPrefixCache())
    executor = MlxGenerationExecutor(
        architecture_class="FakeForCausalLM",
        model=active_model,
        cache_backend=cache_backend,
        attention_backend=DenseReferenceAttentionBackend(),
    )
    return _ExecutorFixture(executor, cache_backend, cache_coordinator)


def _fingerprint(
    *,
    tokenizer_assets_hash: str = "tokenizer-a",
    page_size: int = 2,
) -> PrefixCompatibilityFingerprint:
    return PrefixCompatibilityFingerprint(
        checkpoint="checkpoint-a",
        architecture_class="FakeForCausalLM",
        tokenizer_assets_hash=tokenizer_assets_hash,
        model_dtype="float16",
        kv_dtype="float16",
        quantization="none",
        page_size=page_size,
    )


def _commit_dense_tokens(
    backend: DenseKVCacheBackend,
    handle: str,
    request_id: str,
    length: int,
) -> None:
    cache = backend.get(handle, request_id)
    reservation = backend.reserve_batch((cache,), (length,))
    keys = mx.ones((1, 1, length, 2), dtype=mx.float16)
    values = mx.ones((1, 1, length, 2), dtype=mx.float16)
    reservation.layer_views[0].update_and_fetch(keys, values)
    reservation.commit()


def test_models_directory_has_registered_architecture_modules() -> None:
    models = Path(__file__).parents[1] / "mlx_worker/native_mlx/models"
    assert sorted(path.name for path in models.glob("*.py")) == [
        "__init__.py",
        "gemma3.py",
        "lfm2.py",
        "qwen2.py",
        "qwen3.py",
    ]
    assert not (models / "Qwen2ForCausalLM").exists()


def test_qwen2_model_module_has_no_runtime_dependencies() -> None:
    source = (
        Path(__file__).parents[1] / "mlx_worker/native_mlx/models/qwen2.py"
    ).read_text()
    tree = ast.parse(source)
    forbidden = {"runtime", "scheduler", "executor", "worker", "ipc"}
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        any(part == forbidden_name for part in name.split("."))
        for name in imports
        for forbidden_name in forbidden
    )
    assert "request_id" not in source
    assert "cache_handle" not in source
    assert "ExecutionBatch" not in source


def test_registry_composes_shared_model_and_cache_backend() -> None:
    spec = get_architecture_spec("Qwen2ForCausalLM")
    assert spec is not None
    assert spec.create_model is not None
    execution_plan = spec.resolve()
    assert execution_plan.architecture_class == "Qwen2ForCausalLM"
    assert execution_plan.cache_family == "kv"
    geometry = spec.cache_geometry(_tiny_qwen2_config())
    assert geometry.num_layers == 2
    assert geometry.num_kv_heads == 2
    assert geometry.head_dim == 4
    assert Qwen2ForCausalLM.__module__.endswith("models.qwen2")
    assert Qwen2WeightAdapter.__module__.endswith("models.qwen2")


def test_graph_profile_wraps_model_tree_without_architecture_hooks() -> None:
    model = _TinyGraphModel()
    profiled = GraphProfiledModel(model)
    profiled.reset_graph_profile()

    logits = profiled(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        mx.array([[0, 1, 2]], dtype=mx.int32),
        ForwardBatch(
            forward_mode=ForwardMode.PREFILL,
            token_lengths=(3,),
            cache_lengths=(0,),
            attention_mask="causal",
            layer_attention=(),
        ),
    )
    mx.eval(logits)

    metrics = profiled.graph_profile_metrics()
    assert metrics["model_graph_embedding_ms"] >= 0
    assert metrics["model_graph_attention_ms"] >= 0
    assert metrics["model_graph_mlp_ms"] >= 0
    assert metrics["model_graph_norm_ms"] >= 0
    assert metrics["model_graph_lm_head_ms"] >= 0
    assert metrics["model_graph_layer_total_ms"] >= (
        metrics["model_graph_attention_ms"] + metrics["model_graph_mlp_ms"]
    )
    assert metrics["model_graph_worst_layer_index"] in {0, 1}
    assert (
        "reset_graph_profile"
        not in Path(__file__)
        .parents[1]
        .joinpath("mlx_worker/native_mlx/models/qwen2.py")
        .read_text()
    )


def test_detect_native_architecture_rejects_unsupported_class(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"]})
    )
    with pytest.raises(NativeBootstrapFailure) as caught:
        detect_native_architecture(str(tmp_path))
    assert caught.value.error.category == "unsupported_class"


def test_bootstrap_classifies_invalid_supported_config_as_malformed(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]})
    )
    with pytest.raises(NativeBootstrapFailure) as caught:
        build_native_artifacts(str(tmp_path))
    assert caught.value.error.code == "INVALID_NATIVE_CONFIG"
    assert caught.value.error.category == "malformed_checkpoint"
    assert caught.value.error.stage == "artifact_validation"


def test_executor_physically_batches_unequal_prefill_rows() -> None:
    model = _RecordingModel()
    fixture = _executor(model)
    first = fixture.acquire("first")
    second = fixture.acquire("second")

    result = fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("first", (1, 2, 3), (0, 1, 2), first),
                _request("second", (4,), (0,), second),
            ),
        )
    )

    assert model.calls == [((2, 3), (3, 1))]
    assert [item.next_token_id for item in result.results] == [3, 3]
    assert [item.cache_length for item in result.results] == [3, 1]


def test_hybrid_cache_isolated_from_pure_paged_kv_state() -> None:
    pure = PagedKVCacheBackend(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=4096,
    )
    pure_handle = pure.create("pure")
    assert not hasattr(pure.get(pure_handle, "pure"), "conv_states")

    hybrid = HybridPagedKVCacheBackend(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=4096,
    )
    handle = hybrid.create("hybrid")
    cache = hybrid.get(handle, "hybrid")
    reservation = hybrid.reserve_batch((cache,), (3,))
    reservation.stage_conv_state(0, mx.zeros((1, 3, 4), dtype=mx.float16), 2)
    keys = mx.zeros((1, 1, 3, 2), dtype=mx.float16)
    reservation.stage_layer(1, keys, keys)
    reservation.commit()
    assert 0 in cache.conv_states


def test_executor_physically_batches_decode_with_different_cache_lengths() -> None:
    model = _RecordingModel()
    fixture = _executor(model)
    first = fixture.acquire("first")
    second = fixture.acquire("second")
    fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("first", (1, 2), (0, 1), first),
                _request("second", (3,), (0,), second),
            ),
        )
    )
    model.calls.clear()

    result = fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("first", (7,), (2,), first, phase="decode"),
                _request("second", (8,), (1,), second, phase="decode"),
            ),
        )
    )

    assert model.calls == [((2, 1), (1, 1))]
    assert [item.cache_length for item in result.results] == [3, 2]


def test_executor_failure_does_not_commit_or_release_request_caches() -> None:
    model = _RecordingModel(fail_after_cache_stage=True)
    fixture = _executor(model)
    first = fixture.acquire("first")
    second = fixture.acquire("second")

    with pytest.raises(BatchExecutionError, match="model failed") as caught:
        fixture.executor.execute_batch(
            ExecutionBatch(
                requests=(
                    _request("first", (1,), (0,), first),
                    _request("second", (2,), (0,), second),
                ),
            )
        )

    assert caught.value.code == "MODEL_EXECUTION_FAILED"
    assert fixture.cache_coordinator.length(first) == 0
    assert fixture.cache_coordinator.length(second) == 0
    assert fixture.cache_backend.get(first, "first").request_id == "first"
    assert fixture.cache_backend.get(second, "second").request_id == "second"


def test_executor_mixes_decode_and_prefill_in_one_model_invocation() -> None:
    model = _RecordingModel()
    fixture = _executor(model)
    decode_handle = fixture.acquire("decode")
    prefill_handle = fixture.acquire("prefill")
    initial = fixture.executor.execute_batch(
        ExecutionBatch(requests=(_request("decode", (1, 2), (0, 1), decode_handle),))
    )
    model.calls.clear()

    result = fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request(
                    "decode",
                    (int(initial.results[0].next_token_id),),
                    (2,),
                    decode_handle,
                    phase="decode",
                ),
                _request("prefill", (3, 4, 5), (0, 1, 2), prefill_handle),
            )
        )
    )

    assert model.calls == [((2, 3), (1, 3))]
    assert result.forward_mode is ForwardMode.MIXED
    assert result.physical_batch_size == 2
    assert result.model_forward_count == 1
    assert [item.phase for item in result.results] == ["decode", "prefill"]
    assert [item.cache_length for item in result.results] == [3, 3]


def test_executor_isolates_request_local_preflight_failure() -> None:
    model = _RecordingModel()
    fixture = _executor(model)
    valid_handle = fixture.acquire("valid")

    result = fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("invalid", (1,), (0,), "missing-cache"),
                _request("valid", (2, 3), (0, 1), valid_handle),
            )
        )
    )

    invalid, valid = result.results
    assert invalid.error_code == "INVALID_EXECUTION_REQUEST"
    assert invalid.error_message == "invalid cache handle"
    assert valid.error_code is None
    assert valid.cache_length == 2
    assert fixture.cache_coordinator.length(valid_handle) == 2
    assert model.calls == [((1, 2), (2,))]
    assert result.physical_batch_size == 1
    assert result.model_forward_count == 1


def test_executor_skips_model_when_every_request_fails_preflight() -> None:
    model = _RecordingModel()
    fixture = _executor(model)

    result = fixture.executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("prefill", (1,), (0,), "missing-prefill"),
                _request(
                    "decode",
                    (2,),
                    (0,),
                    "missing-decode",
                    phase="decode",
                ),
            )
        )
    )

    assert result.forward_mode is ForwardMode.MIXED
    assert result.physical_batch_size == 0
    assert result.model_forward_count == 0
    assert all(
        item.error_code == "INVALID_EXECUTION_REQUEST" for item in result.results
    )
    assert model.calls == []


def test_paged_backend_allocates_pages_and_commits_block_tables() -> None:
    backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=512,
        dtype=mx.float16,
    )
    first = backend.get(backend.create("first"), "first")
    second = backend.get(backend.create("second"), "second")
    reservation = backend.reserve_batch((first, second), (3, 1))
    keys = mx.ones((2, 1, 3, 2), dtype=mx.float16)
    values = mx.ones((2, 1, 3, 2), dtype=mx.float16)
    reservation.stage_layer(0, keys, values)

    assert reservation.commit() == (3, 1)
    assert len(first.block_table) == 1
    assert len(second.block_table) == 1
    metrics = backend.metrics()
    assert metrics["cache_backend"] == "paged-mlx"
    assert metrics["used_pages"] == 2
    assert metrics["internal_fragmentation_tokens"] == 12


def test_paged_backend_capacity_failure_is_pre_mutation() -> None:
    backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=64,
        dtype=mx.float16,
    )
    first = backend.get(backend.create("first"), "first")
    second = backend.get(backend.create("second"), "second")

    errors = backend.preflight((first, second), (1, 1))

    assert errors == (None, "native paged KV capacity exhausted before model execution")
    assert first.size() == 0
    assert second.size() == 0
    assert backend.metrics()["used_pages"] == 0


def test_paged_backend_fork_uses_copy_on_write_for_shared_tail() -> None:
    backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=256,
        dtype=mx.float16,
    )
    parent_handle = backend.create("parent")
    parent = backend.get(parent_handle, "parent")
    first = backend.reserve_batch((parent,), (3,))
    first.stage_layer(
        0,
        mx.ones((1, 1, 3, 2), dtype=mx.float16),
        mx.ones((1, 1, 3, 2), dtype=mx.float16),
    )
    first.commit()
    child_handle = backend.fork(parent_handle, "child")
    child = backend.get(child_handle, "child")

    append = backend.reserve_batch((child,), (1,))
    append.stage_layer(
        0,
        mx.ones((1, 1, 1, 2), dtype=mx.float16),
        mx.ones((1, 1, 1, 2), dtype=mx.float16),
    )
    assert append.commit() == (4,)

    assert parent.size() == 3
    assert child.size() == 4
    assert parent.block_table != child.block_table
    assert backend.metrics()["used_pages"] == 2


def test_block_hash_prefix_cache_reuses_only_exact_full_pages() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 4)
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=4096,
    )

    publication = cache.publish(handle, (1, 2, 3, 4), 4)
    exact = cache.probe((1, 2, 3, 4, 9))
    partial = cache.probe((1, 2, 8))
    token_mismatch = cache.probe((1, 9, 3, 4))
    partial_page_only = cache.probe((1,))

    assert publication.published_tokens == 4
    assert publication.published_pages == 2
    assert exact.matched_tokens == 4
    assert exact.matched_pages == 2
    assert partial.matched_tokens == 2
    assert partial.matched_pages == 1
    assert token_mismatch.matched_tokens == 0
    assert partial_page_only.matched_tokens == 0
    assert cache.metrics()["prefix_hits"] == 2
    assert cache.metrics()["prefix_misses"] == 2


def test_block_hash_prefix_cache_rejects_incompatible_fingerprint() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 2)
    first = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(tokenizer_assets_hash="tokenizer-a"),
        page_size=2,
    )
    second = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(tokenizer_assets_hash="tokenizer-b"),
        page_size=2,
    )

    first.publish(handle, (1, 2), 2)

    assert first.probe((1, 2, 3)).matched_tokens == 2
    assert second.probe((1, 2, 3)).matched_tokens == 0


def test_block_hash_prefix_cache_acquire_pins_and_release_unpins() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 4)
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=4096,
    )
    cache.publish(handle, (1, 2, 3, 4), 4)
    probe = cache.probe((1, 2, 3, 4, 5))

    admission = cache.acquire("child", (1, 2, 3, 4, 5), probe)

    assert admission is not None
    assert admission.reused_tokens == 4
    assert backend.length(admission.cache_handle) == 4
    assert cache.metrics()["prefix_reused_pages"] == 2
    assert cache.metrics()["prefix_pinned_pages"] == 2

    cache.release(admission.cache_handle)
    backend.release(admission.cache_handle)

    assert cache.metrics()["prefix_pinned_pages"] == 0


def test_block_hash_prefix_cache_evicts_unpinned_lru_entries() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    first = backend.create("first")
    second = backend.create("second")
    _commit_dense_tokens(backend, first, "first", 2)
    _commit_dense_tokens(backend, second, "second", 2)
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=1,
        max_bytes=4096,
    )

    cache.publish(first, (1, 2), 2)
    cache.publish(second, (3, 4), 2)

    assert cache.probe((1, 2, 9)).matched_tokens == 0
    assert cache.probe((3, 4, 9)).matched_tokens == 2
    assert cache.metrics()["prefix_evictions"] == 1


def test_block_hash_prefix_cache_entry_limit_keeps_reusable_root_chain() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 12)
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=4,
        max_bytes=4096,
    )

    cache.publish(handle, tuple(range(12)), 12)
    cache.publish(handle, tuple(range(12)), 12)

    assert cache.probe(tuple(range(12)) + (99,)).matched_tokens == 8
    assert cache.metrics()["prefix_entries"] == 4
    assert cache.metrics()["prefix_evictions"] == 0


def test_block_hash_prefix_cache_byte_limit_counts_shared_pages_once() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 8)
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=8,
    )

    cache.publish(handle, tuple(range(8)), 8)

    assert cache.probe(tuple(range(8)) + (99,)).matched_tokens == 8
    assert cache.metrics()["prefix_entries"] == 4
    assert cache.metrics()["prefix_bytes"] == 8


def test_radix_prefix_cache_returns_longest_page_aligned_match() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 6)
    cache = RadixPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=4096,
    )

    publication = cache.publish(handle, (1, 2, 3, 4, 5, 6), 6)
    exact = cache.probe((1, 2, 3, 4, 5, 6, 9))
    partial = cache.probe((1, 2, 3, 8))
    partial_page_only = cache.probe((1,))

    assert publication.published_tokens == 6
    assert publication.published_pages == 3
    assert exact.matched_tokens == 6
    assert exact.matched_pages == 3
    assert partial.matched_tokens == 2
    assert partial.matched_pages == 1
    assert partial_page_only.matched_tokens == 0
    assert cache.metrics()["prefix_strategy"] == "radix"
    assert cache.metrics()["prefix_hits"] == 2
    assert cache.metrics()["prefix_misses"] == 1


def test_radix_prefix_cache_splits_branching_edges() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    first = backend.create("first")
    second = backend.create("second")
    _commit_dense_tokens(backend, first, "first", 6)
    _commit_dense_tokens(backend, second, "second", 6)
    cache = RadixPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=4096,
    )

    cache.publish(first, (1, 2, 3, 4, 5, 6), 6)
    cache.publish(second, (1, 2, 3, 9, 10, 11), 6)

    assert cache.probe((1, 2, 3, 4, 5, 6, 7)).matched_tokens == 6
    assert cache.probe((1, 2, 3, 9, 10, 11, 12)).matched_tokens == 6
    assert cache.probe((1, 2, 3, 8)).matched_tokens == 2
    metrics = cache.metrics()
    assert metrics["radix_splits"] >= 1
    assert metrics["radix_shared_pages"] >= 1


def test_radix_prefix_cache_acquire_pins_and_release_unpins() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    handle = backend.create("source")
    _commit_dense_tokens(backend, handle, "source", 4)
    cache = RadixPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=8,
        max_bytes=4096,
    )
    cache.publish(handle, (1, 2, 3, 4), 4)
    probe = cache.probe((1, 2, 3, 4, 5))

    admission = cache.acquire("child", (1, 2, 3, 4, 5), probe)

    assert admission is not None
    assert admission.reused_tokens == 4
    assert backend.length(admission.cache_handle) == 4
    assert cache.metrics()["prefix_reused_pages"] == 2
    assert cache.metrics()["radix_protected_pages"] == 2

    cache.release(admission.cache_handle)
    backend.release(admission.cache_handle)

    assert cache.metrics()["radix_protected_pages"] == 0


def test_radix_prefix_cache_evicts_unpinned_leaves_first() -> None:
    backend = DenseKVCacheBackend(num_layers=1)
    first = backend.create("first")
    second = backend.create("second")
    _commit_dense_tokens(backend, first, "first", 4)
    _commit_dense_tokens(backend, second, "second", 4)
    cache = RadixPrefixCache(
        backend=backend,
        compatibility=_fingerprint(),
        page_size=2,
        max_entries=2,
        max_bytes=4096,
    )

    cache.publish(first, (1, 2, 3, 4), 4)
    cache.publish(second, (1, 2, 5, 6), 4)

    assert cache.probe((1, 2, 9)).matched_tokens == 2
    assert cache.metrics()["prefix_entries"] == 2
    assert cache.metrics()["radix_leaf_evictions"] == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_paged_metal_attention_matches_dense_reference() -> None:
    dense_backend = DenseKVCacheBackend(num_layers=1)
    dense_cache = dense_backend.get(dense_backend.create("dense"), "dense")
    dense_reservation = dense_backend.reserve_batch((dense_cache,), (2,))
    paged_backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        page_size=8,
        budget_bytes=256,
        dtype=mx.float16,
    )
    paged_cache = paged_backend.get(paged_backend.create("paged"), "paged")
    paged_reservation = paged_backend.reserve_batch((paged_cache,), (2,))
    queries = mx.array(
        [[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]]],
        dtype=mx.float16,
    )
    keys = mx.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]], dtype=mx.float16)
    values = mx.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=mx.float16)
    dense_context = DenseReferenceAttentionBackend().contexts(
        dense_reservation,
        ForwardMode.PREFILL,
    )[0]
    paged_attention = PagedMetalAttentionBackend()
    paged_context = paged_attention.contexts(
        paged_reservation,
        ForwardMode.PREFILL,
    )[0]

    dense = dense_context.append_and_attend(
        queries,
        keys,
        values,
        scale=0.5,
        mask="causal",
    )
    paged = paged_context.append_and_attend(
        queries,
        keys,
        values,
        scale=0.5,
        mask="causal",
    )
    mx.eval(dense, paged)

    assert mx.allclose(dense, paged, atol=1e-3, rtol=1e-3).item()
    assert paged_attention.metrics()["attention_backend"] == "native-metal-paged-sdpa"
    assert "attention_backend" not in paged_backend.metrics()


def test_attention_capabilities_reject_incompatible_cache_backend() -> None:
    capabilities = AttentionBackendCapabilities(
        backend_id="test-dense",
        cache_backend_types=(DenseKVCacheBackend,),
        reservation_types=(DenseBatchReservation,),
        supported_masks=frozenset(("causal",)),
        supported_forward_modes=frozenset(ForwardMode),
        requires_metal=False,
    )
    paged_backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        page_size=8,
        budget_bytes=256,
        dtype=mx.float16,
    )

    with pytest.raises(TypeError, match="test-dense requires cache backend type"):
        capabilities.validate_cache_backend(paged_backend)
    paged_cache = paged_backend.get(paged_backend.create("paged"), "paged")
    paged_reservation = paged_backend.reserve_batch((paged_cache,), (1,))
    with pytest.raises(TypeError, match="test-dense requires reservation type"):
        capabilities.validate_context(paged_reservation, ForwardMode.PREFILL)


def test_native_execution_backend_registry_exposes_stable_default() -> None:
    assert DEFAULT_NATIVE_EXECUTION_BACKEND in available_native_execution_backends()


def test_bootstrap_rejects_unknown_execution_backend_before_model_loading() -> None:
    with pytest.raises(NativeBootstrapFailure) as caught:
        build_native_artifacts("missing-model", execution_backend_id="unknown")

    assert caught.value.error.code == "UNSUPPORTED_NATIVE_EXECUTION_BACKEND"
    assert caught.value.error.stage == "backend_selection"
    assert caught.value.error.category == "invalid_configuration"


def test_bootstrap_accepts_overlap_mode_before_model_resolution() -> None:
    with pytest.raises(NativeBootstrapFailure) as caught:
        build_native_artifacts("missing-model", execution_mode="overlap")

    assert caught.value.error.code == "MODEL_RESOLUTION_FAILED"
    assert caught.value.error.stage == "artifact_validation"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_native_execution_backend_registry_builds_compatible_bundle() -> None:
    bundle = build_native_execution_backend(
        DEFAULT_NATIVE_EXECUTION_BACKEND,
        KVCacheGeometry(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            dtype=mx.float16,
        ),
        page_size=8,
        cache_budget_bytes=256,
    )

    bundle.validate()
    assert isinstance(bundle.cache_backend, PagedKVCacheBackend)
    assert isinstance(bundle.attention_backend, PagedMetalAttentionBackend)
    assert bundle.attention_backend.capabilities.reservation_types == (
        PagedBatchReservation,
    )
    assert bundle.attention_backend.capabilities.consumes_page_tables_directly is False


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_executor_isolates_paged_capacity_failure_before_model_call() -> None:
    model = _RecordingModel()
    cache_backend = PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=8,
        budget_bytes=64,
        dtype=mx.float16,
    )
    executor = MlxGenerationExecutor(
        architecture_class="FakeForCausalLM",
        model=model,
        cache_backend=cache_backend,
        attention_backend=PagedMetalAttentionBackend(),
    )
    first = cache_backend.create("first")
    second = cache_backend.create("second")

    result = executor.execute_batch(
        ExecutionBatch(
            requests=(
                _request("first", (1,), (0,), first),
                _request("second", (2,), (0,), second),
            )
        )
    )

    assert [item.error_code for item in result.results] == [
        None,
        "KV_CAPACITY_EXHAUSTED",
    ]
    assert [item.cache_length for item in result.results] == [1, 0]
    assert model.calls == [((1, 1), (1,))]
    assert result.physical_batch_size == 1


@dataclass
class _FakeExecutor:
    lengths: dict[str, int] = field(default_factory=dict)
    batches: list[ExecutionBatch] = field(default_factory=list)
    request_failures: set[str] = field(default_factory=set)

    def load(self, options: Any) -> None:
        del options

    def execute_batch(self, batch: ExecutionBatch) -> StepResult:
        self.batches.append(batch)
        results = []
        for request in batch.requests:
            assert request.cache_handle is not None
            if request.request_id in self.request_failures:
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        phase=request.phase,
                        token_ids=request.token_ids,
                        cache_handle=request.cache_handle,
                        cache_length=self.lengths[request.cache_handle],
                        error_code="INVALID_EXECUTION_REQUEST",
                        error_message="request-local failure",
                    )
                )
                continue
            self.lengths[request.cache_handle] += len(request.token_ids)
            results.append(
                StepRequestResult(
                    request_id=request.request_id,
                    phase=request.phase,
                    token_ids=request.token_ids,
                    cache_handle=request.cache_handle,
                    cache_length=self.lengths[request.cache_handle],
                    next_token_id=9,
                )
            )
        return StepResult(
            forward_mode=batch.forward_mode,
            results=tuple(results),
            step_time_ms=1,
            physical_batch_size=sum(
                request.request_id not in self.request_failures
                for request in batch.requests
            ),
            model_forward_count=int(
                any(
                    request.request_id not in self.request_failures
                    for request in batch.requests
                )
            ),
        )


@dataclass
class _FakeCacheCoordinator:
    lengths: dict[str, int]
    reused_tokens: dict[str, int] = field(default_factory=dict)
    released: list[str] = field(default_factory=list)

    def probe(self, token_ids: tuple[int, ...]) -> Any:
        return type(
            "Probe",
            (),
            {
                "matched_tokens": self.reused_tokens.get(str(token_ids), 0),
                "matched_pages": self.reused_tokens.get(str(token_ids), 0) // 2,
                "cache_handle": "cached"
                if str(token_ids) in self.reused_tokens
                else None,
            },
        )()

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: Any = None,
    ) -> Any:
        del token_ids
        handle = f"cache-{request_id}"
        reused = 0
        if probe is not None:
            reused = int(getattr(probe, "matched_tokens", 0))
        self.lengths[handle] = reused
        return type(
            "Admission",
            (),
            {
                "cache_handle": handle,
                "cache_length": reused,
                "reused_tokens": reused,
                "reused_pages": reused // 2,
            },
        )()

    def publish_committed(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> Any:
        del cache_handle, token_ids, committed_length
        return None

    def length(self, cache_handle: str | None) -> int:
        return self.lengths.get(cache_handle or "", 0)

    def release(self, cache_handle: str | None) -> None:
        if cache_handle is not None:
            self.released.append(cache_handle)
            self.lengths.pop(cache_handle, None)

    def metrics(self) -> dict[str, Any]:
        return {}


def _scheduler(
    executor: _FakeExecutor,
    *,
    prefill_batch_size: int = 4,
    prefill_step_size: int = 256,
    prioritize_decode: bool = True,
    scheduling_policy: str = "fcfs",
) -> tuple[NativeContinuousScheduler, _FakeCacheCoordinator]:
    cache_coordinator = _FakeCacheCoordinator(executor.lengths)
    return (
        NativeContinuousScheduler(
            executor,
            cache_coordinator,  # type: ignore[arg-type]
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
            prioritize_decode=prioritize_decode,
            scheduling_policy=scheduling_policy,
        ),
        cache_coordinator,
    )


def _overlap_scheduler(
    *,
    max_tokens: int = 2,
) -> tuple[
    NativeContinuousScheduler,
    DenseKVCacheBackend,
    _RecordingModel,
    SchedulableRequest,
]:
    model = _RecordingModel()
    cache_backend = DenseKVCacheBackend(num_layers=model.num_layers)
    cache_coordinator = NativeCacheCoordinator(cache_backend, NoPrefixCache())
    executor = MlxOverlapGenerationExecutor(
        architecture_class="FakeForCausalLM",
        model=model,
        cache_backend=cache_backend,
        attention_backend=DenseReferenceAttentionBackend(),
    )
    scheduler = NativeContinuousScheduler(
        executor,
        cache_coordinator,
        execution_mode="overlap",
        terminal_token_ids=(0,),
    )
    request = _schedulable("overlap", (1,), max_tokens=max_tokens)
    return scheduler, cache_backend, model, request


def _schedulable(
    request_id: str,
    tokens: tuple[int, ...],
    *,
    max_tokens: int = 1,
    priority: int = 0,
) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling=SamplingParams(),
        enqueued_at=time.perf_counter(),
        max_tokens=max_tokens,
        priority=priority,
    )


def test_scheduler_owns_chunking_and_emits_typed_token_events() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(executor, prefill_step_size=2)
    scheduler.submit(_schedulable("request", (1, 2, 3, 4, 5)))

    first = scheduler.tick()
    second = scheduler.tick()
    third = scheduler.tick()

    assert [
        [request.phase for request in batch.requests] for batch in executor.batches
    ] == [
        ["prefill"],
        ["prefill"],
        ["prefill"],
    ]
    assert [batch.requests[0].token_ids for batch in executor.batches] == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    assert not any(event.kind == "token" for event in first + second)
    assert any(event.kind == "token" and event.token_id == 9 for event in third)


def test_scheduler_dispatches_decode_and_new_prefill_in_one_mixed_step() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_step_size=8,
        prioritize_decode=False,
    )
    scheduler.submit(_schedulable("running", (1,)))
    scheduler.tick()
    scheduler.submit(_schedulable("waiting", (2,)))

    scheduler.tick()

    assert len(executor.batches) == 2
    mixed = executor.batches[-1]
    assert mixed.forward_mode is ForwardMode.MIXED
    assert [request.phase for request in mixed.requests] == ["decode", "prefill"]


def test_scheduler_prioritizes_decode_without_starving_new_prefill() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(executor, prefill_step_size=8)
    scheduler.submit(_schedulable("running", (1,)))
    scheduler.tick()
    scheduler.submit(_schedulable("waiting-first", (2,)))
    scheduler.submit(_schedulable("waiting-second", (3,)))

    scheduler.tick()

    assert len(executor.batches) == 2
    mixed = executor.batches[-1]
    assert mixed.forward_mode is ForwardMode.MIXED
    assert [request.phase for request in mixed.requests] == ["decode", "prefill"]
    assert [request.request_id for request in mixed.requests] == [
        "running",
        "waiting-first",
    ]


def test_scheduler_respects_prefill_batch_size() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_batch_size=4,
        prefill_step_size=8,
        prioritize_decode=False,
    )
    for index in range(5):
        scheduler.submit(_schedulable(f"request-{index}", (index + 1,)))

    scheduler.tick()

    first = executor.batches[-1]
    assert [request.request_id for request in first.requests] == [
        "request-0",
        "request-1",
        "request-2",
        "request-3",
    ]

    scheduler.tick()

    second = executor.batches[-1]
    assert [request.request_id for request in second.requests] == [
        "request-0",
        "request-1",
        "request-2",
        "request-3",
        "request-4",
    ]
    assert [request.phase for request in second.requests] == [
        "decode",
        "decode",
        "decode",
        "decode",
        "prefill",
    ]


def test_scheduler_prefix_hit_reduces_scheduled_prefill_tokens() -> None:
    executor = _FakeExecutor()
    scheduler, cache = _scheduler(executor, prefill_step_size=8)
    tokens = (1, 2, 3, 4, 5)
    cache.reused_tokens[str(tokens)] = 4

    scheduler.submit(_schedulable("request", tokens))
    events = scheduler.tick()

    assert executor.batches[0].requests[0].token_ids == (5,)
    assert executor.batches[0].requests[0].positions == (4,)
    assert any(
        event.kind == "metrics"
        and event.metrics is not None
        and event.metrics["scheduled_tokens"] == 1
        for event in events
    )


def test_scheduler_isolates_request_local_executor_failure() -> None:
    executor = _FakeExecutor(request_failures={"invalid"})
    scheduler, _ = _scheduler(executor)
    scheduler.submit(_schedulable("invalid", (1,)))
    scheduler.submit(_schedulable("valid", (2,)))

    events = scheduler.tick()

    assert any(
        event.kind == "execution_error" and event.request_id == "invalid"
        for event in events
    )
    assert any(
        event.kind == "token" and event.request_id == "valid" for event in events
    )
    assert not any(
        event.kind == "execution_error" and event.request_id == "valid"
        for event in events
    )
    assert executor.lengths["cache-invalid"] == 0
    assert executor.lengths["cache-valid"] == 1


def test_scheduler_cancellation_releases_at_safe_point() -> None:
    executor = _FakeExecutor()
    scheduler, cache_coordinator = _scheduler(executor)
    scheduler.submit(_schedulable("request", (1,)))
    scheduler.tick()

    assert scheduler.cancel("request")
    events = scheduler.tick()
    cancelled = [event for event in events if event.kind == "cancelled"]

    assert len(cancelled) == 1
    assert cancelled[0].metrics["cancellation_stage"] == "decode"
    assert cancelled[0].metrics["cancellation_latency_ms"] >= 0
    assert cache_coordinator.released == ["cache-request"]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_overlap_scheduler_delays_one_result_and_preserves_cache_lifetime() -> None:
    scheduler, cache_backend, model, request = _overlap_scheduler(max_tokens=2)
    scheduler.submit(request)

    first = scheduler.tick()
    handle = scheduler._running["overlap"].cache_handle
    assert handle is not None
    assert [event.token_id for event in first if event.kind == "token"] == [3]
    assert cache_backend.length(handle) == 1

    second = scheduler.tick()
    assert second == ()
    assert cache_backend.length(handle) == 1

    third = scheduler.tick()
    assert [event.token_id for event in third if event.kind == "token"] == [3]
    assert cache_backend.length(handle) == 2
    assert len(model.calls) == 2

    scheduler.finish("overlap")
    assert scheduler.idle()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_overlap_scheduler_defers_in_flight_cancellation_until_resolve() -> None:
    scheduler, cache_backend, _, request = _overlap_scheduler(max_tokens=4)
    scheduler.submit(request)
    scheduler.tick()
    scheduler.tick()
    handle = scheduler._running["overlap"].cache_handle
    assert handle is not None

    assert scheduler.cancel("overlap")
    events = scheduler.tick()

    assert not any(event.kind == "token" for event in events)
    assert any(event.kind == "cancelled" for event in events)
    assert cache_backend.length(handle) == 0
    assert scheduler.idle()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_overlap_scheduler_discards_speculative_result_after_finish() -> None:
    scheduler, cache_backend, model, request = _overlap_scheduler(max_tokens=4)
    scheduler.submit(request)
    first_result = scheduler.tick()
    scheduler.tick()
    handle = scheduler._running["overlap"].cache_handle
    assert handle is not None
    assert any(event.kind == "token" for event in first_result)

    scheduler.finish("overlap")
    assert not scheduler.idle()
    discarded = scheduler.tick()

    assert not any(event.kind == "token" for event in discarded)
    assert cache_backend.length(handle) == 0
    assert len(model.calls) == 2
    assert scheduler.idle()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_overlap_scheduler_close_synchronizes_before_cache_release() -> None:
    scheduler, cache_backend, _, request = _overlap_scheduler(max_tokens=4)
    scheduler.submit(request)
    scheduler.tick()
    scheduler.tick()
    handle = scheduler._running["overlap"].cache_handle
    assert handle is not None

    scheduler.close()

    assert cache_backend.length(handle) == 0
    assert scheduler.idle()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_overlap_scheduler_dispatch_failure_releases_cache_same_tick() -> None:
    model = _RecordingModel(fail_after_cache_stage=True)
    cache_backend = DenseKVCacheBackend(num_layers=model.num_layers)
    scheduler = NativeContinuousScheduler(
        MlxOverlapGenerationExecutor(
            architecture_class="FakeForCausalLM",
            model=model,
            cache_backend=cache_backend,
            attention_backend=DenseReferenceAttentionBackend(),
        ),
        NativeCacheCoordinator(cache_backend, NoPrefixCache()),
        execution_mode="overlap",
    )
    scheduler.submit(_schedulable("failure", (1,), max_tokens=2))

    events = scheduler.tick()

    error = next(event for event in events if event.kind == "execution_error")
    assert error.error_code == "MODEL_EXECUTION_FAILED"
    assert scheduler.idle()
    assert cache_backend.metrics()["active_kv_bytes"] == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_runtime_runs_overlap_pipeline_to_terminal_response() -> None:
    scheduler, cache_backend, _, _ = _overlap_scheduler(max_tokens=2)
    tokenizer = _FakeTokenizer()
    runtime = NativeRuntime(
        scheduler,
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )
    runtime.submit(_chat_request(max_tokens=2))

    events = []
    while not runtime.idle():
        events.extend(runtime.tick())

    assert [event.kind for event in events if event.kind != "metrics"] == [
        "delta",
        "delta",
        "response",
    ]
    response = next(event.payload for event in events if event.kind == "response")
    assert response.finish_reason == "length"
    assert response.completion_tokens == 2
    assert response.text == "cc"
    assert cache_backend.metrics()["active_kv_bytes"] == 0


def test_scheduler_fcfs_policy_orders_by_arrival() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_batch_size=2,
        prioritize_decode=False,
        scheduling_policy="fcfs",
    )
    scheduler.submit(_schedulable("first", (1,), priority=0, max_tokens=1))
    scheduler.submit(_schedulable("second", (2,), priority=10, max_tokens=10))

    scheduler.tick()

    assert [request.request_id for request in executor.batches[-1].requests] == [
        "first",
        "second",
    ]


def test_scheduler_priority_policy_orders_by_internal_priority() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_batch_size=2,
        prioritize_decode=False,
        scheduling_policy="priority",
    )
    scheduler.submit(_schedulable("low", (1,), priority=0))
    scheduler.submit(_schedulable("high", (2,), priority=5))

    scheduler.tick()

    assert [request.request_id for request in executor.batches[-1].requests] == [
        "high",
        "low",
    ]


def test_scheduler_lof_policy_orders_by_max_tokens() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_batch_size=2,
        prioritize_decode=False,
        scheduling_policy="lof",
    )
    scheduler.submit(_schedulable("short", (1,), max_tokens=1))
    scheduler.submit(_schedulable("long", (2,), max_tokens=8))

    scheduler.tick()

    assert [request.request_id for request in executor.batches[-1].requests] == [
        "long",
        "short",
    ]


def test_scheduler_lpm_policy_orders_by_probe_match_length() -> None:
    executor = _FakeExecutor()
    scheduler, cache = _scheduler(
        executor,
        prefill_batch_size=2,
        prioritize_decode=False,
        scheduling_policy="lpm",
    )
    low_tokens = (1, 2, 3)
    high_tokens = (4, 5, 6)
    cache.reused_tokens[str(low_tokens)] = 1
    cache.reused_tokens[str(high_tokens)] = 2
    scheduler.submit(_schedulable("low", low_tokens))
    scheduler.submit(_schedulable("high", high_tokens))

    scheduler.tick()

    assert [request.request_id for request in executor.batches[-1].requests] == [
        "high",
        "low",
    ]
    assert executor.batches[-1].requests[0].positions == (2,)


def test_scheduler_non_fcfs_policy_ages_old_request_ahead_of_newer_work() -> None:
    executor = _FakeExecutor()
    scheduler, _ = _scheduler(
        executor,
        prefill_batch_size=1,
        prioritize_decode=False,
        scheduling_policy="priority",
    )
    scheduler.submit(_schedulable("old-low", (1,), priority=0))
    for index in range(3):
        scheduler.submit(_schedulable(f"new-high-{index}", (index + 2,), priority=10))
        scheduler.tick()
        scheduler.finish(f"new-high-{index}")

    scheduler.submit(_schedulable("last-high", (9,), priority=10))
    scheduler.tick()

    assert executor.batches[-1].requests[0].request_id == "old-low"


@dataclass
class _FakeScheduler:
    submitted: list[SchedulableRequest] = field(default_factory=list)
    events: list[tuple[SchedulerEvent, ...]] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)

    def submit(self, request: SchedulableRequest) -> None:
        self.submitted.append(request)

    def cancel(self, request_id: str) -> bool:
        handled = any(item.request_id == request_id for item in self.submitted)
        if handled:
            self.events.append(
                (
                    SchedulerEvent(
                        kind="cancelled",
                        request_id=request_id,
                        metrics={
                            "cancellation_stage": "python_waiting",
                            "cancellation_latency_ms": 2,
                        },
                    ),
                )
            )
        return handled

    def finish(self, request_id: str) -> None:
        self.finished.append(request_id)
        self.submitted = [
            item for item in self.submitted if item.request_id != request_id
        ]

    def tick(self) -> tuple[SchedulerEvent, ...]:
        return self.events.pop(0) if self.events else ()

    def idle(self) -> bool:
        return not self.submitted

    def close(self) -> None:
        self.submitted.clear()


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert messages
        assert tokenize and add_generation_prompt
        return [1, 2]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "".join(chr(96 + token) for token in token_ids)


class _StreamingFakeTokenizer(_FakeTokenizer):
    class _Detokenizer:
        def __init__(self) -> None:
            self._last = ""

        def add_token(self, token_id: int) -> None:
            self._last = chr(96 + token_id)

        @property
        def last_segment(self) -> str:
            segment, self._last = self._last, ""
            return segment

        def finalize(self) -> None:
            return None

    @property
    def detokenizer(self) -> _Detokenizer:
        return self._Detokenizer()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        raise AssertionError("streaming detokenizer should avoid full-output decode")


def _chat_request(**overrides: Any):
    from mlx_worker.ipc import ChatCompletionRequest, ChatMessage

    values = {
        "request_id": "request",
        "model": "test-model",
        "messages": [ChatMessage(role="user", content="hello")],
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_prompt_tokens": 16,
        "max_completion_tokens": 4,
        "max_total_tokens_per_request": 20,
        "stream": True,
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def test_runtime_normalizes_public_request_and_owns_terminal_text() -> None:
    scheduler = _FakeScheduler()
    tokenizer = _FakeTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )
    runtime.submit(_chat_request())
    submitted_prompt = scheduler.submitted[0].prompt_token_ids
    scheduler.events.append(
        (
            SchedulerEvent(
                kind="token",
                request_id="request",
                token_id=1,
                cache_length=2,
                phase="prefill",
            ),
        )
    )
    scheduler.events.append(
        (
            SchedulerEvent(
                kind="token",
                request_id="request",
                token_id=2,
                cache_length=3,
                phase="decode",
                metrics={"step_time_ms": 1, "batch_size": 1},
            ),
        )
    )

    first = runtime.tick()
    second = runtime.tick()

    assert submitted_prompt == (1, 2)
    assert [event.kind for event in first] == ["delta"]
    assert [event.kind for event in second] == ["delta", "response"]
    response = next(
        event.payload
        for event in second
        if isinstance(event, RuntimeEvent) and event.kind == "response"
    )
    assert response.text == "ab"
    assert response.finish_reason == "length"
    assert scheduler.finished == ["request"]


def test_runtime_uses_request_local_streaming_detokenizer() -> None:
    scheduler = _FakeScheduler()
    tokenizer = _StreamingFakeTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )
    runtime.submit(_chat_request())
    scheduler.events.extend(
        [
            (
                SchedulerEvent(
                    kind="token",
                    request_id="request",
                    token_id=1,
                    cache_length=2,
                    phase="prefill",
                ),
            ),
            (
                SchedulerEvent(
                    kind="token",
                    request_id="request",
                    token_id=2,
                    cache_length=3,
                    phase="decode",
                ),
            ),
        ]
    )

    first = runtime.tick()
    second = runtime.tick()

    assert [event.kind for event in first] == ["delta"]
    assert [event.kind for event in second] == ["delta", "response"]
    response = next(event.payload for event in second if event.kind == "response")
    assert response.text == "ab"


def test_runtime_passes_scheduler_policy_metadata() -> None:
    scheduler = _FakeScheduler()
    tokenizer = _FakeTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )

    runtime.submit(_chat_request(max_tokens=7, max_completion_tokens=8))

    assert scheduler.submitted[0].max_tokens == 7
    assert scheduler.submitted[0].priority == 0


def test_runtime_reports_scheduler_queue_wait_separately() -> None:
    scheduler = _FakeScheduler()
    tokenizer = _FakeTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )
    runtime.submit(_chat_request(max_tokens=1, stream=False))
    scheduler.events.append(
        (
            SchedulerEvent(
                kind="prefill_progress",
                request_id="request",
                cache_length=2,
                phase="prefill",
                metrics={
                    "step_time_ms": 1,
                    "batch_size": 1,
                    "scheduler_queue_wait_ms": 12,
                },
            ),
            SchedulerEvent(
                kind="token",
                request_id="request",
                token_id=1,
                cache_length=2,
                phase="prefill",
            ),
        )
    )

    events = runtime.tick()
    response = next(event.payload for event in events if event.kind == "response")

    assert response.scheduler_queue_wait_ms == 12
    assert response.queue_time_ms == 12


def test_runtime_cancellation_terminal_response_carries_stage_and_latency() -> None:
    scheduler = _FakeScheduler()
    tokenizer = _FakeTokenizer()
    runtime = NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(0,),
    )
    runtime.submit(_chat_request(stream=False))

    assert runtime.cancel("request")
    events = runtime.tick()
    response = next(event.payload for event in events if event.kind == "response")

    assert response.finish_reason == "cancelled"
    assert response.cancellation_stage == "python_waiting"
    assert response.cancellation_latency_ms == 2
