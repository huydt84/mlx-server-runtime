"""Interface-level tests for the native MLX architecture."""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.bootstrap import (
    NativeBootstrapFailure,
    build_native_artifacts,
    detect_native_architecture,
)
from mlx_worker.native_mlx.attention import (
    DenseReferenceAttentionBackend,
    PagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.cache import DenseKVCacheBackend, PagedKVCacheBackend
from mlx_worker.native_mlx.cache_coordinator import NativeCacheCoordinator
from mlx_worker.native_mlx.executor import MlxGenerationExecutor
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


def test_models_directory_has_one_qwen2_module() -> None:
    models = Path(__file__).parents[1] / "mlx_worker/native_mlx/models"
    assert sorted(path.name for path in models.glob("*.py")) == [
        "__init__.py",
        "qwen2.py",
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
    geometry = spec.cache_geometry(_tiny_qwen2_config())
    assert geometry.num_layers == 2
    assert geometry.num_kv_heads == 2
    assert geometry.head_dim == 4
    assert Qwen2ForCausalLM.__module__.endswith("models.qwen2")
    assert Qwen2WeightAdapter.__module__.endswith("models.qwen2")


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
    paged_context = PagedMetalAttentionBackend().contexts(
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
    assert paged_backend.metrics()["attention_backend"] == "native-metal-paged"


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
    released: list[str] = field(default_factory=list)

    def probe(self, token_ids: tuple[int, ...]) -> Any:
        del token_ids
        return None

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: Any = None,
    ) -> Any:
        del token_ids, probe
        handle = f"cache-{request_id}"
        self.lengths[handle] = 0
        return type(
            "Admission",
            (),
            {
                "cache_handle": handle,
                "cache_length": 0,
                "reused_tokens": 0,
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
    prefill_step_size: int = 256,
) -> tuple[NativeContinuousScheduler, _FakeCacheCoordinator]:
    cache_coordinator = _FakeCacheCoordinator(executor.lengths)
    return (
        NativeContinuousScheduler(
            executor,
            cache_coordinator,  # type: ignore[arg-type]
            prefill_step_size=prefill_step_size,
        ),
        cache_coordinator,
    )


def _schedulable(request_id: str, tokens: tuple[int, ...]) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling=SamplingParams(),
        enqueued_at=time.perf_counter(),
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
    scheduler, _ = _scheduler(executor, prefill_step_size=8)
    scheduler.submit(_schedulable("running", (1,)))
    scheduler.tick()
    scheduler.submit(_schedulable("waiting", (2,)))

    scheduler.tick()

    assert len(executor.batches) == 2
    mixed = executor.batches[-1]
    assert mixed.forward_mode is ForwardMode.MIXED
    assert [request.phase for request in mixed.requests] == ["decode", "prefill"]


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


def test_scheduler_cancellation_releases_only_after_runtime_finish() -> None:
    executor = _FakeExecutor()
    scheduler, cache_coordinator = _scheduler(executor)
    scheduler.submit(_schedulable("request", (1,)))
    scheduler.tick()

    assert scheduler.cancel("request")
    events = scheduler.tick()
    assert any(event.kind == "cancelled" for event in events)
    assert cache_coordinator.released == []

    scheduler.finish("request")
    assert cache_coordinator.released == ["cache-request"]


@dataclass
class _FakeScheduler:
    submitted: list[SchedulableRequest] = field(default_factory=list)
    events: list[tuple[SchedulerEvent, ...]] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)

    def submit(self, request: SchedulableRequest) -> None:
        self.submitted.append(request)

    def cancel(self, request_id: str) -> bool:
        return any(item.request_id == request_id for item in self.submitted)

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
