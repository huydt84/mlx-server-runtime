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
from mlx_worker.native_mlx.cache import DenseKVCacheBackend
from mlx_worker.native_mlx.executor import MlxGenerationExecutor
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
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
) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
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
        keys = mx.zeros((batch, 1, sequence, 2))
        values = mx.zeros((batch, 1, sequence, 2))
        for cache in forward_batch.layer_caches:
            cache.update_and_fetch(keys, values)
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


def _executor(model: _RecordingModel | None = None) -> MlxGenerationExecutor:
    active_model = model or _RecordingModel()
    return MlxGenerationExecutor(
        architecture_class="FakeForCausalLM",
        model=active_model,
        cache_backend=DenseKVCacheBackend(num_layers=active_model.num_layers),
    )


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
    assert spec.create_cache_backend(_tiny_qwen2_config()).num_layers == 2
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
    executor = _executor(model)
    first = executor.create_cache("first")
    second = executor.create_cache("second")

    result = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
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
    executor = _executor(model)
    first = executor.create_cache("first")
    second = executor.create_cache("second")
    executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                _request("first", (1, 2), (0, 1), first),
                _request("second", (3,), (0,), second),
            ),
        )
    )
    model.calls.clear()

    result = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                _request("first", (7,), (2,), first),
                _request("second", (8,), (1,), second),
            ),
        )
    )

    assert model.calls == [((2, 1), (1, 1))]
    assert [item.cache_length for item in result.results] == [3, 2]


def test_executor_failure_does_not_commit_or_release_request_caches() -> None:
    model = _RecordingModel(fail_after_cache_stage=True)
    executor = _executor(model)
    first = executor.create_cache("first")
    second = executor.create_cache("second")

    with pytest.raises(RuntimeError, match="model failed"):
        executor.prefill_batch(
            ExecutionBatch(
                phase="prefill",
                requests=(
                    _request("first", (1,), (0,), first),
                    _request("second", (2,), (0,), second),
                ),
            )
        )

    assert executor.cache_len(first) == 0
    assert executor.cache_len(second) == 0
    assert executor.cache_backend.get(first, "first").request_id == "first"
    assert executor.cache_backend.get(second, "second").request_id == "second"


@dataclass
class _FakeExecutor:
    lengths: dict[str, int] = field(default_factory=dict)
    batches: list[ExecutionBatch] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    def load(self, options: Any) -> None:
        del options

    def create_cache(self, request_id: str) -> str:
        handle = f"cache-{request_id}"
        self.lengths[handle] = 0
        return handle

    def prefill_batch(self, batch: ExecutionBatch) -> StepResult:
        return self._step(batch)

    def decode_batch(self, batch: ExecutionBatch) -> StepResult:
        return self._step(batch)

    def _step(self, batch: ExecutionBatch) -> StepResult:
        self.batches.append(batch)
        results = []
        for request in batch.requests:
            assert request.cache_handle is not None
            self.lengths[request.cache_handle] += len(request.token_ids)
            results.append(
                StepRequestResult(
                    request_id=request.request_id,
                    token_ids=request.token_ids,
                    cache_handle=request.cache_handle,
                    cache_length=self.lengths[request.cache_handle],
                    next_token_id=9,
                )
            )
        return StepResult(batch.phase, tuple(results), 1)

    def cache_len(self, cache_handle: str | None) -> int:
        return self.lengths.get(cache_handle or "", 0)

    def release(self, cache_handle: str | None) -> None:
        if cache_handle is not None:
            self.released.append(cache_handle)
            self.lengths.pop(cache_handle, None)


def _schedulable(request_id: str, tokens: tuple[int, ...]) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling=SamplingParams(),
        enqueued_at=time.perf_counter(),
    )


def test_scheduler_owns_chunking_and_emits_typed_token_events() -> None:
    executor = _FakeExecutor()
    scheduler = NativeContinuousScheduler(executor, prefill_step_size=2)
    scheduler.submit(_schedulable("request", (1, 2, 3, 4, 5)))

    first = scheduler.tick()
    second = scheduler.tick()
    third = scheduler.tick()

    assert [batch.phase for batch in executor.batches] == [
        "prefill",
        "prefill",
        "prefill",
    ]
    assert [batch.requests[0].token_ids for batch in executor.batches] == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    assert not any(event.kind == "token" for event in first + second)
    assert any(event.kind == "token" and event.token_id == 9 for event in third)


def test_scheduler_decode_is_dispatched_before_new_prefill() -> None:
    executor = _FakeExecutor()
    scheduler = NativeContinuousScheduler(executor, prefill_step_size=8)
    scheduler.submit(_schedulable("running", (1,)))
    scheduler.tick()
    scheduler.submit(_schedulable("waiting", (2,)))

    scheduler.tick()

    assert [batch.phase for batch in executor.batches[-2:]] == [
        "decode",
        "prefill",
    ]


def test_scheduler_cancellation_releases_only_after_runtime_finish() -> None:
    executor = _FakeExecutor()
    scheduler = NativeContinuousScheduler(executor)
    scheduler.submit(_schedulable("request", (1,)))
    scheduler.tick()

    assert scheduler.cancel("request")
    events = scheduler.tick()
    assert any(event.kind == "cancelled" for event in events)
    assert executor.released == []

    scheduler.finish("request")
    assert executor.released == ["cache-request"]


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
