from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_worker.ipc import (
    ChatCompletionResponse,
    ModelStatus,
    SchedulerMetricsEvent,
    WorkerReady,
    decode_bootstrap_message,
    decode_event,
)
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    StepRequestResult,
    StepResult,
    execution_batch_field_names,
    execution_request_field_names,
)
from mlx_worker.native_mlx.mapping import load_weight_index
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.config import Qwen2ModelConfig
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.model import (
    Qwen2ForCausalLm,
    Qwen2NativeMlxExecutor,
)
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.weights import (
    Qwen2WeightAdapter,
)
from mlx_worker.native_mlx.registry import get_architecture_spec
from mlx_worker.native_mlx.worker import create_native_worker, run_native_worker


class FakeSocket:
    def __init__(self, payload: bytes = b"") -> None:
        self.sent: list[bytes] = []
        self._reader = io.BytesIO(payload)

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        return self._reader.read(size)


def _write_config_json(tmp_path: Path, payload: dict[str, object]) -> str:
    (tmp_path / "config.json").write_text(json.dumps(payload))
    return str(tmp_path)


def _write_valid_qwen2_model_dir(tmp_path: Path) -> str:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": False,
            }
        )
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ bos_token }}{{ messages }}"})
    )
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>"})
    )
    (tmp_path / "model.safetensors").write_text("placeholder")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "model.norm.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                }
            }
        )
    )
    return str(tmp_path)


def _tiny_qwen2_config() -> Qwen2ModelConfig:
    return Qwen2ModelConfig(
        architecture_class="Qwen2ForCausalLM",
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=False,
        quantization=None,
    )


def _native_request(model_path: str, request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        model=model_path,
        messages=[SimpleNamespace(role="user", content=f"hello {request_id}")],
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
        stop=(),
    )


def test_execution_boundaries_exclude_http_prompt_and_sse_fields() -> None:
    request_fields = set(execution_request_field_names())
    assert "request_id" in request_fields
    assert "token_ids" in request_fields
    assert "positions" in request_fields
    assert "cache_handle" in request_fields
    assert "temperature" in request_fields
    assert "top_p" in request_fields
    assert "messages" not in request_fields
    assert "prompt" not in request_fields
    assert "http_request" not in request_fields
    assert "sse_state" not in request_fields
    assert "queue_policy" not in request_fields
    assert execution_batch_field_names() == ("phase", "requests")


def test_registry_exposes_qwen2_known_good_checkpoint_and_probes() -> None:
    spec = get_architecture_spec("Qwen2ForCausalLM")

    assert spec is not None
    assert spec.known_good_checkpoint == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert [probe.name for probe in spec.compatibility_probes] == [
        "unsupported-llama-class",
        "missing-tokenizer-assets",
    ]


def test_create_native_worker_rejects_unsupported_architecture(tmp_path: Path) -> None:
    model_path = _write_config_json(
        tmp_path,
        {"architectures": ["LlamaForCausalLM"]},
    )

    try:
        create_native_worker(type("Cfg", (), {"model": model_path})())
    except Exception as exc:
        error = exc.error
    else:  # pragma: no cover
        raise AssertionError("expected native bootstrap failure")

    assert error.category == "unsupported_class"
    assert error.stage == "architecture_detection"
    assert error.code == "UNSUPPORTED_ARCHITECTURE_CLASS"


def test_create_native_worker_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    model_path = _write_config_json(tmp_path, {"not_architectures": []})

    try:
        create_native_worker(type("Cfg", (), {"model": model_path})())
    except Exception as exc:
        error = exc.error
    else:  # pragma: no cover
        raise AssertionError("expected native bootstrap failure")

    assert error.category == "malformed_checkpoint"
    assert error.stage == "architecture_detection"
    assert error.code == "MISSING_ARCHITECTURES"


def test_run_native_worker_reports_ready_and_serves_native_completion(
    tmp_path: Path,
) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)
    fake_socket = FakeSocket(
        b'{"type":"chat_completion","request":{"request_id":"req-1","model":"'
        + model_path.encode("utf-8")
        + b'","messages":[{"role":"user","content":"hello"}],"max_tokens":2,'
        + b'"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,'
        + b'"max_completion_tokens":32,"max_total_tokens_per_request":64,'
        + b'"stop":["LO"],"stream":false}}\n'
    )
    config = type("Cfg", (), {"model": model_path})()

    class FakeRawTokenizer:
        chat_template = "tmpl"
        eos_token_id = 99

        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": [1, 2, 3]}

        def decode(self, token_ids, **_kwargs):
            mapping = {7: "HE", 8: "LLO", 99: ""}
            return "".join(mapping[token] for token in token_ids)

    def fake_tokenizer_loader(_model_path: Path):
        return SimpleNamespace(tokenizer=FakeRawTokenizer())

    class FakeExecutor:
        def __init__(self) -> None:
            self.cache_lengths: dict[str, int] = {}

        def load(self, options):
            self.options = options

        def create_cache(self, request_id):
            handle = f"cache-{request_id}"
            self.cache_lengths[handle] = 0
            return handle

        def prefill_batch(self, batch):
            request = batch.requests[0]
            self.cache_lengths[request.cache_handle] = len(request.token_ids)
            return StepResult(
                phase="prefill",
                results=(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids),
                        finished=False,
                        next_token_id=7,
                    ),
                ),
                step_time_ms=1,
            )

        def decode_batch(self, batch):
            request = batch.requests[0]
            self.cache_lengths[request.cache_handle] = (
                self.cache_lengths[request.cache_handle] + 1
            )
            return StepResult(
                phase="decode",
                results=(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=self.cache_lengths[request.cache_handle],
                        finished=False,
                        next_token_id=8,
                    ),
                ),
                step_time_ms=1,
            )

        def cache_len(self, cache_handle):
            return self.cache_lengths.get(cache_handle, 0)

        def release(self, cache_handle):
            self.cache_lengths.pop(cache_handle, None)
            return None

    import mlx_worker.native_mlx.worker as native_worker

    original_loader = native_worker._load_tokenizer_wrapper
    original_executor = native_worker.build_native_executor
    original_select = native_worker.select.select
    native_worker._load_tokenizer_wrapper = fake_tokenizer_loader
    native_worker.build_native_executor = (
        lambda architecture, model_config, weight_index, weight_plan: FakeExecutor()
    )
    native_worker.select.select = lambda *args, **kwargs: ([], [], [])
    try:
        exit_code = run_native_worker(fake_socket, config)
    finally:
        native_worker._load_tokenizer_wrapper = original_loader
        native_worker.build_native_executor = original_executor
        native_worker.select.select = original_select

    assert exit_code == 0
    decoded = [decode_bootstrap_message(chunk) for chunk in fake_socket.sent]
    statuses = [item for item in decoded if isinstance(item, ModelStatus)]
    assert [
        status.progress.current_phase for status in statuses if status.progress
    ] == [
        "architecture_detection",
        "artifact_validation",
        "weight_mapping",
        "native_executor_construction",
        "prompt_tokenizer_readiness",
        "deterministic_warmup",
        "deterministic_warmup",
    ]
    assert statuses[-1].state == "ready"
    assert any(isinstance(item, WorkerReady) for item in decoded)
    responses = []
    for chunk in fake_socket.sent:
        try:
            item = decode_event(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(item, ChatCompletionResponse):
            responses.append(item)
    response = responses[-1]
    assert isinstance(response, ChatCompletionResponse)
    assert response.backend == "native-mlx"
    assert response.text == "HEL"
    assert response.finish_reason == "stop"


def test_native_scheduler_allows_request_to_join_active_decode_batch(
    tmp_path: Path,
) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)

    class FakeRawTokenizer:
        chat_template = "tmpl"
        eos_token_id = 99

        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": [1, 2, 3]}

        def decode(self, token_ids, **_kwargs):
            mapping = {7: "A", 8: "B", 99: ""}
            return "".join(mapping[token] for token in token_ids)

    class FakeExecutor:
        def __init__(self) -> None:
            self.cache_lengths: dict[str, int] = {}
            self.prefill_calls: list[tuple[str, ...]] = []
            self.decode_calls: list[tuple[str, ...]] = []

        def load(self, _options):
            return None

        def create_cache(self, request_id):
            handle = f"cache-{request_id}"
            self.cache_lengths[handle] = 0
            return handle

        def prefill_batch(self, batch):
            self.prefill_calls.append(tuple(item.request_id for item in batch.requests))
            results = []
            for request in batch.requests:
                self.cache_lengths[request.cache_handle] = len(request.token_ids)
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids),
                        finished=False,
                        next_token_id=7,
                    )
                )
            return StepResult(
                phase="prefill",
                results=tuple(results),
                step_time_ms=1,
            )

        def decode_batch(self, batch):
            self.decode_calls.append(tuple(item.request_id for item in batch.requests))
            results = []
            for request in batch.requests:
                self.cache_lengths[request.cache_handle] = (
                    self.cache_lengths[request.cache_handle] + 1
                )
                next_token_id = (
                    8 if self.cache_lengths[request.cache_handle] == 4 else 99
                )
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=self.cache_lengths[request.cache_handle],
                        finished=False,
                        next_token_id=next_token_id,
                    )
                )
            return StepResult(
                phase="decode",
                results=tuple(results),
                step_time_ms=1,
            )

        def cache_len(self, cache_handle):
            return self.cache_lengths.get(cache_handle, 0)

        def release(self, cache_handle):
            self.cache_lengths.pop(cache_handle, None)
            return None

    import mlx_worker.native_mlx.worker as native_worker

    original_loader = native_worker._load_tokenizer_wrapper
    native_worker._load_tokenizer_wrapper = lambda _path: SimpleNamespace(
        tokenizer=FakeRawTokenizer()
    )
    try:
        responses: list[ChatCompletionResponse] = []
        metrics: list[SchedulerMetricsEvent] = []
        executor = FakeExecutor()
        scheduler = native_worker.NativeContinuousScheduler(
            executor=executor,
            options=SimpleNamespace(
                model=model_path, architecture_class="Qwen2ForCausalLM"
            ),
            model_path=Path(model_path),
            model_ref=model_path,
            stage_callback=None,
            emit_response=responses.append,
            emit_error=lambda error: (_ for _ in ()).throw(AssertionError(str(error))),
            emit_metrics=metrics.append,
        )
        scheduler.warmup()
        emitted_1: list[str] = []
        emitted_2: list[str] = []
        scheduler.submit(_native_request(model_path, "req-1"), emitted_1.append)
        scheduler.tick()
        scheduler.submit(_native_request(model_path, "req-2"), emitted_2.append)
        while not scheduler.idle():
            scheduler.tick()
    finally:
        native_worker._load_tokenizer_wrapper = original_loader

    assert executor.prefill_calls == [("warmup",), ("req-1",), ("req-2",)]
    assert ("req-1", "req-2") in executor.decode_calls
    assert emitted_1 == ["A", "B"]
    assert emitted_2 == ["A", "B"]
    assert [response.request_id for response in responses] == ["req-1", "req-2"]
    assert any(
        event.phase == "decode"
        and event.batch_size == 2
        and event.running_requests == 1
        for event in metrics
    )
    assert scheduler._executor.cache_lengths == {}


def test_native_scheduler_rejects_non_positive_prefill_chunk_size() -> None:
    try:
        __import__(
            "mlx_worker.native_mlx.worker", fromlist=["NativeContinuousScheduler"]
        ).NativeContinuousScheduler(
            executor=SimpleNamespace(),
            options=SimpleNamespace(
                model="test-model", architecture_class="Qwen2ForCausalLM"
            ),
            model_path=Path("/tmp/model"),
            model_ref="test-model",
            stage_callback=None,
            emit_response=lambda _response: None,
            emit_error=lambda _error: None,
            emit_metrics=lambda _metrics: None,
            prefill_step_size=0,
        )
    except ValueError as exc:
        assert str(exc) == "native-mlx prefill chunk size must be positive"
    else:  # pragma: no cover
        raise AssertionError("expected invalid prefill chunk size failure")


def test_native_scheduler_interleaves_decode_before_long_prefill_chunks(
    tmp_path: Path,
) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)

    def native_request(request_id: str, content: str) -> SimpleNamespace:
        request = _native_request(model_path, request_id)
        request.messages = [SimpleNamespace(role="user", content=content)]
        return request

    class FakeRawTokenizer:
        chat_template = "tmpl"
        eos_token_id = 99

        def apply_chat_template(self, messages, **_kwargs):
            content = messages[0]["content"]
            if content == "ping":
                return {"input_ids": [0]}
            if content == "long":
                return {"input_ids": [1, 2, 3, 4, 5]}
            if content == "short":
                return {"input_ids": [9]}
            return {"input_ids": [1, 2, 3]}

        def decode(self, token_ids, **_kwargs):
            mapping = {7: "A", 8: "B", 99: ""}
            return "".join(mapping[token] for token in token_ids)

    class FakeExecutor:
        def __init__(self) -> None:
            self.cache_lengths: dict[str, int] = {}
            self.call_order: list[
                tuple[str, tuple[str, ...], tuple[tuple[int, ...], ...]]
            ] = []

        def load(self, _options):
            return None

        def create_cache(self, request_id):
            handle = f"cache-{request_id}"
            self.cache_lengths[handle] = 0
            return handle

        def prefill_batch(self, batch):
            self.call_order.append(
                (
                    "prefill",
                    tuple(item.request_id for item in batch.requests),
                    tuple(item.token_ids for item in batch.requests),
                )
            )
            results = []
            for request in batch.requests:
                self.cache_lengths[request.cache_handle] += len(request.token_ids)
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=self.cache_lengths[request.cache_handle],
                        finished=False,
                        next_token_id=7,
                    )
                )
            return StepResult(
                phase="prefill",
                results=tuple(results),
                step_time_ms=1,
            )

        def decode_batch(self, batch):
            self.call_order.append(
                (
                    "decode",
                    tuple(item.request_id for item in batch.requests),
                    tuple(item.token_ids for item in batch.requests),
                )
            )
            results = []
            for request in batch.requests:
                self.cache_lengths[request.cache_handle] += 1
                next_token_id = (
                    99
                    if request.request_id == "req-short"
                    or self.cache_lengths[request.cache_handle] >= 7
                    else 8
                )
                results.append(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=self.cache_lengths[request.cache_handle],
                        finished=False,
                        next_token_id=next_token_id,
                    )
                )
            return StepResult(
                phase="decode",
                results=tuple(results),
                step_time_ms=1,
            )

        def cache_len(self, cache_handle):
            return self.cache_lengths.get(cache_handle, 0)

        def release(self, cache_handle):
            self.cache_lengths.pop(cache_handle, None)
            return None

    import mlx_worker.native_mlx.worker as native_worker

    original_loader = native_worker._load_tokenizer_wrapper
    native_worker._load_tokenizer_wrapper = lambda _path: SimpleNamespace(
        tokenizer=FakeRawTokenizer()
    )
    try:
        responses: list[ChatCompletionResponse] = []
        metrics: list[SchedulerMetricsEvent] = []
        executor = FakeExecutor()
        scheduler = native_worker.NativeContinuousScheduler(
            executor=executor,
            options=SimpleNamespace(
                model=model_path, architecture_class="Qwen2ForCausalLM"
            ),
            model_path=Path(model_path),
            model_ref=model_path,
            stage_callback=None,
            emit_response=responses.append,
            emit_error=lambda error: (_ for _ in ()).throw(AssertionError(str(error))),
            emit_metrics=metrics.append,
            prefill_step_size=2,
        )
        scheduler.warmup()
        executor.call_order.clear()
        metrics.clear()

        scheduler.submit(native_request("req-long", "long"), None)
        scheduler.tick()
        scheduler.submit(native_request("req-short", "short"), None)
        while not scheduler.idle():
            scheduler.tick()
    finally:
        native_worker._load_tokenizer_wrapper = original_loader

    assert executor.call_order[:4] == [
        ("prefill", ("req-long",), ((1, 2),)),
        ("prefill", ("req-long", "req-short"), ((3, 4), (9,))),
        ("decode", ("req-short",), ((7,),)),
        ("prefill", ("req-long",), ((5,),)),
    ]
    assert [event.phase for event in metrics[:4]] == [
        "prefill",
        "prefill",
        "decode",
        "prefill",
    ]
    assert [event.scheduled_tokens for event in metrics[:4]] == [2, 3, 1, 1]
    assert [response.request_id for response in responses] == ["req-short", "req-long"]
    assert scheduler._executor.cache_lengths == {}


def test_native_scheduler_cancels_running_request_and_keeps_other_request_progressing(
    tmp_path: Path,
) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)

    class FakeRawTokenizer:
        chat_template = "tmpl"
        eos_token_id = 99

        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": [1, 2, 3]}

        def decode(self, token_ids, **_kwargs):
            mapping = {7: "A", 8: "B", 99: ""}
            return "".join(mapping[token] for token in token_ids)

    class FakeExecutor:
        def __init__(self) -> None:
            self.cache_lengths: dict[str, int] = {}
            self.decode_calls: list[tuple[str, ...]] = []

        def load(self, _options):
            return None

        def create_cache(self, request_id):
            handle = f"cache-{request_id}"
            self.cache_lengths[handle] = 0
            return handle

        def prefill_batch(self, batch):
            return StepResult(
                phase="prefill",
                results=tuple(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids),
                        finished=False,
                        next_token_id=7,
                    )
                    for request in batch.requests
                ),
                step_time_ms=1,
            )

        def decode_batch(self, batch):
            self.decode_calls.append(tuple(item.request_id for item in batch.requests))
            return StepResult(
                phase="decode",
                results=tuple(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids) + 1,
                        finished=False,
                        next_token_id=99 if request.request_id == "req-1" else 8,
                    )
                    for request in batch.requests
                ),
                step_time_ms=1,
            )

        def cache_len(self, cache_handle):
            return self.cache_lengths.get(cache_handle, 0)

        def release(self, cache_handle):
            self.cache_lengths.pop(cache_handle, None)

    import mlx_worker.native_mlx.worker as native_worker

    original_loader = native_worker._load_tokenizer_wrapper
    native_worker._load_tokenizer_wrapper = lambda _path: SimpleNamespace(
        tokenizer=FakeRawTokenizer()
    )
    try:
        responses: list[ChatCompletionResponse] = []
        executor = FakeExecutor()
        scheduler = native_worker.NativeContinuousScheduler(
            executor=executor,
            options=SimpleNamespace(
                model=model_path, architecture_class="Qwen2ForCausalLM"
            ),
            model_path=Path(model_path),
            model_ref=model_path,
            stage_callback=None,
            emit_response=responses.append,
            emit_error=lambda error: (_ for _ in ()).throw(AssertionError(str(error))),
            emit_metrics=lambda _metrics: None,
        )
        scheduler.warmup()
        scheduler.submit(_native_request(model_path, "req-1"), None)
        scheduler.submit(_native_request(model_path, "req-2"), None)
        scheduler.tick()
        assert scheduler.cancel("req-2")
        while not scheduler.idle():
            scheduler.tick()
    finally:
        native_worker._load_tokenizer_wrapper = original_loader

    assert executor.decode_calls == [("req-1",)]
    assert [
        (response.request_id, response.finish_reason) for response in responses
    ] == [
        ("req-2", "cancelled"),
        ("req-1", "stop"),
    ]
    assert scheduler._executor.cache_lengths == {}


def test_native_scheduler_cleans_up_request_error_without_leaking_other_state(
    tmp_path: Path,
) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)

    class FakeRawTokenizer:
        chat_template = "tmpl"
        eos_token_id = 99

        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": [1, 2, 3]}

        def decode(self, token_ids, **_kwargs):
            mapping = {7: "A", 99: ""}
            return "".join(mapping[token] for token in token_ids)

    class FakeExecutor:
        def __init__(self) -> None:
            self.cache_lengths: dict[str, int] = {}

        def load(self, _options):
            return None

        def create_cache(self, request_id):
            handle = f"cache-{request_id}"
            self.cache_lengths[handle] = 0
            return handle

        def prefill_batch(self, batch):
            return StepResult(
                phase="prefill",
                results=tuple(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids),
                        finished=False,
                        next_token_id=7 if request.request_id == "req-1" else None,
                        error_code=None
                        if request.request_id == "req-1"
                        else "KV_EXHAUSTED",
                    )
                    for request in batch.requests
                ),
                step_time_ms=1,
            )

        def decode_batch(self, batch):
            return StepResult(
                phase="decode",
                results=tuple(
                    StepRequestResult(
                        request_id=request.request_id,
                        token_ids=request.token_ids,
                        logits=None,
                        cache_handle=request.cache_handle,
                        cache_length=len(request.token_ids) + 1,
                        finished=False,
                        next_token_id=99,
                    )
                    for request in batch.requests
                ),
                step_time_ms=1,
            )

        def cache_len(self, cache_handle):
            return self.cache_lengths.get(cache_handle, 0)

        def release(self, cache_handle):
            self.cache_lengths.pop(cache_handle, None)

    import mlx_worker.native_mlx.worker as native_worker

    original_loader = native_worker._load_tokenizer_wrapper
    native_worker._load_tokenizer_wrapper = lambda _path: SimpleNamespace(
        tokenizer=FakeRawTokenizer()
    )
    try:
        responses: list[ChatCompletionResponse] = []
        errors: list[tuple[str, str]] = []
        executor = FakeExecutor()
        scheduler = native_worker.NativeContinuousScheduler(
            executor=executor,
            options=SimpleNamespace(
                model=model_path, architecture_class="Qwen2ForCausalLM"
            ),
            model_path=Path(model_path),
            model_ref=model_path,
            stage_callback=None,
            emit_response=responses.append,
            emit_error=lambda error: errors.append((error.request_id, error.code)),
            emit_metrics=lambda _metrics: None,
        )
        scheduler.warmup()
        scheduler.submit(_native_request(model_path, "req-1"), None)
        scheduler.submit(_native_request(model_path, "req-2"), None)
        while not scheduler.idle():
            scheduler.tick()
    finally:
        native_worker._load_tokenizer_wrapper = original_loader

    assert errors == [("req-2", "KV_EXHAUSTED")]
    assert [
        (response.request_id, response.finish_reason) for response in responses
    ] == [("req-1", "stop")]
    assert scheduler._executor.cache_lengths == {}


def test_run_native_worker_rejects_missing_tokenizer_assets(tmp_path: Path) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    fake_socket = FakeSocket()
    config = type("Cfg", (), {"model": model_path})()

    class FakeExecutor:
        def load(self, options):
            self.options = options

        def create_cache(self, request_id):
            return f"cache-{request_id}"

        def prefill_batch(self, batch):  # pragma: no cover - startup test only.
            raise NotImplementedError

        def decode_batch(self, batch):  # pragma: no cover - startup test only.
            raise NotImplementedError

        def release(self, cache_handle):
            return None

        def forward_token_ids(self, token_ids):
            return mx.array([[[0.0, 1.0, 2.0, 3.0]]], dtype=mx.float32)

        def prefill_then_decode_tokens(self, token_ids, decode_steps):
            return (
                [7] * (decode_steps + 1),
                [len(token_ids) + i for i in range(decode_steps + 1)],
                1,
            )

    import mlx_worker.native_mlx.worker as native_worker

    original_executor = native_worker.build_native_executor
    native_worker.build_native_executor = (
        lambda architecture, model_config, weight_index, weight_plan: FakeExecutor()
    )
    try:
        exit_code = run_native_worker(fake_socket, config)
    finally:
        native_worker.build_native_executor = original_executor

    assert exit_code == 1
    decoded = [decode_bootstrap_message(chunk) for chunk in fake_socket.sent]
    failed_status = next(
        item
        for item in decoded
        if isinstance(item, ModelStatus) and item.state == "failed"
    )
    assert failed_status.last_error is not None
    assert failed_status.last_error.category == "malformed_checkpoint"
    assert failed_status.last_error.stage == "prompt_tokenizer_readiness"
    assert failed_status.last_error.code == "MISSING_TOKENIZER_ASSET"


def test_qwen2_weight_adapter_builds_canonical_plan(tmp_path: Path) -> None:
    _write_valid_qwen2_model_dir(tmp_path)

    plan = Qwen2WeightAdapter().build_plan(load_weight_index(tmp_path))

    assert plan.architecture_class == "Qwen2ForCausalLM"
    assert plan.source_files == ("model.safetensors",)
    assert any(
        entry.canonical_name == "model.layers.0.self_attn.q_proj.weight"
        for entry in plan.entries
    )


def test_qwen2_model_matches_mlx_lm_reference_for_small_config() -> None:
    from mlx_lm.models.qwen2 import Model as ReferenceModel
    from mlx_lm.models.qwen2 import ModelArgs as ReferenceArgs

    config = _tiny_qwen2_config()
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)
    mx.random.seed(0)
    reference_model = ReferenceModel(
        ReferenceArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=1000000.0,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=False,
        )
    )

    native_logits = native_model(inputs)
    reference_logits = reference_model(inputs)
    mx.eval(native_logits, reference_logits)

    assert native_logits.shape == reference_logits.shape == (1, 4, 256)
    assert str(native_logits.dtype) == str(reference_logits.dtype)
    assert bool(mx.all(mx.isfinite(native_logits)).item())
    assert bool(
        mx.allclose(native_logits, reference_logits, atol=1e-5, rtol=1e-5).item()
    )


def test_qwen2_executor_prefill_returns_logits_and_greedy_token() -> None:
    config = _tiny_qwen2_config()
    weight_index = SimpleNamespace(model_path=Path("/tmp"))
    weight_plan = SimpleNamespace(entries=(), source_files=())
    executor = Qwen2NativeMlxExecutor.__new__(Qwen2NativeMlxExecutor)
    executor.model_path = Path("/tmp")
    executor.model_config = config
    executor.weight_plan = weight_plan
    executor.weight_index = weight_index
    mx.random.seed(0)
    executor.model = Qwen2ForCausalLm(config)
    executor._request_caches = {}
    cache_handle = executor.create_cache("req-1")

    batch = ExecutionBatch(
        phase="prefill",
        requests=(
            ExecutionRequest(
                request_id="req-1",
                token_ids=(1, 2, 3),
                positions=(0, 1, 2),
                cache_handle=cache_handle,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
            ),
        ),
    )

    result = executor.prefill_batch(batch)

    assert result.phase == "prefill"
    assert result.step_time_ms >= 1
    assert len(result.results) == 1
    request_result = result.results[0]
    assert request_result.logits.shape == (3, 256)
    assert bool(mx.all(mx.isfinite(request_result.logits)).item())
    assert request_result.next_token_id is not None
    assert request_result.cache_length == 3


def test_qwen2_executor_decode_appends_one_token_without_recomputing_prompt() -> None:
    config = _tiny_qwen2_config()
    executor = Qwen2NativeMlxExecutor.__new__(Qwen2NativeMlxExecutor)
    executor.model_path = Path("/tmp")
    executor.model_config = config
    executor.weight_plan = SimpleNamespace(entries=(), source_files=())
    executor.weight_index = SimpleNamespace(model_path=Path("/tmp"))
    executor._request_caches = {}
    mx.random.seed(0)
    executor.model = Qwen2ForCausalLm(config)

    cache_handle = executor.create_cache("req-1")
    prefill = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                ExecutionRequest(
                    request_id="req-1",
                    token_ids=(1, 2, 3),
                    positions=(0, 1, 2),
                    cache_handle=cache_handle,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
            ),
        )
    )
    prefill_next = int(prefill.results[0].next_token_id)

    decode = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                ExecutionRequest(
                    request_id="req-1",
                    token_ids=(prefill_next,),
                    positions=(3,),
                    cache_handle=cache_handle,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
            ),
        )
    )

    assert decode.results[0].cache_length == 4
    assert executor.cache_len(cache_handle) == 4
    assert decode.results[0].next_token_id is not None


def test_qwen2_executor_prefill_resumes_cache_across_chunks() -> None:
    config = _tiny_qwen2_config()
    executor = Qwen2NativeMlxExecutor.__new__(Qwen2NativeMlxExecutor)
    executor.model_path = Path("/tmp")
    executor.model_config = config
    executor.weight_plan = SimpleNamespace(entries=(), source_files=())
    executor.weight_index = SimpleNamespace(model_path=Path("/tmp"))
    executor._request_caches = {}
    mx.random.seed(0)
    executor.model = Qwen2ForCausalLm(config)

    full_handle = executor.create_cache("full")
    chunked_handle = executor.create_cache("chunked")
    try:
        full_prefill = executor.prefill_batch(
            ExecutionBatch(
                phase="prefill",
                requests=(
                    ExecutionRequest(
                        request_id="full",
                        token_ids=(1, 2, 3, 4),
                        positions=(0, 1, 2, 3),
                        cache_handle=full_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )
        first_chunk = executor.prefill_batch(
            ExecutionBatch(
                phase="prefill",
                requests=(
                    ExecutionRequest(
                        request_id="chunked",
                        token_ids=(1, 2),
                        positions=(0, 1),
                        cache_handle=chunked_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )
        second_chunk = executor.prefill_batch(
            ExecutionBatch(
                phase="prefill",
                requests=(
                    ExecutionRequest(
                        request_id="chunked",
                        token_ids=(3, 4),
                        positions=(2, 3),
                        cache_handle=chunked_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )

        full_next = int(full_prefill.results[0].next_token_id)
        chunked_next = int(second_chunk.results[0].next_token_id)
        assert first_chunk.results[0].cache_length == 2
        assert second_chunk.results[0].cache_length == 4
        assert chunked_next == full_next
        assert bool(
            mx.allclose(
                second_chunk.results[0].logits[-1],
                full_prefill.results[0].logits[-1],
                atol=1e-5,
                rtol=1e-5,
            ).item()
        )

        full_decode = executor.decode_batch(
            ExecutionBatch(
                phase="decode",
                requests=(
                    ExecutionRequest(
                        request_id="full",
                        token_ids=(full_next,),
                        positions=(4,),
                        cache_handle=full_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )
        chunked_decode = executor.decode_batch(
            ExecutionBatch(
                phase="decode",
                requests=(
                    ExecutionRequest(
                        request_id="chunked",
                        token_ids=(chunked_next,),
                        positions=(4,),
                        cache_handle=chunked_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )

        assert (
            full_decode.results[0].next_token_id
            == chunked_decode.results[0].next_token_id
        )
        assert chunked_decode.results[0].cache_length == 5
        assert executor.cache_len(chunked_handle) == 5
    finally:
        executor.release(full_handle)
        executor.release(chunked_handle)


def test_qwen2_executor_rejects_cross_request_cache_handle() -> None:
    config = _tiny_qwen2_config()
    executor = Qwen2NativeMlxExecutor.__new__(Qwen2NativeMlxExecutor)
    executor.model_path = Path("/tmp")
    executor.model_config = config
    executor.weight_plan = SimpleNamespace(entries=(), source_files=())
    executor.weight_index = SimpleNamespace(model_path=Path("/tmp"))
    executor._request_caches = {}
    mx.random.seed(0)
    executor.model = Qwen2ForCausalLm(config)

    cache_handle = executor.create_cache("req-1")
    try:
        executor.prefill_batch(
            ExecutionBatch(
                phase="prefill",
                requests=(
                    ExecutionRequest(
                        request_id="req-2",
                        token_ids=(1, 2),
                        positions=(0, 1),
                        cache_handle=cache_handle,
                        max_new_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            )
        )
    except ValueError as exc:
        assert "different request" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected request/cache isolation failure")
    assert executor.cache_len(cache_handle) == 0


def test_qwen2_executor_release_clears_native_state() -> None:
    config = _tiny_qwen2_config()
    executor = Qwen2NativeMlxExecutor.__new__(Qwen2NativeMlxExecutor)
    executor.model_path = Path("/tmp")
    executor.model_config = config
    executor.weight_plan = SimpleNamespace(entries=(), source_files=())
    executor.weight_index = SimpleNamespace(model_path=Path("/tmp"))
    executor._request_caches = {}
    mx.random.seed(0)
    executor.model = Qwen2ForCausalLm(config)

    cache_handle = executor.create_cache("req-1")
    executor.release(cache_handle)

    assert executor.cache_len(cache_handle) == 0
