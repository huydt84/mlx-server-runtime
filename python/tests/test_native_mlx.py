from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_worker.ipc import ModelStatus, WorkerError, decode_bootstrap_message
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
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
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)


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


def test_run_native_worker_reports_supported_class_bug(tmp_path: Path) -> None:
    model_path = _write_valid_qwen2_model_dir(tmp_path)
    fake_socket = FakeSocket()
    config = type("Cfg", (), {"model": model_path})()

    def fake_tokenizer_loader(_model_path: Path):
        return SimpleNamespace(tokenizer=SimpleNamespace(chat_template="tmpl"))

    def fake_build_token_ids(_model_path: Path, _messages):
        return [1, 2, 3]

    def fake_compare(_checkpoint: str, _executor, token_ids, **_kwargs):
        return SimpleNamespace(
            checkpoint=_checkpoint,
            token_ids=tuple(token_ids),
            prefill_logits_shape=(1, len(token_ids), 4),
            prefill_logits_dtype="float32",
            prefill_max_abs_diff=0.0,
            decode_max_abs_diff=0.0,
            tolerance_atol=0.02,
            tolerance_rtol=0.02,
            tolerance_ok=True,
            native_tokens=(7, 7, 7),
            reference_tokens=(7, 7, 7),
            cache_lengths=(len(token_ids), len(token_ids) + 1, len(token_ids) + 2),
            prefill_time_ms=1,
            token_ok=True,
        )

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

    original_loader = native_worker._load_tokenizer_wrapper
    original_build = native_worker.build_finalized_token_ids
    original_compare = native_worker.compare_native_prefill_decode_to_mlx_lm
    original_executor = native_worker.build_native_executor
    native_worker._load_tokenizer_wrapper = fake_tokenizer_loader
    native_worker.build_finalized_token_ids = fake_build_token_ids
    native_worker.compare_native_prefill_decode_to_mlx_lm = fake_compare
    native_worker.build_native_executor = (
        lambda architecture, model_config, weight_index, weight_plan: FakeExecutor()
    )
    try:
        exit_code = run_native_worker(fake_socket, config)
    finally:
        native_worker._load_tokenizer_wrapper = original_loader
        native_worker.build_finalized_token_ids = original_build
        native_worker.compare_native_prefill_decode_to_mlx_lm = original_compare
        native_worker.build_native_executor = original_executor

    assert exit_code == 1
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
    ]
    failed_status = next(
        item
        for item in decoded
        if isinstance(item, ModelStatus) and item.state == "failed"
    )
    assert failed_status.last_error is not None
    assert failed_status.last_error.category == "supported_class_bug"
    assert failed_status.last_error.stage == "deterministic_warmup"
    assert failed_status.last_error.code == "NATIVE_PUBLIC_SERVING_NOT_IMPLEMENTED"
    worker_error = decoded[-1]
    assert isinstance(worker_error, WorkerError)
    assert worker_error.error is not None
    assert worker_error.error.category == "supported_class_bug"


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
