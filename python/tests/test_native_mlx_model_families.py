"""Architecture registration and deterministic graph tests for native-v2 families."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import mlx_worker.native_mlx.attention as native_attention
from mlx_worker.native_mlx.attention import DenseReferenceAttentionBackend
from mlx_worker.native_mlx.cache import (
    DenseKVCacheBackend,
    HybridPagedKVCacheBackend,
    KVCacheGeometry,
)
from mlx_worker.native_mlx.execution_backends import build_native_execution_backend
from mlx_worker.native_mlx.executor import MlxGenerationExecutor
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    ForwardMode,
    SamplingParams,
    select_logits_positions,
)
from mlx_worker.native_mlx.models.gemma3 import (
    Gemma3ForCausalLM,
    parse_gemma3_config,
)
from mlx_worker.native_mlx.models.lfm2 import (
    Lfm2MoE,
    Lfm2MoeForCausalLM,
    parse_lfm2_config,
)
from mlx_worker.native_mlx.models.qwen2 import Qwen2ForCausalLM
from mlx_worker.native_mlx.models.qwen3 import (
    Qwen3ForCausalLM,
    parse_qwen3_config,
)
from mlx_worker.native_mlx.registry import get_architecture_spec


def test_registered_model_families_match_real_checkpoint_architecture_classes() -> None:
    assert get_architecture_spec("Lfm2MoeForCausalLM").known_good_checkpoint == (
        "mlx-community/LFM2.5-8B-A1B-MLX-4bit"
    )
    assert get_architecture_spec("Qwen3ForCausalLM").known_good_checkpoint == (
        "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    )
    assert get_architecture_spec("Gemma3ForCausalLM").known_good_checkpoint == (
        "mlx-community/gemma-3-270m-it-qat-8bit"
    )
    assert not get_architecture_spec("Lfm2MoeForCausalLM").supports_prefix_cache


def test_lfm2_selects_hybrid_cache_bundle_at_startup() -> None:
    bundle = build_native_execution_backend(
        "native-metal-paged-sdpa",
        KVCacheGeometry(
            num_layers=2,
            num_kv_heads=1,
            head_dim=4,
            dtype=mx.float16,
        ),
        page_size=8,
        cache_budget_bytes=4096,
        cache_family="hybrid",
    )
    assert isinstance(bundle.cache_backend, HybridPagedKVCacheBackend)
    assert bundle.attention_backend.capabilities.reservation_types[0].__name__ == (
        "HybridBatchReservation"
    )


def _tiny_qwen3_and_gemma3() -> tuple[Qwen3ForCausalLM, Gemma3ForCausalLM]:
    qwen = Qwen3ForCausalLM(
        parse_qwen3_config(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
            }
        )
    )
    gemma = Gemma3ForCausalLM(
        parse_gemma3_config(
            {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3_text",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "sliding_window": 4,
                "_sliding_window_pattern": 2,
            }
        )
    )
    return qwen, gemma


def _tiny_prefill(
    model: Qwen3ForCausalLM | Gemma3ForCausalLM,
    logits_positions: tuple[int, ...] | None,
) -> mx.array:
    backend = DenseKVCacheBackend(model.num_layers)
    cache = backend.get(backend.create("tiny"), "tiny")
    reservation = backend.reserve_batch((cache,), (3,))
    output = model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        mx.array([[0, 1, 2]], dtype=mx.int32),
        ForwardBatch(
            forward_mode=ForwardMode.PREFILL,
            token_lengths=(3,),
            cache_lengths=(0,),
            attention_mask="causal",
            layer_attention=DenseReferenceAttentionBackend().contexts(
                reservation, ForwardMode.PREFILL
            ),
            logits_positions=logits_positions,
        ),
    )
    mx.eval(output)
    reservation.commit()
    return output


def test_models_compute_logits_only_for_requested_positions() -> None:
    for model in _tiny_qwen3_and_gemma3():
        full = _tiny_prefill(model, None)
        last = _tiny_prefill(model, (2,))
        assert last.shape == (1, 1, 64)
        assert mx.allclose(last[:, 0, :], full[:, 2, :], atol=1e-4).item()


def test_select_logits_positions_picks_one_token_per_row() -> None:
    hidden = mx.arange(2 * 3 * 2, dtype=mx.float32).reshape(2, 3, 2)

    def batch(positions: tuple[int, ...] | None) -> ForwardBatch:
        return ForwardBatch(
            forward_mode=ForwardMode.PREFILL,
            token_lengths=(3, 2),
            cache_lengths=(0, 0),
            attention_mask="causal",
            layer_attention=(),
            logits_positions=positions,
        )

    assert select_logits_positions(hidden, batch(None)) is hidden
    assert select_logits_positions(hidden, batch((1, 1))).tolist() == [
        [[2.0, 3.0]],
        [[8.0, 9.0]],
    ]
    assert select_logits_positions(hidden, batch((2, 1))).tolist() == [
        [[4.0, 5.0]],
        [[8.0, 9.0]],
    ]


def test_qwen3_and_gemma3_tiny_graphs_produce_logits() -> None:
    for model in _tiny_qwen3_and_gemma3():
        backend = DenseKVCacheBackend(model.num_layers)
        handle = backend.create("tiny")
        cache = backend.get(handle, "tiny")
        reservation = backend.reserve_batch((cache,), (3,))
        output = model(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            mx.array([[0, 1, 2]], dtype=mx.int32),
            ForwardBatch(
                forward_mode=ForwardMode.PREFILL,
                token_lengths=(3,),
                cache_lengths=(0,),
                attention_mask="causal",
                layer_attention=DenseReferenceAttentionBackend().contexts(
                    reservation, ForwardMode.PREFILL
                ),
            ),
        )
        mx.eval(output)
        assert output.shape == (1, 3, 64)
        reservation.commit()


@pytest.mark.parametrize(
    ("architecture_class", "model_type"),
    [
        ("Qwen2ForCausalLM", "qwen2"),
        ("Qwen3ForCausalLM", "qwen3"),
        ("Gemma3ForCausalLM", "gemma3_text"),
    ],
)
@pytest.mark.parametrize(
    ("config_dtype", "weight_dtype"),
    [("bfloat16", mx.float16), ("float16", mx.bfloat16)],
)
def test_paged_decode_runs_when_config_dtype_differs_from_weights(
    architecture_class: str,
    model_type: str,
    config_dtype: str,
    weight_dtype: mx.Dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_architecture_spec(architecture_class)
    config = spec.parse_config(
        {
            "architectures": [architecture_class],
            "model_type": model_type,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "sliding_window": 4,
            "_sliding_window_pattern": 2,
            "torch_dtype": config_dtype,
        }
    )
    model_class = {
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "Qwen3ForCausalLM": Qwen3ForCausalLM,
        "Gemma3ForCausalLM": Gemma3ForCausalLM,
    }[architecture_class]
    weights = [
        (name, value.astype(weight_dtype))
        for name, value in tree_flatten(model_class(config).parameters())
    ]
    model = spec.create_model(config, weights)
    config = getattr(model, "resolved_config", config)
    bundle = build_native_execution_backend(
        "native-metal-paged-sdpa",
        spec.cache_geometry(config),
        page_size=8,
        cache_budget_bytes=1 << 20,
        cache_family=spec.resolve.cache_family,
    )
    executor = MlxGenerationExecutor(
        architecture_class=architecture_class,
        model=model,
        cache_backend=bundle.cache_backend,
        attention_backend=bundle.attention_backend,
    )
    query_dtypes = []
    paged_decode = native_attention.paged_decode_attention

    def recording_paged_decode(queries, *args, **kwargs):
        query_dtypes.append(queries.dtype)
        return paged_decode(queries, *args, **kwargs)

    monkeypatch.setattr(
        native_attention, "paged_decode_attention", recording_paged_decode
    )
    handle = bundle.cache_backend.create("mixed")

    def step(phase: str, token_ids: tuple[int, ...], start: int) -> int:
        result = executor.execute_batch(
            ExecutionBatch(
                requests=(
                    ExecutionRequest(
                        request_id="mixed",
                        phase=phase,  # type: ignore[arg-type]
                        token_ids=token_ids,
                        positions=tuple(range(start, start + len(token_ids))),
                        cache_handle=handle,
                        sampling=SamplingParams(),
                    ),
                )
            )
        )
        token_id = result.results[0].next_token_id
        assert token_id is not None
        return token_id

    token = step("prefill", (1, 2, 3), 0)
    token = step("decode", (token,), 3)
    step("decode", (token,), 4)

    assert query_dtypes == [weight_dtype] * 4
    assert bundle.cache_backend.length(handle) == 5


def test_lfm2_tiny_graph_commits_hybrid_conv_state_and_length() -> None:
    model = Lfm2MoeForCausalLM(
        parse_lfm2_config(
            {
                "architectures": ["Lfm2MoeForCausalLM"],
                "model_type": "lfm2_moe",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 3,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "num_dense_layers": 1,
                "norm_eps": 1e-5,
                "conv_L_cache": 3,
                "layer_types": ["conv", "full_attention", "conv"],
            }
        )
    )
    backend = DenseKVCacheBackend(model.num_layers)
    handle = backend.create("tiny")
    cache = backend.get(handle, "tiny")
    reservation = backend.reserve_batch((cache,), (3,))
    output = model(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        mx.array([[0, 1, 2]], dtype=mx.int32),
        ForwardBatch(
            forward_mode=ForwardMode.PREFILL,
            token_lengths=(3,),
            cache_lengths=(0,),
            attention_mask="causal",
            layer_attention=DenseReferenceAttentionBackend().contexts(
                reservation, ForwardMode.PREFILL
            ),
        ),
    )
    mx.eval(output)
    reservation.commit()
    assert backend.length(handle) == 3
    assert cache.layers[0].conv_state is not None
    assert cache.layers[2].conv_state is not None


def _tiny_lfm2_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": 64,
        "hidden_size": 2,
        "intermediate_size": 4,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "num_dense_layers": 0,
        "norm_eps": 1e-5,
        "conv_L_cache": 3,
        "layer_types": ["conv"],
    }
    payload.update(overrides)
    return payload


def test_lfm2_config_reads_routed_scaling_factor_with_default() -> None:
    assert parse_lfm2_config(_tiny_lfm2_payload()).routed_scaling_factor == 1.0
    config = parse_lfm2_config(_tiny_lfm2_payload(routed_scaling_factor=2.5))
    assert config.routed_scaling_factor == 2.5


def test_lfm2_moe_routes_with_sigmoid_and_selection_only_expert_bias() -> None:
    moe = Lfm2MoE(
        parse_lfm2_config(
            _tiny_lfm2_payload(use_expert_bias=True, routed_scaling_factor=2.0)
        )
    )
    moe.gate.weight = mx.array([[2.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
    moe.expert_bias = mx.array([0.0, -1.0, 1.0])

    indices, scores = moe.route(mx.array([[[1.0, 0.0]]]))
    mx.eval(indices, scores)

    picked = dict(
        zip(indices.reshape(-1).tolist(), scores.reshape(-1).tolist(), strict=True)
    )
    assert sorted(picked) == [0, 2]
    sigmoid_0 = 1.0 / (1.0 + math.exp(-2.0))
    sigmoid_2 = 1.0 / (1.0 + math.exp(1.0))
    total = sigmoid_0 + sigmoid_2 + 1e-6
    assert math.isclose(picked[0], 2.0 * sigmoid_0 / total, rel_tol=1e-5)
    assert math.isclose(picked[2], 2.0 * sigmoid_2 / total, rel_tol=1e-5)


def test_lfm2_moe_without_normalization_keeps_raw_sigmoid_scores() -> None:
    moe = Lfm2MoE(
        parse_lfm2_config(
            _tiny_lfm2_payload(use_expert_bias=False, norm_topk_prob=False)
        )
    )
    moe.gate.weight = mx.array([[2.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])

    indices, scores = moe.route(mx.array([[[1.0, 0.0]]]))
    mx.eval(indices, scores)

    picked = dict(
        zip(indices.reshape(-1).tolist(), scores.reshape(-1).tolist(), strict=True)
    )
    assert sorted(picked) == [0, 1]
    assert math.isclose(picked[0], 1.0 / (1.0 + math.exp(-2.0)), rel_tol=1e-5)
    assert math.isclose(picked[1], 0.5, rel_tol=1e-5)
