"""Architecture registration and deterministic graph tests for native-v2 families."""

from __future__ import annotations

import mlx.core as mx

from mlx_worker.native_mlx.attention import DenseReferenceAttentionBackend
from mlx_worker.native_mlx.cache import DenseKVCacheBackend
from mlx_worker.native_mlx.interfaces import ForwardBatch, ForwardMode
from mlx_worker.native_mlx.models.gemma3 import (
    Gemma3ForCausalLM,
    parse_gemma3_config,
)
from mlx_worker.native_mlx.models.lfm2 import (
    Lfm2MoeForCausalLM,
    parse_lfm2_config,
)
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


def test_qwen3_and_gemma3_tiny_graphs_produce_logits() -> None:
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
    for model in (qwen, gemma):
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
