"""Qwen2 native MLX architecture implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..cache import LayerKVCache
from ..interfaces import ForwardBatch
from ..mapping import (
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)


@dataclass(frozen=True)
class Qwen2ModelConfig:
    """Qwen2 fields required by the native MLX graph."""

    architecture_class: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_traditional: bool
    rope_scaling: dict[str, Any] | None
    tie_word_embeddings: bool
    quantization: dict[str, Any] | None


def _silu_gate(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class Qwen2Attention(nn.Module):
    """Qwen2 self-attention using MLX built-in SDPA."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        cache: LayerKVCache | None = None,
        attention_mask: mx.array | str | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            offsets = positions[:, 0]
            queries = mx.fast.rope(
                queries,
                self.head_dim,
                traditional=self.rope.traditional,
                base=self.rope.base,
                scale=self.rope.scale,
                offset=offsets,
            )
            keys = mx.fast.rope(
                keys,
                self.head_dim,
                traditional=self.rope.traditional,
                base=self.rope.base,
                scale=self.rope.scale,
                offset=offsets,
            )
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        sdpa_mask = attention_mask
        if sdpa_mask is not None and not isinstance(sdpa_mask, str):
            sdpa_mask = sdpa_mask.astype(queries.dtype)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=sdpa_mask
            if sdpa_mask is not None
            else "causal"
            if seq_len > 1 or cache is not None
            else None,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class Qwen2Mlp(nn.Module):
    """Qwen2 feed-forward block."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu_gate(self.gate_proj(x), self.up_proj(x)))


class Qwen2TransformerBlock(nn.Module):
    """Qwen2 transformer block."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2Mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        cache: LayerKVCache | None = None,
        attention_mask: mx.array | str | None = None,
    ) -> mx.array:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            positions=positions,
            cache=cache,
            attention_mask=attention_mask,
        )
        hidden = x + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp_out


class Qwen2Backbone(nn.Module):
    """Qwen2 decoder backbone."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        positions: mx.array,
        cache: list[LayerKVCache] | None = None,
        attention_mask: mx.array | str | None = None,
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        active_cache = cache or [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, active_cache):
            hidden = layer(
                hidden,
                positions=positions,
                cache=layer_cache,
                attention_mask=attention_mask,
            )
        return self.norm(hidden)


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 decoder-only causal LM."""

    def __init__(self, config: Qwen2ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.model = Qwen2Backbone(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        hidden = self.model(
            input_ids,
            positions=positions,
            cache=(
                list(forward_batch.layer_caches) if forward_batch.layer_caches else None
            ),
            attention_mask=forward_batch.attention_mask,
        )
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_qwen2_config(payload: dict[str, Any]) -> Qwen2ModelConfig:
    """Parse and validate a Qwen2 Hugging Face config."""

    architectures = payload.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("missing or invalid architectures[0]")
    if payload.get("model_type") != "qwen2":
        raise ValueError("expected model_type='qwen2'")
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
    )
    for name in required:
        if not isinstance(payload.get(name), (int, float)):
            raise ValueError(f"missing or invalid {name}")
    heads = int(payload["num_attention_heads"])
    kv_heads = int(payload.get("num_key_value_heads", heads))
    hidden = int(payload["hidden_size"])
    if hidden % heads or heads % kv_heads:
        raise ValueError("invalid Qwen2 head dimensions")
    return Qwen2ModelConfig(
        architecture_class=architecture,
        model_type="qwen2",
        hidden_size=hidden,
        intermediate_size=int(payload["intermediate_size"]),
        num_attention_heads=heads,
        num_hidden_layers=int(payload["num_hidden_layers"]),
        num_key_value_heads=kv_heads,
        vocab_size=int(payload["vocab_size"]),
        max_position_embeddings=int(payload["max_position_embeddings"]),
        rms_norm_eps=float(payload["rms_norm_eps"]),
        rope_theta=float(payload.get("rope_theta", 1_000_000.0)),
        rope_traditional=bool(payload.get("rope_traditional", False)),
        rope_scaling=payload.get("rope_scaling"),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        quantization=payload.get("quantization"),
    )


class Qwen2WeightAdapter:
    """Map Qwen2 checkpoint names into the native graph."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        entries = tuple(
            WeightMappingEntry(
                canonical_name=_canonicalize_qwen2_name(name),
                source_name=name,
                source_file=source_file,
            )
            for name, source_file in index.weight_map.items()
        )
        if not entries:
            raise WeightMappingBug("Qwen2 checkpoint produced empty mapping plan")
        return WeightMappingPlan(
            architecture_class="Qwen2ForCausalLM",
            source_files=index.source_files,
            entries=entries,
        )


def build_qwen2_model(
    config: Qwen2ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> Qwen2ForCausalLM:
    """Construct, quantize, and load the Qwen2 graph."""

    model = Qwen2ForCausalLM(config)
    if config.quantization is not None:
        weight_names = {name for name, _ in weights}

        def class_predicate(path: str, module: nn.Module) -> bool:
            return hasattr(module, "to_quantized") and f"{path}.scales" in weight_names

        nn.quantize(
            model,
            group_size=int(config.quantization["group_size"]),
            bits=int(config.quantization["bits"]),
            mode=str(config.quantization.get("mode", "affine")),
            class_predicate=class_predicate,
        )
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _canonicalize_qwen2_name(source_name: str) -> str:
    if source_name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
        return source_name
    match = _LAYER_RE.match(source_name)
    if match is None:
        raise WeightMappingBug(f"Qwen2 has no mapping rule for tensor {source_name!r}")
    return f"model.layers.{int(match.group(1))}.{match.group(2)}"


EntryClass = Qwen2ForCausalLM
