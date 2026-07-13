"""Qwen3 native MLX architecture implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..interfaces import ForwardBatch, LayerAttentionContext
from ..mapping import (
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)


@dataclass(frozen=True)
class Qwen3ModelConfig:
    """Qwen3 fields required by the native MLX graph."""

    architecture_class: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    tie_word_embeddings: bool
    quantization: dict[str, Any] | None
    kv_cache_dtype: str = "float16"


def _silu_gate(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


class Qwen3Attention(nn.Module):
    """Qwen3 projections, per-head RMSNorm, and RoPE."""

    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
            scale=_rope_scale(config.rope_scaling),
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: LayerAttentionContext,
        attention_mask: str | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        queries = self.q_proj(x).reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        )
        keys = self.k_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )
        values = self.v_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
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
        output = attention_context.append_and_attend(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class Qwen3Mlp(nn.Module):
    """Qwen3 SwiGLU feed-forward block."""

    def __init__(self, config: Qwen3ModelConfig):
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


class Qwen3TransformerBlock(nn.Module):
    """Qwen3 decoder block."""

    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3Mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: LayerAttentionContext,
        attention_mask: str | None = None,
    ) -> mx.array:
        residual = self.self_attn(
            self.input_layernorm(x),
            positions,
            attention_context,
            attention_mask,
        )
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3Backbone(nn.Module):
    """Qwen3 decoder backbone."""

    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        positions: mx.array,
        layer_attention: tuple[LayerAttentionContext, ...],
        attention_mask: str | None = None,
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if len(layer_attention) != len(self.layers):
            raise ValueError("Qwen3 requires one attention context per layer")
        for layer, attention_context in zip(self.layers, layer_attention, strict=True):
            hidden = layer(
                hidden,
                positions,
                attention_context,
                attention_mask,
            )
        return self.norm(hidden)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 decoder-only causal LM."""

    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.model = Qwen3Backbone(config)
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
            positions,
            forward_batch.layer_attention,
            forward_batch.attention_mask,
        )
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_qwen3_config(payload: dict[str, Any]) -> Qwen3ModelConfig:
    """Parse and validate a Qwen3 Hugging Face config."""

    architecture = _architecture(payload)
    if payload.get("model_type") != "qwen3":
        raise ValueError("expected model_type='qwen3'")
    required = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
    )
    for name in required:
        if not isinstance(payload.get(name), (int, float)):
            raise ValueError(f"missing or invalid {name}")
    heads = int(payload["num_attention_heads"])
    kv_heads = int(payload["num_key_value_heads"])
    head_dim = int(payload["head_dim"])
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or head_dim <= 0:
        raise ValueError("invalid Qwen3 head dimensions")
    kv_cache_dtype = str(payload.get("torch_dtype", "float16")).lower()
    if kv_cache_dtype not in {"float16", "bfloat16"}:
        raise ValueError("native paged KV requires float16 or bfloat16 torch_dtype")
    quantization = payload.get("quantization") or payload.get("quantization_config")
    return Qwen3ModelConfig(
        architecture_class=architecture,
        model_type="qwen3",
        hidden_size=int(payload["hidden_size"]),
        intermediate_size=int(payload["intermediate_size"]),
        num_attention_heads=heads,
        num_hidden_layers=int(payload["num_hidden_layers"]),
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=int(payload["vocab_size"]),
        max_position_embeddings=int(payload["max_position_embeddings"]),
        rms_norm_eps=float(payload["rms_norm_eps"]),
        rope_theta=float(payload.get("rope_theta", 1_000_000.0)),
        rope_scaling=payload.get("rope_scaling"),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
    )


class Qwen3WeightAdapter:
    """Map Qwen3 checkpoint names into the native graph."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        entries = tuple(
            WeightMappingEntry(
                canonical_name=_canonicalize_qwen3_name(name),
                source_name=name,
                source_file=source_file,
            )
            for name, source_file in index.weight_map.items()
        )
        if not entries:
            raise WeightMappingBug("Qwen3 checkpoint produced empty mapping plan")
        return WeightMappingPlan(
            architecture_class="Qwen3ForCausalLM",
            source_files=index.source_files,
            entries=entries,
        )


def build_qwen3_model(
    config: Qwen3ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> Qwen3ForCausalLM:
    """Construct, quantize, and load the Qwen3 graph."""

    model = Qwen3ForCausalLM(config)
    _quantize_model(model, config, weights)
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _quantize_model(
    model: nn.Module,
    config: Qwen3ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> None:
    if config.quantization is None:
        return
    weight_names = {name for name, _ in weights}

    def class_predicate(path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" not in weight_names:
            return False
        return {
            "group_size": int(config.quantization["group_size"]),
            "bits": int(config.quantization["bits"]),
            "mode": str(config.quantization.get("mode", "affine")),
        }

    nn.quantize(model, class_predicate=class_predicate)


def _canonicalize_qwen3_name(source_name: str) -> str:
    if source_name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
        return source_name
    match = _LAYER_RE.match(source_name)
    if match is None:
        raise WeightMappingBug(f"Qwen3 has no mapping rule for tensor {source_name!r}")
    return f"model.layers.{int(match.group(1))}.{match.group(2)}"


def _architecture(payload: dict[str, Any]) -> str:
    architectures = payload.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("missing or invalid architectures[0]")
    if architecture != "Qwen3ForCausalLM":
        raise ValueError("expected architectures[0]='Qwen3ForCausalLM'")
    return architecture


def _rope_scale(rope_scaling: dict[str, Any] | None) -> float:
    if not rope_scaling:
        return 1.0
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type == "linear":
        factor = float(rope_scaling.get("factor", 1.0))
        if factor <= 0:
            raise ValueError("Qwen3 rope scaling factor must be positive")
        return 1.0 / factor
    if rope_type in (None, "default"):
        return 1.0
    raise ValueError(f"unsupported Qwen3 RoPE scaling type: {rope_type}")


EntryClass = Qwen3ForCausalLM
