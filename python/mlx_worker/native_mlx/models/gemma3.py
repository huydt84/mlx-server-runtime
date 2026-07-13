"""Gemma 3 text-only native MLX architecture implementation."""

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
class Gemma3ModelConfig:
    """Gemma 3 text fields required by the native MLX graph."""

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
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    sliding_window_pattern: int
    rope_scaling: dict[str, Any] | None
    quantization: dict[str, Any] | None
    kv_cache_dtype: str = "float16"
    layer_types: tuple[str, ...] = ()


class Gemma3RMSNorm(nn.Module):
    """Gemma RMSNorm, whose checkpoint weight is stored as a residual scale."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Gemma3Attention(nn.Module):
    """Gemma 3 attention with local/global RoPE and per-head norms."""

    def __init__(self, config: Gemma3ModelConfig, layer_index: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.query_pre_attn_scalar**-0.5
        self.is_sliding = _is_sliding(config, layer_index)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)
        base = config.rope_local_base_freq if self.is_sliding else config.rope_theta
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=base,
            scale=_rope_scale(config.rope_scaling) if not self.is_sliding else 1.0,
        )
        self.window_size = config.sliding_window if self.is_sliding else None

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: LayerAttentionContext,
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
        mask = (
            f"sliding_window:{self.window_size}"
            if self.window_size is not None
            else "causal"
        )
        output = attention_context.append_and_attend(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class Gemma3Mlp(nn.Module):
    """Gemma 3 GELU feed-forward block."""

    def __init__(self, config: Gemma3ModelConfig):
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
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Gemma3TransformerBlock(nn.Module):
    """Gemma 3 decoder block with four RMSNorm boundaries."""

    def __init__(self, config: Gemma3ModelConfig, layer_index: int):
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_index)
        self.mlp = Gemma3Mlp(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: LayerAttentionContext,
    ) -> mx.array:
        residual = self.self_attn(self.input_layernorm(x), positions, attention_context)
        hidden = _clip_residual(x, self.post_attention_layernorm(residual))
        residual = self.mlp(self.pre_feedforward_layernorm(hidden))
        return _clip_residual(hidden, self.post_feedforward_layernorm(residual))


class Gemma3Backbone(nn.Module):
    """Gemma 3 decoder backbone."""

    def __init__(self, config: Gemma3ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma3TransformerBlock(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        positions: mx.array,
        layer_attention: tuple[LayerAttentionContext, ...],
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        hidden = hidden * mx.array(
            self.config.hidden_size**0.5, dtype=mx.bfloat16
        ).astype(hidden.dtype)
        if len(layer_attention) != len(self.layers):
            raise ValueError("Gemma3 requires one attention context per layer")
        for layer, attention_context in zip(self.layers, layer_attention, strict=True):
            hidden = layer(hidden, positions, attention_context)
        return self.norm(hidden)


class Gemma3ForCausalLM(nn.Module):
    """Gemma 3 text-only causal LM."""

    def __init__(self, config: Gemma3ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.model = Gemma3Backbone(config)
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
        )
        return self.lm_head(hidden)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_gemma3_config(payload: dict[str, Any]) -> Gemma3ModelConfig:
    """Parse and validate a Gemma 3 text Hugging Face config."""

    architecture = _architecture(payload)
    if payload.get("model_type") not in {"gemma3_text", "gemma3"}:
        raise ValueError("expected model_type='gemma3_text'")
    text = payload.get("text_config")
    if isinstance(text, dict):
        merged = {**payload, **text}
    else:
        merged = payload
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
        "sliding_window",
    )
    for name in required:
        if not isinstance(merged.get(name), (int, float)):
            raise ValueError(f"missing or invalid {name}")
    heads = int(merged["num_attention_heads"])
    kv_heads = int(merged["num_key_value_heads"])
    head_dim = int(merged["head_dim"])
    pattern = int(merged.get("_sliding_window_pattern", 6))
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or head_dim <= 0:
        raise ValueError("invalid Gemma3 head dimensions")
    if pattern <= 0:
        raise ValueError("Gemma3 sliding-window pattern must be positive")
    kv_cache_dtype = str(merged.get("torch_dtype", "float16")).lower()
    if kv_cache_dtype not in {"float16", "bfloat16"}:
        raise ValueError("native paged KV requires float16 or bfloat16 torch_dtype")
    quantization = merged.get("quantization") or merged.get("quantization_config")
    layer_types = tuple(str(value) for value in merged.get("layer_types", ()))
    if layer_types and len(layer_types) != int(merged["num_hidden_layers"]):
        raise ValueError("Gemma3 layer_types length does not match layer count")
    return Gemma3ModelConfig(
        architecture_class=architecture,
        model_type="gemma3_text",
        hidden_size=int(merged["hidden_size"]),
        intermediate_size=int(merged["intermediate_size"]),
        num_attention_heads=heads,
        num_hidden_layers=int(merged["num_hidden_layers"]),
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=int(merged["vocab_size"]),
        max_position_embeddings=int(merged["max_position_embeddings"]),
        rms_norm_eps=float(merged["rms_norm_eps"]),
        rope_theta=float(merged.get("rope_theta", 1_000_000.0)),
        rope_local_base_freq=float(merged.get("rope_local_base_freq", 10_000.0)),
        query_pre_attn_scalar=float(merged.get("query_pre_attn_scalar", head_dim)),
        sliding_window=int(merged["sliding_window"]),
        sliding_window_pattern=pattern,
        rope_scaling=merged.get("rope_scaling"),
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        layer_types=layer_types,
    )


class Gemma3WeightAdapter:
    """Map Gemma 3 text checkpoint names into the native graph."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        entries = tuple(
            WeightMappingEntry(
                canonical_name=_canonicalize_gemma3_name(name),
                source_name=name,
                source_file=source_file,
            )
            for name, source_file in index.weight_map.items()
        )
        if not entries:
            raise WeightMappingBug("Gemma3 checkpoint produced empty mapping plan")
        return WeightMappingPlan(
            architecture_class="Gemma3ForCausalLM",
            source_files=index.source_files,
            entries=entries,
        )


def build_gemma3_model(
    config: Gemma3ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> Gemma3ForCausalLM:
    """Construct, quantize, and load the Gemma 3 graph."""

    model = Gemma3ForCausalLM(config)
    _quantize_model(model, config, weights)
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _quantize_model(
    model: nn.Module,
    config: Gemma3ModelConfig,
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


def _canonicalize_gemma3_name(source_name: str) -> str:
    if source_name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
        return source_name
    match = _LAYER_RE.match(source_name)
    if match is None:
        raise WeightMappingBug(f"Gemma3 has no mapping rule for tensor {source_name!r}")
    return f"model.layers.{int(match.group(1))}.{match.group(2)}"


def _architecture(payload: dict[str, Any]) -> str:
    architectures = payload.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if architecture != "Gemma3ForCausalLM":
        raise ValueError("expected architectures[0]='Gemma3ForCausalLM'")
    return architecture


def _is_sliding(config: Gemma3ModelConfig, layer_index: int) -> bool:
    if config.layer_types:
        return config.layer_types[layer_index] == "sliding_attention"
    return (layer_index + 1) % config.sliding_window_pattern != 0


def _rope_scale(rope_scaling: dict[str, Any] | None) -> float:
    if not rope_scaling:
        return 1.0
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type == "linear":
        factor = float(rope_scaling.get("factor", 1.0))
        if factor <= 0:
            raise ValueError("Gemma3 rope scaling factor must be positive")
        return 1.0 / factor
    if rope_type in (None, "default"):
        return 1.0
    raise ValueError(f"unsupported Gemma3 RoPE scaling type: {rope_type}")


def _clip_residual(x: mx.array, y: mx.array) -> mx.array:
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


EntryClass = Gemma3ForCausalLM
