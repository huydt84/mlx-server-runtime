"""LFM2-MoE native MLX architecture implementation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..interfaces import ForwardBatch, HybridLayerAttentionContext
from ..mapping import (
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)


@dataclass(frozen=True)
class Lfm2ModelConfig:
    """LFM2-MoE fields required by the native MLX graph."""

    architecture_class: str
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    use_expert_bias: bool
    num_dense_layers: int
    norm_eps: float
    conv_bias: bool
    conv_l_cache: int
    rope_theta: float
    layer_types: tuple[str, ...]
    quantization: dict[str, Any] | None
    tie_word_embeddings: bool
    kv_cache_dtype: str = "float16"

    @property
    def full_attn_idxs(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        )


class SwitchLinear(nn.Module):
    """Expert-indexed linear layer with MLX quantized dispatch support."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        sorted_indices: bool = False,
    ) -> mx.array:
        output = mx.gather_mm(
            x,
            self.weight.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if hasattr(self, "bias"):
            output = output + mx.expand_dims(self.bias[indices], -2)
        return output

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> "QuantizedSwitchLinear":
        quantized = QuantizedSwitchLinear(
            self.weight.shape[-1],
            self.weight.shape[-2],
            self.weight.shape[0],
            bias=hasattr(self, "bias"),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        quantized.weight, quantized.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        quantized.biases = biases[0] if biases else None
        if hasattr(self, "bias"):
            quantized.bias = self.bias
        return quantized


class QuantizedSwitchLinear(nn.Module):
    """Quantized expert-indexed linear layer."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        *,
        bias: bool,
        group_size: int,
        bits: int,
        mode: str,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-math.sqrt(1 / input_dims),
                high=math.sqrt(1 / input_dims),
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    @property
    def input_dims(self) -> int:
        return self.scales.shape[-1] * self.group_size

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        sorted_indices: bool = False,
    ) -> mx.array:
        output = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if hasattr(self, "bias"):
            output = output + mx.expand_dims(self.bias[indices], -2)
        return output


class SwitchGLU(nn.Module):
    """SwiGLU expert bank with top-k routing."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        selected = indices
        inverse = None
        if do_sort:
            x, selected, inverse = _gather_sort(x, indices)
        up = self.up_proj(x, selected, sorted_indices=do_sort)
        gate = self.gate_proj(x, selected, sorted_indices=do_sort)
        output = self.down_proj(
            nn.silu(gate) * up,
            selected,
            sorted_indices=do_sort,
        )
        if do_sort:
            output = _scatter_unsort(output, inverse, indices.shape)
        return output.squeeze(-2)


class Lfm2Attention(nn.Module):
    """LFM2 full-attention layer."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: HybridLayerAttentionContext,
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
        queries = self.q_layernorm(queries).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offsets = positions[:, 0]
        queries = mx.fast.rope(
            queries,
            self.head_dim,
            traditional=False,
            base=self.rope.base,
            scale=self.rope.scale,
            offset=offsets,
        )
        keys = mx.fast.rope(
            keys,
            self.head_dim,
            traditional=False,
            base=self.rope.base,
            scale=self.rope.scale,
            offset=offsets,
        )
        output = attention_context.append_and_attend(
            queries,
            keys,
            values,
            scale=self.scale,
            mask="causal",
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)


class Lfm2ShortConv(nn.Module):
    """LFM2 depthwise short convolution with request-local state."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        self.cache_size = config.conv_l_cache
        self.hidden_size = config.hidden_size
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_l_cache,
            groups=config.hidden_size,
            bias=config.conv_bias,
        )
        self.in_proj = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.conv_bias,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.conv_bias,
        )

    def __call__(
        self,
        x: mx.array,
        attention_context: HybridLayerAttentionContext,
    ) -> mx.array:
        projected = self.in_proj(x)
        b_gate, c_gate, value = mx.split(projected, 3, axis=-1)
        gated = b_gate * value
        combined = attention_context.prepare_conv_state(gated, self.cache_size)
        conv_out = self.conv(combined)
        attention_context.stage_conv_state(combined, self.cache_size)
        return self.out_proj(c_gate * conv_out)


class Lfm2Mlp(nn.Module):
    """Dense SwiGLU block used by the initial LFM2 layers."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Lfm2MoE(nn.Module):
    """Top-k routed SwiGLU expert block."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.use_expert_bias = config.use_expert_bias
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        )
        if config.use_expert_bias:
            self.expert_bias = mx.zeros((config.num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        if self.use_expert_bias:
            gates = gates + self.expert_bias
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)
        return (self.switch_mlp(x, indices) * scores[..., None]).sum(axis=-2)


class Lfm2DecoderLayer(nn.Module):
    """LFM2 hybrid convolution/attention decoder layer."""

    def __init__(self, config: Lfm2ModelConfig, layer_index: int):
        super().__init__()
        self.is_attention_layer = layer_index in config.full_attn_idxs
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config)
        self.feed_forward = (
            Lfm2Mlp(config)
            if layer_index < config.num_dense_layers
            else Lfm2MoE(config)
        )
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: HybridLayerAttentionContext,
    ) -> mx.array:
        if self.is_attention_layer:
            residual = self.self_attn(
                self.operator_norm(x), positions, attention_context
            )
        else:
            residual = self.conv(self.operator_norm(x), attention_context)
        hidden = x + residual
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm2Backbone(nn.Module):
    """LFM2 hybrid decoder backbone."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        positions: mx.array,
        layer_attention: tuple[HybridLayerAttentionContext, ...],
    ) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if len(layer_attention) != len(self.layers):
            raise ValueError("LFM2 requires one context per layer")
        for layer, attention_context in zip(self.layers, layer_attention, strict=True):
            hidden = layer(hidden, positions, attention_context)
        return self.embedding_norm(hidden)


class Lfm2MoeForCausalLM(nn.Module):
    """LFM2-MoE causal LM with tied input/output embeddings."""

    def __init__(self, config: Lfm2ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.model = Lfm2Backbone(config)

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
        return self.model.embed_tokens.as_linear(hidden)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parse_lfm2_config(payload: dict[str, Any]) -> Lfm2ModelConfig:
    """Parse and validate an LFM2-MoE Hugging Face config."""

    architecture = _architecture(payload)
    if payload.get("model_type") != "lfm2_moe":
        raise ValueError("expected model_type='lfm2_moe'")
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "num_dense_layers",
        "norm_eps",
        "conv_L_cache",
        "layer_types",
    )
    for name in required:
        if name not in payload:
            raise ValueError(f"missing {name}")
    layer_types = tuple(str(value) for value in payload["layer_types"])
    layers = int(payload["num_hidden_layers"])
    if len(layer_types) != layers:
        raise ValueError("LFM2 layer_types length does not match layer count")
    heads = int(payload["num_attention_heads"])
    kv_heads = int(payload["num_key_value_heads"])
    hidden = int(payload["hidden_size"])
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or hidden % heads:
        raise ValueError("invalid LFM2 head dimensions")
    if not all(value in {"conv", "full_attention"} for value in layer_types):
        raise ValueError("unsupported LFM2 layer type")
    dtype = str(payload.get("dtype", payload.get("torch_dtype", "float16"))).lower()
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError("native paged KV requires float16 or bfloat16 dtype")
    quantization = payload.get("quantization") or payload.get("quantization_config")
    return Lfm2ModelConfig(
        architecture_class=architecture,
        model_type="lfm2_moe",
        vocab_size=int(payload["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(payload["intermediate_size"]),
        moe_intermediate_size=int(payload["moe_intermediate_size"]),
        num_hidden_layers=layers,
        num_experts=int(payload["num_experts"]),
        num_experts_per_tok=int(payload["num_experts_per_tok"]),
        norm_topk_prob=bool(payload.get("norm_topk_prob", True)),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=int(payload["max_position_embeddings"]),
        use_expert_bias=bool(payload.get("use_expert_bias", False)),
        num_dense_layers=int(payload["num_dense_layers"]),
        norm_eps=float(payload["norm_eps"]),
        conv_bias=bool(payload.get("conv_bias", False)),
        conv_l_cache=int(payload["conv_L_cache"]),
        rope_theta=float(
            (payload.get("rope_parameters") or {}).get(
                "rope_theta", payload.get("rope_theta", 1_000_000.0)
            )
        ),
        layer_types=layer_types,
        quantization=quantization,
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        kv_cache_dtype=dtype,
    )


class Lfm2WeightAdapter:
    """Map LFM2-MoE checkpoint names into the native graph."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        entries = tuple(
            WeightMappingEntry(
                canonical_name=_canonicalize_lfm2_name(name),
                source_name=name,
                source_file=source_file,
            )
            for name, source_file in index.weight_map.items()
        )
        if not entries:
            raise WeightMappingBug("LFM2 checkpoint produced empty mapping plan")
        return WeightMappingPlan(
            architecture_class="Lfm2MoeForCausalLM",
            source_files=index.source_files,
            entries=entries,
        )


def build_lfm2_model(
    config: Lfm2ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> Lfm2MoeForCausalLM:
    """Construct, quantize, and load the LFM2-MoE graph."""

    model = Lfm2MoeForCausalLM(config)
    _quantize_model(model, config, weights)
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _quantize_model(
    model: nn.Module,
    config: Lfm2ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> None:
    if config.quantization is None:
        return
    weight_names = {name for name, _ in weights}
    base = config.quantization

    def class_predicate(path: str, module: nn.Module) -> bool | dict[str, Any]:
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" not in weight_names:
            return False
        override = base.get(path) if isinstance(base.get(path), dict) else base
        return {
            "group_size": int(override["group_size"]),
            "bits": int(override["bits"]),
            "mode": str(override.get("mode", "affine")),
        }

    nn.quantize(model, class_predicate=class_predicate)


def _canonicalize_lfm2_name(source_name: str) -> str:
    if source_name.startswith(("model.embed_tokens.", "model.embedding_norm.")):
        return source_name
    match = _LAYER_RE.match(source_name)
    if match is None:
        raise WeightMappingBug(f"LFM2 has no mapping rule for tensor {source_name!r}")
    return f"model.layers.{int(match.group(1))}.{match.group(2)}"


def _architecture(payload: dict[str, Any]) -> str:
    architectures = payload.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if architecture != "Lfm2MoeForCausalLM":
        raise ValueError("expected architectures[0]='Lfm2MoeForCausalLM'")
    return architecture


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    *_, width = indices.shape
    flat_indices = indices.flatten()
    order = mx.argsort(flat_indices)
    inverse = mx.argsort(order)
    return x.flatten(0, -3)[order // width], flat_indices[order], inverse


def _scatter_unsort(x: mx.array, inverse: mx.array, shape: tuple[int, ...]) -> mx.array:
    return mx.unflatten(x[inverse], 0, shape)


EntryClass = Lfm2MoeForCausalLM
