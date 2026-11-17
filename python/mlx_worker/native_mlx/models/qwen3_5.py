"""Qwen3.5 dense text native MLX architecture implementation.

Qwen3.5 interleaves Gated DeltaNet linear-attention layers with gated
full-attention layers: every ``full_attention_interval``-th layer attends over
paged KV, every other layer keeps request-local convolution and recurrent
state in the hybrid cache. Only the language model is built; vision tower and
multi-token-prediction weights are ignored.

Two weight formats are accepted:

- MLX affine-quantized ``Qwen3_5ForConditionalGeneration`` exports, including
  1-bit packs such as ``prism-ml/Bonsai-27B-mlx-1bit``;
- Prism ML Hadamard-rotated packs (``model_type = prism_hadamard_qwen35``,
  for example ``prism-ml/Ternary-Bonsai-2-27B-mlx-2bit``), whose projections
  need an activation rotation (see :mod:`mlx_worker.prism_hadamard`).

The Gated DeltaNet recurrence and its helpers come from ``mlx-lm`` so that the
native graph and the ``mlx-lm`` reference run the same Metal kernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.activations import precise_swiglu, swiglu
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from ...prism_hadamard import (
    HadamardPackConfig,
    install_hadamard_modules,
    is_prism_hadamard_config,
    parse_hadamard_pack,
)
from ..interfaces import (
    ForwardBatch,
    HybridLayerAttentionContext,
    select_logits_positions,
)
from ..mapping import (
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)

ARCHITECTURE_CLASS = "Qwen3_5ForConditionalGeneration"

_KV_DTYPES = {mx.float16: "float16", mx.bfloat16: "bfloat16", mx.float32: "float32"}


@dataclass(frozen=True)
class Qwen3_5ModelConfig:
    """Qwen3.5 text fields required by the native MLX graph."""

    architecture_class: str
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    full_attention_interval: int
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    quantization: dict[str, Any] | None
    hadamard: HadamardPackConfig | None = None
    kv_cache_dtype: str = "float16"
    """Activation dtype of the K/V projections; resolved after weights load."""

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def is_linear_layer(self, layer_index: int) -> bool:
        """Return whether a layer is Gated DeltaNet rather than full attention."""

        return (layer_index + 1) % self.full_attention_interval != 0

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self.num_hidden_layers)
            if not self.is_linear_layer(index)
        )


class Qwen3_5Attention(nn.Module):
    """Full attention with q/k RMSNorm, partial RoPE, and a sigmoid output gate."""

    def __init__(self, config: Qwen3_5ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: HybridLayerAttentionContext,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        queries, gate = mx.split(
            self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch_size, seq_len, -1)
        keys = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, -1)
        values = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offsets = positions[:, 0]
        queries = mx.fast.rope(
            queries,
            self.rotary_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offsets,
        )
        keys = mx.fast.rope(
            keys,
            self.rotary_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
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
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3_5RMSNormGated(nn.Module):
    """RMSNorm followed by a float32 SiLU gate."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(hidden, self.weight, self.eps)
        return precise_swiglu(hidden, gate, normed)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention with hybrid-cache request state."""

    def __init__(self, config: Qwen3_5ModelConfig):
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        hidden = config.hidden_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        attention_context: HybridLayerAttentionContext,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        conv_input = attention_context.prepare_conv_state(qkv, self.conv_kernel_size)
        attention_context.stage_conv_state(conv_input, self.conv_kernel_size)
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = (
            part.reshape(batch_size, seq_len, heads, dims)
            for part, heads, dims in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                (self.num_k_heads, self.num_k_heads, self.num_v_heads),
                (self.head_k_dim, self.head_k_dim, self.head_v_dim),
                strict=True,
            )
        )
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        state = attention_context.recurrent_state(
            (self.num_v_heads, self.head_v_dim, self.head_k_dim), mx.float32
        )
        output, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            attention_context.token_mask(seq_len),
            use_kernel=True,
        )
        attention_context.stage_recurrent_state(state)
        output = self.norm(output, z)
        return self.out_proj(output.reshape(batch_size, seq_len, -1))


class Qwen3_5Mlp(nn.Module):
    """Dense SwiGLU feed-forward block."""

    def __init__(self, config: Qwen3_5ModelConfig):
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3_5DecoderLayer(nn.Module):
    """Qwen3.5 decoder layer with linear or full attention."""

    def __init__(self, config: Qwen3_5ModelConfig, layer_index: int):
        super().__init__()
        self.is_linear = config.is_linear_layer(layer_index)
        if self.is_linear:
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3_5Mlp(config)

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        attention_context: HybridLayerAttentionContext,
    ) -> mx.array:
        normed = self.input_layernorm(x)
        if self.is_linear:
            residual = self.linear_attn(normed, attention_context)
        else:
            residual = self.self_attn(normed, positions, attention_context)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3_5Backbone(nn.Module):
    """Qwen3.5 hybrid decoder backbone."""

    def __init__(self, config: Qwen3_5ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3_5DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        positions: mx.array,
        layer_attention: tuple[HybridLayerAttentionContext, ...],
    ) -> mx.array:
        if len(layer_attention) != len(self.layers):
            raise ValueError("Qwen3.5 requires one context per layer")
        hidden = self.embed_tokens(inputs)
        for layer, attention_context in zip(self.layers, layer_attention, strict=True):
            hidden = layer(hidden, positions, attention_context)
        return self.norm(hidden)


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 causal language model."""

    def __init__(self, config: Qwen3_5ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.model = Qwen3_5Backbone(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        hidden = self.model(input_ids, positions, forward_batch.layer_attention)
        hidden = select_logits_positions(hidden, forward_batch)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


def parse_qwen3_5_config(payload: dict[str, Any]) -> Qwen3_5ModelConfig:
    """Parse and validate a Qwen3.5 dense text config.

    Args:
        payload: ``config.json`` of a ``Qwen3_5ForConditionalGeneration``
            export or of a Hadamard-rotated Qwen3.5 pack.

    Raises:
        ValueError: If the config is not a supported dense Qwen3.5 text model.
    """

    hadamard: HadamardPackConfig | None = None
    if is_prism_hadamard_config(payload):
        hadamard = parse_hadamard_pack(payload)
        model_type = str(payload["model_type"])
        quantization = None
    else:
        _architecture(payload)
        model_type = str(payload.get("model_type"))
        if model_type != "qwen3_5":
            raise ValueError("expected model_type='qwen3_5'")
        quantization = payload.get("quantization") or payload.get("quantization_config")
    text = payload.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Qwen3.5 config is missing text_config")
    if text.get("model_type", "qwen3_5_text") != "qwen3_5_text":
        raise ValueError("expected text_config.model_type='qwen3_5_text'")
    if int(text.get("num_experts", 0) or 0) > 0:
        raise ValueError("native-mlx does not implement Qwen3.5 mixture-of-experts")
    if not text.get("attn_output_gate", True):
        raise ValueError("native-mlx requires Qwen3.5 gated attention output")
    if text.get("mlp_only_layers"):
        raise ValueError("native-mlx does not implement Qwen3.5 mlp_only_layers")
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    for name in required:
        if name not in text:
            raise ValueError(f"missing text_config.{name}")
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text.get("head_dim") or hidden // heads)
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or head_dim <= 0:
        raise ValueError("invalid Qwen3.5 attention head dimensions")
    v_heads = int(text["linear_num_value_heads"])
    k_heads = int(text["linear_num_key_heads"])
    if k_heads <= 0 or v_heads <= 0 or v_heads % k_heads:
        raise ValueError("invalid Qwen3.5 Gated DeltaNet head dimensions")
    interval = int(text.get("full_attention_interval", 4))
    if interval <= 0:
        raise ValueError("invalid Qwen3.5 full_attention_interval")
    rope = text.get("rope_parameters") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        raise ValueError(f"native-mlx does not implement Qwen3.5 rope {rope_type!r}")
    partial = float(
        rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    )
    if not 0 < partial <= 1 or int(head_dim * partial) % 2:
        raise ValueError("invalid Qwen3.5 partial_rotary_factor")
    return Qwen3_5ModelConfig(
        architecture_class=ARCHITECTURE_CLASS,
        model_type=model_type,
        vocab_size=int(text["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        max_position_embeddings=int(text.get("max_position_embeddings", 262144)),
        linear_num_value_heads=v_heads,
        linear_num_key_heads=k_heads,
        linear_key_head_dim=int(text["linear_key_head_dim"]),
        linear_value_head_dim=int(text["linear_value_head_dim"]),
        linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
        full_attention_interval=interval,
        rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000_000))),
        partial_rotary_factor=partial,
        tie_word_embeddings=bool(
            text.get("tie_word_embeddings", payload.get("tie_word_embeddings", False))
        ),
        quantization=quantization,
        hadamard=hadamard,
    )


_LANGUAGE_PREFIXES = ("language_model.", "model.language_model.")
_IGNORED_PREFIXES = ("vision_tower.", "model.visual.", "visual.", "mtp.")
_CANONICAL_RE = re.compile(r"^(model\.(embed_tokens|norm|layers\.\d+)\.|lm_head\.)")


class Qwen3_5WeightAdapter:
    """Map Qwen3.5 language-model tensors into the native graph."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        entries = []
        for name, source_file in index.weight_map.items():
            canonical = _canonicalize_qwen3_5_name(name)
            if canonical is None:
                continue
            entries.append(
                WeightMappingEntry(
                    canonical_name=canonical,
                    source_name=name,
                    source_file=source_file,
                )
            )
        if not entries:
            raise WeightMappingBug("Qwen3.5 checkpoint produced empty mapping plan")
        return WeightMappingPlan(
            architecture_class=ARCHITECTURE_CLASS,
            source_files=index.source_files,
            entries=tuple(entries),
        )


def build_qwen3_5_model(
    config: Qwen3_5ModelConfig,
    weights: list[tuple[str, mx.array]],
) -> Qwen3_5ForCausalLM:
    """Construct, quantize, and load the Qwen3.5 graph.

    The returned model's ``resolved_config`` records the K/V activation dtype
    implied by the loaded weights, which sizes the paged KV pool.
    """

    for name, value in weights:
        if name.endswith("conv1d.weight") and int(value.shape[-1]) != 1:
            raise ValueError(
                "Qwen3.5 conv1d weights are not in MLX layout; convert the "
                "checkpoint with mlx-lm first"
            )
    model = Qwen3_5ForCausalLM(config)
    if config.hadamard is not None:
        install_hadamard_modules(model, config.hadamard)
    else:
        _quantize_model(model, config, weights)
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    model.resolved_config = replace(config, kv_cache_dtype=_kv_dtype_name(model))
    return model


def qwen3_5_kv_layers(config: Qwen3_5ModelConfig) -> tuple[int, ...]:
    """Return the layers that store paged KV."""

    return config.full_attention_layers


def _kv_dtype_name(model: Qwen3_5ForCausalLM) -> str:
    """Return the dtype the K/V projections produce for this model's weights.

    MLX keeps each tensor's stored dtype, so the residual stream is the
    promotion of the embedding output and the norm weights (float32 norms in a
    Hadamard pack keep it in float32), and a quantized projection's output is
    promoted with its scales.
    """

    hidden = model.model.embed_tokens(mx.array([0], dtype=mx.int32))
    attention = model.model.layers[model.config.full_attention_layers[0]].self_attn
    probe = attention.k_proj(model.model.layers[0].input_layernorm(hidden))
    dtype = (mx.zeros((1,), hidden.dtype) + mx.zeros((1,), probe.dtype)).dtype
    if dtype not in _KV_DTYPES:
        raise ValueError(f"unsupported Qwen3.5 activation dtype {dtype}")
    return _KV_DTYPES[dtype]


def _quantize_model(
    model: nn.Module,
    config: Qwen3_5ModelConfig,
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


def _canonicalize_qwen3_5_name(source_name: str) -> str | None:
    name = source_name
    for prefix in _LANGUAGE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            if prefix == "model.language_model.":
                name = "model." + name
            break
    if name.startswith(_IGNORED_PREFIXES) or source_name.startswith(_IGNORED_PREFIXES):
        return None
    if _CANONICAL_RE.match(name) is None:
        raise WeightMappingBug(
            f"Qwen3.5 has no mapping rule for tensor {source_name!r}"
        )
    return name


def _architecture(payload: dict[str, Any]) -> str:
    architectures = payload.get("architectures")
    architecture = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if architecture != ARCHITECTURE_CLASS:
        raise ValueError(f"expected architectures[0]={ARCHITECTURE_CLASS!r}")
    return architecture


EntryClass = Qwen3_5ForCausalLM
