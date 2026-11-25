"""Inkling-2-Mini text model served from LookAtMyQuant packs.

The graph has 24 physical layers. Layers 6 to 17 run three times per token,
which makes 48 layer visits; each visit keeps its own KV and convolution
history, so the cache has 48 layers. Passes after the first receive messages
computed from the previous pass.

The model never pads: the tokens of every request of a step form one token
axis, and only attention and the convolutions know which request a token
belongs to (see ``causal_conv.TokenLayout``).

The graph follows the reference runtime published with the packs
(lookatmyquant, MIT License, Copyright (c) 2026 Shin Megami Boson).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx

from ..causal_conv import TokenLayout, causal_conv, causal_keys_values
from ..inkling2_kernels import (
    blend_messages,
    mix_states,
    route_experts,
    row_products,
    scaled_clip,
)
from ..interfaces import ForwardBatch
from ..packed_matmul import (
    FORMAT_IDS,
    ExpertPlan,
    PackedBank,
    PackedRows,
    build_trellis_luts,
    plan_experts,
    single_token_plan,
)
from ..rotation import rotate_out, rotate_packed, rotate_ranges
from ..windowed_attention import WindowedLayerContext
from .inkling2_checkpoint import (
    ARCHITECTURE,
    Inkling2PackAdapter,
    load_packed_weight_index,
)

_KV_CONV, _ATTENTION_CONV, _MLP_CONV = range(3)
_SHORT_CONV_SCALE = 1.0 / math.sqrt(5.0)
_DYNAMIC_CONV_SCALE = 1.0 / math.sqrt(9.0)
_PACKED_INPUT_DTYPE = mx.float16
_GATE_DTYPE = mx.bfloat16
_FUSED_GATE_TOKENS = 32
_ROW_PRODUCT_TOKENS = 16
_VISITS_PER_LAUNCH = 12
_GATES = ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")


@dataclass(frozen=True)
class Inkling2Config:
    """Text-graph dimensions of an Inkling-2 looped checkpoint."""

    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    output_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rel_dim: int
    rel_extent: int
    log_scaling_floor: int
    log_scaling_alpha: float
    norm_eps: float
    qk_clip: float
    rel_clip: float
    window_layers: frozenset[int]
    window_size: int
    final_softcap: float
    block_layers: int
    num_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    gate_softcap: float
    up_softcap: float
    route_scale: float
    loop_start: int
    loop_end: int
    loop_passes: int
    kv_cache_dtype: str = "float16"

    @property
    def num_visits(self) -> int:
        """Return layer visits per token, which is the number of KV layers."""

        return len(self.visits)

    @property
    def visits(self) -> tuple[tuple[int, int], ...]:
        """Return ``(physical layer, pass)`` per visit in execution order."""

        loop = range(self.loop_start, self.loop_end)
        visits = [(layer, 0) for layer in range(self.loop_end)]
        for step in range(1, self.loop_passes):
            visits += [(layer, step) for layer in loop]
        visits += [(layer, 0) for layer in range(self.loop_end, self.num_hidden_layers)]
        return tuple(visits)

    @property
    def window_visits(self) -> tuple[int, ...]:
        """Return the visits whose attention uses the sliding window."""

        return tuple(
            visit
            for visit, (layer, _) in enumerate(self.visits)
            if layer in self.window_layers
        )


def parse_inkling2_config(payload: dict[str, Any]) -> Inkling2Config:
    """Validate ``config.json`` of an Inkling-2 looped checkpoint.

    Args:
        payload: Parsed ``config.json``.

    Returns:
        The text-graph configuration.

    Raises:
        ValueError: If the architecture or a dimension is unsupported.
    """

    architectures = payload.get("architectures")
    if not isinstance(architectures, list) or architectures[:1] != [ARCHITECTURE]:
        raise ValueError(f"expected architectures[0]={ARCHITECTURE!r}")
    text = payload.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Inkling-2 config is missing text_config")
    loop = text.get("loop_config")
    if not isinstance(loop, dict):
        raise ValueError("Inkling-2 config is missing loop_config")
    try:
        config = Inkling2Config(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            vocab_size=int(text["vocab_size"]),
            output_size=int(text.get("unpadded_vocab_size") or text["vocab_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            rel_dim=int(text["d_rel"]),
            rel_extent=int(text["rel_extent"]),
            log_scaling_floor=int(text["log_scaling_n_floor"]),
            log_scaling_alpha=float(text["log_scaling_alpha"]),
            norm_eps=float(text["rms_norm_eps"]),
            qk_clip=float(text["qk_l2_clip_threshold"]),
            rel_clip=float(text["rel_l2_clip_threshold"]),
            window_layers=frozenset(int(layer) for layer in text["local_layer_ids"]),
            window_size=int(text["sliding_window_size"]),
            final_softcap=float(text["final_logit_softcapping"]),
            block_layers=int(text["n_layers_per_attn_res_block"]),
            num_experts=int(text["n_routed_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            intermediate_size=int(text["intermediate_size"]),
            gate_softcap=float(text["mlp_gate_softcap"]),
            up_softcap=float(text["mlp_up_softcap"]),
            route_scale=float(text["route_scale"]),
            loop_start=int(loop["core_start"]),
            loop_end=int(loop["core_end"]),
            loop_passes=int(loop["num_passes"]),
        )
    except KeyError as exc:
        raise ValueError(f"Inkling-2 config is missing {exc.args[0]!r}") from exc
    if int(text.get("sconv_kernel_size", 4)) != 4:
        raise ValueError("Inkling-2 convolutions must have kernel size 4")
    if int(text.get("n_shared_experts") or 0) or loop.get("is_varloop"):
        raise ValueError("Inkling-2 shared experts and variable loops are unsupported")
    if not 0 < config.loop_start < config.loop_end < config.num_hidden_layers:
        raise ValueError("Inkling-2 loop must lie strictly inside the layers")
    if config.loop_start % config.block_layers or config.loop_passes < 1:
        raise ValueError("Inkling-2 loop must start at a block boundary")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return config


def _rms_norm(x: mx.array, weight: mx.array | None = None) -> mx.array:
    return mx.fast.rms_norm(x, weight, 1e-6)


@partial(mx.compile, shapeless=True)
def _blend(hidden: mx.array, message: mx.array, forget: mx.array, keep: mx.array):
    forget = mx.tanh(forget.astype(mx.float32))
    keep = mx.tanh(keep.astype(mx.float32))
    return (1.0 - forget) * hidden + keep * message


def _small_product(x: mx.array, weight: mx.array) -> mx.array:
    if int(x.shape[0]) <= _ROW_PRODUCT_TOKENS:
        return row_products(x, weight)
    return x @ weight.T


class MessageGates:
    """Blend hidden states with messages from the previous loop pass.

    The gates of one step share a batched product, one gate per batch entry.
    """

    def __init__(self, gates: Sequence[tuple[mx.array, mx.array, mx.array]]):
        self.down = mx.stack([down.T for down, _, _ in gates]).astype(_GATE_DTYPE)
        self.up = mx.stack(
            [mx.concatenate([forget, message]).T for _, forget, message in gates]
        ).astype(_GATE_DTYPE)

    def __call__(self, hidden: mx.array, messages: mx.array) -> mx.array:
        """Blend ``(gates, tokens, hidden)`` states and messages.

        ``hidden`` may hold one state for all gates.
        """

        if int(messages.shape[1]) <= _FUSED_GATE_TOKENS:
            return blend_messages(hidden, messages, self.down, self.up)
        normed = _rms_norm(hidden)
        if int(hidden.shape[0]) != int(messages.shape[0]):
            normed = mx.broadcast_to(normed, messages.shape)
        joined = mx.concatenate([normed, _rms_norm(messages)], axis=-1)
        gate = joined.astype(_GATE_DTYPE) @ self.down
        forget, keep = mx.split(gate @ self.up, 2, axis=-1)
        return _blend(hidden, messages, forget, keep)


class _Weights:
    """Mapped tensors, handed out once so their buffers can be released."""

    def __init__(self, weights: list[tuple[str, mx.array]]):
        self._tensors = dict(weights)
        weights.clear()

    def take(self, name: str, dtype: Any = None) -> mx.array:
        try:
            tensor = self._tensors.pop(name)
        except KeyError as exc:
            raise ValueError(f"Inkling-2 checkpoint is missing {name!r}") from exc
        return tensor if dtype is None else tensor.astype(dtype)

    def maybe(self, name: str, dtype: Any = None) -> mx.array | None:
        return self.take(name, dtype) if name in self._tensors else None

    def has(self, name: str) -> bool:
        return name in self._tensors

    def peek(self, name: str) -> mx.array:
        return self._tensors[name]

    def formats(self, prefix: str) -> list[str]:
        """Return the formats stored under ``prefix.{format}.codes``."""

        found = {
            name.removeprefix(prefix + ".").split(".", 1)[0]
            for name in self._tensors
            if name.startswith(prefix + ".") and name.endswith(".codes")
        }
        return sorted(found, key=lambda fmt: FORMAT_IDS[fmt])

    def packed(self, prefix: str, rows: int, **placement: int) -> PackedRows:
        (fmt,) = self.formats(prefix)
        return PackedRows(
            fmt,
            rows,
            self.take(f"{prefix}.{fmt}.codes"),
            self.take(f"{prefix}.{fmt}.scales"),
            levels=self.maybe(f"{prefix}.{fmt}.levels"),
            **placement,
        )

    def signs(self, name: str) -> mx.array:
        return self.take(name, mx.float32)


class PackedProjections:
    """Packed matrices that share an input width, each with its own rotation.

    Attributes:
        inputs: Input each matrix reads, as an index into the call's ``x``.
        ranges: Output columns of each matrix.
    """

    def __init__(
        self,
        weights: _Weights,
        prefixes: Sequence[str],
        widths: Sequence[int],
        input_width: int,
        luts: mx.array,
        inputs: Sequence[int],
    ):
        matrices = []
        signs = []
        self.ranges: list[tuple[int, int]] = []
        self.out_signs: list[mx.array | None] = []
        start = 0
        for index, (prefix, width) in enumerate(zip(prefixes, widths, strict=True)):
            (fmt,) = weights.formats(prefix)
            self.out_signs.append(
                weights.maybe(f"{prefix}.{fmt}.signs_out", mx.float32)
            )
            signs.append(weights.signs(f"{prefix}.{fmt}.signs_in"))
            matrices.append(
                weights.packed(prefix, width, output_row=start, input_index=index)
            )
            self.ranges.append((start, start + width))
            start += width
        self.bank = PackedBank(input_width, matrices, luts)
        self.signs = mx.stack(signs)
        self.inputs = tuple(inputs)
        self._sources = mx.array(self.inputs, dtype=mx.uint32)
        self.rotates_out = any(value is not None for value in self.out_signs)

    def arrays(self) -> list[mx.array]:
        """Return the arrays the projections keep alive."""

        return [
            *self.bank.arrays(),
            self.signs,
            *(value for value in self.out_signs if value is not None),
        ]

    def __call__(
        self, x: mx.array, gate: tuple[mx.array, int] | None = None
    ) -> mx.array:
        """Multiply every matrix with its rotated input.

        Args:
            x: ``(inputs, tokens, width)`` inputs.
            gate: ``(tokens, columns)`` array and the first of the columns whose
                sigmoid scales the inputs.

        Returns:
            ``(tokens, output columns)`` products of all matrices.
        """

        rotated = rotate_packed(
            x, self.signs, inputs=self._sources, dtype=_PACKED_INPUT_DTYPE, gate=gate
        )
        y = self.bank.dense(rotated)
        if not self.rotates_out:
            return y
        return rotate_ranges(y, self.ranges, self.out_signs)


class PackedExperts:
    """The expert matrices of one side (gate-up or down) of a layer."""

    def __init__(
        self,
        weights: _Weights,
        prefix: str,
        rows: int,
        input_width: int,
        luts: mx.array,
    ):
        matrices = []
        for fmt in weights.formats(prefix):
            codes = weights.take(f"{prefix}.{fmt}.codes")
            matrices.append(
                PackedRows(
                    fmt,
                    int(codes.shape[1]),
                    codes,
                    weights.take(f"{prefix}.{fmt}.scales"),
                    outer=weights.maybe(f"{prefix}.{fmt}.outer"),
                )
            )
        self.signs = weights.signs(f"{prefix}.signs")
        table = weights.take(f"{prefix}.map").tolist()
        groups = [
            [tuple(group) for group in expert]
            if isinstance(expert[0], list)
            else [tuple(expert)]
            for expert in table
        ]
        self.bank = PackedBank(input_width, matrices, luts, expert_groups=groups)
        if self.bank.output_rows != rows:
            raise ValueError(f"{prefix} experts must have {rows} rows")

    def arrays(self) -> list[mx.array]:
        """Return the arrays the experts keep alive."""

        return [*self.bank.arrays(), self.signs]

    def __call__(
        self,
        x: mx.array,
        experts: mx.array,
        plan: ExpertPlan,
        *,
        x_div: int,
        soft_caps: mx.array | None = None,
    ) -> mx.array:
        """Multiply each slot's input with its expert.

        Args:
            x: ``(slots / x_div, input width)`` inputs, or the halves of
                soft-gated inputs with ``soft_caps``.
            experts: ``(slots,)`` expert per slot.
            plan: Grouping of ``experts``.
            x_div: Consecutive slots that share one input row.
            soft_caps: Caps of a soft gate.

        Returns:
            ``(slots, rows)`` products.
        """

        rotated = rotate_packed(
            x[None],
            self.signs[None],
            dtype=_PACKED_INPUT_DTYPE,
            soft_caps=soft_caps,
            keep_input=self.bank.needs_raw_input,
        )
        return self.bank.expert(rotated, experts, plan, x_div=x_div)


class Inkling2Experts:
    """Sigmoid-routed mixture of packed experts."""

    def __init__(
        self, config: Inkling2Config, weights: _Weights, layer: int, luts: mx.array
    ):
        self.config = config
        prefix = f"model.layers.{layer}.mlp.gate"
        self.router = weights.take(f"{prefix}.weight", mx.float32)
        self.bias = weights.take(f"{prefix}.e_score_correction_bias", mx.float32)
        self.soft_caps = mx.array(
            (config.gate_softcap, config.up_softcap), dtype=mx.float32
        )
        self.gate_up = PackedExperts(
            weights,
            f"layers.{layer}.experts.gu",
            2 * config.intermediate_size,
            config.hidden_size,
            luts,
        )
        self.down = PackedExperts(
            weights,
            f"layers.{layer}.experts.dn",
            config.hidden_size,
            config.intermediate_size,
            luts,
        )

    def arrays(self) -> list[mx.array]:
        """Return the arrays the experts keep alive."""

        return [self.router, self.bias, *self.gate_up.arrays(), *self.down.arrays()]

    def __call__(self, x: mx.array) -> mx.array:
        config = self.config
        tokens, hidden = (int(size) for size in x.shape)
        top = config.num_experts_per_tok
        chosen, weights = route_experts(
            _small_product(x, self.router), self.bias, top=top, scale=config.route_scale
        )
        experts = chosen.reshape(-1)
        plan = (
            single_token_plan(top)
            if tokens == 1
            else plan_experts(experts, config.num_experts, tokens)
        )
        gate_up = self.gate_up(x, experts, plan, x_div=top)
        down = self.down(gate_up, experts, plan, x_div=1, soft_caps=self.soft_caps)
        return (weights[:, None] @ down.reshape(tokens, top, hidden))[:, 0]


class Inkling2Layer:
    """One physical layer, shared by every pass that visits it."""

    def __init__(
        self, config: Inkling2Config, weights: _Weights, layer: int, luts: mx.array
    ):
        self.config = config
        self.layer = layer
        self.sliding = layer in config.window_layers
        self.starts_block = layer % config.block_layers == 0
        prefix = f"model.layers.{layer}"
        hidden = config.hidden_size
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        query_width = heads * config.head_dim
        kv_width = kv_heads * config.head_dim

        def take(name: str) -> mx.array:
            return weights.take(f"{prefix}.{name}", mx.float32)

        attention_query = weights.maybe(f"{prefix}.attention_input.query", mx.float32)
        self.attention_query = (
            None if attention_query is None else attention_query.reshape(-1, hidden)
        )
        self.mlp_query = take("mlp_input.query").reshape(1, hidden)
        norms = (
            ("input_layernorm",)
            if self.sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        self.input_norms = mx.stack([take(f"{name}.weight") for name in norms])
        self.mlp_norm = take("post_attention_layernorm.weight")
        self.q_scale = take("self_attn.q_proj.norm.scale")
        self.rel_scale = take("self_attn.relative_position_bias.norm.scale")
        self.rel_proj = mx.contiguous(take("self_attn.relative_position_bias.proj").T)
        self.sink = take("self_attn.sink_proj.weight")
        self.dynamic: tuple[mx.array, mx.array] | None = None
        convs = []
        dynamic = []
        for kind in ("k", "v"):
            name = f"self_attn.{kind}_proj.sconv"
            if self.sliding:
                convs.append(_lag_first(take(f"{name}.conv1d.weight")))
            else:
                convs.append(take(f"{name}.static_weight"))
                dynamic.append(take(f"{name}.weight_proj.weight"))
        self.kv_conv = mx.concatenate(convs, axis=-1)
        if dynamic:
            self.dynamic = (dynamic[0], dynamic[1])
        self.attention_conv = _lag_first(
            take("post_attention_convolution.conv1d.weight")
        )
        self.mlp_conv = _lag_first(take("post_mlp_convolution.conv1d.weight"))
        self.gates: tuple[MessageGates, MessageGates] | None = None
        if config.loop_start <= layer < config.loop_end:
            gates = [
                (
                    take(f"{name}.down_proj.weight"),
                    take(f"{name}.forget_gate_proj.weight"),
                    take(f"{name}.message_gate_proj.weight"),
                )
                for name in _GATES
            ]
            self.gates = (MessageGates(gates[:3]), MessageGates(gates[3:]))
        linear = f"layers.{layer}.linear"
        names = ("q", "k", "v", "rel", "gate")
        self.projections = PackedProjections(
            weights,
            [f"{linear}.{name}" for name in names],
            (query_width, kv_width, kv_width, heads * config.rel_dim, query_width),
            hidden,
            luts,
            inputs=(0, 1, 2, 0, 0),
        )
        self.output = PackedProjections(
            weights, [f"{linear}.o"], (hidden,), query_width, luts, inputs=(0,)
        )
        self.experts = Inkling2Experts(config, weights, layer, luts)
        mx.eval(self.arrays())

    def arrays(self) -> list[mx.array]:
        """Return the arrays the layer keeps alive."""

        values: list[Any] = [
            self.attention_query,
            self.mlp_query,
            self.input_norms,
            self.mlp_norm,
            self.q_scale,
            self.rel_scale,
            self.rel_proj,
            self.sink,
            self.kv_conv,
            self.attention_conv,
            self.mlp_conv,
            *(self.dynamic or ()),
            *(array for gate in self.gates or () for array in (gate.down, gate.up)),
            *self.projections.arrays(),
            *self.output.arrays(),
            *self.experts.arrays(),
        ]
        return [value for value in values if value is not None]

    def __call__(
        self,
        hidden: mx.array,
        blocks: Sequence[mx.array],
        context: WindowedLayerContext,
        log_scale: mx.array | None,
        messages: mx.array | None,
    ) -> mx.array:
        """Run one visit.

        Args:
            hidden: ``(tokens, hidden)`` state entering the visit.
            blocks: Residual states of earlier blocks of this pass.
            context: Cache access of this visit.
            log_scale: ``(tokens,)`` scale of global queries, if any exceeds 1.
            messages: ``(4, tokens, hidden)`` query, key, value, and MLP
                messages on passes after the first.

        Returns:
            The state leaving the visit.
        """

        mixed = (
            hidden[None]
            if self.attention_query is None
            else mix_states(self.attention_query, [*blocks, hidden])
        )
        if messages is not None:
            assert self.gates is not None
            mixed = self.gates[0](mixed, messages[:3])
        normed = _rms_norm(mixed) * self.input_norms[:, None, :]
        normed = mx.broadcast_to(normed, (3, *hidden.shape))
        attention = self._attend(normed, context, None if self.sliding else log_scale)
        if self.starts_block:
            blocks = [*blocks, hidden]
            residual = None
        else:
            residual = hidden
        residual = self._convolve(
            attention, context, _ATTENTION_CONV, self.attention_conv, residual
        )
        mlp_input = mix_states(self.mlp_query, [*blocks, residual])
        if messages is not None:
            assert self.gates is not None
            mlp_input = self.gates[1](mlp_input, messages[3:])
        mlp = self.experts(_rms_norm(mlp_input[0], self.mlp_norm))
        return self._convolve(mlp, context, _MLP_CONV, self.mlp_conv, residual)

    def _attend(
        self,
        normed: mx.array,
        context: WindowedLayerContext,
        log_scale: mx.array | None,
    ) -> mx.array:
        config = self.config
        heads, head_dim = config.num_attention_heads, config.head_dim
        projected = self.projections(normed)
        _, (k_start, _), _, (rel_start, _), (gate_start, _) = self.projections.ranges
        queries, rel = scaled_clip(
            projected,
            self.q_scale,
            self.rel_scale,
            heads=heads,
            head_dim=head_dim,
            rel_start=rel_start,
            rel_dim=config.rel_dim,
            clips=(config.qk_clip, config.rel_clip),
            token_scale=log_scale,
        )
        kv = self._keys_values(projected, k_start, normed, context)
        attention = context.append_and_attend(
            queries,
            kv,
            rel=rel,
            rel_proj=self.rel_proj,
            sinks=_small_product(normed[0], self.sink),
            scale=head_dim**-0.5,
        )
        return self.output(attention[None], gate=(projected, gate_start))

    def _keys_values(
        self,
        projected: mx.array,
        offset: int,
        normed: mx.array,
        context: WindowedLayerContext,
    ) -> mx.array:
        config = self.config
        kv_heads, head_dim = config.num_key_value_heads, config.head_dim
        channels = 2 * kv_heads * head_dim
        history = context.conv_history(_KV_CONV, channels)
        dynamic = None
        if self.dynamic is not None:
            dynamic = mx.concatenate(
                [
                    _small_product(normed[1], self.dynamic[0]),
                    _small_product(normed[2], self.dynamic[1]),
                ],
                axis=-1,
            )
        kv, after = causal_keys_values(
            projected,
            history,
            self.kv_conv,
            context.layout,
            offset=offset,
            heads=2 * kv_heads,
            head_dim=head_dim,
            scale=_SHORT_CONV_SCALE if self.sliding else _DYNAMIC_CONV_SCALE,
            clip=config.qk_clip,
            eps=config.norm_eps,
            dynamic_weights=dynamic,
        )
        context.stage_conv(
            _KV_CONV, history, projected[:, offset : offset + channels], after
        )
        return kv

    def _convolve(
        self,
        x: mx.array,
        context: WindowedLayerContext,
        conv_index: int,
        weights: mx.array,
        residual: mx.array | None,
    ) -> mx.array:
        history = context.conv_history(conv_index, int(x.shape[-1]))
        out, after = causal_conv(
            x,
            history,
            weights,
            context.layout,
            scale=_SHORT_CONV_SCALE,
            residual=residual,
        )
        context.stage_conv(conv_index, history, x, after)
        return out


def _lag_first(weight: mx.array) -> mx.array:
    """Return a depthwise ``(channels, 1, lags)`` weight, current token first."""

    return mx.contiguous(weight[:, 0, ::-1].T)


class Inkling2Embedding:
    """Token embedding stored as packed frequency buckets."""

    def __init__(self, config: Inkling2Config, weights: _Weights, luts: mx.array):
        prefixes, rows = _bucket_layout(weights, "embed")
        signs = []
        matrices = []
        start = 0
        ends = []
        for prefix, count in zip(prefixes, rows, strict=True):
            (fmt,) = weights.formats(prefix)
            signs.append(weights.signs(f"{prefix}.{fmt}.signs_in"))
            matrices.append(weights.packed(prefix, count, output_row=start))
            start += count
            ends.append(start)
        self.bank = PackedBank(config.hidden_size, matrices, luts)
        self.signs = mx.stack(signs)
        self.ends = mx.array(ends, dtype=mx.int32)
        self.position = _inverse(weights.take("embed.order"))
        self.norm = weights.take("model.embed_tokens.embed_norm.weight", mx.float32)
        mx.eval(self.arrays())

    def arrays(self) -> list[mx.array]:
        """Return the arrays the embedding keeps alive."""

        return [
            *self.bank.arrays(),
            self.signs,
            self.ends,
            self.position,
            self.norm,
        ]

    def __call__(self, tokens: mx.array) -> mx.array:
        rows = self.position[tokens]
        bucket = (rows[:, None] >= self.ends[None]).sum(axis=-1)
        decoded = self.bank.rows(rows.astype(mx.uint32))
        return _rms_norm(rotate_out(decoded, self.signs[bucket]), self.norm)


class Inkling2Head:
    """Output projection stored as packed frequency buckets."""

    def __init__(self, config: Inkling2Config, weights: _Weights, luts: mx.array):
        prefixes, rows = _bucket_layout(weights, "head")
        self.projections = PackedProjections(
            weights,
            prefixes,
            rows,
            config.hidden_size,
            luts,
            inputs=(0,) * len(prefixes),
        )
        self.order = weights.take("head.order", mx.int32)
        self.position = _inverse(self.order)
        self.softcap = config.final_softcap
        mx.eval(*self.projections.arrays(), self.position)

    def shortlist(self, buckets: int, bits: int) -> TokenShortlist:
        """Return a compact head over the rows of the first buckets.

        Args:
            buckets: Leading frequency buckets to include.
            bits: Bits of the requantized rows.

        Returns:
            A head that names the best of those tokens.
        """

        return TokenShortlist(self, buckets, bits)

    def __call__(self, hidden: mx.array) -> mx.array:
        ordered = self.projections(hidden[None])
        logits = ordered[:, self.position]
        return mx.tanh(logits / self.softcap) * self.softcap


class TokenShortlist:
    """Requantized output rows of the most frequent tokens.

    Drafting only needs a likely token, so it scores these rows instead of the
    whole packed head. Verification always uses the packed head.
    """

    def __init__(self, head: Inkling2Head, buckets: int, bits: int):
        projections = head.projections
        self.mode = {"group_size": 64, "bits": bits}
        parts = []
        for bucket, (start, end) in enumerate(projections.ranges[:buckets]):
            for first in range(start, end, 4096):
                rows = mx.arange(first, min(end, first + 4096), dtype=mx.uint32)
                decoded = rotate_out(
                    projections.bank.rows(rows), projections.signs[bucket]
                )
                parts.append(mx.quantize(decoded.astype(mx.float16), **self.mode))
                mx.eval(*parts[-1])
        self.weight, self.scales, self.biases = (
            mx.concatenate(list(arrays)) for arrays in zip(*parts, strict=True)
        )
        self.order = head.order[: int(self.weight.shape[0])]
        mx.eval(self.weight, self.scales, self.biases, self.order)

    def __call__(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        """Return the best token of a ``(hidden,)`` state and its probability.

        Both arrays have shape ``(1,)``.
        """

        scores = mx.quantized_matmul(
            hidden[None].astype(mx.float16),
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            **self.mode,
        ).astype(mx.float32)
        confidence = mx.exp(mx.max(scores, axis=-1) - mx.logsumexp(scores, axis=-1))
        return self.order[mx.argmax(scores, axis=-1)], confidence


def _bucket_layout(weights: _Weights, part: str) -> tuple[list[str], list[int]]:
    prefixes: list[str] = []
    rows: list[int] = []
    while formats := weights.formats(f"{part}.{len(prefixes)}"):
        prefix = f"{part}.{len(prefixes)}"
        rows.append(int(weights.peek(f"{prefix}.{formats[0]}.scales").size))
        prefixes.append(prefix)
    return prefixes, rows


def _inverse(order: mx.array) -> mx.array:
    inverse = mx.zeros(order.shape, dtype=mx.int32)
    inverse[order] = mx.arange(int(order.size), dtype=mx.int32)
    return inverse


class Inkling2LoopForConditionalGeneration:
    """Looped Inkling-2 text graph over packed weights."""

    def __init__(self, config: Inkling2Config, weights: list[tuple[str, mx.array]]):
        source = _Weights(weights)
        tables = {
            fmt: source.take(f"tables.{fmt}")
            for fmt in FORMAT_IDS
            if source.has(f"tables.{fmt}")
        }
        self.config = config
        self.num_layers = config.num_visits
        self.luts = build_trellis_luts(tables)
        mx.eval(self.luts)
        self.embedding = Inkling2Embedding(config, source, self.luts)
        self.layers = [
            Inkling2Layer(config, source, layer, self.luts)
            for layer in range(config.num_hidden_layers)
        ]
        self.head = Inkling2Head(config, source, self.luts)
        self.message_queries = source.take("model.message_input.query", mx.float32)
        self.final_query = source.take("model.final_input.query", mx.float32)[None]
        self.final_norm = source.take("model.norm.weight", mx.float32)
        mx.eval(self.message_queries, self.final_query, self.final_norm)

    def load_weights(
        self, weights: Sequence[tuple[str, mx.array]], *, strict: bool = True
    ) -> None:
        """Reject reloading: packed banks are assembled at construction."""

        raise NotImplementedError("Inkling-2 packs load once, at construction")

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        """Run one step.

        Args:
            input_ids: ``(rows, tokens)`` token ids, padded at the end.
            positions: Unused; positions follow from the cache lengths.
            forward_batch: Step metadata.

        Returns:
            ``(rows, 1, vocab)`` logits at the requested positions, or
            ``(rows, tokens, vocab)`` logits when none are named.
        """

        del positions
        hidden, _ = self.forward_hidden(input_ids, forward_batch)
        return self._logits(hidden, forward_batch, int(input_ids.shape[1]))

    def _logits(
        self, hidden: mx.array, forward_batch: ForwardBatch, width: int
    ) -> mx.array:
        selected = forward_batch.logits_positions
        if selected is None:
            return self._padded(self.head(hidden), forward_batch, width)
        contexts: Any = forward_batch.layer_attention
        layout: TokenLayout = contexts[0].layout
        if int(hidden.shape[0]) != len(selected):
            hidden = hidden[layout.starts + mx.array(selected, dtype=mx.int32)]
        return self.head(hidden)[:, None, :]

    def forward_with_taps(
        self,
        input_ids: mx.array,
        forward_batch: ForwardBatch,
        taps: tuple[int, ...],
    ) -> tuple[mx.array, mx.array]:
        """Run one step and return tapped states beside the logits.

        Args:
            input_ids: ``(rows, tokens)`` token ids, padded at the end.
            forward_batch: Step metadata.
            taps: Visits whose input state a drafter reads.

        Returns:
            Logits as :meth:`__call__` returns them, and the
            ``(total tokens, taps, hidden)`` tapped states.
        """

        hidden, captured = self.forward_hidden(
            input_ids, forward_batch, frozenset(taps)
        )
        logits = self._logits(hidden, forward_batch, int(input_ids.shape[1]))
        return logits, mx.stack([captured[visit] for visit in taps], axis=1)

    def forward_hidden(
        self,
        input_ids: mx.array,
        forward_batch: ForwardBatch,
        taps: frozenset[int] = frozenset(),
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Run the layers.

        Args:
            input_ids: ``(rows, tokens)`` token ids, padded at the end.
            forward_batch: Step metadata.
            taps: Visits whose input state to return; the number of visits
                names the final normalized state.

        Returns:
            ``(total tokens, hidden)`` normalized output states of every row's
            tokens, concatenated, and the tapped states.
        """

        config = self.config
        contexts: Any = forward_batch.layer_attention
        layout: TokenLayout = contexts[0].layout
        tokens = input_ids.reshape(-1)
        if layout.tokens != int(tokens.shape[0]):
            width = int(input_ids.shape[1])
            tokens = tokens[layout.rows * width + layout.index]
        hidden = self.embedding(tokens)
        log_scale = self._log_scale(layout)
        captured: dict[int, mx.array] = {}
        blocks: list[mx.array] = []
        entry: tuple[mx.array, list[mx.array]] = (hidden, blocks)
        previous: tuple[mx.array, list[mx.array]] = entry
        for visit, (index, step) in enumerate(config.visits):
            if index == config.loop_start:
                if step == 0:
                    entry = (hidden, blocks)
                else:
                    previous = (hidden, blocks[len(entry[1]) + 1 :])
                    hidden, blocks = entry
            if visit in taps:
                captured[visit] = hidden
            messages = None
            if step:
                messages = mix_states(
                    self.message_queries[index - config.loop_start],
                    [*previous[1], previous[0]],
                )
            layer = self.layers[index]
            output = layer(hidden, blocks, contexts[visit], log_scale, messages)
            if layer.starts_block:
                blocks = [*blocks, hidden]
            hidden = output
            if forward_batch.device_idle and visit % _VISITS_PER_LAUNCH == 3:
                mx.async_eval(hidden)
        hidden = mix_states(self.final_query, [*blocks, hidden])[0]
        hidden = _rms_norm(hidden, self.final_norm)
        if config.num_visits in taps:
            captured[config.num_visits] = hidden
        return hidden, captured

    def _log_scale(self, layout: TokenLayout) -> mx.array | None:
        config = self.config
        last = max(
            cached + count
            for cached, count in zip(
                layout.cache_lengths, layout.token_lengths, strict=True
            )
        )
        if last <= config.log_scaling_floor:
            return None
        ratio = (layout.positions.astype(mx.float32) + 1.0) / config.log_scaling_floor
        return 1.0 + config.log_scaling_alpha * mx.log(mx.maximum(ratio, 1.0))

    @staticmethod
    def _padded(logits: mx.array, forward_batch: ForwardBatch, width: int) -> mx.array:
        rows = []
        start = 0
        for count in forward_batch.token_lengths:
            row = logits[start : start + count]
            rows.append(mx.pad(row, [(0, width - count), (0, 0)]))
            start += count
        return mx.stack(rows)


def build_inkling2_model(
    config: Inkling2Config, weights: list[tuple[str, mx.array]]
) -> Inkling2LoopForConditionalGeneration:
    """Construct the packed graph, consuming ``weights``."""

    return Inkling2LoopForConditionalGeneration(config, weights)


__all__ = [
    "Inkling2Config",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2PackAdapter",
    "build_inkling2_model",
    "load_packed_weight_index",
    "parse_inkling2_config",
]

EntryClass = Inkling2LoopForConditionalGeneration
