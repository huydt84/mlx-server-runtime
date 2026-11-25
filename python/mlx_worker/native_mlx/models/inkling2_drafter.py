"""DSpark drafter that proposes token blocks for Inkling-2-Mini.

The drafter is a three-layer transformer that predicts a block of tokens in one
pass. It reads the target model's hidden states at four visits ("taps") through
its own cross-attention keys and values, then a small recurrent head turns the
block's states into tokens one by one. Candidate tokens are scored with a 4-bit
copy of the most frequent rows of the target's output head.

The graph follows the reference runtime published with the packs
(lookatmyquant, MIT License, Copyright (c) 2026 Shin Megami Boson).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from .inkling2 import (
    Inkling2Config,
    Inkling2LoopForConditionalGeneration,
    parse_inkling2_config,
)

_FORMAT = "inkling2-dspark-4bit-v1"
_SHORTLIST_BUCKETS = 5
_SHORTLIST_TOKENS = 32768
_SHORTLIST_BITS = 4
_GROW_TOKENS = 256


@dataclass(frozen=True)
class Inkling2DrafterCheckpoint:
    """Resolved drafter checkpoint.

    Attributes:
        weights: Weight file used for the served pack.
        config: ``mtp_config`` section of the drafter's ``config.json``.
        target_config: Target architecture the drafter was made for.
        bits: Bits of quantized matrices.
        group_size: Group size of quantized matrices.
    """

    weights: Path
    config: dict[str, Any]
    target_config: Inkling2Config
    bits: int
    group_size: int

    @property
    def nbytes(self) -> int:
        """Return the bytes of the drafter and its token shortlist."""

        entries = _SHORTLIST_TOKENS * int(self.config["hidden_size"])
        shortlist = entries * _SHORTLIST_BITS // 8 + entries // 64 * 4
        return self.weights.stat().st_size + shortlist

    @property
    def kv_share(self) -> float:
        """Return the part of the KV budget that drafter contexts need.

        Windowed drafter layers are a smaller part of the target's than the
        global ones, so the global layers bound the share.
        """

        target = self.target_config.num_visits - len(self.target_config.window_visits)
        layers = int(self.config["num_hidden_layers"])
        own = layers - len(self.config["local_layer_ids"])
        return own / (own + target)

    def build(self, target: Any) -> Inkling2Drafter:
        """Load the drafter for a loaded target model."""

        return Inkling2Drafter(self, target)


def load_drafter(drafter_path: Path, model_path: Path) -> Inkling2DrafterCheckpoint:
    """Select the drafter weights that fit a pack.

    A drafter distilled for the served pack is preferred; its file name carries
    the pack's size. Otherwise the file named by the manifest is used.

    Args:
        drafter_path: Local drafter checkpoint directory.
        model_path: Local directory of the served pack.

    Returns:
        The drafter checkpoint.

    Raises:
        ValueError: If the directory is not a supported drafter.
    """

    try:
        manifest = json.loads((drafter_path / "manifest.json").read_text())
        raw = json.loads((drafter_path / "config.json").read_text())
        config, target_config = raw["mtp_config"], parse_inkling2_config(raw)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{drafter_path} is not a DSpark drafter: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT:
        raise ValueError(f"drafter manifest format must be {_FORMAT!r}")
    weights = drafter_path / str(manifest.get("weights"))
    tag = _pack_tag(model_path)
    if tag is not None:
        distilled = sorted(drafter_path.glob(f"mtp_4bit_{tag}_v*.safetensors"))
        if distilled:
            weights = distilled[-1]
    if not weights.is_file():
        raise ValueError(f"drafter weights {weights.name!r} are missing")
    return Inkling2DrafterCheckpoint(
        weights,
        config,
        target_config,
        int(manifest["bits"]),
        int(manifest["group_size"]),
    )


def _pack_tag(model_path: Path) -> str | None:
    """Return a pack's size as drafter file names spell it, such as ``5p4gb``."""

    try:
        manifest = json.loads((model_path / "manifest.json").read_text())
        size = float(manifest["plan_total_gb"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return f"{size:.1f}gb".replace(".", "p")


class _Matrix:
    """Quantized matrix stored as weight, scales, and biases."""

    def __init__(
        self,
        weights: dict[str, mx.array],
        names: list[str],
        files: Inkling2DrafterCheckpoint,
    ):
        keys = [f"{name}.weight" for name in names]
        missing = [key for key in keys if f"{key}.scales" not in weights]
        if missing:
            raise ValueError(f"drafter matrix {missing[0]!r} is not quantized")
        self.weight = mx.concatenate([weights[key] for key in keys])
        self.scales = mx.concatenate([weights[f"{key}.scales"] for key in keys])
        self.biases = mx.concatenate([weights[f"{key}.biases"] for key in keys])
        self.mode = {"group_size": files.group_size, "bits": files.bits}

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x.astype(mx.bfloat16),
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            **self.mode,
        )

    def rows(self, index: mx.array) -> mx.array:
        return mx.dequantize(
            self.weight[index], self.scales[index], self.biases[index], **self.mode
        ).astype(mx.float32)


@dataclass
class _LayerContext:
    """Keys and values of one drafter layer for one request."""

    keys: mx.array | None = None
    values: mx.array | None = None
    used: int = 0


@dataclass
class DraftContext:
    """Target states one request's drafter has read.

    Attributes:
        length: Committed target tokens read so far.
        summary: State the recurrent head starts from.
        layers: Per-layer keys and values.
    """

    length: int = 0
    summary: mx.array | None = None
    layers: list[_LayerContext] = field(default_factory=list)

    def arrays(self) -> list[mx.array]:
        """Return the arrays holding this context."""

        values = [self.summary]
        for layer in self.layers:
            values += [layer.keys, layer.values]
        return [value for value in values if value is not None]


class _DrafterLayer:
    def __init__(
        self,
        weights: dict[str, mx.array],
        index: int,
        files: Inkling2DrafterCheckpoint,
    ):
        prefix = f"layers.{index}"

        def norm(name: str, dtype: Any = mx.bfloat16) -> mx.array:
            return weights[f"{prefix}.{name}.weight"].astype(dtype)

        def matrix(*names: str) -> _Matrix:
            return _Matrix(weights, [f"{prefix}.{name}" for name in names], files)

        self.tap_norm = norm("dflare_norm", mx.float32)
        self.input_norm = norm("input_layernorm")
        self.mlp_norm = norm("post_attention_layernorm")
        self.query_norm = norm("self_attn.q_norm")
        self.key_norm = norm("self_attn.k_norm")
        self.context_key_norm = norm("self_attn.k_norm_ctx")
        self.qkv = matrix("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        self.context_kv = matrix("self_attn.k_proj_ctx", "self_attn.v_proj_ctx")
        self.output = matrix("self_attn.o_proj")
        self.gate_up = matrix("mlp.gate_proj", "mlp.up_proj")
        self.down = matrix("mlp.down_proj")


class Inkling2Drafter:
    """Drafter bound to one target model."""

    def __init__(
        self,
        files: Inkling2DrafterCheckpoint,
        target: Inkling2LoopForConditionalGeneration,
    ):
        config = files.config
        self.target = target
        self.hidden_size = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.eps = float(config["rms_norm_eps"])
        self.rope_theta = float(config["rope_theta"])
        self.local_layers = frozenset(int(i) for i in config["local_layer_ids"])
        self.window = int(config["sliding_window_size"])
        self.block_size = int(config["block_size"])
        self.max_draft_tokens = self.block_size - 1
        self.taps = tuple(int(visit) for visit in config["target_layer_ids"])
        if self.hidden_size != target.config.hidden_size:
            raise ValueError("drafter and target hidden sizes differ")
        if max(self.taps) > target.num_layers:
            raise ValueError("drafter taps a visit the target does not have")
        weights = dict(mx.load(str(files.weights)))
        self.fuse = mx.softmax(weights["fuse_weight"].astype(mx.float32), axis=-1)
        self.mask_embed = weights["mask_embed"].astype(mx.bfloat16)
        self.final_norm = weights["norm.weight"].astype(mx.float32)
        self.layers = [
            _DrafterLayer(weights, index, files)
            for index in range(int(config["num_hidden_layers"]))
        ]
        self.summary_norm = weights["markov_head.dflare_norm.weight"].astype(mx.float32)
        self.token_rows = _Matrix(weights, ["markov_head.markov_w1"], files)
        self.rnn = _Matrix(weights, ["markov_head.rnn_proj"], files)
        self.rnn_bias = weights["markov_head.rnn_proj.bias"].astype(mx.float32)
        self.hidden_out = _Matrix(weights, ["markov_head.hidden_out"], files)
        self.hidden_out_bias = weights["markov_head.hidden_out.bias"].astype(mx.float32)
        self.rank = int(config["markov_rank"])
        self.shortlist = target.head.shortlist(_SHORTLIST_BUCKETS, _SHORTLIST_BITS)
        mx.eval(
            list(weights.values()),
            self.fuse,
            [(layer.qkv.weight, layer.context_kv.weight) for layer in self.layers],
        )

    def new_context(self) -> DraftContext:
        """Return the context of a request with no committed tokens."""

        return DraftContext(layers=[_LayerContext() for _ in self.layers])

    def read(self, context: DraftContext, taps: mx.array) -> None:
        """Append committed target states to a request's context.

        Args:
            context: The request's context.
            taps: ``(tokens, taps, hidden)`` target states.
        """

        tokens = int(taps.shape[0])
        taps = taps.astype(mx.float32)
        for index, (layer, state) in enumerate(
            zip(self.layers, context.layers, strict=True)
        ):
            mixed = mx.fast.rms_norm(
                (taps * self.fuse[index][None, :, None]).sum(axis=1),
                layer.tap_norm,
                self.eps,
            )
            kv = layer.context_kv(mixed).reshape(
                1, tokens, 2 * self.kv_heads, self.head_dim
            )
            keys = mx.fast.rms_norm(
                kv[:, :, : self.kv_heads], layer.context_key_norm, self.eps
            )
            keys = self._rope(keys.transpose(0, 2, 1, 3), context.length)
            values = kv[:, :, self.kv_heads :].transpose(0, 2, 1, 3)
            self._append(
                state,
                keys.astype(mx.float16),
                values.astype(mx.float16),
                context.length,
                local=index in self.local_layers,
            )
        context.length += tokens
        context.summary = mx.fast.rms_norm(
            (taps[-1] * self.fuse[len(self.layers)][:, None]).sum(axis=0),
            self.summary_norm,
            self.eps,
        )

    def propose(
        self, context: DraftContext, token: mx.array, count: int
    ) -> tuple[mx.array, mx.array]:
        """Propose the tokens that follow ``token``.

        Args:
            context: Context of every committed token before ``token``.
            token: ``(1,)`` last sampled token.
            count: Tokens to propose, below the block size.

        Returns:
            ``(count,)`` int32 proposed tokens and the float32 probability of
            each among the shortlisted tokens.
        """

        hidden = self._block(context, token)
        summary = context.summary
        assert summary is not None
        state = mx.zeros((self.rank,), dtype=mx.float32)
        previous = token
        proposed = []
        confidences = []
        for step in range(1, count + 1):
            inputs = mx.concatenate(
                [state, self.token_rows.rows(previous)[0], hidden[step], summary]
            )
            gate, candidate, output = mx.split(
                self.rnn(inputs[None])[0].astype(mx.float32) + self.rnn_bias, 3
            )
            gate = mx.sigmoid(gate)
            state = gate * state + (1.0 - gate) * mx.tanh(candidate)
            delta = self.hidden_out(mx.tanh(output)[None])[0].astype(mx.float32)
            delta = delta + self.hidden_out_bias
            previous, confidence = self.shortlist(hidden[step] + delta)
            proposed.append(previous)
            confidences.append(confidence)
        return mx.concatenate(proposed), mx.concatenate(confidences)

    def _block(self, context: DraftContext, token: mx.array) -> mx.array:
        """Return the hidden states of one proposed block."""

        size = self.block_size
        masks = mx.broadcast_to(self.mask_embed[None], (size - 1, self.hidden_size))
        hidden = mx.concatenate(
            [self.target.embedding(token).astype(mx.bfloat16), masks]
        )
        query_dim = self.heads * self.head_dim
        for layer, state in zip(self.layers, context.layers, strict=True):
            qkv = layer.qkv(mx.fast.rms_norm(hidden, layer.input_norm, self.eps))
            queries = qkv[:, :query_dim].reshape(1, size, self.heads, self.head_dim)
            kv = qkv[:, query_dim:].reshape(1, size, 2 * self.kv_heads, self.head_dim)
            queries = mx.fast.rms_norm(queries, layer.query_norm, self.eps)
            keys = mx.fast.rms_norm(kv[:, :, : self.kv_heads], layer.key_norm, self.eps)
            queries = self._rope(queries.transpose(0, 2, 1, 3), context.length)
            keys = self._rope(keys.transpose(0, 2, 1, 3), context.length)
            values = kv[:, :, self.kv_heads :].transpose(0, 2, 1, 3)
            used, end = state.used, state.used + size
            assert state.keys is not None and state.values is not None
            state.keys[:, :, used:end] = keys.astype(mx.float16)
            state.values[:, :, used:end] = values.astype(mx.float16)
            attention = mx.fast.scaled_dot_product_attention(
                queries.astype(mx.float16),
                state.keys[:, :, :end],
                state.values[:, :, :end],
                scale=self.head_dim**-0.5,
            )
            attention = attention.transpose(0, 2, 1, 3).reshape(size, query_dim)
            hidden = hidden + layer.output(attention)
            gate, up = mx.split(
                layer.gate_up(mx.fast.rms_norm(hidden, layer.mlp_norm, self.eps)),
                2,
                axis=-1,
            )
            hidden = hidden + layer.down(nn.silu(gate) * up)
        return mx.fast.rms_norm(hidden.astype(mx.float32), self.final_norm, self.eps)

    def _append(
        self,
        state: _LayerContext,
        keys: mx.array,
        values: mx.array,
        length: int,
        *,
        local: bool,
    ) -> None:
        """Append keys and values, overwriting the oldest of a local layer.

        Positions are already encoded in the keys, so a local layer keeps its
        window as a ring.
        """

        tokens = int(keys.shape[2])
        if local and tokens > self.window:
            keys, values = keys[:, :, -self.window :], values[:, :, -self.window :]
            length += tokens - self.window
            tokens = self.window
        limit = self.window if local else length + tokens
        self._reserve(state, min(limit, length + tokens) + self.block_size)
        start = length % limit if local else length
        first = min(tokens, limit - start)
        for name, value in (("keys", keys), ("values", values)):
            buffer = getattr(state, name)
            buffer[:, :, start : start + first] = value[:, :, :first]
            if first < tokens:
                buffer[:, :, : tokens - first] = value[:, :, first:]
        state.used = min(limit, length + tokens)

    def _reserve(self, state: _LayerContext, tokens: int) -> None:
        held = 0 if state.keys is None else int(state.keys.shape[2])
        if tokens <= held:
            return
        capacity = -(-tokens // _GROW_TOKENS) * _GROW_TOKENS
        shape = (1, self.kv_heads, capacity - held, self.head_dim)
        for name in ("keys", "values"):
            buffer = getattr(state, name)
            grown = mx.zeros(shape, dtype=mx.float16)
            if buffer is not None:
                grown = mx.concatenate([buffer, grown], axis=2)
            setattr(state, name, grown)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )
