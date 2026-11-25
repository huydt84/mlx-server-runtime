"""Tiny random Inkling-2 pack and drafter with the published file layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_worker.native_mlx.packed_matmul import FORMAT_BITS, FORMAT_IDS

from .packed_weights import random_codes, random_tables

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, REL_DIM = 64, 4, 2, 64, 16
LAYERS, EXPERTS, TOP, INTERMEDIATE = 6, 8, 2, 64
VOCAB, OUTPUT, GATE_RANK = 256, 228, 8
WINDOW_LAYERS = (0, 2, 4)
LOOP = range(2, 4)
PASSES = 3
WINDOW = 16

_LINEARS = {
    "self_attn.q_proj.linear": (HEADS * HEAD_DIM, HIDDEN),
    "self_attn.k_proj.linear": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj.linear": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.relative_position_bias.linear": (HEADS * REL_DIM, HIDDEN),
    "self_attn.o_proj.gate_proj": (HEADS * HEAD_DIM, HIDDEN),
    "self_attn.o_proj.linear": (HIDDEN, HEADS * HEAD_DIM),
}


def pack_config() -> dict[str, Any]:
    """Return the ``config.json`` of the tiny checkpoint."""

    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "text_config": {
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "vocab_size": VOCAB,
            "unpadded_vocab_size": OUTPUT,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "d_rel": REL_DIM,
            "rel_extent": 32,
            "log_scaling_n_floor": 24,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 8.0,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": list(WINDOW_LAYERS),
            "sliding_window_size": WINDOW,
            "sconv_kernel_size": 4,
            "final_logit_softcapping": 30.0,
            "n_layers_per_attn_res_block": 2,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": TOP,
            "n_shared_experts": 0,
            "intermediate_size": INTERMEDIATE,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 2.0,
            "loop_config": {
                "core_start": LOOP.start,
                "core_end": LOOP.stop,
                "num_passes": PASSES,
                "gate_rank": GATE_RANK,
                "is_varloop": False,
            },
        },
    }


def write_pack(path: Path, seed: int = 0) -> None:
    """Write a checkpoint that mixes packed formats and both expert map kinds."""

    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    _json(path / "config.json", pack_config())
    _json(
        path / "manifest.json",
        {
            "format": "inkling2-packed-v1-sqyfr",
            "subformat": "rg128-f16signs-v3",
            "plan_total_gb": 5.4,
        },
    )
    tables = {name: table * 0.5 for name, table in random_tables(rng).items()}
    np.savez(path / "luts.npz", **tables)
    for layer in range(LAYERS):
        _write_layer(path, rng, layer)
    _write_buckets(path, rng, "embed", [("lm6", 128), ("rtr4", 128)], VOCAB)
    _write_buckets(path, rng, "head", [("lm8", 128), ("rtr2", OUTPUT - 128)], OUTPUT)
    mx.save_safetensors(str(path / "steering.safetensors"), _steering(rng))


def write_drafter(path: Path, seed: int = 1) -> None:
    """Write a one-global, one-local layer drafter for the tiny pack."""

    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    visits = LAYERS + len(LOOP) * (PASSES - 1)
    config = pack_config()
    config["mtp_config"] = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "local_layer_ids": [1],
        "sliding_window_size": WINDOW,
        "block_size": 4,
        "markov_rank": 64,
        "target_layer_ids": [1, visits],
    }
    _json(path / "config.json", config)
    _json(
        path / "manifest.json",
        {
            "format": "inkling2-dspark-4bit-v1",
            "weights": "mtp_4bit.safetensors",
            "bits": 4,
            "group_size": 64,
        },
    )
    tensors: dict[str, mx.array] = {
        "fuse_weight": _normal(rng, 3, 2),
        "mask_embed": _normal(rng, HIDDEN),
        "norm.weight": mx.ones((HIDDEN,)),
        "markov_head.dflare_norm.weight": mx.ones((HIDDEN,)),
        "markov_head.hidden_out.bias": _normal(rng, HIDDEN),
        "markov_head.rnn_proj.bias": _normal(rng, 3 * 64),
    }
    shapes = {
        "markov_head.markov_w1": (VOCAB, 64),
        "markov_head.rnn_proj": (3 * 64, 4 * 64),
        "markov_head.hidden_out": (HIDDEN, 64),
    }
    for layer in range(2):
        prefix = f"layers.{layer}"
        for name in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            tensors[f"{prefix}.{name}.weight"] = mx.ones((HIDDEN,))
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            tensors[f"{prefix}.self_attn.{name}.weight"] = mx.ones((64,))
        shapes[f"{prefix}.self_attn.q_proj"] = (128, HIDDEN)
        for name in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"):
            shapes[f"{prefix}.self_attn.{name}"] = (64, HIDDEN)
        shapes[f"{prefix}.self_attn.o_proj"] = (HIDDEN, 128)
        shapes[f"{prefix}.mlp.gate_proj"] = (128, HIDDEN)
        shapes[f"{prefix}.mlp.up_proj"] = (128, HIDDEN)
        shapes[f"{prefix}.mlp.down_proj"] = (HIDDEN, 128)
    for name, shape in shapes.items():
        weight, scales, biases = mx.quantize(
            _normal(rng, *shape) * 0.1, group_size=64, bits=4
        )
        tensors[f"{name}.weight"] = weight
        tensors[f"{name}.weight.scales"] = scales
        tensors[f"{name}.weight.biases"] = biases
    mx.save_safetensors(str(path / "mtp_4bit.safetensors"), tensors)


def _write_layer(path: Path, rng: np.random.Generator, layer: int) -> None:
    arrays: dict[str, np.ndarray] = {}
    linears: dict[str, Any] = {}
    formats = ("rtr4", "ht3", "lm6", "rtr2", "ht1", "rtr3")
    for index, (name, (rows, columns)) in enumerate(_LINEARS.items()):
        fmt = formats[(index + layer) % len(formats)]
        safe = name.replace(".", "_")
        rotates_out = layer == 1 and name.endswith("o_proj.linear")
        linears[name] = {
            "fmt": fmt,
            "k": FORMAT_BITS[fmt],
            "safe": safe,
            "out_rot": rotates_out,
        }
        parts = _matrix(rng, fmt, rows, columns)
        if rotates_out:
            parts["signs_out"] = _signs(rng, rows)
        arrays.update({f"lin_{safe}_{part}": value for part, value in parts.items()})
    experts: dict[str, dict[str, int]] = {}
    for side, rows, columns in (
        ("gu", 2 * INTERMEDIATE, HIDDEN),
        ("dn", HIDDEN, INTERMEDIATE),
    ):
        per_group = layer % 2 == 0
        blocks = 2 * EXPERTS if per_group else EXPERTS
        block_rows = rows // 2 if per_group else rows
        mixed = ["ht2", "nvfp4"] if layer == 5 else ["ht2", "rtr1"]
        choice = [mixed[block % 3 % 2] for block in range(blocks)]
        table = []
        for block, fmt in enumerate(choice):
            table.append([FORMAT_IDS[fmt], choice[:block].count(fmt)])
        for fmt in mixed:
            count = choice.count(fmt)
            arrays[f"{side}_{fmt}_codes"] = np.stack(
                [random_codes(rng, fmt, block_rows, columns) for _ in range(count)]
            )
            if fmt == "nvfp4":
                shape = (count, block_rows, columns // 16)
                arrays[f"{side}_{fmt}_scales"] = rng.integers(
                    16, 32, size=shape, dtype=np.uint8
                )
                arrays[f"{side}_{fmt}_outer"] = rng.uniform(
                    1.0, 2.0, size=count
                ).astype(np.float32)
            else:
                arrays[f"{side}_{fmt}_scales"] = _scales(rng, count, block_rows)
        arrays[f"{side}_signs_in"] = _signs(rng, columns)
        name = f"{side}_rg_map" if per_group else f"{side}_map"
        shape = (EXPERTS, 2, 2) if per_group else (EXPERTS, 2)
        arrays[name] = np.array(table, dtype=np.int32).reshape(shape)
        experts[side] = {fmt: choice.count(fmt) for fmt in mixed}
    np.savez(path / f"layer_{layer:02d}_packed.npz", **arrays)
    _json(
        path / f"layer_{layer:02d}_meta.json",
        {"layer": layer, "linears": linears, "experts": experts},
    )


def _write_buckets(
    path: Path,
    rng: np.random.Generator,
    part: str,
    buckets: list[tuple[str, int]],
    tokens: int,
) -> None:
    arrays: dict[str, np.ndarray] = {"order": rng.permutation(tokens).astype(np.int32)}
    meta = []
    for index, (fmt, rows) in enumerate(buckets):
        meta.append({"kb": index, "opt": fmt, "rows": rows, "k": FORMAT_BITS[fmt]})
        for name, value in _matrix(rng, fmt, rows, HIDDEN).items():
            arrays[f"b{index}_{name}"] = value
    np.savez(path / f"{part}_packed.npz", **arrays)
    _json(path / f"{part}_meta.json", meta)


def _steering(rng: np.random.Generator) -> dict[str, mx.array]:
    tensors = {
        "model.final_input.query": _normal(rng, HIDDEN),
        "model.message_input.query": _normal(rng, len(LOOP), 4, HIDDEN),
        "model.embed_tokens.embed_norm.weight": mx.ones((HIDDEN,)),
        "model.norm.weight": mx.ones((HIDDEN,)),
        "model.visual.patch.weight": _normal(rng, 2, 2),
    }
    kv = KV_HEADS * HEAD_DIM
    for layer in range(LAYERS):
        sliding = layer in WINDOW_LAYERS
        prefix = f"model.layers.{layer}"
        values = {
            "mlp.gate.weight": _normal(rng, EXPERTS, HIDDEN),
            "mlp.gate.e_score_correction_bias": _normal(rng, EXPERTS) * 0.05,
            "mlp_input.query": _normal(rng, HIDDEN),
            "post_attention_convolution.conv1d.weight": _normal(rng, HIDDEN, 1, 4),
            "post_mlp_convolution.conv1d.weight": _normal(rng, HIDDEN, 1, 4),
            "post_attention_layernorm.weight": mx.ones((HIDDEN,)),
            "self_attn.q_proj.norm.scale": mx.ones((HEADS, HEAD_DIM)),
            "self_attn.relative_position_bias.norm.scale": mx.ones((HEADS, REL_DIM)),
            "self_attn.relative_position_bias.proj": _normal(rng, REL_DIM, 32),
            "self_attn.sink_proj.weight": _normal(rng, HEADS, HIDDEN) * 0.1,
        }
        if layer:
            shape = (HIDDEN,) if sliding else (3, HIDDEN)
            values["attention_input.query"] = _normal(rng, *shape)
        for kind in ("k", "v"):
            name = f"self_attn.{kind}_proj.sconv"
            if sliding:
                values[f"{name}.conv1d.weight"] = _normal(rng, kv, 1, 4)
            else:
                values[f"{name}.static_weight"] = _normal(rng, 4, kv)
                values[f"{name}.weight_proj.weight"] = (
                    _normal(rng, KV_HEADS * 4, HIDDEN) * 0.1
                )
        if sliding:
            norms = ("input_layernorm",)
        else:
            norms = ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        values.update({f"{name}.weight": mx.ones((HIDDEN,)) for name in norms})
        if layer in LOOP:
            for gate in ("q", "k", "v", "mlp"):
                name = f"{gate}_message_gate"
                values[f"{name}.down_proj.weight"] = _normal(rng, GATE_RANK, 2 * HIDDEN)
                for half in ("forget", "message"):
                    values[f"{name}.{half}_gate_proj.weight"] = (
                        _normal(rng, HIDDEN, GATE_RANK) * 0.1
                    )
        tensors.update({f"{prefix}.{name}": value for name, value in values.items()})
    return tensors


def _matrix(
    rng: np.random.Generator, fmt: str, rows: int, columns: int
) -> dict[str, np.ndarray]:
    parts = {
        "codes": random_codes(rng, fmt, rows, columns),
        "scales": _scales(rng, rows),
        "signs_in": _signs(rng, columns),
    }
    if fmt.startswith("lm"):
        parts["levels"] = rng.normal(size=1 << FORMAT_BITS[fmt]).astype(np.float32)
    return parts


def _scales(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.uniform(0.1, 0.3, size=shape).astype(np.float16)


def _signs(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=size).astype(np.float16)


def _normal(rng: np.random.Generator, *shape: int) -> mx.array:
    return mx.array(rng.normal(size=shape).astype(np.float32))


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))
