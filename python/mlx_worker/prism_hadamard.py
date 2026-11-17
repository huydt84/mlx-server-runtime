"""Prism ML Hadamard-rotated MLX packs (``model_type = prism_hadamard_qwen35``).

These packs (for example ``prism-ml/Ternary-Bonsai-2-27B-mlx-2bit``) store a
Qwen3.5 checkpoint whose projection weights were multiplied by a blockwise
Walsh-Hadamard rotation before affine quantization. The rotation must be
applied to activations at run time:

- a rotated linear layer multiplies its input by per-column ``signs`` and
  applies a normalized Hadamard transform over blocks of ``block`` columns
  before the quantized matmul;
- the rotated embedding table returns rows in the rotated basis, so each
  looked-up row is dequantized and rotated back with the inverse transform.

An ordinary affine loader skips both transforms and produces wrong output
without an error, so the model type is deliberately not ``qwen3_5``. This
module mirrors the runtime bundled with the pack (``runtime/runtime.py``) and
is shared by the native backend and the ``mlx-lm`` (v1) backend.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

PRISM_HADAMARD_MODEL_TYPE = "prism_hadamard_qwen35"
"""``config.json`` model type of a Hadamard-rotated Qwen3.5 pack."""

PRISM_HADAMARD_BASE_MODEL_TYPE = "qwen3_5"

_SUPPORTED_SCHEMA_VERSIONS = frozenset((1, 2))
_SUPPORTED_BLOCK_SIZES = frozenset((512, 1024, 2048, 4096))


@dataclass(frozen=True)
class HadamardModuleRecord:
    """One packed module listed in the pack's ``config.json``.

    Attributes:
        path: Module path inside the language model, for example
            ``model.layers.0.mlp.up_proj``.
        block: Hadamard block size, or ``0`` for an unrotated packed module.
        embedding: Whether the module is the token embedding table.
    """

    path: str
    block: int
    embedding: bool


@dataclass(frozen=True)
class HadamardPackConfig:
    """Validated packed-module manifest and its affine quantization."""

    records: tuple[HadamardModuleRecord, ...]
    group_size: int
    bits: int


def is_prism_hadamard_config(payload: Mapping[str, Any]) -> bool:
    """Return whether a ``config.json`` payload declares a Hadamard pack."""

    return payload.get("model_type") == PRISM_HADAMARD_MODEL_TYPE


def is_prism_hadamard_checkpoint(model_path: Path) -> bool:
    """Return whether a local checkpoint directory holds a Hadamard pack."""

    config_path = Path(model_path) / "config.json"
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and is_prism_hadamard_config(payload)


def parse_hadamard_pack(payload: Mapping[str, Any]) -> HadamardPackConfig:
    """Validate a Hadamard pack's module manifest.

    Args:
        payload: The pack's ``config.json``.

    Returns:
        The packed modules and their affine quantization settings.

    Raises:
        ValueError: If the pack uses a schema, transform, layout, or
            quantization this runtime does not implement.
    """

    if not is_prism_hadamard_config(payload):
        raise ValueError(f"expected model_type={PRISM_HADAMARD_MODEL_TYPE!r}")
    if payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported Hadamard pack schema_version")
    base = payload.get("base_model_type", PRISM_HADAMARD_BASE_MODEL_TYPE)
    if base != PRISM_HADAMARD_BASE_MODEL_TYPE:
        raise ValueError(f"unsupported Hadamard pack base_model_type {base!r}")
    layout = payload.get("gdn_activation_layout", "grouped")
    if layout != "grouped":
        raise ValueError(f"unsupported Gated DeltaNet activation layout {layout!r}")
    quantization = payload.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("Hadamard pack is missing quantization settings")
    if quantization.get("mode", "affine") != "affine":
        raise ValueError("Hadamard pack quantization must be affine")
    group_size = int(quantization.get("group_size", 0))
    bits = int(quantization.get("bits", 0))
    if group_size <= 0 or bits not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("invalid Hadamard pack quantization settings")
    modules = payload.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Hadamard pack is missing its module manifest")
    records: list[HadamardModuleRecord] = []
    seen: set[str] = set()
    for module in modules:
        if not isinstance(module, Mapping):
            raise ValueError("Hadamard pack module records must be objects")
        path = str(module.get("path", ""))
        if not path or path in seen:
            raise ValueError(f"invalid or duplicate packed module path {path!r}")
        seen.add(path)
        if module.get("dtype") != "float16":
            raise ValueError(f"unsupported packed module dtype for {path}")
        block = int(module.get("block", 0))
        if block and block not in _SUPPORTED_BLOCK_SIZES:
            raise ValueError(f"unsupported Hadamard block size {block} for {path}")
        records.append(
            HadamardModuleRecord(
                path=path,
                block=block,
                embedding=bool(module.get("embedding", False)),
            )
        )
    return HadamardPackConfig(
        records=tuple(records),
        group_size=group_size,
        bits=bits,
    )


def hadamard_rotate(
    x: mx.array,
    block: int,
    signs: mx.array,
    *,
    inverse: bool = False,
) -> mx.array:
    """Apply the pack's blockwise sign-and-Hadamard rotation to the last axis.

    The forward rotation multiplies by ``signs`` and then applies a normalized
    Hadamard transform over each block; the inverse applies the transform and
    then the signs. The transform runs in float32 and the result keeps the
    input dtype.
    """

    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide the activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class HadamardQuantizedLinear(nn.Module):
    """Affine-quantized linear layer stored in a Hadamard-rotated basis."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        block: int,
        group_size: int,
        bits: int,
    ) -> None:
        super().__init__()
        if input_dims % group_size or (input_dims * bits) % 32:
            raise ValueError("packed linear width is not group- and word-aligned")
        if block and input_dims % block:
            raise ValueError("Hadamard block does not divide the packed width")
        self.block = block
        self.group_size = group_size
        self.bits = bits
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size), mx.float16)
        self.biases = mx.zeros((output_dims, input_dims // group_size), mx.float16)
        if block:
            self.signs = mx.ones((input_dims,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = hadamard_rotate(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


class HadamardQuantizedEmbedding(nn.Module):
    """Affine-quantized embedding table stored in a Hadamard-rotated basis.

    Looked-up rows are dequantized to float16, the pack's activation dtype,
    and rotated back with the inverse transform.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        *,
        block: int,
        group_size: int,
        bits: int,
    ) -> None:
        super().__init__()
        if dims % group_size or (dims * bits) % 32:
            raise ValueError("packed embedding width is not group- and word-aligned")
        if block and dims % block:
            raise ValueError("Hadamard block does not divide the embedding width")
        self.block = block
        self.group_size = group_size
        self.bits = bits
        self.weight = mx.zeros((num_embeddings, dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, dims // group_size), mx.float16)
        self.biases = mx.zeros((num_embeddings, dims // group_size), mx.float16)
        if block:
            self.signs = mx.ones((dims,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        indices = x.reshape(-1)
        rows = mx.dequantize(
            self.weight[indices],
            self.scales[indices],
            self.biases[indices],
            group_size=self.group_size,
            bits=self.bits,
        )
        rows = rows.reshape(*shape, -1).astype(mx.float16)
        if self.block:
            rows = hadamard_rotate(rows, self.block, self.signs, inverse=True)
        return rows


def install_hadamard_modules(model: nn.Module, pack: HadamardPackConfig) -> None:
    """Replace every manifest module of a language model with its packed form.

    Args:
        model: Language model whose module paths match the manifest, with
            ``nn.Linear`` and ``nn.Embedding`` placeholders at those paths.
        pack: Validated pack manifest.

    Raises:
        ValueError: If a manifest path is missing or names the wrong module
            kind.
    """

    for record in pack.records:
        parent, name = _resolve_parent(model, record.path)
        original = parent[int(name)] if name.isdigit() else getattr(parent, name, None)
        if record.embedding:
            if not isinstance(original, nn.Embedding):
                raise ValueError(f"packed embedding {record.path} is not an embedding")
            rows, width = (int(size) for size in original.weight.shape)
            replacement: nn.Module = HadamardQuantizedEmbedding(
                rows,
                width,
                block=record.block,
                group_size=pack.group_size,
                bits=pack.bits,
            )
        else:
            if not isinstance(original, nn.Linear) or "bias" in original:
                raise ValueError(
                    f"packed linear {record.path} is not a bias-free linear"
                )
            rows, width = (int(size) for size in original.weight.shape)
            replacement = HadamardQuantizedLinear(
                width,
                rows,
                block=record.block,
                group_size=pack.group_size,
                bits=pack.bits,
            )
        setattr(parent, name, replacement)


def load_mlx_lm_model(model_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load a Hadamard pack into ``mlx-lm``'s Qwen3.5 text model.

    This is the reference the pack's own runtime builds: ``mlx-lm``'s
    ``qwen3_5`` graph with packed modules swapped in. The vision tower is not
    loaded.

    Args:
        model_path: Local pack directory.

    Returns:
        The evaluated ``mlx-lm`` model and the pack's ``config.json``.
    """

    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    pack = parse_hadamard_pack(config)
    model = Model(
        ModelArgs.from_dict(
            {
                "model_type": PRISM_HADAMARD_BASE_MODEL_TYPE,
                "text_config": config["text_config"],
            }
        )
    )
    install_hadamard_modules(model.language_model, pack)
    weights: dict[str, mx.array] = {}
    for weights_file in sorted(model_path.glob("*.safetensors")):
        weights.update(
            (name, value)
            for name, value in mx.load(str(weights_file)).items()
            if name.startswith("language_model.") and ".mtp." not in name
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config


def load_mlx_lm(model_path: Path) -> tuple[nn.Module, Any]:
    """Load a Hadamard pack and its tokenizer the way ``mlx_lm.load`` would.

    End-of-generation ids come from ``generation_config.json`` when it lists
    them; otherwise the tokenizer's own EOS token is used.
    """

    from mlx_lm.utils import load_tokenizer

    model, _ = load_mlx_lm_model(model_path)
    eos = generation_eos_token_ids(model_path)
    tokenizer = load_tokenizer(Path(model_path), eos_token_ids=eos or None)
    return model, tokenizer


def generation_eos_token_ids(model_path: Path) -> list[int]:
    """Return the end-of-generation ids listed in ``generation_config.json``."""

    path = Path(model_path) / "generation_config.json"
    try:
        value = json.loads(path.read_text()).get("eos_token_id")
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [int(item) for item in values]


def _resolve_parent(model: nn.Module, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    parent: Any = model
    for part in parts[:-1]:
        try:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        except (AttributeError, IndexError, KeyError) as exc:
            raise ValueError(f"packed module path {path!r} does not exist") from exc
    return parent, parts[-1]


__all__ = [
    "PRISM_HADAMARD_MODEL_TYPE",
    "HadamardModuleRecord",
    "HadamardPackConfig",
    "HadamardQuantizedEmbedding",
    "HadamardQuantizedLinear",
    "generation_eos_token_ids",
    "hadamard_rotate",
    "install_hadamard_modules",
    "is_prism_hadamard_checkpoint",
    "is_prism_hadamard_config",
    "load_mlx_lm",
    "load_mlx_lm_model",
    "parse_hadamard_pack",
]
