"""Qwen3.5 (Gated DeltaNet hybrid) and Hadamard-pack native MLX tests."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model as ReferenceModel
from mlx_lm.models.qwen3_5 import ModelArgs as ReferenceModelArgs

from mlx_worker.native_mlx.attention import (
    DenseReferenceAttentionBackend,
    HybridPagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.bootstrap import detect_native_architecture
from mlx_worker.native_mlx.cache import (
    DenseKVCacheBackend,
    HybridPagedKVCacheBackend,
    RecurrentStateRow,
    gather_recurrent_rows,
)
from mlx_worker.native_mlx.interfaces import ForwardBatch, ForwardMode
from mlx_worker.native_mlx.mapping import (
    WeightMappingBug,
    load_mapped_weights,
    load_weight_index,
)
from mlx_worker.native_mlx.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5WeightAdapter,
    build_qwen3_5_model,
    parse_qwen3_5_config,
)
from mlx_worker.native_mlx.registry import (
    architecture_for_model_type,
    get_architecture_spec,
)
from mlx_worker.prism_hadamard import (
    HadamardQuantizedEmbedding,
    HadamardQuantizedLinear,
    hadamard_rotate,
    load_mlx_lm_model,
    parse_hadamard_pack,
)

_TEXT_CONFIG: dict[str, object] = {
    "model_type": "qwen3_5_text",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attn_output_gate": True,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    },
}


def _payload(**text_overrides: object) -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {**_TEXT_CONFIG, **text_overrides},
    }


def _reference_and_weights() -> tuple[ReferenceModel, list[tuple[str, mx.array]]]:
    """Build an mlx-lm reference with random norms and return its text weights."""

    mx.random.seed(7)
    reference = ReferenceModel(
        ReferenceModelArgs.from_dict(
            {"model_type": "qwen3_5", "text_config": dict(_TEXT_CONFIG)}
        )
    )
    weights = []
    for name, value in tree_flatten(reference.parameters()):
        if name.endswith(("norm.weight", "layernorm.weight")):
            value = 1.0 + 0.1 * mx.random.normal(value.shape)
        weights.append((name, value))
    reference.load_weights(weights)
    mx.eval(reference.parameters())
    native = [(name.removeprefix("language_model."), value) for name, value in weights]
    return reference, native


def _native_model() -> tuple[ReferenceModel, Qwen3_5ForCausalLM]:
    reference, weights = _reference_and_weights()
    return reference, build_qwen3_5_model(parse_qwen3_5_config(_payload()), weights)


def _paged_step(
    model: Qwen3_5ForCausalLM,
    backend: HybridPagedKVCacheBackend,
    handles: list[str],
    rows: list[list[int]],
) -> mx.array:
    """Run one padded paged step and return each row's last-token logits."""

    caches = tuple(backend.get(handle, handle.split("-")[3]) for handle in handles)
    lengths = tuple(len(row) for row in rows)
    width = max(lengths)
    starts = [backend.length(handle) for handle in handles]
    input_ids = mx.array([row + [0] * (width - len(row)) for row in rows])
    positions = mx.array(
        [
            list(range(start, start + len(row)))
            + [start + len(row) - 1] * (width - len(row))
            for start, row in zip(starts, rows, strict=True)
        ],
        dtype=mx.int32,
    )
    mode = ForwardMode.DECODE if width == 1 else ForwardMode.PREFILL
    reservation = backend.reserve_batch(caches, lengths)
    logits = model(
        input_ids,
        positions,
        ForwardBatch(
            forward_mode=mode,
            token_lengths=lengths,
            cache_lengths=tuple(starts),
            attention_mask="causal",
            layer_attention=HybridPagedMetalAttentionBackend().contexts(
                reservation, mode
            ),
            logits_positions=tuple(length - 1 for length in lengths),
        ),
    )
    mx.eval(logits)
    reservation.commit()
    return logits[:, 0, :]


def _hybrid_backend(model: Qwen3_5ForCausalLM) -> HybridPagedKVCacheBackend:
    config = model.config
    return HybridPagedKVCacheBackend(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        page_size=8,
        budget_bytes=1 << 20,
        dtype=mx.float32,
        kv_layer_indices=config.full_attention_layers,
    )


def test_qwen3_5_is_registered_as_hybrid_without_prefix_cache() -> None:
    spec = get_architecture_spec("Qwen3_5ForConditionalGeneration")
    assert spec is not None
    assert spec.known_good_checkpoint == "prism-ml/Bonsai-27B-mlx-1bit"
    assert spec.cache_family == "hybrid"
    assert not spec.supports_prefix_cache
    assert architecture_for_model_type("prism_hadamard_qwen35") == (
        "Qwen3_5ForConditionalGeneration"
    )
    assert architecture_for_model_type("qwen3_5") is None


def test_qwen3_5_cache_geometry_holds_kv_for_attention_layers_only() -> None:
    spec = get_architecture_spec("Qwen3_5ForConditionalGeneration")
    assert spec is not None
    config = parse_qwen3_5_config(_payload())
    geometry = spec.cache_geometry(config)
    assert geometry.kv_layer_indices == (1, 3)
    assert geometry.head_dim == 32
    assert config.rotary_dim == 8


def test_parse_qwen3_5_rejects_unsupported_variants() -> None:
    with pytest.raises(ValueError, match="mixture-of-experts"):
        parse_qwen3_5_config(_payload(num_experts=8))
    with pytest.raises(ValueError, match="rope"):
        parse_qwen3_5_config(
            _payload(rope_parameters={"rope_type": "yarn", "rope_theta": 1.0})
        )
    with pytest.raises(ValueError, match="architectures"):
        parse_qwen3_5_config({**_payload(), "architectures": ["Qwen3ForCausalLM"]})


def test_weight_adapter_strips_language_prefix_and_skips_vision_and_mtp() -> None:
    from mlx_worker.native_mlx.mapping import WeightIndex

    index = WeightIndex(
        model_path=Path("."),
        weight_map={
            "language_model.model.layers.0.mlp.up_proj.weight": "a",
            "language_model.lm_head.weight": "a",
            "model.language_model.norm.weight": "a",
            "vision_tower.blocks.0.attn.qkv.weight": "a",
            "mtp.layers.0.mlp.up_proj.weight": "a",
            "language_model.mtp.fc.weight": "a",
        },
        source_files=("a",),
    )
    names = {
        entry.source_name: entry.canonical_name
        for entry in Qwen3_5WeightAdapter().build_plan(index).entries
    }
    assert names == {
        "language_model.model.layers.0.mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
        "language_model.lm_head.weight": "lm_head.weight",
        "model.language_model.norm.weight": "model.norm.weight",
    }
    with pytest.raises(WeightMappingBug):
        Qwen3_5WeightAdapter().build_plan(
            WeightIndex(Path("."), {"language_model.unknown.weight": "a"}, ("a",))
        )


def test_native_qwen3_5_matches_mlx_lm_forward() -> None:
    reference, model = _native_model()
    tokens = [3, 17, 42, 5, 9, 33, 1]
    expected = reference(mx.array([tokens]))
    backend = DenseKVCacheBackend(model.num_layers)
    cache = backend.get(backend.create("ref"), "ref")
    reservation = backend.reserve_batch((cache,), (len(tokens),))
    actual = model(
        mx.array([tokens]),
        mx.array([list(range(len(tokens)))], dtype=mx.int32),
        ForwardBatch(
            forward_mode=ForwardMode.PREFILL,
            token_lengths=(len(tokens),),
            cache_lengths=(0,),
            attention_mask="causal",
            layer_attention=DenseReferenceAttentionBackend().contexts(
                reservation, ForwardMode.PREFILL
            ),
        ),
    )
    reservation.commit()
    assert model.resolved_config.kv_cache_dtype == "float32"
    assert mx.allclose(actual, expected, atol=1e-4, rtol=1e-4).item()
    assert cache.layers[0].recurrent_state is not None
    assert cache.layers[0].conv_state is not None
    assert cache.layers[1].recurrent_state is None


def test_paged_padded_batch_keeps_each_request_state_isolated() -> None:
    reference, model = _native_model()
    prompts = [[3, 17, 42, 5, 9, 33, 1, 8, 2, 11], [7, 7, 21]]

    batched = _hybrid_backend(model)
    handles = [batched.create(f"r{index}") for index in range(2)]
    logits = [_paged_step(model, batched, handles, prompts)]
    next_tokens = [int(token) for token in mx.argmax(logits[0], axis=-1).tolist()]
    for _ in range(3):
        logits.append(_paged_step(model, batched, handles, [[t] for t in next_tokens]))
        next_tokens = [int(t) for t in mx.argmax(logits[-1], axis=-1).tolist()]

    for row, prompt in enumerate(prompts):
        cache = reference.make_cache()
        sequence = list(prompt)
        out = reference(mx.array([sequence]), cache=cache)[:, -1, :]
        assert mx.allclose(logits[0][row], out[0], atol=1e-4, rtol=1e-4).item()
        for step in range(3):
            token = int(mx.argmax(out[0]).item())
            out = reference(mx.array([[token]]), cache=cache)[:, -1, :]
            assert mx.allclose(
                logits[step + 1][row], out[0], atol=1e-4, rtol=1e-4
            ).item()
    assert batched.length(handles[0]) == len(prompts[0]) + 3
    assert batched.length(handles[1]) == len(prompts[1]) + 3


def test_recurrent_state_batch_is_reused_only_for_the_same_request_order() -> None:
    batch = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    rows = [RecurrentStateRow(batch, 0), RecurrentStateRow(batch, 1)]
    assert gather_recurrent_rows(rows, (3,), mx.float32) is batch
    swapped = gather_recurrent_rows(rows[::-1], (3,), mx.float32)
    assert swapped is not batch
    assert swapped.tolist() == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    fresh = gather_recurrent_rows([rows[1], None], (3,), mx.float32)
    assert fresh.tolist() == [[3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]


def test_hybrid_backend_sizes_pages_for_kv_layers_only() -> None:
    full = HybridPagedKVCacheBackend(
        num_layers=4,
        num_kv_heads=1,
        head_dim=32,
        page_size=8,
        budget_bytes=1 << 20,
        dtype=mx.float32,
    )
    subset = HybridPagedKVCacheBackend(
        num_layers=4,
        num_kv_heads=1,
        head_dim=32,
        page_size=8,
        budget_bytes=1 << 20,
        dtype=mx.float32,
        kv_layer_indices=(1, 3),
    )
    assert subset.bytes_per_page * 2 == full.bytes_per_page
    assert subset.num_pages == 2 * full.num_pages
    assert subset.key_pages[0].shape[0] == 0
    assert subset.key_pages[1].shape[0] == subset.num_pages
    with pytest.raises(ValueError, match="layer indices"):
        HybridPagedKVCacheBackend(
            num_layers=2,
            num_kv_heads=1,
            head_dim=32,
            page_size=8,
            budget_bytes=1 << 20,
            kv_layer_indices=(5,),
        )


def _rotation_matrix(width: int, block: int, signs: mx.array) -> mx.array:
    return hadamard_rotate(mx.eye(width), block, signs)


def test_hadamard_rotation_is_orthogonal_and_inverted() -> None:
    mx.random.seed(3)
    signs = mx.where(mx.random.uniform(shape=(1024,)) < 0.5, -1.0, 1.0)
    x = mx.random.normal((2, 3, 1024))
    rotated = hadamard_rotate(x, 512, signs)
    assert mx.allclose(
        mx.linalg.norm(rotated, axis=-1), mx.linalg.norm(x, axis=-1), rtol=1e-4
    ).item()
    restored = hadamard_rotate(rotated, 512, signs, inverse=True)
    assert mx.allclose(restored, x, atol=1e-4).item()


def test_hadamard_linear_and_embedding_undo_folded_rotation() -> None:
    mx.random.seed(5)
    width, rows, block = 1024, 8, 512
    signs = mx.where(mx.random.uniform(shape=(width,)) < 0.5, -1.0, 1.0)
    rotation = _rotation_matrix(width, block, signs)
    dense = mx.random.normal((rows, width)) * 0.05
    folded = (dense @ rotation).astype(mx.float16)
    linear = HadamardQuantizedLinear(width, rows, block=block, group_size=128, bits=8)
    linear.weight, linear.scales, linear.biases = mx.quantize(folded, 128, 8)
    linear.signs = signs
    x = mx.random.normal((2, width)).astype(mx.float16)
    expected = x.astype(mx.float32) @ dense.T
    assert mx.allclose(linear(x).astype(mx.float32), expected, atol=5e-2).item()
    rotated_rows = hadamard_rotate(dense, block, signs).astype(mx.float16)
    embedding = HadamardQuantizedEmbedding(
        rows, width, block=block, group_size=128, bits=8
    )
    embedding.weight, embedding.scales, embedding.biases = mx.quantize(
        rotated_rows, 128, 8
    )
    embedding.signs = signs
    looked_up = embedding(mx.array([[2, 5]]))
    assert looked_up.dtype == mx.float16
    assert mx.allclose(
        looked_up.astype(mx.float32), dense[mx.array([[2, 5]])], atol=2e-3
    ).item()


def _hadamard_payload() -> dict[str, object]:
    modules = [{"path": "model.embed_tokens", "block": 512, "embedding": True}]
    modules.append({"path": "lm_head", "block": 512, "embedding": False})
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        for name in ("mlp.gate_proj", "mlp.up_proj"):
            modules.append(
                {"path": f"{prefix}.{name}", "block": 512, "embedding": False}
            )
        modules.append(
            {"path": f"{prefix}.mlp.down_proj", "block": 1024, "embedding": False}
        )
        attention = (
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
            if layer == 0
            else ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        )
        for name in attention:
            modules.append(
                {"path": f"{prefix}.{name}", "block": 512, "embedding": False}
            )
    return {
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "gdn_activation_layout": "grouped",
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "text_config": {
            **_TEXT_CONFIG,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "vocab_size": 32,
        },
        "modules": [{**module, "dtype": "float16"} for module in modules],
    }


def _write_hadamard_pack(directory: Path) -> dict[str, object]:
    """Write a tiny Hadamard pack with float32 auxiliary tensors."""

    payload = _hadamard_payload()
    pack = parse_hadamard_pack(payload)
    reference = ReferenceModel(
        ReferenceModelArgs.from_dict(
            {"model_type": "qwen3_5", "text_config": payload["text_config"]}
        )
    )
    from mlx_worker.prism_hadamard import install_hadamard_modules

    install_hadamard_modules(reference.language_model, pack)
    mx.random.seed(11)
    weights: dict[str, mx.array] = {}
    for name, value in tree_flatten(reference.parameters()):
        if name.endswith(".signs"):
            value = mx.where(mx.random.uniform(shape=value.shape) < 0.5, -1.0, 1.0)
        elif name.endswith(".weight") and value.dtype == mx.uint32:
            value = mx.random.randint(0, 2**31, value.shape).astype(mx.uint32)
        elif name.endswith(".scales"):
            value = (0.02 * mx.random.uniform(shape=value.shape)).astype(mx.float16)
        elif name.endswith(".biases"):
            value = (-0.02 * mx.random.uniform(shape=value.shape)).astype(mx.float16)
        elif name.endswith(("norm.weight", "layernorm.weight")):
            value = 1.0 + 0.1 * mx.random.normal(value.shape)
        weights[name] = value
    weights["vision_tower.patch_embed.proj.weight"] = mx.zeros((2, 2))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    (directory / "config.json").write_text(json.dumps(payload))
    return payload


def test_hadamard_pack_native_model_matches_mlx_lm_reference(tmp_path: Path) -> None:
    payload = _write_hadamard_pack(tmp_path)
    reference, _ = load_mlx_lm_model(tmp_path)

    architecture = detect_native_architecture(str(tmp_path))
    assert architecture.architecture_class == "Qwen3_5ForConditionalGeneration"
    config = architecture.execution_plan.parse_config(payload)
    assert config.hadamard is not None and config.quantization is None
    index = load_weight_index(tmp_path)
    assert index.source_files == ("model.safetensors",)
    weights = load_mapped_weights(index, Qwen3_5WeightAdapter().build_plan(index))
    model = build_qwen3_5_model(config, weights)
    assert isinstance(model.model.embed_tokens, HadamardQuantizedEmbedding)
    assert isinstance(model.lm_head, HadamardQuantizedLinear)
    assert model.resolved_config.kv_cache_dtype == "float32"

    tokens = [1, 4, 9, 16, 25, 3]
    expected = reference(mx.array([tokens]))
    backend = _hybrid_backend(model)
    handle = backend.create("pack")
    actual = _paged_step(model, backend, [handle], [tokens])
    assert mx.allclose(actual[0], expected[0, -1], atol=1e-3, rtol=1e-3).item()


def test_parse_hadamard_pack_rejects_unknown_layouts() -> None:
    payload = _hadamard_payload()
    with pytest.raises(ValueError, match="layout"):
        parse_hadamard_pack({**payload, "gdn_activation_layout": "interleaved"})
    with pytest.raises(ValueError, match="schema_version"):
        parse_hadamard_pack({**payload, "schema_version": 9})
    bad_block = [dict(module) for module in payload["modules"]]
    bad_block[0]["block"] = 300
    with pytest.raises(ValueError, match="block size"):
        parse_hadamard_pack({**payload, "modules": bad_block})


def test_engine_loads_hadamard_packs_with_pack_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlx_worker import engine
    from mlx_worker import prism_hadamard as prism

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "prism_hadamard_qwen35"})
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        prism, "load_mlx_lm", lambda path: calls.append(path) or ("m", "t")
    )
    load_model, _, _ = engine._load_mlx_components()
    assert load_model(str(tmp_path)) == ("m", "t")
    assert calls == [tmp_path]


def test_native_tokenizer_stops_on_every_hadamard_pack_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlx_lm.utils

    from mlx_worker.native_mlx import bootstrap

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "prism_hadamard_qwen35"})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 3]})
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")

    class Raw:
        eos_token_id = 7

    class Wrapper:
        _tokenizer = Raw()

    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda path: Wrapper())
    _, _, eos = bootstrap._load_tokenizer(tmp_path)
    assert eos == (7, 3)
