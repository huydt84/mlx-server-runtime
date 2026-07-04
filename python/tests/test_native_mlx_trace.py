from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from mlx_worker.native_mlx.models.Qwen2ForCausalLM.cache import Qwen2LayerCache
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.config import Qwen2ModelConfig
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.debug_trace import (
    TraceCheckpoint,
    compare_trace_runs,
    trace_qwen2_run,
    write_trace_artifacts,
)
from mlx_worker.native_mlx.models.Qwen2ForCausalLM.model import Qwen2ForCausalLm
from mlx_worker.native_mlx.worker import build_prompt_fingerprint


def _tiny_qwen2_config(num_hidden_layers: int = 2) -> Qwen2ModelConfig:
    return Qwen2ModelConfig(
        architecture_class="Qwen2ForCausalLM",
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=False,
        quantization=None,
    )


def _with_backend(run, backend: str):
    return replace(
        run,
        backend=backend,
        prefill=tuple(
            TraceCheckpoint(
                record=replace(checkpoint.record, backend=backend),
                values=checkpoint.values.copy(),
            )
            for checkpoint in run.prefill
        ),
        decode=tuple(
            TraceCheckpoint(
                record=replace(checkpoint.record, backend=backend),
                values=checkpoint.values.copy(),
            )
            for checkpoint in run.decode
        ),
    )


def _reference_qwen2_model(config: Qwen2ModelConfig):
    from mlx_lm.models.qwen2 import Model as ReferenceModel
    from mlx_lm.models.qwen2 import ModelArgs as ReferenceArgs

    return ReferenceModel(
        ReferenceArgs(
            model_type="qwen2",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_traditional=config.rope_traditional,
            rope_scaling=config.rope_scaling,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    )


def test_build_prompt_fingerprint_is_stable_and_prompt_sensitive() -> None:
    messages = [{"role": "user", "content": "ping"}]

    first = build_prompt_fingerprint(messages)
    second = build_prompt_fingerprint(messages)
    different = build_prompt_fingerprint([{"role": "user", "content": "pong"}])

    assert first == second
    assert first != different


def test_qwen2_trace_run_covers_required_checkpoints_and_bounds_samples() -> None:
    config = _tiny_qwen2_config()
    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)

    run = trace_qwen2_run(
        model=native_model,
        model_config=config,
        backend="native-mlx",
        prompt_token_ids=(1, 2, 3),
        prompt_fingerprint="prompt-fingerprint",
        cache=[Qwen2LayerCache() for _ in range(config.num_hidden_layers)],
        decode_input_token_ids=(7,),
        sample_size=3,
        selected_dumps=("prefill.step0.layer0.attention.hidden_states",),
    )

    checkpoint_ids = {
        checkpoint.record.checkpoint_id for checkpoint in run.prefill + run.decode
    }
    assert "prefill.step0.global.embedding.hidden_states" in checkpoint_ids
    assert "prefill.step0.layer0.kv_append.keys" in checkpoint_ids
    assert "prefill.step0.layer1.kv_append.values" in checkpoint_ids
    assert "decode.step0.layer0.attention.hidden_states" in checkpoint_ids
    assert all(
        len(checkpoint.record.sample_values) <= 3
        for checkpoint in run.prefill + run.decode
    )
    dumped = next(
        checkpoint
        for checkpoint in run.prefill
        if checkpoint.record.checkpoint_id
        == "prefill.step0.layer0.attention.hidden_states"
    )
    assert dumped.record.full_values is not None
    assert all(
        checkpoint.record.full_values is None
        for checkpoint in run.prefill + run.decode
        if checkpoint.record.checkpoint_id
        != "prefill.step0.layer0.attention.hidden_states"
    )


def test_qwen2_trace_run_covers_many_layers_and_decode_steps() -> None:
    config = _tiny_qwen2_config(num_hidden_layers=4)
    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)

    run = trace_qwen2_run(
        model=native_model,
        model_config=config,
        backend="native-mlx",
        prompt_token_ids=(1, 2, 3, 4),
        prompt_fingerprint="prompt-fingerprint",
        cache=[Qwen2LayerCache() for _ in range(config.num_hidden_layers)],
        decode_input_token_ids=(9, 10, 11),
    )

    checkpoints_per_step = 3 + (config.num_hidden_layers * 5)
    assert len(run.prefill) == checkpoints_per_step
    assert len(run.decode) == 3 * checkpoints_per_step
    for layer_index in range(config.num_hidden_layers):
        assert any(
            checkpoint.record.checkpoint_id
            == f"prefill.step0.layer{layer_index}.attention.hidden_states"
            for checkpoint in run.prefill
        )
        assert any(
            checkpoint.record.checkpoint_id
            == f"prefill.step0.layer{layer_index}.mlp.hidden_states"
            for checkpoint in run.prefill
        )
        assert any(
            checkpoint.record.checkpoint_id
            == f"prefill.step0.layer{layer_index}.residual.hidden_states"
            for checkpoint in run.prefill
        )
        assert any(
            checkpoint.record.checkpoint_id
            == f"prefill.step0.layer{layer_index}.kv_append.keys"
            for checkpoint in run.prefill
        )
        assert any(
            checkpoint.record.checkpoint_id
            == f"decode.step2.layer{layer_index}.kv_append.values"
            for checkpoint in run.decode
        )


def test_qwen2_trace_prefill_logits_match_direct_model_forward() -> None:
    config = _tiny_qwen2_config()
    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    direct_logits = native_model(inputs)
    mx.eval(direct_logits)

    run = trace_qwen2_run(
        model=native_model,
        model_config=config,
        backend="native-mlx",
        prompt_token_ids=(1, 2, 3),
        prompt_fingerprint="prompt-fingerprint",
        cache=[Qwen2LayerCache() for _ in range(config.num_hidden_layers)],
    )

    logits_checkpoint = next(
        checkpoint
        for checkpoint in run.prefill
        if checkpoint.record.checkpoint_id == "prefill.step0.global.logits.logits"
    )
    assert logits_checkpoint.values.shape == (1, 3, 256)
    assert bool(mx.allclose(mx.array(logits_checkpoint.values), direct_logits).item())


def test_reference_qwen2_trace_prefill_logits_match_direct_model_forward() -> None:
    from mlx_lm.models.cache import make_prompt_cache

    config = _tiny_qwen2_config()
    mx.random.seed(0)
    reference_model = _reference_qwen2_model(config)
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    direct_logits = reference_model(inputs, cache=make_prompt_cache(reference_model))
    mx.eval(direct_logits)

    run = trace_qwen2_run(
        model=reference_model,
        model_config=config,
        backend="mlx-lm",
        prompt_token_ids=(1, 2, 3),
        prompt_fingerprint="prompt-fingerprint",
        cache=make_prompt_cache(reference_model),
    )

    logits_checkpoint = next(
        checkpoint
        for checkpoint in run.prefill
        if checkpoint.record.checkpoint_id == "prefill.step0.global.logits.logits"
    )
    assert logits_checkpoint.values.shape == (1, 3, 256)
    assert bool(mx.allclose(mx.array(logits_checkpoint.values), direct_logits).item())


def test_native_and_reference_trace_align_across_many_layers_and_decode_steps() -> None:
    from mlx_lm.models.cache import make_prompt_cache

    config = _tiny_qwen2_config(num_hidden_layers=4)
    prompt_token_ids = (1, 2, 3, 4)
    prompt_fingerprint = "prompt-fingerprint"

    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)
    mx.random.seed(0)
    reference_model = _reference_qwen2_model(config)

    reference_run = trace_qwen2_run(
        model=reference_model,
        model_config=config,
        backend="mlx-lm",
        prompt_token_ids=prompt_token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=make_prompt_cache(reference_model),
        decode_steps=3,
    )
    native_run = trace_qwen2_run(
        model=native_model,
        model_config=config,
        backend="native-mlx",
        prompt_token_ids=prompt_token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=[Qwen2LayerCache() for _ in range(config.num_hidden_layers)],
        decode_input_token_ids=reference_run.decode_input_token_ids,
    )

    comparison = compare_trace_runs(
        native_run,
        reference_run,
        tolerance_atol=2e-2,
        tolerance_rtol=2e-2,
    )

    assert comparison.aligned
    assert comparison.first_mismatch is None
    assert native_run.generated_token_ids == reference_run.generated_token_ids
    assert native_run.decode_input_token_ids == reference_run.decode_input_token_ids
    assert len(native_run.decode) == len(reference_run.decode)


def test_trace_comparison_distinguishes_missing_and_numeric_mismatches(
    tmp_path: Path,
) -> None:
    config = _tiny_qwen2_config()
    mx.random.seed(0)
    native_model = Qwen2ForCausalLm(config)
    prompt_token_ids = (1, 2, 3)
    prompt_fingerprint = "prompt-fingerprint"

    native_run = trace_qwen2_run(
        model=native_model,
        model_config=config,
        backend="native-mlx",
        prompt_token_ids=prompt_token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=[Qwen2LayerCache() for _ in range(config.num_hidden_layers)],
        decode_input_token_ids=(4,),
    )

    reference_run = _with_backend(native_run, "mlx-lm")

    missing_reference = replace(
        reference_run,
        prefill=tuple(
            checkpoint
            for checkpoint in reference_run.prefill
            if checkpoint.record.checkpoint_id
            != "prefill.step0.layer0.attention.hidden_states"
        ),
    )
    missing = compare_trace_runs(
        native_run,
        missing_reference,
        tolerance_atol=2e-2,
        tolerance_rtol=2e-2,
    )
    assert not missing.aligned
    assert missing.first_mismatch is not None
    assert missing.first_mismatch.kind == "missing_checkpoint"
    assert (
        missing.first_mismatch.checkpoint_id
        == "prefill.step0.layer0.attention.hidden_states"
    )

    mutated_reference = replace(
        reference_run,
        prefill=tuple(
            TraceCheckpoint(record=checkpoint.record, values=checkpoint.values + 0.5)
            if checkpoint.record.checkpoint_id
            == "prefill.step0.layer0.attention.hidden_states"
            else checkpoint
            for checkpoint in reference_run.prefill
        ),
    )
    numeric = compare_trace_runs(
        native_run,
        mutated_reference,
        tolerance_atol=2e-2,
        tolerance_rtol=2e-2,
    )
    assert not numeric.aligned
    assert numeric.first_mismatch is not None
    assert numeric.first_mismatch.kind == "numeric_mismatch"

    artifacts = write_trace_artifacts(
        output_dir=tmp_path,
        checkpoint="tiny-qwen2",
        native_run=native_run,
        reference_run=mutated_reference,
        comparison=numeric,
        tolerance_atol=2e-2,
        tolerance_rtol=2e-2,
    )
    assert artifacts.prefill_jsonl_path.exists()
    assert artifacts.decode_jsonl_path.exists()
    assert artifacts.summary_markdown_path.exists()
    summary = artifacts.summary_markdown_path.read_text()
    assert "numeric_mismatch" in summary
    assert "prefill.step0.layer0.attention.hidden_states" in summary
    prefill_jsonl = artifacts.prefill_jsonl_path.read_text()
    assert '"backend": "native-mlx"' in prefill_jsonl
    assert '"backend": "mlx-lm"' in prefill_jsonl
