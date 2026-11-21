# 0008. Checkpoint loading compatibility

- Status: Accepted
- Date: 2026-11-22

## Context

Native-mlx loads MLX exports from many publishers. Valid exports differ in
which sidecar files they ship and in how they declare their architecture.

## Decision

- **Only `tokenizer.json` and `tokenizer_config.json` are required.** Some
  valid exports, including LFM2, keep special-token metadata in those files
  and omit the optional sidecar.
- **Prism Hadamard checkpoints add every end-of-generation id from
  `generation_config.json`.** The tokenizer alone knows only one of them.
- **Checkpoints with no `architectures` field map by `model_type`** to the
  registered architecture whose weight format they use (for example
  `prism_hadamard_qwen35` to `Qwen3_5ForConditionalGeneration`).
- **Hadamard-folded weights:** a linear layer stores `W' = W Rᵀ`, so
  `W' · rotate(x) = W · x`. An embedding stores rotated rows and returns the
  original rows.
- **A model may refine its config from the loaded weights.** For example it
  may set the K/V activation dtype that sizes the KV pool. Bootstrap reads
  `resolved_config` after the model is built. Hadamard checkpoints with
  float32 norms keep K/V in float32.
- **The v1 batching path feature-detects mlx-lm's optional `all_tokens`
  keyword** instead of requiring it ([#7](https://github.com/huydt84/mlx-server-runtime/issues/7)).

## Code

- `python/mlx_worker/native_mlx/bootstrap.py` — `build_native_artifacts`,
  `_load_tokenizer`.
- `python/mlx_worker/native_mlx/registry.py` — `_MODEL_TYPE_ARCHITECTURES`.
- `python/mlx_worker/native_mlx/models/` — Hadamard quantized layers.
- `python/mlx_worker/batching.py` — `_validate_batch_generator_contract`.
