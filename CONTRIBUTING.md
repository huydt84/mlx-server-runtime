# Contributing

Thank you for improving MLX Server Runtime. This guide is for contributors
working on the native MLX model path, especially new model architectures,
model-output debugging, and inference-graph profiling.

Before editing, set up the repository toolchain:

```bash
cargo build --workspace
cd python
uv sync
cd ..
```

Use the repository tools exactly: `cargo` for Rust and `uv`, `ruff`, and
`pytest` for Python. Keep hardware-dependent model checks separate from unit
tests.

## Understand the Model Boundary

The native serving path is:

```text
Rust gateway
  -> Python worker transport
  -> native runtime and scheduler
  -> model-step executor
  -> architecture model module
  -> MLX / Metal
```

Architecture-specific code belongs in
`python/mlx_worker/native_mlx/models/`. A model module owns configuration,
tensor operations, weight mapping, and model construction. It must not own
request lifecycle, scheduling, HTTP/IPC, cache policy, or whole-request
generation.

The shared executor contract remains:

```text
ExecutionBatch -> StepResult
```

Do not add a second serving path or delegate native execution to
`mlx_lm.generate()`.

## Add a Model Architecture

Use `python/mlx_worker/native_mlx/models/qwen2.py` as the working example.

### 1. Define the architecture module

Create a module such as:

```text
python/mlx_worker/native_mlx/models/<architecture>.py
```

It should provide:

- a frozen configuration dataclass containing only fields used by the graph;
- a strict configuration parser for the Hugging Face `config.json`;
- MLX modules for the backbone, attention projections, MLP, normalization, and
  causal-LM head;
- a weight adapter that maps checkpoint tensor names to the native graph;
- a model builder that applies quantization when supported, loads weights
  strictly, switches to evaluation mode, and evaluates parameters;
- a `num_layers` attribute and a model call compatible with `NativeModel`.

Keep the model-facing attention interface semantic. The model supplies
projected Q/K/V tensors, positions, scale, and mask intent through
`LayerAttentionContext`; it must not manipulate request cache handles or
physical page tables.

### 2. Register the architecture

Add an `ArchitectureSpec` entry in
`python/mlx_worker/native_mlx/registry.py`. The entry connects:

- the exact class named in `config.json` under `architectures[0]`;
- one known-good checkpoint;
- compatibility probes;
- the configuration parser;
- the weight adapter;
- the model builder;
- KV-cache geometry: layer count, KV heads, head dimension, and dtype.

Registration is explicit. Unsupported architecture classes must fail startup;
do not silently select a related implementation or fall back to v1.

### 3. Add deterministic unit tests

Extend `python/tests/test_native_mlx.py` with tests for:

- valid and invalid configuration parsing;
- registry lookup and cache geometry;
- checkpoint-name canonicalization and unsupported tensor names;
- strict weight loading and quantization selection;
- tensor shapes for prefill and decode;
- unequal-length batching;
- KV continuation and release;
- absence of runtime, scheduler, executor, worker, and IPC dependencies in the
  model module.

Use a tiny deterministic configuration. Unit tests should not download a full
checkpoint.

### 4. Prove numerical parity

Compare the native graph with the corresponding `mlx-lm` reference starting
from identical finalized token IDs. Validate direct forward logits first, then
prefill followed by forced-token decode. If output diverges, use the semantic
trace workflow below before changing tolerances.

If the model fails to load or execute with `mlx-lm`, use the corresponding
Hugging Face Transformers implementation as the reference instead. Compare
direct forward logits first, then prefill followed by forced-token decode from
the same finalized token IDs. When diagnosing a mismatch, compare equivalent
intermediate layer outputs and record any expected dtype or quantization
differences between the implementations.

### 5. Validate the full serving path

On Apple Silicon, start the gateway with the native backend and the candidate
checkpoint. Exercise both streaming and non-streaming requests through
`/v1/chat/completions`. Verify startup classification, token accounting,
cancellation, cache cleanup, and request-correlated metrics.

Do not claim support based only on imports, model construction, or a direct
forward call.

## Debug Model Layer Output

Semantic tracing compares native MLX and `mlx-lm` at model checkpoints and
reports the first meaningful divergence. It records prefill and decode outputs
for embeddings, attention, MLP, residual paths, normalization, logits, and KV
state.

Run a bounded trace from the repository root:

```bash
CHECKPOINT=mlx-community/Qwen2.5-7B-Instruct-4bit \
TRACE_DIR="$PWD/model-trace" \
uv --directory python run python - <<'PY'
import os
from pathlib import Path

from mlx_worker.native_mlx.bootstrap import (
    build_finalized_token_ids,
    build_native_artifacts,
)
from mlx_worker.native_mlx.diagnostics import (
    build_prompt_fingerprint,
    trace_native_debug_to_mlx_lm,
)

checkpoint = os.environ["CHECKPOINT"]
output_dir = Path(os.environ["TRACE_DIR"])
messages = [{"role": "user", "content": "Explain paged KV cache briefly."}]

artifacts = build_native_artifacts(checkpoint)
token_ids = build_finalized_token_ids(
    artifacts.architecture.model_path,
    messages,
)
result = trace_native_debug_to_mlx_lm(
    checkpoint,
    artifacts.diagnostics,
    token_ids,
    prompt_fingerprint=build_prompt_fingerprint(messages),
    output_dir=output_dir,
    decode_steps=2,
    selected_dumps=("logits",),
)

print(result.prefill_jsonl_path)
print(result.decode_jsonl_path)
print(result.summary_markdown_path)
PY
```

Read the Markdown summary first. Then inspect the matching native and reference
records in the prefill or decode JSONL file. Compare:

- checkpoint name and layer index;
- dtype and shape;
- finite, NaN, and infinity counts;
- min, max, mean, and standard deviation;
- stable hash and bounded sample values.

Fix the earliest divergent checkpoint, not the final text symptom. Common
causes include incorrect weight names, RoPE offsets, attention head layout,
mask semantics, residual ordering, normalization epsilon, tied embeddings,
quantization, or KV append position.

Only request full tensor dumps for selected checkpoints. Tracing forces MLX
evaluation and is intentionally separate from performance measurement.

Tests for trace coverage and comparison behavior live in
`python/tests/test_native_mlx_trace.py`.

## Profile Model-Graph Inference

Model-graph profiling measures MLX module categories inside a model step. It
is different from whole-pipeline profiling: graph profiling focuses on model
components, while `scripts/profile.sh` includes gateway, transport, scheduler,
executor, cache, synchronization, and streaming overhead.

The graph profiler is model-agnostic. `GraphProfiledModel` walks the MLX module
tree and recognizes common paths such as `layers`, `blocks`, `self_attn`,
`attention`, `mlp`, `ffn`, `norm`, embeddings, and `lm_head`. Name new model
modules clearly so the generic wrapper can classify them without
architecture-specific hooks.

### Enable graph profiling

Configure `config/runtime.toml` with `worker.backend = "native-mlx"`, then
start a diagnostic server:

```bash
MLX_RUNTIME_NATIVE_GRAPH_PROFILE=1 cargo run -p mlx_runtime_gateway
```

Send a representative request from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Write four short tokens."}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 4
  }'
```

Inspect the graph metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | rg 'model_graph|executor_stage'
```

The reported categories include embedding, projections, attention, MLP,
normalization, LM head, total layer time, worst-layer time, and worst-layer
index when those modules exist.

Graph profiling calls `mx.eval` around selected modules. This makes component
cost visible but changes execution timing and synchronization. Keep it off in
fair latency or throughput benchmarks, and confirm suspected bottlenecks with
whole-pipeline or Metal evidence before optimizing.

If a new architecture produces missing categories, add a tiny synthetic model
test in `python/tests/test_native_mlx.py` before changing the generic path
classifier. Avoid architecture-name checks in `graph_profile.py`.

## Run Validation

Run focused tests while developing, then the repository checks before opening
a pull request:

```bash
cd python
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest
cd ..

cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Report any Apple Silicon, Metal, or checkpoint validation that remains to be
run. Do not replace a required host check with mocks or a direct import test.

## Keep Changes Reviewable

- Keep architecture changes inside the model and registry seams.
- Add tests for every behavior change and failure boundary.
- Do not add dependencies for small helpers already covered by the standard
  library or current packages.
- Do not include model files, generated traces, `.gputrace` captures, or
  benchmark output in a pull request unless they are intentional review
  artifacts.
- Update user documentation when support, configuration, commands, metrics, or
  limitations change.
