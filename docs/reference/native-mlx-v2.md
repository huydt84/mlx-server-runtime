# Native MLX v2 Reference

The native MLX backend is experimental and only supports explicitly implemented
Hugging Face architecture classes. It is never selected automatically.

This reference is for operators validating or running the `native-mlx` backend.
It describes the current support boundary, configuration surface, validation
entrypoints, evidence artifacts, metrics, known limitations, and v1 fallback.

## Support Boundary

`native-mlx` is the default backend. It serves only the architecture classes
below; select `v1` for any other `mlx-lm` checkpoint and for `vlm_model`.

The registered architecture classes and their known-good checkpoints are:

| Architecture class | Known-good checkpoints |
| --- | --- |
| `Qwen2ForCausalLM` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| `Qwen3ForCausalLM` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `Gemma3ForCausalLM` | `mlx-community/gemma-3-270m-it-qat-8bit` |
| `Lfm2MoeForCausalLM` | `mlx-community/LFM2.5-8B-A1B-MLX-4bit` |
| `Qwen3_5ForConditionalGeneration` (dense, text only) | `prism-ml/Bonsai-27B-mlx-1bit`, `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` |

A checkpoint without `architectures` is matched by model type only where a
registered class implements its weight format: `prism_hadamard_qwen35` maps to
`Qwen3_5ForConditionalGeneration`.

Support is architecture-class based, not model-family-name based. A checkpoint
declaring an unsupported architecture must fail before serving. A supported
architecture that fails because of weight layout, artifact, tokenizer,
attention, cache, or runtime incompatibility is treated as a compatibility bug
unless the checkpoint itself is malformed.

`native-mlx` does not claim:

- universal Hugging Face model support;
- automatic replacement of v1;
- VLM support;
- SGLang or vLLM feature parity;
- FlashAttention or FlashInfer equivalence;
- multi-model serving in one process.

## Ownership Summary

The serving path is:

```text
Client
  -> Rust gateway and worker supervisor
  -> backend-gated IPC
  -> Python worker transport
  -> Python native runtime
  -> Python NativeScheduler
  -> Python model-step executor
  -> architecture model module
  -> MLX / Metal
```

Rust owns HTTP/SSE, outer admission, readiness, health, worker supervision,
cancellation initiation, and gateway telemetry.

Python owns prompt construction, tokenization, request lifecycle,
detokenization, terminal semantics, scheduling, executor batch construction,
cache lifecycle policy, model-step execution, and native MLX tensors.

The core model-step boundary remains:

```text
ExecutionBatch -> StepResult
```

The executor is not a whole-request `generate()` abstraction. The scheduler
chooses token work and cache lifecycle timing; the executor performs one model
step and returns typed per-request results or structured errors.

The executor pads every row of a step to the longest row, so the scheduler never
puts decode rows and prefill chunks in the same step. When both kinds of work
are pending, steps alternate: one step decodes every running request, the next
prefills one chunk for one request (or up to the prompt concurrency when no
request is decoding).

## Configuration

Select the native backend explicitly:

```toml
[worker]
backend = "native-mlx"
model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

Useful native-v2 environment variables:

| Variable | Values | Purpose |
| --- | --- | --- |
| `MLX_RUNTIME_BACKEND` | `v1`, `native-mlx` | Worker backend selector |
| `MLX_RUNTIME_NATIVE_EXECUTION_BACKEND` | `native-metal-paged-sdpa` | Compatible native cache-and-attention bundle; defaults to the existing production path |
| `MLX_RUNTIME_NATIVE_EXECUTION_MODE` | `overlap`, `serial` | Startup execution coordinator; `overlap` (the default) pipelines decode on the owner thread, `serial` resolves every step before building the next |
| `MLX_RUNTIME_NATIVE_KV_PAGE_SIZE` | `8`, `16`, `32` | Native paged-KV page size |
| `MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY` | `radix`, `block-hash` | Prefix-cache strategy; `radix` is the native-v2 default |
| `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY` | `fcfs`, `lpm`, `lof`, `priority` | Python scheduler waiting-queue policy |
| `MLX_RUNTIME_NATIVE_GRAPH_PROFILE` | `0`, `1` | Enables diagnostic graph-profile metrics; keep off for fair benchmarks |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE` | `0`, `1` | Enables bounded whole-pipeline diagnostic events; disabled by default |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR` | directory path | Required output directory when pipeline profiling is enabled |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID` | string | Optional run ID shared by gateway-side and worker-side events |
| `MLX_RUNTIME_NATIVE_METAL_CAPTURE` | `0`, `1` | Requests optional heavy Metal capture preflight; requires process-start `MTL_CAPTURE_ENABLED=1` |
| `MLX_RUNTIME_TEXT_PROMPT_CONCURRENCY` | positive integer | Text prompt/prefill admission width |
| `MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE` | positive integer | Chunked-prefill token budget |
| `MLX_RUNTIME_MEMORY_UTILIZATION` | float in (0, 1] | Share of the Metal recommended working set for weights, activations, and the KV pool; the pool gets the remainder after the weights load. Set from `[memory] utilization` / `--memory-utilization` |
| `MLX_RUNTIME_KV_CACHE_BYTES` | positive integer | Absolute KV page-pool size; overrides the utilization. Set from `[memory] kv_cache_bytes` / `--kv-cache-memory` |
| `MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES` | positive integer | Legacy name for `MLX_RUNTIME_KV_CACHE_BYTES`, used when that is unset. The pool is allocated at startup and shared by in-flight requests and the prefix cache; new requests wait until their prompt plus `max_tokens` fits |
| `MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES` | positive integer | Prefix-cache entry bound; each cached KV page (`MLX_RUNTIME_NATIVE_KV_PAGE_SIZE` tokens) is one entry. Unset, the prefix cache may hold every page of the KV pool: when admitting a request needs pages, the scheduler first evicts least recently used prefix entries that no running request uses |
| `MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES` | non-negative integer | Freed MLX device buffers kept for reuse; defaults to 1 GiB so changing batch shapes cannot fill device memory and push the host into swap |

Invalid native execution-backend, execution-mode, page-size, prefix-cache
strategy, or scheduler-policy values must fail startup. They must not silently
select dense attention, another prefix strategy, v1, `mlx-lm`, or `mlx-vlm`.
The default `overlap` mode never silently falls back to serial. It
builds every step, prefill included, before the previous step's tokens are
read on the CPU: a decode step is fed the previous step's device-resident
tokens, and a prefill step does not depend on the previous step. Use
`mlx-host-validation/scripts/native_overlap.sh` to require exact serial/overlap
output parity, a 2% performance confidence-bound gate, CPU overlap evidence,
an MLX `.gputrace`, and matched serial/overlap Metal System Traces from the
Python worker. The Metal gate uses the same sustained four-request concurrent
decode workload in both modes and rejects a p95 GPU interval-gap regression
beyond 2% or 50 microseconds, whichever allowance is larger.

## Hybrid Models

`Lfm2MoeForCausalLM` and `Qwen3_5ForConditionalGeneration` use the hybrid
cache: attention layers store paged KV, and the other layers keep
request-local state committed with each step.

- LFM2 short-convolution layers keep the last `conv_L_cache - 1` inputs.
- Qwen3.5 Gated DeltaNet layers keep the last `linear_conv_kernel_dim - 1`
  convolution inputs and a float32 recurrent state of
  `linear_num_value_heads x linear_value_head_dim x linear_key_head_dim` per
  layer. On a 27B Qwen3.5 checkpoint that is about 150 MiB per in-flight
  request, outside `MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES`.
- Qwen3.5 allocates KV pages only for its full-attention layers, so the
  budget is not spent on Gated DeltaNet layers. The KV dtype is the dtype the
  loaded weights produce, which can be float32.
- Recurrent state cannot be truncated to a prefix, so hybrid models run
  without a prefix cache; requests never reuse cached tokens.
- Padded prefill rows mask padding out of both the convolution state and the
  recurrence, so batching does not change a request's output.

The Gated DeltaNet recurrence uses the `mlx-lm` Metal kernel
(`mlx_lm.models.gated_delta`), so native and `mlx-lm` runs execute the same
recurrence code.

### Hadamard-Rotated Packs

`prism_hadamard_qwen35` packs store projection weights folded with a
blockwise sign-and-Hadamard rotation. The native model, and the v1 backend
when it loads such a pack, apply the activation rotation before each packed
matmul and rotate looked-up embedding rows back. Plain `mlx_lm.load` cannot
load these packs. Their norms and other unquantized tensors are float32, so
the residual stream and the KV cache are float32.

## Attention Backend Contract

The native executor selects attention through a registered execution-backend
bundle. A bundle pairs one KV-cache backend with one compatible attention
backend and validates that pairing before serving. Architecture models continue
to depend only on `LayerAttentionContext`; they do not select kernels or inspect
page tables.

Each attention backend declares a stable identifier, compatible cache and
reservation types, supported masks and forward modes, Metal requirements, and
whether it consumes page tables directly. Adding another causal backend with
the existing model-facing operation requires a new identifier, factory entry,
and backend contract tests, but no Qwen2 or executor changes. New attention
semantics such as sinks or sliding windows still require an explicit model
interface extension.

The current `native-metal-paged-sdpa` bundle stores KV in fixed-size pages and
uses two attention paths:

- **Decode steps** (every row adds one token) run a Metal kernel that follows
  each row's block table while it attends, so rows of different lengths need
  neither a dense copy nor padding. One threadgroup serves one row and KV head
  together with all query heads that share it. When a step has few rows, each
  row's keys are split into ranges of at least 32 keys on separate
  threadgroups and merged by a second kernel. Head dimensions that are not a multiple of 32 fall back to
  the gather path below.
- **Prefill and mixed steps** gather each row's pages into padded dense K/V
  and call `mx.fast.scaled_dot_product_attention`.

Each layer writes the step's new K/V into the pool before attending: one
slice copy when the step's cache slots are consecutive (a single row, for
one), otherwise one indexed scatter.

Its capability metadata therefore reports that it consumes page tables
directly.

Attention backend identity and dispatch telemetry are owned by the attention
adapter. Cache metrics remain owned by the KV backend. Because MLX evaluation
is lazy, `attention_time_ms` measures attention graph construction/dispatch;
`executor_eval_ms` contains the synchronized MLX execution boundary.

## Profiling

Whole-pipeline profiling is an opt-in diagnostic mode. It writes
request-correlated events across the gateway, worker transport, runtime,
scheduler, executor, cache, model, MLX synchronization, detokenization, and
streaming boundaries.

See [Profile inference](../how-to/profiling.md) for the runnable workflow,
artifact formats, optional Metal capture, and interpretation guidance.

## Benchmark and Trace Artifacts

Benchmarks and traces are separate artifacts.

Benchmark rows must keep tracing off and label backend, mode, checkpoint,
tokenizer/template, workload, prompt/completion token counts, TTFT, ITL,
latency, throughput, cache/KV metrics, and scheduler/executor metrics.

Semantic traces are diagnostic. They compare finalized token IDs through
bounded checkpoints such as embeddings, attention, MLP, logits, and KV append
state. Trace output must not be mixed into benchmark leaderboards.

Pipeline profiles are also diagnostic, but cover gateway, worker transport,
runtime, scheduler, executor/cache/model, MLX synchronization, detokenization,
and streaming stages. They complement rather than replace semantic traces or
paged-attention kernel captures.

Native-v2 does not need to beat v1 to be correct. Performance regressions must
be reported honestly and routed to follow-up optimization work.

## Metrics

Operators should inspect at least these metric families during validation:

- `mlx_latency_by_backend_ms`
- `mlx_scheduler_tick_latency_by_backend_ms`
- `mlx_scheduler_stage_latency_by_backend_ms`
- `mlx_scheduler_requests_by_backend`
- `mlx_scheduler_policy_by_backend`
- `mlx_scheduled_tokens_by_backend`
- `mlx_executor_physical_batch_size_by_backend`
- `mlx_executor_model_forward_count_by_backend`
- `mlx_executor_stage_latency_by_backend_ms`
- `mlx_kv_cache_pages_by_backend`
- `mlx_kv_cache_active_bytes_by_backend`
- `mlx_kv_cache_fragmentation_tokens_by_backend`
- `mlx_attention_time_by_backend_ms`
- `mlx_prefix_cache_hits_by_backend`
- `mlx_prefix_cache_reused_tokens_by_backend`
- `mlx_radix_cache_by_backend`

Metrics are owned by the layer that produces them. Rust projects typed worker
metrics to Prometheus; it does not reconstruct scheduler or executor state from
request totals.

## Known Limitations

- The only registered production execution backend is currently `native-metal-paged-sdpa`.
- Only explicitly implemented architecture classes are supported.
- Hybrid models (LFM2-MoE, Qwen3.5) run without a prefix cache.
- Qwen3.5 mixture-of-experts checkpoints and the Qwen3.5 vision tower are not
  implemented.
- Native decoding supports `temperature` and `top_p`. `temperature: 0` is greedy and bit-identical to `argmax`; a positive temperature scales the logits, keeps the smallest set of tokens whose probability reaches `top_p`, and draws on the device with the Gumbel-max trick. Each request gets a random seed, and every draw is keyed by that seed and the token position, so the random draw does not depend on serial or overlap execution, batch composition, or prompt chunking. The logits themselves can differ in the last bits with batch composition and with how much of the prompt came from the prefix cache, so two runs of the same prompt are not guaranteed to produce identical text. A per-request `seed` field is not exposed yet. `top_k`, penalties, and logprobs are not supported.
- VLM requests remain outside `native-mlx`; a `vlm_model` with the native backend fails at startup with `VLM_UNSUPPORTED`.
- `overlap` is the default execution mode; set
  `MLX_RUNTIME_NATIVE_EXECUTION_MODE=serial` for the synchronous coordinator.

## v1 Backend

Select v1 for checkpoints native-mlx does not implement and for VLMs:

```toml
[worker]
backend = "v1"
```

or by setting:

```bash
MLX_RUNTIME_BACKEND=v1
```

or with `mlx-air serve --backend v1`. `native-mlx` startup failures tell the
operator to start with `--backend v1`. Native startup never silently falls
back to v1.
