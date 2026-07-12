# Native MLX v2 Reference

The native MLX backend is experimental and only supports explicitly implemented
Hugging Face architecture classes. It is never selected automatically.

This reference is for operators validating or running the `native-mlx` backend.
It describes the current support boundary, configuration surface, validation
entrypoints, evidence artifacts, metrics, known limitations, and v1 fallback.

## Support Boundary

`native-mlx` is an explicit backend selector. The default backend remains `v1`.

The first supported architecture class is:

```text
Qwen2ForCausalLM
```

The known-good checkpoint used by the native-v2 plan and host scripts is:

```text
mlx-community/Qwen2.5-7B-Instruct-4bit
```

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
| `MLX_RUNTIME_NATIVE_KV_PAGE_SIZE` | `8`, `16`, `32` | Native paged-KV page size |
| `MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY` | `radix`, `block-hash` | Prefix-cache strategy; `radix` is the native-v2 default |
| `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY` | `fcfs`, `lpm`, `lof`, `priority` | Python scheduler waiting-queue policy |
| `MLX_RUNTIME_NATIVE_GRAPH_PROFILE` | `0`, `1` | Enables diagnostic graph-profile metrics; keep off for fair benchmarks |
| `MLX_RUNTIME_TEXT_PROMPT_CONCURRENCY` | positive integer | Text prompt/prefill admission width |
| `MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE` | positive integer | Chunked-prefill token budget |
| `MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES` | positive integer | Text KV/prefix-cache byte budget |
| `MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES` | positive integer | Prefix-cache entry bound |

Invalid native page-size, prefix-cache strategy, or scheduler-policy values must
fail startup. They must not silently select dense attention, another prefix
strategy, v1, `mlx-lm`, or `mlx-vlm`.

## Validation Entry Points

Each native-v2 phase owns a host script under `mlx-host-validation/scripts/`.
Phase 14 owns the full completion gate:

```bash
bash mlx-host-validation/scripts/v2_phase_14.sh
```

The Phase 14 gate checks:

- required phase scripts `v2_phase_1.sh` through `v2_phase_12.sh`;
- Python sync, formatting, linting, and tests;
- Rust formatting, clippy, and workspace tests;
- host-only phase workstreams for startup, serving, parity, streaming,
  batching, chunked prefill, paged KV/attention, prefix strategies, queue
  policies, cancellation, metrics, unsupported-class behavior, and v1
  non-regression.

The script writes a durable report to:

```text
benchmarks/results/v2_phase_14_completion.md
```

Phase 14 passes only when the full gate actually runs on compatible Apple
Silicon/Metal and every required workstream succeeds. A blocked report is
evidence, not a completion claim.

## Benchmark and Trace Artifacts

Benchmarks and traces are separate artifacts.

Benchmark rows must keep tracing off and label backend, mode, checkpoint,
tokenizer/template, workload, prompt/completion token counts, TTFT, ITL,
latency, throughput, cache/KV metrics, and scheduler/executor metrics.

Semantic traces are diagnostic. They compare finalized token IDs through
bounded checkpoints such as embeddings, attention, MLP, logits, and KV append
state. Trace output must not be mixed into benchmark leaderboards.

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

- `native-mlx` is experimental and explicit opt-in.
- Only explicitly implemented architecture classes are supported.
- Greedy decoding is the validated native path.
- VLM requests remain outside `native-mlx`.
- Phase 13 overlap scheduling is superseded and does not block Phase 14.
  Future MLX-safe overlap or measured scheduler-gap reduction work belongs in
  the Phase 17 appendix.
- Appendix phases are follow-up work and do not weaken Phase 14 completion.

## v1 Fallback

Use v1 by leaving the default backend unchanged:

```toml
[worker]
backend = "v1"
```

or by setting:

```bash
MLX_RUNTIME_BACKEND=v1
```

`native-mlx` startup failures should tell the operator that v1 remains
available. Native startup must never silently fall back to v1.
