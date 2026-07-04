# Phase 10 Review Ledger

## Phase contract summary

- Required behavior: text LLM continuous batching, VLM continuous batching, prefix/cache reuse, cancellation, telemetry, and concurrent routing across baseline and continuous modes.
- Required inputs/outputs: OpenAI-compatible `POST /v1/chat/completions` for text and VLM requests; `GET /metrics` for observability.
- Required validation signals: `baseline_ready=1`, `continuous_ready=1`, `baseline_metrics_ok=1`, `continuous_metrics_ok=1`, `vlm_metrics_ok=1`, `continuous_join_while_decoding_ok=1`, `vlm_continuous_join_while_decoding_ok=1`, `cache_hits_ok=1`, `vlm_local_image_ok=1`, `cancellation_ok=1`, `phase_10_validation_ok=1`.
- Required failure behavior: readiness or request failure must surface stable HTTP / error responses and exit non-zero in validation.
- Required performance constraints: measurable TTFT, latency, tokens/sec, and batch-join behavior; no exact model-text requirement.
- Required observability/metrics: cache hits, cached tokens, memory, cancellation, VLM request/image counts, VLM media-prep timings.
- Required host model/config: text model `mlx-community/Qwen2.5-7B-Instruct-4bit`; VLM model `mlx-community/Qwen2-VL-2B-Instruct-4bit`; local image fixture `benchmarks/images/fruits.png` unless overridden.
- Required public API request path(s): `POST /v1/chat/completions`, `GET /metrics`, readiness probe via `GET /health` during startup wait.
- Required completion request(s): text decode-only, text short/long/shared-prefix/cancel-followup/cache-pressure, VLM decode-only, VLM text-only, VLM image-bearing, VLM repeated-image, VLM cancel-followup.
- Required non-completion request(s): cancellation during streaming decode for text and VLM, plus concurrent join-while-decoding requests.
- Required phase-specific assertions: real continuous-batch join while decoding for text and VLM, prompt-cache hits, VLM image handling, non-empty real responses, and deterministic metrics presence.

## Reviewer rounds

- R0:
  - Reviewer called after current diff: no
  - Result: pending
  - Blockers: none yet

- R1:
  - Reviewer called after final code/test/script changes: yes
  - Result: none
  - Blockers: none

## Open blockers

- None.

## Resolved blockers

- PH10-BLK-001: VLM image-bearing cache path now uses media-safe local-file cache namespaces and a dedicated no-op-trim image cache store; URL images are excluded from fallback reuse.
- PH10-BLK-002: VLM cancellation during preparation is now observable through reader-thread cancel tracking plus scheduler `request_cancelled` hook; unit coverage retained.
- PH10-BLK-003: Host script now proves mixed text+VLM concurrent progress with absolute first-chunk/finish timestamps and `mixed_backend_fairness_ok=1`.
- PH10-BLK-004: Host script now gates throughput on text-suite wall-clock completion/prompt throughput and prints `throughput_improved_ok=1`.

## Pre-review self-check

- Latest reviewer call happened after final code/test/validation changes: no
- No open blockers: yes
- All resolved blockers still have valid deterministic proof: yes
- Tests/lints/host validation passed: yes
- Host validation signals seen: `baseline_ready=1`, `continuous_ready=1`, `baseline_metrics_ok=1`, `continuous_metrics_ok=1`, `vlm_metrics_ok=1`, `mixed_backend_fairness_ok=1`, `continuous_join_while_decoding_ok=1`, `vlm_continuous_join_while_decoding_ok=1`, `cache_hits_ok=1`, `vlm_cache_hits_ok=1`, `vlm_local_image_ok=1`, `cancellation_ok=1`, `throughput_improved_ok=1`, `phase_10_validation_ok=1`.

## Final proof

- Host proof saved under `/var/folders/.../mlx-runtime-phase-10/` for baseline and continuous summaries/logs.
- Final reviewer re-check complete; no blockers remain.
