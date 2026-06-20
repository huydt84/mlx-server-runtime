# Phase 8 Review Ledger

## Phase contract summary

- Required behavior: VLM chat completion with 1 local image or 1 web image returns text through runtime; text-only requests still use text engine; streaming works; cancellation/backpressure/lifecycle/metrics still apply.
- Required inputs/outputs: OpenAI-style `/v1/chat/completions` body with text and `image_url` parts; 200 JSON completion or SSE stream; `/models/{model}/status`, `/models/{model}/ready`, `/metrics`.
- Required validation signals: non-empty text for text and VLM completions; SSE `data:` chunks and `[DONE]`; VLM model `ready`; warmup passed with positive latency; metrics include VLM counters and latency histograms.
- Required failure behavior: invalid `http://` image rejected with `400` / `INVALID_IMAGE_URL` before model execution.
- Required performance constraints: no added blocking on gateway path; host gate must not regress load/stream completion.
- Required observability/metrics: `mlx_vlm_requests_total`, `mlx_vlm_image_count_total`, `mlx_vlm_image_preprocess_latency_ms`, `mlx_vlm_prompt_template_latency_ms`, `mlx_vlm_load_errors_total`, `mlx_ttft_ms`, `mlx_request_latency_ms`, `mlx_decode_tokens_per_second`.
- Required host model/config: text `mlx-community/Qwen2.5-7B-Instruct-4bit`; VLM `mlx-community/Qwen2-VL-2B-Instruct-4bit`; `MLX_RUNTIME_CONFIG` overrides `config/runtime.toml`.
- Required public API request path(s): `GET /health`, `GET /models`, `GET /models/{model}/status`, `GET /models/{model}/ready`, `GET /metrics`, `POST /v1/chat/completions`.
- Required completion request(s): text-only completion; VLM completion with 1 local image; VLM completion with 1 loopback web image.
- Required non-completion request(s): invalid `http://` image request rejected before model execution.
- Required phase-specific assertions: VLM request streams non-empty text and ends with `[DONE]`; VLM lifecycle reports `ready` and warmup passed with nonzero latency; metrics show exactly 3 VLM requests and 2 images, 0 load errors.

## Reviewer rounds

- R0:
  - Reviewer called after current diff: yes
  - Result: blockers found
  - Blockers: P8-B001, P8-B002, P8-S001, P8-S002

## Open blockers

- All previously open blockers resolved; kept here for traceability.

### R01 — synthetic VLM readiness

- Issue IDs: P8-B001
- Severity: must fix
- Category: error handling/lifecycle bug
- Root cause: VLM readiness is derived from first successful user traffic, and `mark_vlm_ready()` stamps zero warmup latency.
- Invariant: VLM `ready` / `warmup_passed` must come from explicit warmup transition with real latency, not first user completion.
- Affected files/boundaries: `rust/crates/gateway/src/http.rs`, `rust/crates/gateway/src/supervisor.rs`, `mlx-host-validation/scripts/phase_8.sh`.
- Regression guard: supervisor/unit test for explicit VLM warmup transition; host script assertion that VLM status is ready before user traffic and warmup latency is positive.
- Host validation signal, if relevant: `vlm_warmup_ready_ok=1`.
- Deterministic proof: `/models/{vlm}/status` shows `ready=true`, `warmup_passed=true`, `last_warmup_latency_ms > 0` before VLM user requests.
- Patch order: fix lifecycle boundary first, then host script proof.
- Status: resolved
- Proof: initial reviewer found readiness fabricated from first successful request.

### R02 — external image dependency

- Issue IDs: P8-B002
- Severity: must fix
- Category: host validation gap
- Root cause: host script uses third-party `raw.githubusercontent.com` image URL.
- Invariant: host gate must use repo-controlled or locally served image input.
- Affected files/boundaries: `mlx-host-validation/scripts/phase_8.sh`.
- Regression guard: script uses checked-in local image fixture or local static file server.
- Host validation signal, if relevant: `vlm_local_image_stream_ok=1` or equivalent local-image proof.
- Deterministic proof: no external network dependency for pass/fail.
- Patch order: replace image source, then rerun host gate.
- Status: resolved
- Proof: initial reviewer found external asset dependency.

### R03 — weak metric proof

- Issue IDs: P8-S001
- Severity: should fix
- Category: observability/telemetry gap
- Root cause: host script checks metric family names and a decode-throughput string, not positive observation counts.
- Invariant: real VLM request must increment TTFT and request-latency histograms, and throughput gauge must be positive.
- Affected files/boundaries: `mlx-host-validation/scripts/phase_8.sh`, `rust/crates/gateway/src/telemetry.rs`.
- Regression guard: assert `mlx_ttft_ms_count > 0`, `mlx_request_latency_ms_count > 0`, `mlx_decode_tokens_per_second > 0` after real VLM traffic.
- Host validation signal, if relevant: `metrics_ok=1` plus positive metric assertions.
- Deterministic proof: metric counts/gauge values parsed structurally from Prometheus output.
- Patch order: strengthen host script assertions after lifecycle/image fix.
- Status: resolved
- Proof: metric families exist, but proof of non-zero observations is missing.

### R04 — local image contract mismatch

- Issue IDs: P8-S002
- Severity: should fix
- Category: contract mismatch
- Root cause: implementation and tests reject local image paths even though phase contract includes local paths.
- Invariant: runtime accepts both local path and HTTPS image sources under the explicit security policy.
- Affected files/boundaries: `PLAN.md` phase 8 text, `python/mlx_worker/vlm_engine.py`, `rust/crates/gateway/src/http.rs`, `python/tests/test_vlm_engine.py`.
- Regression guard: unit tests for local-path acceptance plus HTTPS acceptance.
- Host validation signal, if relevant: local-image request succeeds through public API path.
- Deterministic proof: local path request uses repo-controlled fixture; HTTPS request remains supported.
- Patch order: widen validation boundary, then update host script.
- Status: resolved
- Proof: current script still depends on remote URL, and local paths are rejected.



## Resolved blockers

### R01 — VLM prompt template dropped image token count

- Invariant: prompt builder must forward image count into `mlx_vlm.prompt_utils.apply_chat_template`, or VLM models see 0 image tokens and fail with token/feature mismatch.
- Regression guard: `python/tests/test_vlm_engine.py::test_vlm_engine_forwards_image_count_to_prompt_template`.
- Test/check/validation command: `uv run pytest tests/test_vlm_engine.py tests/test_main.py`
- Host validation signal, if relevant: `vlm_json_response_non_empty=1` and `vlm_status=ready` from `bash mlx-host-validation/scripts/phase_8.sh`.
- Proof summary: `bash mlx-host-validation/scripts/phase_8.sh` completed real gateway startup, loaded VLM model, returned real VLM stream and JSON responses, rejected invalid image URL, and reported VLM metrics.
- Status: resolved

### R02 — non-stream VLM path reroutes into stream path

- Invariant: non-stream VLM completions stay on `mlx_vlm.generate`; cancellation may short-circuit before generation but must not force streaming path.
- Regression guard: `python/tests/test_vlm_engine.py::test_vlm_engine_complete_chat_uses_non_stream_generate_when_cancel_checked`.
- Test/check/validation command: `uv run pytest tests/test_vlm_engine.py tests/test_main.py tests/test_config.py`.
- Host validation signal, if relevant: `vlm_stream_done=1` and `vlm_json_response_non_empty=1` from `bash mlx-host-validation/scripts/phase_8.sh`.
- Proof summary: engine now keeps non-stream path on `mlx_vlm.generate`; host gate returned real VLM stream and JSON completions.
- Status: resolved

### R03 — VLM image cap split between gateway and worker

- Invariant: one source of truth for VLM image cap; gateway and worker use same configured limit.
- Regression guard: `python/tests/test_vlm_engine.py::test_vlm_engine_honors_configured_image_cap` and `python/tests/test_main.py::test_main_default_vlm_engine_lazy_construction`.
- Test/check/validation command: `uv run pytest tests/test_config.py tests/test_main.py tests/test_vlm_engine.py`.
- Host validation signal, if relevant: gateway/worker bootstrap now threads `MLX_RUNTIME_MAX_VLM_IMAGES` from config.
- Proof summary: gateway passes configured image cap into worker bootstrap; worker reads env var and engine honors instance cap.
- Status: resolved

### R04 — VLM timing metrics never leave worker boundary

- Invariant: successful VLM request updates gateway metrics from real worker timings, not just metric names.
- Regression guard: `python/tests/test_ipc.py::IpcEncodingTests::test_vlm_response_round_trip_preserves_timing_fields` and host metric assertions.
- Test/check/validation command: `uv run pytest tests/test_ipc.py tests/test_vlm_engine.py`.
- Host validation signal, if relevant: `vlm_stream_metrics_ok=1` and `vlm_metrics_ok=1` only after asserting metric values > 0.
- Proof summary: worker emits timing fields, gateway records them, host gate checks positive VLM metrics on both stream and JSON runs.
- Status: resolved

## Pre-review self-check

- Latest reviewer call happened after final code/test/validation changes: yes
- No open blockers: yes
- All resolved blockers still have valid deterministic proof: yes
- Tests/lints/host validation passed: yes
- `cargo fmt --check` passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed.
- `cargo test --workspace --all-features` passed.
- `uv run ruff format --check .` passed.
- `uv run ruff check .` passed.
- `uv run pytest` passed.
- `bash mlx-host-validation/scripts/phase_8.sh` passed with real gateway startup, VLM load, stream and JSON responses, lifecycle checks, and metrics checks.

## Final proof

- `machine=arm64`
- `mlx_import_ok=1`
- `mlx_lm_import_ok=1`
- `mlx_vlm_import_ok=1`
- `health_response=healthy`
- `text_response_non_empty=1`
- `vlm_stream_chunk_ok=1`
- `vlm_stream_done=1`
- `vlm_stream_metrics_ok=1`
- `vlm_stream_response_non_empty=1`
- `vlm_json_response_non_empty=1`
- `vlm_status=ready`
- `vlm_ready_response=ready`
- `invalid_image_rejected=1`
- `metrics_ok=1`
- `vlm_metrics_ok=1`
