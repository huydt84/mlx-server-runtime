# 0009. Benchmark measurement integrity

- Status: Accepted
- Date: 2026-11-22

## Context

Benchmark totals (requests, tokens, cache hits, runtime counters) and timeline
analyses are used to draw performance conclusions. Small harness shortcuts can
make them wrong without any visible error.

## Decision

- **The load generator never retries a request.** A transport failure after
  the request was written is ambiguous: the server may already have run it.
  A retry would duplicate measured inference work and corrupt request, cache,
  and runtime-counter totals.
- **Synthetic mail-assistant prompts are sized from words, not tokens.** Each
  body is sized at about 1.3 tokens per word, plus 64 tokens for the header,
  wrapper, language line, and chat template. Workload results record the
  actual token counts.
- **Overlap timelines pair each `synchronize_eval` with the preceding
  `async_dispatch` of the same request.** A step's events are recorded
  together when it resolves. Pairing by timestamp would be wrong because the
  next step is often dispatched before the previous one synchronizes.
- **Pipeline-profile traces are appended without re-serializing.** The trace
  file is `json.dumps({"traceEvents": [...]}, indent=2)` split into a fixed
  head, one block per event, and a fixed tail, so new events are spliced in
  before the tail.
- **Metal refuses to overwrite an existing capture**, so profiling runs delete
  the previous capture directory first.
- **Host checks that replace the staged gateway with a proxy run last.** The
  `cli_bench.sh` calibration and failed-trial checks inject a known delay or
  request failures, so no other check can run after them.

## Code

- `python/mlx_benchmark/loadgen.py` — `post_json`.
- `python/mlx_benchmark/mail_assistant.py` — `_ENVELOPE_TOKENS`,
  `_TOKENS_PER_WORD`.
- `python/mlx_worker/native_mlx/pipeline_profile.py` — `_TRACE_HEAD`,
  `_TRACE_TAIL`.
- `mlx-host-validation/scripts/python/overlap_benchmark.py` —
  `analyze_timeline`.
