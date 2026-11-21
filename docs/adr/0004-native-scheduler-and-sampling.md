# 0004. Native scheduler admission, chaining, and sampling

- Status: Accepted
- Date: 2026-11-22

## Context

The native-mlx scheduler runs continuous batching over a paged KV pool with an
overlap pipeline: while step N runs on the GPU, step N+1 is already built and
queued. KV pages are finite, so admission has to stay deadlock-free and fair.

## Decision

- **FCFS order is the arrival order of an `OrderedDict`.** The FCFS policy
  returns states in insertion order and does not build a sorted copy on each
  decode tick.
- **Admission reserves pages for the whole request up front:**
  `ceil((prompt + max_tokens) / page_size) - reused_prefix_pages + 1`. The
  extra page covers a copy-on-write split of a shared last page.
- **A blocked head-of-line request blocks later arrivals.** When the oldest
  waiting request does not fit and requests are still running, later arrivals
  wait behind it, so short requests cannot starve a long one. Reclaiming
  prefix-cache pages can evict the entry the request matched, so the prefix
  cache is probed again after a reclaim.
- **When every prefill candidate is waiting for pages, decode keeps running**
  so running requests finish and free their pages.
- **Decode and prefill run as separate steps.** Batch growth shows up on the
  decode batch gauge, not the mixed gauge.
- **Chained decode steps can run one token past a terminal token.** The next
  step is dispatched before the current one resolves, so its token is thrown
  away when the current step produces end-of-sequence. When `max_tokens` is
  known ahead of time, no step runs past it.
- **Chained decode inputs read the previous token from the device.** The
  rows carry a placeholder host token id ([#6](https://github.com/huydt84/mlx-server-runtime/issues/6)).
- **Step failures are reported when the step resolves.** A failed launch or
  commit is swallowed at the call site because resolving the step emits the
  failure for each of its rows.
- **Model, cache, and async evaluation share the owner thread's default GPU
  stream.** A separate stream adds a command-queue handoff and enables no
  extra overlap for a single-model pipeline.
- **Transcription requests run to completion inline** in the worker loop, and
  text scheduling resumes after. The worker reads up to 1 MiB per socket read
  because audio uploads arrive as multi-megabyte lines.
- **Startup warmup decodes four tokens per request**, so the overlap pipeline
  runs more than one decode step and compiles every kernel it will use.
- **Sampling:** top-p drops a token once the more likely tokens already reach
  `top_p`, and always keeps the most likely token. Per-position random keys
  use SplitMix64-style mixing so neighboring positions get unrelated keys.

## Consequences

- Head-of-line blocking can leave pages idle while a large request waits.
- Each chained terminal step wastes up to one decode step of GPU time.

## Code

- `python/mlx_worker/native_mlx/scheduler.py` — `_BasePolicy.order` (FCFS),
  `_admission_pages`, `_select_prefill`, `_select_work`,
  `_advance_step_state`, `_chained_work`, `_tick_overlap`,
  `_dispatch_chained`.
- `python/mlx_worker/native_mlx/executor.py` — `MlxOverlapGenerationExecutor`
  constructor.
- `python/mlx_worker/native_mlx/worker.py` — `run_native_worker`,
  `_RECV_BYTES`.
- `python/mlx_worker/native_mlx/runtime.py` — `_WARMUP_DECODE_TOKENS`.
- `python/mlx_worker/native_mlx/sampling.py` — `_sample_row`, `_row_seed`.
