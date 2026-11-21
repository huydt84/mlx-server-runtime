---
layout: post
title: "Measuring Inference Honestly"
author: "MLX Air Team"
date: 2026-11-22
tags: [learning, benchmarking, profiling, methodology]
excerpt: "A faster number is easy to produce and hard to trust. The rules behind mlx-air bench: primary metrics, trial-level statistics, calibration, order alternation, cold-cache validation, no retries, and profiling that never touches a benchmark row."
---

**Table of contents**

- [Ways a benchmark lies](#ways-a-benchmark-lies)
- [One question per workload](#one-question-per-workload)
- [Suites, focuses, and model roles](#suites-focuses-and-model-roles)
- [Statistics at the right level](#statistics-at-the-right-level)
- [Calibrate before you conclude](#calibrate-before-you-conclude)
- [Controlling the environment you cannot control](#controlling-the-environment-you-cannot-control)
- [Cache state is part of the workload](#cache-state-is-part-of-the-workload)
- [Never retry](#never-retry)
- [Profiling lives in a separate process](#profiling-lives-in-a-separate-process)
- [The pipeline profiler](#the-pipeline-profiler)
- [Comparing two versions](#comparing-two-versions)
- [Reporting](#reporting)
- [Acknowledgments](#acknowledgments)

Every number in this blog series comes from one tool, `mlx-air bench`, and
one configuration file, `benchmarks/config/default.toml`. This post explains
the rules behind them. They are useful for any inference engine, and most of
them were learned by getting a number wrong first.

## Ways a benchmark lies

A non-exhaustive list we have personally hit:

- **The baseline was broken.** The `v1` backend's exact-prompt cache hit
  passed an empty suffix to `mlx-lm`, so "native vs. v1 on cache hits"
  compared against a bug.
- **The workload never exercised the problem.** A host script published
  "first delta seen" only after a stream finished, so its second stream never
  overlapped the first, and a decode-padding bug that made mixed steps 36x
  slower went unseen.
- **The instrument changed the result.** The model-graph profiler forces
  extra evaluation at layer boundaries; profiled timings are not normal
  timings.
- **The mean hid the tail.** Scheduler bugs that delayed short streams'
  *second* token were invisible in mean TTFT, which they improved.
- **Retries hid failures** and duplicated work (below).

## One question per workload

Every workload names a single **primary metric**: the one number that
answers the question the workload asks. Other metrics are reported, but
they explain the primary metric; they do not compete with it.

| Workload | Shape | Primary metric | Question |
| --- | --- | --- | --- |
| `interactive_stream_c1` | 32-token prompt, 32 out, streaming | TTFT | How fast does a chat feel to start? |
| `cold_prefill_2k_c1` | 2k-token prompt, 16 out | TTFT | How fast is prefill? |
| `decode_128_c1` | Short prompt, 128 out | TPOT | How fast is single-stream decode? |
| `steady_short_c4` / `_c8` | Short prompts, 4 or 8 in flight | Aggregate tokens/s | How well does batching scale? |
| `mixed_c8` | 512-token prompts, 8 in flight | Aggregate tokens/s | Prefill and decode together |
| `shared_prefix_warm_c16` | 8 groups x 8 suffixes, warm prefix | TTFT | Is the prefix cache paying off? |
| `cache_pressure_c4` | 2k-token prompts | End-to-end latency | Behavior near KV capacity |

*Table 1: A selection of workloads. TTFT and TPOT: lower is better.
Throughput: higher is better.*

## Suites, focuses, and model roles

A **suite** decides breadth and repetition (`smoke` 2 rounds x 1 trial,
`optimization` 2 x 2, `release` 2 x 3, `exhaustive` 2 x 4). A **focus**
(`gateway`, `prefill`, `decode`, `scheduler`, `cache`) narrows a suite to the
workloads and models relevant to one subsystem.

Models are assigned **roles**, because one checkpoint is a smoke test, not a
conclusion:

| Role | Checkpoint | Why |
| --- | --- | --- |
| `primary` | Qwen3-4B-Instruct-2507 4-bit | The main dense target |
| `dense` | Qwen2.5-7B-Instruct 4-bit | A larger dense model, different GQA ratio |
| `moe` | LFM2.5-8B-A1B 4-bit | Mixture-of-experts and hybrid cache |
| `runtime_control` | gemma-3-270m-it QAT 8-bit | So small that runtime overhead dominates |
| `hybrid_binary` / `hybrid_ternary` | Qwen3.5 27B 1-bit / ternary | Gated DeltaNet at sub-2-bit |

*Table 2: Model roles.* The `runtime_control` model deserves a note: a 270M
model's forward pass is so cheap that its numbers are mostly gateway, IPC,
scheduler, and Python overhead. If an "engine" optimization does not move
it, the optimization was not in the engine.

The rule: a performance claim uses every model role the selected suite and
focus require.

## Statistics at the right level

Requests within one trial are not independent: they share a server process,
a thermal state, and a cache. So `mlx-air bench` treats **each trial's mean
as one observation** and reports, per row, the mean, median, min-max, and
coefficient of variation (CV) *across trials*. It also reports a
request-level p95 that pools all requests, to show the tail, but the
headline statistics are trial-level.

## Calibrate before you conclude

How large must a difference be before it is real? That depends on the
run-to-run noise of each row on this machine, which you have to measure:

```bash
mlx-air bench calibrate --suite optimization --focus scheduler --repetitions 5
```

Calibration repeats the same run and reports each primary metric's CV across
repetitions. A row whose CV exceeds `max_cv_percent` (5% by default) is
marked **unstable**, and no conclusion may be drawn from it.

Unstable rows happen. In run `decode-f5524d2`, the MoE model's overlap
`decode_128_c1` TPOT had a CV of 22% across trials, and the primary model's
serial row 16%, where most decode rows sit under 1%. Those runs said "this
measurement is not repeatable," not "this change is slower." In the next
decode run (`decode-96bf4d2`), the same two rows came in at 0.3% and 4.3%,
both under the threshold.

## Controlling the environment you cannot control

A laptop is a noisy benchmark host: thermal throttling, background indexing,
power source. The harness cannot remove the noise, but it can stop the noise
from becoming bias:

- **Alternate order.** Each round starts one server per (model, runtime
  configuration) pair, and the order alternates between rounds and models.
  If the machine heats up during a run, both configurations see both the
  cool and the hot half.
- **Warm up with different prompts.** Before measuring, each workload runs
  once with distinct warmup prompts of the same shape. That compiles kernels
  and fills allocator caches without pre-filling the prefix cache with the
  measured prompts. Warmup requests are recorded but never measured.
- **Record the host.** `results.json` records the OS version and the power
  state, so a run on battery is visible after the fact.
- **Stop at the first failure.** A failed request ends the run with a named
  stage (`readiness`, `measurement`, `validation`) and exit code 50. A
  "mean latency" over requests that partly failed is not a measurement.

## Cache state is part of the workload

A prefix-cache hit can make a request ten times faster, so the cache state
must be declared and enforced, not hoped for:

- **`cold`** workloads clear the prefix cache before each trial and **fail
  validation** if any request reuses cached tokens. A model whose chat template
  puts the same tokens before every prompt declares an allowance: Qwen2.5's
  default system prompt fills exactly one 16-token KV page, so its
  `cold_cache_allowance_tokens` is 16.
- **`warm-prefix`** workloads send each shared prefix once before every
  trial and fail validation unless every request reuses part of its prompt.
- **`primed`** workloads send the prefixes but accept whatever the backend
  reuses, so a backend with no prefix cache (a hybrid model, or `v1` with its
  cache off) can be compared on identical traffic.

## Never retry

The load generator **never retries a request**
([ADR-0009](../adr/0009-benchmark-measurement-integrity.md)). A transport
failure after the request was written is ambiguous: the server may have run
it. A retry would duplicate inference work and corrupt request counts, cache
hit counts, and runtime counters, and it would hide exactly the kind of
failure (the macOS socket bug in [post 10](2026-11-22-gateway-sse-cancellation.md))
that the benchmark should surface.

## Profiling lives in a separate process

Performance rows are always collected with profiling **off**. Diagnostics
run afterwards, in separate server processes:

```bash
mlx-air bench run --suite optimization --profile representative
mlx-air bench diagnose --result artifacts/benchmark/<run>/results.json --workload-family decode
```

Diagnostic timings never become benchmark rows. This is not pedantry. The
model-graph profiler (`MLX_RUNTIME_NATIVE_GRAPH_PROFILE=1`) has to force
evaluation at module boundaries to time them, which destroys the very
asynchrony that [post 6](2026-11-22-overlap-scheduling-mlx-async-eval.md)
works to create. Profiles tell you *where to look*; benchmarks tell you
*whether it helped*.

Profiles are still enormously useful. In the radix-cache investigation
([post 8](2026-11-22-prefix-caching-radix-vs-block-hash.md)), a graph
profile showed model layers taking only 6 to 8% of the regressed workload's
wall time. That ruled out every kernel and pointed at host-side cache work,
where the fix was.

## The pipeline profiler

`mlx-air profile run` captures one request-correlated timeline across the
client, the Rust gateway, worker transport, runtime, scheduler, executor,
cache, MLX synchronization, detokenization, and streaming, and renders a
self-contained `profile.html`.

Joining events from three processes into one timeline needs a few rules
([ADR-0010](../adr/0010-pipeline-profile-timing.md)):

- **One clock.** The profiling client stamps requests with
  `time.perf_counter_ns()`, the same clock the worker uses, so no clock
  translation is needed.
- **Anchor spans at their start.** A prefill span is recorded when its step
  finishes, so its start is "now minus step time."
- **Keep envelopes out of the breakdown.** `runtime/terminal` and
  `gateway/http_round_trip` cover a whole request. They are reported as
  request latency, not as stages, or they would dwarf the stages inside
  them.
- **Pair overlapped events by request, not by time.** With chaining, the next
  synchronization in time usually belongs to the *previous* step.

`mlx-air profile compare before/ after/` prints per-stage deltas, labeled
`better (lower)` or worse, but it points at where to look. The conclusion
still comes from `mlx-air bench run`.

## Comparing two versions

The procedure for "is version B faster than version A?":

1. Prepare each version's distribution independently.
2. Run the same suite, focus, and configuration file against both.
3. Require matching workload inputs and output/token parity. If the outputs
   differ, you are comparing different work.
4. Judge each workload by its primary metric, and discard unstable rows.
5. State exactly which suite, focus, models, and workloads the verdict
   covers.

## Reporting

Finally, how a number is written down:

- Absolute values first, with units and direction: "TPOT 11.04 ms (lower is
  better)", not "24% faster".
- A signed percentage always names its baseline, candidate, metric, unit,
  and whether the change is better or worse.
- Correctness and parity, end-to-end latency, TTFT, TPOT, throughput, CPU
  overhead, graph profiles, and Metal idle gaps are separate results. They
  are not blended into one score.
- Regressions are reported as plainly as improvements. Native-mlx does not
  have to beat `v1` to be correct, and pretending it does would make every
  other number less trustworthy.

## Acknowledgments

The trial-level statistics and calibration approach owe a lot to the
benchmarking practices of the
[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
rules and the vLLM benchmark scripts, adapted to a single noisy laptop.
