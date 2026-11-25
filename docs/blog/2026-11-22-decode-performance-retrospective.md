---
layout: post
title: "Decode Performance Retrospective: From 18.9 ms to 5.8 ms per Token"
author: "MLX Air Team"
date: 2026-11-22
tags: [performance, retrospective, decode, benchmarks]
excerpt: "Four commits, four benchmark runs, two models. A change-by-change account of where native-mlx decode time went, including the rows we could not draw conclusions from and the changes that did nothing measurable."
---

**Table of contents**

- [The setup](#the-setup)
- [The headline](#the-headline)
- [Before the baseline](#before-the-baseline)
- [Change 1: read attention straight from the pages](#change-1-read-attention-straight-from-the-pages)
- [Change 2: chain decode steps on the device](#change-2-chain-decode-steps-on-the-device)
- [Change 3: overlap prefill too](#change-3-overlap-prefill-too)
- [Change 4: close the short-context gap](#change-4-close-the-short-context-gap)
- [Throughput under load](#throughput-under-load)
- [What did not move](#what-did-not-move)
- [What we learned](#what-we-learned)
- [What is next](#what-is-next)
- [Acknowledgments](#acknowledgments)

The other posts in this series explain techniques one at a time. This one
puts them in order and shows what each bought, using only result files that
are checked into `benchmarks/results/`.

## The setup

All runs use `mlx-air bench run --suite optimization --focus decode` with the
unchanged `benchmarks/config/default.toml`: 2 rounds x 2 trials per row,
runtime order alternating between rounds, profiling off, greedy decoding.
All ran on the same Apple Silicon laptop on AC power. The result files record
the OS and power state; they do not record the chip model.

Two model roles:

- **primary**: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (dense, 36 layers,
  8 KV heads);
- **moe**: `mlx-community/LFM2.5-8B-A1B-MLX-4bit` (mixture-of-experts, about
  1B active parameters, 6 attention layers out of 24).

Two workloads:

- `decode_128_c1`: one stream, short prompt, 128 output tokens. Primary
  metric **TPOT** (lower is better).
- `mixed_c8`: 512-token prompts, 8 in flight, 32 output tokens. Primary metric
  **aggregate output tokens/s** (higher is better).

| Run ID | Commit | Change |
| --- | --- | --- |
| `decode-f70afde` | `f70afde` | Baseline: cheaper paged attention masks |
| `decode-1793318` | `1793318` | + paged decode kernel (`500de9e`) + decode chaining |
| `decode-f5524d2` | `f5524d2` | + prefill overlap, overlap default (`71f01a4`) |
| `decode-96bf4d2` | `96bf4d2` | + smaller key splits, slice-copy KV writes |

*Table 1: The four decode runs.*

## The headline

`decode_128_c1` mean TPOT, lower is better:

| Run | primary, serial | primary, overlap | moe, serial | moe, overlap |
| --- | ---: | ---: | ---: | ---: |
| `decode-f70afde` | 14.62 ms | 14.52 ms | 18.89 ms | 18.95 ms |
| `decode-1793318` | 13.13 ms | 11.40 ms | 7.63 ms | 5.88 ms |
| `decode-f5524d2` | 14.39 ms (unstable) | 11.39 ms | 7.58 ms | 6.61 ms (unstable) |
| `decode-96bf4d2` | 13.40 ms | **11.04 ms** | 7.42 ms | **5.80 ms** |

*Table 2: "Unstable" marks rows whose coefficient of variation across trials
exceeded the 5% calibration threshold (16.5% and 22.0%). No conclusion is
drawn from them.*

In single-stream throughput terms (higher is better), the primary model went
from 61.5 to 78.0 output tokens/s in overlap mode, and the MoE model from
48.7 to 134.3 tokens/s.

Baseline to today, overlap mode, which is the default:

- primary: 14.52 ms -> 11.04 ms TPOT, 3.48 ms lower (24% lower, better);
- moe: 18.95 ms -> 5.80 ms TPOT, 13.15 ms lower (69% lower, better).

## Before the baseline

The baseline is not naive. Several changes before `f70afde` are why it is a
fair starting point:

- **Logits only for the sampled token** (`c942ba0`). Every model projected
  every prefill position onto the vocabulary; now each row projects only its
  last token ([post 7](2026-11-22-on-device-reproducible-sampling.md)).
- **Never pad decode rows to a prefill chunk** (`8816786`). Eight-request
  mixed steps went from about 3.6 s to about 0.1 s
  ([post 5](2026-11-22-continuous-batching-chunked-prefill.md)).
- **Cheaper attention masks** (`f70afde` itself). Each layer rebuilt a
  float32 additive mask and gathered every row from pages. Now a single row
  uses MLX's built-in causal mask, equal-length decode rows use no mask, other
  steps build one boolean mask shared by all layers, and a single row with no
  cached prefix attends to its new K/V directly.

At the baseline, `serial` and `overlap` were within 0.1 ms of each other on
both models. The overlap executor existed but had nothing useful to overlap.

## Change 1: read attention straight from the pages

`500de9e` replaced gather + SDPA with the paged decode Metal kernel for
decode steps ([post 4](2026-11-22-paged-decode-attention-metal.md)). The same
commit stopped the KV commit path from rescanning every page with
list-membership tests, about 1.2 ms of CPU per step at 910 pages
([post 3](2026-11-22-paged-kv-cache-unified-memory.md)).

Serial mode isolates it from the scheduler change in the same run:

| Model | serial TPOT before | after | Change |
| --- | ---: | ---: | --- |
| primary | 14.62 ms | 13.13 ms | 1.49 ms lower (better) |
| moe | 18.89 ms | 7.63 ms | 11.26 ms lower (better) |

*Table 3: `decode-f70afde` vs. `decode-1793318`, serial rows.*

The MoE model's gain is far larger than the dense model's, and we want to be
careful about why, because these two parts landed together and we did not
measure them separately. Our reading: LFM2.5-8B-A1B does very little GPU work
per token (about 1B active parameters), so host-side per-step costs (the page
rescan, and building per-row gather, pad, and stack operations for every
attention layer) were a large share of its step. Removing them showed up
almost one-for-one. The dense model's step is dominated by streaming 2 GB of
weights, so the same savings are a smaller share. Testing that reading would
take a profile of each part on its own; it is a hypothesis, not a result.

## Change 2: chain decode steps on the device

`1793318` let the overlap scheduler dispatch decode step N+1, fed by step N's
sampled tokens while they are still on the GPU, before resolving N
([post 6](2026-11-22-overlap-scheduling-mlx-async-eval.md)). Same run, serial
vs. overlap:

| Model | serial | overlap | Change |
| --- | ---: | ---: | --- |
| primary | 13.13 ms | 11.40 ms | 1.73 ms lower (better) |
| moe | 7.63 ms | 5.88 ms | 1.75 ms lower (better) |

*Table 4: `decode-1793318`.*

The absolute saving is the same for both models, which is what removing a
fixed CPU gap per step should look like. Overlap became the default in the
next commit (`71f01a4`).

## Change 3: overlap prefill too

`f5524d2` built every step before the previous one resolved, prefill
included, and fixed the command-queue trap that had added about 7 ms to TTFT
when a decode step was enqueued right behind a long prefill.

For single-stream decode, there is nothing to measure: `decode_128_c1` has
one prefill per request and 128 decode steps. The primary overlap row is
unchanged (11.40 -> 11.39 ms), and two rows in this run were unstable. The
change was aimed at mixed traffic. In the scheduler-focus runs, comparing
`71f01a4` to `f5524d2` in overlap mode (aggregate tokens/s, higher is
better):

| Model | Workload | `71f01a4` | `f5524d2` |
| --- | --- | ---: | ---: |
| primary | `mixed_c8` | 21.23 | 21.91 |
| primary | `steady_short_c8` | 89.22 | 91.21 |
| moe | `mixed_c8` | 37.28 | 38.70 |
| moe | `steady_short_c4` | 141.81 | 144.68 |

*Table 5: Run IDs `scheduler-71f01a4` and `scheduler-f5524d2`. All four
improve by 2 to 4%, with trial CVs of 1 to 4%, so treat these as small,
consistent gains rather than large ones.*

## Change 4: close the short-context gap

`96bf4d2` came from a GPU-bound profile: 128 chained Qwen3-4B decode steps
with a single synchronization, so CPU overhead was out of the picture. The
whole remaining gap to `mlx_lm` was in attention plus the KV write. At 64
keys the paged kernel took 0.48 ms per step against 0.20 ms for MLX SDPA,
because one threadgroup per (row, KV head) left a single short row on 8
threadgroups.

Two fixes: split keys down to 32 per split instead of 256, and write a
step's K/V with one slice copy when its slots are consecutive. GPU-bound step
time, lower is better:

| Keys | native before | native after | `mlx_lm` |
| ---: | ---: | ---: | ---: |
| 64 | 11.34 ms | 11.06 ms | 10.80 ms |
| 256 | 11.77 ms | 11.14 ms | 10.98 ms |
| 1,024 | 11.95 ms | 11.75 ms | 11.83 ms |

*Table 6: From the commit's own GPU-bound measurement, not a `bench run`.*

In the end-to-end benchmark, primary overlap TPOT went from 11.40 ms
(`decode-1793318`) to 11.04 ms (`decode-96bf4d2`), and MoE overlap from 5.88
to 5.80 ms. Short-context native decode is now within about 2.5% of `mlx_lm`
on the GPU, and ahead of it at 1,024 keys, where reading pages in place beats
growing a dense cache.

## Throughput under load

`mixed_c8` aggregate output tokens/s, overlap mode, higher is better:

| Run | primary | moe |
| --- | ---: | ---: |
| `decode-f70afde` | 21.17 | 36.34 |
| `decode-1793318` | 23.58 | 40.54 |
| `decode-f5524d2` | 23.54 | 40.99 |
| `decode-96bf4d2` | 23.70 | 41.06 |

*Table 7: Gains of 12% (primary) and 13% (moe) from baseline, much smaller
than the single-stream gains.*

This is expected. `mixed_c8` sends 512-token prompts with only 32 output
tokens each, so most of its time is prefill, and every change in this
retrospective targeted decode. Prefill is compute-bound and still runs
through gather + SDPA. It is the obvious next target.

## What did not move

Honest accounting includes the null results:

- **Gateway-focus runs** (`gateway-71f01a4` vs. `gateway-f5524d2`): the
  primary model's `short_nonstream_c1` latency was 573.5 ms and 575.1 ms, and
  its `interactive_stream_c1` TTFT 178.0 ms and 178.2 ms. No change, as
  expected for a scheduler-internal commit.
- **The `runtime_control` TTFT** row (gemma-3-270m) moved from 151.5 ms to
  140.1 ms, but with trial CVs of 4.7% and 5.3%. The second is over the 5%
  threshold, so we do not claim it.
- **Prefill overlap on single-stream decode**, as noted above: nothing to see.

One more comparison that is not a change but is worth knowing: in the gateway
runs, overlap mode's `short_nonstream_c1` latency (573.5 ms) is 65 ms lower
than serial mode's (638.8 ms) for the primary model, and 40 ms lower for
`runtime_control` (284.3 vs. 324.1 ms). For short requests, the CPU gap per
step that overlap hides is a large share of the whole request.

## What we learned

1. **Isolate with modes.** Keeping `serial` as a selectable mode turned
   every overlap change into an in-run A/B test on identical builds.
2. **Small models expose host overhead.** The MoE and the 270M control model
   reacted most to CPU-side fixes. If you only benchmark the big model, you
   will underrate them.
3. **GPU-bound microprofiles find kernel gaps; end-to-end runs confirm
   them.** The 0.48 vs. 0.20 ms attention gap was invisible in TPOT until the
   CPU overhead around it was removed.
4. **Unstable rows are information.** Three of the eight primary-metric rows
   in `decode-f5524d2` were unstable (the two in Table 2, plus primary serial
   `mixed_c8` at 12.7%). Discarding them cost nothing; trusting them would
   have suggested a regression that did not exist.

## What is next

- **Prefill.** A paged prefill path that avoids the dense gather, and better
  chunk scheduling for mixed traffic.
- **Microsecond host timers.** Scheduler and cache stage timers now round to
  0 ms; the next profile needs microsecond resolution.
- **Sampling, detokenization, IPC, and SSE** as separately timed stages in
  `mlx-air profile`, since model layers are a minority of wall time in some
  mixed workloads.
- **MLP-oriented kernel work.** Model-graph profiles put MLP as the largest
  model bucket and attention second.

## Acknowledgments

This retrospective exists because the benchmark results were checked in with
their configuration, commit, and raw samples. Every number above can be
regenerated from `benchmarks/results/<run>/results.json`.
