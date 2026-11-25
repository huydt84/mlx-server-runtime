---
layout: post
title: "Prefill on Packed Weights: Decode Once, Multiply in 8x8 Blocks"
author: "MLX Air Team"
date: 2026-11-26
tags: [performance, prefill, metal, quantization, kernels]
excerpt: "Inkling-2-Mini's 5.4 GB pack prefilled long prompts at about 200 tokens/s. A kernel that decodes each weight tile once and multiplies it with up to 32 tokens as SIMD-group matrices cut a 3,725-token prompt's time to first token from 17.8 s to 9.7 s, without changing a single generated token on that workload."
---

**Table of contents**

- [The problem](#the-problem)
- [How much work a prompt token is](#how-much-work-a-prompt-token-is)
- [Where the time went](#where-the-time-went)
- [Why the decode kernel is wrong for prefill](#why-the-decode-kernel-is-wrong-for-prefill)
- [First attempt: slower, for two reasons](#first-attempt-slower-for-two-reasons)
- [The kernel](#the-kernel)
- [Small units keep the old loops](#small-units-keep-the-old-loops)
- [Dense projections too](#dense-projections-too)
- [A rotation that computed every value eight times](#a-rotation-that-computed-every-value-eight-times)
- [Is it still the same model?](#is-it-still-the-same-model)
- [Results](#results)
- [What is left](#what-is-left)

The [hybrid models post](2026-11-22-serving-hybrid-models.md) showed how MLX
Air serves weights that never get expanded. Inkling-2-Mini's LookAtMyQuant
packs push that further: every matrix is stored as trellis codes, NVFP4 blocks,
or Lloyd-Max indices, decoded inside the kernel that multiplies it
([ADR-0011](../adr/0011-packed-weight-kernels.md)). Decode was tuned first, and
it shows: one stream generates a token every 31.7 ms. Prefill was not. A long
prompt went in at about 200 tokens/s, so a 3,700-token mail thread waited 18
seconds for its first token.

This post is about that prefill: why the kernel that is right for decode is
wrong for prompts, a first attempt that made things slower, and the kernel that
nearly doubled the speed.

## The problem

Inkling-2-Mini is a looped mixture of experts: 24 physical layers, of which 12
run three times, so a token makes 48 layer visits. Each visit routes the token
to 8 of 128 experts. The 5.4 GB pack mixes formats per projection and per
128-row group of an expert: `rtr1` and `ht2` trellis codes for most experts,
`rtr4`, `ht3`, and `lm6` for the attention projections.

All measurements here ran on a 16-core M4 Pro GPU with 24 GB of memory.

## How much work a prompt token is

One prompt token costs, in multiply-adds (MACs):

| Part | Shape per visit | MACs per token |
| --- | --- | ---: |
| Expert gate-up | 8 experts × 2304 × 2304 | 2.04 G |
| Expert down | 8 experts × 2304 × 1152 | 1.02 G |
| Q, K, V, relative, and gate projections | 8576 × 2304 | 0.95 G |
| Output projection | 2304 × 3072 | 0.34 G |
| **Total over 48 visits** | | **4.35 G** |

*Table 1: Matrix work per prompt token.*

A raw loop of 8x8 SIMD-group matrix products on this GPU sustains about
2.8 TMAC/s (more on that below). At that rate the matrix work alone would allow
roughly 640 tokens/s. At 200 tokens/s we were using about a third of the GPU.

## Where the time went

A per-operation profile of one 256-token step (each operation synchronized
separately, so small operations are inflated by roughly 0.15 ms each; this is a
diagnostic, not a benchmark row):

| Operation | Before (ms/step) | Calls |
| --- | ---: | ---: |
| Expert gate-up products | 577.7 | 48 |
| Expert down products | 292.1 | 48 |
| Q/K/V/relative/gate products | 149.2 | 48 |
| Input rotations | 94.1 | 193 |
| Attention | 58.6 | 48 |
| Output products | 54.3 | 48 |
| Everything else | 173 | — |

*Table 2: Profile of a 256-token prefill step before the change, 1,399 ms in
total.*

Expert products were 62% of the step, at about 0.9 TMAC/s.

## Why the decode kernel is wrong for prefill

The packed kernel was built for steps of one to eight tokens. One threadgroup
takes one *unit* (up to eight inputs that use the same rows) and one 128-row
group. Each lane decodes 16 columns of a row and takes dot products with every
input of the unit:

```text
for each 16-column chunk:
    w = decode(row, chunk)        # trellis lookups, ~8 ops per weight
    for each of up to 8 inputs:
        acc[input] += dot(w, x[input][chunk])
```

For decode that is ideal: a token's eight experts are distinct, so there is
nothing to share, and the kernel never writes an expanded weight to memory.
For a prompt it has two costs.

**Rows are decoded again for every eight inputs.** A 256-token step routes
2,048 slots to about 106 experts, 19 slots each on average and up to 166 for the
busiest. In units of eight that is 308 decodes of an expert's rows. In units of
32 it would be 135.

**The dot products are load-bound.** Each input value a lane loads feeds two
multiply-adds (two rows at a time). The GPU can issue far more arithmetic than
that.

So the plan was obvious: bigger units, and multiply with
`simdgroup_matrix`, which does an 8x8x8 product in one instruction sequence
with the data shared across the SIMD group's 32 lanes.

## First attempt: slower, for two reasons

The first version decoded a 32-row by 16-column tile into threadgroup memory
and multiplied it with 32 tokens. It ran at 0.75 TMAC/s, slower than what it
replaced, and returned `NaN`.

**The fragment layout was wrong.** To feed values into a `simdgroup_matrix`
without going through memory, you need to know which two elements of the 8x8
block each lane holds. The obvious guess, row `lane / 4` and columns
`2 * (lane % 4)`, is not what the hardware does. MLX's own matmul (`BaseMMAFrag`
in `steel/gemm/mma.h`) spells it out: lane `l` holds row
`(l / 16 % 2) * 4 + (l / 2) % 4` and columns `(l / 8 % 2) * 4 + (l % 2) * 2` and
the next one, accessed through `thread_elements()`. Casting a `float2` to a
matrix compiles, and reads garbage.

With the layout fixed, results matched the old kernel to 8.5e-7, and the
kernel ran at 1.05 TMAC/s. Still barely faster.

**There is a register cliff.** A synthetic loop of matrix products, with no
decoding and no device loads, showed why:

| Accumulators per SIMD group (8x8 float) | TMAC/s |
| --- | ---: |
| 4 × 4 = 16 | 1.61 |
| 2 × 4 = 8 | 2.74 |
| 4 × 2 = 8 | 2.75 |
| 2 × 2 = 4 | 2.58 |
| 2 × 4 = 8, half precision | 2.87 |
| 8 × 4 = 32, half precision | 0.59 |

*Table 3: Raw `simdgroup_multiply_accumulate` throughput on the M4 Pro GPU
against the number of live accumulator blocks.*

Sixteen accumulators, the 32-row by 32-token tile, falls off a cliff. Eight
reach 98% of what this loop can do at all. An earlier tiling attempt,
which [ADR-0011](../adr/0011-packed-weight-kernels.md) records as measured
slower, had used the same guessed layout.

## The kernel

The final shape keeps eight accumulators: **16 rows × up to 32 tokens per
SIMD group, 32 columns per tile.**

```text
for each 32-column tile:
    lane l decodes chunk l / 16 of row l % 16   # one chunk per lane
    store the 16 x 32 tile to threadgroup memory
    for each 8-column block of the tile:
        load 2 weight blocks (16 rows)
        for each 8-token block (up to 4):
            load the input block straight from device memory
            2 multiply-accumulates
```

A few details matter:

- **Every lane decodes exactly one 16-column chunk per tile**, the same
  decoding code the decode kernel uses, so every format works unchanged.
- **The tile keeps the format's own type.** Trellis weights are halves.
  NVFP4's block scale is folded in while storing, and an E2M1 value times an
  E4M3 scale has at most 7 significant bits, so the product is exact in half
  precision. Lloyd-Max levels are floats and stay floats.
- **Inputs are read directly in fragment layout**, so only weights go through
  threadgroup memory, and a SIMD group only waits on itself
  (`simdgroup_barrier`, not a threadgroup barrier).
- **Accumulation is in float**, like MLX's matmul.

On layer 10's real routing, the expert products became:

| Step | Products | Before (ms) | After (ms) | Effective TMAC/s after |
| --- | --- | ---: | ---: | ---: |
| 256 tokens | gate-up | 11.86 | 5.58 | 1.95 |
| 256 tokens | down | 6.12 | 2.78 | 1.95 |
| 512 tokens | gate-up | 23.26 | 9.34 | 2.33 |
| 512 tokens | down | 11.74 | 4.82 | 2.26 |

*Table 4: Expert product microbenchmark on one layer with recorded routing.
Lower ms is better.*

Effective TMAC/s counts only real tokens. Tokens come in blocks of eight, so
an expert with 19 slots multiplies 24. Counting that padding, the kernel runs
at about 90% of Table 3's ceiling.

## Small units keep the old loops

An 8x8 block always multiplies eight tokens. When an expert gets one or two
slots, the old per-lane loops win:

| Slots per expert | Old (ms) | Blocked (ms) |
| ---: | ---: | ---: |
| 1 | 2.66 | 3.45 |
| 2 | 2.77 | 3.46 |
| 4 | 3.42 | 3.58 |
| 5 | 5.95 | 3.65 |
| 8 | 5.95 | 3.91 |
| 16 | 11.53 | 5.01 |
| 32 | 23.07 | 8.24 |

*Table 5: Gate-up products for 128 experts with a fixed number of slots each.
Lower is better.*

The kernel branches per unit: one to four inputs run the old loops, five or
more run the blocked products. Steps of at most eight tokens never reach the
new code. That matters, because [speculative
decoding](../adr/0013-speculative-decoding.md) relies on short steps computing
exactly what one-token steps compute, bit for bit. They still do.

## Dense projections too

Before, a step of 128 or more tokens decoded each dense projection bank into a
float32 copy, about 80 MB per visit for the attention projections, and handed
it to MLX's matmul. The blocked kernel beats that without the copy: the
Q/K/V/relative/gate products went from 3.1 to 2.0 ms per 256-token step.
Peak memory of a single-request prefill fell from 7.66 to 7.40 GB at 256-token
chunks.

## A rotation that computed every value eight times

Packed weights expect their input in a randomized Hadamard basis, so every
product starts with a rotation ([ADR-0011](../adr/0011-packed-weight-kernels.md)).
The expert down products rotate a soft-gated input,
`tanh(g/a)·a·sigmoid(g)·tanh(u/b)·b`. Eight threads share each row, and each
computed the whole row's gate, two `tanh`s and an `exp` per element, eight
times over.

Staging the gated row in threadgroup memory once took that rotation from 0.66
to 0.36 ms at 256 tokens. Staging every rotation made the cheap ones slower
(0.11 to 0.19 ms for the output rotation), because 12 KB of threadgroup memory
per threadgroup lowers occupancy, so only the soft-gated one stages.

## Is it still the same model?

Steps longer than eight tokens already differed from one-token steps in the
last bits, so the question is whether the new kernel is less precise. We
compared each version's logits for the first 384 prompt tokens against the
logits of one-token steps, the bit-exact reference:

| Chunk | Version | Mean KL | Greedy agreement |
| ---: | --- | ---: | ---: |
| 256 | before | 1.16e-2 | 98.2% |
| 256 | after | 5.50e-3 | 98.7% |
| 512 | before | 1.23e-2 | 98.2% |
| 512 | after | 9.63e-3 | 97.9% |

*Table 6: Distance of chunked-prefill logits from one-token-step logits on the
5.4 GB pack. Lower KL is better.*

The new kernel is no further from the reference. Each version differs from the
reference by about as much as the two differ from each other. The host
validation script `native_packed_checkpoints.sh` passed on the 5.4 GB pack:
short steps are still exact, step-shape KL is 0.00017, and speculative
decoding returns the plain tokens.

## Results

Both versions were staged as separate distributions and run with the new
`packed_prefill` suite: 2 rounds × 3 trials, the `packed` runtime
configuration, profiling off, greedy decoding, same configuration file. Both
ran on battery power on the same machine.

| Workload | Metric | Baseline `fe4abf6` | Candidate | CV (base / cand.) |
| --- | --- | ---: | ---: | ---: |
| `cold_prefill_2k_c1` | **TTFT** (ms, lower is better) | 17,820 | 9,718 | 0.49% / 0.73% |
| `interactive_stream_c1` | **TTFT** (ms, lower is better) | 478 | 305 | 6.65% / 11.56% |
| `decode_128_c1` | **TPOT** (ms, lower is better) | 31.73 | 31.63 | 0.06% / 0.10% |

*Table 7: `mlx-air bench run --suite packed_prefill` results. Runs:
`benchmarks/results/packed-prefill-fe4abf6` and
`benchmarks/results/packed-prefill-blocked`.*

- **Long-prompt prefill:** `cold_prefill_2k_c1` sends prompts of 3,725 tokens.
  TTFT fell from 17,820 ms to 9,718 ms, 45% lower (better) than the baseline.
  That is 209 → 383 prompt tokens/s. All 12 outputs are identical between
  versions.
- **Short prompts:** `interactive_stream_c1` TTFT is lower too, but its rows
  vary more than the 5% calibration threshold, so it is not a verdict.
- **Decode:** TPOT is unchanged, as expected. Short steps run the same
  kernels. Eight of the twelve 128-token outputs differ between versions,
  because their 84-token prompts are now prefilled with different rounding, and
  greedy text diverges after enough tokens. So this row is not a parity-matched
  comparison either.

Chunk size matters more now. In a single-request side measurement over 2,048
tokens (our own harness, not a benchmark row):

| Chunk | Before (tokens/s) | After (tokens/s) |
| ---: | ---: | ---: |
| 64 | 144 | 247 |
| 256 | 198 | 376 |
| 512 | 206 | 401 |
| 1024 | 211 | 420 |

*Table 8: Prefill throughput by chunk size. Higher is better.*

Larger chunks give each expert more tokens per unit. The default chunk stays
at 256: a 512-token chunk would hold up running decodes for about 1.3 s per
step instead of 0.7 s.

## What is left

The profile of a 256-token step after the change:

| Operation | Before (ms/step) | After (ms/step) |
| --- | ---: | ---: |
| Expert gate-up products | 577.7 | 259.2 |
| Expert down products | 292.1 | 133.8 |
| Q/K/V/relative/gate products | 149.2 | 98.3 |
| Input rotations | 94.1 | 73.3 |
| Output products | 54.3 | 40.3 |

*Table 9: The same diagnostic profile as Table 2, after the change.*

Matrix products are still about 80% of the step, now close to the 8x8 block
ceiling. What remains:

- **Padding.** About a fifth of expert work at 256-token steps multiplies
  padding, because token blocks are eight wide.
- **The projection rotation** (0.37 ms per visit) is latency-bound: one
  threadgroup per token walks a 72-step loop.
- **The 12.8 GB pack** shares every kernel but was not benchmarked here. On a
  24 GB machine it needs the other GPU work stopped first.

The decisions are recorded in
[ADR-0014](../adr/0014-blocked-packed-products.md). The kernel is
`packed_rows_blocked` in
`python/mlx_worker/native_mlx/kernels/packed_matmul.metal`.
