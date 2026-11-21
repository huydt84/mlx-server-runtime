---
layout: post
title: "Zero-Overhead Scheduling with MLX Lazy Evaluation"
author: "MLX Air Team"
date: 2026-11-22
tags: [scheduling, cpu-overhead, mlx, overlap, deep-dive]
excerpt: "A GPU that waits for Python is a GPU that is idle. How MLX Air builds step N+1 while step N runs, feeds it tokens that have not left the device, and why one Metal command-queue detail cost 7 ms of time-to-first-token."
---

**Table of contents**

- [The idle gap](#the-idle-gap)
- [What "zero overhead" means](#what-zero-overhead-means)
- [MLX gives us the pieces](#mlx-gives-us-the-pieces)
- [Prepare, dispatch, resolve](#prepare-dispatch-resolve)
- [Chaining decode on device-resident tokens](#chaining-decode-on-device-resident-tokens)
- [Speculating one token past the end](#speculating-one-token-past-the-end)
- [Overlapping prefill too](#overlapping-prefill-too)
- [The command-queue trap](#the-command-queue-trap)
- [One stream, not two](#one-stream-not-two)
- [How we verify it](#how-we-verify-it)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

Every decode step has a CPU part and a GPU part. The CPU picks rows, builds
input arrays, records the model graph, reads back the sampled tokens,
detokenizes, and sends them to the gateway. The GPU runs the model. In a
naive loop these happen one after the other, and the GPU sits idle during
every CPU part. This post shows how MLX Air removes that idle time.

## The idle gap

A serial loop looks like this:

```python
while running:
    batch = scheduler.select()          # CPU
    logits = model(batch)               # CPU builds the graph
    tokens = sample(logits)
    mx.eval(tokens)                     # GPU runs; CPU waits
    emit(tokens.tolist())               # CPU; GPU waits
```

```text
CPU  |build N|           |read N, emit|build N+1|           |read N+1 ...
GPU  |       |== run N ==|            |         |= run N+1 =|
                          ^^^^^^^^^^^^^^^^^^^^^^^
                          GPU idle every step
```

*Figure 1: A serial loop. The gap is the CPU's time to read results and
build the next graph.*

With a datacenter batch of 256 rows, a 1 ms CPU gap hides behind a 30 ms
step. On a Mac serving one to eight users, a decode step for a 4B model takes
about 11 ms, and the CPU part is not much smaller than the GPU part could be.
At these batch sizes, CPU overhead *is* TPOT.

## What "zero overhead" means

We use SGLang's definition: CPU work still exists, but batch N+1 is prepared
and launched while batch N runs, so the accelerator has effectively no idle
gap. The CPU is never free; it is just never on the critical path.

```text
CPU  |build N|build N+1|read N, emit|build N+2|read N+1, emit|...
GPU  |       |== run N ==|== run N+1 ==|== run N+2 ==|
```

*Figure 2: Overlap. While the GPU runs N, the CPU builds N+1 and then
processes N's results.*

The hard part is the dependency hiding in Figure 2: step N+1's input is the
token step N sampled. You cannot build N+1 from a value you have not read.
Unless you never read it.

## MLX gives us the pieces

MLX is lazily evaluated. Calling the model does not run it; it records a
graph whose leaves are arrays. Three operations matter:

- `mx.eval(x)` runs the graph and blocks until `x` is ready.
- `mx.async_eval(x)` submits the graph to the GPU and returns at once.
- `x.tolist()` / `x.item()` forces evaluation and copies to Python, so it
  blocks.

Because the sampled-token array from step N is just an MLX array, we can use
it *as an input* to step N+1's graph without ever calling `.tolist()`. MLX
records the dependency, and the GPU runs N+1 after N finishes, with no CPU
round trip in between.

## Prepare, dispatch, resolve

The executor exposes the step as three calls instead of one
(`python/mlx_worker/native_mlx/executor.py`):

```text
prepare_batch(batch)        -> PreparedStep   # host-side layout, no MLX work
dispatch_batch(prepared)    -> InFlightStep   # reserve KV, build graph, async_eval
resolve_batch(in_flight)    -> StepResult     # mx.eval, read tokens, commit KV
```

The overlap scheduler (`_tick_overlap`) keeps at most one step in flight.
Each tick:

1. If the in-flight step N can be followed by work that does not need N's
   resolved values, **dispatch N+1 first** (`_dispatch_chained`).
2. **Resolve N**: wait for the GPU, read its tokens, update request state.
3. If nothing was chained, select and dispatch fresh work.
4. Return N's events (tokens, completions) so the worker can detokenize and
   stream them while N+1 runs.

At most two steps are ever outstanding: the one being resolved and the one
just dispatched.

## Chaining decode on device-resident tokens

The common case is decode after decode for the same rows. To dispatch N+1
before N resolves, the scheduler needs three things.

**Inputs.** Each continuing row gets a placeholder host token (`0`) and the
next position. The executor substitutes the real input from step N's
device-side token array:

```python
pending = executor.device_next_tokens(step_n, request_ids)   # stays on the GPU
batch = ExecutionBatch(requests=..., pending_token_ids=pending)
```

**KV cache.** Step N+1 must attend to the K/V that step N is still writing.
`commit_dispatched` publishes N's cache append (new lengths, new pages)
*without* synchronizing. N+1 then reserves slots after N's, and its attention
reads N's K/V through the MLX graph, so the GPU orders the write before the
read.

**Rows that are known to stop.** A row that was cancelled, asked to finish, or
will reach `max_tokens` after N is left out of N+1. When `max_tokens` is known
ahead of time, no step ever runs past it.

## Speculating one token past the end

One thing the scheduler *cannot* know before N resolves is whether N sampled
an end-of-sequence token. So a chained N+1 runs every continuing row,
including a row that is about to stop. When N resolves and shows EOS, N+1's
token for that row is thrown away.

This costs at most one decode step of GPU time per finished request, and the
discarded token never reaches the client. Its K/V is never published to the
prefix cache, because decode K/V is not published at all; the request's
cache is simply released. [ADR-0004](../adr/0004-native-scheduler-and-sampling.md)
records the trade-off.

## Overlapping prefill too

Decode chaining was the first step. The second made *every* step overlap,
including prefill (`f5524d2`):

- **Prefill after decode.** A pending prefill chunk does not depend on the
  decode step's tokens at all, so it is built before the decode resolves.
- **Decode after prefill.** If a prefill chunk completes a prompt, that
  row's first token is still on the device; the following decode step reads
  it from there. Rows that were already decoding use their resolved tokens.
- **Prefill after prefill** for the same request still waits, because the
  next chunk's cache length depends on the previous chunk's commit.

Decode and prefill still alternate exactly as in
[post 5](2026-11-22-continuous-batching-chunked-prefill.md). A prefill step
with no decode work after it is resolved before the next step is selected.

## The command-queue trap

The first version of prefill overlap enqueued every chained step at once.
It made TTFT worse by about **7 ms**.

The reason is a Metal detail. Enqueueing work (`mx.async_eval`) is not free
when the command queue is full: the call blocks until there is room. A long
prefill step fills the queue. Enqueueing the following decode step right
behind it blocked the CPU for most of the prefill's GPU time, and that CPU
time was exactly when it should have been reading the *previous* step's
tokens and sending the new request's first token to the client.

The fix separates building a step from enqueueing it
(`dispatch_batch(prepared, launch=False)`):

- A **decode step after a decode step** is built and enqueued at once.
  Decode steps are short and do not fill the queue.
- **Any other chained step** is built before N resolves but enqueued at the
  start of the next tick, *after* N's events have been returned.

So the GPU still never waits for graph construction, and the CPU never blocks
behind a long prefill while holding tokens the client is waiting for.

A smaller cost of the design: a request that arrives while N+1 is already
queued starts after N+1. Overlap trades up to one step of TTFT for no idle
GPU time between steps.

## One stream, not two

Our original plan used two MLX GPU streams and alternated steps between
them, with `probe_cross_stream_dependency` in `overlap.py` checking that MLX
orders a cross-stream dependency correctly. In the end, model, cache, and
async evaluation all share the owner thread's **default GPU stream**. A
second stream adds a command-queue handoff and enables no extra overlap for a
single-model pipeline: the GPU executes one step at a time anyway, and the
CPU/GPU overlap comes from `async_eval`, not from concurrency on the GPU.

We also do not use a background Python thread for the executor. All MLX
graph construction stays on one owner thread, which keeps MLX's
thread-local state simple and avoids the GIL contention that a second Python
thread would bring.

## How we verify it

Overlap is easy to get subtly wrong: a missing dependency gives wrong tokens
only sometimes. MLX Air checks it at three levels.

1. **Exact parity.** `mlx-host-validation/scripts/native_overlap.sh`
   requires identical outputs between `serial` and `overlap` modes for greedy
   decoding. For sampled decoding, the random draw itself does not depend on
   the mode ([post 7](2026-11-22-on-device-reproducible-sampling.md)).
2. **Timeline evidence.** The script records CPU dispatch and synchronization
   events and checks that steady decode transitions queue N+1 before the CPU
   finishes processing N. Each step's synchronization is paired with that
   step's own dispatch, not the next one in time, because with chaining the
   next synchronization often belongs to the previous step
   ([ADR-0009](../adr/0009-benchmark-measurement-integrity.md)).
3. **Metal traces.** It captures matched serial and overlap Metal System
   Traces of the same four-request decode workload and rejects a p95 GPU
   interval-gap regression larger than 2% or 50 microseconds, whichever is
   larger.

## Results

Decode chaining (`1793318`), measured with
`mlx-air bench run --suite optimization --focus decode` (mean TPOT,
`decode_128_c1`, lower is better):

| Model | serial | overlap | Overlap gain |
| --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 4-bit | 13.13 ms | 11.40 ms | 1.73 ms (13% lower) |
| LFM2.5-8B-A1B 4-bit | 7.63 ms | 5.88 ms | 1.75 ms (23% lower) |

*Table 1: Same run, same build, serial vs. overlap mode. Run ID
`decode-1793318`. Before decode chaining, the two modes were within 0.1 ms
of each other (run `decode-f70afde`): overlap did not overlap anything that
mattered.*

The gain is nearly the same in absolute terms for both models, about
1.7 ms. That is what you would expect if the saved time is the CPU gap per
step, which does not depend on the model. For the faster MoE model the same
gap is a larger share of the step.

Overlap is now the default (`MLX_RUNTIME_NATIVE_EXECUTION_MODE=overlap`).
`serial` remains available for A/B comparison and debugging, and an invalid
value fails startup rather than silently falling back.

## Acknowledgments

The zero-overhead design follows the
[SGLang v0.4 overlap scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/),
which introduced future tokens and a one-step result delay; vLLM's
asynchronous scheduling follows the same idea. MLX's
[`async_eval`](https://ml-explore.github.io/mlx/build/html/python/transforms.html)
made it possible to express without threads.
