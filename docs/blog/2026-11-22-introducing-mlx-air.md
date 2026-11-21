---
layout: post
title: "Introducing MLX Air: A Serving Engine for Apple Silicon, Not a Proxy"
author: "MLX Air Team"
date: 2026-11-22
tags: [announcement, architecture, apple-silicon, serving]
excerpt: "Why we split serving into a Rust control plane and a Python MLX worker, what the native-mlx backend owns, and a map of the twelve-part series on inference optimization that follows."
---

**Table of contents**

- [Why build another local server?](#why-build-another-local-server)
- [Two processes, one contract](#two-processes-one-contract)
- [The native-mlx backend](#the-native-mlx-backend)
- [What a request goes through](#what-a-request-goes-through)
- [Where we are today](#where-we-are-today)
- [The series](#the-series)
- [Getting started](#getting-started)
- [Acknowledgments](#acknowledgments)

We are excited to introduce **MLX Air** (`mlx-air`), an inference serving
runtime for Apple Silicon. It serves an OpenAI-compatible
`/v1/chat/completions` API (plus Whisper transcription) from a Mac, and it is
built around one rule:

> Rust owns serving. Python owns model execution.

MLX Air is not a proxy around `mlx_lm.server`. It has its own scheduler, its
own paged KV cache, its own Metal decode-attention kernel, and its own
benchmark harness. This post explains why, and introduces a series of posts
that walk through each piece. The series doubles as a learning resource on
LLM inference optimization in general: every post starts from the general
technique (PagedAttention, continuous batching, CPU/GPU overlap, prefix
caching) and then shows what changes when the hardware is a Mac with unified
memory and the framework is MLX.

```bash
curl -fsSL \
  https://github.com/huydt84/mlx-server-runtime/releases/latest/download/install.sh | sh
~/.local/bin/mlx-air serve --model mlx-community/Qwen3-8B-4bit --memory-utilization 0.6
```

## Why build another local server?

Running a model on a Mac is easy. `mlx-lm` loads a checkpoint and generates
tokens in a few lines of Python. Serving one is different. A server has to:

- accept many requests at once and keep the GPU busy with all of them;
- stream tokens back with low, steady latency;
- stop work when a client disconnects;
- refuse work it cannot hold in memory, instead of crashing or swapping;
- report what it is doing, in numbers an operator can alert on.

The obvious shortcut is an HTTP proxy in front of `mlx_lm.server`:

```text
Client -> Rust -> HTTP -> mlx_lm.server -> MLX
```

We rejected it early. Two HTTP stacks parse and queue the same request. The
front process cannot tell a slow backend from a crashed one. Cancellation
stops at the proxy: the client is gone, but the backend keeps generating. And
every scheduling decision lives in a process we do not control.

## Two processes, one contract

MLX Air runs two processes per model:

```mermaid
flowchart LR
    C[Client] -- HTTP / SSE --> G[Rust gateway]
    G -- "Unix socket, newline-delimited JSON" --> W[Python worker]
    W --> R[native runtime + scheduler]
    R --> E[model-step executor]
    E --> M[MLX / Metal]
```

*Figure 1: The serving path. The gateway never touches a tensor; the worker
never reads an HTTP header.*

**The Rust gateway** owns everything a client can see: HTTP parsing, SSE
framing, admission control and backpressure (HTTP 429 when the queue is full),
client-disconnect detection, worker process supervision, Prometheus metrics,
and structured request logs. It is a hand-written `std::net` server with one
thread per connection and no garbage collector.

**The Python worker** owns everything that touches the model: prompt
construction, tokenization, scheduling, the KV cache, the forward pass,
sampling, and detokenization. Python is the right home for this because MLX,
its tokenizers, and its model zoo live there. Porting model code to Rust
would duplicate thousands of lines and fall behind upstream.

The boundary between them is a Unix domain socket carrying one JSON object
per line. We chose JSON over MessagePack or Protobuf on purpose: every frame
can be read with `nc -U`. The message shapes are fixed, so only the encoder
would change if serialization ever showed up in a profile. So far it has not.

| Owned by Rust | Owned by Python |
| --- | --- |
| HTTP and SSE | Tokenization and chat templates |
| Queue admission and backpressure | Scheduling and batching |
| Cancellation initiation | KV cache lifecycle |
| Worker lifecycle | Forward pass and sampling |
| Metrics and logs | Detokenization and stop handling |

*Table 1: The ownership split.*

## The native-mlx backend

The worker has two backends. `v1` drives `mlx-lm`'s `BatchGenerator` (and
`mlx-vlm` for vision models), so any checkpoint `mlx-lm` can load works.
`native-mlx` is the default. It is our own runtime, and it is where the
optimization work in this series happens:

- a **paged KV cache**: one preallocated pool of 16-token pages shared by
  every request and by the prefix cache;
- a **custom Metal kernel** that reads decode attention straight from those
  pages;
- an **iteration-level scheduler** with chunked prefill and alternating
  decode and prefill steps;
- an **overlap executor** that builds step N+1 while the GPU runs step N;
- a **radix prefix cache** that reuses KV pages across requests that share a
  prompt prefix;
- **on-device sampling** that is reproducible across batch composition;
- a **hybrid cache** for models that mix attention with recurrent layers
  (LFM2, Qwen3.5 Gated DeltaNet).

`native-mlx` supports a fixed list of architecture classes: `Qwen2`, `Qwen3`,
`Gemma3`, `Lfm2Moe`, and dense text-only `Qwen3_5`. A checkpoint that
declares any other class fails at startup and names `--backend v1`. It never
silently falls back.

## What a request goes through

Here is one streaming request, end to end:

1. The gateway parses the request, checks token limits, and waits for an
   admission permit. If the queue is full it answers 429 at once.
2. It sends a `ChatCompletion` frame over the socket.
3. The worker applies the chat template, tokenizes, and submits the request
   to the scheduler.
4. The scheduler admits it only when the KV pool can hold its prompt plus
   `max_tokens`, after subtracting any pages the prefix cache already has.
5. The prompt runs as one or more 256-token prefill chunks, interleaved with
   decode steps for requests that are already generating.
6. Each decode step produces one token per running request. The worker
   detokenizes and sends a delta frame; the gateway writes an SSE event.
7. On a stop token or `max_tokens`, the worker releases the request's pages
   (or hands its prompt pages to the prefix cache) and sends the final frame.

If the client disconnects, the gateway notices on the socket, sends a
`CancelRequest`, and the scheduler drops the request at the next step
boundary.

## Where we are today

All numbers below come from `mlx-air bench run` result files checked into
`benchmarks/results/`, on one Apple Silicon laptop on AC power. Lower
time-per-output-token (TPOT) is better.

| Model (role) | Workload | Before optimization (f70afde) | Today (96bf4d2) |
| --- | --- | ---: | ---: |
| Qwen3-4B-Instruct-2507 4-bit (primary) | `decode_128_c1` TPOT | 14.52 ms | 11.04 ms |
| LFM2.5-8B-A1B 4-bit MoE | `decode_128_c1` TPOT | 18.95 ms | 5.80 ms |

*Table 2: Mean time per output token, overlap mode, single stream, 128
output tokens. The MoE model improved 3.3x; the dense model 1.3x.*

On a GPU-bound single-stream Qwen3-4B decode step, native-mlx is now within
3% of `mlx_lm` at short contexts and slightly ahead at 1,024 keys. The
[retrospective post](2026-11-22-decode-performance-retrospective.md) breaks
down which change bought which millisecond.

We also want to be clear about what MLX Air does **not** claim. It does not
claim vLLM or SGLang feature parity, FlashAttention equivalence, universal
Hugging Face support, or multi-model serving in one process. It runs one model
per worker on one Mac.

## The series

Each post stands on its own. Posts 2, 5, 6, and 11 are the most useful as
general learning material; posts 4, 8, and 9 are the most Apple-specific.

1. **Introducing MLX Air** (this post)
2. [Inference 101: Prefill, Decode, and Why Your Mac Is Memory-Bound](2026-11-22-inference-101-prefill-decode.md)
3. [Paged KV Cache on Unified Memory](2026-11-22-paged-kv-cache-unified-memory.md)
4. [Writing a Paged Decode Attention Kernel in Metal](2026-11-22-paged-decode-attention-metal.md)
5. [Continuous Batching and Chunked Prefill on One GPU](2026-11-22-continuous-batching-chunked-prefill.md)
6. [Zero-Overhead Scheduling with MLX Lazy Evaluation](2026-11-22-overlap-scheduling-mlx-async-eval.md)
7. [On-Device Sampling That Survives Batching](2026-11-22-on-device-reproducible-sampling.md)
8. [Prefix Caching: Radix Trees vs. Block Hashes](2026-11-22-prefix-caching-radix-vs-block-hash.md)
9. [Serving Hybrid Models: Gated DeltaNet, Short Convolutions, and 1-Bit Weights](2026-11-22-serving-hybrid-models.md)
10. [The Boring Milliseconds: HTTP, SSE, and Cancellation](2026-11-22-gateway-sse-cancellation.md)
11. [Measuring Inference Honestly](2026-11-22-measuring-inference-honestly.md)
12. [Decode Performance Retrospective: From 18.9 ms to 5.8 ms per Token](2026-11-22-decode-performance-retrospective.md)

## Getting started

1. Install: `curl -fsSL https://github.com/huydt84/mlx-server-runtime/releases/latest/download/install.sh | sh`
2. Check the host: `mlx-air doctor`
3. Serve: `mlx-air serve --model mlx-community/Qwen3-4B-Instruct-2507-4bit`
4. Benchmark: `mlx-air bench run --suite smoke`
5. Profile: `mlx-air profile run --model mlx-community/Qwen3-4B-Instruct-2507-4bit --requests 4 --concurrency 2 --open`

The [architecture document](../explanation/architecture.md) and the
[ADRs](../adr/README.md) record each design decision with its context.

## Acknowledgments

MLX Air builds directly on [MLX](https://github.com/ml-explore/mlx),
[`mlx-lm`](https://github.com/ml-explore/mlx-lm), `mlx-vlm`, and
`mlx-audio`. Its design borrows openly from
[vLLM](https://github.com/vllm-project/vllm) (PagedAttention, memory
utilization sizing) and [SGLang](https://github.com/sgl-project/sglang)
(RadixAttention, the zero-overhead scheduler). Thanks to the authors of all of
them.
