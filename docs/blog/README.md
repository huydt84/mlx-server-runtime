# Blog

Deep dives into inference optimization: how LLM serving engines work in
general, and how MLX Air does it on Apple Silicon with MLX and Metal.

Each post explains a general technique first, then shows MLX Air's
implementation, the code that implements it, and the benchmark results
checked into [`benchmarks/results/`](../../benchmarks/results/). Read them in
order as a course, or jump to a topic.

---

**Nov 22, 2026 · 7 min read** · Featured

### [Introducing MLX Air: A Serving Engine for Apple Silicon, Not a Proxy](2026-11-22-introducing-mlx-air.md)

Why serving is split into a Rust control plane and a Python MLX worker, what
the `native-mlx` backend owns, and a map of the series. Single-stream decode
on a 4-bit LFM2.5-8B-A1B went from 18.95 ms to 5.80 ms per token.

---

**Nov 22, 2026 · 8 min read**

### [Inference 101: Prefill, Decode, and Why Your Mac Is Memory-Bound](2026-11-22-inference-101-prefill-decode.md)

A first-principles cost model: TTFT, TPOT, arithmetic intensity, and KV
cache size. Why a 4-bit Qwen3-4B decodes at about 11 ms per token, and why an
8B MoE decodes twice as fast.

---

**Nov 22, 2026 · 9 min read**

### [Paged KV Cache on Unified Memory](2026-11-22-paged-kv-cache-unified-memory.md)

One preallocated page pool sized from Metal's recommended working set,
admission that never runs a request out of pages, copy-on-write sharing, and
the MLX buffer-donation rule that keeps writes in place.

---

**Nov 22, 2026 · 11 min read**

### [Writing a Paged Decode Attention Kernel in Metal](2026-11-22-paged-decode-attention-metal.md)

A line-by-line walk through the kernel: block-table indexing, GQA sharing,
simdgroup online softmax, and split-K. Lowering the split minimum from 256 to
32 keys took a 64-token step from 0.48 ms of attention toward SDPA's 0.20 ms.

---

**Nov 22, 2026 · 8 min read**

### [Continuous Batching and Chunked Prefill on One GPU](2026-11-22-continuous-batching-chunked-prefill.md)

Iteration-level scheduling told through three bugs: an unenforced limit, a
missing decode priority, and padding that made an eight-request step take
3.6 s instead of 0.1 s.

---

**Nov 22, 2026 · 9 min read**

### [Zero-Overhead Scheduling with MLX Lazy Evaluation](2026-11-22-overlap-scheduling-mlx-async-eval.md)

Building step N+1 while the GPU runs step N, feeding it tokens that never
left the device. Decode chaining cut 1.7 ms from every token on both models;
a Metal command-queue detail cost 7 ms of TTFT until it was fixed.

---

**Nov 22, 2026 · 5 min read**

### [On-Device Sampling That Survives Batching](2026-11-22-on-device-reproducible-sampling.md)

Temperature, top-p, and the Gumbel-max trick in MLX, with random keys derived
from request seed and token position so batch composition and overlap mode
cannot change the draw.

---

**Nov 22, 2026 · 8 min read**

### [Prefix Caching: Radix Trees vs. Block Hashes](2026-11-22-prefix-caching-radix-vs-block-hash.md)

vLLM-style hash chains and SGLang-style radix trees over the same paged
pool. Exact-hit latency of 139 ms vs. 516 ms without reuse, and how equal hit
rates still produced unequal latency.

---

**Nov 22, 2026 · 8 min read**

### [Serving Hybrid Models: Gated DeltaNet, Short Convolutions, and 1-Bit Weights](2026-11-22-serving-hybrid-models.md)

What recurrent state does to a paged cache, to batching, and to prefix
reuse, and how a 27B Qwen3.5 runs from a 1-bit or Hadamard-rotated ternary
pack.

---

**Nov 22, 2026 · 7 min read**

### [The Boring Milliseconds: HTTP, SSE, and Cancellation](2026-11-22-gateway-sse-cancellation.md)

Nagle's algorithm, a non-blocking flag that macOS sockets inherit, admission
control that says no early, cancellation that frees KV pages, and failing
fast at startup.

---

**Nov 22, 2026 · 9 min read**

### [Measuring Inference Honestly](2026-11-22-measuring-inference-honestly.md)

Primary metrics, trial-level statistics, calibration, order alternation,
cold-cache validation, no retries, and profiling that never becomes a
benchmark row.

---

**Nov 22, 2026 · 10 min read**

### [Decode Performance Retrospective: From 18.9 ms to 5.8 ms per Token](2026-11-22-decode-performance-retrospective.md)

Four commits, four benchmark runs, two models: which change bought which
millisecond, plus the unstable rows and null results.

---

**Nov 26, 2026 · 10 min read**

### [Prefill on Packed Weights: Decode Once, Multiply in 8x8 Blocks](2026-11-26-packed-prefill-simdgroup-matrices.md)

Why a decode-tuned packed-weight kernel stalls on prompts, a register cliff in
`simdgroup_matrix` products, and the kernel that cut a 3,725-token prompt's
time to first token on Inkling-2-Mini from 17.8 s to 9.7 s.

---

## Reading paths

- **New to LLM inference:** 2, 5, 3, 6, 8, 11.
- **Apple Silicon and MLX specifics:** 3, 4, 6, 9, 13.
- **Running or operating MLX Air:** 1, 10, 11.

## Writing a post

Posts live in this directory as `YYYY-MM-DD-slug.md` with front matter
(`layout`, `title`, `author`, `date`, `tags`, `excerpt`). Every performance
number must come from a `mlx-air bench run` result that is checked in or
linked, stated as an absolute value with its unit and direction (see
[Measuring Inference Honestly](2026-11-22-measuring-inference-honestly.md)).
