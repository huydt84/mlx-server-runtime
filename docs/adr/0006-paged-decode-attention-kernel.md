# 0006. Paged decode attention kernel

- Status: Accepted
- Date: 2026-11-22

## Context

Gathering each row's pages into dense K/V before attention cost a copy per
decode step. Decode reads the whole cache for one query token per row, so it
can read keys and values straight from the page pool instead.

## Decision

- **A custom Metal kernel reads decode attention directly from the paged KV
  pool.** It is used for decode steps whose head dimension is a multiple of
  32. Other steps use the gather path.
- **Launch geometry:**
  - `grid.x = rows * KV_HEADS` threadgroups, one per (row, KV head) pair, each
    handling every query head that shares that KV head (GQA);
  - `grid.y = SPLITS` key ranges per row, merged by `paged_decode_merge` when
    `SPLITS > 1` (one threadgroup per head, one thread per head element);
  - `SIMDGROUPS` simdgroups of 32 lanes per threadgroup; each lane owns
    `HEAD_DIM / 32` elements of a head. Each simdgroup accumulates an online
    softmax over a strided subset of its split's keys, and the simdgroups
    merge through threadgroup memory.
- **Split keys only when few threadgroups would run.** A row's keys are split
  until a step launches about 32 threadgroups, with at least 32 keys per split
  and at most 32 splits. Split count `s` covers `[32 * s, 64 * s)` keys.
  Startup warmup decodes key counts in the middle of each range, so a slightly
  longer or shorter prompt compiles the same variant.
- **Online softmax:** a running maximum and a running sum of exponentials are
  rescaled whenever the maximum grows. Partial results over disjoint key
  ranges are stored unnormalized at their own maximum and merged by rescaling
  to the global maximum. An empty partial (maximum `-inf`) contributes
  nothing.
- **The query dtype and the KV dtype are separate template parameters.**
  Quantized checkpoints can run float16 activations over a bfloat16 cache or
  the reverse, and each pairing is its own kernel variant.
- **Table and length pointer types are template parameters** because MLX
  passes single-element inputs in the constant address space and larger ones
  in the device address space.
- **Kernel bodies live in `.metal` files.** The Python `_SOURCE` strings only
  declare threadgroup scratch and forward MLX's generated arguments. When the
  sources are concatenated into one string, `#pragma once` is dropped because
  it does nothing there and the compiler warns about it.

## Consequences

- Each `(query dtype, KV dtype, head dim, splits)` combination is a separate
  compiled kernel. Startup warmup has to cover the combinations a checkpoint
  uses, or the first real request pays the compile.

## Code

- `python/mlx_worker/native_mlx/kernels/paged_decode.metal`,
  `python/mlx_worker/native_mlx/kernels/online_softmax.metal`.
- `python/mlx_worker/native_mlx/paged_decode.py` — `splits_for`,
  `warmup_key_counts`, `_TARGET_THREADGROUPS`, `_MIN_KEYS_PER_SPLIT`.
- `python/mlx_worker/native_mlx/metal_source.py` — `_expand`.
