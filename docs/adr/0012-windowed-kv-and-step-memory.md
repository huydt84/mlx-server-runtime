# 0012. Windowed KV pool, relative attention, and step memory

- Status: Accepted
- Date: 2026-11-25

## Context

40 of Inkling-2-Mini's 48 layer visits attend only to the last 1,024 tokens, and
every visit adds a relative-position bias and a sink logit that the paged
decode kernel of [ADR-0006](0006-paged-decode-attention-kernel.md) lacks. Every
visit also keeps convolution history. On a 24 GB Mac the 12.8 GB pack leaves
little room, so transient memory decides whether it loads.

## Decision

- **Sliding-window visits use a second page pool** whose pages return once they
  fall behind the window. The next query at position `length` reads back to
  `length - window + 1`, and a window may start and end inside a page. The
  windowed pool gets at most half the KV budget and at most 16 full windows,
  which the default number of active requests cannot exceed.
- **Attention reads a per-row base position.** Entry 0 of a windowed row's block
  table holds the position `base` whose earlier pages were released. Query
  tokens of different rows need no padding: each names its row and position.
- **A query uses the key splits its own keys need**, so its result does not
  depend on the other queries of the launch. Steps of at most eight tokens split
  keys as one-token steps do.
- **The pool's reference is replaced by each write**, which lets the write reuse
  the old buffer instead of copying the layer.
- **Convolutions compute the new history together with their outputs.**
  Otherwise every layer's inputs stay alive until the step commits. Steps of
  more than 16 tokens keep no inputs at all, so they cannot be committed partly.
- **The family's step memory is measured, not derived**: 1 GiB plus 0.5 MiB per
  prefill token, because prefill keeps several visits in flight and each decodes
  its projections. The buffer cache is capped at 256 MiB, which costs no speed.
- **Loading keeps no freed buffers**, and layers take their source tensors out of
  the caller's list, so those are released once packed into a bank.
- **A step waits for nothing between visits when the device is idle**: every
  twelfth visit is launched early so the GPU starts while the rest is built.

## Consequences

- A request needs pages in both pools before it is admitted.
- Inkling-2-Mini keeps no prefix cache.
- Another windowed model can reuse the bundle, but its step memory must be
  measured again.

## Code

- `python/mlx_worker/native_mlx/windowed_cache.py` —
  `WindowedPagedKVCacheBackend`, `MAX_PARTIAL_TOKENS`.
- `python/mlx_worker/native_mlx/windowed_attention.py`, `relative_attention.py`,
  `kernels/relative_attention.metal` — attention.
- `python/mlx_worker/native_mlx/causal_conv.py` — `causal_keys_values`,
  `causal_conv`.
- `python/mlx_worker/native_mlx/registry.py` — `StepMemory`.
- `python/mlx_worker/native_mlx/bootstrap.py` — `_loading_buffers`.
- `python/mlx_worker/native_mlx/models/inkling2.py` — `_Weights`,
  `_VISITS_PER_LAUNCH`.
