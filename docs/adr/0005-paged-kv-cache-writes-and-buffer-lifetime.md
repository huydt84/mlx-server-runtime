# 0005. Paged KV cache writes and buffer lifetime

- Status: Accepted
- Date: 2026-11-22

## Context

Each layer's KV pool is one MLX array. MLX can update an array in place only
when nothing else holds a reference to the old buffer (buffer donation).
Otherwise every page write copies the whole pool layer.

## Decision

- **Pack rows without per-row slices when nothing is padded.** When every row
  appends the same number of tokens (every decode step does), the slot order
  is row-major `(row, position)`, so one reshape packs the step.
- **Write consecutive slots with a slice copy.** When a reservation's slots
  are consecutive (for example one decode row, even across a page boundary),
  a plain slice assignment is cheaper on the GPU than an indexed scatter.
- **Drop the pool's reference before the write lands.** `stage_layer` replaces
  the pool's array with the updated one so the scatter can donate the old
  buffer. `_finish_reservation` clears the staged arrays so a finished
  reservation does not keep a buffer alive and force a full copy on the next
  step.
- **Free only pages that this step stopped referencing.** Pages whose last pin
  drops are freed by `_finish_reservation`.
- **Layers that hold only state keep an empty placeholder** so per-layer
  indexing stays uniform. Staging KV into one fails on its zero-page shape.
- **Every prefix-cache entry holds exactly one page of its own**, so stored
  bytes are `entries * page_bytes`.
- **Decode reads keys directly from the pages** (see
  [ADR-0006](0006-paged-decode-attention-kernel.md)). A single row with no
  cached prefix attends only to its own new K/V, which is already dense, so
  the page gather is skipped.
- **Attention masks are built lazily on the device.** Building them in Python
  with `tolist` once per layer synchronized MLX and rebuilt a nested list for
  every row.

## Consequences

- A new reference to a pool array (for debugging, metrics, or a closure)
  silently turns in-place writes into full-layer copies. Watch
  `native_pipeline_profile` timings when changing cache code.

## Code

- `python/mlx_worker/native_mlx/cache.py` — `PagedBatchReservation.stage_layer`,
  `PagedKVCacheBackend.__post_init__`, `_commit_reservation`,
  `_finish_reservation`.
- `python/mlx_worker/native_mlx/prefix_cache.py` — `_stored_bytes`.
- `python/mlx_worker/native_mlx/attention.py` — `append_and_attend`,
  `_dense_causal_mask`.
