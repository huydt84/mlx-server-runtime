# 0011. Packed weight kernels

- Status: Accepted
- Date: 2026-11-25

## Context

LookAtMyQuant packs store every matrix as trellis codes (`ht*`, `rtr*`), NVFP4
blocks, or Lloyd-Max level indices (`lm*`), and one layer mixes formats per
projection and per 128-row group of an expert. Expanding the weights does not
fit the machines the packs are made for. Inkling-2-Mini visits 48 layers per
token, so one kernel launch per format would spend most of a step on launch
overhead. The bit layouts follow the reference runtime
[lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant) (MIT License,
Copyright (c) 2026 Shin Megami Boson; see
`python/mlx_worker/native_mlx/kernels/LICENSE.lookatmyquant`), whose kernels run
one request at a time.

## Decision

- **A bank concatenates every row group that shares an input width into one
  buffer.** Each row group has an eight-word descriptor: format, code offset
  (words), scale offset (halves), aux offset, rows, first output row, input
  index, and outer scale (float bits). The aux offset locates Lloyd-Max levels
  or NVFP4 block scales. The input index picks the copy of the input: each
  dense matrix has its own rotation, and NVFP4 experts read the unrotated one.
- **A trellis row is one little-endian bit stream that stores the last column
  first.** The 12-bit window ending column group `last` starts at bit
  `(N - 1 - last) * K`; bits past the row are zero.
- **One threadgroup handles one (unit, row group) pair.** A unit is up to eight
  inputs that multiply the same rows, so each row is decoded once for all of
  them. Units of a dense bank are consecutive tokens; units of an expert bank
  are runs of one expert in the slots sorted by expert. Each expert adds at
  most one partly filled unit.
- **A simdgroup is split into sub-warps of `LANES` lanes** that multiply two
  rows (four for a single input) at once, so one load of the input serves all
  of them. Four rows with eight inputs, and decoding tiles into threadgroup
  memory for simdgroup matrix products, were both measured slower.
- **From 128 inputs on, a dense bank is decoded once and multiplied with MLX.**
  Below that, decoding inside the kernel is faster.
- **Steps of at most eight tokens use every kernel path a one-token step uses,
  so their results are bit-identical.** The router, sink, dynamic-convolution,
  and message-gate products run in kernels that split work by column, never by
  token, because MLX chooses different matmul kernels for one and several rows.
- **Rotations are one launch each.** The Hadamard matrix is a Kronecker product
  of a power-of-two factor, applied as SIMD shuffles between the lanes that hold
  rows, and a small factor read from a buffer. No threadgroup memory or barrier
  is needed.
- **Message gates and small products spread one token over several
  threadgroups.** A forward runs them one after another, so their latency
  matters more than their throughput. They win up to 32 tokens (gates) and 16
  tokens (small products).

## Consequences

- Adding a format means adding a decoder to `packed_formats.metal` and an entry
  to the format tables; the launch code does not change.
- Expert products reach about 1.0 TMAC/s on an M4 Pro GPU, 40% of the time being
  decoding. Long prefill is limited by this, at about 200 tokens/s on the
  5.4 GB pack.
- Steps longer than eight tokens differ from short ones in the last bits, so
  batching and prompt chunking can change greedy text.

## Code

- `python/mlx_worker/native_mlx/packed_matmul.py` — `PackedBank`,
  `plan_experts`, `_DECODE_TOKENS`.
- `python/mlx_worker/native_mlx/kernels/packed_formats.metal`,
  `packed_matmul.metal` — decoders and products.
- `python/mlx_worker/native_mlx/rotation.py` — `rotate_packed`, `rotate_ranges`.
- `python/mlx_worker/native_mlx/inkling2_kernels.py` — `blend_messages`,
  `row_products`.
- `python/tests/test_native_mlx_inkling2.py` —
  `test_short_steps_compute_exactly_what_one_token_steps_do`.
