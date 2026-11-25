# 0014. Blocked packed products for long steps

- Status: Accepted
- Date: 2026-11-26

## Context

[ADR-0011](0011-packed-weight-kernels.md) multiplies packed rows with at most
eight inputs per decoded row. Prefill steps pass hundreds of tokens, so the
same rows were decoded again for every eight inputs, and each lane's dot
products loaded one input value for every two multiply-adds. Expert products
reached about 1.0 TMAC/s on an M4 Pro GPU, against 2.9 TMAC/s for MLX's own
half-precision matmul. Long prompts prefilled at about 200 tokens/s on the
5.4 GB Inkling-2-Mini pack.

A 256-token step of that pack routes its 2,048 expert slots to about 106
experts, 19 slots each on average, so units of 32 decode each expert's rows
once for most experts.

A microbenchmark of `simdgroup_multiply_accumulate` on this GPU showed a
register cliff. Sixteen live 8x8 float accumulators per SIMD group ran at
about 1.6 TMAC/s, while eight ran at about 2.8 TMAC/s. A first tiled kernel
with sixteen accumulators was therefore slower than the per-lane loops.

## Decision

- **Steps of more than eight tokens use units of up to 32 inputs.** Expert
  units are runs of one expert in the slots sorted by expert; the next unit's
  start ends a unit. Dense units are 32 consecutive tokens.
- **Each SIMD group owns 16 rows and decodes a 16-row by 32-column tile per
  iteration**: lane `l` decodes the 16-column chunk `l / 16` of row `l % 16`
  into threadgroup memory. The tile holds the format's weight type: half for
  trellis and NVFP4 (whose block scale is folded in exactly), float for
  Lloyd-Max levels.
- **The tile is multiplied with the unit's inputs as 8x8 SIMD-group matrix
  products accumulated in float**, two row blocks by up to four token blocks,
  so a SIMD group holds at most eight accumulators. Inputs are read straight
  from device memory in the fragment layout of MLX's `BaseMMAFrag`: lane `l`
  holds row `(l / 16 % 2) * 4 + (l / 2) % 4` and columns
  `(l / 8 % 2) * 4 + (l % 2) * 2` and the next one.
- **Units of one to four inputs keep the per-lane loops**, which are faster
  there. Five or more inputs use the blocked products.
- **Dense banks are no longer decoded into device memory** for long steps;
  the blocked products replace the decode-then-MLX-matmul path.
- **Packed input widths must be multiples of 32 columns**, one tile.
- **The soft-gated rotation stages its gated, signed row in threadgroup
  memory**, so each `tanh`/`exp` is evaluated once per element instead of once
  per thread that shares the row. Staging the other rotation inputs was slower,
  because the extra threadgroup memory lowers occupancy, so they read their
  inputs directly.

## Consequences

- Steps of at most eight tokens run the kernels of ADR-0011 unchanged, so they
  still compute exactly what one-token steps compute.
- Token blocks are eight wide, so a unit whose size is not a multiple of eight
  multiplies padding; at 256-token steps about a fifth of expert work is
  padding.
- Long steps no longer allocate an expanded copy of each dense bank, about
  80 MB per visit on the 5.4 GB pack. The family's measured step memory
  ([ADR-0012](0012-windowed-kv-and-step-memory.md)) was not lowered and is now
  conservative.
- Revisit the tile shape on GPUs with matrix hardware or a different register
  file, and remeasure the one-to-four-input crossover.

## Code

- `python/mlx_worker/native_mlx/kernels/packed_matmul.metal` —
  `packed_rows_blocked`, `packed_decode_tile`, `packed_unit_member`,
  `packed_dense_blocks`, `packed_expert_blocks`, `packed_run_group`.
- `python/mlx_worker/native_mlx/packed_matmul.py` — `_BLOCKED_TOKENS`,
  `lanes_for`, `PackedBank.dense`, `PackedBank.expert`, `plan_experts`.
- `python/mlx_worker/native_mlx/kernels/rotation.metal` — `rotate_input`,
  `RotationStaged`.
- `python/tests/test_native_mlx_packed_matmul.py` —
  `test_dense_bank_multiplies_decoded_rows`,
  `test_expert_bank_multiplies_each_slot_with_its_expert`.
