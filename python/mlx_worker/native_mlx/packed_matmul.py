"""Matrix products with LookAtMyQuant packed weights.

The kernel sources are ``kernels/packed_formats.metal`` and
``kernels/packed_matmul.metal``.

A packed checkpoint stores every matrix as rows of trellis codes, Lloyd-Max
level indices, or NVFP4 blocks, and one model layer mixes several formats:
each attention projection has its own, and each expert (or each 128-row group
of an expert) has its own. Dispatching one kernel per format would multiply
the launches of a 48-visit forward, so a :class:`PackedBank` concatenates all
row groups that share an input width into one buffer and describes each group
with a small descriptor. One launch then covers all formats of the bank.

Weights are never expanded: the kernels decode 16 columns of a row at a time
while they multiply. Steps of more than eight tokens multiply up to 32 inputs
with each decoded tile as 8x8 blocks. The bit layouts follow the reference runtime published with the
packs (lookatmyquant, MIT License, Copyright (c) 2026 Shin Megami Boson).
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header

FORMAT_IDS: Mapping[str, int] = {
    "ht1": 0,
    "ht2": 1,
    "ht3": 2,
    "rtr1": 3,
    "rtr2": 4,
    "rtr3": 5,
    "rtr4": 6,
    "nvfp4": 7,
    "lm6": 8,
    "lm8": 9,
}
FORMAT_BITS: Mapping[str, int] = {
    "ht1": 1,
    "ht2": 2,
    "ht3": 3,
    "rtr1": 1,
    "rtr2": 2,
    "rtr3": 3,
    "rtr4": 4,
    "nvfp4": 4,
    "lm6": 6,
    "lm8": 8,
}
"""Code bits per weight, as stored in each pack's metadata ``k``."""

TRELLIS_LUT_SHAPES: Mapping[str, tuple[int, ...]] = {
    "ht1": (512, 4),
    "ht2": (512, 4),
    "ht3": (1024, 2),
    "rtr1": (4096,),
    "rtr2": (4096,),
    "rtr3": (4096,),
    "rtr4": (4096,),
}

ROWS_PER_GROUP = 128
"""Rows handled by one threadgroup; a multiple of every sub-warp's rows."""

_SIMDGROUPS = 8
_SIMD_WIDTH = 32
_CHUNK = 16
_MAX_TOKENS = 8
_BLOCKED_TOKENS = 32
_BLOCKED_COLUMNS = 32

_DENSE_SOURCE = """
    threadgroup half lut[mlx_air::PACKED_LUT_HALVES];
    mlx_air::packed_dense_matmul<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
        blob, luts, descriptors, x, meta, y, lut,
        threadgroup_position_in_grid.xy, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""

_EXPERT_SOURCE = """
    threadgroup half lut[mlx_air::PACKED_LUT_HALVES];
    mlx_air::packed_expert_matmul<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
        blob, luts, descriptors, x, experts, order, units, meta, y, lut,
        threadgroup_position_in_grid.xy, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""

_ROWS_SOURCE = """
    threadgroup half lut[mlx_air::PACKED_LUT_HALVES];
    mlx_air::packed_row_lookup<FORMATS, N, SIMDGROUPS, GROUP_ROWS>(
        blob, luts, descriptors, rows, y, lut,
        threadgroup_position_in_grid.x, thread_position_in_threadgroup.x);
"""

_DENSE_BLOCKS_SOURCE = """
    threadgroup half lut[mlx_air::PACKED_LUT_HALVES];
    threadgroup float tiles[SIMDGROUPS * mlx_air::PACKED_TILE_WEIGHTS];
    mlx_air::packed_dense_blocks<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
        blob, luts, descriptors, x, meta, y, lut, tiles,
        threadgroup_position_in_grid.xy, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""

_EXPERT_BLOCKS_SOURCE = """
    threadgroup half lut[mlx_air::PACKED_LUT_HALVES];
    threadgroup float tiles[SIMDGROUPS * mlx_air::PACKED_TILE_WEIGHTS];
    mlx_air::packed_expert_blocks<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
        blob, luts, descriptors, x, experts, order, units, units_shape[0], meta,
        y, lut, tiles, threadgroup_position_in_grid.xy,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_PLAN_SOURCE = """
    mlx_air::packed_plan_experts<EXPERTS>(
        experts, order, units, experts_shape[0], thread_position_in_grid.x);
"""


@lru_cache(maxsize=1)
def _plan_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_packed_plan_experts",
        input_names=["experts"],
        output_names=["order", "units"],
        source=_PLAN_SOURCE,
        header=metal_header("packed_matmul.metal"),
    )


@lru_cache(maxsize=1)
def _rows_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_packed_row_lookup",
        input_names=["blob", "luts", "descriptors", "rows"],
        output_names=["y"],
        source=_ROWS_SOURCE,
        header=metal_header("packed_matmul.metal"),
    )


@lru_cache(maxsize=2)
def _dense_kernel(blocked: bool) -> Any:
    return mx.fast.metal_kernel(
        name="native_packed_dense_" + ("blocks" if blocked else "matmul"),
        input_names=["blob", "luts", "descriptors", "x", "meta"],
        output_names=["y"],
        source=_DENSE_BLOCKS_SOURCE if blocked else _DENSE_SOURCE,
        header=metal_header("packed_matmul.metal"),
    )


@lru_cache(maxsize=2)
def _expert_kernel(blocked: bool) -> Any:
    return mx.fast.metal_kernel(
        name="native_packed_expert_" + ("blocks" if blocked else "matmul"),
        input_names=[
            "blob",
            "luts",
            "descriptors",
            "x",
            "experts",
            "order",
            "units",
            "meta",
        ],
        output_names=["y"],
        source=_EXPERT_BLOCKS_SOURCE if blocked else _EXPERT_SOURCE,
        header=metal_header("packed_matmul.metal"),
    )


@lru_cache(maxsize=256)
def _meta(*values: int) -> mx.array:
    return mx.array(values, dtype=mx.uint32)


@dataclass(frozen=True)
class PackedRows:
    """One matrix, or one matrix per expert, stored in one packed format.

    Attributes:
        fmt: Format name, a key of :data:`FORMAT_IDS`.
        rows: Rows per block.
        codes: Codes as ``uint8`` or ``uint32``; the first axis may index
            blocks (experts or row groups).
        scales: Per-row float16 scales, or NVFP4 ``uint8`` block scales.
        levels: Lloyd-Max levels for ``lm6`` and ``lm8``.
        outer: NVFP4 scale per block.
        output_row: First output row of block 0 in a dense bank.
        input_index: Rotated input this matrix multiplies in a dense bank.
    """

    fmt: str
    rows: int
    codes: mx.array
    scales: mx.array
    levels: mx.array | None = None
    outer: mx.array | None = None
    output_row: int = 0
    input_index: int = 0


def build_trellis_luts(tables: Mapping[str, mx.array]) -> mx.array:
    """Concatenate a pack's trellis tables in the kernel's fixed order.

    Args:
        tables: Table per trellis format, as stored in ``luts.npz``; formats
            the pack does not use may be missing.

    Returns:
        A flat float16 array holding every trellis table.

    Raises:
        ValueError: If a table has an unexpected shape.
    """

    parts = []
    for fmt, shape in TRELLIS_LUT_SHAPES.items():
        table = tables.get(fmt)
        if table is None:
            table = mx.zeros(shape, dtype=mx.float16)
        elif tuple(int(size) for size in table.shape) != shape:
            raise ValueError(f"packed table {fmt!r} must have shape {shape}")
        parts.append(table.astype(mx.float16).reshape(-1))
    return mx.concatenate(parts)


def as_words(array: mx.array) -> mx.array:
    """Return an array's bytes as a flat uint32 vector, zero-padded at the end."""

    flat = mx.view(array.reshape(-1), mx.uint8) if array.nbytes else array
    remainder = int(flat.size) % 4
    if remainder:
        flat = mx.concatenate([flat, mx.zeros((4 - remainder,), dtype=mx.uint8)])
    return mx.view(flat, mx.uint32)


def lanes_for(input_width: int) -> int:
    """Return the sub-warp width whose lanes evenly cover one row.

    Raises:
        ValueError: If the width does not hold whole 32-column tiles.
    """

    if input_width <= 0 or input_width % _BLOCKED_COLUMNS:
        raise ValueError(
            f"packed input width {input_width} must be a multiple of {_BLOCKED_COLUMNS}"
        )
    chunks = input_width // _CHUNK
    lanes = 16
    while chunks % lanes:
        lanes //= 2
    return lanes


class PackedBank:
    """Row groups of several packed matrices that share one input width."""

    def __init__(
        self,
        input_width: int,
        matrices: Sequence[PackedRows],
        luts: mx.array,
        *,
        expert_groups: Sequence[Sequence[tuple[int, int]]] | None = None,
    ):
        """Assemble the bank.

        Args:
            input_width: Columns of every matrix.
            matrices: Packed matrices. In an expert bank each holds one format's
                blocks.
            luts: Result of :func:`build_trellis_luts`.
            expert_groups: For an expert bank, per expert the ordered
                ``(format id, block index)`` pairs that make up its rows.
                ``None`` builds a dense bank.

        Raises:
            ValueError: If codes, scales, or the expert map are inconsistent.
        """

        self.input_width = input_width
        self.luts = luts
        self.lanes = lanes_for(input_width)
        self._words: list[mx.array] = []
        self._size = 0
        self.format_mask = 0
        placed: dict[int, list[tuple[int, ...]]] = {}
        dense: list[tuple[int, ...]] = []
        self.output_rows = 0
        for matrix in matrices:
            blocks = self._place(matrix)
            fid = FORMAT_IDS[matrix.fmt]
            self.format_mask |= 1 << fid
            if expert_groups is None:
                for block_index, block in enumerate(blocks):
                    base = matrix.output_row + block_index * matrix.rows
                    dense.extend(self._split(block, base, matrix.input_index))
                    self.output_rows = max(self.output_rows, base + matrix.rows)
            else:
                if fid in placed:
                    raise ValueError(f"expert bank repeats format {matrix.fmt!r}")
                placed[fid] = blocks
        if expert_groups is None:
            descriptors = dense
            self.experts = 0
            self.groups = len(dense)
        else:
            descriptors, self.groups, self.output_rows = self._expert_descriptors(
                placed, expert_groups
            )
            self.experts = len(expert_groups)
        self.whole_groups = all(
            entry[5] == index * ROWS_PER_GROUP for index, entry in enumerate(dense)
        )
        self.blob = mx.concatenate(self._words)
        self._words = []
        self.descriptors = mx.array(descriptors, dtype=mx.uint32).reshape(-1)

    def arrays(self) -> list[mx.array]:
        """Return the arrays this bank keeps alive."""

        return [self.blob, self.descriptors, self.luts]

    @property
    def needs_raw_input(self) -> bool:
        """Return whether some experts read their input unrotated."""

        return bool(self.experts and self.format_mask >> FORMAT_IDS["nvfp4"] & 1)

    @property
    def nbytes(self) -> int:
        """Return packed bytes, excluding the shared tables."""

        return int(self.blob.nbytes + self.descriptors.nbytes)

    def _append(self, array: mx.array) -> int:
        words = as_words(array)
        offset = self._size
        self._words.append(words)
        self._size += int(words.size)
        return offset

    def _place(self, matrix: PackedRows) -> list[tuple[int, ...]]:
        bits = FORMAT_BITS[matrix.fmt]
        row_bytes = self.input_width * bits // 8
        block_bytes = matrix.rows * row_bytes
        if row_bytes % 4 or not block_bytes or matrix.codes.nbytes % block_bytes:
            raise ValueError(
                f"packed {matrix.fmt} codes do not hold whole blocks of "
                f"{matrix.rows} rows by {self.input_width} columns"
            )
        blocks = matrix.codes.nbytes // block_bytes
        code_offset = self._append(matrix.codes)
        scale_offset = 0
        aux_offset = 0
        outer: list[float] = [1.0] * blocks
        if matrix.fmt == "nvfp4":
            scale_bytes = matrix.rows * self.input_width // _CHUNK
            if matrix.scales.nbytes != blocks * scale_bytes or matrix.outer is None:
                raise ValueError("NVFP4 rows need block scales and outer scales")
            aux_offset = self._append(matrix.scales)
            outer = [float(value) for value in matrix.outer.reshape(-1).tolist()]
            if len(outer) != blocks or scale_bytes % 4:
                raise ValueError("NVFP4 outer scales do not match the blocks")
        else:
            if int(matrix.scales.size) != blocks * matrix.rows:
                raise ValueError(f"packed {matrix.fmt} scales do not match the rows")
            scale_offset = 2 * self._append(matrix.scales.astype(mx.float16))
            if matrix.fmt in ("lm6", "lm8"):
                if matrix.levels is None or int(matrix.levels.size) != 1 << bits:
                    raise ValueError(f"packed {matrix.fmt} needs {1 << bits} levels")
                aux_offset = self._append(matrix.levels.astype(mx.float32))
        fid = FORMAT_IDS[matrix.fmt]
        return [
            (
                fid,
                code_offset + block * block_bytes // 4,
                scale_offset + block * matrix.rows,
                aux_offset
                + (
                    block * matrix.rows * self.input_width // _CHUNK // 4
                    if matrix.fmt == "nvfp4"
                    else 0
                ),
                matrix.rows,
                row_bytes,
                _float_bits(outer[block]),
            )
            for block in range(blocks)
        ]

    def _split(
        self, block: tuple[int, ...], output_row: int, input_index: int
    ) -> list[tuple[int, ...]]:
        """Split one block into descriptors of at most one row group each."""

        fid, codes, scales, aux, rows, row_bytes, outer = block
        groups = []
        for first in range(0, rows, ROWS_PER_GROUP):
            block_scale_words = (
                first * self.input_width // _CHUNK // 4 if fid == 7 else 0
            )
            groups.append(
                (
                    fid,
                    codes + first * row_bytes // 4,
                    scales + (0 if fid == 7 else first),
                    aux + block_scale_words,
                    min(ROWS_PER_GROUP, rows - first),
                    output_row + first,
                    input_index,
                    outer,
                )
            )
        return groups

    def _expert_descriptors(
        self,
        placed: dict[int, list[tuple[int, ...]]],
        expert_groups: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[list[tuple[int, ...]], int, int]:
        descriptors: list[tuple[int, ...]] = []
        groups: int | None = None
        output_rows: int | None = None
        for expert in expert_groups:
            row = 0
            entries: list[tuple[int, ...]] = []
            for fid, index in expert:
                blocks = placed.get(int(fid))
                if blocks is None or not 0 <= int(index) < len(blocks):
                    raise ValueError("expert map names a missing packed block")
                block = blocks[int(index)]
                entries.extend(self._split(block, row, int(fid == FORMAT_IDS["nvfp4"])))
                row += block[4]
            if groups is None:
                groups, output_rows = len(entries), row
            elif groups != len(entries) or output_rows != row:
                raise ValueError("experts of one bank must have equal rows")
            descriptors.extend(entries)
        if groups is None or output_rows is None:
            raise ValueError("expert bank has no experts")
        return descriptors, groups, output_rows

    def dense(self, x: mx.array) -> mx.array:
        """Multiply every dense matrix of the bank with its rotated input.

        Args:
            x: ``(inputs, tokens, input_width)`` inputs.

        Returns:
            ``(tokens, output_rows)`` float32 products.
        """

        tokens = int(x.shape[1])
        blocked = tokens > _MAX_TOKENS
        max_tokens = _BLOCKED_TOKENS if blocked else tokens
        units = (tokens + max_tokens - 1) // max_tokens
        return _dense_kernel(blocked)(
            inputs=[
                self.blob,
                self.luts,
                self.descriptors,
                x,
                _meta(tokens, self.output_rows, 0, 0),
            ],
            template=self._template(max_tokens),
            grid=(units * _SIMDGROUPS * _SIMD_WIDTH, self.groups, 1),
            threadgroup=(_SIMDGROUPS * _SIMD_WIDTH, 1, 1),
            output_shapes=[(tokens, self.output_rows)],
            output_dtypes=[mx.float32],
        )[0]

    def rows(self, rows: mx.array) -> mx.array:
        """Decode individual rows of a dense bank.

        Args:
            rows: ``(count,)`` uint32 output rows.

        Returns:
            ``(count, input_width)`` float32 rows in the packed basis.

        Raises:
            ValueError: If a matrix other than the last has a partial group.
        """

        if not self.whole_groups:
            raise ValueError("packed row lookup needs whole 128-row groups")
        count = int(rows.shape[0])
        return _rows_kernel()(
            inputs=[self.blob, self.luts, self.descriptors, rows],
            template=[
                ("FORMATS", self.format_mask),
                ("N", self.input_width),
                ("SIMDGROUPS", _SIMDGROUPS),
                ("GROUP_ROWS", ROWS_PER_GROUP),
            ],
            grid=(count * _SIMDGROUPS * _SIMD_WIDTH, 1, 1),
            threadgroup=(_SIMDGROUPS * _SIMD_WIDTH, 1, 1),
            output_shapes=[(count, self.input_width)],
            output_dtypes=[mx.float32],
        )[0]

    def expert(
        self,
        x: mx.array,
        experts: mx.array,
        plan: ExpertPlan,
        *,
        x_div: int,
    ) -> mx.array:
        """Multiply each slot's input with that slot's expert.

        Args:
            x: ``(slots / x_div, input_width)`` rotated inputs, or
                ``(2, slots / x_div, input_width)`` with the unrotated inputs
                second when :attr:`needs_raw_input`.
            experts: ``(slots,)`` uint32 expert per slot.
            plan: Result of :func:`plan_experts` for ``experts``.
            x_div: Consecutive slots that share one input row.

        Returns:
            ``(slots, output_rows)`` float32 products.
        """

        slots = int(experts.shape[0])
        return _expert_kernel(plan.max_tokens > _MAX_TOKENS)(
            inputs=[
                self.blob,
                self.luts,
                self.descriptors,
                x,
                experts,
                plan.order,
                plan.units,
                _meta(slots, self.output_rows, self.groups, x_div),
            ],
            template=self._template(plan.max_tokens),
            grid=(plan.unit_count * _SIMDGROUPS * _SIMD_WIDTH, self.groups, 1),
            threadgroup=(_SIMDGROUPS * _SIMD_WIDTH, 1, 1),
            output_shapes=[(slots, self.output_rows)],
            output_dtypes=[mx.float32],
        )[0]

    def _template(self, max_tokens: int) -> list[tuple[str, Any]]:
        return [
            ("FORMATS", self.format_mask),
            ("N", self.input_width),
            ("LANES", self.lanes),
            ("SIMDGROUPS", _SIMDGROUPS),
            ("MAX_TOKENS", max_tokens),
        ]


@dataclass(frozen=True)
class ExpertPlan:
    """Grouping of expert slots so each expert's rows are decoded once.

    Attributes:
        order: Slots sorted by expert.
        units: Position in ``order`` where each unit starts; ``slots`` marks an
            unused unit.
        unit_count: Launched units.
        max_tokens: Slots one unit may hold.
    """

    order: mx.array
    units: mx.array
    unit_count: int
    max_tokens: int


def plan_experts(experts: mx.array, num_experts: int, tokens: int) -> ExpertPlan:
    """Group slots that use the same expert into units.

    A unit holds at most eight slots when the step has at most eight tokens, and
    at most 32 otherwise.

    Args:
        experts: ``(slots,)`` uint32 expert per slot.
        num_experts: Experts in the bank.
        tokens: Tokens the slots belong to. A token's experts are distinct, so
            no expert owns more slots than this.

    Returns:
        The grouping used by :meth:`PackedBank.expert`.
    """

    slots = int(experts.shape[0])
    if tokens <= _MAX_TOKENS:
        order, units = _plan_kernel()(
            inputs=[experts],
            template=[("EXPERTS", num_experts)],
            grid=(slots, 1, 1),
            threadgroup=(min(slots, _SIMD_WIDTH), 1, 1),
            output_shapes=[(slots,), (slots,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        return ExpertPlan(order, units, slots, _MAX_TOKENS)
    order = mx.argsort(experts).astype(mx.uint32)
    ordered = experts[order]
    positions = mx.arange(slots, dtype=mx.uint32)
    first = mx.concatenate([mx.array([True]), ordered[1:] != ordered[:-1]])
    run_start = mx.cummax(mx.where(first, positions, mx.array(0, dtype=mx.uint32)))
    leads = (positions - run_start) % _BLOCKED_TOKENS == 0
    starts = mx.where(leads, positions, mx.array(slots, dtype=mx.uint32))
    unit_count = min(slots, slots // _BLOCKED_TOKENS + num_experts)
    units = mx.sort(starts)[:unit_count]
    return ExpertPlan(order, units, unit_count, _BLOCKED_TOKENS)


_NO_PLAN: dict[int, ExpertPlan] = {}


def single_token_plan(slots: int) -> ExpertPlan:
    """Return the plan for one token, whose experts are always distinct."""

    plan = _NO_PLAN.get(slots)
    if plan is None:
        empty = mx.zeros((1,), dtype=mx.uint32)
        plan = ExpertPlan(empty, empty, slots, 1)
        _NO_PLAN[slots] = plan
    return plan


def _float_bits(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", value))[0])
