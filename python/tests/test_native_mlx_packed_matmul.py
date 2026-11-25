"""Packed matrix products against a NumPy decoder."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_worker.native_mlx.packed_matmul import (
    FORMAT_IDS,
    PackedBank,
    PackedRows,
    build_trellis_luts,
    lanes_for,
    plan_experts,
    single_token_plan,
)
from mlx_worker.native_mlx.rotation import (
    hadamard_factors,
    rotate_out,
    rotate_packed,
    rotate_ranges,
)

from .packed_weights import decode, random_codes, random_rows, random_tables

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="packed kernels require Metal"
)

_COLUMNS = 96


def _luts(tables: dict[str, np.ndarray]) -> mx.array:
    return build_trellis_luts({fmt: mx.array(table) for fmt, table in tables.items()})


def _close(actual: mx.array, expected: np.ndarray) -> None:
    scale = np.abs(expected).max() + 1.0
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5 * scale, rtol=0)


@pytest.mark.parametrize("fmt", sorted(FORMAT_IDS))
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 9, 20, 32, 33, 34, 35, 36, 40, 60])
def test_dense_bank_multiplies_decoded_rows(fmt: str, tokens: int) -> None:
    rng = np.random.default_rng(FORMAT_IDS[fmt])
    tables = random_tables(rng)
    packed, weights = random_rows(rng, fmt, 132, _COLUMNS, tables)
    bank = PackedBank(_COLUMNS, [packed], _luts(tables))
    x = rng.normal(size=(tokens, _COLUMNS)).astype(np.float16)

    actual = bank.dense(mx.array(x)[None])

    _close(actual, x.astype(np.float32) @ weights.T)


@pytest.mark.parametrize("tokens", [1, 5, 40])
def test_dense_bank_mixes_formats_and_inputs(tokens: int) -> None:
    rng = np.random.default_rng(3)
    tables = random_tables(rng)
    specs = [("rtr4", 64, 0), ("ht3", 32, 1), ("lm6", 130, 1), ("rtr1", 16, 0)]
    matrices, weights, start = [], [], 0
    for fmt, rows, index in specs:
        packed, dense = random_rows(
            rng, fmt, rows, _COLUMNS, tables, output_row=start, input_index=index
        )
        matrices.append(packed)
        weights.append(dense)
        start += rows
    bank = PackedBank(_COLUMNS, matrices, _luts(tables))
    x = rng.normal(size=(2, tokens, _COLUMNS)).astype(np.float16)

    actual = bank.dense(mx.array(x))

    expected = [
        x[index].astype(np.float32) @ dense.T
        for (_, _, index), dense in zip(specs, weights, strict=True)
    ]
    _close(actual, np.concatenate(expected, axis=1))


def test_rows_decode_individual_rows_of_every_format() -> None:
    rng = np.random.default_rng(4)
    tables = random_tables(rng)
    formats = [fmt for fmt in FORMAT_IDS if fmt != "nvfp4"]
    matrices, weights = [], []
    for index, fmt in enumerate(formats):
        packed, dense = random_rows(
            rng, fmt, 128, _COLUMNS, tables, output_row=128 * index
        )
        matrices.append(packed)
        weights.append(dense)
    bank = PackedBank(_COLUMNS, matrices, _luts(tables))
    rows = rng.integers(0, 128 * len(formats), size=40)

    actual = bank.rows(mx.array(rows, dtype=mx.uint32))

    _close(actual, np.concatenate(weights)[rows])


def test_rows_need_whole_groups() -> None:
    rng = np.random.default_rng(5)
    tables = random_tables(rng)
    first, _ = random_rows(rng, "lm8", 64, _COLUMNS, tables)
    second, _ = random_rows(rng, "lm8", 64, _COLUMNS, tables, output_row=64)
    bank = PackedBank(_COLUMNS, [first, second], _luts(tables))

    with pytest.raises(ValueError, match="whole 128-row groups"):
        bank.rows(mx.array([0], dtype=mx.uint32))


def _expert_bank(
    rng: np.random.Generator, formats: list[str], groups_per_expert: int
) -> tuple[PackedBank, list[np.ndarray]]:
    """Return a bank of eight experts of mixed formats and their weights."""

    tables = random_tables(rng)
    experts, rows = 8, 48 if groups_per_expert == 1 else 128
    blocks = experts * groups_per_expert
    choice = [formats[index % len(formats)] for index in range(blocks)]
    matrices, weights = [], {}
    for fmt in formats:
        owned = [index for index in range(blocks) if choice[index] == fmt]
        codes = np.stack([random_codes(rng, fmt, rows, _COLUMNS) for _ in owned])
        outer = None
        if fmt == "nvfp4":
            shape = (len(owned), rows, _COLUMNS // 16)
            scales = rng.integers(0, 120, size=shape, dtype=np.uint8)
            outer = rng.uniform(0.5, 2.0, size=(len(owned),)).astype(np.float32)
        else:
            scales = rng.uniform(0.5, 2.0, size=(len(owned), rows)).astype(np.float16)
        matrices.append(
            PackedRows(
                fmt,
                rows,
                mx.array(codes),
                mx.array(scales),
                outer=None if outer is None else mx.array(outer),
            )
        )
        for position, index in enumerate(owned):
            weights[index] = decode(
                fmt,
                codes[position],
                _COLUMNS,
                scales[position],
                tables,
                outer=None if outer is None else outer[position : position + 1],
            )
    taken = dict.fromkeys(formats, 0)
    table = []
    for expert in range(experts):
        entry = []
        for group in range(groups_per_expert):
            fmt = choice[expert * groups_per_expert + group]
            entry.append((FORMAT_IDS[fmt], taken[fmt]))
            taken[fmt] += 1
        table.append(entry)
    bank = PackedBank(_COLUMNS, matrices, _luts(tables), expert_groups=table)
    dense = [
        np.concatenate(
            [
                weights[expert * groups_per_expert + group]
                for group in range(groups_per_expert)
            ]
        )
        for expert in range(experts)
    ]
    return bank, dense


@pytest.mark.parametrize(
    ("formats", "groups"),
    [(["ht1", "ht2", "rtr1"], 1), (["nvfp4", "ht3"], 1), (["ht2", "rtr2", "rtr3"], 2)],
)
@pytest.mark.parametrize("tokens", [1, 4, 9, 30, 120])
def test_expert_bank_multiplies_each_slot_with_its_expert(
    formats: list[str], groups: int, tokens: int
) -> None:
    rng = np.random.default_rng(tokens)
    bank, weights = _expert_bank(rng, formats, groups)
    top = 3
    chosen = np.stack([rng.permutation(8)[:top] for _ in range(tokens)]).reshape(-1)
    experts = mx.array(chosen, dtype=mx.uint32)
    rotated = rng.normal(size=(tokens, _COLUMNS)).astype(np.float16)
    raw = rng.normal(size=(tokens, _COLUMNS)).astype(np.float16)
    x = np.stack([rotated, raw]) if bank.needs_raw_input else rotated
    plan = single_token_plan(top) if tokens == 1 else plan_experts(experts, 8, tokens)

    actual = bank.expert(mx.array(x), experts, plan, x_div=top)

    assert bank.needs_raw_input == ("nvfp4" in formats)
    expected = []
    for slot, expert in enumerate(chosen):
        unrotated = bank.needs_raw_input and expert % len(formats) == 0
        source = raw if unrotated else rotated
        expected.append(weights[expert] @ source[slot // top].astype(np.float32))
    _close(actual, np.stack(expected))


def test_plan_puts_at_most_32_slots_of_one_expert_in_a_unit() -> None:
    experts = mx.array([5] * 70 + [2] * 3 + [7], dtype=mx.uint32)

    plan = plan_experts(experts, 8, 70)

    ordered = np.array(experts)[np.array(plan.order)]
    starts = [int(start) for start in np.array(plan.units) if start < 74]
    assert ordered.tolist() == [2] * 3 + [5] * 70 + [7]
    assert starts == [0, 3, 35, 67, 73]
    assert plan.unit_count >= len(starts)


def test_plan_of_a_small_step_lists_every_expert_once() -> None:
    experts = mx.array([7, 2, 33, 2, 7, 0, 33, 64, 2], dtype=mx.uint32)

    plan = plan_experts(experts, 65, 3)

    assert np.array(plan.order).tolist() == [5, 1, 3, 8, 0, 4, 2, 6, 7]
    assert np.array(plan.units).tolist() == [0, 1, 4, 6, 8, 9, 9, 9, 9]
    assert plan.unit_count == 9


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"codes": mx.zeros((5,), dtype=mx.uint8)}, "whole blocks"),
        ({"scales": mx.zeros((3,), dtype=mx.float16)}, "scales do not match"),
        ({"fmt": "lm6", "codes": mx.zeros((4, 72), dtype=mx.uint8)}, "needs 64 levels"),
        (
            {"fmt": "nvfp4", "codes": mx.zeros((4, 48), dtype=mx.uint8)},
            "block scales and outer",
        ),
    ],
)
def test_bank_rejects_inconsistent_matrices(rows: dict, message: str) -> None:
    values = {
        "fmt": "rtr1",
        "rows": 4,
        "codes": mx.zeros((4, 12), dtype=mx.uint8),
        "scales": mx.zeros((4,), dtype=mx.float16),
    }

    with pytest.raises(ValueError, match=message):
        PackedBank(_COLUMNS, [PackedRows(**{**values, **rows})], build_trellis_luts({}))


def test_bank_rejects_a_missing_expert_block() -> None:
    rows = PackedRows(
        "rtr1",
        4,
        mx.zeros((1, 4, 12), dtype=mx.uint8),
        mx.zeros((1, 4), dtype=mx.float16),
    )

    with pytest.raises(ValueError, match="missing packed block"):
        PackedBank(
            _COLUMNS,
            [rows],
            build_trellis_luts({}),
            expert_groups=[[(FORMAT_IDS["rtr1"], 1)]],
        )


def test_tables_and_widths_are_validated() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        build_trellis_luts({"ht1": mx.zeros((4, 4))})
    with pytest.raises(ValueError, match="multiple of 32"):
        lanes_for(40)
    assert [lanes_for(width) for width in (32, 96, 1152, 2304)] == [2, 2, 8, 16]


@pytest.mark.parametrize("width", [32, 64, 96, 1152, 2304, 3072])
def test_rotation_is_an_orthogonal_hadamard_transform(width: int) -> None:
    rng = np.random.default_rng(width)
    first, second = (np.array(factor) for factor in hadamard_factors(width))
    matrix = np.kron(first, second)
    np.testing.assert_array_equal(matrix @ matrix.T, width * np.eye(width))

    x = rng.normal(size=(3, width)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=width).astype(np.float32)
    rotated = rotate_packed(mx.array(x)[None], mx.array(signs)[None], dtype=mx.float32)[
        0
    ]

    expected = (x * signs) @ matrix.T / np.sqrt(width)
    np.testing.assert_allclose(np.array(rotated), expected, atol=1e-5)
    np.testing.assert_allclose(
        np.array(rotate_out(rotated, mx.array(signs))), x, atol=1e-5
    )
    np.testing.assert_allclose(
        np.array(rotate_ranges(rotated, [(0, width)], [mx.array(signs)])), x, atol=1e-5
    )


def test_one_launch_rotates_several_matrices_and_gated_inputs() -> None:
    rng = np.random.default_rng(3)
    width, tokens = 96, 5
    matrix = np.kron(*(np.array(factor) for factor in hadamard_factors(width)))
    x = rng.normal(size=(2, tokens, width)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(3, width)).astype(np.float32)

    def rotated(values: np.ndarray, row: int) -> np.ndarray:
        return (values * signs[row]) @ matrix.T / np.sqrt(width)

    several = rotate_packed(
        mx.array(x),
        mx.array(signs),
        inputs=mx.array([1, 0, 1], dtype=mx.uint32),
        dtype=mx.float32,
        keep_input=True,
    )
    expected = [rotated(x[1], 0), rotated(x[0], 1), rotated(x[1], 2), x[1]]
    np.testing.assert_allclose(np.array(several), np.stack(expected), atol=1e-5)

    gate = rng.normal(size=(tokens, width + 7)).astype(np.float32)
    gated = rotate_packed(
        mx.array(x[:1]), mx.array(signs[:1]), dtype=mx.float32, gate=(mx.array(gate), 7)
    )
    np.testing.assert_allclose(
        np.array(gated[0]), rotated(x[0] / (1.0 + np.exp(-gate[:, 7:])), 0), atol=1e-5
    )

    halves = rng.normal(size=(1, tokens, 2 * width)).astype(np.float32) * 3.0
    soft = rotate_packed(
        mx.array(halves),
        mx.array(signs[:1]),
        dtype=mx.float32,
        soft_caps=mx.array([4.0, 2.0]),
        keep_input=True,
    )
    g, u = halves[0, :, :width], halves[0, :, width:]
    product = np.tanh(g / 4.0) * 4.0 / (1.0 + np.exp(-g)) * np.tanh(u / 2.0) * 2.0
    np.testing.assert_allclose(np.array(soft[0]), rotated(product, 0), atol=1e-4)
    np.testing.assert_allclose(np.array(soft[1]), product, atol=1e-4)


def test_one_launch_rotates_column_ranges_back() -> None:
    rng = np.random.default_rng(4)
    ranges = [(0, 96), (96, 128), (128, 224)]
    signs: list[mx.array | None] = [
        mx.array(rng.choice([-1.0, 1.0], size=96).astype(np.float32)),
        None,
        mx.array(rng.choice([-1.0, 1.0], size=96).astype(np.float32)),
    ]
    y = mx.array(rng.normal(size=(5, 224)).astype(np.float32))

    out = np.array(rotate_ranges(y, ranges, signs))

    np.testing.assert_array_equal(out[:, 96:128], np.array(y[:, 96:128]))
    for (start, end), flips in zip(ranges, signs, strict=True):
        if flips is not None:
            expected = np.array(rotate_out(y[:, start:end], flips))
            np.testing.assert_allclose(out[:, start:end], expected, atol=1e-5)


def test_rotation_rejects_widths_without_a_construction() -> None:
    with pytest.raises(ValueError, match="no Hadamard construction"):
        hadamard_factors(22)
