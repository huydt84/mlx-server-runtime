"""Relative-bias paged attention checks against a dense reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_worker.native_mlx import relative_attention
from mlx_worker.native_mlx.relative_attention import relative_paged_attention

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="relative attention requires Metal"
)

_HEADS, _KV_HEADS, _DIM, _PAGE, _REL = 6, 2, 64, 8, 16


def _reference(queries, keys, values, rel, rel_proj, sinks, positions, base, window):
    """Return dense attention output for one row's query tokens."""

    groups = _HEADS // _KV_HEADS
    outputs = np.zeros((len(positions), _HEADS, _DIM))
    for token, position in enumerate(positions):
        for head in range(_HEADS):
            scores, rows = [sinks[token, head]], [np.zeros(_DIM)]
            first = max(base, position - window + 1) if window else base
            for key in range(first, position + 1):
                bias = 0.0
                if position - key < len(rel_proj):
                    bias = rel[token, head] @ rel_proj[position - key]
                scores.append(
                    queries[token, head] @ keys[key, head // groups] * 0.1 + bias
                )
                rows.append(values[key, head // groups])
            weights = np.exp(np.array(scores) - np.max(scores))
            outputs[token, head] = (weights / weights.sum()) @ np.array(rows)
    return outputs.reshape(len(positions), -1)


@pytest.mark.parametrize("window", [None, 20])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("tokens", [1, 5])
def test_relative_attention_matches_dense_reference(
    monkeypatch, window, splits, tokens
) -> None:
    monkeypatch.setattr(relative_attention, "splits_for", lambda *_: splits)
    monkeypatch.setattr(relative_attention, "_MIN_KEYS_PER_SPLIT", 4)
    rng = np.random.default_rng(7)
    offsets = [37, 9]
    counts = [tokens, max(1, tokens - 2)]
    bases = [16, 0] if window else [0, 0]
    extent = 12
    pages = rng.normal(size=(40, _PAGE, 2, _KV_HEADS, _DIM)).astype(np.float16)
    order = rng.permutation(40)
    tables = np.stack([order[:8], order[8:16]]).astype(np.int32)
    queries = rng.normal(size=(2, tokens, _HEADS, _DIM)).astype(np.float32)
    rel = rng.normal(size=(2, tokens, _HEADS, _REL)).astype(np.float32)
    rel_proj = rng.normal(size=(extent, _REL)).astype(np.float32)
    sinks = rng.normal(size=(2, tokens, _HEADS)).astype(np.float32)

    token_rows = [0] * counts[0] + [1] * counts[1]
    positions = [offsets[0] + token for token in range(counts[0])]
    positions += [offsets[1] + token for token in range(counts[1])]
    flat = [(row, token) for row in range(2) for token in range(counts[row])]
    flat_rows, flat_tokens = (list(axis) for axis in zip(*flat, strict=True))
    actual = np.array(
        relative_paged_attention(
            mx.array(queries[flat_rows, flat_tokens]),
            mx.array(pages),
            mx.array(tables),
            mx.array(bases, dtype=mx.int32),
            mx.array(token_rows, dtype=mx.int32),
            mx.array(positions, dtype=mx.int32),
            mx.array(rel[flat_rows, flat_tokens]),
            mx.array(rel_proj),
            mx.array(sinks[flat_rows, flat_tokens]),
            max_length=max(offsets) + tokens,
            scale=0.1,
            window_size=window,
        )
    )

    start = 0
    for row in range(2):
        slots = pages[tables[row]].reshape(-1, 2, _KV_HEADS, _DIM).astype(np.float64)
        padded = np.zeros((bases[row], 2, _KV_HEADS, _DIM))
        history = np.concatenate([padded, slots])
        expected = _reference(
            queries[row],
            history[:, 0],
            history[:, 1],
            rel[row],
            rel_proj,
            sinks[row],
            [offsets[row] + token for token in range(counts[row])],
            bases[row],
            window,
        )
        np.testing.assert_allclose(
            actual[start : start + counts[row]], expected, rtol=2e-4, atol=2e-5
        )
        start += counts[row]
