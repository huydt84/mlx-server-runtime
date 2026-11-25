"""Inkling-2 kernels against the MLX operations they replace."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_worker.native_mlx.causal_conv import (
    CONV_LAGS,
    TokenLayout,
    causal_conv,
    causal_keys_values,
)
from mlx_worker.native_mlx.inkling2_kernels import (
    blend_messages,
    mix_states,
    route_experts,
    row_products,
    scaled_clip,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Inkling-2 kernels require Metal"
)

_LAYOUT = TokenLayout((10, 0, 4), (1, 6, 2))


def _rows() -> list[range]:
    return [range(1), range(1, 7), range(7, 9)]


def _convolved(x: np.ndarray, history: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return ``x + conv(x)`` computed one row and lag at a time."""

    out = x.astype(np.float64).copy()
    for row, tokens in enumerate(_rows()):
        joined = np.concatenate([history[row], x[list(tokens)]])
        for index, token in enumerate(tokens):
            for lag in range(CONV_LAGS):
                weight = weights[token, lag] if weights.ndim == 3 else weights[lag]
                out[token] += weight * joined[index + CONV_LAGS - 1 - lag]
    return out


def _next_history(x: np.ndarray, history: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.concatenate([history[row], x[list(tokens)]])[-(CONV_LAGS - 1) :]
            for row, tokens in enumerate(_rows())
        ]
    )


@pytest.mark.parametrize("residual", [False, True])
def test_causal_conv_reads_history_across_unpadded_rows(residual: bool) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(9, 40)).astype(np.float32)
    history = rng.normal(size=(3, CONV_LAGS - 1, 40)).astype(np.float32)
    weights = rng.normal(size=(CONV_LAGS, 40)).astype(np.float32)
    extra = rng.normal(size=(9, 40)).astype(np.float32)

    out, following = causal_conv(
        mx.array(x),
        mx.array(history),
        mx.array(weights),
        _LAYOUT,
        scale=0.4,
        residual=mx.array(extra) if residual else None,
    )

    expected = _convolved(x, history, weights) * 0.4 + (extra if residual else 0.0)
    np.testing.assert_allclose(np.array(out), expected, atol=1e-5)
    np.testing.assert_array_equal(np.array(following), _next_history(x, history))


@pytest.mark.parametrize("dynamic", [False, True])
def test_keys_are_clipped_and_values_normalized(dynamic: bool) -> None:
    rng = np.random.default_rng(2)
    kv_heads, dim, offset = 2, 64, 24
    heads = 2 * kv_heads
    channels = heads * dim
    x = rng.normal(size=(9, offset + channels + 8)).astype(np.float32)
    history = rng.normal(size=(3, CONV_LAGS - 1, channels)).astype(np.float32)
    weights = rng.normal(size=(CONV_LAGS, channels)).astype(np.float32)
    per_head = rng.normal(size=(9, heads, CONV_LAGS)).astype(np.float32)

    kv, following = causal_keys_values(
        mx.array(x),
        mx.array(history),
        mx.array(weights),
        _LAYOUT,
        offset=offset,
        heads=heads,
        head_dim=dim,
        scale=0.4,
        clip=5.0,
        eps=1e-6,
        dynamic_weights=mx.array(per_head) if dynamic else None,
        dtype=mx.float32,
    )

    inputs = x[:, offset : offset + channels]
    full = np.broadcast_to(weights, (9, CONV_LAGS, channels)).copy()
    if dynamic:
        full += np.repeat(per_head.transpose(0, 2, 1), dim, axis=-1)
    out = (_convolved(inputs, history, full) * 0.4).reshape(9, 2, kv_heads, dim)
    norms = np.linalg.norm(out[:, 0], axis=-1, keepdims=True)
    keys = out[:, 0] * np.minimum(5.0 / (norms + 1e-6), 1.0)
    values = out[:, 1] / np.sqrt((out[:, 1] ** 2).mean(-1, keepdims=True) + 1e-6)
    assert (norms > 5.0).any()
    np.testing.assert_allclose(np.array(kv), np.stack([keys, values], 1), atol=2e-4)
    np.testing.assert_array_equal(np.array(following), _next_history(inputs, history))


@pytest.mark.parametrize("states", [1, 3, 6])
@pytest.mark.parametrize("hidden", [64, 2304])
def test_mix_states_is_a_softmax_over_normalized_states(
    states: int, hidden: int
) -> None:
    rng = np.random.default_rng(states)
    queries = rng.normal(size=(4, hidden)).astype(np.float32)
    values = rng.normal(size=(states, 5, hidden)).astype(np.float32) * 3.0

    actual = mix_states(mx.array(queries), [mx.array(value) for value in values])

    normed = mx.fast.rms_norm(mx.array(values), None, 1e-6)
    scores = normed @ mx.array(queries).T / hidden**0.5
    weights = mx.softmax(scores, axis=0)
    expected = mx.einsum("stq,sth->qth", weights, normed)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-5)


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_scaled_clip_limits_each_head(per_head: bool, scaled: bool) -> None:
    rng = np.random.default_rng(3)
    heads, dim, rel_dim, start = 4, 64, 16, 300
    x = rng.normal(size=(5, start + heads * rel_dim + 7)).astype(np.float32)
    lead = (heads,) if per_head else ()
    q_scale = rng.uniform(0.5, 2.0, size=(*lead, dim)).astype(np.float32)
    rel_scale = rng.uniform(0.5, 2.0, size=(*lead, rel_dim)).astype(np.float32)
    factor = rng.uniform(1.0, 2.0, size=(5,)).astype(np.float32)

    queries, rel = scaled_clip(
        mx.array(x),
        mx.array(q_scale),
        mx.array(rel_scale),
        heads=heads,
        head_dim=dim,
        rel_start=start,
        rel_dim=rel_dim,
        clips=(8.0, 4.0),
        token_scale=mx.array(factor) if scaled else None,
    )

    def expected(values: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
        values = values * scale
        norm = np.linalg.norm(values, axis=-1, keepdims=True)
        values = values * np.minimum(clip / (norm + 1e-6), 1.0)
        return values * factor[:, None, None] if scaled else values

    wanted = expected(x[:, : heads * dim].reshape(5, heads, dim), q_scale, 8.0)
    np.testing.assert_allclose(np.array(queries), wanted, atol=1e-5)
    wanted = expected(
        x[:, start : start + heads * rel_dim].reshape(5, heads, rel_dim), rel_scale, 4.0
    )
    np.testing.assert_allclose(np.array(rel), wanted, atol=1e-5)


def test_route_experts_biases_the_choice_but_not_the_weights() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(7, 32)).astype(np.float32)
    bias = rng.normal(size=(32,)).astype(np.float32) * 0.2

    chosen, weights = route_experts(mx.array(logits), mx.array(bias), top=4, scale=1.5)

    gates = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(-(gates + bias), axis=-1)[:, :4]
    np.testing.assert_array_equal(np.array(chosen), order)
    picked = np.take_along_axis(gates, order, axis=-1)
    np.testing.assert_allclose(
        np.array(weights), picked / picked.sum(-1, keepdims=True) * 1.5, atol=1e-5
    )


@pytest.mark.parametrize(("hidden", "rank"), [(64, 16), (256, 64)])
@pytest.mark.parametrize(("gates", "shared"), [(1, False), (3, True), (3, False)])
def test_messages_blend_through_low_rank_gates(
    hidden: int, rank: int, gates: int, shared: bool
) -> None:
    rng = np.random.default_rng(6)
    tokens = 5
    states = rng.normal(size=(1 if shared else gates, tokens, hidden))
    messages = rng.normal(size=(gates, tokens, hidden))
    down = rng.normal(size=(gates, 2 * hidden, rank)) / np.sqrt(hidden)
    up = rng.normal(size=(gates, rank, 2 * hidden)) / np.sqrt(rank)

    out = blend_messages(
        *(mx.array(value.astype(np.float32)) for value in (states, messages, down, up))
    )

    def normed(value: np.ndarray) -> np.ndarray:
        return value / np.sqrt((value**2).mean(axis=-1, keepdims=True) + 1e-6)

    wide = np.broadcast_to(states, messages.shape)
    joined = np.concatenate([normed(wide), normed(messages)], axis=-1)
    forget, keep = np.split(joined @ down @ up, 2, axis=-1)
    expected = (1.0 - np.tanh(forget)) * wide + np.tanh(keep) * messages
    np.testing.assert_allclose(np.array(out), expected, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 5, 24, 128])
def test_row_products_do_not_depend_on_the_step_size(rows: int) -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(5, 96)).astype(np.float32)
    weight = rng.normal(size=(rows, 96)).astype(np.float32)

    out = np.array(row_products(mx.array(x), mx.array(weight)))

    np.testing.assert_allclose(out, x @ weight.T, atol=1e-4)
    for token in range(5):
        alone = row_products(mx.array(x[token : token + 1]), mx.array(weight))
        np.testing.assert_array_equal(np.array(alone[0]), out[token])
