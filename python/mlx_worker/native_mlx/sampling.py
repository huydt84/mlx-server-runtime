"""Device-side next-token selection for native batches.

Greedy rows take ``argmax`` exactly as before. Sampled rows divide their
logits by the temperature, keep the smallest set of tokens whose probability
reaches ``top_p``, and draw one token with the Gumbel-max trick.

Each sampled row draws its noise from a key derived from the request's seed
and the absolute position of the token being chosen. A request therefore gets
the same tokens whichever other requests share its batch and however its
prompt was chunked, and the draw never leaves the device, so overlap mode can
feed sampled tokens straight into the next step.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .interfaces import SamplingParams

_MIN_TEMPERATURE = 1e-5
_SEED_MASK = (1 << 63) - 1


def is_greedy(sampling: SamplingParams) -> bool:
    """Return whether ``sampling`` selects the most likely token."""

    return sampling.temperature <= 0.0


def sample_next_tokens(
    logits: mx.array,
    samplings: Sequence[SamplingParams],
    positions: Sequence[int],
) -> mx.array:
    """Select one next token per row.

    Args:
        logits: ``(batch, vocab)`` last-position logits.
        samplings: Per-row sampling parameters.
        positions: Per-row absolute position of the token being chosen.

    Returns:
        ``(batch,)`` int32 token ids, still on the device.
    """

    if all(is_greedy(sampling) for sampling in samplings):
        return mx.argmax(logits, axis=-1)
    rows = [
        _sample_row(logits[index], sampling, position)
        if not is_greedy(sampling)
        else mx.argmax(logits[index], axis=-1)
        for index, (sampling, position) in enumerate(
            zip(samplings, positions, strict=True)
        )
    ]
    return mx.stack(rows).astype(mx.uint32)


def _sample_row(logits: mx.array, sampling: SamplingParams, position: int) -> mx.array:
    scaled = logits.astype(mx.float32) / max(sampling.temperature, _MIN_TEMPERATURE)
    noise = _gumbel(int(scaled.shape[-1]), sampling.seed, position)
    if sampling.top_p >= 1.0:
        return mx.argmax(scaled + noise, axis=-1)
    order = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take(scaled, order, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    exceeded = (mx.cumsum(sorted_probs, axis=-1) - sorted_probs) >= sampling.top_p
    kept = mx.where(exceeded, -mx.inf, sorted_logits)
    choice = mx.argmax(kept + noise, axis=-1)
    return mx.take(order, choice, axis=-1)


def _gumbel(vocab: int, seed: int, position: int) -> mx.array:
    key = mx.random.key(_row_seed(seed, position))
    uniform = mx.random.uniform(
        low=1e-20, high=1.0, shape=(vocab,), dtype=mx.float32, key=key
    )
    return -mx.log(-mx.log(uniform))


def _row_seed(seed: int, position: int) -> int:
    value = (seed * 0x9E3779B97F4A7C15 + position + 1) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    return value & _SEED_MASK
