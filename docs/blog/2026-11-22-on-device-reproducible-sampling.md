---
layout: post
title: "On-Device Sampling That Survives Batching"
author: "MLX Air Team"
date: 2026-11-22
tags: [learning, sampling, determinism, mlx]
excerpt: "Temperature and top-p in a few lines of MLX, the Gumbel-max trick, and why every random draw is keyed by request seed and token position so that batch composition, prompt chunking, and overlap mode cannot change it."
---

**Table of contents**

- [Why sampling belongs on the device](#why-sampling-belongs-on-the-device)
- [Only the logits you need](#only-the-logits-you-need)
- [Greedy is argmax](#greedy-is-argmax)
- [Temperature and top-p](#temperature-and-top-p)
- [The Gumbel-max trick](#the-gumbel-max-trick)
- [Randomness that does not depend on the batch](#randomness-that-does-not-depend-on-the-batch)
- [What is and is not reproducible](#what-is-and-is-not-reproducible)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)

Sampling is the smallest part of a decode step, a few hundred microseconds at
most. It is also the part most likely to drag the whole engine back to
serial execution or to make outputs depend on who else was in the batch. This
short post covers how MLX Air's native backend samples
(`python/mlx_worker/native_mlx/sampling.py`).

## Why sampling belongs on the device

The overlap scheduler in [post 6](2026-11-22-overlap-scheduling-mlx-async-eval.md)
feeds step N's sampled tokens into step N+1 *while they are still on the
GPU*. That only works if the sampled token is an MLX array produced by GPU
operations. If sampling copied logits to the CPU and used Python's `random`,
every step would need a device-to-host read before the next step could be
built, and the overlap would be gone.

So sampling is written entirely in MLX operations and returns a `(batch,)`
token array that never leaves the device until the step resolves.

## Only the logits you need

The language-model head is a `hidden x vocab` matrix, and vocabularies are
large (about 150k tokens for Qwen). Projecting every position of a 256-token
prefill chunk onto the vocabulary produces a `256 x 150k` float tensor, and
then sampling uses one row of it.

`ForwardBatch` carries `logits_positions`, one per row: the last real token of
that row. Models apply the LM head to those positions only (commit `c942ba0`).
For a prefill chunk this skips 255 of 256 vocabulary projections and the
memory for their logits.

## Greedy is argmax

`temperature: 0` means greedy decoding. If every row in the batch is greedy,
the whole step is one `mx.argmax(logits, axis=-1)`. It is bit-identical to
argmax by definition, which is what makes the serial/overlap parity checks
in post 6 exact.

## Temperature and top-p

For a sampled row:

```python
def _sample_row(logits, sampling, position):
    scaled = logits.astype(mx.float32) / max(sampling.temperature, 1e-5)
    noise = _gumbel(vocab, sampling.seed, position)
    if sampling.top_p >= 1.0:
        return mx.argmax(scaled + noise, axis=-1)
    order = mx.argsort(-scaled, axis=-1)
    sorted_logits = mx.take(scaled, order, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    exceeded = (mx.cumsum(sorted_probs, axis=-1) - sorted_probs) >= sampling.top_p
    kept = mx.where(exceeded, -mx.inf, sorted_logits)
    choice = mx.argmax(kept + noise, axis=-1)
    return mx.take(order, choice, axis=-1)
```

**Temperature** divides the logits. Below 1 it sharpens the distribution,
above 1 it flattens it.

**Top-p (nucleus) sampling** keeps the smallest set of most-likely tokens
whose probabilities sum to at least `top_p`, and samples among those. The
mask line reads: drop a token if the tokens *before it* in sorted order
already reach `top_p`. Subtracting the token's own probability from the
cumulative sum means the most likely token is always kept, even when it
alone exceeds `top_p`.

Note that the Gumbel noise is indexed in *sorted* order here. That is fine:
any fixed assignment of independent noise to candidates gives a correct
sample.

## The Gumbel-max trick

Sampling from a categorical distribution usually means computing a CDF and
binary-searching a uniform random number in it. The **Gumbel-max trick** is
simpler on a GPU:

```text
g_i = -log(-log(u_i)),  u_i ~ Uniform(0, 1)
sample = argmax_i (logit_i + g_i)
```

Adding independent Gumbel noise to each logit and taking the argmax yields
token `i` with probability exactly `softmax(logits)_i`. It needs no
normalization, no cumulative sum, and no search: one random vector, one add,
one argmax. With top-p, the dropped tokens are `-inf` and can never win.

## Randomness that does not depend on the batch

The usual way to sample in a batch is to draw one random tensor of shape
`(batch, vocab)` from a global generator. That makes a request's output
depend on its row index and on how many random numbers earlier steps used,
which in turn depend on who else was running. Run the same prompt twice under
different load and you get different text even with the same seed.

MLX Air instead derives the key for each draw from two numbers that belong to
the request alone: its **seed** and the **absolute position** of the token
being chosen.

```python
def _row_seed(seed, position):
    value = (seed * 0x9E3779B97F4A7C15 + position + 1) & MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    return value & ((1 << 63) - 1)

key = mx.random.key(_row_seed(sampling.seed, position))
```

The mixing is SplitMix64-style, so neighboring positions get unrelated keys
rather than keys that differ in one bit.

Because the key depends only on (seed, position), the random draw for a
given token is the same whether the request:

- runs alone or in a batch of eight;
- sits in row 0 or row 5;
- had its prompt prefilled in one chunk or four;
- ran in `serial` or `overlap` mode, including the one discarded token after
  EOS in overlap mode.

## What is and is not reproducible

We want to be precise here, because "deterministic sampling" is often
over-claimed.

- **The random draw** is a pure function of seed and position. That part is
  reproducible.
- **The logits** are not bit-identical across batch compositions. A matrix
  multiply over a batch of 8 may accumulate in a different order than over a
  batch of 1, and a prompt that came partly from the prefix cache went
  through different kernels than one prefilled from scratch. Those last-bit
  differences can flip an argmax when two tokens are nearly tied.

So two runs of the same prompt with the same seed are *much* more likely to
agree than with a batch-dependent generator, but are not guaranteed to
produce identical text. Each request currently gets a random seed; a
per-request `seed` field is not exposed in the API yet.

## Limitations

Native decoding supports `temperature` and `top_p`. It does not support
`top_k`, repetition or frequency penalties, or logprobs. Sampled rows are
processed one row at a time inside the step graph (greedy rows stay fully
batched); for the batch sizes a local server sees this has not shown up in
profiles.

## Acknowledgments

The Gumbel-max trick is classical (Gumbel, 1954); for a modern treatment see
[Maddison et al., A* Sampling](https://arxiv.org/abs/1411.0030). Nucleus
sampling is from [Holtzman et al.](https://arxiv.org/abs/1904.09751). The
SplitMix64 mixing constants are from Steele, Lea, and Flood's
[SplittableRandom](https://dl.acm.org/doi/10.1145/2714064.2660195).
