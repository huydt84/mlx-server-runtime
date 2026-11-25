---
layout: post
title: "Serving Hybrid Models: Gated DeltaNet, Short Convolutions, and 1-Bit Weights"
author: "MLX Air Team"
date: 2026-11-22
tags: [model-support, hybrid-models, linear-attention, quantization, kv-cache]
excerpt: "LFM2 and Qwen3.5 mix attention with layers that carry a running state instead of a KV history. What that does to a paged cache, to batching, and to prefix reuse, plus how a 27B model fits on a laptop as a 1-bit or Hadamard-rotated ternary pack."
---

**Table of contents**

- [Two kinds of memory in one model](#two-kinds-of-memory-in-one-model)
- [The models](#the-models)
- [A hybrid cache](#a-hybrid-cache)
- [Batching layers that carry state](#batching-layers-that-carry-state)
- [Why hybrid models have no prefix cache](#why-hybrid-models-have-no-prefix-cache)
- [Sub-2-bit weights and Hadamard rotations](#sub-2-bit-weights-and-hadamard-rotations)
- [Adding a family without slowing the others](#adding-a-family-without-slowing-the-others)
- [Correctness first](#correctness-first)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)

## Two kinds of memory in one model

A transformer layer remembers the past as a list: one key and one value
vector per previous token. Memory grows linearly with context, and attention
reads all of it every step.

A growing family of models replaces most attention layers with layers that
remember the past as a **fixed-size state**:

- A **short convolution** looks at only the last few inputs, so it needs to
  keep just those.
- A **linear-attention / delta-rule** layer, such as Gated DeltaNet, folds
  every token into a matrix-valued state with an update rule. The state's
  size does not depend on context length.

These **hybrid** models keep a few full-attention layers for precise recall
and use cheap state layers everywhere else. They decode quickly and their
memory grows slowly. They also break assumptions a paged-KV engine makes
everywhere.

## The models

MLX Air's native backend serves two hybrid families:

| Architecture | Layers | State layers | Per-request state |
| --- | --- | --- | --- |
| `Lfm2MoeForCausalLM` (LFM2.5-8B-A1B) | 24 | 18 short-convolution, 6 attention | Last `conv_L_cache - 1` inputs per conv layer |
| `Qwen3_5ForConditionalGeneration` (dense, text) | varies | Gated DeltaNet on 3 of every 4 layers (`full_attention_interval`, default 4) | Last `linear_conv_kernel_dim - 1` conv inputs, plus a float32 recurrent state per layer |

*Table 1: Hybrid families in `native-mlx`. LFM2.5-8B-A1B is also a
mixture-of-experts model with sigmoid expert routing.*

The Qwen3.5 recurrent state is
`linear_num_value_heads x linear_value_head_dim x linear_key_head_dim`
float32 values per layer. On a 27B checkpoint that comes to about **150 MiB
per in-flight request**, a fixed cost that does not shrink for short prompts
and does not grow for long ones.

## A hybrid cache

The pure-transformer families use `PagedKVCacheBackend`. Hybrid families use
a separate `HybridPagedKVCacheBackend` composed of two parts:

- a **paged KV pool for the attention layers only**. A startup-built mapping
  from logical layer index to KV slot means LFM2's 6 attention layers get
  pages and its 18 convolution layers get none. The KV budget is not spent on
  layers that never read it.
- a **request-local state store** for the other layers: convolution tails
  and recurrent matrices, committed with each step.

Layers that hold only state keep an empty placeholder in the KV pool so that
per-layer indexing stays uniform; staging K/V into one of those is an error
by design ([ADR-0005](../adr/0005-paged-kv-cache-writes-and-buffer-lifetime.md)).

State commits are transactional, like KV appends. A step stages each layer's
new state, and only `commit` makes it the request's state. A failed step
leaves every request's state exactly as it was.

The recurrent state lives outside the KV byte budget. Operators sizing memory
for Qwen3.5 must leave room for it: eight concurrent requests on a 27B model
hold over a gigabyte of recurrent state on top of the KV pool.

## Batching layers that carry state

Batching prefill rows of different lengths means padding the short rows. For
attention, padding is handled by the mask: padded positions are never
attended to. For a state layer, padding is more dangerous. A convolution
would shift padding values into its window, and a recurrence would fold
padding tokens into the state. Either way, a request's output would depend on
who it was batched with.

The hybrid attention context exposes a `token_mask` built from each row's
real append length. Padded positions are masked out of both the convolution
state update and the recurrence, so batching does not change a request's
output.

## Why hybrid models have no prefix cache

Prefix caching ([post 8](2026-11-22-prefix-caching-radix-vs-block-hash.md))
relies on one property of KV: the cache after `n` tokens is a *prefix* of the
cache after `m > n` tokens. To reuse the first 512 tokens, take the first 32
pages.

A recurrent state has no such property. The state after 2,000 tokens is one
matrix that blends all 2,000. There is no way to recover from it the state
after token 512. Reusing a prefix would require snapshotting the state at
every page boundary that might be shared, which costs 150 MiB per snapshot on
Qwen3.5 27B.

So hybrid families declare `supports_prefix_cache = False` in the registry
and run with `NoPrefixCache`. Every request prefills in full. Startup warmup,
which leans on the prefix cache for dense models, prefills each warmup prompt
in full for hybrids; it took 1.7 s for LFM2.5-8B-A1B on an M4 Pro.

## Sub-2-bit weights and Hadamard rotations

A 27B model at 16 bits is 54 GB, which no laptop serves. MLX Air's
known-good Qwen3.5 checkpoints are much smaller packs:

- `prism-ml/Bonsai-27B-mlx-1bit`: an MLX affine-quantized 1-bit export.
- `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`: a ternary pack stored in a
  **Hadamard-rotated** format.

Low-bit quantization suffers from outliers: a few large weights stretch each
group's quantization range and waste the rest. Multiplying the weights by an
orthogonal rotation spreads outliers across the group first. For a rotation
`R`, a linear layer can store `W' = W R^T` and compute

```text
W' . rotate(x) = W R^T R x = W x
```

So at run time the activation must be rotated before every packed matmul:
multiply by per-column signs, then apply a normalized blockwise
Walsh-Hadamard transform. The embedding table stores rotated rows, so each
looked-up row is rotated back. ([ADR-0008](../adr/0008-checkpoint-loading-compatibility.md).)

These packs use `model_type = prism_hadamard_qwen35`, deliberately not
`qwen3_5`: an ordinary affine loader would skip both transforms and produce
wrong output without an error. Plain `mlx_lm.load` cannot load them. MLX
Air's registry maps that model type to `Qwen3_5ForConditionalGeneration`
because the packs omit the `architectures` field, and both the native and
`v1` backends install the rotation modules. Two more details mattered:

- The packs keep norms and other unquantized tensors in float32, so the
  residual stream and the K/V activations are float32. The model refines its
  config after loading (`resolved_config`), and the KV pool is sized and
  typed as float32. The paged decode kernel's separate query and KV dtype
  parameters (post 4) exist partly for this.
- The tokenizer knows only one end-of-generation token. The pack's
  `generation_config.json` lists several, and the loader adds all of them.

## Adding a family without slowing the others

Hybrid support is a test of the architecture. It would be easy to add a
`conv_states` dictionary to every reservation, an `if hybrid:` to every
layer, and a string-parsed mask to every attention call. Each is cheap; all
of them together are per-token overhead paid by every pure transformer
forever.

MLX Air resolves the architecture **entirely at startup**
(`python/mlx_worker/native_mlx/registry.py`). Each family is a small manifest
naming its module and factory functions, not an import:

```python
@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_class: str
    module_name: str
    model_factory_name: str
    cache_geometry_factory: Callable[[Any], KVCacheGeometry]
    supports_prefix_cache: bool = True
    cache_family: str = "kv"
```

Bootstrap does one dictionary lookup, imports only the selected module, and
freezes an `ArchitectureExecutionPlan` binding the model, the cache family,
the attention bundle, and the prefix-cache capability. After that, the
scheduler and executor hot path contain no registry access, no
architecture-name checks, and no lazy imports. Pure transformers get the
pure paged-KV bundle with no state fields at all; LFM2 and Qwen3.5 get the
hybrid bundle. Unrelated model families are never imported into the worker.

The rule for contributors: a new family may add a manifest, a model module,
and a family-specific execution factory, but may not add branches or optional
fields to another family's steady-state path.

## Correctness first

Hybrid models are where subtle bugs live. Two examples:

- **Sigmoid routing.** LFM2-MoE scores experts with `sigmoid(gate(x))`, and
  its `expert_bias` only affects which experts are chosen, not their weights.
  The first native router copied the softmax gating of older `mlx-lm`
  releases, which also folded `expert_bias` into the routing weights. Outputs
  looked plausible; the parity check against `mlx-lm` 0.32.2 showed a
  maximum absolute logit difference of 29.1. Matching the Transformers
  reference (sigmoid scores, bias for selection only, renormalize, scale by
  `routed_scaling_factor`) brought it to 0.0.
- **Same kernel as the reference.** The Gated DeltaNet recurrence uses the
  `mlx-lm` Metal kernel (`mlx_lm.models.gated_delta`) directly, so native and
  `mlx-lm` runs execute the same recurrence code. Parity failures then point
  at our cache and batching, not at a reimplemented kernel.

The `mlx_lm_comparison` benchmark suite runs native `overlap` against the
`mlx_lm` baseline on both Qwen3.5 packs behind the same gateway and load
generator, with the `v1` prompt cache off so neither side reuses prefixes.

## Limitations

- Hybrid models run without a prefix cache.
- Qwen3.5 mixture-of-experts checkpoints and the Qwen3.5 vision tower are not
  implemented.
- Recurrent state is outside the KV memory budget and must be planned for
  separately.

## Acknowledgments

Gated DeltaNet is from [Yang et al., Gated Delta Networks](https://arxiv.org/abs/2412.06464).
Rotation-based outlier smoothing for quantization follows
[QuaRot](https://arxiv.org/abs/2404.00456) and related work. Thanks to the
`mlx-lm` maintainers for the Gated DeltaNet kernel, and to Liquid AI and the
Qwen team for open hybrid checkpoints.
