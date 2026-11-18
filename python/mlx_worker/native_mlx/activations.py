"""Fused gated MLP activations shared by native model families.

An uncompiled ``activation(gate) * up`` launches the activation kernel and a
separate multiply kernel for every MLP of every step. Compiling the pair fuses
it into one elementwise kernel, which matters at decode, where each kernel
launch is a measurable share of a step. ``shapeless=True`` keeps one compiled
graph across batch and sequence shapes.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
from mlx import nn


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Return ``silu(gate) * up`` as one fused kernel.

    Args:
        gate: Gate projection output.
        up: Up projection output with the same shape as ``gate``.

    Returns:
        The gated activation.
    """

    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def geglu_approx(gate: mx.array, up: mx.array) -> mx.array:
    """Return ``gelu_approx(gate) * up`` as one fused kernel.

    Args:
        gate: Gate projection output.
        up: Up projection output with the same shape as ``gate``.

    Returns:
        The gated activation.
    """

    return nn.gelu_approx(gate) * up
