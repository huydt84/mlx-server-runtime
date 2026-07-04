"""Qwen2 KV cache primitives for native MLX executor."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Qwen2LayerCache:
    """Per-layer KV cache with append/read semantics."""

    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = int(self.keys.shape[2])
        return self.keys, self.values

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass
class Qwen2RequestCache:
    """Opaque per-request cache owned by Python executor."""

    request_id: str
    layers: list[Qwen2LayerCache]

    def size(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].size()

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)
