"""Shared dense KV-cache backend for native MLX execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

import mlx.core as mx


@dataclass
class LayerKVCache(Protocol):
    """Layer-cache surface consumed by architecture model modules."""

    offsets: mx.array

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...


@dataclass
class DenseLayerCache:
    """Per-layer KV cache with append/read semantics."""

    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0

    @property
    def offsets(self) -> mx.array:
        """Expose a batch-compatible RoPE offset."""

        return mx.array([self.offset], dtype=mx.int32)

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
class DenseBatchLayerCache:
    """Temporary batched KV view used for one physical model step."""

    request_layers: tuple[DenseLayerCache, ...]
    append_lengths: tuple[int, ...]
    keys: mx.array | None = None
    values: mx.array | None = None
    offsets: mx.array = field(init=False)
    _appended_keys: tuple[mx.array, ...] = field(init=False, default=())
    _appended_values: tuple[mx.array, ...] = field(init=False, default=())
    _committed_keys: tuple[mx.array, ...] = field(init=False, default=())
    _committed_values: tuple[mx.array, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.offsets = mx.array(
            [layer.size() for layer in self.request_layers],
            dtype=mx.int32,
        )

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        max_total = max(
            layer.size() + append_length
            for layer, append_length in zip(
                self.request_layers,
                self.append_lengths,
                strict=True,
            )
        )
        key_rows: list[mx.array] = []
        value_rows: list[mx.array] = []
        appended_keys: list[mx.array] = []
        appended_values: list[mx.array] = []
        committed_keys: list[mx.array] = []
        committed_values: list[mx.array] = []
        for index, (layer, append_length) in enumerate(
            zip(self.request_layers, self.append_lengths, strict=True)
        ):
            valid_keys = keys[index : index + 1, :, :append_length, :]
            valid_values = values[index : index + 1, :, :append_length, :]
            appended_keys.append(valid_keys)
            appended_values.append(valid_values)
            if layer.size() == 0:
                row_keys = valid_keys
                row_values = valid_values
            else:
                if layer.keys is None or layer.values is None:
                    raise ValueError("cache layer missing KV state")
                row_keys = mx.concatenate([layer.keys, valid_keys], axis=2)
                row_values = mx.concatenate([layer.values, valid_values], axis=2)
            committed_keys.append(row_keys)
            committed_values.append(row_values)
            pad = max_total - int(row_keys.shape[2])
            if pad > 0:
                row_keys = mx.pad(row_keys, [(0, 0), (0, 0), (0, pad), (0, 0)])
                row_values = mx.pad(
                    row_values,
                    [(0, 0), (0, 0), (0, pad), (0, 0)],
                )
            key_rows.append(row_keys)
            value_rows.append(row_values)
        self.keys = mx.concatenate(key_rows, axis=0)
        self.values = mx.concatenate(value_rows, axis=0)
        self._appended_keys = tuple(appended_keys)
        self._appended_values = tuple(appended_values)
        self._committed_keys = tuple(committed_keys)
        self._committed_values = tuple(committed_values)
        self.offsets = mx.array(
            [
                layer.size() + append
                for layer, append in zip(
                    self.request_layers, self.append_lengths, strict=True
                )
            ],
            dtype=mx.int32,
        )
        return self.keys, self.values

    def commit(self) -> tuple[int, ...]:
        lengths: list[int] = []
        for layer, keys, values in zip(
            self.request_layers,
            self._committed_keys,
            self._committed_values,
            strict=True,
        ):
            layer.keys = keys
            layer.values = values
            layer.offset = int(keys.shape[2])
            lengths.append(layer.size())
        return tuple(lengths)

    def validate_commit(self) -> None:
        """Validate every staged append before mutating request caches."""

        if len(self._appended_keys) != len(self.request_layers):
            raise ValueError("batched cache has incomplete staged keys")
        if len(self._appended_values) != len(self.request_layers):
            raise ValueError("batched cache has incomplete staged values")
        if len(self._committed_keys) != len(self.request_layers):
            raise ValueError("batched cache has incomplete committed keys")
        if len(self._committed_values) != len(self.request_layers):
            raise ValueError("batched cache has incomplete committed values")
        for layer, keys, values, committed_keys, committed_values, append_length in zip(
            self.request_layers,
            self._appended_keys,
            self._appended_values,
            self._committed_keys,
            self._committed_values,
            self.append_lengths,
            strict=True,
        ):
            if int(keys.shape[2]) != append_length:
                raise ValueError("staged cache key length mismatch")
            if int(values.shape[2]) != append_length:
                raise ValueError("staged cache value length mismatch")
            expected_length = layer.size() + append_length
            if int(committed_keys.shape[2]) != expected_length:
                raise ValueError("committed cache key length mismatch")
            if int(committed_values.shape[2]) != expected_length:
                raise ValueError("committed cache value length mismatch")
            if layer.keys is None and layer.values is not None:
                raise ValueError("cache layer has values without keys")
            if layer.values is None and layer.keys is not None:
                raise ValueError("cache layer has keys without values")


@dataclass
class DenseRequestCache:
    """Opaque per-request cache owned by Python executor."""

    request_id: str
    layers: list[DenseLayerCache]

    def size(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].size()

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


class KVCacheBackend(Protocol):
    """Opaque cache lifecycle and batch-view interface for an executor."""

    def create(self, request_id: str) -> str: ...

    def get(self, handle: str | None, request_id: str) -> DenseRequestCache: ...

    def length(self, handle: str | None) -> int: ...

    def release(self, handle: str | None) -> None: ...

    def batch_layers(
        self,
        caches: tuple[DenseRequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> list[DenseBatchLayerCache]: ...


@dataclass
class DenseKVCacheBackend:
    """Reusable dense per-request KV backend for causal decoder models."""

    num_layers: int
    _caches: dict[str, DenseRequestCache] = field(default_factory=dict)

    def create(self, request_id: str) -> str:
        handle = f"dense-kv-{request_id}-{uuid.uuid4().hex}"
        self._caches[handle] = DenseRequestCache(
            request_id=request_id,
            layers=[DenseLayerCache() for _ in range(self.num_layers)],
        )
        return handle

    def get(self, handle: str | None, request_id: str) -> DenseRequestCache:
        if handle is None or handle not in self._caches:
            raise ValueError("invalid cache handle")
        cache = self._caches[handle]
        if cache.request_id != request_id:
            raise ValueError("cache handle belongs to different request")
        return cache

    def length(self, handle: str | None) -> int:
        if handle is None or handle not in self._caches:
            return 0
        return self._caches[handle].size()

    def release(self, handle: str | None) -> None:
        if handle is not None:
            self._caches.pop(handle, None)

    def batch_layers(
        self,
        caches: tuple[DenseRequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> list[DenseBatchLayerCache]:
        return [
            DenseBatchLayerCache(
                request_layers=tuple(cache.layers[index] for cache in caches),
                append_lengths=append_lengths,
            )
            for index in range(self.num_layers)
        ]
