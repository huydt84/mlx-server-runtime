"""Repo-owned canonical weight-mapping abstractions for native MLX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx


class WeightArtifactValidationError(ValueError):
    """Raised when checkpoint weight artifacts are missing or malformed."""


class WeightMappingBug(RuntimeError):
    """Raised when a supported-class checkpoint needs adapter work."""


@dataclass(frozen=True)
class WeightIndex:
    """Weight artifact index loaded from checkpoint metadata.

    Attributes:
        model_path: Local checkpoint directory.
        weight_map: Mapping from source tensor name to source file.
        source_files: Unique source files referenced by the index.
    """

    model_path: Path
    weight_map: dict[str, str]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class WeightMappingEntry:
    """One repo-owned canonical mapping entry."""

    canonical_name: str
    source_name: str
    source_file: str


@dataclass(frozen=True)
class WeightMappingPlan:
    """Canonical mapping plan produced by one architecture adapter."""

    architecture_class: str
    source_files: tuple[str, ...]
    entries: tuple[WeightMappingEntry, ...]


class WeightMappingAdapter(Protocol):
    """Per-class adapter that maps raw checkpoint names into canonical names."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        """Build canonical mapping plan from raw checkpoint metadata."""


def load_mapped_weights(
    index: WeightIndex,
    plan: WeightMappingPlan,
) -> list[tuple[str, mx.array]]:
    """Load a validated mapping plan without architecture-specific I/O code."""

    by_file: dict[str, list[WeightMappingEntry]] = {}
    for entry in plan.entries:
        by_file.setdefault(entry.source_file, []).append(entry)
    loaded: list[tuple[str, mx.array]] = []
    for source_file, entries in by_file.items():
        try:
            tensors = mx.load(str(index.model_path / source_file))
        except Exception as exc:
            raise WeightArtifactValidationError(
                f"could not load native weights from {source_file}: {exc}"
            ) from exc
        for entry in entries:
            if entry.source_name not in tensors:
                raise WeightArtifactValidationError(
                    f"missing tensor {entry.source_name!r} in {source_file}"
                )
            loaded.append((entry.canonical_name, tensors[entry.source_name]))
    return loaded


def load_weight_index(model_path: Path) -> WeightIndex:
    """Load weight index metadata from checkpoint directory.

    Raises:
        WeightArtifactValidationError: If index file or referenced weight files are
            missing or malformed.
    """

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise WeightArtifactValidationError(
            f"missing native weight index: {index_path.name}"
        )

    try:
        payload = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightArtifactValidationError(
            f"could not parse {index_path.name}: {exc}"
        ) from exc

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise WeightArtifactValidationError(
            f"{index_path.name} is missing non-empty weight_map"
        )

    source_files = sorted(
        {str(value) for value in weight_map.values() if isinstance(value, str)}
    )
    if not source_files:
        raise WeightArtifactValidationError(
            f"{index_path.name} does not reference any weight files"
        )

    missing_files = [name for name in source_files if not (model_path / name).exists()]
    if missing_files:
        raise WeightArtifactValidationError(
            "missing native weight files: " + ", ".join(missing_files)
        )

    invalid_keys = [
        key
        for key, value in weight_map.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_keys:
        raise WeightArtifactValidationError(
            f"{index_path.name} contains non-string weight entries"
        )

    return WeightIndex(
        model_path=model_path,
        weight_map={str(key): str(value) for key, value in weight_map.items()},
        source_files=tuple(source_files),
    )
