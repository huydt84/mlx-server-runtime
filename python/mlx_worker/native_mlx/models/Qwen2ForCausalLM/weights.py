"""Qwen2 weight-mapping adapter and loader."""

from __future__ import annotations

import re

import mlx.core as mx

from ...mapping import (
    WeightArtifactValidationError,
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


class Qwen2WeightAdapter:
    """Build canonical weight plan for Qwen2 checkpoints."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        """Map raw checkpoint names into repo-owned canonical names."""

        entries: list[WeightMappingEntry] = []
        for source_name, source_file in index.weight_map.items():
            entries.append(
                WeightMappingEntry(
                    canonical_name=_canonicalize_qwen2_name(source_name),
                    source_name=source_name,
                    source_file=source_file,
                )
            )

        if not entries:
            raise WeightMappingBug("Qwen2 checkpoint produced empty mapping plan")

        return WeightMappingPlan(
            architecture_class="Qwen2ForCausalLM",
            source_files=index.source_files,
            entries=tuple(entries),
        )


def load_mapped_weights(
    index: WeightIndex, plan: WeightMappingPlan
) -> list[tuple[str, mx.array]]:
    """Load mapped checkpoint weights into canonical-name list."""

    by_file: dict[str, list[WeightMappingEntry]] = {}
    for entry in plan.entries:
        by_file.setdefault(entry.source_file, []).append(entry)

    loaded: list[tuple[str, mx.array]] = []
    for source_file, entries in by_file.items():
        source_path = index.model_path / source_file
        try:
            tensors = mx.load(str(source_path))
        except Exception as exc:  # pragma: no cover - host/runtime dependent
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


def _canonicalize_qwen2_name(source_name: str) -> str:
    if source_name in {
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
    }:
        return source_name
    if source_name in {
        "model.norm.weight",
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
    }:
        return source_name

    match = _LAYER_RE.match(source_name)
    if match is None:
        raise WeightMappingBug(
            f"Qwen2 weight adapter has no canonical rule for source tensor {source_name!r}"
        )

    layer_index = int(match.group(1))
    suffix = match.group(2)
    if not suffix:
        raise WeightMappingBug(
            f"Qwen2 weight adapter saw empty layer suffix for {source_name!r}"
        )
    return f"model.layers.{layer_index}.{suffix}"
