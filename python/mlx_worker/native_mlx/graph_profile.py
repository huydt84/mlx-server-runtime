"""Model-agnostic MLX module graph profiling."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

from .interfaces import ForwardBatch, NativeModel


_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)(?:layers|blocks|h|decoder_layers)\.(\d+)(?:\.|$)"
)


@dataclass
class _GraphProfileRecorder:
    """Collect per-step model graph timings."""

    metrics: dict[str, float] = field(default_factory=dict)
    layer_totals: dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.metrics.clear()
        self.layer_totals.clear()

    def record(self, category: str, elapsed_ms: float, layer_index: int | None) -> None:
        key = f"model_graph_{category}_ms"
        self.metrics[key] = self.metrics.get(key, 0.0) + elapsed_ms
        if layer_index is not None:
            self.layer_totals[layer_index] = (
                self.layer_totals.get(layer_index, 0.0) + elapsed_ms
            )

    def summary(self) -> dict[str, int]:
        metrics = {
            name: max(0, int(round(value))) for name, value in self.metrics.items()
        }
        total = sum(self.layer_totals.values())
        if total:
            worst_layer_index, worst_layer_ms = max(
                self.layer_totals.items(), key=lambda item: item[1]
            )
            metrics["model_graph_layer_total_ms"] = max(0, int(round(total)))
            metrics["model_graph_worst_layer_ms"] = max(
                0,
                int(round(worst_layer_ms)),
            )
            metrics["model_graph_worst_layer_index"] = worst_layer_index
        return metrics


class _ProfiledModule(nn.Module):
    """Proxy that times a registered MLX module without changing model code."""

    def __init__(
        self,
        *,
        target: nn.Module,
        category: str,
        layer_index: int | None,
        recorder: _GraphProfileRecorder,
    ) -> None:
        super().__init__()
        self.target = target
        self.category = category
        self.layer_index = layer_index
        self.recorder = recorder

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        output = self.target(*args, **kwargs)
        _eval_arrays(output)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.recorder.record(self.category, elapsed_ms, self.layer_index)
        return output

    def __getattr__(self, name: str) -> Any:
        """Preserve helper methods exposed by specialized MLX modules."""

        try:
            return super().__getattr__(name)
        except AttributeError:
            target = self.get("target") or self.__dict__.get("target")
            if target is None:
                raise
            return getattr(target, name)


class GraphProfiledModel:
    """NativeModel wrapper that profiles an MLX module tree generically."""

    def __init__(self, model: NativeModel) -> None:
        self._model = model
        self._recorder = _GraphProfileRecorder()
        self.num_layers = int(getattr(model, "num_layers"))
        self._wrap_selected_modules()

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        forward_batch: ForwardBatch,
    ) -> mx.array:
        return self._model(input_ids, positions, forward_batch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def load_weights(
        self,
        weights: Sequence[tuple[str, mx.array]],
        *,
        strict: bool = True,
    ) -> None:
        self._model.load_weights(weights, strict=strict)

    def reset_graph_profile(self) -> None:
        self._recorder.reset()

    def graph_profile_metrics(self) -> dict[str, int]:
        return self._recorder.summary()

    def _wrap_selected_modules(self) -> None:
        if not isinstance(self._model, nn.Module):
            return
        candidates = [
            _ProfileCandidate(
                path=path,
                module=module,
                category=category,
                layer_index=_layer_index(path),
            )
            for path, module in self._model.named_modules()
            if path and (category := _category_for(path, module)) is not None
        ]
        selected: list[_ProfileCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.path.count(".")):
            if any(_is_descendant(candidate.path, item.path) for item in selected):
                continue
            selected.append(candidate)
        for candidate in selected:
            _replace_module(
                self._model,
                candidate.path,
                _ProfiledModule(
                    target=candidate.module,
                    category=candidate.category,
                    layer_index=candidate.layer_index,
                    recorder=self._recorder,
                ),
            )


@dataclass(frozen=True)
class _ProfileCandidate:
    path: str
    module: nn.Module
    category: str
    layer_index: int | None


def _category_for(path: str, module: nn.Module) -> str | None:
    lower_path = path.lower()
    class_name = type(module).__name__.lower()
    if "embed" in lower_path or "embedding" in class_name:
        return "embedding"
    if "lm_head" in lower_path or "output_projection" in lower_path:
        return "lm_head"
    if any(name in lower_path for name in ("self_attn", "attention", "attn")):
        return "attention"
    if any(name in lower_path for name in ("mlp", "ffn", "feed_forward")):
        return "mlp"
    if "norm" in lower_path or "norm" in class_name or lower_path.endswith(".ln"):
        return "norm"
    if "linear" in class_name or lower_path.endswith("_proj"):
        return "projection"
    return None


def _layer_index(path: str) -> int | None:
    match = _LAYER_INDEX_RE.search(path)
    if match is None:
        return None
    return int(match.group(1))


def _is_descendant(path: str, parent: str) -> bool:
    return path.startswith(f"{parent}.")


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent: Any = root
    index = 0
    while index < len(parts) - 1:
        part = parts[index]
        if isinstance(parent, list):
            parent = parent[int(part)]
        else:
            parent = parent[part]
        index += 1
    leaf = parts[-1]
    if isinstance(parent, list):
        parent[int(leaf)] = replacement
    else:
        parent[leaf] = replacement


def _eval_arrays(value: Any) -> None:
    arrays = tuple(_iter_arrays(value))
    if arrays:
        mx.eval(*arrays)


def _iter_arrays(value: Any) -> Any:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)


__all__ = ["GraphProfiledModel"]
