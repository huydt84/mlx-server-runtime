"""LookAtMyQuant packed checkpoints of Inkling-2-Mini.

A pack stores every layer's matrices in one ``layer_NN_packed.npz`` archive and
describes their formats in ``layer_NN_meta.json``. The embedding and output
head are split into frequency buckets with their own formats, trellis tables
live in ``luts.npz``, and the unquantized tensors in ``steering.safetensors``.

The adapter folds the formats into canonical weight names, so the model builder
needs only names and arrays:

- ``layers.{i}.linear.{matrix}.{format}.{part}``
- ``layers.{i}.experts.{side}.{format}.{part}``, ``layers.{i}.experts.{side}.signs``
  and ``layers.{i}.experts.{side}.map``
- ``{embed|head}.{bucket}.{format}.{part}`` and ``{embed|head}.order``
- ``tables.{format}``
- the steering tensor names unchanged.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from ..mapping import (
    WeightArtifactValidationError,
    WeightIndex,
    WeightMappingBug,
    WeightMappingEntry,
    WeightMappingPlan,
)
from ..packed_matmul import FORMAT_BITS, FORMAT_IDS, TRELLIS_LUT_SHAPES

ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
PACK_FORMAT = "inkling2-packed-v1-sqyfr"
PACK_SUBFORMATS = frozenset((None, "rg128-f16signs-v3"))
"""Pack revisions: one format per expert, or one per 128-row group."""

_STEERING = "steering.safetensors"
_LINEARS = {
    "self_attn.q_proj.linear": "q",
    "self_attn.k_proj.linear": "k",
    "self_attn.v_proj.linear": "v",
    "self_attn.relative_position_bias.linear": "rel",
    "self_attn.o_proj.gate_proj": "gate",
    "self_attn.o_proj.linear": "o",
}
_SIDES = ("gu", "dn")


def load_packed_weight_index(model_path: Path) -> WeightIndex:
    """Index a pack's archives without reading any tensor.

    Keys are ``{file}/{member}`` because archives of different layers reuse
    member names.

    Args:
        model_path: Local checkpoint directory.

    Returns:
        The weight index.

    Raises:
        WeightArtifactValidationError: If the checkpoint is not a supported
            pack or one of its files is missing or unreadable.
    """

    manifest = _read_json(model_path, "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != PACK_FORMAT:
        raise WeightArtifactValidationError(
            f"native-mlx serves Inkling-2-Mini only from LookAtMyQuant packs "
            f"(manifest.json format {PACK_FORMAT!r})"
        )
    if manifest.get("subformat") not in PACK_SUBFORMATS:
        raise WeightArtifactValidationError(
            f"unsupported LookAtMyQuant subformat {manifest.get('subformat')!r}"
        )
    archives = sorted(
        [
            "embed_packed.npz",
            "head_packed.npz",
            "luts.npz",
            *(path.name for path in model_path.glob("layer_*_packed.npz")),
        ]
    )
    missing = [
        name for name in (*archives, _STEERING) if not (model_path / name).exists()
    ]
    if missing:
        raise WeightArtifactValidationError(
            "missing native weight files: " + ", ".join(missing)
        )
    weight_map: dict[str, str] = {}
    for name in archives:
        try:
            with zipfile.ZipFile(model_path / name) as archive:
                members = archive.namelist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise WeightArtifactValidationError(
                f"could not read {name}: {exc}"
            ) from exc
        for member in members:
            weight_map[f"{name}/{member.removesuffix('.npy')}"] = name
    for tensor in _safetensors_names(model_path / _STEERING):
        weight_map[f"{_STEERING}/{tensor}"] = _STEERING
    return WeightIndex(
        model_path=model_path,
        weight_map=weight_map,
        source_files=(*archives, _STEERING),
    )


class Inkling2PackAdapter:
    """Map a pack's archive members into canonical names."""

    def build_plan(self, index: WeightIndex) -> WeightMappingPlan:
        """Build the mapping plan from archive members and format metadata.

        Args:
            index: Result of :func:`load_packed_weight_index`.

        Returns:
            The mapping plan; members no serving path reads are left out.

        Raises:
            WeightMappingBug: If metadata names an unknown matrix or format, or
                a member the pack does not contain.
        """

        plan = _PlanBuilder(index)
        layers = [
            name.removesuffix("_packed.npz")
            for name in index.source_files
            if name.startswith("layer_")
        ]
        for layer, stem in enumerate(layers):
            if stem != f"layer_{layer:02d}":
                raise WeightMappingBug(f"pack is missing layer {layer}")
            self._map_layer(plan, layer, stem)
        for part in ("embed", "head"):
            self._map_buckets(plan, part)
        for fmt in TRELLIS_LUT_SHAPES:
            if f"luts.npz/{fmt}" in index.weight_map:
                plan.add(f"tables.{fmt}", "luts.npz", fmt)
        prefix = f"{_STEERING}/"
        for key in index.weight_map:
            if key.startswith(prefix):
                tensor = key.removeprefix(prefix)
                if tensor.startswith("model.") and not tensor.startswith(
                    "model.visual."
                ):
                    plan.add(tensor, _STEERING, tensor)
        return WeightMappingPlan(
            architecture_class=ARCHITECTURE,
            source_files=index.source_files,
            entries=tuple(plan.entries),
        )

    @staticmethod
    def _map_layer(plan: _PlanBuilder, layer: int, stem: str) -> None:
        meta = plan.metadata(f"{stem}_meta.json")
        archive = f"{stem}_packed.npz"
        linears = meta.get("linears")
        experts = meta.get("experts")
        if not isinstance(linears, dict) or set(linears) != set(_LINEARS):
            raise WeightMappingBug(f"{stem} does not describe its six projections")
        for name, matrix in _LINEARS.items():
            record = linears[name]
            fmt = _format(record.get("fmt"), record.get("k"), f"{stem} {name}")
            source = f"lin_{record['safe']}"
            target = f"layers.{layer}.linear.{matrix}.{fmt}"
            parts = ["codes", "scales", "signs_in"]
            if fmt.startswith("lm"):
                parts.append("levels")
            if record.get("out_rot"):
                parts.append("signs_out")
            for part in parts:
                plan.add(f"{target}.{part}", archive, f"{source}_{part}")
        if not isinstance(experts, dict) or set(experts) != set(_SIDES):
            raise WeightMappingBug(f"{stem} does not describe its experts")
        for side in _SIDES:
            target = f"layers.{layer}.experts.{side}"
            for fmt in experts[side]:
                _format(fmt, None, f"{stem} {side}")
                plan.add(f"{target}.{fmt}.codes", archive, f"{side}_{fmt}_codes")
                plan.add(f"{target}.{fmt}.scales", archive, f"{side}_{fmt}_scales")
                if fmt == "nvfp4":
                    plan.add(f"{target}.{fmt}.outer", archive, f"{side}_{fmt}_outer")
            plan.add(f"{target}.signs", archive, f"{side}_signs_in")
            grouped = f"{side}_rg_map"
            plan.add(
                f"{target}.map",
                archive,
                grouped if plan.has(archive, grouped) else f"{side}_map",
            )

    @staticmethod
    def _map_buckets(plan: _PlanBuilder, part: str) -> None:
        archive = f"{part}_packed.npz"
        buckets = plan.metadata(f"{part}_meta.json")
        if not isinstance(buckets, list) or not buckets:
            raise WeightMappingBug(f"{part}_meta.json lists no buckets")
        for bucket, record in enumerate(buckets):
            source = f"b{record['kb']}"
            fmt = _format(record.get("opt"), record.get("k"), f"{part} {source}")
            parts = ["codes", "scales", "signs_in"]
            if fmt.startswith("lm"):
                parts.append("levels")
            for name in parts:
                plan.add(f"{part}.{bucket}.{fmt}.{name}", archive, f"{source}_{name}")
        plan.add(f"{part}.order", archive, "order")


class _PlanBuilder:
    """Collect mapping entries and check that each source exists."""

    def __init__(self, index: WeightIndex):
        self.index = index
        self.entries: list[WeightMappingEntry] = []

    def has(self, source_file: str, member: str) -> bool:
        return f"{source_file}/{member}" in self.index.weight_map

    def add(self, canonical: str, source_file: str, member: str) -> None:
        if not self.has(source_file, member):
            raise WeightMappingBug(f"pack is missing {member!r} in {source_file}")
        self.entries.append(WeightMappingEntry(canonical, member, source_file))

    def metadata(self, name: str) -> Any:
        try:
            return _read_json(self.index.model_path, name)
        except WeightArtifactValidationError as exc:
            raise WeightMappingBug(str(exc)) from exc


def _format(fmt: object, bits: object, where: str) -> str:
    if not isinstance(fmt, str) or fmt not in FORMAT_IDS:
        raise WeightMappingBug(f"{where} uses unknown packed format {fmt!r}")
    if bits is not None and bits != FORMAT_BITS[fmt]:
        raise WeightMappingBug(f"{where} stores {fmt} with {bits} bits")
    return fmt


def _read_json(model_path: Path, name: str) -> Any:
    try:
        return json.loads((model_path / name).read_text())
    except FileNotFoundError as exc:
        raise WeightArtifactValidationError(
            f"missing {name}: native-mlx serves Inkling-2-Mini only from "
            "LookAtMyQuant packs"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightArtifactValidationError(f"could not parse {name}: {exc}") from exc


def _safetensors_names(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if not 0 < size <= path.stat().st_size:
                raise ValueError("invalid header size")
            header = json.loads(handle.read(size))
    except (OSError, ValueError) as exc:
        raise WeightArtifactValidationError(
            f"could not read {path.name} header: {exc}"
        ) from exc
    return [name for name in header if name != "__metadata__"]
