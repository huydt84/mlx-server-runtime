"""Inkling-2 packs: loading, forward consistency, and speculative decoding."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_worker.native_mlx.bootstrap import (
    NativeArchitecture,
    NativeBootstrapFailure,
    _open_drafter,
)
from mlx_worker.native_mlx.cache_coordinator import NativeCacheCoordinator
from mlx_worker.native_mlx.execution_backends import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    build_native_execution_backend,
)
from mlx_worker.native_mlx.executor import (
    MlxGenerationExecutor,
    MlxOverlapGenerationExecutor,
)
from mlx_worker.native_mlx.interfaces import (
    BatchExecutionError,
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    ForwardMode,
    SamplingParams,
    SchedulableRequest,
)
from mlx_worker.native_mlx.mapping import (
    WeightArtifactValidationError,
    WeightMappingBug,
    load_mapped_weights,
)
from mlx_worker.native_mlx.models import inkling2
from mlx_worker.native_mlx.models.inkling2 import parse_inkling2_config
from mlx_worker.native_mlx.models.inkling2_drafter import load_drafter
from mlx_worker.native_mlx.prefix_cache import NoPrefixCache
from mlx_worker.native_mlx.registry import get_architecture_spec
from mlx_worker.native_mlx.scheduler import NativeContinuousScheduler
from mlx_worker.native_mlx.speculative import Speculator

from . import inkling2_pack

_ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
_PROMPT = tuple(range(3, 3 + 45))

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="packed kernels require Metal"
)


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("pack")
    inkling2_pack.write_pack(path)
    return path


@pytest.fixture(scope="module")
def drafter_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("drafter")
    inkling2_pack.write_drafter(path)
    return path


@pytest.fixture(scope="module")
def model(pack: Path) -> Iterator[Any]:
    """Build the tiny model with full-precision inputs, gates, and KV.

    Half precision amplifies the rounding differences between step shapes, and
    the random weights leave many routing scores close together.
    """

    plan = get_architecture_spec(_ARCHITECTURE).resolve
    index = plan.load_weight_index(pack)
    adapter = plan.create_weight_adapter()
    weights = load_mapped_weights(index, adapter.build_plan(index))
    config = plan.parse_config(json.loads((pack / "config.json").read_text()))
    config = replace(config, kv_cache_dtype="float32")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inkling2, "_PACKED_INPUT_DTYPE", mx.float32)
        patch.setattr(inkling2, "_GATE_DTYPE", mx.float32)
        built = plan.create_model(config, weights)
        assert weights == []
        yield built


def _executor(
    model: Any, *, overlap: bool = False, speculator: Speculator | None = None
) -> MlxGenerationExecutor:
    geometry = get_architecture_spec(_ARCHITECTURE).cache_geometry_factory(model.config)
    bundle = build_native_execution_backend(
        DEFAULT_NATIVE_EXECUTION_BACKEND,
        geometry,
        page_size=8,
        cache_budget_bytes=64 << 20,
        cache_family="windowed",
    )
    kind = MlxOverlapGenerationExecutor if overlap else MlxGenerationExecutor
    return kind(
        architecture_class=_ARCHITECTURE,
        model=model,
        cache_backend=bundle.cache_backend,
        attention_backend=bundle.attention_backend,
        speculator=speculator,
    )


class _Session:
    """Step a model directly, without sampling."""

    def __init__(self, model: Any):
        self.model = model
        self.executor = _executor(model)
        self.backend = self.executor.cache_backend
        self.handles: dict[str, str] = {}

    def step(self, work: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
        """Append tokens per request and return every token's logits."""

        caches = []
        for name in work:
            if name not in self.handles:
                self.handles[name] = self.backend.create(name)
            caches.append(self.backend.get(self.handles[name], name))
        lengths = tuple(len(tokens) for tokens in work.values())
        reservation = self.backend.reserve_batch(tuple(caches), lengths)
        width = max(lengths)
        batch = ForwardBatch(
            ForwardMode.PREFILL,
            lengths,
            tuple(cache.size() for cache in caches),
            "causal",
            self.executor.attention_backend.contexts(reservation, ForwardMode.PREFILL),
        )
        padded = [
            list(tokens) + [0] * (width - len(tokens)) for tokens in work.values()
        ]
        logits = self.model(mx.array(padded), None, batch)
        mx.eval(logits)
        reservation.commit()
        return {
            name: np.array(logits[row, :length])
            for row, (name, length) in enumerate(zip(work, lengths, strict=True))
        }


@needs_metal
def test_chunked_and_batched_steps_match_one_forward(model: Any) -> None:
    whole = _Session(model).step({"a": _PROMPT})["a"]
    assert whole.shape == (len(_PROMPT), inkling2_pack.OUTPUT)
    assert np.isfinite(whole).all() and whole.std() > 1e-3

    session = _Session(model)
    parts = [
        session.step({"a": _PROMPT[:20]})["a"],
        session.step({"b": (9, 8, 7), "a": _PROMPT[20:21]})["a"],
        session.step({"a": _PROMPT[21:40], "c": (1,)})["a"],
        *(session.step({"a": (token,)})["a"] for token in _PROMPT[40:]),
    ]
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=2e-2)


@needs_metal
def test_short_steps_compute_exactly_what_one_token_steps_do(model: Any) -> None:
    tokens = tuple(range(7, 7 + 72))
    one = _Session(model)
    one.step({"a": _PROMPT})
    expected = np.concatenate([one.step({"a": (token,)})["a"] for token in tokens])

    for size in (2, 3, 8):
        several = _Session(model)
        several.step({"a": _PROMPT})
        logits = np.concatenate(
            [
                several.step({"a": tokens[start : start + size]})["a"]
                for start in range(0, len(tokens), size)
            ]
        )
        np.testing.assert_array_equal(logits, expected)


@needs_metal
def test_sliding_layers_hold_only_their_window(model: Any) -> None:
    session = _Session(model)
    for start in range(0, 160, 8):
        session.step({"a": tuple(range(start, start + 8))})

    metrics = session.backend.metrics()
    assert metrics["used_pages"] == 20
    assert metrics["window_used_pages"] == 2


@needs_metal
def test_requested_positions_and_taps_come_from_the_same_forward(model: Any) -> None:
    session = _Session(model)
    cache = session.backend.get(session.backend.create("a"), "a")
    taps = (0, 4, model.num_layers)

    def run(positions: tuple[int, ...] | None) -> tuple[mx.array, mx.array]:
        reservation = session.backend.reserve_batch((cache,), (6,))
        contexts = session.executor.attention_backend.contexts(
            reservation, ForwardMode.PREFILL
        )
        batch = ForwardBatch(
            ForwardMode.PREFILL, (6,), (0,), "causal", contexts, positions
        )
        result = model.forward_with_taps(mx.array([_PROMPT[:6]]), batch, taps)
        mx.eval(*result)
        reservation.abort()
        return result

    logits, states = run(None)
    last, again = run((5,))
    assert states.shape == (6, 3, inkling2_pack.HIDDEN)
    assert last.shape == (1, 1, inkling2_pack.OUTPUT)
    np.testing.assert_allclose(np.array(last[0, 0]), np.array(logits[0, 5]), atol=1e-3)
    np.testing.assert_allclose(np.array(states), np.array(again), atol=1e-3)
    np.testing.assert_allclose(
        np.array(model.head(states[:, 2])), np.array(logits[0]), atol=1e-3
    )


class _ScriptedDrafter:
    """Propose tokens from a script indexed by committed length."""

    taps = (1, 4)
    max_draft_tokens = 3

    def __init__(self, script: dict[int, int]):
        self.script = script
        self.read_tokens: list[int] = []

    def new_context(self) -> Any:
        return _ScriptedContext()

    def read(self, context: Any, taps: mx.array) -> None:
        assert taps.shape[1:] == (2, inkling2_pack.HIDDEN)
        self.read_tokens.append(int(taps.shape[0]))
        context.length += int(taps.shape[0])

    def propose(
        self, context: Any, token: mx.array, count: int
    ) -> tuple[mx.array, mx.array]:
        first = context.length + 1
        tokens = mx.array([self.script[first + index] for index in range(count)])
        return tokens, mx.arange(count, 0, -1) / 10


class _ScriptedContext:
    length = 0

    @staticmethod
    def arrays() -> list[mx.array]:
        return []


class _FixedLength:
    """Length controller that proposes three tokens and verifies ``verified``."""

    def __init__(self, verified: int = 3) -> None:
        self.verified = verified
        self.steps: list[tuple[int, int]] = []

    def choose(self, limit: int) -> int:
        return min(limit, 3)

    def trim(self, confidences: Sequence[float]) -> int:
        return min(len(confidences), self.verified)

    def record(
        self, confidences: Sequence[float], kept: int, elapsed_ms: float
    ) -> None:
        assert elapsed_ms > 0
        self.steps.append((len(confidences), kept))

    def record_plain(self, elapsed_ms: float | None) -> None:
        del elapsed_ms


def _generate(
    executor: MlxGenerationExecutor,
    prompts: dict[str, tuple[int, ...]],
    *,
    sampling: SamplingParams,
    max_tokens: int = 24,
    terminal: tuple[int, ...] = (),
) -> dict[str, list[int]]:
    scheduler = NativeContinuousScheduler(
        executor,
        NativeCacheCoordinator(executor.cache_backend, NoPrefixCache()),
        execution_mode=executor.execution_mode,
        terminal_token_ids=terminal,
    )
    for name, prompt in prompts.items():
        scheduler.submit(
            SchedulableRequest(
                request_id=name,
                prompt_token_ids=prompt,
                sampling=sampling,
                enqueued_at=time.perf_counter(),
                max_tokens=max_tokens,
            )
        )
    tokens: dict[str, list[int]] = {name: [] for name in prompts}
    for _ in range(400):
        if scheduler.idle():
            break
        for event in scheduler.tick():
            assert event.kind != "execution_error", event.message
            if event.kind == "token":
                tokens[event.request_id].append(event.token_id)
                done = len(tokens[event.request_id]) >= max_tokens
                if done or event.token_id in terminal:
                    scheduler.finish(event.request_id)
    assert scheduler.idle()
    backend = executor.cache_backend
    assert backend.free_page_count() == backend.num_pages
    assert backend.metrics()["window_used_pages"] == 0
    return tokens


def _speculator(script: dict[int, int], verified: int = 3) -> Speculator:
    return Speculator(_ScriptedDrafter(script), _FixedLength(verified))


@needs_metal
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "sampling",
    [SamplingParams(), SamplingParams(temperature=0.9, top_p=0.95, seed=11)],
    ids=["greedy", "sampled"],
)
def test_speculation_keeps_exactly_the_tokens_the_target_draws(
    model: Any, overlap: bool, sampling: SamplingParams
) -> None:
    plain = _generate(
        _executor(model, overlap=overlap), {"a": _PROMPT}, sampling=sampling
    )["a"]
    script = {len(_PROMPT) + index: token for index, token in enumerate(plain)}
    script.update({len(_PROMPT) + len(plain) + index: 0 for index in range(4)})
    wrong = {
        position: token + 1 if position % 3 == 0 else token
        for position, token in script.items()
    }

    for proposals, keeps_all, verified in (
        (script, True, 3),
        (wrong, False, 3),
        (script, True, 1),
        (script, True, 0),
    ):
        speculator = _speculator(proposals, verified)
        executor = _executor(model, overlap=overlap, speculator=speculator)
        tokens = _generate(executor, {"a": _PROMPT}, sampling=sampling)["a"]
        assert tokens == plain
        steps = speculator.controller.steps
        assert steps
        assert all(count <= verified for count, _ in steps)
        assert all(kept == count for count, kept in steps) == keeps_all
        assert len(plain) - 1 - sum(kept + 1 for _, kept in steps) in (0, 1)
        assert sum(speculator.drafter.read_tokens) == len(_PROMPT) + len(plain) - 1


@needs_metal
def test_speculation_stops_at_a_terminal_token_inside_a_step(model: Any) -> None:
    plain = _generate(_executor(model), {"a": _PROMPT}, sampling=SamplingParams())["a"]
    script = {len(_PROMPT) + index: token for index, token in enumerate(plain)}
    terminal = plain[2]
    expected = plain[: plain.index(terminal) + 1]

    executor = _executor(model, speculator=_speculator(script))
    tokens = _generate(
        executor, {"a": _PROMPT}, sampling=SamplingParams(), terminal=(terminal,)
    )["a"]

    assert tokens == expected


@needs_metal
def test_speculation_waits_until_one_request_decodes_alone(model: Any) -> None:
    prompts = {"a": _PROMPT, "b": _PROMPT[:7]}
    speculator = _speculator(dict.fromkeys(range(400), 0))
    executor = _executor(model, speculator=speculator)

    tokens = _generate(executor, prompts, sampling=SamplingParams(), max_tokens=8)

    assert all(len(values) == 8 for values in tokens.values())
    assert speculator.controller.steps == []
    plain = _generate(
        _executor(model), prompts, sampling=SamplingParams(), max_tokens=8
    )
    assert tokens == plain


@needs_metal
def test_executor_rejects_proposals_it_cannot_verify(model: Any) -> None:
    def batch(executor: MlxGenerationExecutor, names: str) -> ExecutionBatch:
        return ExecutionBatch(
            requests=tuple(
                ExecutionRequest(
                    request_id=name,
                    phase="decode",
                    token_ids=(1,),
                    positions=(0,),
                    cache_handle=executor.cache_backend.create(name),
                    sampling=SamplingParams(),
                    draft_tokens=2,
                )
                for name in names
            )
        )

    without = _executor(model)
    assert without.draft_tokens("a", without.cache_backend.create("a"), 5) == 0
    for executor, names in (
        (without, "a"),
        (_executor(model, speculator=_speculator({})), "ab"),
    ):
        with pytest.raises(BatchExecutionError, match="proposed tokens"):
            executor.execute_batch(batch(executor, names))


@needs_metal
def test_packed_drafter_reads_and_proposes(model: Any, drafter_path: Path) -> None:
    checkpoint = load_drafter(drafter_path, drafter_path)
    drafter = checkpoint.build(model)
    rng = np.random.default_rng(0)
    states = mx.array(rng.normal(size=(40, 2, inkling2_pack.HIDDEN)).astype(np.float32))
    token = mx.array([5])

    whole = drafter.new_context()
    drafter.read(whole, states)
    chunked = drafter.new_context()
    for start, end in ((0, 11), (11, 12), (12, 31), (31, 40)):
        drafter.read(chunked, states[start:end])

    assert whole.length == chunked.length == 40
    proposed, confidences = drafter.propose(whole, token, 3)
    assert proposed.shape == confidences.shape == (3,)
    assert all(0 <= int(value) < inkling2_pack.VOCAB for value in proposed.tolist())
    assert all(0.0 < value <= 1.0 for value in confidences.tolist())
    assert drafter.propose(whole, token, 2)[0].tolist() == proposed.tolist()[:2]
    assert drafter.propose(chunked, token, 3)[0].tolist() == proposed.tolist()
    assert 0.0 < checkpoint.kv_share < 1.0
    assert checkpoint.nbytes > 0


@needs_metal
def test_drafter_generates_the_same_tokens(model: Any, drafter_path: Path) -> None:
    plain = _generate(_executor(model), {"a": _PROMPT}, sampling=SamplingParams())
    drafter = load_drafter(drafter_path, drafter_path).build(model)
    speculator = Speculator(drafter, _FixedLength())

    tokens = _generate(
        _executor(model, overlap=True, speculator=speculator),
        {"a": _PROMPT},
        sampling=SamplingParams(),
    )

    assert tokens == plain
    assert speculator.controller.steps


def test_drafter_prefers_weights_distilled_for_the_pack(
    pack: Path, drafter_path: Path, tmp_path: Path
) -> None:
    inkling2_pack.write_drafter(tmp_path)
    assert load_drafter(tmp_path, pack).weights.name == "mtp_4bit.safetensors"
    for name in ("mtp_4bit_5p4gb_v1", "mtp_4bit_5p4gb_v2", "mtp_4bit_8p9gb_v2"):
        (tmp_path / f"{name}.safetensors").write_bytes(b"")
    chosen = load_drafter(tmp_path, pack).weights.name
    assert chosen == "mtp_4bit_5p4gb_v2.safetensors"
    assert load_drafter(tmp_path, drafter_path).weights.name == "mtp_4bit.safetensors"


def test_drafter_errors_name_the_problem(pack: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a DSpark drafter"):
        load_drafter(pack, pack)
    inkling2_pack.write_drafter(tmp_path)
    (tmp_path / "mtp_4bit.safetensors").unlink()
    with pytest.raises(ValueError, match="are missing"):
        load_drafter(tmp_path, pack)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "format": "x"}))
    with pytest.raises(ValueError, match="manifest format"):
        load_drafter(tmp_path, pack)


def test_startup_rejects_a_drafter_the_family_cannot_use(tmp_path: Path) -> None:
    spec = get_architecture_spec("Qwen2ForCausalLM")
    architecture = NativeArchitecture(
        str(tmp_path), tmp_path, "Qwen2ForCausalLM", {}, spec, spec.resolve
    )

    assert _open_drafter(None, architecture) is None
    with pytest.raises(NativeBootstrapFailure) as error:
        _open_drafter(str(tmp_path), architecture)
    assert error.value.error.code == "DRAFTER_UNSUPPORTED"
    assert "no speculative decoding" in error.value.error.message


def test_config_lists_every_visit_of_the_loop() -> None:
    config = parse_inkling2_config(inkling2_pack.pack_config())

    layers = [layer for layer, _ in config.visits]
    assert layers == [0, 1, 2, 3, 2, 3, 2, 3, 4, 5]
    assert [step for _, step in config.visits] == [0, 0, 0, 0, 1, 1, 2, 2, 0, 0]
    assert config.window_visits == (0, 2, 4, 6, 8)


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        (None, {"architectures": ["Other"]}, "expected architectures"),
        (None, {"text_config": None}, "missing text_config"),
        ("text_config", {"loop_config": None}, "missing loop_config"),
        ("text_config", {"hidden_size": ...}, "missing 'hidden_size'"),
        ("text_config", {"sconv_kernel_size": 3}, "kernel size 4"),
        ("text_config", {"n_shared_experts": 1}, "unsupported"),
        ("text_config", {"num_key_value_heads": 3}, "divisible"),
        ("loop_config", {"is_varloop": True}, "unsupported"),
        ("loop_config", {"core_end": 6}, "strictly inside"),
        ("loop_config", {"core_start": 1}, "block boundary"),
    ],
)
def test_config_rejects_unsupported_graphs(
    section: str | None, changes: dict[str, Any], message: str
) -> None:
    payload = inkling2_pack.pack_config()
    target = payload
    if section is not None:
        target = payload["text_config"]
    if section == "loop_config":
        target = target["loop_config"]
    for name, value in changes.items():
        if value is ...:
            del target[name]
        else:
            target[name] = value

    with pytest.raises(ValueError, match=message):
        parse_inkling2_config(payload)


def _plan(path: Path) -> Any:
    plan = get_architecture_spec(_ARCHITECTURE).resolve
    return plan.create_weight_adapter().build_plan(plan.load_weight_index(path))


def test_mapping_covers_the_text_graph_and_skips_vision(pack: Path) -> None:
    names = {entry.canonical_name for entry in _plan(pack).entries}

    assert "layers.1.linear.o.rtr4.signs_out" in names
    assert "layers.5.experts.gu.nvfp4.outer" in names
    assert "layers.0.experts.dn.map" in names
    assert "embed.order" in names and "head.0.lm8.levels" in names
    assert not any("visual" in name for name in names)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("manifest.json", None, "only from LookAtMyQuant packs"),
        ("manifest.json", '{"format": "gguf"}', "only from LookAtMyQuant packs"),
        (
            "manifest.json",
            '{"format": "inkling2-packed-v1-sqyfr", "subformat": "v9"}',
            "unsupported LookAtMyQuant subformat 'v9'",
        ),
        ("luts.npz", None, "missing native weight files: luts.npz"),
        ("luts.npz", "not an archive", "could not read luts.npz"),
        ("steering.safetensors", "short", "could not read steering.safetensors"),
    ],
)
def test_index_rejects_other_layouts_before_reading_tensors(
    tmp_path: Path, name: str, content: str | None, message: str
) -> None:
    inkling2_pack.write_pack(tmp_path)
    (tmp_path / name).unlink()
    if content is not None:
        (tmp_path / name).write_text(content)

    with pytest.raises(WeightArtifactValidationError, match=message):
        _plan(tmp_path)


def test_mapping_rejects_incomplete_or_unknown_metadata(tmp_path: Path) -> None:
    inkling2_pack.write_pack(tmp_path)
    meta_path = tmp_path / "layer_02_meta.json"
    meta = json.loads(meta_path.read_text())

    linear = meta["linears"]["self_attn.q_proj.linear"]
    for changes, message in (
        ({"fmt": "xyz"}, "unknown packed format"),
        ({"k": 7}, "with 7 bits"),
        ({"safe": "nothing"}, "pack is missing"),
    ):
        changed = json.loads(json.dumps(meta))
        changed["linears"]["self_attn.q_proj.linear"] = {**linear, **changes}
        meta_path.write_text(json.dumps(changed))
        with pytest.raises(WeightMappingBug, match=message):
            _plan(tmp_path)

    changed = json.loads(json.dumps(meta))
    del changed["experts"]["dn"]
    meta_path.write_text(json.dumps(changed))
    with pytest.raises(WeightMappingBug, match="does not describe its experts"):
        _plan(tmp_path)

    meta_path.write_text(json.dumps(meta))
    (tmp_path / "layer_03_packed.npz").unlink()
    with pytest.raises(WeightMappingBug, match="missing layer 3"):
        _plan(tmp_path)
