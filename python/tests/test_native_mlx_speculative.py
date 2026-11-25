"""Draft length choice and drafter context bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.speculative import DraftLengthController, Speculator

_SURE, _UNSURE = 0.95, 0.35


def _settled(
    kept: int,
    *,
    confidence: float = _SURE,
    verified: int = 3,
    plain_ms: float = 40.0,
    step_ms: float = 60.0,
) -> DraftLengthController:
    controller = DraftLengthController(7)
    for _ in range(100):
        controller.record_plain(plain_ms)
        controller.record((confidence,) * verified, kept, step_ms)
    return controller


def test_accepted_proposals_keep_drafting() -> None:
    assert _settled(3).choose(7) == 7
    assert _settled(3).choose(2) == 2
    assert _settled(3).choose(0) == 0


def test_rejected_proposals_stop_drafting() -> None:
    assert _settled(0).choose(7) == 0


def test_expensive_steps_stop_drafting() -> None:
    assert _settled(1, step_ms=120.0).choose(7) == 0
    assert _settled(1, step_ms=48.0).choose(7) > 0


def test_steps_verify_the_confident_proposals() -> None:
    controller = DraftLengthController(7)

    assert controller.trim([_SURE] * 7) >= 5
    assert controller.trim([_SURE, _SURE, _UNSURE, _SURE]) == 2
    assert controller.trim([_UNSURE, _SURE]) == 0
    assert controller.trim([]) == 0


def test_confidence_means_what_acceptance_shows() -> None:
    unreliable = _settled(0, confidence=_SURE)
    reliable = _settled(3, confidence=_UNSURE)

    assert unreliable.trim([_SURE] * 4) == 0
    assert reliable.trim([_UNSURE] * 4) >= 3


def test_plain_steps_probe_again_periodically() -> None:
    controller = _settled(0)
    chosen = []
    for _ in range(40):
        chosen.append(controller.choose(7))
        controller.record_plain(40.0)

    probes = [index for index, count in enumerate(chosen) if count]
    assert probes and probes[0] >= 30
    assert controller.choose(1) == 1
    controller.record((_SURE,) * 2, 0, 60.0)
    assert controller.choose(7) == 0


def test_a_stalled_step_does_not_look_like_a_lasting_slowdown() -> None:
    controller = _settled(3)

    controller.record((_SURE,) * 3, 3, 60_000.0)

    assert controller.choose(7) == 7


def test_first_steps_of_a_length_are_not_timed() -> None:
    controller = DraftLengthController(7)
    controller.record_plain(5_000.0)
    controller.record((_SURE,) * 3, 3, 5_000.0)
    for _ in range(30):
        controller.record_plain(40.0)
        controller.record((_SURE,) * 3, 3, 60.0)

    assert controller.choose(7) == 7
    assert controller._plain_ms == pytest.approx(40.0)
    assert controller._cost_ms[3] == pytest.approx(60.0)


def test_unknown_intervals_are_not_recorded_as_step_times() -> None:
    controller = DraftLengthController(7)
    controller.record_plain(None)
    controller.record((_SURE,) * 2, 2, 70.0)

    assert controller.choose(7) == 7


@dataclass
class _Context:
    length: int = 0

    def arrays(self) -> list[mx.array]:
        return [mx.array([self.length])]


@dataclass
class _Cache:
    draft_context: Any = None


class _Drafter:
    taps = (0,)
    max_draft_tokens = 4

    def new_context(self) -> _Context:
        return _Context()

    def read(self, context: _Context, taps: mx.array) -> None:
        context.length += int(taps.shape[0])

    def propose(
        self, context: Any, token: mx.array, count: int
    ) -> tuple[mx.array, mx.array]:
        del context, token
        tokens = mx.arange(10, 10 + count)
        return tokens, mx.array([_SURE, _SURE, _UNSURE, _SURE])[:count]


def test_contexts_follow_the_tokens_of_their_request() -> None:
    speculator = Speculator.from_drafter(_Drafter())
    first, second = _Cache(), _Cache()
    taps = mx.zeros((7, 1, 8))

    arrays = speculator.read((first, second), (0, 0), (5, 2), taps)

    assert (first.draft_context.length, second.draft_context.length) == (5, 2)
    assert [int(array[0]) for array in arrays] == [5, 2]
    assert speculator.draft_tokens(first, 3) > 0
    assert speculator.draft_tokens(_Cache(), 3) == 0


def test_a_context_that_missed_tokens_is_dropped() -> None:
    speculator = Speculator.from_drafter(_Drafter())
    cache = _Cache(_Context(length=4))

    assert speculator.read((cache,), (6,), (1,), mx.zeros((1, 1, 8))) == []

    assert cache.draft_context is None
    assert speculator.draft_tokens(cache, 3) == 0


@pytest.mark.parametrize("limit", [1, 4, 9])
def test_choice_never_exceeds_the_drafter_or_the_request(limit: int) -> None:
    controller = DraftLengthController(4)
    assert controller.choose(limit) <= min(limit, 4)


def test_proposals_are_cut_at_the_first_unlikely_one() -> None:
    speculator = Speculator.from_drafter(_Drafter())
    cache: Any = _Cache(_Context(length=4))

    tokens, confidences = speculator.propose(cache, mx.array([3]), 4)

    assert tokens.tolist() == [10, 11]
    assert confidences == pytest.approx((_SURE, _SURE))
