"""Speculative decoding for one request at a time.

A drafter proposes tokens from the target model's hidden states, and the target
checks all of them in one step. A proposed token is kept only when it equals
the token the target itself would have drawn at that position, so a request
gets the tokens it would get without a drafter.

Verifying more tokens makes a step slower whether or not they are kept.
:class:`DraftLengthController` therefore tracks how often proposals of a given
confidence are kept and what steps cost. Each step verifies as many of its
proposals as yield the most tokens per second, and a request stops drafting
while plain steps are faster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx

from .cache import RequestCache


class Drafter(Protocol):
    """Model-specific proposer of likely next tokens."""

    taps: tuple[int, ...]
    """Target states to read, as ``forward_with_taps`` names them."""

    max_draft_tokens: int
    """Most tokens one proposal holds."""

    def new_context(self) -> Any:
        """Return the context of a request with no committed tokens."""

    def read(self, context: Any, taps: mx.array) -> None:
        """Append committed target states to a context."""

    def propose(
        self, context: Any, token: mx.array, count: int
    ) -> tuple[mx.array, mx.array]:
        """Return ``(count,)`` tokens proposed to follow ``token``.

        The second array holds the drafter's probability of each token.
        """


class DrafterCheckpoint(Protocol):
    """Drafter checkpoint that has not been loaded yet."""

    nbytes: int
    """Bytes the drafter holds once built."""

    kv_share: float
    """Part of the KV budget its request contexts need."""

    def build(self, target: Any) -> Drafter:
        """Load the drafter for a loaded target model."""


class SpeculativeModel(Protocol):
    """Target model whose hidden states a drafter reads."""

    def forward_with_taps(
        self, input_ids: mx.array, forward_batch: Any, taps: tuple[int, ...]
    ) -> tuple[mx.array, mx.array]:
        """Return logits and ``(total tokens, taps, hidden)`` states."""


class DraftLengthController:
    """Choose how many tokens to propose and how many of them to verify.

    Acceptance is tracked per range of drafter confidence as the chance that a
    token is kept given that the ones before it were. Costs are tracked per
    verified length; lengths that have not run yet use a prior scaled by the
    ones that have.
    """

    _BINS = 10
    _ACCEPT_PRIOR_WEIGHT = 4.0
    _ACCEPT_DECAY = 0.98
    _COST_DECAY = 0.9
    _RATE_DECAY = 0.95
    _MIN_RATE_STEPS = 8
    _EMPTY_COST = 1.2
    _BASE_COST = 1.35
    _TOKEN_COST = 0.24
    _PROBE_INTERVAL = 32

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._seen = [self._ACCEPT_PRIOR_WEIGHT] * self._BINS
        self._kept = [
            self._ACCEPT_PRIOR_WEIGHT * ((index + 0.5) / self._BINS) ** 3
            for index in range(self._BINS)
        ]
        self._plain_ms: float | None = None
        self._cost_ms: dict[int, float] = {}
        self._warm: set[int] = set()
        self._plain_steps = 0
        self._rate_steps = 0
        self._step_tokens = 0.0
        self._step_ms = 0.0

    def choose(self, limit: int) -> int:
        """Return how many tokens to propose for the next step.

        Args:
            limit: Most tokens the request can still use beyond the next one.

        Returns:
            A count from zero through ``limit``; zero means a plain step.
        """

        count = min(limit, self.max_tokens)
        if count <= 0:
            return 0
        if self._rate_steps < self._MIN_RATE_STEPS:
            return count
        plain_ms = self._plain_ms or self._unit_ms()
        pays = self._step_tokens * plain_ms > self._step_ms
        return count if pays or self._plain_steps >= self._PROBE_INTERVAL else 0

    def trim(self, confidences: Sequence[float]) -> int:
        """Return how many leading proposals to verify.

        Args:
            confidences: The drafter's probability of each proposed token.
        """

        best, best_rate = 0, 1.0 / self._cost(0)
        chain, expected = 1.0, 1.0
        for count, confidence in enumerate(confidences, start=1):
            index = self._bin(confidence)
            chain *= self._kept[index] / self._seen[index]
            expected += chain
            rate = expected / self._cost(count)
            if rate > best_rate:
                best, best_rate = count, rate
        return best

    def record(
        self, confidences: Sequence[float], kept: int, elapsed_ms: float
    ) -> None:
        """Record one step that proposed tokens.

        Args:
            confidences: Drafter probability of each verified token.
            kept: Verified tokens the request keeps.
            elapsed_ms: Time the step took, with drafting.
        """

        self._plain_steps = 0
        verified = len(confidences)
        for position in range(min(verified, kept + 1)):
            index = self._bin(confidences[position])
            self._seen[index] = self._seen[index] * self._ACCEPT_DECAY + 1.0
            self._kept[index] = self._kept[index] * self._ACCEPT_DECAY + float(
                position < kept
            )
        if not self._is_warm(verified):
            return
        cost = self._blend(self._cost_ms.get(verified), elapsed_ms)
        self._cost_ms[verified] = cost
        self._rate_steps += 1
        self._step_tokens = self._step_tokens * self._RATE_DECAY + kept + 1
        self._step_ms = self._step_ms * self._RATE_DECAY + min(elapsed_ms, 2.0 * cost)

    def record_plain(self, elapsed_ms: float | None) -> None:
        """Record one single-request step without proposals.

        Args:
            elapsed_ms: Time since the previous step, or ``None`` when the step
                did not directly follow another one.
        """

        self._plain_steps += 1
        if elapsed_ms is not None and self._is_warm(-1):
            self._plain_ms = self._blend(self._plain_ms, elapsed_ms)

    def _bin(self, confidence: float) -> int:
        return min(self._BINS - 1, max(0, int(confidence * self._BINS)))

    def _is_warm(self, verified: int) -> bool:
        """Return whether a length ran before; its first step compiles kernels."""

        if verified in self._warm:
            return True
        self._warm.add(verified)
        return False

    def _blend(self, current: float | None, value: float) -> float:
        if current is None:
            return value
        value = min(value, 2.0 * current)
        return self._COST_DECAY * current + (1.0 - self._COST_DECAY) * value

    def _cost(self, verified: int) -> float:
        """Return the time of a step, measured or scaled from measured ones."""

        measured = self._cost_ms.get(verified)
        if measured is not None:
            return measured
        return self._prior_cost(verified) * self._unit_ms()

    def _unit_ms(self) -> float:
        """Return the plain-step time that measured steps imply."""

        if not self._cost_ms:
            return self._plain_ms or 1.0
        ratios = [
            cost / self._prior_cost(length) for length, cost in self._cost_ms.items()
        ]
        return sum(ratios) / len(ratios)

    def _prior_cost(self, verified: int) -> float:
        if not verified:
            return self._EMPTY_COST
        return self._BASE_COST + self._TOKEN_COST * verified


@dataclass
class Speculator:
    """Drafter of one target model with its length controller."""

    drafter: Drafter
    controller: DraftLengthController

    @classmethod
    def from_drafter(cls, drafter: Drafter) -> Speculator:
        """Return a speculator that uses ``drafter``."""

        return cls(drafter, DraftLengthController(drafter.max_draft_tokens))

    def read(
        self,
        caches: tuple[RequestCache, ...],
        cache_lengths: tuple[int, ...],
        token_lengths: tuple[int, ...],
        taps: mx.array,
    ) -> list[mx.array]:
        """Give each request's drafter context its tokens of one step.

        A request whose context misses earlier tokens loses it and continues
        without proposals.

        Args:
            caches: Request caches of the step.
            cache_lengths: Committed tokens before the step.
            token_lengths: Tokens each request commits.
            taps: Tapped states of every request's tokens, concatenated.

        Returns:
            Arrays to evaluate with the step.
        """

        arrays: list[mx.array] = []
        start = 0
        for cache, length, count in zip(
            caches, cache_lengths, token_lengths, strict=True
        ):
            owner: Any = cache
            if length == 0:
                owner.draft_context = self.drafter.new_context()
            context = owner.draft_context
            if context is not None and context.length != length:
                context = owner.draft_context = None
            if context is not None:
                self.drafter.read(context, taps[start : start + count])
                arrays.extend(context.arrays())
            start += count
        return arrays

    def draft_tokens(self, cache: RequestCache, limit: int) -> int:
        """Return how many tokens to propose for a request's next step."""

        context = getattr(cache, "draft_context", None)
        if context is None:
            return 0
        return self.controller.choose(limit)

    def propose(
        self, cache: RequestCache, token: mx.array, count: int
    ) -> tuple[mx.array, tuple[float, ...]]:
        """Propose tokens and keep those worth verifying.

        The proposals are read back, so a step verifies only tokens that are
        likely enough.

        Args:
            cache: Request whose context holds every committed token before
                ``token``.
            token: ``(1,)`` last sampled token.
            count: Tokens to propose.

        Returns:
            The int32 tokens to verify, possibly none, and their confidences.
        """

        owner: Any = cache
        tokens, confidences = self.drafter.propose(owner.draft_context, token, count)
        mx.eval(tokens, confidences)
        values = [float(value) for value in confidences.tolist()]
        kept = self.controller.trim(values)
        return tokens[:kept].astype(mx.int32), tuple(values[:kept])
