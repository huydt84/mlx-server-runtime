"""Deterministic prompt-bank generation for MLX Air benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any


_WORDS = (
    "amber",
    "cedar",
    "indigo",
    "quartz",
    "silver",
    "violet",
    "maple",
    "cobalt",
    "willow",
    "copper",
    "saffron",
    "granite",
    "juniper",
    "pearl",
    "crimson",
    "linen",
)


@dataclass(frozen=True)
class Prompt:
    """One generated prompt and its reproducibility metadata."""

    group: str
    name: str
    index: int
    target_tokens: int
    text: str
    sha256: str


def generate_prompt_bank(configuration: dict[str, Any]) -> dict[str, list[Prompt]]:
    """Generate every selected measured and warmup prompt group.

    Args:
        configuration: Fully selected benchmark configuration.

    Returns:
        Prompt groups keyed by configuration name.
    """
    seed = int(configuration["sampling"]["seed"])
    return {
        name: _generate_group(name, definition, seed)
        for name, definition in configuration["prompt_bank"].items()
    }


def shared_prefix_prime_prompts(
    prompts: list[Prompt], *, trial_index: int, request_count: int
) -> list[Prompt]:
    """Build one non-measured suffix per shared prefix used by a trial."""

    selected = [
        prompts[(trial_index * request_count + index) % len(prompts)]
        for index in range(request_count)
    ]
    primes: dict[str, Prompt] = {}
    marker = ". Unique suffix "
    for prompt in selected:
        shared, separator, _suffix = prompt.text.partition(marker)
        if not separator:
            raise ValueError(
                f"shared-prefix prompt {prompt.name!r} has no unique-suffix marker"
            )
        digest = hashlib.sha256(shared.encode()).hexdigest()
        if digest in primes:
            continue
        text = (
            f"{shared}. Benchmark-only priming suffix {trial_index}-{len(primes)}; "
            "this suffix is intentionally absent from measured prompts"
        )
        primes[digest] = Prompt(
            group=prompt.group,
            name=f"prime-{prompt.name}",
            index=prompt.index,
            target_tokens=prompt.target_tokens,
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
    return list(primes.values())


def _generate_group(name: str, definition: dict[str, Any], seed: int) -> list[Prompt]:
    group_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"
    )
    rng = random.Random(group_seed)
    count = int(definition["count"])
    target_tokens = int(definition["target_tokens"])
    kind = str(definition["kind"])
    material = str(definition["material"])
    prompts: list[Prompt] = []
    for index in range(count):
        words = [rng.choice(_WORDS) for _ in range(max(1, target_tokens - 8))]
        marker = f"mlx-air {material} {name} {index}"
        if kind == "decode-promoting":
            text = (
                f"{marker}. Produce a detailed numbered sequence with exactly the "
                f"requested number of steps. Source material: {' '.join(words)}"
            )
        elif kind == "shared-prefix":
            suffixes = int(definition["suffixes_per_group"])
            prefix_index = index // suffixes
            prefix_rng = random.Random(
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:{name}:prefix:{prefix_index}".encode()
                    ).digest()[:8],
                    "big",
                )
            )
            shared_tokens = max(1, target_tokens * 3 // 4)
            shared = " ".join(prefix_rng.choice(_WORDS) for _ in range(shared_tokens))
            suffix = " ".join(words[: max(1, target_tokens - shared_tokens - 8)])
            text = (
                f"mlx-air shared prefix {prefix_index}: {shared}. "
                f"Unique suffix {index}: {suffix}"
            )
        elif kind == "cache-pressure":
            unique = hashlib.sha256(
                f"{seed}:{name}:unique:{index}".encode()
            ).hexdigest()
            text = f"{marker} unique-prefix-{unique}. Summarize: {' '.join(words)}"
        else:
            text = f"{marker}. Summarize the following material: {' '.join(words)}"
        prompts.append(
            Prompt(
                group=name,
                name=f"{name}-{index:04d}",
                index=index,
                target_tokens=target_tokens,
                text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return prompts
