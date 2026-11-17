"""Deterministic prompt-bank generation for MLX Air benchmarks."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from mlx_benchmark import mail_assistant

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
    system: str | None = None


def prompt_messages(prompt: Prompt) -> list[dict[str, str]]:
    """Return the chat messages sent for ``prompt``."""

    messages = [{"role": "user", "content": prompt.text}]
    if prompt.system is not None:
        messages.insert(0, {"role": "system", "content": prompt.system})
    return messages


def _digest(text: str, system: str | None) -> str:
    material = text if system is None else f"{system}\n\n{text}"
    return hashlib.sha256(material.encode()).hexdigest()


def generate_prompt_bank(configuration: dict[str, Any]) -> dict[str, list[Prompt]]:
    """Generate every selected measured prompt group, keyed by group name."""
    seed = int(configuration["sampling"]["seed"])
    return {
        name: _generate_group(name, definition, seed, "measured")
        for name, definition in configuration["prompts"].items()
    }


def generate_warmup_prompts(
    configuration: dict[str, Any], workload: dict[str, Any]
) -> list[Prompt]:
    """Generate one concurrency's worth of warmup prompts shaped like a workload.

    Warmup prompts use a separate seed namespace, so they never match measured
    prompt text or share a measured prompt's prefix.
    """
    group = workload["prompt_group"]
    definition = {
        **configuration["prompts"][group],
        "count": int(workload["concurrency"]),
    }
    seed = int(configuration["sampling"]["seed"])
    return _generate_group(f"warmup-{group}", definition, seed, "warmup")


def shared_prefix_prime_prompts(
    prompts: list[Prompt], *, trial_index: int, request_count: int
) -> list[Prompt]:
    """Build one non-measured suffix per shared prefix used by a trial.

    A prompt with a system message shares that message with every prompt of
    the same task, so its priming request repeats the system message with a
    benchmark-only user message.
    """

    selected = [
        prompts[(trial_index * request_count + index) % len(prompts)]
        for index in range(request_count)
    ]
    primes: dict[str, Prompt] = {}
    marker = ". Unique suffix "
    for prompt in selected:
        if prompt.system is not None:
            digest = hashlib.sha256(prompt.system.encode()).hexdigest()
            if digest in primes:
                continue
            text = (
                f"Output language: {mail_assistant.OUTPUT_LANGUAGE}\n\n<email>\n"
                f"Benchmark-only priming email {trial_index}-{len(primes)}; this "
                "email is intentionally absent from measured prompts.\n</email>"
            )
            primes[digest] = Prompt(
                group=prompt.group,
                name=f"prime-{prompt.name}",
                index=prompt.index,
                target_tokens=prompt.target_tokens,
                text=text,
                sha256=_digest(text, prompt.system),
                system=prompt.system,
            )
            continue
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


def _generate_group(
    name: str, definition: dict[str, Any], seed: int, material: str
) -> list[Prompt]:
    group_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"
    )
    rng = random.Random(group_seed)
    count = int(definition["count"])
    target_tokens = int(definition["target_tokens"])
    kind = str(definition["kind"])
    prompts: list[Prompt] = []
    tasks = mail_assistant.task_names()
    for index in range(count):
        system: str | None = None
        if kind == "mail-assistant":
            task = tasks[index % len(tasks)]
            system = mail_assistant.system_message(task)
            text = mail_assistant.user_message(
                rng, index, mail_assistant.body_tokens_for(rng, task, target_tokens)
            )
            prompts.append(
                Prompt(
                    group=name,
                    name=f"{name}-{index:04d}-{task}",
                    index=index,
                    target_tokens=target_tokens,
                    text=text,
                    sha256=_digest(text, system),
                    system=system,
                )
            )
            continue
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
