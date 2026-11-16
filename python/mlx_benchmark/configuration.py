"""Load and select declarative MLX Air benchmark configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_benchmark.statistics import METRICS, STREAMING_ONLY_METRICS

import tomllib


SCHEMA_VERSION = 2
_PROMPT_KINDS = {
    "short",
    "medium",
    "long",
    "decode-promoting",
    "shared-prefix",
    "cache-pressure",
}
_LOAD_MODES = {"sequential", "burst", "closed-loop"}
_CACHE_STATES = {"cold", "warm-prefix", "cache-pressure"}
_BACKENDS = {"v1", "native-mlx"}
_PROFILERS = {"pipeline", "graph"}
_DEFAULT_REVISION = "resolve-at-run"


class ConfigurationError(ValueError):
    """A benchmark configuration error tied to an exact path and field."""

    def __init__(self, path: Path, field: str, message: str) -> None:
        super().__init__(f"{path}: {field}: {message}")
        self.path = path
        self.field = field


@dataclass(frozen=True)
class SelectedConfiguration:
    """The fully expanded values selected for one benchmark command."""

    source_path: Path
    values: dict[str, Any]


def load_selected_configuration(
    path: Path,
    *,
    suite_name: str,
    focus_name: str | None,
    profile: str,
    server_mode: str,
) -> SelectedConfiguration:
    """Load, validate, and select a benchmark configuration.

    Raises:
        ConfigurationError: If parsing, validation, or selection fails.
    """
    path = path.expanduser().resolve()
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ConfigurationError(path, "<file>", "file does not exist") from error
    except OSError as error:
        raise ConfigurationError(
            path, "<file>", f"cannot read file: {error}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(path, "<toml>", str(error)) from error

    checker = _Checker(path)
    _validate(checker, raw)
    return SelectedConfiguration(
        source_path=path,
        values=_select(
            checker,
            raw,
            suite_name=suite_name,
            focus_name=focus_name,
            profile=profile,
            server_mode=server_mode,
        ),
    )


class _Checker:
    """Typed field access that reports the exact failing field."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def fail(self, field: str, message: str) -> None:
        raise ConfigurationError(self.path, field, message)

    def table(self, values: dict[str, Any], key: str, field: str) -> dict[str, Any]:
        value = values.get(key)
        if not isinstance(value, dict):
            self.fail(field, "must be a table")
        return value

    def named_tables(
        self, raw: dict[str, Any], key: str, *, required: bool = True
    ) -> dict[str, dict[str, Any]]:
        if key not in raw and not required:
            return {}
        values = self.table(raw, key, key)
        if required and not values:
            self.fail(key, "must contain at least one named table")
        for name, value in values.items():
            if not isinstance(value, dict):
                self.fail(f"{key}.{name}", "must be a table")
        return values

    def string(
        self,
        values: dict[str, Any],
        key: str,
        field: str,
        *,
        choices: set[str] | dict[str, Any] | None = None,
    ) -> str:
        value = values.get(key)
        if not isinstance(value, str) or not value:
            self.fail(field, "must be a non-empty string")
        if choices is not None and value not in choices:
            self.fail(field, f"unsupported value {value!r}")
        return value

    def boolean(self, values: dict[str, Any], key: str, field: str) -> bool:
        value = values.get(key)
        if not isinstance(value, bool):
            self.fail(field, "must be a boolean")
        return value

    def integer(
        self,
        values: dict[str, Any],
        key: str,
        field: str,
        minimum: int,
        maximum: int,
    ) -> int:
        value = values.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            self.fail(field, "must be an integer")
        if not minimum <= value <= maximum:
            self.fail(field, f"must be between {minimum} and {maximum}")
        return value

    def number(
        self,
        values: dict[str, Any],
        key: str,
        field: str,
        minimum: float,
        maximum: float,
    ) -> float:
        value = values.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            self.fail(field, "must be a number")
        if not minimum <= value <= maximum:
            self.fail(field, f"must be between {minimum} and {maximum}")
        return float(value)

    def names(
        self,
        values: dict[str, Any],
        key: str,
        field: str,
        choices: set[str] | dict[str, Any],
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        value = values.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            self.fail(field, "must be an array of non-empty strings")
        if not allow_empty and not value:
            self.fail(field, "must not be empty")
        if len(value) != len(set(value)):
            self.fail(field, "must not contain duplicates")
        unknown = [item for item in value if item not in choices]
        if unknown:
            self.fail(field, f"unknown values {unknown!r}")
        return value


def _validate(check: _Checker, raw: dict[str, Any]) -> None:
    if raw.get("schema_version") != SCHEMA_VERSION:
        check.fail("schema_version", f"must be {SCHEMA_VERSION}")

    sampling = check.table(raw, "sampling", "sampling")
    check.integer(sampling, "seed", "sampling.seed", 0, 2**63 - 1)
    check.number(sampling, "temperature", "sampling.temperature", 0.0, 2.0)
    check.number(sampling, "top_p", "sampling.top_p", 0.0, 1.0)
    for key in ("request_timeout_seconds", "readiness_timeout_seconds"):
        check.integer(sampling, key, f"sampling.{key}", 1, 3_600)

    calibration = check.table(raw, "calibration", "calibration")
    check.number(
        calibration, "max_cv_percent", "calibration.max_cv_percent", 0.0, 100.0
    )

    prompts = check.named_tables(raw, "prompts")
    for name, prompt in prompts.items():
        field = f"prompts.{name}"
        kind = check.string(prompt, "kind", f"{field}.kind", choices=_PROMPT_KINDS)
        count = check.integer(prompt, "count", f"{field}.count", 1, 10_000)
        check.integer(prompt, "target_tokens", f"{field}.target_tokens", 1, 65_536)
        if kind == "shared-prefix":
            groups = check.integer(
                prompt, "prefix_groups", f"{field}.prefix_groups", 1, count
            )
            suffixes = check.integer(
                prompt, "suffixes_per_group", f"{field}.suffixes_per_group", 1, count
            )
            if groups * suffixes != count:
                check.fail(
                    f"{field}.count", "must equal prefix_groups * suffixes_per_group"
                )

    models = check.named_tables(raw, "models")
    for name, model in models.items():
        field = f"models.{name}"
        check.string(model, "checkpoint", f"{field}.checkpoint")
        for key in ("tokenizer", "revision"):
            if key in model:
                check.string(model, key, f"{field}.{key}")
        if "cold_cache_allowance_tokens" in model:
            check.integer(
                model,
                "cold_cache_allowance_tokens",
                f"{field}.cold_cache_allowance_tokens",
                0,
                1_000_000,
            )

    runtimes = check.named_tables(raw, "runtime_configurations")
    for name, runtime in runtimes.items():
        field = f"runtime_configurations.{name}"
        check.string(runtime, "backend", f"{field}.backend", choices=_BACKENDS)
        environment = check.table(runtime, "environment", f"{field}.environment")
        for key, value in environment.items():
            if not key.startswith("MLX_RUNTIME_"):
                check.fail(f"{field}.environment", "keys must start with MLX_RUNTIME_")
            if not isinstance(value, str):
                check.fail(f"{field}.environment.{key}", "must be a string")

    workloads = check.named_tables(raw, "workloads")
    for name, workload in workloads.items():
        _validate_workload(check, f"workloads.{name}", workload, prompts)

    for name, focus in check.named_tables(raw, "focuses", required=False).items():
        check.names(focus, "models", f"focuses.{name}.models", models)
        check.names(focus, "workloads", f"focuses.{name}.workloads", workloads)

    diagnostics = check.named_tables(raw, "diagnostics", required=False)
    for name, diagnostic in diagnostics.items():
        field = f"diagnostics.{name}"
        check.names(diagnostic, "models", f"{field}.models", models)
        check.names(diagnostic, "workloads", f"{field}.workloads", workloads)
        check.names(diagnostic, "profilers", f"{field}.profilers", _PROFILERS)

    for name, suite in check.named_tables(raw, "suites").items():
        field = f"suites.{name}"
        check.integer(suite, "rounds", f"{field}.rounds", 1, 10)
        check.integer(suite, "trials", f"{field}.trials", 1, 100)
        check.names(
            suite, "runtime_configurations", f"{field}.runtime_configurations", runtimes
        )
        check.names(
            suite, "diagnostics", f"{field}.diagnostics", diagnostics, allow_empty=True
        )
        _suite_coverage(check, field, suite, models, workloads)


def _validate_workload(
    check: _Checker,
    field: str,
    workload: dict[str, Any],
    prompts: dict[str, Any],
) -> None:
    check.string(workload, "prompt_group", f"{field}.prompt_group", choices=prompts)
    check.integer(workload, "output_tokens", f"{field}.output_tokens", 1, 4_096)
    streaming = check.boolean(workload, "streaming", f"{field}.streaming")
    mode = check.string(
        workload, "load_mode", f"{field}.load_mode", choices=_LOAD_MODES
    )
    concurrency = check.integer(workload, "concurrency", f"{field}.concurrency", 1, 256)
    requests = check.integer(
        workload, "requests_per_trial", f"{field}.requests_per_trial", 1, 10_000
    )
    check.string(workload, "cache_state", f"{field}.cache_state", choices=_CACHE_STATES)
    metric = check.string(
        workload, "primary_metric", f"{field}.primary_metric", choices=METRICS
    )
    if not streaming and metric in STREAMING_ONLY_METRICS:
        check.fail(f"{field}.primary_metric", f"{metric!r} requires streaming = true")
    if mode == "sequential" and concurrency != 1:
        check.fail(f"{field}.concurrency", "must be 1 for sequential mode")
    if mode == "burst" and concurrency != requests:
        check.fail(
            f"{field}.concurrency", "must equal requests_per_trial for burst mode"
        )
    if mode == "closed-loop" and concurrency > requests:
        check.fail(
            f"{field}.concurrency",
            "must not exceed requests_per_trial for closed-loop mode",
        )


def _suite_coverage(
    check: _Checker,
    field: str,
    suite: dict[str, Any],
    models: dict[str, Any],
    workloads: dict[str, Any],
) -> dict[str, list[str]]:
    coverage = check.table(suite, "coverage", f"{field}.coverage")
    if not coverage:
        check.fail(f"{field}.coverage", "must cover at least one model")
    expanded: dict[str, list[str]] = {}
    for model, selected in coverage.items():
        entry = f"{field}.coverage.{model}"
        if model not in models:
            check.fail(entry, f"unknown model {model!r}")
        if selected == "all":
            expanded[model] = list(workloads)
        else:
            expanded[model] = check.names(coverage, model, entry, workloads)
    return expanded


def _select(
    check: _Checker,
    raw: dict[str, Any],
    *,
    suite_name: str,
    focus_name: str | None,
    profile: str,
    server_mode: str,
) -> dict[str, Any]:
    suites = raw["suites"]
    if suite_name not in suites:
        check.fail("--suite", f"unknown suite {suite_name!r}")
    suite = suites[suite_name]
    coverage = _suite_coverage(
        check, f"suites.{suite_name}", suite, raw["models"], raw["workloads"]
    )

    if focus_name is not None:
        focus = raw.get("focuses", {}).get(focus_name)
        if focus is None:
            check.fail("--focus", f"unknown focus {focus_name!r}")
        coverage = {
            model: [name for name in names if name in focus["workloads"]]
            for model, names in coverage.items()
            if model in focus["models"]
        }
        coverage = {model: names for model, names in coverage.items() if names}
        if not coverage:
            check.fail(
                "--focus",
                f"focus {focus_name!r} selects nothing from suite {suite_name!r}",
            )

    workload_names = list(
        dict.fromkeys(name for names in coverage.values() for name in names)
    )
    workloads = [
        {"name": name, **deepcopy(raw["workloads"][name]), "trials": suite["trials"]}
        for name in workload_names
    ]
    runtime_names = list(suite["runtime_configurations"])
    executions = [
        {
            "round": round_index + 1,
            "model": model,
            "runtime_configuration": runtime,
        }
        for round_index in range(suite["rounds"])
        for model_index, model in enumerate(coverage)
        for runtime in (
            runtime_names
            if (round_index + model_index) % 2 == 0
            else list(reversed(runtime_names))
        )
    ]
    if server_mode == "external" and len(executions) != 1:
        check.fail(
            "--server-mode",
            "external server mode requires exactly one declared execution",
        )

    diagnostic_names = list(suite["diagnostics"])
    if focus_name in diagnostic_names:
        diagnostic_names = [focus_name]
    if profile != "none" and not diagnostic_names:
        check.fail("--profile", "selected suite has no diagnostics")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(check.path),
        "suite": suite_name,
        "focus": focus_name,
        "profile": profile,
        "server_mode": server_mode,
        "sampling": deepcopy(raw["sampling"]),
        "calibration": deepcopy(raw["calibration"]),
        "models": [
            {
                "name": name,
                "checkpoint": raw["models"][name]["checkpoint"],
                "tokenizer": raw["models"][name].get(
                    "tokenizer", raw["models"][name]["checkpoint"]
                ),
                "revision": raw["models"][name].get("revision", _DEFAULT_REVISION),
                "cold_cache_allowance_tokens": raw["models"][name].get(
                    "cold_cache_allowance_tokens", 0
                ),
            }
            for name in coverage
        ],
        "runtime_configurations": [
            {"name": name, **deepcopy(raw["runtime_configurations"][name])}
            for name in runtime_names
        ],
        "prompts": {
            name: deepcopy(raw["prompts"][name])
            for name in dict.fromkeys(
                workload["prompt_group"] for workload in workloads
            )
        },
        "workloads": workloads,
        "coverage": coverage,
        "executions": executions,
        "diagnostics": [
            {"name": name, **deepcopy(raw["diagnostics"][name])}
            for name in diagnostic_names
        ],
    }
