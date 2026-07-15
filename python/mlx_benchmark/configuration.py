"""Load and select declarative MLX Air benchmark configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


_MAX_PROMPTS_PER_GROUP = 10_000
_MAX_PROMPT_TOKENS = 65_536
_MAX_OUTPUT_TOKENS = 4_096
_MAX_REQUESTS_PER_TRIAL = 10_000
_MAX_TRIALS = 100
_MAX_CONCURRENCY = 256
_PROMPT_KINDS = {
    "short",
    "medium",
    "long",
    "decode-promoting",
    "shared-prefix",
    "cache-pressure",
}
_PROMPT_MATERIALS = {"measured", "warmup"}
_CACHE_MODES = {"cold", "warm-prefix", "cache-pressure"}
_METRIC_UNITS = {"ms", "tokens/s"}
_CACHE_UNITS = {"tokens", "percent", "bytes"}


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

    Args:
        path: Required TOML configuration path.
        suite_name: Suite selected by ``--suite``.
        focus_name: Optional focus selected by ``--focus``.
        profile: Profiling selector recorded with the selected values.
        server_mode: Server ownership mode selected by the CLI.

    Returns:
        The expanded configuration values used by the run.

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

    _validate_configuration(path, raw)
    return SelectedConfiguration(
        source_path=path,
        values=_select_configuration(
            path,
            raw,
            suite_name=suite_name,
            focus_name=focus_name,
            profile=profile,
            server_mode=server_mode,
        ),
    )


def _validate_configuration(path: Path, raw: dict[str, Any]) -> None:
    schema_version = _integer(path, raw, "schema_version", "schema_version")
    if schema_version != 1:
        _fail(path, "schema_version", "must be 1")

    execution = _table(path, raw, "execution", "execution")
    server_modes = _string_list(
        path, execution, "server_modes", "execution.server_modes"
    )
    load_modes = _string_list(path, execution, "load_modes", "execution.load_modes")
    directions = _string_list(
        path, execution, "metric_directions", "execution.metric_directions"
    )
    units = _string_list(path, execution, "metric_units", "execution.metric_units")
    _exact_values(
        path,
        "execution.server_modes",
        server_modes,
        {"self-launched", "external"},
    )
    _exact_values(
        path,
        "execution.load_modes",
        load_modes,
        {"sequential", "burst", "closed-loop"},
    )
    _exact_values(path, "execution.metric_directions", directions, {"lower", "higher"})
    if not units or not set(units).issubset(_METRIC_UNITS):
        _fail(
            path,
            "execution.metric_units",
            f"contains an unsupported unit; supported units are {sorted(_METRIC_UNITS)!r}",
        )

    sampling = _table(path, raw, "sampling", "sampling")
    _integer(path, sampling, "seed", "sampling.seed")
    _number(path, sampling, "temperature", "sampling.temperature", minimum=0.0)
    _number(path, sampling, "top_p", "sampling.top_p", minimum=0.0, maximum=1.0)
    _bounded_integer(
        path,
        sampling,
        "request_timeout_seconds",
        "sampling.request_timeout_seconds",
        1,
        3_600,
    )
    _bounded_integer(
        path,
        sampling,
        "readiness_timeout_seconds",
        "sampling.readiness_timeout_seconds",
        1,
        3_600,
    )

    prompt_bank = _named_tables(path, raw, "prompt_bank")
    for name, prompt in prompt_bank.items():
        field = f"prompt_bank.{name}"
        kind = _string(path, prompt, "kind", f"{field}.kind")
        if kind not in _PROMPT_KINDS:
            _fail(path, f"{field}.kind", f"unsupported prompt kind {kind!r}")
        material = _string(path, prompt, "material", f"{field}.material")
        if material not in _PROMPT_MATERIALS:
            _fail(path, f"{field}.material", f"unsupported material {material!r}")
        count = _bounded_integer(
            path, prompt, "count", f"{field}.count", 1, _MAX_PROMPTS_PER_GROUP
        )
        _bounded_integer(
            path,
            prompt,
            "target_tokens",
            f"{field}.target_tokens",
            1,
            _MAX_PROMPT_TOKENS,
        )
        if kind == "shared-prefix":
            groups = _bounded_integer(
                path,
                prompt,
                "prefix_groups",
                f"{field}.prefix_groups",
                1,
                count,
            )
            suffixes = _bounded_integer(
                path,
                prompt,
                "suffixes_per_group",
                f"{field}.suffixes_per_group",
                1,
                count,
            )
            if groups * suffixes != count:
                _fail(
                    path,
                    f"{field}.count",
                    "must equal prefix_groups * suffixes_per_group",
                )

    cache_states = _named_tables(path, raw, "cache_states")
    for name, cache_state in cache_states.items():
        field = f"cache_states.{name}"
        mode = _string(path, cache_state, "mode", f"{field}.mode")
        if mode not in _CACHE_MODES:
            _fail(path, f"{field}.mode", f"unsupported cache mode {mode!r}")
        unit = _string(path, cache_state, "unit", f"{field}.unit")
        if unit not in _CACHE_UNITS:
            _fail(path, f"{field}.unit", f"unsupported cache unit {unit!r}")
        _number(path, cache_state, "value", f"{field}.value", minimum=0.0)

    models = _named_tables(path, raw, "models")
    for name, model in models.items():
        field = f"models.{name}"
        _string(path, model, "checkpoint", f"{field}.checkpoint")
        _string(path, model, "tokenizer", f"{field}.tokenizer")
        _string(path, model, "revision", f"{field}.revision")

    runtime_configurations = _named_tables(path, raw, "runtime_configurations")
    for name, runtime in runtime_configurations.items():
        field = f"runtime_configurations.{name}"
        backend = _string(path, runtime, "backend", f"{field}.backend")
        if backend not in {"v1", "native-mlx"}:
            _fail(path, f"{field}.backend", f"unsupported backend {backend!r}")
        environment = _table(path, runtime, "environment", f"{field}.environment")
        for key, value in environment.items():
            if not isinstance(key, str) or not key.startswith("MLX_RUNTIME_"):
                _fail(
                    path,
                    f"{field}.environment",
                    "keys must start with MLX_RUNTIME_",
                )
            if not isinstance(value, str):
                _fail(path, f"{field}.environment.{key}", "must be a string")

    warmup_groups = _named_tables(path, raw, "warmup_groups")
    for name, warmup in warmup_groups.items():
        field = f"warmup_groups.{name}"
        prompt_group = _reference(
            path, warmup, "prompt_group", f"{field}.prompt_group", prompt_bank
        )
        if prompt_bank[prompt_group]["material"] != "warmup":
            _fail(path, f"{field}.prompt_group", "must reference warmup material")
        _bounded_integer(
            path, warmup, "concurrency", f"{field}.concurrency", 1, _MAX_CONCURRENCY
        )
        _bounded_integer(
            path,
            warmup,
            "output_tokens",
            f"{field}.output_tokens",
            1,
            _MAX_OUTPUT_TOKENS,
        )

    workloads = _named_tables(path, raw, "workloads")
    for name, workload in workloads.items():
        field = f"workloads.{name}"
        prompt_group = _reference(
            path, workload, "prompt_group", f"{field}.prompt_group", prompt_bank
        )
        if prompt_bank[prompt_group]["material"] != "measured":
            _fail(path, f"{field}.prompt_group", "must reference measured material")
        _bounded_integer(
            path,
            workload,
            "output_tokens",
            f"{field}.output_tokens",
            1,
            _MAX_OUTPUT_TOKENS,
        )
        _boolean(path, workload, "streaming", f"{field}.streaming")
        load_mode = _string(path, workload, "load_mode", f"{field}.load_mode")
        if load_mode not in load_modes:
            _fail(
                path, f"{field}.load_mode", f"unsupported execution mode {load_mode!r}"
            )
        concurrency = _bounded_integer(
            path,
            workload,
            "concurrency",
            f"{field}.concurrency",
            1,
            _MAX_CONCURRENCY,
        )
        _reference(path, workload, "cache_state", f"{field}.cache_state", cache_states)
        _string(path, workload, "primary_metric", f"{field}.primary_metric")
        direction = _string(
            path, workload, "metric_direction", f"{field}.metric_direction"
        )
        if direction not in directions:
            _fail(
                path,
                f"{field}.metric_direction",
                f"unsupported direction {direction!r}",
            )
        unit = _string(path, workload, "metric_unit", f"{field}.metric_unit")
        if unit not in units:
            _fail(path, f"{field}.metric_unit", f"unsupported unit {unit!r}")
        _reference_list(
            path,
            workload,
            "runtime_configurations",
            f"{field}.runtime_configurations",
            runtime_configurations,
        )
        _bounded_integer(path, workload, "trials", f"{field}.trials", 1, _MAX_TRIALS)
        requests = _bounded_integer(
            path,
            workload,
            "requests_per_trial",
            f"{field}.requests_per_trial",
            1,
            _MAX_REQUESTS_PER_TRIAL,
        )
        if load_mode == "sequential" and concurrency != 1:
            _fail(path, f"{field}.concurrency", "must be 1 for sequential mode")
        if load_mode == "burst" and concurrency != requests:
            _fail(
                path,
                f"{field}.concurrency",
                "must equal requests_per_trial for burst mode",
            )
        if load_mode == "closed-loop" and concurrency > requests:
            _fail(
                path,
                f"{field}.concurrency",
                "must not exceed requests_per_trial for closed-loop mode",
            )

    focuses = _named_tables(path, raw, "focuses")
    for name, focus in focuses.items():
        field = f"focuses.{name}"
        _reference_list(path, focus, "models", f"{field}.models", models)
        _reference_list(path, focus, "workloads", f"{field}.workloads", workloads)

    configuration_orders = _named_tables(path, raw, "configuration_orders")
    for name, order in configuration_orders.items():
        field = f"configuration_orders.{name}"
        _reference(path, order, "model", f"{field}.model", models)
        _reference_list(
            path,
            order,
            "runtime_configurations",
            f"{field}.runtime_configurations",
            runtime_configurations,
        )

    tail_sets = _named_tables(path, raw, "tail_sets")
    for name, tail_set in tail_sets.items():
        _reference_list(
            path,
            tail_set,
            "workloads",
            f"tail_sets.{name}.workloads",
            workloads,
            allow_empty=True,
        )

    diagnostics = _named_tables(path, raw, "diagnostics")
    for name, diagnostic in diagnostics.items():
        field = f"diagnostics.{name}"
        _reference_list(path, diagnostic, "models", f"{field}.models", models)
        _reference_list(path, diagnostic, "workloads", f"{field}.workloads", workloads)

    suites = _named_tables(path, raw, "suites")
    for name, suite in suites.items():
        field = f"suites.{name}"
        _reference_list(path, suite, "models", f"{field}.models", models)
        _reference_list(path, suite, "workloads", f"{field}.workloads", workloads)
        _reference_list(
            path,
            suite,
            "runtime_configurations",
            f"{field}.runtime_configurations",
            runtime_configurations,
        )
        _reference_list(
            path,
            suite,
            "warmup_groups",
            f"{field}.warmup_groups",
            warmup_groups,
            allow_empty=True,
        )
        _reference_list(
            path,
            suite,
            "configuration_orders",
            f"{field}.configuration_orders",
            configuration_orders,
        )
        _reference_list(
            path,
            suite,
            "focuses",
            f"{field}.focuses",
            focuses,
            allow_empty=True,
        )
        _boolean(path, suite, "focus_required", f"{field}.focus_required")
        _bounded_integer(path, suite, "trials", f"{field}.trials", 1, _MAX_TRIALS)
        _bounded_integer(
            path,
            suite,
            "max_model_starts",
            f"{field}.max_model_starts",
            1,
            100,
        )
        _reference(path, suite, "tail_set", f"{field}.tail_set", tail_sets)
        _reference_list(
            path,
            suite,
            "diagnostic_families",
            f"{field}.diagnostic_families",
            diagnostics,
            allow_empty=True,
        )
        coverage = suite.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            _fail(path, f"{field}.coverage", "must be a non-empty array of tables")
        covered_models: set[str] = set()
        covered_workloads: set[str] = set()
        for index, entry in enumerate(coverage):
            entry_field = f"{field}.coverage[{index}]"
            if not isinstance(entry, dict):
                _fail(path, entry_field, "must be a table")
            model = _reference(path, entry, "model", f"{entry_field}.model", models)
            selected = _reference_list(
                path,
                entry,
                "workloads",
                f"{entry_field}.workloads",
                workloads,
            )
            if model not in suite["models"]:
                _fail(path, f"{entry_field}.model", "is not selected by the suite")
            outside_suite = set(selected).difference(suite["workloads"])
            if outside_suite:
                _fail(
                    path,
                    f"{entry_field}.workloads",
                    f"contains workloads outside the suite: {sorted(outside_suite)!r}",
                )
            covered_models.add(model)
            covered_workloads.update(selected)
        if covered_models != set(suite["models"]):
            _fail(path, f"{field}.coverage", "must cover every suite model")
        if covered_workloads != set(suite["workloads"]):
            _fail(path, f"{field}.coverage", "must cover every suite workload")


def _select_configuration(
    path: Path,
    raw: dict[str, Any],
    *,
    suite_name: str,
    focus_name: str | None,
    profile: str,
    server_mode: str,
) -> dict[str, Any]:
    execution = raw["execution"]
    if server_mode not in execution["server_modes"]:
        _fail(path, "execution.server_modes", f"does not support {server_mode!r}")
    suites = raw["suites"]
    if suite_name not in suites:
        _fail(path, "--suite", f"unknown suite {suite_name!r}")
    suite = suites[suite_name]
    allowed_focuses = suite["focuses"]
    if focus_name is None and suite["focus_required"]:
        _fail(path, f"suites.{suite_name}.focus_required", "--focus is required")
    if focus_name is not None and focus_name not in allowed_focuses:
        _fail(
            path,
            "--focus",
            f"focus {focus_name!r} is not available in suite {suite_name!r}",
        )

    selected_models = list(suite["models"])
    selected_workloads = list(suite["workloads"])
    if focus_name is not None:
        focus = raw["focuses"][focus_name]
        selected_models = _ordered_intersection(selected_models, focus["models"])
        selected_workloads = _ordered_intersection(
            selected_workloads, focus["workloads"]
        )
    if not selected_models:
        _fail(path, "--focus", "selection contains no models")
    if not selected_workloads:
        _fail(path, "--focus", "selection contains no workloads")

    selected_runtime_names = list(suite["runtime_configurations"])
    selected_workload_values: list[dict[str, Any]] = []
    prompt_names: list[str] = []
    cache_names: list[str] = []
    for name in selected_workloads:
        workload = deepcopy(raw["workloads"][name])
        runtimes = _ordered_intersection(
            selected_runtime_names, workload["runtime_configurations"]
        )
        if not runtimes:
            _fail(
                path,
                f"workloads.{name}.runtime_configurations",
                f"does not overlap suite {suite_name!r}",
            )
        workload["name"] = name
        workload["runtime_configurations"] = runtimes
        workload["trials"] = suite["trials"]
        selected_workload_values.append(workload)
        prompt_names.append(workload["prompt_group"])
        cache_names.append(workload["cache_state"])

    warmup_names = list(suite["warmup_groups"])
    prompt_names.extend(
        raw["warmup_groups"][name]["prompt_group"] for name in warmup_names
    )
    tail_name = suite["tail_set"]
    diagnostic_names = list(suite["diagnostic_families"])
    if focus_name not in {None, "all"} and focus_name in diagnostic_names:
        diagnostic_names = [focus_name]
    order_names = [
        name
        for name in suite["configuration_orders"]
        if raw["configuration_orders"][name]["model"] in selected_models
    ]
    selected_coverage = []
    for entry in suite["coverage"]:
        if entry["model"] not in selected_models:
            continue
        workload_names = _ordered_intersection(selected_workloads, entry["workloads"])
        if workload_names:
            selected_coverage.append(
                {"model": entry["model"], "workloads": workload_names}
            )

    return {
        "schema_version": raw["schema_version"],
        "source_path": str(path),
        "suite": suite_name,
        "focus": focus_name,
        "profile": profile,
        "server_mode": server_mode,
        "execution": deepcopy(execution),
        "sampling": deepcopy(raw["sampling"]),
        "models": [
            {"name": name, **deepcopy(raw["models"][name])} for name in selected_models
        ],
        "runtime_configurations": [
            {"name": name, **deepcopy(raw["runtime_configurations"][name])}
            for name in selected_runtime_names
        ],
        "prompt_bank": {
            name: deepcopy(raw["prompt_bank"][name])
            for name in dict.fromkeys(prompt_names)
        },
        "cache_states": {
            name: deepcopy(raw["cache_states"][name])
            for name in dict.fromkeys(cache_names)
        },
        "warmup_groups": [
            {"name": name, **deepcopy(raw["warmup_groups"][name])}
            for name in warmup_names
        ],
        "configuration_orders": [
            {"name": name, **deepcopy(raw["configuration_orders"][name])}
            for name in order_names
        ],
        "tail_selection": {
            "name": tail_name,
            **deepcopy(raw["tail_sets"][tail_name]),
        },
        "diagnostic_routing": [
            {"name": name, **deepcopy(raw["diagnostics"][name])}
            for name in diagnostic_names
        ],
        "cost_limits": {"max_model_starts": suite["max_model_starts"]},
        "workloads": selected_workload_values,
        "coverage": selected_coverage,
    }


def _table(path: Path, values: dict[str, Any], key: str, field: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        _fail(path, field, "must be a table")
    return value


def _named_tables(
    path: Path, raw: dict[str, Any], key: str
) -> dict[str, dict[str, Any]]:
    values = _table(path, raw, key, key)
    if not values:
        _fail(path, key, "must contain at least one named table")
    for name, value in values.items():
        if not isinstance(value, dict):
            _fail(path, f"{key}.{name}", "must be a table")
    return values


def _string(path: Path, values: dict[str, Any], key: str, field: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        _fail(path, field, "must be a non-empty string")
    return value


def _integer(path: Path, values: dict[str, Any], key: str, field: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, field, "must be an integer")
    return value


def _bounded_integer(
    path: Path,
    values: dict[str, Any],
    key: str,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    value = _integer(path, values, key, field)
    if not minimum <= value <= maximum:
        _fail(path, field, f"must be between {minimum} and {maximum}")
    return value


def _number(
    path: Path,
    values: dict[str, Any],
    key: str,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, field, "must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(path, field, f"must be at least {minimum}")
    if maximum is not None and number > maximum:
        _fail(path, field, f"must be at most {maximum}")
    return number


def _boolean(path: Path, values: dict[str, Any], key: str, field: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        _fail(path, field, "must be a boolean")
    return value


def _string_list(
    path: Path,
    values: dict[str, Any],
    key: str,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = values.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        _fail(path, field, "must be an array of non-empty strings")
    if not allow_empty and not value:
        _fail(path, field, "must not be empty")
    if len(value) != len(set(value)):
        _fail(path, field, "must not contain duplicates")
    return value


def _reference(
    path: Path,
    values: dict[str, Any],
    key: str,
    field: str,
    choices: dict[str, Any],
) -> str:
    value = _string(path, values, key, field)
    if value not in choices:
        _fail(path, field, f"unknown reference {value!r}")
    return value


def _reference_list(
    path: Path,
    values: dict[str, Any],
    key: str,
    field: str,
    choices: dict[str, Any],
    *,
    allow_empty: bool = False,
) -> list[str]:
    selected = _string_list(path, values, key, field, allow_empty=allow_empty)
    for value in selected:
        if value not in choices:
            _fail(path, field, f"unknown reference {value!r}")
    return selected


def _exact_values(
    path: Path, field: str, actual: list[str], expected: set[str]
) -> None:
    if set(actual) != expected:
        _fail(path, field, f"must contain exactly {sorted(expected)!r}")


def _ordered_intersection(left: list[str], right: list[str]) -> list[str]:
    right_values = set(right)
    return [value for value in left if value in right_values]


def _fail(path: Path, field: str, message: str) -> None:
    raise ConfigurationError(path, field, message)
