from __future__ import annotations

import pytest

from mlx_worker.config import load_config


def test_load_config_parses_max_vlm_images(monkeypatch) -> None:
    """Worker config reads VLM image cap from environment."""
    monkeypatch.setenv("MLX_RUNTIME_MAX_VLM_IMAGES", "7")
    config = load_config()
    assert config.max_vlm_images == 7


def test_load_config_defaults_backend_to_native(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_BACKEND", raising=False)

    config = load_config()

    assert config.backend == "native-mlx"


def test_load_config_parses_the_tool_call_parser(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_TOOL_CALL_PARSER", raising=False)
    assert load_config().tool_call_parser == "auto"

    for setting in ("", "auto", "none", "json_tools", "inkling2"):
        monkeypatch.setenv("MLX_RUNTIME_TOOL_CALL_PARSER", setting)
        assert load_config().tool_call_parser == (setting or "auto")


def test_load_config_rejects_an_unknown_tool_call_parser(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_TOOL_CALL_PARSER", "mistral")

    with pytest.raises(ValueError, match="MLX_RUNTIME_TOOL_CALL_PARSER.*json_tools"):
        load_config()


def test_load_config_parses_v1_backend(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_BACKEND", "v1")

    config = load_config()

    assert config.backend == "v1"


def test_load_config_parses_continuous_batching_controls(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_CONTINUOUS_BATCHING", "true")
    monkeypatch.setenv("MLX_RUNTIME_PROMPT_CONCURRENCY", "6")
    monkeypatch.setenv("MLX_RUNTIME_DECODE_CONCURRENCY", "3")
    monkeypatch.setenv("MLX_RUNTIME_PREFILL_CHUNK_SIZE", "128")

    config = load_config()

    assert config.continuous_batching is True
    assert config.prompt_concurrency == 6
    assert config.decode_concurrency == 3
    assert config.prefill_chunk_size == 128
    assert config.text_prompt_concurrency == 6
    assert config.text_decode_concurrency == 3
    assert config.text_prefill_chunk_size == 128
    assert config.vlm_prompt_concurrency == 6
    assert config.vlm_decode_concurrency == 3
    assert config.vlm_prefill_chunk_size == 128


def test_load_config_prefers_backend_specific_controls(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_PROMPT_CONCURRENCY", "6")
    monkeypatch.setenv("MLX_RUNTIME_TEXT_PROMPT_CONCURRENCY", "2")
    monkeypatch.setenv("MLX_RUNTIME_VLM_PROMPT_CONCURRENCY", "5")
    monkeypatch.setenv("MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES", "1024")
    monkeypatch.setenv("MLX_RUNTIME_VLM_APC_CACHE_BUDGET_BYTES", "2048")
    monkeypatch.setenv("MLX_RUNTIME_VISION_FEATURE_CACHE_MAX_ENTRIES", "7")

    config = load_config()

    assert config.prompt_concurrency == 6
    assert config.text_prompt_concurrency == 2
    assert config.vlm_prompt_concurrency == 5
    assert config.text_cache_budget_bytes == 1024
    assert config.vlm_apc_cache_budget_bytes == 2048
    assert config.vision_feature_cache_max_entries == 7


def test_load_config_parses_native_kv_page_size(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE", "32")

    config = load_config()

    assert config.native_kv_page_size == 32


def test_load_config_defaults_native_execution_backend(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_EXECUTION_BACKEND", raising=False)

    config = load_config()

    assert config.native_execution_backend == "native-metal-paged-sdpa"


def test_load_config_parses_native_execution_backend(monkeypatch) -> None:
    monkeypatch.setenv(
        "MLX_RUNTIME_NATIVE_EXECUTION_BACKEND", "native-metal-paged-sdpa"
    )

    config = load_config()

    assert config.native_execution_backend == "native-metal-paged-sdpa"


def test_load_config_defaults_native_execution_mode_to_overlap(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_EXECUTION_MODE", raising=False)

    config = load_config()

    assert config.native_execution_mode == "overlap"


def test_load_config_parses_native_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_EXECUTION_MODE", "serial")

    config = load_config()

    assert config.native_execution_mode == "serial"


def test_load_config_rejects_invalid_native_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_EXECUTION_MODE", "threaded")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_EXECUTION_MODE"):
        load_config()


def test_load_config_defaults_native_mlx_cache_limit_to_one_gib(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES", raising=False)

    config = load_config()

    assert config.native_mlx_cache_limit_bytes == 1024**3


def test_load_config_parses_native_mlx_cache_limit(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES", "0")

    config = load_config()

    assert config.native_mlx_cache_limit_bytes == 0


def test_load_config_rejects_negative_native_mlx_cache_limit(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES", "-1")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES"):
        load_config()


def test_load_config_parses_native_prefix_cache_strategy(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY", "radix")

    config = load_config()

    assert config.native_prefix_cache_strategy == "radix"


def test_load_config_defaults_native_prefix_cache_strategy_to_radix(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY", raising=False)

    config = load_config()

    assert config.native_prefix_cache_strategy == "radix"


def test_load_config_parses_native_scheduling_policy(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_SCHEDULING_POLICY", "lpm")

    config = load_config()

    assert config.native_scheduling_policy == "lpm"


def test_load_config_defaults_native_scheduling_policy_to_fcfs(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_SCHEDULING_POLICY", raising=False)

    config = load_config()

    assert config.native_scheduling_policy == "fcfs"


def test_load_config_rejects_invalid_native_scheduling_policy(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_SCHEDULING_POLICY", "random")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_SCHEDULING_POLICY"):
        load_config()


def test_load_config_rejects_invalid_native_prefix_cache_strategy(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY", "none")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY"):
        load_config()


def test_load_config_rejects_invalid_native_kv_page_size(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE", "7")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_KV_PAGE_SIZE"):
        load_config()


def test_load_config_rejects_invalid_native_execution_backend(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_EXECUTION_BACKEND", "unknown")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_EXECUTION_BACKEND"):
        load_config()


def test_load_config_defaults_pipeline_profiling_off(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", raising=False)
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR", raising=False)

    config = load_config()

    assert config.native_pipeline_profile is False
    assert config.native_pipeline_profile_dir is None


def test_load_config_requires_pipeline_profile_directory(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "1")
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR", raising=False)

    with pytest.raises(ValueError, match="PIPELINE_PROFILE_DIR"):
        load_config()


_MEMORY_ENV = (
    "MLX_RUNTIME_MEMORY_UTILIZATION",
    "MLX_RUNTIME_KV_CACHE_BYTES",
    "MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES",
    "MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST",
    "MLX_RUNTIME_MAX_TOTAL_TOKENS_EXPLICIT",
)


def test_load_config_defaults_memory_sizing(monkeypatch) -> None:
    for name in _MEMORY_ENV:
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.memory_utilization == 0.5
    assert config.kv_cache_bytes is None
    assert config.max_total_tokens_per_request == 65_536
    assert config.max_total_tokens_explicit is False


def test_load_config_parses_memory_sizing(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_MEMORY_UTILIZATION", "0.8")
    monkeypatch.setenv("MLX_RUNTIME_KV_CACHE_BYTES", "1073741824")
    monkeypatch.setenv("MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES", "4096")
    monkeypatch.setenv("MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST", "8192")
    monkeypatch.setenv("MLX_RUNTIME_MAX_TOTAL_TOKENS_EXPLICIT", "1")

    config = load_config()

    assert config.memory_utilization == 0.8
    assert config.kv_cache_bytes == 1 << 30
    assert config.max_total_tokens_per_request == 8192
    assert config.max_total_tokens_explicit is True


def test_load_config_accepts_legacy_kv_budget_variable(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_KV_CACHE_BYTES", raising=False)
    monkeypatch.setenv("MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES", "268435456")

    assert load_config().kv_cache_bytes == 268_435_456


def test_legacy_zero_budget_disables_v1_cache_without_pinning_native_pool(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MLX_RUNTIME_KV_CACHE_BYTES", raising=False)
    monkeypatch.setenv("MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES", "0")

    config = load_config()

    assert config.text_cache_budget_bytes == 0
    assert config.kv_cache_bytes is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLX_RUNTIME_MEMORY_UTILIZATION", "0"),
        ("MLX_RUNTIME_MEMORY_UTILIZATION", "1.2"),
        ("MLX_RUNTIME_KV_CACHE_BYTES", "-1"),
        ("MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST", "0"),
    ],
)
def test_load_config_rejects_invalid_memory_sizing(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_config()
