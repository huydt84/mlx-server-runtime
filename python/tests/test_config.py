from __future__ import annotations

import pytest

from mlx_worker.config import load_config


def test_load_config_parses_max_vlm_images(monkeypatch) -> None:
    """Worker config reads VLM image cap from environment."""
    monkeypatch.setenv("MLX_RUNTIME_MAX_VLM_IMAGES", "7")
    config = load_config()
    assert config.max_vlm_images == 7


def test_load_config_defaults_backend_to_v1(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_BACKEND", raising=False)

    config = load_config()

    assert config.backend == "v1"


def test_load_config_parses_native_backend(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_BACKEND", "native-mlx")

    config = load_config()

    assert config.backend == "native-mlx"


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


def test_load_config_rejects_invalid_native_prefix_cache_strategy(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY", "none")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY"):
        load_config()


def test_load_config_rejects_invalid_native_kv_page_size(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE", "7")

    with pytest.raises(ValueError, match="MLX_RUNTIME_NATIVE_KV_PAGE_SIZE"):
        load_config()
