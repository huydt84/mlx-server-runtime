from __future__ import annotations

from mlx_worker.config import load_config


def test_load_config_parses_max_vlm_images(monkeypatch) -> None:
    """Worker config reads VLM image cap from environment."""
    monkeypatch.setenv("MLX_RUNTIME_MAX_VLM_IMAGES", "7")
    config = load_config()
    assert config.max_vlm_images == 7


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
