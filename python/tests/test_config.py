from __future__ import annotations

from mlx_worker.config import load_config


def test_load_config_parses_max_vlm_images(monkeypatch) -> None:
    """Worker config reads VLM image cap from environment."""
    monkeypatch.setenv("MLX_RUNTIME_MAX_VLM_IMAGES", "7")
    config = load_config()
    assert config.max_vlm_images == 7
