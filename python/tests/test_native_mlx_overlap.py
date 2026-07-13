from __future__ import annotations

import pytest
import mlx.core as mx

from mlx_worker.native_mlx.overlap import probe_cross_stream_dependency


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_cross_stream_probe_orders_dependency() -> None:
    result = probe_cross_stream_dependency()

    assert result.supported is True
    assert "native model/cache evidence" in result.detail
