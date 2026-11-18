"""Template arguments chosen for the paged-decode attention kernel."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.paged_decode import attention_template


def _arrays(query_dtype, key_dtype, value_dtype=None):
    queries = mx.zeros((2, 28, 1, 128), dtype=query_dtype)
    key_pages = mx.zeros((3, 16, 4, 128), dtype=key_dtype)
    value_pages = mx.zeros((3, 16, 4, 128), dtype=value_dtype or key_dtype)
    return queries, key_pages, value_pages


@pytest.mark.parametrize(
    ("query_dtype", "kv_dtype"),
    [
        (mx.float16, mx.float16),
        (mx.float16, mx.bfloat16),
        (mx.bfloat16, mx.float16),
        (mx.float32, mx.bfloat16),
    ],
)
def test_query_and_kv_dtypes_bind_separate_template_types(
    query_dtype, kv_dtype
) -> None:
    template = dict(
        attention_template(*_arrays(query_dtype, kv_dtype), simdgroups=8, splits=2)
    )

    assert template["Q"] == query_dtype
    assert template["KV"] == kv_dtype


def test_template_carries_launch_geometry_in_kernel_order() -> None:
    template = attention_template(
        *_arrays(mx.float16, mx.bfloat16), simdgroups=8, splits=4
    )

    assert template == [
        ("Q", mx.float16),
        ("KV", mx.bfloat16),
        ("QUERY_HEADS", 28),
        ("KV_HEADS", 4),
        ("HEAD_DIM", 128),
        ("PAGE_SIZE", 16),
        ("SIMDGROUPS", 8),
        ("SPLITS", 4),
    ]


def test_key_and_value_pools_of_different_dtypes_are_rejected() -> None:
    with pytest.raises(ValueError, match="one dtype"):
        attention_template(
            *_arrays(mx.float16, mx.bfloat16, mx.float16), simdgroups=8, splits=1
        )
