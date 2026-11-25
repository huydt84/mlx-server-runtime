"""Loading ``.metal`` kernel files and fusing gated MLP activations."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx import nn

from mlx_worker.native_mlx import metal_source
from mlx_worker.native_mlx.activations import geglu_approx, swiglu


@pytest.fixture
def kernel_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(metal_source, "KERNEL_DIR", tmp_path.resolve())
    return tmp_path


def test_local_includes_are_inlined_once_with_line_markers(kernel_dir) -> None:
    (kernel_dir / "common.metal").write_text(
        "#pragma once\ninline float twice(float x) { return 2 * x; }\n"
    )
    (kernel_dir / "a.metal").write_text('#include "common.metal"\nfloat a();\n')
    (kernel_dir / "main.metal").write_text(
        '#include <metal_stdlib>\n#include "common.metal"\n'
        '#include "a.metal"\nfloat main_fn();\n'
    )

    source = "".join(metal_source._expand(kernel_dir / "main.metal", set()))

    assert source.count("inline float twice") == 1
    assert "#pragma once" not in source
    assert "#include <metal_stdlib>" in source
    assert source.splitlines() == [
        '#line 1 "main.metal"',
        "#include <metal_stdlib>",
        '#line 1 "common.metal"',
        "",
        "inline float twice(float x) { return 2 * x; }",
        '#line 3 "main.metal"',
        '#line 1 "a.metal"',
        '#line 2 "a.metal"',
        "float a();",
        '#line 4 "main.metal"',
        "float main_fn();",
    ]


def test_include_outside_kernel_directory_is_rejected(kernel_dir) -> None:
    (kernel_dir / "main.metal").write_text('#include "../outside.metal"\n')
    (kernel_dir.parent / "outside.metal").write_text("float outside();\n")

    with pytest.raises(ValueError, match="escapes"):
        metal_source._expand(kernel_dir / "main.metal", set())


def test_missing_include_fails() -> None:
    with pytest.raises(FileNotFoundError):
        metal_source._expand(metal_source.KERNEL_DIR / "missing.metal", set())


def test_shipped_kernels_resolve() -> None:
    source = metal_source.metal_header("paged_decode.metal")

    assert '#line 1 "online_softmax.metal"' in source
    assert "paged_decode_attention" in source
    assert '#include "' not in source


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("shape", [(1, 1, 64), (3, 5, 64)])
def test_fused_activations_match_unfused(dtype, shape) -> None:
    mx.random.seed(0)
    gate = (mx.random.normal(shape) * 4).astype(dtype)
    up = mx.random.normal(shape).astype(dtype)

    assert mx.array_equal(swiglu(gate, up), nn.silu(gate) * up).item()
    assert mx.array_equal(geglu_approx(gate, up), nn.gelu_approx(gate) * up).item()
