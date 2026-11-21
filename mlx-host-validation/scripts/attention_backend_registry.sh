#!/usr/bin/env bash
#
# Host-only validation for the native attention-backend registry.
# Run this on an Apple Silicon Mac with MLX Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/attention_backend_registry.sh [options]
#   bash mlx-host-validation/scripts/attention_backend_registry.sh --help
#
# Expected success signals:
#   - `attention_backend_config_tests_ok=1`
#   - `attention_backend_registry_ok=1`
#   - `attention_backend_capabilities_ok=1`
#   - `attention_backend_parity_ok=1`
#   - `attention_backend_decode_kernel_parity_ok=1`
#   - `attention_backend_hot_path_identity_ok=1`
#   - `attention_backend_validation_ok=1`
#
# Expected failure signals:
#   - non-arm64 host or unavailable MLX Metal
#   - unknown/default backend mismatch
#   - incompatible cache/reservation contract
#   - dense-reference parity failure
#   - registry adds a wrapper around the production attention hot path

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate the native attention-backend registry: contract tests, Metal capability and parity probes, and hot-path identity."
cli_option PYTEST_FILTER --pytest-filter MLX_ATTENTION_REGISTRY_PYTEST_FILTER string \
    "execution_backend or attention_capabilities or paged_metal_attention" \
    "pytest -k expression for the contract tests."
cli_option SKIP_CONTRACT_TESTS --skip-contract-tests MLX_ATTENTION_REGISTRY_SKIP_CONTRACT_TESTS bool 0 \
    "Skip the pytest contract tests and run only the Metal probes."
cli_parse "$@"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "attention_backend_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi

echo "[1/2] Run focused configuration and backend contract tests"
if [[ "$SKIP_CONTRACT_TESTS" == "1" ]]; then
    echo "attention_backend_config_tests_skipped=1"
else
    (
        cd "$PYTHON_DIR"
        uv run pytest tests/test_config.py tests/test_native_mlx.py -k "$PYTEST_FILTER" -q
    )
    echo "attention_backend_config_tests_ok=1"
fi

echo "[2/2] Run direct Metal capability and parity probes"
(
    cd "$PYTHON_DIR"
    uv run python - <<'PY'
import mlx.core as mx

from mlx_worker.native_mlx.attention import (
    DenseReferenceAttentionBackend,
    PagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.cache import (
    DenseKVCacheBackend,
    KVCacheGeometry,
    PagedKVCacheBackend,
)
from mlx_worker.native_mlx.execution_backends import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    available_native_execution_backends,
    build_native_execution_backend,
)
from mlx_worker.native_mlx.interfaces import ForwardMode

if not mx.metal.is_available():
    raise SystemExit("MLX Metal is unavailable")
if available_native_execution_backends() != (DEFAULT_NATIVE_EXECUTION_BACKEND,):
    raise SystemExit("default backend registry is inconsistent")

bundle = build_native_execution_backend(
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    KVCacheGeometry(1, 1, 4, mx.float16),
    page_size=8,
    cache_budget_bytes=256,
)
bundle.validate()
print("attention_backend_registry_ok=1")

capabilities = bundle.attention_backend.capabilities
if capabilities.supported_masks != frozenset(("causal", None)):
    raise SystemExit("production attention mask capability is inaccurate")
if not capabilities.consumes_page_tables_directly:
    raise SystemExit("paged decode kernel is not reported as reading page tables")
print("attention_backend_capabilities_ok=1")

dense_backend = DenseKVCacheBackend(num_layers=1)
dense_cache = dense_backend.get(dense_backend.create("dense"), "dense")
dense_reservation = dense_backend.reserve_batch((dense_cache,), (2,))
paged_backend = bundle.cache_backend
paged_cache = paged_backend.get(paged_backend.create("paged"), "paged")
paged_reservation = paged_backend.reserve_batch((paged_cache,), (2,))

queries = mx.array(
    [[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]]],
    dtype=mx.float16,
)
keys = mx.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]], dtype=mx.float16)
values = mx.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=mx.float16)
dense = DenseReferenceAttentionBackend().contexts(
    dense_reservation, ForwardMode.PREFILL
)[0].append_and_attend(queries, keys, values, scale=0.5, mask="causal")
paged = bundle.attention_backend.contexts(
    paged_reservation, ForwardMode.PREFILL
)[0].append_and_attend(queries, keys, values, scale=0.5, mask="causal")
mx.eval(dense, paged)
if not mx.allclose(dense, paged, atol=1e-3, rtol=1e-3).item():
    raise SystemExit("registered backend does not match dense reference")
print("attention_backend_parity_ok=1")

decode_geometry = KVCacheGeometry(1, 2, 64, mx.float16)
decode_bundle = build_native_execution_backend(
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    decode_geometry,
    page_size=8,
    cache_budget_bytes=1 << 20,
)
decode_dense = DenseKVCacheBackend(num_layers=1)
cache_lengths = (5, 17, 9)


def decode_step(cache_backend, contexts_for):
    caches = []
    for row, length in enumerate(cache_lengths):
        cache = cache_backend.get(cache_backend.create(f"row-{row}"), f"row-{row}")
        mx.random.seed(row)
        prefix = mx.random.normal((1, 2, length, 64)).astype(mx.float16)
        reservation = cache_backend.reserve_batch((cache,), (length,))
        (context,) = contexts_for(reservation, ForwardMode.PREFILL)
        mx.eval(
            context.append_and_attend(
                mx.random.normal((1, 4, length, 64)).astype(mx.float16),
                prefix,
                prefix,
                scale=0.125,
                mask="causal",
            )
        )
        reservation.commit()
        caches.append(cache)
    mx.random.seed(99)
    step_keys = mx.random.normal((3, 2, 1, 64)).astype(mx.float16)
    reservation = cache_backend.reserve_batch(tuple(caches), (1, 1, 1))
    (context,) = contexts_for(reservation, ForwardMode.DECODE)
    output = context.append_and_attend(
        mx.random.normal((3, 4, 1, 64)).astype(mx.float16),
        step_keys,
        step_keys,
        scale=0.125,
        mask="causal",
    )
    mx.eval(output)
    reservation.commit()
    return output


kernel_output = decode_step(
    decode_bundle.cache_backend, decode_bundle.attention_backend.contexts
)
reference_output = decode_step(
    decode_dense, DenseReferenceAttentionBackend().contexts
)
if not mx.allclose(kernel_output, reference_output, atol=2e-2).item():
    raise SystemExit("paged decode kernel does not match dense reference")
print("attention_backend_decode_kernel_parity_ok=1")

if type(bundle.attention_backend) is not PagedMetalAttentionBackend:
    raise SystemExit("registry inserted a production hot-path wrapper")
if type(bundle.cache_backend) is not PagedKVCacheBackend:
    raise SystemExit("registry inserted a cache hot-path wrapper")
print("attention_backend_hot_path_identity_ok=1")
PY
)

echo "attention_backend_validation_ok=1"
