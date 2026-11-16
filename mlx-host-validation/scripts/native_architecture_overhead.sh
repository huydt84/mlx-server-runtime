#!/usr/bin/env bash
#
# Native-v2 architecture-isolation and CPU-overhead gate.
#
# This is the aggregate near-zero-overhead gate.  It combines the
# hardware-independent architecture-count proof with the real public-gateway,
# MLX capture, and Metal System Trace gate in native_overlap.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACE_DIR="${MLX_OVERHEAD_TRACE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-v2-architecture-overhead}"
ARCH_HELPER="$ROOT/mlx-host-validation/scripts/python/architecture_overhead.py"
OVERLAP_SCRIPT="$ROOT/mlx-host-validation/scripts/native_overlap.sh"
OVERLAP_DIR="$TRACE_DIR/overlap"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "overhead_host_error=Apple Silicon arm64 is required for the full gate" >&2
    exit 1
fi

mkdir -p "$TRACE_DIR" "$OVERLAP_DIR"
echo "[1/3] Measure one versus 101 lazy architecture manifests"
PYTHONPATH="$ROOT/python" uv --directory "$ROOT/python" run python "$ARCH_HELPER" \
    --output "$TRACE_DIR/architecture-overhead.json" \
    --output-markdown "$TRACE_DIR/architecture-overhead.md"

echo "[2/3] Run public-gateway parity, performance, MLX, and Metal gates"
MTL_CAPTURE_ENABLED=1 \
MLX_OVERLAP_TRACE_DIR="$OVERLAP_DIR" \
MLX_OVERLAP_CHECKPOINT="${MLX_OVERHEAD_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}" \
MLX_OVERLAP_WARMUPS="${MLX_OVERHEAD_WARMUPS:-5}" \
MLX_OVERLAP_SAMPLES="${MLX_OVERHEAD_SAMPLES:-10}" \
MLX_OVERLAP_MAX_TOKENS="${MLX_OVERHEAD_MAX_TOKENS:-8}" \
MLX_OVERLAP_MAX_REGRESSION="${MLX_OVERHEAD_MAX_REGRESSION:-0.02}" \
MLX_OVERLAP_METAL_CAPTURE=1 \
MLX_OVERLAP_XCTRACE=1 \
bash "$OVERLAP_SCRIPT"

echo "[3/3] Verify durable artifacts"
test -s "$TRACE_DIR/architecture-overhead.json"
test -s "$TRACE_DIR/architecture-overhead.md"
test -s "$OVERLAP_DIR/comparison.json"
test -s "$OVERLAP_DIR/overlap-report.md"
test -s "$OVERLAP_DIR/timeline-analysis.json"
test -s "$OVERLAP_DIR/serial-metal-analysis.json"
test -s "$OVERLAP_DIR/metal-analysis.json"
test -s "$OVERLAP_DIR/metal-comparison.json"
test -s "$OVERLAP_DIR/overlap-profile/pipeline.gputrace"
echo "overhead_validation_artifacts=$TRACE_DIR"
echo "overhead_validation_ok=1"
