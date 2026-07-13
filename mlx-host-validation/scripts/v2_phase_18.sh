#!/usr/bin/env bash
#
# Native-v2 Phase 18 architecture-isolation and CPU-overhead gate.
#
# This is the aggregate near-zero-overhead gate.  It combines the
# hardware-independent architecture-count proof with the real public-gateway,
# MLX capture, and Metal System Trace gate owned by Phase 17.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACE_DIR="${MLX_PHASE18_TRACE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-v2-phase-18}"
ARCH_HELPER="$ROOT/mlx-host-validation/scripts/python/phase18_architecture_overhead.py"
PHASE17_SCRIPT="$ROOT/mlx-host-validation/scripts/v2_phase_17.sh"
PHASE17_DIR="$TRACE_DIR/phase17"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "phase18_host_error=Apple Silicon arm64 is required for the full gate" >&2
    exit 1
fi

mkdir -p "$TRACE_DIR" "$PHASE17_DIR"
echo "[1/3] Measure one versus 101 lazy architecture manifests"
PYTHONPATH="$ROOT/python" uv --directory "$ROOT/python" run python "$ARCH_HELPER" \
    --output "$TRACE_DIR/architecture-overhead.json" \
    --output-markdown "$TRACE_DIR/architecture-overhead.md"

echo "[2/3] Run public-gateway parity, performance, MLX, and Metal gates"
MTL_CAPTURE_ENABLED=1 \
MLX_PHASE17_TRACE_DIR="$PHASE17_DIR" \
MLX_PHASE17_CHECKPOINT="${MLX_PHASE18_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}" \
MLX_PHASE17_WARMUPS="${MLX_PHASE18_WARMUPS:-5}" \
MLX_PHASE17_SAMPLES="${MLX_PHASE18_SAMPLES:-10}" \
MLX_PHASE17_MAX_TOKENS="${MLX_PHASE18_MAX_TOKENS:-8}" \
MLX_PHASE17_MAX_REGRESSION="${MLX_PHASE18_MAX_REGRESSION:-0.02}" \
MLX_PHASE17_METAL_CAPTURE=1 \
MLX_PHASE17_XCTRACE=1 \
bash "$PHASE17_SCRIPT"

echo "[3/3] Verify durable artifacts"
test -s "$TRACE_DIR/architecture-overhead.json"
test -s "$TRACE_DIR/architecture-overhead.md"
test -s "$PHASE17_DIR/comparison.json"
test -s "$PHASE17_DIR/phase17-report.md"
test -s "$PHASE17_DIR/timeline-analysis.json"
test -s "$PHASE17_DIR/serial-metal-analysis.json"
test -s "$PHASE17_DIR/metal-analysis.json"
test -s "$PHASE17_DIR/metal-comparison.json"
test -s "$PHASE17_DIR/overlap-profile/pipeline.gputrace"
echo "phase18_validation_artifacts=$TRACE_DIR"
echo "phase18_validation_ok=1"
