#!/usr/bin/env bash
#
# Native-v2 architecture-isolation and CPU-overhead gate.
#
# This is the aggregate near-zero-overhead gate.  It combines the
# hardware-independent architecture-count proof with the real public-gateway,
# MLX capture, and Metal System Trace gate in native_overlap.sh.
#
# Usage:
#   bash mlx-host-validation/scripts/native_architecture_overhead.sh [options]
#   bash mlx-host-validation/scripts/native_architecture_overhead.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Gate native-v2 architecture isolation and CPU overhead: lazy manifest counts plus the serial/overlap parity, latency, MLX, and Metal gates."
cli_option CHECKPOINT --checkpoint MLX_OVERHEAD_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served by the overlap gate."
cli_option PORT --port MLX_OVERHEAD_PORT port 18017 "Gateway port for the overlap gate."
cli_option WARMUPS --warmups MLX_OVERHEAD_WARMUPS int 5 "Warmup requests per mode before timing."
cli_option SAMPLES --samples MLX_OVERHEAD_SAMPLES int 10 "Timed requests per mode; an even number >= 2."
cli_option MAX_TOKENS --max-tokens MLX_OVERHEAD_MAX_TOKENS int 8 "Completion tokens per timed request."
cli_option MAX_REGRESSION --max-regression MLX_OVERHEAD_MAX_REGRESSION number 0.02 \
    "Largest allowed overlap latency regression, as a fraction of serial."
cli_option TRACE_DIR --artifacts-dir MLX_OVERHEAD_TRACE_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-architecture-overhead" "Directory for manifest results and overlap artifacts."
cli_parse "$@"
cli_require_free_port "$PORT" --port
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
MTL_CAPTURE_ENABLED=1 bash "$OVERLAP_SCRIPT" \
    --artifacts-dir "$OVERLAP_DIR" \
    --checkpoint "$CHECKPOINT" \
    --port "$PORT" \
    --warmups "$WARMUPS" \
    --samples "$SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --max-regression "$MAX_REGRESSION" \
    --metal-capture \
    --xctrace

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
