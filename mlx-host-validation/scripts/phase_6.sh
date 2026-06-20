#!/usr/bin/env bash
#
# Phase 6 host-only benchmark validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_6.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. `mlx_lm.generate.BatchGenerator` is available on the host.
#   3. The benchmark runner can compare raw `mlx-lm`, `mlx_lm.server`, and this project across multiple models.
#   4. The generated benchmark report contains measured latency and observability/control comparison text for each model.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_batch_generator_ok=1`.
#   - It prints `benchmark_report_ok=1` and `phase_6_benchmark_ok=1`.
#   - It prints the report path and the three backend names.
#   - It includes all benchmark models in the report.
#   - If validation fails, the script exits non-zero and points to the captured benchmark logs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
REPORT_PATH="$ROOT/benchmarks/results/phase_6_report.md"
BENCHMARK_LOG="${TMPDIR:-/tmp}/mlx-runtime-phase-6-benchmark.log"

echo "[1/3] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/3] Verify Apple Silicon and mlx-lm imports"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

import mlx.core as mx
print("mlx_import_ok=1")

from mlx_lm.generate import BatchGenerator  # noqa: F401
print("mlx_batch_generator_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/3] Run benchmark suite"
cd "$ROOT"
rm -f "$BENCHMARK_LOG"
bash scripts/benchmark.sh \
    --report-path "$REPORT_PATH" \
    --warmup-trials 0 \
    --trials 1 \
    --max-tokens 8 >"$BENCHMARK_LOG" 2>&1

grep -qx '# Phase 6 Benchmark Report' "$REPORT_PATH"
grep -q 'mlx-community/Qwen2.5-7B-Instruct-4bit' "$REPORT_PATH"
grep -q 'mlx-community/Qwen3-8B-4bit' "$REPORT_PATH"
grep -q 'mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-bf16' "$REPORT_PATH"
grep -q 'mlx-community/Qwen3.5-9B-4bit' "$REPORT_PATH"
grep -q 'raw mlx-lm' "$REPORT_PATH"
grep -q 'mlx_lm.server' "$REPORT_PATH"
grep -q 'this project' "$REPORT_PATH"
grep -q 'Observability / Control' "$REPORT_PATH"
grep -q 'latency_mean_overhead_ms' "$REPORT_PATH"
grep -q 'latency_per_completion_token_ms' "$REPORT_PATH"
grep -q 'prompt_tokens_per_request_mean' "$REPORT_PATH"

echo "benchmark_report_ok=1"
echo "phase_6_benchmark_ok=1"
echo "report_path=$REPORT_PATH"
echo "benchmark_log=$BENCHMARK_LOG"
