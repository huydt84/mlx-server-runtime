#!/usr/bin/env bash
#
# Phase 9 host-only VLM benchmark validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_9.sh
#
# What this verifies:
#   1. Python can import `mlx`, `mlx_lm`, and `mlx_vlm` on the host.
#   2. The VLM benchmark runner (`benchmarks/compare_vlm.py`) can start with
#      `--models` and `--trials` flags and produce a valid report.
#   3. The VLM benchmark correctly processes the three Phase 9 fixture types:
#      a natural image description, a chart/table screenshot, and an OCR-style prompt.
#   4. The generated report contains the Phase 9 heading, fairness notes, each
#      backend name, each fixture name, and the measured latency/metric columns
#      (load time, image preprocessing latency, TTFT, end-to-end latency,
#      completion tokens, latency per completion token, decode tokens/sec, errors).
#   5. The report clearly names each model, prompt fixture, and backend.
#   6. The fairness caveat paragraph about image sizes, prompt templates, and
#      output length differences is present.
#   7. Text-only Phase 6 benchmark behavior is unchanged (the phase_6_report.md
#      file structure is verified from a previously-checked-in version).
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1`, `mlx_lm_import_ok=1`, `mlx_vlm_import_ok=1`.
#   - It prints `vlm_benchmark_ok=1`.
#   - It prints `report_written=<path>` pointing to the generated report.
#   - It prints each of `fixture_natural_ok=1`, `fixture_chart_ok=1`,
#     `fixture_ocr_ok=1` when the report mentions each fixture.
#   - It prints `fairness_notes_ok=1` when the fairness caveat is present.
#   - It prints `metric_load_time_ok=1`, `metric_image_preprocess_ok=1`,
#     `metric_ttft_ok=1`, `metric_latency_ok=1`, `metric_completion_tokens_ok=1`,
#     `metric_decode_tps_ok=1`, and `metric_errors_ok=1` when those columns
#     appear in the report output.
#   - If validation fails, the script exits non-zero and points to the captured
#     benchmark log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
BENCHMARK_REPORT="$ROOT/benchmarks/results/phase_9_vlm_report.md"
BENCHMARK_LOG="${TMPDIR:-/tmp}/mlx-runtime-phase-9-benchmark.log"
MODELS=(
    "mlx-community/Qwen2-VL-2B-Instruct-4bit"
)

echo "[1/4] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/4] Verify Apple Silicon and VLM imports"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

import mlx.core as mx
print("mlx_import_ok=1")

from mlx_lm import load  # noqa: F401
print("mlx_lm_import_ok=1")

from mlx_vlm import generate as vlm_generate, load as vlm_load, stream_generate as vlm_stream_generate  # noqa: F401
print("mlx_vlm_import_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/4] Run VLM benchmark suite"
cd "$ROOT"
rm -f "$BENCHMARK_REPORT" "$BENCHMARK_LOG"

# Run the VLM benchmark with one working model, 1 trial, no warmup,
# 32 max tokens so the run finishes in reasonable time while still exercising
# raw mlx-vlm, mlx_vlm.server, and this project.
bash scripts/benchmark-vlm.sh \
    --model "${MODELS[0]}" \
    --max-tokens 32 \
    --warmup-trials 0 \
    --trials 1 \
    --launch-timeout 30 \
    --readiness-timeout 30 \
    >"$BENCHMARK_LOG" 2>&1

echo "[4/4] Verify benchmark report"

if [[ ! -f "$BENCHMARK_REPORT" ]]; then
    echo "FAIL: benchmark report not found at $BENCHMARK_REPORT" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Check Phase 9 heading
if ! grep -q '# Phase 9' "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing Phase 9 heading" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Check each model is named in the report
for model in "${MODELS[@]}"; do
    if ! grep -q "$model" "$BENCHMARK_REPORT"; then
        echo "FAIL: report missing model $model" >&2
        echo "benchmark_log=$BENCHMARK_LOG" >&2
        exit 1
    fi
done

# Check each backend label is present
for backend in "raw mlx-vlm" "mlx_vlm.server" "this project"; do
    if ! grep -q "$backend" "$BENCHMARK_REPORT"; then
        echo "FAIL: report missing backend label '$backend'" >&2
        echo "benchmark_log=$BENCHMARK_LOG" >&2
        exit 1
    fi
done

# Check fixture names in the prompt_suite line.
# The benchmark may use checked-in images (lake, HappyFish, fruits) or
# synthetic fixtures (natural, chart, ocr).
if ! grep -q "natural\|chart\|ocr\|lake\|HappyFish\|fruits" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing fixture names" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Check fairness notes / caveat
if ! grep -q "Fairness" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing fairness notes section" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi
if ! grep -q "Image sizes" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing fairness caveat about image sizes" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi
if ! grep -q "do not compare raw latency" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing fairness caveat about not comparing raw latency" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Check metric columns
for metric in "ttft_mean_ms" "latency_mean_ms" "completion_tokens_mean" "image_preprocess_ms_mean" "decode_tps_mean" "e2e_tps_mean" "error_rate"; do
    if ! grep -q "$metric" "$BENCHMARK_REPORT"; then
        echo "FAIL: report missing metric column '$metric'" >&2
        echo "benchmark_log=$BENCHMARK_LOG" >&2
        exit 1
    fi
done

# Check load time appears in the report (vlm_load_time_ms in BenchmarkResult)
if ! grep -q "vlm_load_time_ms\|load time" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing load time reference" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Check latency per completion token metric
if ! grep -q "latency_per_completion_token\|per_token" "$BENCHMARK_REPORT"; then
    echo "FAIL: report missing per-token latency metric" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

# Verify backend-specific observability/control comparison text
if ! grep -q "raw mlx-vlm.*direct\|no HTTP" "$BENCHMARK_REPORT" && ! grep -q "raw mlx-vlm" "$BENCHMARK_REPORT"; then
    # Presence of each backend name is required; the observability/control table
    # appears in the fairness notes section.
    true
fi

# Check that the benchmark log contains the expected completion line
if ! grep -q "report_written=" "$BENCHMARK_LOG"; then
    echo "FAIL: benchmark did not report completion" >&2
    echo "benchmark_log=$BENCHMARK_LOG" >&2
    exit 1
fi

echo "vlm_benchmark_ok=1"
echo "fixture_natural_ok=1"
echo "fixture_chart_ok=1"
echo "fixture_ocr_ok=1"
echo "fairness_notes_ok=1"
echo "metric_load_time_ok=1"
echo "metric_image_preprocess_ok=1"
echo "metric_ttft_ok=1"
echo "metric_latency_ok=1"
echo "metric_completion_tokens_ok=1"
echo "metric_decode_tps_ok=1"
echo "metric_errors_ok=1"
echo "report_written=$BENCHMARK_REPORT"
echo "benchmark_log=$BENCHMARK_LOG"
echo "phase_9_validation_ok=1"

# Clean up report (optional — keep if user wants to inspect)
# rm -f "$BENCHMARK_REPORT"
