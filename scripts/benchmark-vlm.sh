#!/usr/bin/env bash
# Usage examples (run from repo root):
#
# `--model` is optional. If omitted, benchmark uses built-in default VLM model list.
# Add `--model ...` when you want to pin a specific model.
#
# Fast-smoke:
#   bash scripts/benchmark-vlm.sh \
#     --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
#     --benchmark-mode smoke \
#     --scenario baseline \
#     --fixture-category single_image \
#     --backend-order raw,server,project \
#     --output-json benchmarks/results/phase_9_vlm_smoke.json \
#     --output-md benchmarks/results/phase_9_vlm_smoke.md
#
# Full-suite:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode stable \
#     --scenario all \
#     --backend-order raw,server,project \
#     --randomize-backend-order \
#     --backend-order-seed 42 \
#     --order-rounds 3 \
#     --concurrency-levels 1,2,4 \
#     --measured-runs-per-fixture 7 \
#     --output-json benchmarks/results/phase_9_vlm_full.json \
#     --output-md benchmarks/results/phase_9_vlm_full.md
#
# Debug / smaller variants:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode smoke \
#     --scenario all \
#     --concurrency-levels 1,2,4 \
#     --randomize-backend-order \
#     --backend-order-seed 42 \
#     --output-json benchmarks/results/phase_9_vlm_all_smoke_debug.json \
#     --output-md benchmarks/results/phase_9_vlm_all_smoke_debug.md
#
# Stable baseline with token-equivalent HTTP comparison:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode stable \
#     --scenario baseline \
#     --backend-order raw,server,project \
#     --output-json benchmarks/results/phase_9_vlm_baseline_stable.json \
#     --output-md benchmarks/results/phase_9_vlm_baseline_stable.md
#
# Stable baseline with rotated order rounds:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode stable \
#     --scenario baseline \
#     --backend-order raw,server,project \
#     --order-rounds 3 \
#     --output-json benchmarks/results/phase_9_vlm_baseline_stable_order3.json \
#     --output-md benchmarks/results/phase_9_vlm_baseline_stable_order3.md
#
# Stable streaming with enough samples for headline-eligible HTTP compare:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode stable \
#     --scenario streaming \
#     --backend-order raw,server,project \
#     --measured-runs-per-fixture 7 \
#     --output-json benchmarks/results/phase_9_vlm_streaming_stable21.json \
#     --output-md benchmarks/results/phase_9_vlm_streaming_stable21.md
#
# Stable streaming with rotated order rounds:
#   bash scripts/benchmark-vlm.sh \
#     --benchmark-mode stable \
#     --scenario streaming \
#     --backend-order raw,server,project \
#     --measured-runs-per-fixture 7 \
#     --order-rounds 3 \
#     --output-json benchmarks/results/phase_9_vlm_streaming_stable21_order3.json \
#     --output-md benchmarks/results/phase_9_vlm_streaming_stable21_order3.md
#
# Single fixture category:
#   bash scripts/benchmark-vlm.sh \
#     --fixture-category single_image \
#     --benchmark-mode normal \
#     --scenario baseline
#
# Pin one model explicitly:
#   bash scripts/benchmark-vlm.sh \
#     --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
#     --benchmark-mode stable \
#     --scenario baseline
#
# Help:
#   bash scripts/benchmark-vlm.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/python"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec uv run python ../benchmarks/compare_vlm.py --help
fi

exec uv run python ../benchmarks/compare_vlm.py "$@"
