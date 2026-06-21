#!/usr/bin/env bash
# Usage examples (run from repo root):
#
# Phase 6 text-only fast-smoke:
#   bash scripts/benchmark.sh \
#     --model mlx-community/Llama-3.2-3B-Instruct-4bit \
#     --prompt "Say hello in one short sentence." \
#     --warmup-trials 0 \
#     --trials 1 \
#     --max-tokens 8 \
#     --report-path benchmarks/results/phase_6_smoke.md
#
# Phase 6 full-suite:
#   bash scripts/benchmark.sh \
#     --prompt-limit 0 \
#     --include-long-prompts \
#     --warmup-trials 1 \
#     --trials 1 \
#     --report-path benchmarks/results/phase_6_report.md
#
# Help:
#   bash scripts/benchmark.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/python"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec uv run python ../benchmarks/compare.py --help
fi

exec uv run python ../benchmarks/compare.py "$@"
