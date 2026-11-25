#!/usr/bin/env bash
#
# native-v2 packed-checkpoint host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# LookAtMyQuant packs have no mlx_lm reference, so this script checks that the
# native graph agrees with itself across step shapes, answers a question with a
# known answer, bounds the sliding-window pages, and rejects other layouts. It
# also requires a speculative run to return the tokens of a plain run.
#
# Usage:
#   bash mlx-host-validation/scripts/native_packed_checkpoints.sh [options]
#   bash mlx-host-validation/scripts/native_packed_checkpoints.sh --help
#
# Host requirements:
#   - Apple Silicon (`arm64`) with a Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - checkpoint and drafter already in the local Hugging Face cache, or local
#     directories
#   - free memory for the weights plus 4 GiB; on a 24 GB Mac stop other GPU work
#     before validating the 12.8 GB pack
#
# Expected success signals:
#   - `step_shapes_ok=1`
#   - `short_steps_ok=1`
#   - `answer_ok=1`
#   - `window_pages_ok=1`
#   - `other_layout_rejected_ok=1`
#   - `speculation_ok=1` (or `speculation_skipped=1` with an empty --drafter)
#   - `packed_checkpoints_ok=1`
#
# Expected failure signals:
#   - non-zero exit with the failed check on stderr

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate native serving of LookAtMyQuant packed checkpoints: step-shape consistency, window pages, layout rejection, and speculation."
cli_option CHECKPOINT --checkpoint MLX_PACKED_CHECKPOINTS_CHECKPOINT string \
    "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB" "Packed checkpoint: a cached repository or a local directory."
cli_option DRAFTER --drafter MLX_PACKED_CHECKPOINTS_DRAFTER string \
    "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter" "Drafter checkpoint; empty skips the speculation check."
cli_option KV_CACHE_MIB --kv-cache-mib MLX_PACKED_CHECKPOINTS_KV_CACHE_MIB int 768 \
    "KV page budget in MiB, split across both page pools."
cli_option TOKENS --tokens MLX_PACKED_CHECKPOINTS_TOKENS int 64 \
    "Generated tokens per run."
cli_option SKIP_SYNC --skip-sync MLX_PACKED_CHECKPOINTS_SKIP_SYNC bool 0 \
    "Skip uv sync; reuse the existing Python environment."
cli_example "bash mlx-host-validation/scripts/native_packed_checkpoints.sh --skip-sync --drafter ''"
cli_parse "$@"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "expected Apple Silicon arm64 host" >&2
    exit 1
fi

echo "[1/2] Sync Python environment"
if [[ "$SKIP_SYNC" == "1" ]]; then
    echo "sync_skipped=1"
else
    uv --directory "$PYTHON_DIR" sync --group dev
fi

echo "[2/2] Validate the packed checkpoint"
uv --directory "$PYTHON_DIR" run python \
    "$ROOT/mlx-host-validation/scripts/python/packed_checkpoints.py" \
    --checkpoint "$CHECKPOINT" \
    --drafter "$DRAFTER" \
    --kv-cache-mib "$KV_CACHE_MIB" \
    --tokens "$TOKENS"
