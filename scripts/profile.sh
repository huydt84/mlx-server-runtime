#!/usr/bin/env bash
# Run a bounded native-MLX whole-pipeline profile.
#
# Optional environment variables:
#   MLX_PROFILE_CHECKPOINT   Model checkpoint or local path.
#   MLX_PROFILE_DIR          Directory for profile artifacts.
#   MLX_PROFILE_PORT         Gateway port used during capture.
#   MLX_PROFILE_METAL=1      Capture a bounded Metal .gputrace.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MLX_PHASE16_CHECKPOINT="${MLX_PROFILE_CHECKPOINT:-${MLX_PHASE16_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}}"
export MLX_PHASE16_TRACE_DIR="${MLX_PROFILE_DIR:-${MLX_PHASE16_TRACE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-profile}}"
export MLX_PHASE16_PORT="${MLX_PROFILE_PORT:-${MLX_PHASE16_PORT:-18016}}"
export MLX_PHASE16_METAL_CAPTURE="${MLX_PROFILE_METAL:-${MLX_PHASE16_METAL_CAPTURE:-0}}"

exec bash "$ROOT/mlx-host-validation/scripts/v2_phase_16.sh"
