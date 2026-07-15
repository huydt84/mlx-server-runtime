#!/usr/bin/env bash
# Build and stage the relocatable MLX Air distribution layout.

set -euo pipefail

usage() {
    echo "Usage: $0 --output-dir PATH" >&2
}

if [[ $# -ne 2 || "$1" != "--output-dir" || -z "$2" ]]; then
    usage
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$2"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "error: output path already exists: $OUTPUT_DIR" >&2
    exit 1
fi

echo "Building MLX Air release binaries"
cargo build \
    --manifest-path "$ROOT/Cargo.toml" \
    --release \
    --bin mlx-air \
    --bin mlx_runtime_gateway

mkdir -p \
    "$OUTPUT_DIR/bin" \
    "$OUTPUT_DIR/config" \
    "$OUTPUT_DIR/licenses" \
    "$OUTPUT_DIR/python"

cp "$ROOT/target/release/mlx-air" "$OUTPUT_DIR/bin/mlx-air"
cp "$ROOT/target/release/mlx_runtime_gateway" "$OUTPUT_DIR/bin/mlx_runtime_gateway"
cp "$ROOT/config/runtime.toml" "$OUTPUT_DIR/config/runtime.toml"
cp "$ROOT/benchmarks/config/default.toml" "$OUTPUT_DIR/config/benchmark.toml"
cp "$ROOT/LICENSE" "$OUTPUT_DIR/licenses/LICENSE"
cp "$ROOT/python/pyproject.toml" "$OUTPUT_DIR/python/pyproject.toml"
cp "$ROOT/python/uv.lock" "$OUTPUT_DIR/python/uv.lock"
COPYFILE_DISABLE=1 tar \
    -C "$ROOT/python" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - \
    mlx_benchmark \
    mlx_worker | tar -C "$OUTPUT_DIR/python" -xf -

echo "MLX Air distribution staged at $OUTPUT_DIR"
