#!/usr/bin/env bash
#
# native-v2 Phase 2 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_2.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - `local-probe/LlamaForCausalLM` as unsupported-class probe
#   - `local-probe/Qwen2ForCausalLM-missing-tokenizer` as malformed-artifact probe
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_lm_import_ok=1`
#   - `probe=known_good` followed by stage list ending before serving
#   - `probe=unsupported` with `category=unsupported_class`
#   - `probe=malformed` with `category=malformed_checkpoint`
#   - `no_ready_claim=1`
#
# Expected failure signals:
#   - non-zero exit
#   - printed probe label and error reason
#   - no `READY` bootstrap message for any native probe

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"

echo "[1/3] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/3] Verify Apple Silicon, mlx, and mlx_lm imports"
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

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/3] Probe native worker startup stages"
uv run python - <<'PY'
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from mlx_worker.ipc import ModelStatus, WorkerError, WorkerReady, decode_bootstrap_message


def write_local_probe(root: Path, architecture_class: str, *, missing_tokenizer: bool) -> str:
    model_dir = root / architecture_class
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture_class],
                "model_type": "qwen2" if architecture_class == "Qwen2ForCausalLM" else "llama",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
            }
        )
    )
    if not missing_tokenizer:
        (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ bos_token }}{{ messages }}"})
    )
    (model_dir / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>"})
    )
    (model_dir / "model.safetensors").write_text("placeholder")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "model.norm.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                }
            }
        )
    )
    return str(model_dir)


def run_probe(
    label: str,
    model_ref: str,
    expected_category: str | None,
    expected_stage: str | None,
    expected_statuses: list[str],
    *,
    expected_ready: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"phase2-{label}-") as temp_dir:
        root = Path(temp_dir)
        socket_path = root / "worker.sock"
        log_path = root / "worker.log"

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)

        env = os.environ.copy()
        env["MLX_RUNTIME_SOCKET"] = str(socket_path)
        env["MLX_RUNTIME_BACKEND"] = "native-mlx"
        env["MLX_RUNTIME_MODEL"] = model_ref
        env["MLX_RUNTIME_VLM_MODEL"] = ""

        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", "mlx_worker.main"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                conn, _ = listener.accept()
            finally:
                listener.close()

            decoded = []
            with conn:
                buffer = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            item = decode_bootstrap_message(line + b"\n")
                            decoded.append(item)
                            if isinstance(item, WorkerReady):
                                break
                    if any(isinstance(item, WorkerReady) for item in decoded):
                        break

            rc = proc.wait(timeout=60)
            expected_rc = 0 if expected_ready else 1
            if rc != expected_rc:
                raise SystemExit(
                    f"{label}: expected exit {expected_rc}, saw {rc}; inspect {log_path}"
                )

        ready = any(isinstance(item, WorkerReady) for item in decoded)
        if ready != expected_ready:
            raise SystemExit(
                f"{label}: expected ready={expected_ready}, saw {ready}; inspect {log_path}"
            )

        statuses = [item for item in decoded if isinstance(item, ModelStatus)]
        stages = [status.progress.current_phase for status in statuses if status.progress]
        if stages != expected_statuses:
            raise SystemExit(f"{label}: expected stages {expected_statuses}, saw {stages}")

        if expected_ready:
            print(f"probe={label}")
            print(f"stages={','.join(stages)}")
            print("ready=1")
            return

        failed_status = next(item for item in statuses if item.state == "failed")
        if failed_status.last_error is None:
            raise SystemExit(f"{label}: missing last_error; inspect {log_path}")

        terminal_error = decoded[-1]
        if not isinstance(terminal_error, WorkerError) or terminal_error.error is None:
            raise SystemExit(f"{label}: missing structured worker error; inspect {log_path}")

        if terminal_error.error.category != expected_category:
            raise SystemExit(
                f"{label}: expected category {expected_category}, saw {terminal_error.error.category}"
            )
        if terminal_error.error.stage != expected_stage:
            raise SystemExit(
                f"{label}: expected stage {expected_stage}, saw {terminal_error.error.stage}"
            )

        print(f"probe={label}")
        print(f"stages={','.join(stages)}")
        print(f"category={terminal_error.error.category}")
        print(f"stage={terminal_error.error.stage}")


with tempfile.TemporaryDirectory(prefix="phase2-local-probes-") as probe_root:
    probe_root_path = Path(probe_root)
    unsupported_model = write_local_probe(
        probe_root_path / "unsupported",
        "LlamaForCausalLM",
        missing_tokenizer=False,
    )
    malformed_model = write_local_probe(
        probe_root_path / "malformed",
        "Qwen2ForCausalLM",
        missing_tokenizer=True,
    )

    run_probe(
        label="known_good",
        model_ref="mlx-community/Qwen2.5-7B-Instruct-4bit",
        expected_category=None,
        expected_stage=None,
        expected_statuses=[
            "architecture_detection",
            "artifact_validation",
            "weight_mapping",
            "native_executor_construction",
            "prompt_tokenizer_readiness",
            "deterministic_warmup",
            "deterministic_warmup",
        ],
        expected_ready=True,
    )
    run_probe(
        label="unsupported",
        model_ref=unsupported_model,
        expected_category="unsupported_class",
        expected_stage="architecture_detection",
        expected_statuses=["architecture_detection"],
    )
    run_probe(
        label="malformed",
        model_ref=malformed_model,
        expected_category="malformed_checkpoint",
        expected_stage="artifact_validation",
        expected_statuses=[
            "architecture_detection",
            "artifact_validation",
        ],
    )

print("phase_2_startup_validation_ok=1")
PY
