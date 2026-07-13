#!/usr/bin/env python3
"""Run the bounded MLX stream probe and write a durable validation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_worker.native_mlx.overlap import probe_cross_stream_dependency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = probe_cross_stream_dependency()
    payload = {
        "probe": "synthetic_cross_stream_dependency",
        "supported": result.supported,
        "elapsed_ms": result.elapsed_ms,
        "detail": result.detail,
        "serving_overlap_claimed": False,
        "next_gate": "real_model_and_metal_timeline",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_stream_probe_supported={int(result.supported)}")
    print(f"phase17_stream_probe_report={output}")
    print("phase17_serving_overlap_claimed=0")
    return 0 if result.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
