from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_phase14_helper() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    helper = root / "mlx-host-validation/scripts/python/phase14_completion.py"
    spec = importlib.util.spec_from_file_location("phase14_completion", helper)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_phase_script(root: Path, phase: int, body: str = "exit 0\n") -> None:
    path = root / "mlx-host-validation/scripts" / f"v2_phase_{phase}.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)


def test_phase14_check_scripts_reports_missing_required_script(tmp_path: Path) -> None:
    helper = _load_phase14_helper()
    for phase in range(1, 12):
        _write_phase_script(tmp_path, phase)
    output = tmp_path / "status.json"

    result = helper.main(
        [
            "check-scripts",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    missing = [item for item in payload["scripts"] if not item["exists"]]
    assert result == 1
    assert missing == [
        {
            "phase": 12,
            "path": str(tmp_path / "mlx-host-validation/scripts/v2_phase_12.sh"),
            "exists": False,
            "bash_syntax_ok": False,
            "error": "missing required phase script",
        }
    ]


def test_phase14_check_scripts_accepts_all_required_scripts(tmp_path: Path) -> None:
    helper = _load_phase14_helper()
    for phase in range(1, 13):
        _write_phase_script(tmp_path, phase)
    output = tmp_path / "status.json"

    result = helper.main(
        [
            "check-scripts",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert result == 0
    assert all(item["exists"] and item["bash_syntax_ok"] for item in payload["scripts"])
    assert [item["phase"] for item in payload["scripts"]] == list(range(1, 13))


def test_phase14_report_marks_missing_script_as_blocked(tmp_path: Path) -> None:
    helper = _load_phase14_helper()
    script_status = tmp_path / "status.json"
    command_log = tmp_path / "commands.json"
    report = tmp_path / "report.md"
    script_status.write_text(
        json.dumps(
            {
                "scripts": [
                    {
                        "phase": 12,
                        "path": "mlx-host-validation/scripts/v2_phase_12.sh",
                        "exists": False,
                        "bash_syntax_ok": False,
                        "error": "missing required phase script",
                    }
                ]
            }
        )
    )
    command_log.write_text("[]")

    result = helper.main(
        [
            "report",
            "--root",
            str(tmp_path),
            "--script-status",
            str(script_status),
            "--command-log",
            str(command_log),
            "--output",
            str(report),
            "--checkpoint",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ]
    )

    content = report.read_text()
    assert result == 0
    assert "completion_status: `blocked`" in content
    assert "Phase 12" in content
