"""Phase 14 completion-gate helpers."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence


REQUIRED_PHASES: tuple[int, ...] = tuple(range(1, 13))


@dataclass(frozen=True)
class PhaseScriptStatus:
    """Presence and syntax status for one required host-validation script."""

    phase: int
    path: pathlib.Path
    exists: bool
    bash_syntax_ok: bool
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "phase": self.phase,
            "path": str(self.path),
            "exists": self.exists,
            "bash_syntax_ok": self.bash_syntax_ok,
            "error": self.error,
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Phase 14 helper subcommand."""

    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser("check-scripts")
    check.add_argument("--root", required=True)
    check.add_argument("--output", required=True)

    report = subcommands.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--script-status", required=True)
    report.add_argument("--command-log", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--checkpoint", required=True)
    report.add_argument("--host-ran", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "check-scripts":
        return check_scripts(args)
    if args.command == "report":
        return write_report(args)
    raise AssertionError(f"unhandled command {args.command}")


def check_scripts(args: argparse.Namespace) -> int:
    """Check required phase scripts for presence and Bash syntax."""

    root = pathlib.Path(args.root)
    statuses = [
        _phase_script_status(root / "mlx-host-validation" / "scripts", phase)
        for phase in REQUIRED_PHASES
    ]
    payload = {
        "checked_at": _now(),
        "required_phases": list(REQUIRED_PHASES),
        "scripts": [status.to_json() for status in statuses],
    }
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    failures = [
        status for status in statuses if not status.exists or not status.bash_syntax_ok
    ]
    if failures:
        for status in failures:
            print(
                "phase14_required_script_failed "
                f"phase={status.phase} path={status.path} error={status.error}",
                file=sys.stderr,
            )
        return 1
    print("phase14_required_scripts_ok=1")
    return 0


def write_report(args: argparse.Namespace) -> int:
    """Write a durable Phase 14 completion summary."""

    script_status = json.loads(pathlib.Path(args.script_status).read_text())
    command_log = json.loads(pathlib.Path(args.command_log).read_text())
    scripts = script_status.get("scripts", [])
    failed_scripts = [
        item
        for item in scripts
        if not item.get("exists") or not item.get("bash_syntax_ok")
    ]
    failed_commands = [item for item in command_log if item.get("returncode") != 0]
    complete = not failed_scripts and not failed_commands and args.host_ran
    lines = [
        "# Phase 14 Native v2 Completion Gate",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- host: `{platform.platform()}`",
        f"- host_validation_ran: `{str(args.host_ran).lower()}`",
        f"- completion_status: `{'passed' if complete else 'blocked'}`",
        "",
        "## Required Phase Scripts",
        "",
        "| phase | exists | bash_syntax_ok | path | error |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for item in scripts:
        lines.append(
            "| {phase} | {exists} | {syntax} | `{path}` | {error} |".format(
                phase=item["phase"],
                exists=_flag(bool(item["exists"])),
                syntax=_flag(bool(item["bash_syntax_ok"])),
                path=item["path"],
                error=f"`{item['error']}`" if item.get("error") else "",
            )
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "| command | returncode | elapsed_s |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in command_log:
        lines.append(
            "| `{cmd}` | {returncode} | {elapsed:.2f} |".format(
                cmd=item["cmd"],
                returncode=item["returncode"],
                elapsed=float(item["elapsed_s"]),
            )
        )
    if failed_scripts:
        lines.extend(["", "## Blocking Script Failures", ""])
        for item in failed_scripts:
            lines.append(
                "- Phase {phase}: `{path}` -> {error}".format(
                    phase=item["phase"],
                    path=item["path"],
                    error=item.get("error") or "not available",
                )
            )
    if failed_commands:
        lines.extend(["", "## Blocking Command Failures", ""])
        for item in failed_commands:
            lines.append(f"- `{item['cmd']}` exited {item['returncode']}")
    lines.extend(
        [
            "",
            "## Completion Rule",
            "",
            "Phase 14 passes only when every required script exists, local Rust/Python "
            "validation passes, and the full host gate actually runs on Apple "
            "Silicon/Metal. A blocked report is evidence, not a completion claim.",
        ]
    )
    pathlib.Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"phase14_completion_report={args.output}")
    if complete:
        print("phase_14_validation_ok=1")
    else:
        print("phase_14_validation_blocked=1")
    return 0


def _phase_script_status(
    scripts_dir: pathlib.Path,
    phase: int,
) -> PhaseScriptStatus:
    path = scripts_dir / f"v2_phase_{phase}.sh"
    if not path.exists():
        return PhaseScriptStatus(
            phase=phase,
            path=path,
            exists=False,
            bash_syntax_ok=False,
            error="missing required phase script",
        )
    result = subprocess.run(
        ["bash", "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return PhaseScriptStatus(
            phase=phase,
            path=path,
            exists=True,
            bash_syntax_ok=False,
            error=(result.stderr or result.stdout).strip(),
        )
    return PhaseScriptStatus(
        phase=phase,
        path=path,
        exists=True,
        bash_syntax_ok=True,
    )


def _flag(value: bool) -> str:
    return "yes" if value else "no"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
